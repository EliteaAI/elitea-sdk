import json
import logging
import re
import threading
import urllib.parse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Generator, Optional

from azure.devops.connection import Connection
from azure.devops.v7_1.core import CoreClient
from azure.devops.v7_1.wiki import WikiClient
from azure.devops.v7_1.work_item_tracking import TeamContext, Wiql, WorkItemTrackingClient
from azure.devops.v7_1.work_item_tracking.models import CommentCreate
from azure.devops.v7_1.search.models import WorkItemSearchRequest
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_core.tools import ToolException
from msrest.authentication import BasicAuthentication
from pydantic import create_model, PrivateAttr, SecretStr
from pydantic import model_validator
from pydantic.fields import Field

from elitea_sdk.tools.non_code_indexer_toolkit import NonCodeIndexerToolkit
from ..utils import (
    AdoSearchPaging,
    SEARCH_INFO_CODES,
    SearchIndexHints,
    create_search_client,
    describe_search_info_code,
)
from ...utils import get_file_bytes_from_artifact, detect_mime_type
from ...utils.content_parser import parse_file_content
from ....runtime.langchain.document_loaders.EliteAImageLoader import EliteAImageLoader, MAX_IMAGE_READ_BYTES
from ....runtime.langchain.document_loaders.constants import image_loaders_map, image_loaders_map_converted
from ....runtime.langchain.document_loaders.image_cache import ImageDescriptionCache
from ....runtime.utils.utils import IndexerKeywords

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = frozenset(image_loaders_map) | frozenset(image_loaders_map_converted)
_MARKDOWN_IMAGE_PATTERN = re.compile(r'!\[(.*?)\]\(\s*([^)\s]+)(?:\s+"[^"]*")?\s*\)')
_IMAGE_WORKERS = 5
_MAX_COMMENT_IMAGES_PER_CALL = 20
# DoS backstop for the legacy work item field pass, NOT a behavior gate: no plausible real
# work item embeds this many distinct field images, but a hostile one is only bounded by
# ADO field size limits — without a ceiling one call means an unbounded vision-LLM bill.
_WORK_ITEM_IMAGE_CEILING = 200
# The byte cap bounds transfer, not decode: PIL allows 300M-pixel rasters and svglib
# rasterizes whatever canvas the SVG declares. 36 MP covers an 8K UHD screenshot while
# capping worst-case RGBA decode at ~145 MB per image.
_MAX_IMAGE_PIXELS_FOR_LLM = 36_000_000
# DoS backstop for the legacy work item field pass, NOT an image-size gate: comfortably
# above the Azure DevOps Services 60 MB attachment maximum, so no real attachment is affected.
_ATTACHMENT_STREAM_CEILING_BYTES = 100 * 1024 * 1024
# Cumulative DoS backstop for one gates-off call: per-attachment caps alone compose to
# many GB against hostile content; a realistic screenshot-heavy item totals tens of MB.
# The budget is per call, tracked by the serial gates-off loop — concurrent callers
# (e.g. indexer fetch workers) each get their own, so the aggregate scales with workers.
_WORK_ITEM_STREAM_BUDGET_BYTES = 512 * 1024 * 1024
_IMAGE_UNAVAILABLE = "[image unavailable: {reason}]"
_IMAGE_LIMIT_NOTE = "[image skipped: per-call image limit of {limit} reached]"
_IMAGE_BUDGET_NOTE = "[image skipped: per-call download budget reached]"


@dataclass(frozen=True)
class HighlightBudget:
    max_results: int = 5
    max_per_result: int = 3
    max_chars: int = 200


PAGING = AdoSearchPaging(default_top=5, max_top=50)
HIGHLIGHTS = HighlightBudget()

SEARCH_HINTS = SearchIndexHints(
    filter_not_indexed=(
        "Check that the work item type, state, assignee or area path filtered on exists in "
        "this project."
    ),
)

class _ImageNote(str):
    """Failure/skip note from the image-describe pipeline. A distinct type, not text
    matching, so consumers that must drop notes (the indexer keeps them out of
    embeddings) cannot silently break when a note's wording changes."""

create_wi_field = """JSON of the work item fields to create in Azure DevOps, i.e.
                    {
                       "fields":{
                          "System.Title":"Implement Registration Form Validation",
                          "field2":"Value 2",
                       }
                    }
                    """

# Input models for Azure DevOps operations
ADOWorkItemsSearch = create_model(
    "AzureDevOpsSearchModel",
    query=(str, Field(description="WIQL query for searching Azure DevOps work items")),
    limit=(Optional[int], Field(description="Number of items to return. IMPORTANT: Tool returns all items if limit=-1. If parameter is not provided then the value will be taken from tool configuration.", default=None)),
    fields=(Optional[list[str]], Field(description="Comma-separated list of requested fields", default=None))
)

ADOWorkItemsTextSearch = create_model(
    "AzureDevOpsWorkItemsTextSearchModel",
    query=(str, Field(description="Free text to search for: keywords, a quoted phrase, or a person's name. Matches work item title, description, acceptance criteria, tags, history and comments. This is not WIQL - do not pass a SELECT statement.")),
    work_item_type=(Optional[List[str]], Field(default=None, description="Restrict to these work item types, exactly as named in Azure DevOps, e.g. ['Bug', 'User Story'].")),
    state=(Optional[List[str]], Field(default=None, description="Restrict to these states, exactly as named in Azure DevOps, e.g. ['New', 'Active'].")),
    assigned_to=(Optional[List[str]], Field(default=None, description="Restrict to these assignees, in the display-name-and-email form Azure DevOps stores, e.g. ['John Doe <jodoe@contoso.com>'].")),
    area_path=(Optional[List[str]], Field(default=None, description="Restrict to these area paths, e.g. ['MyProject\\Backend'].")),
    top=(Optional[int], Field(default=PAGING.default_top, ge=1, le=PAGING.max_top, description=f"Number of work items to return. Defaults to {PAGING.default_top}; maximum {PAGING.max_top}. Matched-field highlights are only attached to the first {HIGHLIGHTS.max_results} results, so refine the query rather than raising this.")),
    skip=(Optional[int], Field(default=0, ge=0, le=PAGING.max_skip, description=f"Number of results to skip for paging. Maximum {PAGING.max_skip}. Pass back the next_skip value of a previous response whenever that response supplied one, including when it returned no results - but stop and refine the query once a second window in a row comes back empty, which means the token cannot read these matches.")),
    include_highlights=(Optional[bool], Field(default=False, description="Return the matched field snippets that explain why each work item matched. Off by default, so a result set is titles and metadata only. Set true whenever relevance matters - free text also matches description, acceptance criteria, tags, history and comments, so a title alone often does not show why an item was returned.")),
)

ADOCreateWorkItem = create_model(
    "AzureDevOpsCreateWorkItemModel",
    work_item_json=(str, Field(description=create_wi_field)),
    wi_type=(Optional[str], Field(description="Work item type, e.g. 'Task', 'Issue' or  'EPIC'", default="Task"))
)

ADOUpdateWorkItem = create_model(
    "AzureDevOpsUpdateWorkItemModel",
    id=(str, Field(description="ID of work item required to be updated")),
    work_item_json=(str, Field(description=create_wi_field))
)

ADODeleteWorkItem = create_model(
    "AzureDevOpsDeleteWorkItemModel",
    id=(int, Field(description="ID of work item to be deleted"))
)

ADOGetWorkItem = create_model(
    "AzureDevOpsGetWorkItemModel",
    id=(int, Field(description="The work item id")),
    fields=(Optional[list[str]], Field(description="Comma-separated list of requested fields", default=None)),
    as_of=(Optional[str], Field(description="AsOf UTC date time string", default=None)),
    expand=(Optional[str], Field(description="The expand parameters for work item attributes. Possible options are { None, Relations, Fields, Links, All }.", default=None)),
    parse_attachments=(Optional[bool], Field(description="Value that defines is attachment should be parsed.", default=False)),
    image_description_prompt=(Optional[str],
                     Field(description="Prompt which is used for image description", default=None)),
    process_images=(Optional[bool], Field(default=True, description="Whether to process images in work item fields and attachments. Set to False to skip image description processing and return raw content.")),
)

ADOLinkWorkItem = create_model(
    "ADOLinkWorkItem",
    source_id=(int, Field(description="ID of the work item you plan to add link to")),
    target_id=(int, Field(description="ID of the work item linked to source one")),
    link_type=(str, Field(description="Link type: System.LinkTypes.Dependency-forward, etc.")),
    attributes=(Optional[dict], Field(description="Dict with attributes used for work items linking. Example: `comment`, etc. and syntax 'comment': 'Some linking comment'", default=None))
)

ADOGetLinkType = create_model(
    "ADOGetLinkType",
)

ADOGetComments = create_model(
    "ADOGetComments",
    work_item_id=(int, Field(description="The work item id")),
    limit_total=(Optional[int], Field(description="Max number of total comments to return", default=None)),
    include_deleted=(Optional[bool], Field(description="Specify if the deleted comments should be retrieved", default=False)),
    expand=(Optional[str], Field(description="The expand parameters for comments. Possible options are { all, none, reactions, renderedText, renderedTextOnly }.", default="none")),
    order=(Optional[str], Field(description="Order in which the comments should be returned. Possible options are { asc, desc }", default=None)),
    process_images=(Optional[bool], Field(description="Whether to fetch and analyze images embedded in comment text. When True, each embedded ADO attachment image is described by the configured LLM and the description is inserted next to the image reference in the returned comment content.", default=False)),
    image_description_prompt=(Optional[str], Field(description="Prompt which is used for image description", default=None)),
)

ADOGetImageByUrl = create_model(
    "ADOGetImageByUrl",
    attachment_url=(str, Field(description="Full Azure DevOps work item attachment URL, e.g. https://dev.azure.com/{org}/{project}/_apis/wit/attachments/{guid}?fileName=screen.png — typically the src of an <img> tag or markdown image found in work item fields or comments. Both dev.azure.com and {org}.visualstudio.com URL forms are accepted.")),
    file_name=(Optional[str], Field(description="File name with extension (e.g. 'screen.png'). Overrides or supplies the fileName query parameter; required when the URL does not contain one.", default=None)),
    prompt=(Optional[str], Field(description="Custom instruction for the image analysis. Defaults to the standard detailed image description prompt.", default=None)),
)

ADOLinkWorkItemsToWikiPage = create_model(
    "ADOLinkWorkItemsToWikiPage",
    work_item_ids=(List[int], Field(description="List of work item IDs to link to the wiki page")),
    wiki_identified=(str, Field(description="Wiki ID or wiki name")),
    page_name=(str, Field(description="Wiki page path to link the work items to", examples=["/TargetPage"]))
)

ADOUnlinkWorkItemsFromWikiPage = create_model(
    "ADOUnlinkWorkItemsFromWikiPage",
    work_item_ids=(List[int], Field(description="List of work item IDs to unlink from the wiki page")),
    wiki_identified=(str, Field(description="Wiki ID or wiki name")),
    page_name=(str, Field(description="Wiki page path to unlink the work items from", examples=["/TargetPage"]))
)

ADOGetWorkItemTypeFields = create_model(
    "ADOGetWorkItemTypeFields",
    work_item_type=(Optional[str], Field(description="Work item type to get fields for (e.g., 'Task', 'Bug', 'Test Case', 'Epic'). Default is 'Task'.", default="Task")),
    force_refresh=(Optional[bool], Field(description="If True, reload field definitions from Azure DevOps. Use this if project configuration has changed.", default=False))
)

ADOAttachFileToWorkItem = create_model(
    "ADOAttachFileToWorkItem",
    work_item_id=(int, Field(description="ID of the work item to attach the file to")),
    filepath=(str, Field(description="File path in format /{bucket}/{filename} pointing to the artifact to attach. Any file type is supported (image, PDF, document, etc.). Get this from a file/image generation or upload tool response.")),
    filename=(Optional[str], Field(description="Filename to use for the ADO attachment, e.g. 'diagram.png'. If not provided, uses the original filename from the artifact. Should include file extension.", default=None)),
    inline_field=(Optional[str], Field(description="Optional HTML-typed work item field reference name (e.g. 'System.Description', 'Microsoft.VSTS.TCM.ReproSteps'). If provided, an <img> tag (for images) or <a> link (for other file types) is appended to that field's current value so the attachment renders inline. Requires the field to accept HTML.", default=None)),
    add_as_comment=(Optional[bool], Field(description="If True, also add a work item comment containing the inline image/link reference. Default is False.", default=False)),
    comment=(Optional[str], Field(description="Optional 'comment' attribute stored on the AttachedFile relation itself (a short caption/description of the attachment).", default=None)),
)

class AzureDevOpsApiWrapper(NonCodeIndexerToolkit):
    # TODO use ado_configuration instead of organization_url, project and token
    organization_url: str
    project: str
    token: SecretStr
    limit: Optional[int] = 5
    _client: Optional[WorkItemTrackingClient] = PrivateAttr()
    _wiki_client: Optional[WikiClient] = PrivateAttr() # Add WikiClient instance
    _core_client: Optional[CoreClient] = PrivateAttr() # Add CoreClient instance
    _relation_types: Dict = PrivateAttr(default_factory=dict) # track actual relation types for instance
    _work_item_type_fields_cache: Dict[str, Dict] = PrivateAttr(default_factory=dict)  # Cache for work item type field definitions
    _image_cache: ImageDescriptionCache = PrivateAttr(default_factory=ImageDescriptionCache)
    _search_client_instance: Optional[Any] = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types (e.g., WorkItemTrackingClient, WikiClient, CoreClient)

    @model_validator(mode='before')
    @classmethod
    def validate_toolkit(cls, values):
        """Validate and set up the Azure DevOps client."""
        try:
            # Set up connection to Azure DevOps using Personal Access Token (PAT)
            credentials = BasicAuthentication('', values['token'])
            connection = Connection(base_url=values['organization_url'], creds=credentials)

            # Retrieve the work item tracking client and assign it to the private _client attribute
            cls._client = connection.clients_v7_1.get_work_item_tracking_client()
            cls._wiki_client = connection.clients_v7_1.get_wiki_client()
            cls._core_client = connection.clients_v7_1.get_core_client()

        except Exception as e:
            error_msg = str(e).lower()
            if "expired" in error_msg or "token" in error_msg and ("invalid" in error_msg or "unauthorized" in error_msg):
                raise ValueError(
                    "Azure DevOps connection failed: Your access token has expired or is invalid. "
                    "Please refresh your token in the toolkit configuration."
                )
            elif "401" in error_msg or "unauthorized" in error_msg:
                raise ValueError(
                    "Azure DevOps connection failed: Authentication failed. "
                    "Please check your credentials in the toolkit configuration."
                )
            elif "404" in error_msg or "not found" in error_msg:
                raise ValueError(
                    "Azure DevOps connection failed: Organization or project not found. "
                    "Please verify your organization URL and project name."
                )
            elif "timeout" in error_msg or "timed out" in error_msg:
                raise ValueError(
                    "Azure DevOps connection failed: Connection timed out. "
                    "Please check your network connection and try again."
                )
            else:
                raise ValueError(f"Azure DevOps connection failed: {e}")

        return super().validate_toolkit(values)

    def _parse_work_items(self, work_items, fields=None):
        """Parse work items dynamically based on the fields requested."""
        parsed_items = []

        # If no specific fields are provided, default to the basic ones
        if fields is None:
            fields = ["System.Title", "System.State", "System.AssignedTo", "System.WorkItemType", "System.CreatedDate",
                      "System.ChangedDate"]

        # Remove 'System.Id' from the fields list, as it's not a field you request, it's metadata
        fields = [field for field in fields if "System.Id" not in field]
        fields = [field for field in fields if "System.WorkItemType" not in field]
        for item in work_items:
            # Fetch full details of the work item, including the requested fields
            full_item = self._client.get_work_item(id=item.id, project=self.project, fields=fields)
            fields_data = full_item.fields

            # Parse the fields dynamically
            parsed_item = {"id": full_item.id, "url": f"{self.organization_url}/_workitems/edit/{full_item.id}"}

            # Iterate through the requested fields and add them to the parsed result
            for field in fields:
                parsed_item[field] = fields_data.get(field, "N/A")

            parsed_items.append(parsed_item)

        return parsed_items

    def _transform_work_item(self, work_item_json):
        try:
            # Convert the input JSON to a Python dictionary
            if isinstance(work_item_json, str):
                work_item_json = json.loads(work_item_json)
        except (json.JSONDecodeError, ValueError) as e:
            raise ToolException(f"Issues during attempt to parse work_item_json: {e}")

        if 'fields' not in work_item_json:
            raise ToolException("The 'fields' property is missing from the work_item_json.")

            # Transform the dictionary into a list of JsonPatchOperation objects
        patch_document = [
            {
                "op": "add",
                "path": f"/fields/{field}",
                "value": value
            }
            for field, value in work_item_json["fields"].items()
        ]
        return patch_document

    def create_work_item(self, work_item_json, wi_type="Task"):
        """Create a work item in Azure DevOps."""
        try:
            patch_document = self._transform_work_item(work_item_json)
        except Exception as e:
            return ToolException(f"Issues during attempt to parse work_item_json: {str(e)}")

        try:
            # Use the transformed patch_document to create the work item
            work_item = self._client.create_work_item(
                document=patch_document,
                project=self.project,
                type=wi_type
            )
            return {
                "id": work_item.id,
                "message": f"Work item {work_item.id} created successfully. View it at {work_item.url}."
            }
        except Exception as e:
            if "unknown value" in str(e):
                logger.error(f"Unable to create work item due to incorrect assignee: {e}")
                return ToolException(f"Unable to create work item due to incorrect assignee: {e}")
            logger.error(f"Error creating work item: {e}")
            return ToolException(f"Error creating work item: {e}")

    def update_work_item(self, id: str, work_item_json: str):
        """Updates existing work item per defined data"""

        try:
            patch_document = self._transform_work_item(work_item_json)
            work_item = self._client.update_work_item(id=id, document=patch_document, project=self.project)
        except Exception as e:
            return ToolException(f"Issues during attempt to parse work_item_json: {str(e)}")
        return f"Work item ({work_item.id}) was updated."

    def delete_work_item(self, id: int):
        """Delete a work item from Azure DevOps by ID."""
        try:
            self._client.delete_work_item(id=id, project=self.project)
            return f"Work item {id} was successfully deleted."
        except Exception as e:
            logger.error(f"Error deleting work item {id}: {e}")
            return ToolException(f"Error deleting work item {id}: {e}")

    def get_relation_types(self) -> dict:
        """Returns dict of possible relation types per syntax: 'relation name': 'relation reference name'.
        NOTE: reference name is used for adding links to the work item"""

        if not self._relation_types:
            # have to be called only once for session
            relations = self._client.get_relation_types()
            for relation in relations:
                self._relation_types.update({relation.name: relation.reference_name})
        return self._relation_types

    def _get_work_item_type_fields(self, work_item_type: str) -> Dict:
        """
        Get field definitions for a specific work item type using the Azure DevOps client.

        Args:
            work_item_type: The work item type (e.g., 'Task', 'Bug', 'Test Case')

        Returns:
            dict: Mapping of field reference names to their metadata (name, type, required, allowed values)
        """
        try:
            # Use the WorkItemTrackingClient to get work item type fields
            work_item_type_obj = self._client.get_work_item_type(self.project, work_item_type)

            # Get fields for this work item type
            fields = work_item_type_obj.fields

            field_definitions = {}
            for field in fields:
                field_ref_name = field.reference_name
                field_definitions[field_ref_name] = {
                    'name': field.name,
                    'type': field.type if hasattr(field, 'type') else 'Unknown',
                    'required': field.always_required if hasattr(field, 'always_required') else False,
                    'allowed_values': field.allowed_values if hasattr(field, 'allowed_values') else [],
                    'description': field.help_text if hasattr(field, 'help_text') else ''
                }

            return field_definitions

        except Exception as e:
            logger.warning(f"Failed to fetch field definitions for work item type '{work_item_type}' using client: {e}")
            return {}

    def _format_work_item_type_fields_for_display(self, work_item_type: str, field_definitions: Dict) -> str:
        """
        Format field definitions in human-readable format for LLM.

        Args:
            work_item_type: The work item type name
            field_definitions: Output from _get_work_item_type_fields()

        Returns:
            Formatted string with field information
        """
        if not field_definitions:
            return f"Unable to retrieve field definitions for work item type '{work_item_type}'. Please check your Azure DevOps connection and permissions."

        output = [f"Available Fields for Work Item Type '{work_item_type}' in Project '{self.project}':\n"]
        output.append("=" * 80)

        # Separate required and optional fields
        required_fields = []
        optional_fields = []

        for ref_name, field_info in sorted(field_definitions.items()):
            field_entry = {
                'ref_name': ref_name,
                'name': field_info.get('name', ref_name),
                'type': field_info.get('type', 'Unknown'),
                'required': field_info.get('required', False),
                'allowed_values': field_info.get('allowed_values', [])
            }

            if field_entry['required']:
                required_fields.append(field_entry)
            else:
                optional_fields.append(field_entry)

        # Display required fields first
        if required_fields:
            output.append("\n📋 REQUIRED FIELDS:")
            output.append("-" * 80)
            for field in required_fields:
                output.append(f"\n✓ {field['name']} (Reference: {field['ref_name']})")
                output.append(f"  Type: {field['type']}")
                if field['allowed_values']:
                    output.append(f"  Allowed Values: {', '.join(str(v) for v in field['allowed_values'])}")

        # Display optional fields (common ones only)
        if optional_fields:
            output.append("\n\n📝 OPTIONAL FIELDS (Common):")
            output.append("-" * 80)
            # Show only commonly used optional fields
            common_fields = ['System.AssignedTo', 'System.AreaPath', 'System.IterationPath',
                           'Microsoft.VSTS.Common.Priority', 'System.Tags', 'System.State']
            for field in optional_fields:
                if field['ref_name'] in common_fields:
                    output.append(f"\n  {field['name']} (Reference: {field['ref_name']})")
                    output.append(f"    Type: {field['type']}")
                    if field['allowed_values']:
                        output.append(f"    Allowed Values: {', '.join(str(v) for v in field['allowed_values'])}")

        output.append("\n\n" + "=" * 80)
        output.append("\n💡 Usage Instructions:")
        output.append("  • Use the 'Reference' name (e.g., 'System.Title') as the field key in work_item_json")
        output.append("  • Provide all required fields when creating work items")
        output.append("  • For fields with allowed values, use exact value from the list")
        output.append(f"  • Example for {work_item_type}: " + '{"fields": {"System.Title": "My title", "CustomField": "Value"}}')

        return '\n'.join(output)

    def get_work_item_type_fields(self, work_item_type: str = "Task", force_refresh: bool = False) -> str:
        """
        Get formatted information about available fields for a specific work item type.
        This method helps discover which fields are required for work item creation.

        Args:
            work_item_type: The work item type to get fields for (e.g., 'Task', 'Bug', 'Test Case', 'Epic').
                           Default is 'Task'.
            force_refresh: If True, reload field definitions from Azure DevOps instead of using cache.
                          Use this if project configuration has changed (new fields added, etc.).

        Returns:
            Formatted string with field names, types, and requirements
        """
        cache_key = work_item_type

        if force_refresh or cache_key not in self._work_item_type_fields_cache:
            self._work_item_type_fields_cache[cache_key] = self._get_work_item_type_fields(work_item_type)

        return self._format_work_item_type_fields_for_display(work_item_type, self._work_item_type_fields_cache[cache_key])

    def link_work_items(self, source_id, target_id, link_type, attributes: dict = None):
        """Add the relation to the source work item with an appropriate attributes if any. User may pass attributes like name, etc."""

        if not self._relation_types:
            # check cached relation types and trigger its collection if it is empty by that moment
            self.get_relation_types()
        if link_type not in self._relation_types.values():
            return ToolException(f"Link type is incorrect. You have to use proper relation's reference name NOT relation's name: {self._relation_types}")

        relation = {
            "rel": link_type,
            "url": f"{self.organization_url}/_apis/wit/workItems/{target_id}"
        }

        if attributes:
            relation.update({"attributes": attributes})

        try:
            self._client.update_work_item(
                document=[
                    {
                        "op": "add",
                        "path": "/relations/-",
                        "value": relation
                    }
                ],
                id=source_id
            )
        except Exception as e:
            logger.error(f"Error linking work items: {e}")
            return ToolException(f"Error linking work items: {e}")

        return f"Work item {source_id} linked to {target_id} with link type {link_type}"

    def search_work_items(self, query: str, limit: int = None, fields=None):
        """Search for work items with a WIQL query (SELECT ... FROM WorkItems WHERE ...) and dynamically fetch fields based on the query. Requires a valid WIQL statement, not free text - for keyword or phrase search use search_work_items_by_text."""
        try:
            # Create a Wiql object with the query
            wiql = Wiql(query=query)

            # Validate that the Azure DevOps client is initialized
            if not self._client:
                raise ToolException("Azure DevOps client not initialized.")
            logger.info(f"Search for work items using {query}")
            # Execute the WIQL query
            if not limit:
                limit = self.limit
            work_items = self._client.query_by_wiql(wiql, top=None if limit < 0 else limit, team_context=TeamContext(project=self.project)).work_items

            if not work_items:
                return "No work items found."

            # Parse the work items and fetch the fields dynamically
            parsed_work_items = self._parse_work_items(work_items, fields)

            # Return the parsed work items
            return parsed_work_items
        except ValueError as ve:
            logger.error(f"Invalid WIQL query: {ve}")
            return ToolException(f"Invalid WIQL query: {ve}")
        except Exception as e:
            logger.error(f"Error searching work items: {e}")
            return ToolException(f"Error searching work items: {e}")

    @property
    def _search_client(self):
        if self._search_client_instance is None:
            self._search_client_instance = create_search_client(self.organization_url, self.token)
        return self._search_client_instance

    def search_work_items_by_text(
            self,
            query: str,
            work_item_type: Optional[List[str]] = None,
            state: Optional[List[str]] = None,
            assigned_to: Optional[List[str]] = None,
            area_path: Optional[List[str]] = None,
            top: Optional[int] = PAGING.default_top,
            skip: Optional[int] = 0,
            include_highlights: Optional[bool] = False,
    ) -> str:
        """
        Search work items in this project by free text, ranked by relevance.

        Use for keywords, a phrase or a person's name. For a structured query over fields,
        dates or links use search_work_items, which takes WIQL.

        Returns one summary per work item - never the full body - so follow up with
        get_work_item to read one. Set include_highlights to see which field matched. Pass
        next_skip back as skip while a response supplies one, stopping after two empty
        windows in a row.

        Returns:
            str: JSON with total_count, returned, skip, truncated, next_skip, results and
            warnings.
        """
        if not query or not query.strip():
            return ToolException("Search query cannot be empty. Provide text to search for.")

        top = max(1, min(top or PAGING.default_top, PAGING.max_top))
        skip = max(0, min(skip or 0, PAGING.max_skip))
        include_highlights = bool(include_highlights)

        filters = {"System.TeamProject": [self.project]}
        for field_reference_name, requested_values in (
            ("System.WorkItemType", work_item_type),
            ("System.State", state),
            ("System.AssignedTo", assigned_to),
            ("System.AreaPath", area_path),
        ):
            if requested_values:
                filters[field_reference_name] = (
                    [requested_values] if isinstance(requested_values, str) else list(requested_values)
                )

        try:
            response = self._search_client.fetch_work_item_search_results(
                request=WorkItemSearchRequest(
                    search_text=query,
                    filters=filters,
                    top=top,
                    skip=skip,
                ),
                project=self.project,
            )
        except Exception as e:
            msg = f"Unable to search work items for query '{query}': {str(e)}"
            logger.error(msg)
            return ToolException(
                f"{msg}\nWork item search requires at least Basic access and a token with the "
                "Work Items (read) scope. On Azure DevOps Server it also requires the Search "
                "extension to be installed."
            )

        results = response.results or []
        total_count = response.count or 0
        payload_results = []
        results_denied_highlights_by_budget = 0
        results_carrying_highlights = 0
        for result in results:
            fields_by_lowercased_name = {key.lower(): value for key, value in (result.fields or {}).items()}
            work_item_id = fields_by_lowercased_name.get("system.id")
            entry = {
                "id": int(work_item_id) if str(work_item_id).isdigit() else work_item_id,
                "title": fields_by_lowercased_name.get("system.title"),
                "type": fields_by_lowercased_name.get("system.workitemtype"),
                "state": fields_by_lowercased_name.get("system.state"),
                "project": result.project.name if result.project else self.project,
                "url": f"{self.organization_url}/_workitems/edit/{work_item_id}" if work_item_id else result.url,
            }
            assignee = fields_by_lowercased_name.get("system.assignedto")
            if assignee:
                entry["assigned_to"] = assignee
            if include_highlights:
                hits_carrying_highlights = [hit for hit in (result.hits or []) if hit.highlights]
                highlights_fit_in_budget = results_carrying_highlights < HIGHLIGHTS.max_results
                if hits_carrying_highlights and highlights_fit_in_budget:
                    entry["highlights"] = [
                        {
                            "field": hit.field_reference_name,
                            "text": BeautifulSoup(hit.highlights[0], "html.parser")
                            .get_text(" ", strip=True)[:HIGHLIGHTS.max_chars],
                        }
                        for hit in hits_carrying_highlights[:HIGHLIGHTS.max_per_result]
                    ]
                    results_carrying_highlights += 1
                elif hits_carrying_highlights:
                    results_denied_highlights_by_budget += 1
            payload_results.append(entry)

        returned = len(payload_results)
        matches_beyond_this_window = total_count > skip + top
        next_skip = skip + (top if returned else max(top, PAGING.empty_window_stride))
        paging_ceiling_reached = next_skip > PAGING.max_skip
        reported_code = SEARCH_INFO_CODES.get(response.info_code)
        window_worth_continuing_from = returned > 0 or (
            reported_code is not None and reported_code.matches_hidden_by_permissions
        )
        payload = {
            "total_count": total_count,
            "returned": returned,
            "skip": skip,
            "truncated": matches_beyond_this_window,
            "results": payload_results,
        }
        if matches_beyond_this_window and window_worth_continuing_from and not paging_ceiling_reached:
            payload["next_skip"] = next_skip

        warnings = []
        if response.info_code:
            warnings.append(describe_search_info_code(response.info_code, SEARCH_HINTS))
        if matches_beyond_this_window and paging_ceiling_reached:
            warnings.append(
                f"The paging limit of {PAGING.max_skip} results has been reached; refine the "
                "query to reach the remaining matches."
            )
        if not returned:
            warnings.append(
                "No matches in this window. Work item search covers the configured project and "
                "indexes title, description, acceptance criteria, tags, history and comments - it "
                "does not read attachment contents. Filter values must match Azure DevOps exactly, "
                "for example a work item type of 'User Story' rather than 'story'. If a total_count "
                "above zero is reported with no results, the matches in this window are either not "
                "readable with the current token or already past the result set - continue with "
                "next_skip once, and refine the query when it does not come back or when a second "
                "window in a row comes back empty, which means the token cannot read these "
                "matches. Newly created or edited items can take a few minutes to appear in the "
                "index."
            )
        if results_denied_highlights_by_budget:
            warnings.append(
                f"Highlights were attached to the first {HIGHLIGHTS.max_results} result(s); "
                f"{results_denied_highlights_by_budget} further result(s) list metadata alone. "
                "Lower top or refine the query to see why they matched."
            )
        if warnings:
            payload["warnings"] = warnings

        return json.dumps(payload)

    def _extract_attachment_ref(self, attachment_url):
        parsed = urllib.parse.urlparse(attachment_url)
        segments = [segment for segment in parsed.path.split('/') if segment]
        if 'attachments' not in segments:
            return None
        index = segments.index('attachments')
        if index + 1 >= len(segments):
            return None
        attachment_id = segments[index + 1]
        if not re.fullmatch(r'[\w-]+', attachment_id):
            return None
        file_names = urllib.parse.parse_qs(parsed.query).get('fileName', [])
        return attachment_id, (urllib.parse.unquote(file_names[0]) if file_names else None)

    def _get_attachment_content_capped(self, attachment_id, limit):
        content_generator = self._client.get_attachment_content(id=attachment_id, download=True)
        chunks = []
        total = 0
        try:
            for chunk in content_generator:
                total += len(chunk)
                if total > limit:
                    return None
                chunks.append(chunk)
        finally:
            close = getattr(content_generator, 'close', None)
            if close:
                close()
        return b"".join(chunks)

    def _fetch_validated_image(self, attachment_id, name):
        """Gated validate-and-fetch ladder shared by the comment pipeline and
        get_image_by_url; returns (content, None) on success or (None, reason)."""
        suffix = Path(name).suffix
        if suffix.lower() not in _IMAGE_EXTENSIONS:
            return None, f"unsupported image format '{suffix}'; supported: png, jpg, jpeg, gif, webp, bmp, svg"
        try:
            content = self._get_attachment_content_capped(attachment_id, limit=MAX_IMAGE_READ_BYTES)
        except Exception as e:
            return None, f"could not download the attachment: {e}"
        if content is None:
            return None, "image exceeds the 5 MB processing limit"
        width, height = EliteAImageLoader.read_dimensions(name, content)
        if not width or not height:
            return None, "could not read image dimensions"
        if width * height > _MAX_IMAGE_PIXELS_FOR_LLM:
            return None, f"image dimensions {width}x{height} exceed the processing limit"
        return content, None

    def _describe_attachment_safe(self, attachment_url, file_name=None, prompt=None,
                                  enforce_image_gates=True, stream_budget=None):
        try:
            if attachment_url.lower().startswith('data:'):
                return None
            ref = self._extract_attachment_ref(attachment_url)
            if ref is None:
                return _ImageNote(_IMAGE_UNAVAILABLE.format(reason="not an Azure DevOps attachment URL"))
            attachment_id, derived_name = ref
            name = file_name or derived_name
            if not name:
                return _ImageNote(_IMAGE_UNAVAILABLE.format(reason="no file name in URL"))
            if enforce_image_gates:
                content, error = self._fetch_validated_image(attachment_id, name)
                if error:
                    return _ImageNote(_IMAGE_UNAVAILABLE.format(reason=error))
            else:
                if stream_budget is not None and stream_budget[0] <= 0:
                    return _ImageNote(_IMAGE_BUDGET_NOTE)
                content = self._get_attachment_content_capped(
                    attachment_id, limit=_ATTACHMENT_STREAM_CEILING_BYTES)
                if stream_budget is not None:
                    stream_budget[0] -= len(content) if content is not None else _ATTACHMENT_STREAM_CEILING_BYTES
                if content is None:
                    return _ImageNote(_IMAGE_UNAVAILABLE.format(reason="attachment exceeds the download safety limit"))
            result = parse_file_content(file_content=content, file_name=name, llm=self.llm,
                                        prompt=prompt, image_cache=self._image_cache)
            if isinstance(result, ToolException):
                return _ImageNote(_IMAGE_UNAVAILABLE.format(reason=str(result)))
            return result
        except Exception as e:
            logger.warning(f"Failed to describe attachment '{attachment_url}': {e}")
            return _ImageNote(_IMAGE_UNAVAILABLE.format(reason=str(e)))

    def _describe_image_urls(self, urls, prompt, enforce_image_gates=True, max_images=None):
        descriptions = {}
        fetch_urls = []
        for url in dict.fromkeys(urls):
            if url.lower().startswith('data:'):
                descriptions[url] = None
            elif max_images is not None and len(fetch_urls) >= max_images:
                descriptions[url] = _ImageNote(_IMAGE_LIMIT_NOTE.format(limit=max_images))
            else:
                fetch_urls.append(url)

        if not fetch_urls:
            return descriptions

        if not enforce_image_gates:
            # This path serves the pre-existing work item field pass, whose decode
            # behaviour is unbounded; keeping it serial preserves today's one-image-at-a-time
            # memory profile and makes the cumulative download budget a plain running total.
            budget = [_WORK_ITEM_STREAM_BUDGET_BYTES]
            results = [self._describe_attachment_safe(url, prompt=prompt, enforce_image_gates=False,
                                                      stream_budget=budget) for url in fetch_urls]
        else:
            def describe(url):
                return self._describe_attachment_safe(url, prompt=prompt, enforce_image_gates=True)

            # The indexer's work item fetch pool already spends the parallelism budget;
            # nesting another pool inside one of its workers would multiply concurrent
            # LLM calls beyond it. Host runtimes routinely dispatch tool calls off the
            # main thread, so thread identity alone is not a nesting signal.
            in_fetch_pool = threading.current_thread().name.startswith("ado-wi-fetch")
            max_workers = min(_IMAGE_WORKERS, len(fetch_urls))
            if max_workers <= 1 or in_fetch_pool:
                results = [describe(url) for url in fetch_urls]
            else:
                with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ado-img") as executor:
                    results = list(executor.map(describe, fetch_urls))

        descriptions.update(zip(fetch_urls, results))
        return descriptions

    def parse_attachment_by_url(self, attachment_url, file_name=None, image_description_prompt=None):
        ref = self._extract_attachment_ref(attachment_url)
        if ref is None:
            raise ToolException(f"Attachment '{attachment_url}' was not found.")
        attachment_id, derived_name = ref
        file_name = file_name or derived_name
        if not file_name:
            raise ToolException("File name must be provided either in the URL or as a parameter.")
        return self.parse_attachment_by_id(attachment_id, file_name, image_description_prompt)

    def get_image_by_url(self, attachment_url: str, file_name: Optional[str] = None, prompt: Optional[str] = None):
        """Analyze an image attached to an Azure DevOps work item and return its textual content.
        Accepts a work item attachment URL (the src of an <img> tag or a markdown image reference
        found in work item fields or comments), downloads the image by its attachment id through
        the configured Azure DevOps connection and uses the configured vision model to extract
        text and describe what the image shows. Supported formats: png, jpg, jpeg, gif, webp,
        bmp, svg; maximum size 5 MB. Returns the extracted description as plain text, or an
        error message if the image is inaccessible or the format is unsupported."""
        try:
            if not self._client:
                return ToolException("Azure DevOps client not initialized.")
            if self.llm is None:
                return ToolException("Image analysis requires an LLM, but this toolkit is configured without one.")
            if attachment_url.lower().startswith('data:'):
                return ToolException("data: URIs are not supported; provide an Azure DevOps attachment URL.")
            if '/_apis/wit/attachments/' not in urllib.parse.urlparse(attachment_url).path.lower():
                return ToolException(
                    "URL must be an Azure DevOps work item attachment URL (containing /_apis/wit/attachments/).")
            ref = self._extract_attachment_ref(attachment_url)
            if ref is None:
                return ToolException("Could not determine the attachment id from the URL.")
            attachment_id, derived_name = ref
            name = file_name or derived_name
            if not name:
                return ToolException("Provide file_name; the URL has no fileName parameter.")
            content, error = self._fetch_validated_image(attachment_id, name)
            if error:
                return ToolException(error[0].upper() + error[1:])
            return parse_file_content(file_content=content, file_name=name, llm=self.llm,
                                      prompt=prompt, image_cache=self._image_cache)
        except Exception as e:
            logger.error(f"Error analyzing attachment image: {e}")
            return ToolException(f"Error analyzing attachment image: {e}")

    def parse_attachment_by_id(self, attachment_id, file_name, image_description_prompt):
        file_content = self.get_attachment_content(attachment_id)
        return parse_file_content(file_content=file_content, file_name=file_name,
                                            llm=self.llm, prompt=image_description_prompt,
                                            image_cache=self._image_cache)

    def get_work_item(self, id: int, fields: Optional[list[str]] = None, as_of: Optional[str] = None, expand: Optional[str] = None, parse_attachments=False, image_description_prompt=None, process_images: bool = True):
        """Get a single work item by ID."""
        try:
            # Validate that the Azure DevOps client is initialized
            if not self._client:
                raise ToolException("Azure DevOps client not initialized.")

            # Fetch the work item
            work_item = self._client.get_work_item(id=id, project=self.project, fields=fields, as_of=as_of, expand=expand)

            # Parse the fields dynamically
            fields_data = work_item.fields
            parsed_item = {"id": work_item.id, "url": f"{self.organization_url}/_workitems/edit/{work_item.id}"}

            # Iterate through the requested fields and add them to the parsed result
            if fields:
                for field in fields:
                    parsed_item[field] = fields_data.get(field, "N/A")
            else:
                parsed_item.update(fields_data)

            # extract relations if any
            relations_data = None
            if expand and str(expand).lower() in ("relations", "all"):
                try:
                    relations_data = getattr(work_item, 'relations', None)
                except KeyError:
                    relations_data = None
            if relations_data:
                parsed_item['relations'] = [relation.as_dict() for relation in relations_data]

            if parse_attachments:
                # describe images in work item fields if present
                field_soups = {}
                image_urls = []
                for field_name, field_value in fields_data.items():
                    if isinstance(field_value, str):
                        soup = BeautifulSoup(field_value, 'html.parser')
                        field_soups[field_name] = soup
                        if process_images:
                            image_urls += [img.get('src') for img in soup.find_all('img') if img.get('src')]
                # This pre-existing surface must keep describing every attachment it
                # describes today, so the image gates that guard the newer surfaces stay off.
                descriptions = self._describe_image_urls(
                    image_urls, image_description_prompt, enforce_image_gates=False,
                    max_images=_WORK_ITEM_IMAGE_CEILING) if image_urls else {}
                for field_name, soup in field_soups.items():
                    for img in soup.find_all('img'):
                        description = descriptions.get(img.get('src'))
                        if description is not None:
                            img['image-description'] = description
                    parsed_item[field_name] = str(soup)
                # parse attached documents if present
                for relation in parsed_item.get('relations', []):
                    # Only process actual file attachments
                    if relation.get('rel') == 'AttachedFile':
                        file_name = relation.get('attributes', {}).get('name')
                        if file_name:
                            try:
                                if process_images:
                                    relation['content'] = self.parse_attachment_by_url(relation['url'], file_name, image_description_prompt=image_description_prompt)
                                else:
                                    relation['content'] = self.parse_attachment_by_url(relation['url'], file_name, image_description_prompt=None)
                            except Exception as att_e:
                                logger.warning(f"Failed to parse attachment {file_name}: {att_e}")


            return parsed_item
        except Exception as e:
            logger.error(f"Error getting work item: {e}")
            return ToolException(f"Error getting work item: {e}")


    def get_comments(self, work_item_id: int, limit_total: Optional[int] = None, include_deleted: Optional[bool] = None, expand: Optional[str] = None, order: Optional[str] = None, process_images: bool = False, image_description_prompt: Optional[str] = None):
        """Get comments for work item by ID."""
        try:
            # Validate that the Azure DevOps client is initialized
            if not self._client:
                raise ToolException("Azure DevOps client not initialized.")

            # Resolve limits to extract in single portion and for whole set of comment
            limit_portion = self.limit
            limit_all = limit_total if limit_total else self.limit

            # Markdown-formatted comments carry images in `text`, HTML ones in `renderedText`;
            # requesting the rendered form gives a uniform surface to scan without overriding
            # an expand the caller chose deliberately.
            effective_expand = expand
            if process_images and expand in (None, "none"):
                effective_expand = "renderedText"

            # Fetch the work item comments
            comments_portion = self._client.get_comments(project=self.project, work_item_id=work_item_id, top=limit_portion, include_deleted=include_deleted, expand=effective_expand, order=order)
            comments_all = []

            while True:
                comments_all += [comment.as_dict() for comment in comments_portion.comments]

                if not comments_portion.continuation_token or len(comments_all) >= limit_all:
                    comments_all = comments_all[:limit_all]
                    break
                else:
                    comments_portion = self._client.get_comments(continuation_token=comments_portion.continuation_token, project=self.project, work_item_id=int(work_item_id), top=3, include_deleted=include_deleted, expand=effective_expand, order=order)

            if not process_images:
                return comments_all
            if self.llm is None:
                logger.warning("Image processing requested for work item comments but no LLM is configured; "
                               "returning raw comments.")
                return comments_all
            if not any(self._comment_has_image_markup(comment) for comment in comments_all):
                return comments_all
            return self._embed_comment_image_descriptions(comments_all, image_description_prompt)
        except Exception as e:
            logger.error(f"Error getting work item comments: {e}")
            return ToolException(f"Error getting work item comments: {e}")

    _COMMENT_TEXT_FIELDS = ('rendered_text', 'renderedText', 'text')

    def _comment_has_image_markup(self, comment):
        for field in self._COMMENT_TEXT_FIELDS:
            value = comment.get(field)
            if isinstance(value, str) and ('<img' in value or '![' in value):
                return True
        return False

    def _embed_comment_image_descriptions(self, comments, prompt):
        image_urls = []
        for comment in comments:
            for field in self._COMMENT_TEXT_FIELDS:
                value = comment.get(field)
                if not isinstance(value, str):
                    continue
                if '<img' in value:
                    image_urls += [img.get('src') for img in BeautifulSoup(value, 'html.parser').find_all('img')
                                   if img.get('src')]
                if '![' in value:
                    image_urls += [url for _, url in _MARKDOWN_IMAGE_PATTERN.findall(value) if url]

        descriptions = self._describe_image_urls(image_urls, prompt, enforce_image_gates=True,
                                                 max_images=_MAX_COMMENT_IMAGES_PER_CALL)

        for comment in comments:
            for field in self._COMMENT_TEXT_FIELDS:
                value = comment.get(field)
                if not isinstance(value, str):
                    continue
                try:
                    comment[field] = self._patch_image_descriptions(value, descriptions)
                except Exception as e:
                    logger.warning(f"Failed to embed image descriptions into comment field '{field}': {e}")
        return comments

    def _patch_image_descriptions(self, value, descriptions):
        if '<img' in value:
            soup = BeautifulSoup(value, 'html.parser')
            for img in soup.find_all('img'):
                description = descriptions.get(img.get('src'))
                if description is not None:
                    img['image-description'] = description
            value = str(soup)
        if '![' in value:
            references = dict.fromkeys(
                (match.group(0), match.group(2)) for match in _MARKDOWN_IMAGE_PATTERN.finditer(value))
            for reference, url in references:
                description = descriptions.get(url)
                if description is None:
                    continue
                # The URL is kept intact so the model can follow up with get_image_by_url.
                value = value.replace(reference, f"{reference}\n[image-description: {description}]")
        return value

    def _get_wiki_artifact_uri(self, wiki_identified: str, page_name: str) -> str:
        """Helper method to construct the artifact URI for a wiki page."""
        if not self._wiki_client:
            raise ToolException("Wiki client not initialized.")
        if not self._core_client:
            raise ToolException("Core client not initialized.")

        # 1. Get Project ID
        project_details = self._core_client.get_project(self.project)
        if not project_details or not project_details.id:
            raise ToolException(f"Could not retrieve project details or ID for project '{self.project}'.")
        project_id = project_details.id
        # logger.info(f"Found project ID: {project_id}")

        # 2. Get Wiki ID
        wiki_details = self._wiki_client.get_wiki(project=self.project, wiki_identifier=wiki_identified)
        if not wiki_details or not wiki_details.id:
            raise ToolException(f"Could not retrieve wiki details or ID for wiki '{wiki_identified}'.")
        wiki_id = wiki_details.id
        # logger.info(f"Found wiki ID: {wiki_id}")

        # 3. Get Wiki Page
        wiki_page = self._wiki_client.get_page(project=self.project, wiki_identifier=wiki_identified, path=page_name)

        # 4. Construct the Artifact URI
        url = f"{project_id}/{wiki_id}{wiki_page.page.path}"
        encoded_url = urllib.parse.quote(url, safe="")
        artifact_uri = f"vstfs:///Wiki/WikiPage/{encoded_url}"
        # logger.info(f"Constructed Artifact URI: {artifact_uri}")
        return artifact_uri

    def link_work_items_to_wiki_page(self, work_item_ids: List[int], wiki_identified: str, page_name: str):
        """Links one or more work items to a specific wiki page using an ArtifactLink."""
        if not work_item_ids:
            return "No work item IDs provided. No links created."
        if not self._client:
            return ToolException("Work item client not initialized.")

        try:
            # 1. Get Artifact URI using helper method
            artifact_uri = self._get_wiki_artifact_uri(wiki_identified, page_name)

            # 2. Define the relation payload using the Artifact URI
            relation = {
                "rel": "ArtifactLink",
                "url": artifact_uri,
                "attributes": {"name": "Wiki Page"} # Standard attribute for wiki links
            }

            patch_document = [
                {
                    "op": 0,
                    "path": "/relations/-",
                    "value": relation
                }
            ]

            # 3. Update each work item
            successful_links = []
            failed_links = {}
            for work_item_id in work_item_ids:
                try:
                    self._client.update_work_item(
                        document=patch_document,
                        id=work_item_id,
                        project=self.project # Assuming work items are in the same project
                    )
                    successful_links.append(str(work_item_id))
                    # logger.info(f"Successfully linked work item {work_item_id} to wiki page '{page_name}'.")
                except Exception as update_e:
                    error_msg = f"Failed to link work item {work_item_id}: {str(update_e)}"
                    logger.error(error_msg)
                    failed_links[str(work_item_id)] = str(update_e)

            # 4. Construct response message
            response = ""
            if successful_links:
                response += f"Successfully linked work items [{', '.join(successful_links)}] to wiki page '{page_name}' in wiki '{wiki_identified}'.\n"
            if failed_links:
                response += f"Failed to link work items: {json.dumps(failed_links)}"

            return response.strip()

        except Exception as e:
            logger.error(f"Error linking work items to wiki page '{page_name}': {str(e)}")
            return ToolException(f"An unexpected error occurred while linking work items to wiki page '{page_name}': {str(e)}")

    def unlink_work_items_from_wiki_page(self, work_item_ids: List[int], wiki_identified: str, page_name: str):
        """Unlinks one or more work items from a specific wiki page by removing the ArtifactLink."""
        if not work_item_ids:
            return "No work item IDs provided. No links removed."
        if not self._client:
            return ToolException("Work item client not initialized.")

        try:
            # 1. Get Artifact URI using helper method
            artifact_uri = self._get_wiki_artifact_uri(wiki_identified, page_name)

            # 2. Process each work item to remove the link
            successful_unlinks = []
            failed_unlinks = {}
            no_link_found = []

            for work_item_id in work_item_ids:
                try:
                    # Get the work item with its relations
                    work_item = self._client.get_work_item(id=work_item_id, project=self.project, expand='Relations')
                    if not work_item or not work_item.relations:
                        no_link_found.append(str(work_item_id))
                        logger.info(f"Work item {work_item_id} has no relations. Skipping unlink.")
                        continue

                    # Find the index of the relation to remove
                    relation_index_to_remove = -1
                    for i, relation in enumerate(work_item.relations):
                        if relation.rel == "ArtifactLink" and relation.url == artifact_uri:
                            relation_index_to_remove = i
                            break

                    if relation_index_to_remove == -1:
                        no_link_found.append(str(work_item_id))
                        # logger.info(f"No link to wiki page '{page_name}' found on work item {work_item_id}.")
                        continue

                    # Create the patch document to remove the relation by index
                    patch_document = [
                        {
                            "op": "remove", # Use "remove" operation
                            "path": f"/relations/{relation_index_to_remove}"
                        }
                    ]

                    # Update the work item
                    self._client.update_work_item(
                        document=patch_document,
                        id=work_item_id,
                        project=self.project
                    )
                    successful_unlinks.append(str(work_item_id))
                    logger.info(f"Successfully unlinked work item {work_item_id} from wiki page '{page_name}'.")

                except Exception as update_e:
                    error_msg = f"Failed to unlink work item {work_item_id}: {str(update_e)}"
                    logger.error(error_msg)
                    failed_unlinks[str(work_item_id)] = str(update_e)

            # 5. Construct response message
            response = ""
            if successful_unlinks:
                response += f"Successfully unlinked work items [{', '.join(successful_unlinks)}] from wiki page '{page_name}' in wiki '{wiki_identified}'.\n"
            if no_link_found:
                 response += f"No link to wiki page '{page_name}' found for work items [{', '.join(no_link_found)}].\n"
            if failed_unlinks:
                response += f"Failed to unlink work items: {json.dumps(failed_unlinks)}"

            return response.strip() if response else "No action taken or required."

        except Exception as e:
            logger.error(f"Error unlinking work items from wiki page '{page_name}': {str(e)}")
            return ToolException(f"An unexpected error occurred while unlinking work items from wiki page '{page_name}': {str(e)}")

    def attach_file_to_work_item(
        self,
        work_item_id: int,
        filepath: str,
        filename: Optional[str] = None,
        inline_field: Optional[str] = None,
        add_as_comment: bool = False,
        comment: Optional[str] = None,
    ):
        """Attach a file from artifact storage to an Azure DevOps work item.

        Uploads the file as an ADO attachment, adds it to the work item as an
        AttachedFile relation, and optionally embeds it inline in an HTML field
        (e.g. System.Description) and/or as a work item comment. Images render
        inline via an <img> tag; other file types are rendered as a link.
        """
        if not self._client:
            return ToolException("Azure DevOps client not initialized.")

        try:
            file_bytes, artifact_filename = get_file_bytes_from_artifact(self.elitea, filepath)
        except Exception as e:
            return ToolException(f"Failed to retrieve artifact '{filepath}': {e}")

        if not file_bytes:
            return ToolException(f"Artifact '{filepath}' not found or empty")

        resolved_filename = filename or artifact_filename
        if not resolved_filename:
            return ToolException("Filename could not be resolved from artifact or arguments.")

        mime_type = detect_mime_type(file_bytes, resolved_filename)
        is_image = mime_type.startswith("image/")

        try:
            attachment_ref = self._client.create_attachment(
                upload_stream=BytesIO(file_bytes),
                project=self.project,
                file_name=resolved_filename,
                upload_type="Simple",
            )
        except Exception as e:
            logger.error(f"Error uploading attachment '{resolved_filename}' to ADO: {e}")
            return ToolException(f"Error uploading attachment '{resolved_filename}': {e}")

        attachment_url = getattr(attachment_ref, "url", None)
        attachment_id = getattr(attachment_ref, "id", None)
        if not attachment_url:
            return ToolException("ADO did not return an attachment URL after upload.")

        relation_value = {"rel": "AttachedFile", "url": attachment_url, "attributes": {"name": resolved_filename}}
        if comment:
            relation_value["attributes"]["comment"] = comment

        patch_document = [{"op": "add", "path": "/relations/-", "value": relation_value}]

        if inline_field:
            try:
                work_item = self._client.get_work_item(id=work_item_id, project=self.project, fields=[inline_field])
                current_value = (work_item.fields or {}).get(inline_field, "") or ""
            except Exception as e:
                logger.warning(f"Could not read field '{inline_field}' on WI {work_item_id}: {e}")
                current_value = ""
            new_value = current_value + self._build_inline_markup(attachment_url, resolved_filename, is_image)
            patch_document.append({"op": "add", "path": f"/fields/{inline_field}", "value": new_value})

        try:
            self._client.update_work_item(document=patch_document, id=work_item_id, project=self.project)
        except Exception as e:
            logger.error(f"Error attaching file to work item {work_item_id}: {e}")
            return ToolException(f"Error attaching file to work item {work_item_id}: {e}")

        if add_as_comment:
            try:
                comment_html = self._build_inline_markup(attachment_url, resolved_filename, is_image)
                self._client.add_comment(
                    request=CommentCreate(text=comment_html),
                    project=self.project,
                    work_item_id=work_item_id,
                )
            except Exception as e:
                logger.warning(f"Attached file but failed to add comment on WI {work_item_id}: {e}")

        return {
            "work_item_id": work_item_id,
            "attachment_id": attachment_id,
            "attachment_url": attachment_url,
            "filename": resolved_filename,
            "mime_type": mime_type,
            "inline_field": inline_field,
            "message": f"File '{resolved_filename}' attached to work item {work_item_id}.",
        }

    @staticmethod
    def _build_inline_markup(url: str, filename: str, is_image: bool) -> str:
        safe_name = filename.replace('"', '&quot;')
        if is_image:
            return f'<div><img src="{url}" alt="{safe_name}" /></div>'
        return f'<div><a href="{url}">{safe_name}</a></div>'

    # Opt-in parallelism: default 1 preserves pre-refactor serial behaviour for
    # existing callers. Callers set `workers=N` to fan out per-doc pipelines.
    _DEFAULT_WORKERS = 1
    # Hard cap on concurrent per-doc pipelines. Higher values risk ADO REST 429s,
    # LLM rate limits, and pgvector pool exhaustion.
    _MAX_WORKERS = 10

    def _base_loader(
        self,
        wiql: str,
        workers: Optional[int] = None,
        process_images: Optional[bool] = None,
        image_description_prompt: Optional[str] = None,
        fields: Optional[List[str]] = None,
        sanitize: Optional[bool] = True,
        **kwargs,
    ) -> Generator[Document, None, None]:
        self._init_indexing_stats()
        # Expose worker count to _save_index_generator (base-doc executor) and
        # any downstream per-doc work. Defaults to _DEFAULT_WORKERS so the tool
        # works out of the box; pass workers=1 to force serial.
        raw_workers = int(workers) if workers else self._DEFAULT_WORKERS
        if raw_workers > self._MAX_WORKERS:
            logger.warning(
                "workers=%s exceeds cap %s (ADO REST quota + pgvector pool "
                "headroom); clamping to %s.",
                raw_workers, self._MAX_WORKERS, self._MAX_WORKERS,
            )
        self._index_workers = max(1, min(raw_workers, self._MAX_WORKERS))
        # Stash the indexing knobs so _fetch_work_item_document (running on a
        # worker thread) can read them without receiving them as arguments.
        self._index_process_images = bool(process_images) if process_images else False
        self._index_image_description_prompt = image_description_prompt
        self._index_fields = list(fields) if fields else None
        # sanitize=True (default): strip HTML tags and collapse identity dicts
        # to their displayName before serializing page_content. sanitize=False
        # restores the pre-refactor behavior (raw HTML strings, full identity
        # dicts). process_images still injects image descriptions when on;
        # they survive sanitize=False as <img image-description="..."> markup.
        self._index_sanitize = True if sanitize is None else bool(sanitize)
        result = self._client.query_by_wiql(Wiql(query=wiql))
        # Flat queries (FROM workitems) populate .work_items; tree/link queries
        # (FROM workitemLinks ... MODE (Recursive)) populate .work_item_relations
        # with .source/.target references and leave .work_items as None.
        work_item_ids = []
        seen = set()
        for ref in result.work_items or []:
            if ref.id not in seen:
                seen.add(ref.id)
                work_item_ids.append(ref.id)
        for rel in result.work_item_relations or []:
            for endpoint in (getattr(rel, 'target', None), getattr(rel, 'source', None)):
                if endpoint is not None and endpoint.id is not None and endpoint.id not in seen:
                    seen.add(endpoint.id)
                    work_item_ids.append(endpoint.id)

        # Fetch work item details concurrently — each get_work_item is an
        # independent REST call, so this is a straight I/O win. Yield in the
        # order of `work_item_ids` so downstream _reduce_duplicates + stats
        # stay deterministic. State mutation stays on the main thread.
        max_workers = max(1, self._index_workers)
        if max_workers <= 1 or len(work_item_ids) <= 1:
            for wi_id in work_item_ids:
                self._track_processed_item()
                yield self._fetch_work_item_document(wi_id)
            return

        executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ado-wi-fetch",
        )
        try:
            pending: Dict[Any, int] = {}  # future -> index (O(1) lookup on completion)
            id_iter = enumerate(work_item_ids)
            next_yield_idx = 0
            ready: Dict[int, Document] = {}

            def _submit_next() -> bool:
                try:
                    idx, wi_id = next(id_iter)
                except StopIteration:
                    return False
                pending[executor.submit(self._fetch_work_item_document, wi_id)] = idx
                return True

            for _ in range(max_workers):
                if not _submit_next():
                    break

            while pending or ready:
                # Drain any contiguous ready items first (in-order yield).
                while next_yield_idx in ready:
                    self._track_processed_item()
                    yield ready.pop(next_yield_idx)
                    next_yield_idx += 1
                if not pending:
                    break
                done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    idx = pending.pop(future)
                    ready[idx] = future.result()
                    _submit_next()
            # Drain any tail
            while next_yield_idx in ready:
                self._track_processed_item()
                yield ready.pop(next_yield_idx)
                next_yield_idx += 1
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _fetch_work_item_document(self, wi_id) -> Document:
        """Fetch one work item and wrap it as a base Document. Pure I/O — safe
        to call from a worker thread. No shared-state mutation.

        Four optional indexing knobs are read from self (set by _base_loader):
        - _index_process_images: describe embedded <img> tags via the LLM
        - _index_image_description_prompt: prompt override for those calls
        - _index_fields: keep only these field reference names in page_content
        - _index_sanitize: strip HTML + flatten identity dicts before dumping
          the payload. Defaults to True; pass sanitize=False to _base_loader
          to preserve the pre-refactor shape.
        """
        process_images = getattr(self, "_index_process_images", False)
        image_prompt = getattr(self, "_index_image_description_prompt", None)
        fields_filter = getattr(self, "_index_fields", None)
        sanitize = getattr(self, "_index_sanitize", True)

        wi = self._client.get_work_item(id=wi_id, project=self.project, expand='all')
        raw_fields = dict(wi.fields or {})

        # Describe embedded images before HTML gets sanitized so the image
        # text survives as [image: ...] in the final payload.
        if process_images:
            field_soups = {}
            image_urls = []
            for name, value in raw_fields.items():
                if not isinstance(value, str) or '<img' not in value:
                    continue
                try:
                    soup = BeautifulSoup(value, 'html.parser')
                    srcs = [img.get('src') for img in soup.find_all('img') if img.get('src')]
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "process_images pass failed for field %s on work item %s: %s",
                        name, wi_id, exc,
                    )
                    continue
                field_soups[name] = soup
                image_urls += srcs
            descriptions = self._describe_image_urls(
                image_urls, image_prompt, enforce_image_gates=False,
                max_images=_WORK_ITEM_IMAGE_CEILING) if image_urls else {}
            for name, soup in field_soups.items():
                try:
                    for img in soup.find_all('img'):
                        src = img.get('src')
                        if not src:
                            continue
                        description = descriptions.get(src)
                        if isinstance(description, _ImageNote) or description is None:
                            # Notes stay out of the payload — failure text would
                            # pollute the index embeddings — but the failure must
                            # still be visible somewhere, and on this unattended
                            # path the log is the only somewhere.
                            logger.warning(
                                "image description failed for %s on work item %s: %s",
                                src, wi_id, description or "data: URI skipped",
                            )
                        elif isinstance(description, str):
                            img['image-description'] = description
                    raw_fields[name] = str(soup)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "process_images pass failed for field %s on work item %s: %s",
                        name, wi_id, exc,
                    )

        selected_names = fields_filter if fields_filter else list(raw_fields.keys())
        filtered: Dict[str, Any] = {}
        for name in selected_names:
            if name not in raw_fields:
                continue
            value = raw_fields[name]
            if sanitize:
                value = self._flatten_identity(value)
                if isinstance(value, str):
                    value = self._sanitize_html(value)
            filtered[name] = value

        return Document(
            page_content=json.dumps(filtered, ensure_ascii=False, default=str),
            metadata={
                'id': str(wi.id),
                'type': raw_fields.get('System.WorkItemType', ''),
                'title': raw_fields.get('System.Title', ''),
                'state': raw_fields.get('System.State', ''),
                'area': raw_fields.get('System.AreaPath', ''),
                'reason': raw_fields.get('System.Reason', ''),
                'iteration': raw_fields.get('System.IterationPath', ''),
                'updated_on': raw_fields.get('System.ChangedDate', ''),
                'attachment_ids': {
                    rel.url.split('/')[-1]: rel.attributes.get('name', '')
                    for rel in wi.relations or [] if rel.rel == 'AttachedFile'
                },
            },
        )

    @staticmethod
    def _flatten_identity(value):
        """Azure identity fields (AssignedTo, CreatedBy, ...) are dicts with a
        displayName plus a bag of avatar URLs and descriptors. Collapse them to
        just the displayName to cut ~1 KB per identity out of the payload."""
        if isinstance(value, dict) and 'displayName' in value:
            return value.get('displayName')
        return value

    @staticmethod
    def _sanitize_html(value: str) -> str:
        """Strip HTML tags while preserving image-description text added by the
        process_images pass. Non-HTML strings are returned unchanged so plain
        field values (dates, ids, paths) do not go through the parser."""
        if '<' not in value:
            return value
        soup = BeautifulSoup(value, 'html.parser')
        for img in soup.find_all('img'):
            description = img.get('image-description') or img.get('alt')
            if description:
                img.replace_with(f"[image: {description}]")
            else:
                img.extract()
        text = soup.get_text(separator='\n')
        lines = [ln.strip() for ln in text.splitlines()]
        return '\n'.join(ln for ln in lines if ln)

    def get_attachment_content(self, attachment_id):
        content_generator = self._client.get_attachment_content(id=attachment_id, download=True)
        return b"".join(content_generator)

    def _process_document(self, document: Document) -> Generator[Document, None, None]:
        raw_attachment_ids = document.metadata.get('attachment_ids', {})

        # Normalize attachment_ids: accept dict or JSON string, raise otherwise
        if isinstance(raw_attachment_ids, str):
            try:
                loaded = json.loads(raw_attachment_ids)
            except json.JSONDecodeError:
                raise TypeError(
                    f"Expected dict or JSON string for 'attachment_ids', got non-JSON string for id="
                    f"{document.metadata.get('id')}: {raw_attachment_ids!r}"
                )
            if not isinstance(loaded, dict):
                raise TypeError(
                    f"'attachment_ids' JSON did not decode to dict for id={document.metadata.get('id')}: {loaded!r}"
                )
            attachment_ids = loaded
        elif isinstance(raw_attachment_ids, dict):
            attachment_ids = raw_attachment_ids
        else:
            raise TypeError(
                f"Expected 'attachment_ids' to be dict or JSON string, got {type(raw_attachment_ids)} "
                f"for id={document.metadata.get('id')}: {raw_attachment_ids!r}"
            )

        for attachment_id, file_name in attachment_ids.items():
            content = self.get_attachment_content(attachment_id=attachment_id)
            yield Document(
                page_content="",
                metadata={
                    'id': attachment_id,
                    IndexerKeywords.CONTENT_FILE_NAME.value: file_name,
                    IndexerKeywords.CONTENT_IN_BYTES.value: content,
                },
            )

    def _index_tool_params(self):
        """Return the parameters for indexing data."""
        return {
            "wiql": (str, Field(description="WIQL (Work Item Query Language) query string to select and filter Azure DevOps work items.")),
            "workers": (Optional[int], Field(
                default=None,
                ge=1,
                le=10,
                description=(
                    "Maximum number of work items processed concurrently. Applies "
                    "to both the initial REST fetch (get_work_item per id) and the "
                    "per-item indexing pipeline (attachments, chunking). Defaults "
                    "to 1 (serial). Capped at 10 to stay within ADO REST quota, "
                    "LLM rate limits, and the pgvector connection pool. Values "
                    "above 10 are clamped."
                ),
            )),
            "process_images": (Optional[bool], Field(
                default=False,
                description=(
                    "If True, scan HTML work-item fields for <img> tags and "
                    "describe each image via the LLM before sanitizing HTML, "
                    "so screenshots inside System.Description, ReproSteps, "
                    "etc. become searchable as text. Costs one LLM call per "
                    "image. Default False."
                ),
            )),
            "image_description_prompt": (Optional[str], Field(
                default=None,
                description=(
                    "Optional prompt to steer image description output. "
                    "Ignored unless process_images=True."
                ),
            )),
            "fields": (Optional[List[str]], Field(
                default=None,
                description=(
                    "Whitelist of work-item field reference names to include "
                    "in the indexed content — e.g. ['System.Title', "
                    "'System.Description', 'System.State', "
                    "'Microsoft.VSTS.Common.AcceptanceCriteria', "
                    "'Microsoft.VSTS.TCM.ReproSteps']. If omitted or empty, "
                    "all fields returned by Azure DevOps are indexed, which "
                    "includes revision/watermark/board-column bookkeeping "
                    "that inflates the payload. Metadata columns (title, "
                    "state, area, iteration, updated_on) on the resulting "
                    "Document are always populated regardless of this list."
                ),
            )),
            "sanitize": (Optional[bool], Field(
                default=True,
                description=(
                    "If True (default), strip HTML tags from string fields "
                    "and collapse Azure identity dicts (AssignedTo, "
                    "CreatedBy, etc.) to their displayName before indexing, "
                    "cutting payload size and making the JSON dump readable. "
                    "Image descriptions inserted by process_images are "
                    "preserved as '[image: ...]' text. Set to False to "
                    "restore the pre-refactor shape (raw HTML strings and "
                    "full identity dicts) — useful if a downstream consumer "
                    "parses the JSON expecting the original schema."
                ),
            )),
        }

    def get_available_tools(self):
        """Return a list of available tools."""
        return super().get_available_tools() + [
            {
                "name": "search_work_items",
                "description": self.search_work_items.__doc__,
                "args_schema": ADOWorkItemsSearch,
                "ref": self.search_work_items,
            },
            {
                "name": "search_work_items_by_text",
                "description": self.search_work_items_by_text.__doc__,
                "args_schema": ADOWorkItemsTextSearch,
                "ref": self.search_work_items_by_text,
            },
            {
                "name": "create_work_item",
                "description": self.create_work_item.__doc__,
                "args_schema": ADOCreateWorkItem,
                "ref": self.create_work_item,
            },
            {
                "name": "update_work_item",
                "description": self.update_work_item.__doc__,
                "args_schema": ADOUpdateWorkItem,
                "ref": self.update_work_item,
            },
            {
                "name": "delete_work_item",
                "description": self.delete_work_item.__doc__,
                "args_schema": ADODeleteWorkItem,
                "ref": self.delete_work_item,
            },
            {
                "name": "get_work_item",
                "description": self.get_work_item.__doc__,
                "args_schema": ADOGetWorkItem,
                "ref": self.get_work_item,
            },
            {
                "name": "link_work_items",
                "description": self.link_work_items.__doc__,
                "args_schema": ADOLinkWorkItem,
                "ref": self.link_work_items,
            },
            {
                "name": "get_relation_types",
                "description": self.get_relation_types.__doc__,
                "args_schema": ADOGetLinkType,
                "ref": self.get_relation_types,
            },
            {
                "name": "get_comments",
                "description": self.get_comments.__doc__,
                "args_schema": ADOGetComments,
                "ref": self.get_comments,
            },
            {
                "name": "get_image_by_url",
                "description": self.get_image_by_url.__doc__,
                "args_schema": ADOGetImageByUrl,
                "ref": self.get_image_by_url,
            },
            {
                "name": "link_work_items_to_wiki_page",
                "description": self.link_work_items_to_wiki_page.__doc__,
                "args_schema": ADOLinkWorkItemsToWikiPage,
                "ref": self.link_work_items_to_wiki_page,
            },
            {
                "name": "unlink_work_items_from_wiki_page",
                "description": self.unlink_work_items_from_wiki_page.__doc__,
                "args_schema": ADOUnlinkWorkItemsFromWikiPage,
                "ref": self.unlink_work_items_from_wiki_page,
            },
            {
                "name": "get_work_item_type_fields",
                "description": self.get_work_item_type_fields.__doc__,
                "args_schema": ADOGetWorkItemTypeFields,
                "ref": self.get_work_item_type_fields,
            },
            {
                "name": "attach_file_to_work_item",
                "description": self.attach_file_to_work_item.__doc__,
                "args_schema": ADOAttachFileToWorkItem,
                "ref": self.attach_file_to_work_item,
            }
        ]
