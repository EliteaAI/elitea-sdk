import json
import logging
import re
import urllib.parse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from io import BytesIO
from typing import Any, Dict, List, Generator, Optional

from azure.devops.connection import Connection
from azure.devops.v7_1.core import CoreClient
from azure.devops.v7_1.wiki import WikiClient
from azure.devops.v7_1.work_item_tracking import TeamContext, Wiql, WorkItemTrackingClient
from azure.devops.v7_1.work_item_tracking.models import CommentCreate
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_core.tools import ToolException
from msrest.authentication import BasicAuthentication
from pydantic import create_model, PrivateAttr, SecretStr
from pydantic import model_validator
from pydantic.fields import Field

from elitea_sdk.tools.non_code_indexer_toolkit import NonCodeIndexerToolkit
from ...utils import get_file_bytes_from_artifact, detect_mime_type
from ...utils.content_parser import parse_file_content
from ....runtime.utils.utils import IndexerKeywords

logger = logging.getLogger(__name__)

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
    order=(Optional[str], Field(description="Order in which the comments should be returned. Possible options are { asc, desc }", default=None))
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
        """Search for work items using a WIQL query and dynamically fetch fields based on the query."""
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

    def parse_attachment_by_url(self, attachment_url, file_name=None, image_description_prompt=None):
        match = re.search(r'attachments/([\w-]+)(?:\?fileName=([^&]+))?', attachment_url)
        if match:
            attachment_id = match.group(1)
            if not file_name:
                file_name = match.group(2)
            if not file_name:
                raise ToolException("File name must be provided either in the URL or as a parameter.")
            return self.parse_attachment_by_id(attachment_id, file_name, image_description_prompt)
        raise ToolException(f"Attachment '{attachment_url}' was not found.")

    def parse_attachment_by_id(self, attachment_id, file_name, image_description_prompt):
        file_content = self.get_attachment_content(attachment_id)
        return parse_file_content(file_content=file_content, file_name=file_name,
                                            llm=self.llm, prompt=image_description_prompt)

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
                for field_name, field_value in fields_data.items():
                    if isinstance(field_value, str):
                        soup = BeautifulSoup(field_value, 'html.parser')
                        images = soup.find_all('img')
                        for img in images:
                            src = img.get('src')
                            if src and process_images:
                                description = self.parse_attachment_by_url(src, image_description_prompt=image_description_prompt)
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


    def get_comments(self, work_item_id: int, limit_total: Optional[int] = None, include_deleted: Optional[bool] = None, expand: Optional[str] = None, order: Optional[str] = None):
        """Get comments for work item by ID."""
        try:
            # Validate that the Azure DevOps client is initialized
            if not self._client:
                raise ToolException("Azure DevOps client not initialized.")

            # Resolve limits to extract in single portion and for whole set of comment
            limit_portion = self.limit
            limit_all = limit_total if limit_total else self.limit

            # Fetch the work item comments
            comments_portion = self._client.get_comments(project=self.project, work_item_id=work_item_id, top=limit_portion, include_deleted=include_deleted, expand=expand, order=order)
            comments_all = []

            while True:
                comments_all += [comment.as_dict() for comment in comments_portion.comments]

                if not comments_portion.continuation_token or len(comments_all) >= limit_all:
                    return comments_all[:limit_all]
                else:
                    comments_portion = self._client.get_comments(continuation_token=comments_portion.continuation_token, project=self.project, work_item_id=int(work_item_id), top=3, include_deleted=include_deleted, expand=expand, order=order)
        except Exception as e:
            logger.error(f"Error getting work item comments: {e}")
            return ToolException(f"Error getting work item comments: {e}")

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
            for name, value in list(raw_fields.items()):
                if not isinstance(value, str) or '<img' not in value:
                    continue
                try:
                    soup = BeautifulSoup(value, 'html.parser')
                    for img in soup.find_all('img'):
                        src = img.get('src')
                        if not src:
                            continue
                        try:
                            description = self.parse_attachment_by_url(
                                src, image_description_prompt=image_prompt,
                            )
                            img['image-description'] = description
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "image description failed for %s on work item %s: %s",
                                src, wi_id, exc,
                            )
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
