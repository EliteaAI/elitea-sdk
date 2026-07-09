import hashlib
import logging
import re
import time
import requests
from typing import Generator, List, Literal, Optional
from urllib.parse import unquote

from azure.devops.connection import Connection
from azure.devops.exceptions import AzureDevOpsServiceError
from azure.devops.v7_0.core import CoreClient
from azure.devops.v7_0.wiki import WikiClient, WikiPageCreateOrUpdateParameters, WikiCreateParametersV2, \
    WikiPageMoveParameters
from azure.devops.v7_0.wiki.models import GitVersionDescriptor, WikiPagesBatchRequest
from langchain_core.documents import Document
from langchain_core.tools import ToolException
from msrest.authentication import BasicAuthentication
from pydantic import create_model, PrivateAttr, SecretStr, BaseModel
from pydantic import model_validator
from pydantic.fields import Field

import elitea_sdk.tools.ado.work_item
from ..repos import ReposApiWrapper
from ...non_code_indexer_toolkit import NonCodeIndexerToolkit
from ...utils.available_tools_decorator import extend_with_parent_available_tools
from ...utils.content_parser import parse_file_content
from ....runtime.utils.utils import IndexerKeywords

logger = logging.getLogger(__name__)

# Azure DevOps REST resource GUID for POST /wiki/wikis/{wikiIdentifier}/pagesBatch.
# Public API contract, mirrored from azure/devops/v7_0/wiki/wiki_client.get_pages_batch.
# We call Client._send directly (bypassing WikiClient.get_pages_batch) so we can read
# the x-ms-continuationtoken response header for pagination.
ADO_WIKI_PAGES_BATCH_LOCATION_ID = "71323c46-2592-4398-8771-ced73dd87207"

GetWikiInput = create_model(
    "GetWikiInput",
    wiki_identified=(Optional[str], Field(default=None, description="Wiki ID or wiki name. If not provided, uses the default wiki identifier from toolkit configuration."))
)

GetPageByPathInput = create_model(
    "GetPageByPathInput",
    wiki_identified=(Optional[str], Field(default=None, description="Wiki ID or wiki name. If not provided, uses the default wiki identifier from toolkit configuration.")),
    page_name=(str, Field(description="Wiki page path")),
    image_description_prompt=(Optional[str],
                              Field(description="Prompt which is used for image description", default=None)),
    process_images=(Optional[bool], Field(default=True, description="Whether to process images in page content. Set to False to get raw content without image description processing."))
)

GetPageByIdInput = create_model(
    "GetPageByIdInput",
    wiki_identified=(Optional[str], Field(default=None, description="Wiki ID or wiki name. If not provided, uses the default wiki identifier from toolkit configuration.")),
    page_id=(int, Field(description="Wiki page ID")),
    image_description_prompt=(Optional[str],
                              Field(description="Prompt which is used for image description", default=None)),
    process_images=(Optional[bool], Field(default=True, description="Whether to process images in page content. Set to False to get raw content without image description processing."))
)


class GetPageInput(BaseModel):
    """Input schema for get_wiki_page tool with validation."""
    wiki_identified: Optional[str] = Field(default=None, description="Wiki ID or wiki name. If not provided, uses the default wiki identifier from toolkit configuration.")
    page_path: Optional[str] = Field(default=None, description="Wiki page path (e.g., '/MB_Heading/MB_2')")
    page_id: Optional[int] = Field(default=None, description="Wiki page ID")
    include_content: Optional[bool] = Field(default=False, description="Whether to include page content in the response. If True, content will be processed for image descriptions.")
    image_description_prompt: Optional[str] = Field(default=None, description="Prompt which is used for image description when include_content is True")
    process_images: Optional[bool] = Field(default=True, description="Whether to process images in page content. Set to False to get raw content without image description processing.")
    recursion_level: Optional[Literal['none', 'oneLevel', 'oneLevelPlusNestedEmptyFolders', 'full']] = Field(
        default="oneLevel",
        description="Controls how many levels of sub-pages are retrieved along with the main page. "
                    "Options: 'none' (No subpages retrieved - only the requested page metadata), "
                    "'oneLevel' (Direct children only - immediate sub-pages) [default], "
                    "'oneLevelPlusNestedEmptyFolders' (Direct children plus recursive chains of nested child folders that only contain a single folder), "
                    "'full' (All descendants - entire page hierarchy)."
    )

    @model_validator(mode='before')
    @classmethod
    def validate_inputs(cls, values):
        """Validator to ensure at least one of page_path or page_id is provided."""
        page_path = values.get('page_path')
        page_id = values.get('page_id')
        if not page_path and not page_id:
            raise ValueError("At least one of 'page_path' or 'page_id' must be provided")
        return values


DeletePageByPathInput = create_model(
    "DeletePageByPathInput",
    wiki_identified=(Optional[str], Field(default=None, description="Wiki ID or wiki name. If not provided, uses the default wiki identifier from toolkit configuration.")),
    page_name=(str, Field(description="Wiki page path")),
)

DeletePageByIdInput = create_model(
    "DeletePageByIdInput",
    wiki_identified=(Optional[str], Field(default=None, description="Wiki ID or wiki name. If not provided, uses the default wiki identifier from toolkit configuration.")),
    page_id=(int, Field(description="Wiki page ID")),
)

ModifyPageInput = create_model(
    "ModifyPageInput",
    wiki_identified=(Optional[str], Field(default=None, description="Wiki ID or wiki name. If not provided, uses the default wiki identifier from toolkit configuration.")),
    page_name=(str, Field(description="Wiki page name")),
    page_content=(str, Field(description="Wiki page content")),
    version_identifier=(str, Field(description="Version string identifier (name of tag/branch, SHA1 of commit). Usually for wiki the branch is 'wikiMaster'")),
    version_type=(Optional[str], Field(description="Version type (branch, tag, or commit). Determines how Id is interpreted", default="branch")),
    expanded=(Optional[bool], Field(description="Whether to return the full page object or just its simplified version.", default=False))
)

RenamePageInput = create_model(
    "RenamePageInput",
    wiki_identified=(Optional[str], Field(default=None, description="Wiki ID or wiki name. If not provided, uses the default wiki identifier from toolkit configuration.")),
    old_page_name=(str, Field(description="Old Wiki page name to be renamed", examples= ["/TestPageName"])),
    new_page_name=(str, Field(description="New Wiki page name", examples= ["/RenamedName"])),
    version_identifier=(str, Field(description="Version string identifier (name of tag/branch, SHA1 of commit)")),
    version_type=(Optional[str], Field(description="Version type (branch, tag, or commit). Determines how Id is interpreted", default="branch"))
)


def _format_wiki_page_response(wiki_page_response, expanded: bool = False, include_content: bool = False):
    """Format wiki page response.

    Args:
        wiki_page_response: The WikiPageResponse object from Azure DevOps API
        expanded: If True, returns comprehensive page metadata. If False, returns simplified format.
        include_content: If True and expanded=True, includes the page content in the response.

    Returns:
        Dictionary with eTag and page information. Format depends on expanded parameter.
    """
    try:
        if expanded:
            # Comprehensive metadata format
            page = wiki_page_response.page

            # Process sub_pages if present
            sub_pages = []
            if page and hasattr(page, 'sub_pages') and page.sub_pages:
                for sub_page in page.sub_pages:
                    sub_page_dict = {
                        'id': sub_page.id if hasattr(sub_page, 'id') else None,
                        'path': sub_page.path if hasattr(sub_page, 'path') else None,
                        'order': sub_page.order if hasattr(sub_page, 'order') else None,
                        'git_item_path': sub_page.git_item_path if hasattr(sub_page, 'git_item_path') else None,
                        'url': sub_page.url if hasattr(sub_page, 'url') else None,
                        'remote_url': sub_page.remote_url if hasattr(sub_page, 'remote_url') else None,
                    }
                    # Recursively process nested sub_pages if present
                    if hasattr(sub_page, 'sub_pages') and sub_page.sub_pages:
                        sub_page_dict['sub_pages'] = [
                            {
                                'id': sp.id if hasattr(sp, 'id') else None,
                                'path': sp.path if hasattr(sp, 'path') else None,
                                'order': sp.order if hasattr(sp, 'order') else None,
                            }
                            for sp in sub_page.sub_pages
                        ]
                    sub_pages.append(sub_page_dict)

            result = {
                'eTag': wiki_page_response.eTag,
                'page': {
                    'id': page.id if page else None,
                    'path': page.path if page else None,
                    'git_item_path': page.git_item_path if page and hasattr(page, 'git_item_path') else None,
                    'remote_url': page.remote_url if page and hasattr(page, 'remote_url') else None,
                    'url': page.url if page else None,
                    'order': page.order if page and hasattr(page, 'order') else None,
                    'is_parent_page': page.is_parent_page if page and hasattr(page, 'is_parent_page') else None,
                    'is_non_conformant': page.is_non_conformant if page and hasattr(page, 'is_non_conformant') else None,
                    'sub_pages': sub_pages,
                }
            }
            # Include content if requested
            if include_content and page and hasattr(page, 'content'):
                result['page']['content'] = page.content
            return result
        else:
            # Simplified format for backward compatibility
            return {
                "eTag": wiki_page_response.eTag,
                "id": wiki_page_response.page.id,
                "page": wiki_page_response.page.url
            }
    except Exception as e:
        logger.error(f"Unable to format wiki page response: {wiki_page_response}, error: {str(e)}")
        return wiki_page_response


def _format_wiki_response(wiki_response):
    """Format wiki response to a serializable dictionary.

    Args:
        wiki_response: The WikiV2 object from Azure DevOps API

    Returns:
        Dictionary with wiki information that is msgpack/JSON serializable.
    """
    try:
        result = {
            'id': wiki_response.id if hasattr(wiki_response, 'id') else None,
            'name': wiki_response.name if hasattr(wiki_response, 'name') else None,
            'type': wiki_response.type if hasattr(wiki_response, 'type') else None,
            'url': wiki_response.url if hasattr(wiki_response, 'url') else None,
            'project_id': wiki_response.project_id if hasattr(wiki_response, 'project_id') else None,
            'repository_id': wiki_response.repository_id if hasattr(wiki_response, 'repository_id') else None,
            'mapped_path': wiki_response.mapped_path if hasattr(wiki_response, 'mapped_path') else None,
        }

        # Add optional fields if present
        if hasattr(wiki_response, 'remote_url') and wiki_response.remote_url:
            result['remote_url'] = wiki_response.remote_url

        # Format versions list - each version is a GitVersionDescriptor object
        if hasattr(wiki_response, 'versions') and wiki_response.versions:
            result['versions'] = []
            for version_descriptor in wiki_response.versions:
                if hasattr(version_descriptor, 'version'):
                    # Extract primitive values from GitVersionDescriptor
                    version_dict = {
                        'version': version_descriptor.version if hasattr(version_descriptor, 'version') else None,
                        'version_type': version_descriptor.version_type if hasattr(version_descriptor, 'version_type') else None,
                        'version_options': version_descriptor.version_options if hasattr(version_descriptor, 'version_options') else None,
                    }
                    result['versions'].append(version_dict)
                else:
                    # Fallback if it's already a string or dict
                    result['versions'].append(version_descriptor)

        return result
    except Exception as e:
        logger.error(f"Unable to format wiki response: {wiki_response}, error: {str(e)}")
        # Fallback to string representation
        return {"error": f"Unable to format wiki response: {str(e)}"}


class AzureDevOpsApiWrapper(NonCodeIndexerToolkit):
    # TODO use ado_configuration instead of organization_url, project and token
    organization_url: str
    project: str
    token: SecretStr
    default_wiki_identifier: Optional[str] = None
    _client: Optional[WikiClient] = PrivateAttr()  # Private attribute for the wiki client
    _core_client: Optional[CoreClient] = PrivateAttr()  # Private attribute for the CoreClient client

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types (e.g., WorkItemTrackingClient)

    @model_validator(mode='before')
    @classmethod
    def validate_toolkit(cls, values):
        """Validate and set up the Azure DevOps client."""
        try:
            # Set up connection to Azure DevOps using Personal Access Token (PAT)
            credentials = BasicAuthentication('', values['token'])
            connection = Connection(base_url=values['organization_url'], creds=credentials)

            # Retrieve the work item tracking client and assign it to the private _client attribute
            cls._client = connection.clients.get_wiki_client()
            cls._core_client = connection.clients.get_core_client()

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

    def _resolve_wiki_identifier(self, wiki_identified: Optional[str] = None) -> str:
        """Resolve wiki identifier from parameter or default configuration.

        Args:
            wiki_identified: Optional wiki identifier parameter passed to the tool

        Returns:
            Resolved wiki identifier string

        Raises:
            ToolException: If neither parameter nor default wiki identifier is provided
        """
        if wiki_identified:
            return wiki_identified

        if self.default_wiki_identifier:
            return self.default_wiki_identifier

        raise ToolException(
            "Wiki identifier must be provided either as a parameter or configured as default in toolkit settings. "
            "Please either pass 'wiki_identified' parameter or configure 'default_wiki_identifier' in the toolkit."
        )

    def get_wiki(self, wiki_identified: Optional[str] = None):
        """Extract ADO wiki information."""
        try:
            wiki_id = self._resolve_wiki_identifier(wiki_identified)
            wiki_response = self._client.get_wiki(project=self.project, wiki_identifier=wiki_id)
            return _format_wiki_response(wiki_response)
        except Exception as e:
            logger.error(f"Error during the attempt to extract wiki: {str(e)}")
            return ToolException(f"Error during the attempt to extract wiki: {str(e)}")

    def get_wiki_page_by_path(self, wiki_identified: Optional[str] = None, page_name: str = None, image_description_prompt=None, process_images: bool = True):
        """Extract ADO wiki page content."""
        try:
            wiki_id = self._resolve_wiki_identifier(wiki_identified)
            content = self._client.get_page(project=self.project, wiki_identifier=wiki_id, path=page_name,
                                            include_content=True).page.content
            if process_images:
                return self._process_images(content, image_description_prompt=image_description_prompt, wiki_identified=wiki_id)
            return content
        except Exception as e:
            logger.error(f"Error during the attempt to extract wiki page: {str(e)}")
            return ToolException(f"Error during the attempt to extract wiki page: {str(e)}")

    def get_wiki_page_by_id(self, wiki_identified: Optional[str] = None, page_id: int = None, image_description_prompt=None, process_images: bool = True):
        """Extract ADO wiki page content."""
        try:
            wiki_id = self._resolve_wiki_identifier(wiki_identified)
            content = self._client.get_page_by_id(project=self.project, wiki_identifier=wiki_id, id=page_id,
                                                   include_content=True).page.content
            if process_images:
                return self._process_images(content, image_description_prompt=image_description_prompt, wiki_identified=wiki_id)
            return content
        except Exception as e:
            logger.error(f"Error during the attempt to extract wiki page: {str(e)}")
            return ToolException(f"Error during the attempt to extract wiki page: {str(e)}")

    def get_wiki_page(self, wiki_identified: Optional[str] = None, page_path: Optional[str] = None, page_id: Optional[int] = None,
                      include_content: bool = False, image_description_prompt: Optional[str] = None,
                      process_images: bool = True, recursion_level: Literal['none', 'oneLevel', 'oneLevelPlusNestedEmptyFolders', 'full'] = "oneLevel"):
        """Get wiki page metadata and optionally content.

        Retrieves comprehensive metadata for a wiki page including eTag, id, path, git_item_path,
        remote_url, url, sub_pages, order, and other properties. Optionally includes page content.
        Supports lookup by either page_id (takes precedence) or page_path.

        Args:
            wiki_identified: Wiki ID or wiki name. If not provided, uses default from toolkit configuration.
            page_path: Wiki page path (e.g., '/MB_Heading/MB_2'). Optional if page_id is provided.
            page_id: Wiki page ID. Optional if page_path is provided. Takes precedence over page_path.
            include_content: Whether to include page content in response. Defaults to False (metadata only).
            image_description_prompt: Optional prompt for image description when include_content is True.
            process_images: Whether to process/describe images found in page content. Set to False to skip
                           image processing and return raw content. Defaults to True.
            recursion_level: Level of recursion to retrieve sub-pages. Options: 'none' (no subpages),
                           'oneLevel' (direct children only), 'oneLevelPlusNestedEmptyFolders' (direct children
                           plus recursive chains of nested child folders that only contain a single folder),
                           'full' (all descendants). Defaults to 'oneLevel'.

        Returns:
            Dictionary containing eTag and comprehensive page metadata including id, path, git_item_path,
            remote_url, url, sub_pages, order, is_parent_page, is_non_conformant, and optionally content.

        Raises:
            ToolException: If page/wiki not found, authentication fails, or other errors occur.
        """
        try:
            # Resolve wiki identifier
            wiki_id = self._resolve_wiki_identifier(wiki_identified)

            # Validate that at least one identifier is provided
            if not page_path and not page_id:
                raise ToolException("At least one of 'page_path' or 'page_id' must be provided")

            # Fetch page using page_id (priority) or page_path
            if page_id:
                logger.info(f"Fetching wiki page by ID: {page_id} from wiki: {wiki_id}")
                wiki_page_response = self._client.get_page_by_id(
                    project=self.project,
                    wiki_identifier=wiki_id,
                    id=page_id,
                    include_content=include_content,
                    recursion_level=recursion_level
                )
            else:
                logger.info(f"Fetching wiki page by path: {page_path} from wiki: {wiki_id}")
                wiki_page_response = self._client.get_page(
                    project=self.project,
                    wiki_identifier=wiki_id,
                    path=page_path,
                    include_content=include_content,
                    recursion_level=recursion_level
                )

            # Format response with comprehensive metadata
            result = _format_wiki_page_response(
                wiki_page_response,
                expanded=True,
                include_content=include_content
            )

            # Process images in content if requested
            if include_content and process_images and result.get('page', {}).get('content'):
                processed_content = self._process_images(
                    result['page']['content'],
                    image_description_prompt=image_description_prompt,
                    wiki_identified=wiki_id
                )
                result['page']['content'] = processed_content

            return result

        except AzureDevOpsServiceError as e:
            error_msg = str(e).lower()

            # Page not found errors
            if "404" in error_msg or "not found" in error_msg or "does not exist" in error_msg:
                identifier = f"ID {page_id}" if page_id else f"path '{page_path}'"
                wiki_id = wiki_identified or self.default_wiki_identifier or "unknown"
                logger.error(f"Page {identifier} not found in wiki '{wiki_id}': {str(e)}")
                return ToolException(
                    f"Page {identifier} not found in wiki '{wiki_id}'. "
                    f"Please verify the page exists and the identifier is correct."
                )

            # Path validation errors
            elif "path" in error_msg and ("correct" in error_msg or "invalid" in error_msg):
                wiki_id = wiki_identified or self.default_wiki_identifier or "unknown"
                logger.error(f"Invalid page path '{page_path}' in wiki '{wiki_id}': {str(e)}")
                return ToolException(
                    f"Invalid page path '{page_path}'. Please ensure the path format is correct (e.g., '/PageName')."
                )

            # Wiki not found errors
            elif "wiki" in error_msg and ("not found" in error_msg or "does not exist" in error_msg):
                wiki_id = wiki_identified or self.default_wiki_identifier or "unknown"
                logger.error(f"Wiki '{wiki_id}' not found: {str(e)}")
                return ToolException(
                    f"Wiki '{wiki_id}' not found. Please verify the wiki identifier is correct."
                )

            # Authentication/authorization errors
            elif "401" in error_msg or "unauthorized" in error_msg or "authentication" in error_msg:
                wiki_id = wiki_identified or self.default_wiki_identifier or "unknown"
                logger.error(f"Authentication failed for wiki '{wiki_id}': {str(e)}")
                return ToolException(
                    f"Authentication failed. Please check your access token is valid and has permission to access wiki '{wiki_id}'."
                )

            elif "403" in error_msg or "forbidden" in error_msg or "permission" in error_msg:
                wiki_id = wiki_identified or self.default_wiki_identifier or "unknown"
                logger.error(f"Permission denied for wiki '{wiki_id}': {str(e)}")
                return ToolException(
                    f"Permission denied. You do not have access to wiki '{wiki_id}' or page {page_id if page_id else page_path}."
                )

            # Generic Azure DevOps service errors
            else:
                logger.error(f"Azure DevOps service error while fetching page: {str(e)}")
                return ToolException(f"Error accessing wiki page: {str(e)}")

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return ToolException(f"Validation error: {str(e)}")

        except Exception as e:
            error_msg = str(e).lower()

            # Timeout errors
            if "timeout" in error_msg or "timed out" in error_msg:
                logger.error(f"Connection timeout while fetching page: {str(e)}")
                return ToolException(
                    f"Connection timeout. Please check your network connection and try again."
                )

            # Generic errors
            logger.error(f"Unexpected error during wiki page retrieval: {str(e)}")
            return ToolException(f"Unexpected error during wiki page retrieval: {str(e)}")

    def _get_repos_wrapper(self, wiki_identified: str) -> Optional["ReposApiWrapper"]:
        """Return a ReposApiWrapper bound to the wikiMaster branch of the given wiki.

        Cached per wiki identifier on the instance so index-time processing of many
        pages doesn't rebuild the wrapper (and its azure-devops Connection) per page.
        Returns None when the wiki cannot be resolved — callers must handle that.
        """
        cache = self.__dict__.setdefault("_repos_wrapper_cache", {})
        cached = cache.get(wiki_identified)
        if cached is not None:
            return cached
        try:
            wiki = self._client.get_wiki(project=self.project, wiki_identifier=wiki_identified)
            repos_wrapper = ReposApiWrapper(
                organization_url=self.organization_url,
                project=self.project,
                token=self.token.get_secret_value(),
                repository_id=wiki.repository_id,
                base_branch="wikiMaster",
                active_branch="wikiMaster",
                llm=self.llm,
            )
        except Exception as e:
            logger.error(f"Failed to initialize repos wrapper for wiki '{wiki_identified}': {str(e)}")
            return None
        cache[wiki_identified] = repos_wrapper
        return repos_wrapper

    def _process_images(self, page_content: str, wiki_identified: str, image_description_prompt=None):
        if image_description_prompt and self.llm is None:
            raise ToolException(
                "Cannot generate image descriptions: image_description_prompt was provided but no LLM is configured. "
                "Either initialize the toolkit with an LLM or omit image_description_prompt."
            )

        image_pattern = r"!\[(.*?)\]\((.*?)\)"
        matches = re.findall(image_pattern, page_content)

        # Initialize repos_wrapper once for all attachments in this page
        repos_wrapper = None
        has_attachments = any(url.startswith("/.attachments/") for _, url in matches)

        if has_attachments:
            repos_wrapper = self._get_repos_wrapper(wiki_identified)

        for image_name, image_url in matches:
            # BUG 1 fix: skip images with empty URLs — cannot fetch or resolve them
            if not image_url:
                logger.warning(f"Skipping image '{image_name}': empty URL, leaving original markdown unchanged.")
                continue

            if image_url.startswith("/.attachments/"):
                try:
                    if repos_wrapper is None:
                        raise Exception("Repos wrapper not initialized")
                    description = self.process_attachment(attachment_url=image_url,
                                                          attachment_name=image_name,
                                                          image_description_prompt=image_description_prompt,
                                                          repos_wrapper=repos_wrapper)
                except Exception as e:
                    # Skip rather than replace with a corrupted description
                    logger.warning(f"Skipping image '{image_name}': error parsing attachment '{image_url}': {str(e)}")
                    continue
            else:
                try:
                    response = requests.get(image_url)
                    response.raise_for_status()
                    file_content = response.content
                    description = parse_file_content(
                        file_content=file_content,
                        file_name="image.png",
                        llm=self.llm,
                        prompt=image_description_prompt
                    )
                except Exception as e:
                    # BUG 2 fix: skip rather than replace with a corrupted literal description
                    logger.warning(f"Skipping image '{image_name}': error fetching external image '{image_url}': {str(e)}")
                    continue

            new_image_markdown = f"![{image_name}]({description})"
            page_content = page_content.replace(f"![{image_name}]({image_url})", new_image_markdown)
        return page_content

    def process_attachment(self, attachment_url, attachment_name, repos_wrapper, image_description_prompt):
        file_path = unquote(attachment_url.lstrip('/'))
        attachment_content = repos_wrapper.download_file(path=file_path)
        return parse_file_content(
            file_content=attachment_content,
            file_name=attachment_name,
            llm=self.llm,
            prompt=image_description_prompt
        )

    def delete_page_by_path(self, wiki_identified: Optional[str] = None, page_name: str = None):
        """Delete ADO wiki page by path."""
        try:
            wiki_id = self._resolve_wiki_identifier(wiki_identified)
            self._client.delete_page(project=self.project, wiki_identifier=wiki_id, path=page_name)
            return f"Page '{page_name}' in wiki '{wiki_id}' has been deleted"
        except Exception as e:
            logger.error(f"Unable to delete wiki page: {str(e)}")
            return ToolException(f"Unable to delete wiki page: {str(e)}")

    def delete_page_by_id(self, wiki_identified: Optional[str] = None, page_id: int = None):
        """Delete ADO wiki page by ID."""
        try:
            wiki_id = self._resolve_wiki_identifier(wiki_identified)
            self._client.delete_page_by_id(project=self.project, wiki_identifier=wiki_id, id=page_id)
            return f"Page with id '{page_id}' in wiki '{wiki_id}' has been deleted"
        except Exception as e:
            logger.error(f"Unable to delete wiki page: {str(e)}")
            return ToolException(f"Unable to delete wiki page: {str(e)}")

    def rename_wiki_page(self, wiki_identified: Optional[str] = None, old_page_name: str = None, new_page_name: str = None, version_identifier: str = None,
                         version_type: str = "branch"):
        """Rename page

        Args:
         wiki_identified (str): The identifier for the wiki. If not provided, uses default from toolkit configuration.
         old_page_name (str): The current name of the page to be renamed (e.g. '/old_page_name').
         new_page_name (str): The new name for the page (e.g. '/new_page_name').
         version_identifier (str): The identifier for the version (e.g., branch or commit). Defaults to None.
         version_type (str, optional): The type of version identifier. Defaults to "branch".
     """

        try:
            wiki_id = self._resolve_wiki_identifier(wiki_identified)
            try:
                return self._client.create_page_move(
                    project=self.project,
                    wiki_identifier=wiki_id,
                    comment=f"Page rename from '{old_page_name}' to '{new_page_name}'",
                    page_move_parameters=WikiPageMoveParameters(new_path=new_page_name, path=old_page_name),
                    version_descriptor=GitVersionDescriptor(version=version_identifier, version_type=version_type)
                )
            except AzureDevOpsServiceError as e:
                if "The version '{0}' either is invalid or does not exist." in str(e):
                    # Retry the request without version_descriptor
                    return self._client.create_page_move(
                        project=self.project,
                        wiki_identifier=wiki_id,
                        comment=f"Page rename from '{old_page_name}' to '{new_page_name}'",
                        page_move_parameters=WikiPageMoveParameters(new_path=new_page_name, path=old_page_name),
                    )
                else:
                    raise
        except Exception as e:
            logger.error(f"Unable to rename wiki page: {str(e)}")
            return ToolException(f"Unable to rename wiki page: {str(e)}")

    def modify_wiki_page(self, wiki_identified: Optional[str] = None, page_name: str = None, page_content: str = None, version_identifier: str = None, version_type: str = "branch", expanded: Optional[bool] = False):
        """Create or Update ADO wiki page content."""
        try:
            wiki_id = self._resolve_wiki_identifier(wiki_identified)
            all_wikis = [wiki.name for wiki in self._client.get_all_wikis(project=self.project)]
            if wiki_id not in all_wikis:
                logger.info(f"wiki name '{wiki_id}' doesn't exist. New wiki will be created.")
                try:
                    project_id = None
                    projects = self._core_client.get_projects()

                    for project in projects:
                        if project.name == self.project:
                            project_id = project.id
                            break
                    if project_id:
                        self._client.create_wiki(project=self.project, wiki_create_params=WikiCreateParametersV2(name=wiki_id, project_id=project_id))
                    else:
                        return "Project ID has not been found."
                except Exception as create_wiki_e:
                    return ToolException(f"Unable to create new wiki due to error: {create_wiki_e}")
            try:
                page = self._client.get_page(project=self.project, wiki_identifier=wiki_id, path=page_name)
                version = page.eTag
            except Exception as get_page_e:
                if "Ensure that the path of the page is correct and the page exists" in str(get_page_e):
                    logger.info("Path is not found. New page will be created")
                    version = None
                else:
                    return ToolException(f"Unable to extract page by path {page_name}: {str(get_page_e)}")

            try:
                return _format_wiki_page_response(self._client.create_or_update_page(
                    project=self.project,
                    wiki_identifier=wiki_id,
                    path=page_name,
                    parameters=WikiPageCreateOrUpdateParameters(content=page_content),
                    version=version,
                    version_descriptor=GitVersionDescriptor(version=version_identifier, version_type=version_type)
                ), expanded=expanded)
            except AzureDevOpsServiceError as e:
                if "The version '{0}' either is invalid or does not exist." in str(e):
                    # Retry the request without version_descriptor
                    return _format_wiki_page_response(wiki_page_response=self._client.create_or_update_page(
                        project=self.project,
                        wiki_identifier=wiki_id,
                        path=page_name,
                        parameters=WikiPageCreateOrUpdateParameters(content=page_content),
                        version=version
                    ), expanded=expanded)
                else:
                    raise
        except Exception as e:
            logger.error(f"Unable to modify wiki page: {str(e)}")
            return ToolException(f"Unable to modify wiki page: {str(e)}")

    def _iter_wiki_pages(self, wiki_identifier: str, batch_size: int = 100) -> Generator:
        """Yield every WikiPageDetail across all batches.

        The SDK's WikiClient.get_pages_batch returns only the deserialized body and
        discards the raw response, so the x-ms-continuationtoken header — which
        drives pagination — is unreachable through the public API. We reproduce
        the same call via the underlying Client._send so we can read the header.
        """
        route_values = {
            'project': self._client._serialize.url('project', self.project, 'str'),
            'wikiIdentifier': self._client._serialize.url('wiki_identifier', wiki_identifier, 'str'),
        }
        continuation_token: Optional[str] = None
        while True:
            request_body = WikiPagesBatchRequest(top=batch_size, continuation_token=continuation_token)
            content = self._client._serialize.body(request_body, 'WikiPagesBatchRequest')
            response = self._client._send(
                http_method='POST',
                location_id=ADO_WIKI_PAGES_BATCH_LOCATION_ID,
                version='7.0',
                route_values=route_values,
                content=content,
            )
            for page in self._client._deserialize('[WikiPageDetail]', self._client._unwrap_collection(response)):
                yield page
            continuation_token = response.headers.get('x-ms-continuationtoken')
            if not continuation_token:
                break

    _INDEXER_ATTACHMENTS_META_KEY = "_ado_wiki_attachments"

    _ATTACHMENT_MARKDOWN_RE = re.compile(r"!?\[(?:[^\]]*)\]\(([^)]+)\)")

    def _extract_attachment_paths(self, page_content: str) -> List[str]:
        """Return unique /.attachments/ paths referenced anywhere in page_content.

        Matches both image markdown (``![alt](url)``) and regular link markdown
        (``[text](url)``) so that non-image attachments — PDFs, .md/.docx files,
        anything a user drops onto a wiki page — are also picked up. Only
        wiki-repo attachments are returned; external URLs are not stored in the
        wiki git repo and cannot be pulled as dependent docs.
        """
        seen: dict[str, None] = {}
        for url in self._ATTACHMENT_MARKDOWN_RE.findall(page_content or ""):
            url = url.strip().split(" ", 1)[0]  # strip an optional markdown title: [t](url "title")
            if url and url.startswith("/.attachments/"):
                seen.setdefault(url, None)
        return list(seen.keys())

    def _base_loader(self, wiki_identifier: Optional[str] = None, chunking_tool: str = None,
                     path_contains: Optional[str] = None,
                     process_images: bool = False,
                     image_description_prompt: Optional[str] = None,
                     include_attachments: bool = False,
                     include_extensions: Optional[List[str]] = None,
                     skip_extensions: Optional[List[str]] = None,
                     **kwargs) -> Generator[Document, None, None]:
        self._init_indexing_stats()
        wiki_identifier = self._resolve_wiki_identifier(wiki_identifier)

        # Stash indexing-time flags on the instance so _process_document (called
        # later by the base indexer, without kwargs) can read them.
        self._index_wiki_identifier = wiki_identifier
        self._index_process_images = bool(process_images)
        self._index_image_description_prompt = image_description_prompt
        self._index_include_attachments = bool(include_attachments)
        self._index_include_extensions = include_extensions or []
        self._index_skip_extensions = skip_extensions or []

        pages = self._iter_wiki_pages(wiki_identifier)
        # Normalize hyphens to spaces so users can pass either the URL slug form
        # ("Feature-Analysis-and-Background") or the display form ("Feature Analysis and Background").
        needle = path_contains.lower().replace("-", " ") if path_contains else None
        for page in pages:
            self._indexing_stats.total_fetched += 1
            title = page.path.rsplit("/", 1)[-1]
            if needle and needle not in page.path.lower().replace("-", " "):
                self._track_skipped_document(page.path, reason="filtered")
                continue
            self._track_processed_item()
            raw_content = self._client.get_page_by_id(project=self.project, wiki_identifier=wiki_identifier, id=page.id, include_content=True).page.content
            # Hash the raw source, not the LLM-augmented version — LLM output is
            # non-deterministic and would break incremental dedup across runs.
            content_hash = hashlib.sha256((raw_content or "").encode("utf-8")).hexdigest()

            content = raw_content
            if self._index_process_images:
                try:
                    content = self._process_images(
                        raw_content,
                        wiki_identified=wiki_identifier,
                        image_description_prompt=image_description_prompt,
                    )
                except Exception as e:
                    logger.warning(f"Image processing failed for page '{page.path}': {str(e)}. Falling back to raw content.")
                    content = raw_content

            # Capture attachment references off the raw markdown so _process_document
            # can enumerate them even after Option A rewrote the URLs in `content`.
            attachment_paths = (
                self._extract_attachment_paths(raw_content) if self._index_include_attachments else []
            )

            metadata = {
                'id': str(page.id),
                'path': page.path,
                'title': title,
                'updated_on': content_hash,
            }
            if attachment_paths:
                metadata[self._INDEXER_ATTACHMENTS_META_KEY] = attachment_paths
            if chunking_tool:
                metadata[IndexerKeywords.CONTENT_IN_BYTES.value] = (content or "").encode("utf-8")
                yield Document(page_content='', metadata=metadata)
            else:
                yield Document(page_content=content or "", metadata=metadata)

    def _remove_metadata_keys(self) -> List[str]:
        """Drop the transient attachment-paths list before documents are written to the vector store."""
        return super()._remove_metadata_keys() + [self._INDEXER_ATTACHMENTS_META_KEY]

    def _matches_extension_filter(self, name: str) -> bool:
        """Apply include_extensions/skip_extensions using the Confluence convention:
        glob-like patterns matched against the file name (case-insensitive). Empty
        include_extensions means "process everything not in skip_extensions".
        """
        for pattern in self._index_skip_extensions:
            if re.match(re.escape(pattern).replace(r'\*', '.*') + '$', name, re.IGNORECASE):
                return False
        if not self._index_include_extensions:
            return True
        return any(
            re.match(re.escape(pattern).replace(r'\*', '.*') + '$', name, re.IGNORECASE)
            for pattern in self._index_include_extensions
        )

    _ATTACHMENT_DOWNLOAD_ATTEMPTS = 3
    _ATTACHMENT_DOWNLOAD_INITIAL_BACKOFF = 0.5  # seconds; doubles each retry
    _PARTIAL_MARKER = "::partial::"

    # Only retry exceptions whose class is inherently transient. Auth/404 errors
    # from the ADO SDK typically raise AzureDevOpsServiceError too; retrying
    # those is wasted time but bounded (~1.5s worst case) and the caller still
    # sees the original exception on final failure.
    _TRANSIENT_DOWNLOAD_EXCEPTIONS = (
        requests.RequestException,
        ConnectionError,
        TimeoutError,
        AzureDevOpsServiceError,
    )

    def _download_attachment_with_retry(self, repos_wrapper, file_path: str) -> bytes:
        """Fetch an attachment blob with short exponential backoff.

        Retries a bounded number of times on network-class exceptions only;
        after the final attempt the last exception is re-raised so the caller
        can decide how to record the failure.
        """
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self._ATTACHMENT_DOWNLOAD_ATTEMPTS + 1):
            try:
                return repos_wrapper.download_file(path=file_path)
            except self._TRANSIENT_DOWNLOAD_EXCEPTIONS as e:
                last_exc = e
                if attempt >= self._ATTACHMENT_DOWNLOAD_ATTEMPTS:
                    break
                delay = self._ATTACHMENT_DOWNLOAD_INITIAL_BACKOFF * (2 ** (attempt - 1))
                logger.info(
                    f"Attachment download attempt {attempt}/{self._ATTACHMENT_DOWNLOAD_ATTEMPTS} "
                    f"failed for '{file_path}': {e}. Retrying in {delay:.1f}s."
                )
                time.sleep(delay)
        # loop exhausted without a successful return
        raise last_exc  # type: ignore[misc]

    def _mark_parent_partial(self, document: Document, failed_names: List[str]) -> None:
        """Suffix the parent page's updated_on with a deterministic failure marker.

        Without this, a page whose raw markdown is unchanged but whose attachments
        failed to index would be dedup-skipped on the next run — stranding the
        failed attachments until the page itself is edited. Perturbing the hash
        forces `_reduce_duplicates` to yield the parent again next run, which
        re-fires `_process_document` and gives the attachments another chance.

        The marker is derived from the sorted set of failed file names, so it is
        stable across runs with the same failure set (no thrash on the marker
        itself). If failures resolve, the parent's hash returns to the clean
        raw-content form and the loop terminates naturally.
        """
        current = document.metadata.get('updated_on', '')
        if self._PARTIAL_MARKER in current:
            return  # already marked in this run
        signature = ",".join(sorted(set(failed_names)))
        sig_hash = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
        document.metadata['updated_on'] = f"{current}{self._PARTIAL_MARKER}{sig_hash}"

    def _process_document(self, document: Document) -> Generator[Document, None, None]:
        """Emit each referenced /.attachments/ file as a dependent Document.

        The base indexer calls this after _base_loader for every parent document
        that survived dedup. We only act when include_attachments=True; otherwise
        there's nothing to do. Attachment bytes are downloaded from the
        wikiMaster branch via the same ReposApiWrapper the tool-side image
        processing uses, then handed off to the indexer pipeline via
        CONTENT_IN_BYTES so parse_file_content can extract text (or an LLM
        description for images) uniformly with other non-code indexers.
        """
        if not getattr(self, "_index_include_attachments", False):
            return
        attachment_paths = document.metadata.get(self._INDEXER_ATTACHMENTS_META_KEY) or []
        if not attachment_paths:
            return

        wiki_identifier = getattr(self, "_index_wiki_identifier", None) or self._resolve_wiki_identifier(None)
        repos_wrapper = self._get_repos_wrapper(wiki_identifier)
        if repos_wrapper is None:
            for att_url in attachment_paths:
                self._track_dependent_item_skipped(att_url)
            # Whole page's attachments are stranded — force re-processing next run.
            self._mark_parent_partial(document, [unquote(u.lstrip('/')).rsplit('/', 1)[-1] for u in attachment_paths])
            return

        parent_id = document.metadata.get('id')
        parent_path = document.metadata.get('path', '')
        failed_names: List[str] = []

        for att_url in attachment_paths:
            file_path = unquote(att_url.lstrip('/'))
            file_name = file_path.rsplit('/', 1)[-1]
            if not self._matches_extension_filter(file_name):
                # Extension filter is an explicit user choice, not a failure —
                # do not perturb the parent's hash.
                self._track_runtime_skipped(file_name, reason="extension")
                continue
            try:
                attachment_bytes = self._download_attachment_with_retry(repos_wrapper, file_path)
            except Exception as e:
                logger.warning(
                    f"Failed to download attachment '{att_url}' for wiki page "
                    f"'{parent_path}' after {self._ATTACHMENT_DOWNLOAD_ATTEMPTS} attempt(s): {e}"
                )
                self._track_dependent_item_skipped(file_name)
                failed_names.append(file_name)
                continue
            if not attachment_bytes:
                self._track_skipped_file_empty(file_name)
                failed_names.append(file_name)
                continue

            # Use blob content hash as updated_on so unchanged attachments dedup across runs.
            att_hash = hashlib.sha256(attachment_bytes).hexdigest()
            file_ext = ("." + file_name.rsplit('.', 1)[-1]) if '.' in file_name else ''
            attachment_id = f"{parent_id}::{file_path}"

            yield Document(
                page_content='',
                metadata={
                    'id': attachment_id,
                    'name': file_name,
                    'path': file_path,
                    'parent_page_id': parent_id,
                    'parent_page_path': parent_path,
                    'updated_on': att_hash,
                    IndexerKeywords.CONTENT_FILE_NAME.value: file_ext,
                    IndexerKeywords.CONTENT_IN_BYTES.value: attachment_bytes,
                },
            )

        if failed_names:
            self._mark_parent_partial(document, failed_names)

    def _index_tool_params(self):
        """Return the parameters for indexing data."""
        return {
            'chunking_tool': (Literal['markdown', ''], Field(description="Name of chunking tool", default='markdown')),
            "wiki_identifier": (Optional[str], Field(default=None, description="Wiki identifier to index, e.g., 'ABCProject.wiki'. If not provided, uses the default wiki identifier from toolkit configuration.")),
            'path_contains': (Optional[str], Field(
                default=None,
                description=(
                    "Optional case-insensitive substring filter applied to the full wiki page path. "
                    "A page is included when the substring appears anywhere in its path, so "
                    "filtering by a parent folder name also pulls in all descendants. "
                    "Hyphens and spaces are treated as equivalent, so you can pass either the "
                    "URL slug form ('Feature-Analysis-and-Background') or the display form "
                    "('Feature Analysis and Background') and both will match the same pages. "
                    "Examples: 'design' matches '/Architecture/Design Records' and "
                    "'/Architecture/Design Records/API'. Leave empty ('' or omit) to index "
                    "all pages."
                )
            )),
            'process_images': (Optional[bool], Field(
                default=False,
                description=(
                    "If True, replace inline image markdown in each page with an LLM-generated "
                    "description of the image, so image content becomes part of the page's "
                    "searchable text. Requires an LLM to be configured on the toolkit. "
                    "Deduplication (updated_on) is computed against the raw page markdown "
                    "before rewriting, so non-deterministic LLM output does not force a "
                    "re-index every run. Costs one LLM call per referenced image per changed page."
                ),
            )),
            'image_description_prompt': (Optional[str], Field(
                default=None,
                description=(
                    "Optional custom prompt used when generating image descriptions. Only "
                    "applied when process_images=True or include_attachments=True. If omitted, "
                    "the default image description prompt is used."
                ),
            )),
            'include_attachments': (Optional[bool], Field(
                default=False,
                description=(
                    "If True, also index each /.attachments/ file referenced by wiki pages as "
                    "its own dependent document. Attachment bytes are pulled from the wikiMaster "
                    "branch of the wiki's backing git repo and passed through the standard "
                    "content parser (LLM description for images/PDFs, text for supported files). "
                    "External image URLs (http/https) are not indexed — only files stored in the "
                    "wiki repo. Independent of process_images: enabling both means image content "
                    "is searchable both inline in the parent page and as a standalone row."
                ),
            )),
            'include_extensions': (Optional[List[str]], Field(
                default=[],
                description=(
                    "Glob-style file-name patterns to include when indexing attachments, e.g. "
                    "[\"*.png\", \"*.pdf\"]. Empty list means include everything not in skip_extensions."
                ),
            )),
            'skip_extensions': (Optional[List[str]], Field(
                default=[],
                description=(
                    "Glob-style file-name patterns to skip when indexing attachments, e.g. "
                    "[\"*.zip\", \"*.exe\"]. Evaluated before include_extensions."
                ),
            )),
        }

    @extend_with_parent_available_tools
    def get_available_tools(self):
        """Return a list of available tools."""
        # Add default wiki identifier info to descriptions if configured
        default_wiki_info = f"\nDefault wiki: {self.default_wiki_identifier}" if self.default_wiki_identifier else ""

        return [
            {
                "name": "get_wiki",
                "description": (self.get_wiki.__doc__ or "") + default_wiki_info,
                "args_schema": GetWikiInput,
                "ref": self.get_wiki,
            },
            {
                "name": "get_wiki_page",
                "description": (self.get_wiki_page.__doc__ or "") + default_wiki_info,
                "args_schema": GetPageInput,
                "ref": self.get_wiki_page,
            },
            {
                "name": "get_wiki_page_by_path",
                "description": (self.get_wiki_page_by_path.__doc__ or "") + default_wiki_info,
                "args_schema": GetPageByPathInput,
                "ref": self.get_wiki_page_by_path,
            },
            {
                "name": "get_wiki_page_by_id",
                "description": (self.get_wiki_page_by_id.__doc__ or "") + default_wiki_info,
                "args_schema": GetPageByIdInput,
                "ref": self.get_wiki_page_by_id,
            },
            {
                "name": "delete_page_by_path",
                "description": (self.delete_page_by_path.__doc__ or "") + default_wiki_info,
                "args_schema": DeletePageByPathInput,
                "ref": self.delete_page_by_path,
            },
            {
                "name": "delete_page_by_id",
                "description": (self.delete_page_by_id.__doc__ or "") + default_wiki_info,
                "args_schema": DeletePageByIdInput,
                "ref": self.delete_page_by_id,
            },
            {
                "name": "modify_wiki_page",
                "description": (self.modify_wiki_page.__doc__ or "") + default_wiki_info,
                "args_schema": ModifyPageInput,
                "ref": self.modify_wiki_page,
            },
            {
                "name": "rename_wiki_page",
                "description": (self.rename_wiki_page.__doc__ or "") + default_wiki_info,
                "args_schema": RenamePageInput,
                "ref": self.rename_wiki_page,
            }
        ]