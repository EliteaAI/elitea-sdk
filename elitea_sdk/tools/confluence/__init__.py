from typing import List, Literal, Optional

import requests
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_core.tools import BaseTool
from pydantic import create_model, BaseModel, ConfigDict, Field

from .api_wrapper import ConfluenceAPIWrapper
from ..base.tool import BaseAction
from ..elitea_base import filter_missconfigured_index_tools
from ..utils import parse_list, check_connection_response
from ...configurations.confluence import ConfluenceConfiguration, _hosting_to_cloud
from ...configurations.pgvector import PgVectorConfiguration
from ...runtime.utils.constants import TOOLKIT_NAME_META, TOOL_NAME_META, TOOLKIT_TYPE_META

name = "confluence"

def get_toolkit(tool):
    settings = tool['settings']
    confluence_configuration = settings['confluence_configuration']

    # Resolve cloud from credential hosting first, then fall back to toolkit-level cloud
    # (toolkit-level cloud takes precedence during the transition period)
    toolkit_cloud = settings.get('cloud')
    if toolkit_cloud is not None:
        cloud = toolkit_cloud
    else:
        hosting = confluence_configuration.get('hosting', 'Auto') if isinstance(confluence_configuration, dict) else getattr(confluence_configuration, 'hosting', 'Auto')
        base_url = confluence_configuration.get('base_url') if isinstance(confluence_configuration, dict) else getattr(confluence_configuration, 'base_url', None)
        cloud = _hosting_to_cloud(hosting, base_url)

    return ConfluenceToolkit().get_toolkit(
        selected_tools=settings.get('selected_tools', []),
        space=settings.get('space', None),
        cloud=cloud,
        api_version=settings.get('api_version', 'Auto'),
        confluence_configuration=confluence_configuration,
        limit=settings.get('limit', 5),
        labels=parse_list(settings.get('labels', None)),
        custom_headers=settings.get('custom_headers', {}),
        additional_fields=settings.get('additional_fields', []),
        verify_ssl=settings.get('verify_ssl', True),
        elitea=settings.get('elitea'),
        llm=settings.get('llm', None),
        toolkit_name=tool.get('toolkit_name'),
        # indexer settings
        pgvector_configuration=settings.get('pgvector_configuration', {}),
        collection_name=str(tool['toolkit_name']),
        doctype='doc',
        embedding_model=settings.get('embedding_model'),
        vectorstore_type="PGVector"
    )

def get_tools(tool):
    return get_toolkit(tool).get_tools()


class ConfluenceToolkit(BaseToolkit):
    tools: List[BaseTool] = []

    @staticmethod
    def toolkit_config_schema() -> BaseModel:
        selected_tools = {x['name']: x['args_schema'].schema() for x in
                          ConfluenceAPIWrapper.model_construct().get_available_tools()}

        @check_connection_response
        def check_connection(self):
            # Normalize base URL and construct API endpoint
            normalized_url = self.base_url.rstrip('/')
            cloud = getattr(self, 'cloud', True)

            # For cloud instances, ensure /wiki is present in the API path
            # Self-hosted instances may use different paths (e.g., /confluence)
            if cloud:
                # Check if base_url already includes /wiki
                if normalized_url.endswith('/wiki'):
                    url = normalized_url + '/rest/api/space'
                else:
                    url = normalized_url + '/wiki/rest/api/space'
            else:
                # For self-hosted, append /rest/api/space directly
                url = normalized_url + '/rest/api/space'

            headers = {'Accept': 'application/json'}
            auth = None
            confluence_config = self.confluence_configuration or {}
            token = confluence_config.get('token')
            username = confluence_config.get('username')
            api_key = confluence_config.get('api_key')

            if token:
                headers['Authorization'] = f'Bearer {token}'
            elif username and api_key:
                auth = (username, api_key)
            else:
                raise ValueError('Confluence connection requires either token or username+api_key')
            response = requests.get(url, headers=headers, auth=auth, timeout=5, verify=getattr(self, 'verify_ssl', True))
            return response

        model = create_model(
            name,
            space=(str, Field(description="Space")),
            api_version=(Literal['Auto', '2', '3'], Field(
                description="REST API version used for all Confluence operations.\n\n"
                        "• **Auto** (default) — automatically selected based on the Hosting setting "
                        "in the linked credential (Cloud → V3, Server → V2)\n"
                        "• **V3** — for Confluence Cloud (*.atlassian.net). Uses ADF for rich text content\n"
                        "• **V2** — for Confluence Server / Data Center. Uses plain text and wiki markup\n\n"
                        "⚠️ Using the wrong version may cause content formatting issues in pages and comments.",
                default="Auto"
            )),
            limit=(int, Field(
                description="Maximum number of pages to retrieve per individual API request.\n"
                            "Controls the page size of each call — does not limit the total number of pages retrieved.\n"
                            "(Default: 5)",
                default=5, gt=0
            )),
            labels=(Optional[str], Field(
                description="Filter content retrieval to pages that have specific Confluence labels.\n"
                            "Comma-separated list, no spaces around commas.\n"
                            "Example: `meeting-notes,documentation,project-alpha`\n"
                            "(Optional — leave empty to retrieve all content without label filtering)",
                default=None,
                examples="elitea,elitea;another-label"
            )),
            max_pages=(int, Field(
                description="Maximum total number of pages to retrieve across all paginated requests.\n"
                            "Prevents excessive data retrieval for large Confluence spaces.\n"
                            "(Default: 10)",
                default=10, gt=0
            )),
            number_of_retries=(int, Field(
                description="How many times the toolkit should automatically retry a failed API request "
                            "before reporting an error.\n"
                            "Useful for handling transient network issues or temporary Confluence unavailability.\n"
                            "(Default: 2)",
                default=2, ge=0
            )),
            min_retry_seconds=(int, Field(
                description="Minimum number of seconds to wait before attempting a retry after a failure.\n"
                            "Acts as the lower bound of the retry backoff interval.\n"
                            "(Default: 10)",
                default=10, ge=0
            )),
            max_retry_seconds=(int, Field(
                description="Maximum number of seconds to wait between retry attempts.\n"
                            "Acts as the upper bound of the retry backoff interval.\n"
                            "Retries will not wait longer than this value regardless of attempt number.\n"
                            "(Default: 60)",
                default=60, ge=0
            )),
            # optional field for custom headers as dictionary
            custom_headers=(Optional[dict], Field(
                description="Optional additional HTTP headers to include with every API request.\n"
                            "Useful for custom authentication, routing, or proxy requirements.\n"
                            "Must be valid JSON format.\n"
                            "Example: `{\"X-Custom-Header\": \"value\", \"X-Tenant-ID\": \"my-org\"}`\n"
                            "(Optional — leave empty if not required)",
                default={}
            )),
            confluence_configuration=(ConfluenceConfiguration, Field(description="Confluence Configuration", json_schema_extra={'configuration_types': ['confluence']})),
            pgvector_configuration=(Optional[PgVectorConfiguration], Field(default = None,
                                                                           description="PgVector Configuration",
                                                                           json_schema_extra={'configuration_types': ['pgvector']})),
            # embedder settings
            embedding_model=(Optional[str], Field(default=None, description="Embedding configuration.", json_schema_extra={'configuration_model': 'embedding'})),

            selected_tools=(List[Literal[tuple(selected_tools)]],
                            Field(default=[], json_schema_extra={'args_schemas': selected_tools})),
            __config__=ConfigDict(json_schema_extra={
                'metadata': {
                    "label": "Confluence",
                    "icon_url": None,
                    "categories": ["documentation"],
                    "extra_categories": ["confluence", "wiki", "knowledge base", "documentation", "atlassian"]
                }
            })
        )
        model.check_connection = check_connection
        return model

    @classmethod
    @filter_missconfigured_index_tools
    def get_toolkit(cls, selected_tools: list[str] | None = None, toolkit_name: Optional[str] = None, **kwargs):
        if selected_tools is None:
            selected_tools = []
        wrapper_payload = {
            **kwargs,
            # TODO use confluence_configuration fields
            **kwargs['confluence_configuration'],
            **(kwargs.get('pgvector_configuration') or {}),
        }
        confluence_api_wrapper = ConfluenceAPIWrapper(**wrapper_payload)
        available_tools = confluence_api_wrapper.get_available_tools()
        tools = []
        for tool in available_tools:
            if selected_tools:
                if tool["name"] not in selected_tools:
                    continue
            description = tool["description"]
            if toolkit_name:
                description = f"Toolkit: {toolkit_name}\n{description}"
            description = f"Confluence space: {confluence_api_wrapper.space}\n{description}"
            description = description[:1000]
            tools.append(BaseAction(
                api_wrapper=confluence_api_wrapper,
                name=tool["name"],
                description=description,
                args_schema=tool["args_schema"],
                metadata={TOOLKIT_NAME_META: toolkit_name, TOOLKIT_TYPE_META: name, TOOL_NAME_META: tool["name"]} if toolkit_name else {TOOL_NAME_META: tool["name"]}
            ))
        return cls(tools=tools)

    def get_tools(self):
        return self.tools
