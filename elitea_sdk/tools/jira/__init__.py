from typing import List, Optional, Literal
from .api_wrapper import JiraApiWrapper
from langchain_core.tools import BaseTool, BaseToolkit
from ..base.tool import BaseAction
from pydantic import create_model, BaseModel, ConfigDict, Field
import requests

from ..elitea_base import filter_missconfigured_index_tools
from ..utils import parse_list, check_connection_response
from ...configurations.jira import JiraConfiguration, _hosting_to_cloud
from ...configurations.pgvector import PgVectorConfiguration
from ...runtime.utils.constants import TOOLKIT_NAME_META, TOOLKIT_TYPE_META, TOOL_NAME_META

name = "jira"

def get_toolkit(tool):
    settings = tool['settings']
    jira_configuration = settings['jira_configuration']

    # Resolve cloud from credential hosting first, then fall back to toolkit-level cloud
    # (toolkit-level cloud takes precedence during the transition period)
    toolkit_cloud = settings.get('cloud')
    if toolkit_cloud is not None:
        # Explicit toolkit-level value — respect it (backward compatibility)
        cloud = toolkit_cloud
    else:
        # Derive from credential hosting field
        hosting = jira_configuration.get('hosting', 'Auto') if isinstance(jira_configuration, dict) else getattr(jira_configuration, 'hosting', 'Auto')
        base_url = jira_configuration.get('base_url') if isinstance(jira_configuration, dict) else getattr(jira_configuration, 'base_url', None)
        cloud = _hosting_to_cloud(hosting, base_url)

    return JiraToolkit().get_toolkit(
        selected_tools=settings.get('selected_tools', []),
        base_url=settings.get('base_url'),
        cloud=cloud,
        api_version=settings.get('api_version', 'auto'),
        jira_configuration=jira_configuration,
        limit=settings.get('limit', 5),
        labels=parse_list(settings.get('labels', [])),
        custom_headers=settings.get('custom_headers', {}),
        additional_fields=settings.get('additional_fields', []),
        verify_ssl=settings.get('verify_ssl', True),
        # indexer settings
        llm=settings.get('llm', None),
        elitea=settings.get('elitea', None),
        pgvector_configuration=settings.get('pgvector_configuration', {}),
        collection_name=str(tool['toolkit_name']),
        embedding_model=settings.get('embedding_model'),
        vectorstore_type="PGVector",
        toolkit_name=tool.get('toolkit_name')
    )

def get_tools(tool):
    return get_toolkit(tool).get_tools()
            

class JiraToolkit(BaseToolkit):
    tools: List[BaseTool] = []

    @staticmethod
    def toolkit_config_schema() -> BaseModel:
        selected_tools = {x['name']: x['args_schema'].schema() for x in JiraApiWrapper.model_construct().get_available_tools()}

        @check_connection_response
        def check_connection(self):
            jira_config = self.jira_configuration or {}
            base_url = jira_config.get('base_url', '')
            url = base_url.rstrip('/') + '/rest/api/2/myself'
            headers = {'Accept': 'application/json'}
            auth = None
            token = jira_config.get('token')
            username = jira_config.get('username')
            api_key = jira_config.get('api_key')

            if token:
                headers['Authorization'] = f'Bearer {token}'
            elif username and api_key:
                auth = (username, api_key)
            else:
                raise ValueError('Jira connection requires either token or username+api_key')
            response = requests.get(url, headers=headers, auth=auth, timeout=5, verify=getattr(self, 'verify_ssl', True))
            return response

        model = create_model(
            name,
            limit=(int, Field(
                description="Maximum number of issues to retrieve per request.\n"
                            "Keep this value low for better performance.\n"
                            "(Default: 5)",
                gt=0, default=5
            )),
            api_version=(Literal['Auto', '2', '3'], Field(
                description="REST API version used for all Jira operations.\n\n"
                        "• **Auto** (default) — automatically selected based on the Hosting setting "
                        "in the linked credential (Cloud → V3, Server → V2)\n"
                        "• **V3** — required for Jira Cloud (*.atlassian.net). Uses Atlassian Document "
                        "Format (ADF) for comments and descriptions\n"
                        "• **V2** — standard for Jira Server / Data Center (e.g., self-hosted instances). "
                        "Uses plain text and wiki markup\n\n"
                        "⚠️ Using the wrong version for your instance type may cause failures in "
                        "comments, search, and attachments.",
                default="Auto"
            )),
            labels=(Optional[str], Field(
                description="Specify labels to apply to created or updated Jira entities.\n"
                            "Comma-separated list, no spaces around commas.\n"
                            "Example: `alita,elitea,automation`\n"
                            "(Optional)",
                default=None,
                examples="elitea,elitea;another-label"
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
            verify_ssl=(bool, Field(
                description="Enables SSL certificate verification for all API requests to your Jira instance.\n\n"
                            "• **Enabled** (recommended) — validates the server's SSL certificate for secure connections\n"
                            "• **Disabled** — skips SSL verification. Use only for internal/self-signed certificate environments\n\n"
                            "⚠️ Disabling SSL verification is not recommended in production environments.",
                default=True
            )),
            additional_fields=(Optional[str], Field(
                description="Custom Jira field IDs that should be accessible within this toolkit.\n"
                            "Use Jira field IDs as they appear in your instance schema.\n"
                            "Example: `customfield_10045,customfield_10100`\n"
                            "(Optional — leave empty if no custom fields are needed)",
                default=""
            )),
            jira_configuration=(JiraConfiguration, Field(description="Jira Configuration", json_schema_extra={'configuration_types': ['jira']})),
            pgvector_configuration=(Optional[PgVectorConfiguration], Field(default=None,
                                                                           description="PgVector Configuration", json_schema_extra={'configuration_types': ['pgvector']})),
            # embedder settings
            embedding_model=(Optional[str], Field(default=None, description="Embedding configuration.", json_schema_extra={'configuration_model': 'embedding'})),

            selected_tools=(List[Literal[tuple(selected_tools)]], Field(default=[], json_schema_extra={'args_schemas': selected_tools})),
            __config__=ConfigDict(json_schema_extra={
                'metadata': {
                    "label": "Jira",
                    "icon_url": "jira-icon.svg",
                    "categories": ["project management"],
                    "extra_categories": ["jira", "atlassian", "issue tracking", "project management", "task management"],
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
            # TODO use jira_configuration fields
            **kwargs['jira_configuration'],
            **(kwargs.get('pgvector_configuration') or {}),
        }
        jira_api_wrapper = JiraApiWrapper(**wrapper_payload)
        available_tools = jira_api_wrapper.get_available_tools()
        tools = []
        for tool in available_tools:
            if selected_tools:
                if tool["name"] not in selected_tools:
                    continue
            description = tool["description"]
            if toolkit_name:
                description = f"Toolkit: {toolkit_name}\n{description}"
            description = f"Jira instance: {jira_api_wrapper.base_url}\n{description}"
            description = description[:1000]
            tools.append(BaseAction(
                api_wrapper=jira_api_wrapper,
                name=tool["name"],
                description=description,
                args_schema=tool["args_schema"],
                metadata={TOOLKIT_NAME_META: toolkit_name, TOOLKIT_TYPE_META: name, TOOL_NAME_META: tool["name"]} if toolkit_name else {TOOL_NAME_META: tool["name"]}
            ))
        return cls(tools=tools)

    def get_tools(self):
        return self.tools
