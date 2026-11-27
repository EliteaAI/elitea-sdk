"""Microsoft Graph toolkit for elitea-sdk.

Low-level access to any Microsoft Graph endpoint with delegated OAuth, limited by the
toolkit's allowed HTTP methods and path prefixes.
"""

import logging
from typing import List, Literal, Optional

from langchain_core.tools import BaseToolkit, BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from .api_wrapper import (
    DEFAULT_MAX_DOWNLOAD_MB,
    DEFAULT_MAX_RESPONSE_CHARS,
    DEFAULT_MAX_UPLOAD_MB,
    MsGraphApiWrapper,
)
from .rules import DEFAULT_METHODS, METHODS
from ..base.tool import BaseAction
from ...configurations.ms_graph import MsGraphConfiguration
from ..common_tooltips import get_credentials_tooltip

logger = logging.getLogger(__name__)

__all__ = ["MsGraphApiWrapper", "MsGraphToolkit", "get_tools"]

name = "ms_graph"

_TOOLKIT_SETTINGS = (
    "allowed_methods", "allowed_paths", "denied_paths",
    "max_response_chars", "max_upload_mb", "max_download_mb",
)


def get_tools(tool):
    """Entry point for toolkit loading from tools/__init__.py"""
    settings = tool['settings']
    return (MsGraphToolkit()
            .get_toolkit(
        selected_tools=settings.get('selected_tools', []),
        ms_graph_configuration=settings.get('ms_graph_configuration', {}),
        tokens=settings.get('tokens', {}),
        toolkit_name=tool.get('toolkit_name'),
        toolkit_id=tool.get('id'),
        llm=settings.get('llm'),
        elitea=settings.get('elitea', None),
        **{key: settings[key] for key in _TOOLKIT_SETTINGS if settings.get(key) is not None},
    )
            .get_tools())


class MsGraphToolkit(BaseToolkit):
    """Arbitrary Microsoft Graph requests as the signed-in user."""

    tools: List[BaseTool] = []

    @staticmethod
    def toolkit_config_schema() -> BaseModel:
        """Return the configuration schema for this toolkit."""
        available_tools = MsGraphApiWrapper.model_construct().get_available_tools()
        selected_tools = {x['name']: x['args_schema'].model_json_schema() if x.get('args_schema') else {}
                          for x in available_tools}
        tool_groups = {x['name']: x['group'] for x in available_tools if x.get('group')}

        return create_model(
            name,
            ms_graph_configuration=(
                MsGraphConfiguration,
                Field(
                    description=get_credentials_tooltip("Microsoft Graph"),
                    json_schema_extra={'configuration_types': ['ms_graph']}
                )
            ),
            allowed_methods=(
                List[Literal[METHODS]],
                Field(default=list(DEFAULT_METHODS), title="Allowed methods",
                      description="HTTP methods the tools may send. GET only by default; add POST, PATCH, PUT or "
                                  "DELETE to let the agent change data. $batch is checked per request.")
            ),
            allowed_paths=(
                List[str],
                Field(default=[], title="Allowed paths",
                      description="Only these Graph path prefixes (empty = any), e.g. /me/messages, /users/*/events. "
                                  "A prefix matches whole path segments; * matches one segment.")
            ),
            denied_paths=(
                List[str],
                Field(default=[], title="Denied paths",
                      description="Never these Graph path prefixes, e.g. /me/sendMail, /users. "
                                  "Denied paths win over allowed paths.")
            ),
            max_response_chars=(
                int,
                Field(default=DEFAULT_MAX_RESPONSE_CHARS, ge=1000, le=200000, title="Max response characters",
                      description="Longer JSON / text responses are cut to this size")
            ),
            max_upload_mb=(
                int,
                Field(default=DEFAULT_MAX_UPLOAD_MB, ge=1, le=250, title="Max upload size (MB)",
                      description="Largest artifact file sent as a request body")
            ),
            max_download_mb=(
                int,
                Field(default=DEFAULT_MAX_DOWNLOAD_MB, ge=1, le=500, title="Max download size (MB)",
                      description="Largest response read; binary responses are saved to artifact storage")
            ),
            selected_tools=(
                List[Literal[tuple(selected_tools)]],
                Field(default=[], json_schema_extra={'args_schemas': selected_tools, 'tool_groups': tool_groups})
            ),
            __config__=ConfigDict(json_schema_extra={
                'metadata': {
                    "label": "Microsoft Graph",
                    "icon_url": "ms_graph.svg",
                    "categories": ["office"],
                    "extra_categories": ["microsoft", "graph", "office 365", "microsoft 365", "api"]
                }
            })
        )

    @classmethod
    def get_toolkit(
        cls,
        selected_tools: Optional[List[str]] = None,
        toolkit_name: Optional[str] = None,
        **kwargs
    ):
        """Create and return a MsGraphToolkit instance."""
        if selected_tools is None:
            selected_tools = []

        toolkit_id = kwargs.get('toolkit_id')
        graph_config = kwargs.get('ms_graph_configuration', {}) or {}

        wrapper_payload = {
            **kwargs,
            **graph_config,
            'toolkit_name': toolkit_name,
        }

        if kwargs.get('tokens') and graph_config.get('oauth_discovery_endpoint'):
            oauth_endpoint = graph_config['oauth_discovery_endpoint']
            config_uuid = graph_config.get('configuration_uuid')

            token_data = None
            if config_uuid:
                token_data = kwargs['tokens'].get(f"{config_uuid}:{oauth_endpoint}")
            if token_data is None:
                token_data = kwargs['tokens'].get(oauth_endpoint)

            if token_data:
                if isinstance(token_data, dict):
                    wrapper_payload['token'] = token_data.get('access_token')
                    # Keep the credential's stored login when the injected token carries none
                    if token_data.get('refresh_token'):
                        wrapper_payload['refresh_token'] = token_data['refresh_token']
                else:
                    wrapper_payload['token'] = token_data

        # Proactive OAuth guard (same as Teams / Outlook): surface a clean login
        # prompt instead of failing deep inside a tool call.
        if graph_config.get('oauth_discovery_endpoint') and not wrapper_payload.get('token'):
            logger.debug("MS Graph OAuth mode active but no token found — raising McpAuthorizationRequired.")
            raise MsGraphConfiguration._build_mcp_authorization_required(
                message=(
                    "Microsoft Graph requires OAuth authorization. "
                    "Please complete the OAuth flow to obtain an access token."
                ),
                oauth_discovery_endpoint=graph_config['oauth_discovery_endpoint'],
                scopes=graph_config.get('scopes'),
                configuration_uuid=graph_config.get('configuration_uuid'),
                toolkit_id=toolkit_id,
                toolkit_name=toolkit_name,
                client_id=graph_config.get('client_id'),
                # An admin-managed app's secret stays server-side: not even a masked copy
                client_secret=None if graph_config.get('oauth_app_id') else graph_config.get('client_secret'),
                oauth_app_id=graph_config.get('oauth_app_id'),
            )

        wrapper = MsGraphApiWrapper(**wrapper_payload)

        available_tools_list = wrapper.get_available_tools()
        if selected_tools:
            selected_set = set(selected_tools)
            available_tools_list = [t for t in available_tools_list if t["name"] in selected_set]

        tools = [
            BaseAction(
                api_wrapper=wrapper,
                name=tool_info["name"],
                description=tool_info["description"],
                args_schema=tool_info.get("args_schema"),
            )
            for tool_info in available_tools_list
        ]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Return list of tools in this toolkit."""
        return self.tools
