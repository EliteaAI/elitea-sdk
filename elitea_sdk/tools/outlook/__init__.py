"""Outlook toolkit for elitea-sdk.

Provides email operations via Microsoft Graph API with OAuth authentication.
"""

import logging
from typing import List, Literal, Optional

from langchain_core.tools import BaseToolkit, BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from .api_wrapper import OutlookApiWrapper
from ..base.tool import BaseAction
from ...configurations.outlook import OutlookConfiguration
from ..common_tooltips import get_credentials_tooltip

logger = logging.getLogger(__name__)

__all__ = ["OutlookApiWrapper", "OutlookToolkit", "get_tools"]

name = "outlook"


def get_tools(tool):
    """Entry point for toolkit loading from tools/__init__.py"""
    return (OutlookToolkit()
            .get_toolkit(
        selected_tools=tool['settings'].get('selected_tools', []),
        outlook_configuration=tool['settings'].get('outlook_configuration', {}),
        tokens=tool['settings'].get('tokens', {}),
        toolkit_name=tool.get('toolkit_name'),
        toolkit_id=tool.get('id'),
        llm=tool['settings'].get('llm'),
        elitea=tool['settings'].get('elitea', None),
    )
            .get_tools())


class OutlookToolkit(BaseToolkit):
    """Outlook toolkit for email operations via Microsoft Graph API."""

    tools: List[BaseTool] = []

    @staticmethod
    def toolkit_config_schema() -> BaseModel:
        """Return the configuration schema for this toolkit."""
        available_tools = OutlookApiWrapper.model_construct().get_available_tools()
        selected_tools = {x['name']: x['args_schema'].model_json_schema() if x.get('args_schema') else {} for x in available_tools}

        return create_model(
            name,
            outlook_configuration=(
                OutlookConfiguration,
                Field(
                    description=get_credentials_tooltip("Outlook"),
                    json_schema_extra={'configuration_types': ['outlook']}
                )
            ),
            selected_tools=(
                List[Literal[tuple(selected_tools)]],
                Field(default=[], json_schema_extra={'args_schemas': selected_tools})
            ),
            __config__=ConfigDict(json_schema_extra={
                'metadata': {
                    "label": "Outlook",
                    "icon_url": "outlook.svg",
                    "categories": ["office"],
                    "extra_categories": ["microsoft", "email", "mail", "communication"]
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
        """Create and return an OutlookToolkit instance."""
        if selected_tools is None:
            selected_tools = []

        toolkit_id = kwargs.get('toolkit_id')

        outlook_config = kwargs.get('outlook_configuration', {})

        wrapper_payload = {
            **kwargs,
            **outlook_config,
            'toolkit_name': toolkit_name,
        }

        if kwargs.get('tokens') and outlook_config.get('oauth_discovery_endpoint'):
            logger.debug("Outlook configuration includes OAuth discovery endpoint")
            oauth_endpoint = outlook_config['oauth_discovery_endpoint']
            config_uuid = outlook_config.get('configuration_uuid')

            token_data = None
            if config_uuid:
                token_data = kwargs['tokens'].get(f"{config_uuid}:{oauth_endpoint}")
            if token_data is None:
                token_data = kwargs['tokens'].get(oauth_endpoint)

            if token_data:
                if isinstance(token_data, dict):
                    wrapper_payload['token'] = token_data.get('access_token')
                    wrapper_payload['refresh_token'] = token_data.get('refresh_token')
                else:
                    wrapper_payload['token'] = token_data

        # Proactive OAuth guard (same as SharePoint): surface a clean login prompt
        # instead of failing deep inside a tool call.
        if outlook_config.get('oauth_discovery_endpoint') and not wrapper_payload.get('token'):
            logger.debug("Outlook OAuth mode active but no token found — raising McpAuthorizationRequired.")
            raise OutlookConfiguration._build_mcp_authorization_required(
                message=(
                    "Outlook requires OAuth authorization. "
                    "Please complete the OAuth flow to obtain an access token."
                ),
                oauth_discovery_endpoint=outlook_config['oauth_discovery_endpoint'],
                scopes=outlook_config.get('scopes'),
                configuration_uuid=outlook_config.get('configuration_uuid'),
                toolkit_id=toolkit_id,
                toolkit_name=toolkit_name,
                client_id=outlook_config.get('client_id'),
                client_secret=outlook_config.get('client_secret'),
            )

        wrapper = OutlookApiWrapper(**wrapper_payload)

        available_tools_list = wrapper.get_available_tools()

        if selected_tools:
            selected_set = set(selected_tools)
            available_tools_list = [t for t in available_tools_list if t["name"] in selected_set]

        tools = []
        for tool_info in available_tools_list:
            tool = BaseAction(
                api_wrapper=wrapper,
                name=tool_info["name"],
                description=tool_info["description"],
                args_schema=tool_info.get("args_schema"),
            )
            tools.append(tool)

        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Return list of tools in this toolkit."""
        return self.tools
