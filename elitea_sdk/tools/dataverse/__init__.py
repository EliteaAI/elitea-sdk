from typing import List, Literal, Optional

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import BaseModel, ConfigDict, create_model
from pydantic.fields import Field

from .api_wrapper import DataverseApiWrapper
from ..base.tool import BaseAction
from ..elitea_base import filter_missconfigured_index_tools
from ...configurations.dataverse import DataverseConfiguration, entra_authority
from ...runtime.utils.constants import TOOLKIT_NAME_META, TOOL_NAME_META, TOOLKIT_TYPE_META

name = "dataverse"


def get_tools(tool):
    return DataverseToolkit().get_toolkit(
        selected_tools=tool['settings'].get('selected_tools', []),
        dataverse_configuration=tool['settings']['dataverse_configuration'],
        tokens=tool['settings'].get('tokens', {}),
        toolkit_name=tool.get('toolkit_name'),
    ).get_tools()


class DataverseToolkit(BaseToolkit):
    tools: list[BaseTool] = []

    @staticmethod
    def toolkit_config_schema() -> BaseModel:
        available_tools = DataverseApiWrapper.model_construct().get_available_tools()
        selected_tools = {x['name']: x['args_schema'].schema() for x in available_tools}
        tool_groups = {x['name']: x['group'] for x in available_tools if x.get('group')}
        return create_model(
            name,
            dataverse_configuration=(DataverseConfiguration, Field(
                description="Dataverse Configuration", json_schema_extra={'configuration_types': ['dataverse']})),
            selected_tools=(List[Literal[tuple(selected_tools)]], Field(
                default=[], json_schema_extra={'args_schemas': selected_tools, 'tool_groups': tool_groups})),
            __config__=ConfigDict(json_schema_extra={
                'metadata': {
                    "label": "Dataverse (OneCRM)", "icon_url": "dataverse.svg",
                    "categories": ["other"],
                    "extra_categories": ["dataverse", "dynamics", "crm", "onecrm", "microsoft"],
                }
            })
        )

    @classmethod
    @filter_missconfigured_index_tools
    def get_toolkit(cls, selected_tools: list[str] | None = None, toolkit_name: Optional[str] = None, **kwargs):
        selected_tools = selected_tools or []
        config = kwargs.get('dataverse_configuration', {})
        wrapper_payload = {**kwargs, **config}
        injected = (kwargs.get('tokens') or {}).get(f"{config.get('configuration_uuid')}:{entra_authority(config)}")
        if injected:
            wrapper_payload['access_token'] = injected.get('access_token') if isinstance(injected, dict) else injected
        api_wrapper = DataverseApiWrapper(**wrapper_payload)
        tools = []
        for tool in api_wrapper.get_available_tools():
            if selected_tools and tool["name"] not in selected_tools:
                continue
            description = tool["description"]
            if toolkit_name:
                description = f"Toolkit: {toolkit_name}\n{description}"
            tools.append(BaseAction(
                api_wrapper=api_wrapper,
                name=tool["name"],
                description=description[:1000],
                args_schema=tool["args_schema"],
                metadata={TOOLKIT_NAME_META: toolkit_name, TOOLKIT_TYPE_META: name, TOOL_NAME_META: tool["name"]}
                if toolkit_name else {TOOL_NAME_META: tool["name"]}
            ))
        return cls(tools=tools)

    def get_tools(self) -> list[BaseTool]:
        return self.tools
