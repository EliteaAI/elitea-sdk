from typing import List, Literal, Optional

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import BaseModel, ConfigDict, Field, create_model

from .api_wrapper import TavilyApiWrapper
from ..base.tool import BaseAction
from ...configurations.tavily import TavilyConfiguration
from ...runtime.utils.constants import TOOLKIT_NAME_META, TOOL_NAME_META, TOOLKIT_TYPE_META

name = "tavily"

CHAT_TOOLS = ["web_search", "web_extract"]
TOOL_SPEND = {
    "feature": "web_search",
    "label": "Web search",
    "backend": "tavily",
    "configuration_type": "tavily",
    "unit": "credit",
    "operations": ["search", "extract", "crawl", "map"],
    # Largest single call here: crawl at limit 20 with instructions and advanced extract (4 + 8)
    "max_units_per_call": 12,
    "chat_tools": CHAT_TOOLS,
}
POLICY_KEYS = ("max_results", "max_search_depth", "extract_depth", "max_page_chars",
               "allowed_domains", "blocked_domains", "per_run_call_cap")


def get_tools(tool):
    settings = tool['settings']
    return TavilyToolkit().get_toolkit(
        selected_tools=settings.get('selected_tools', []),
        tavily_configuration=settings['tavily_configuration'],
        elitea=settings.get('elitea'),
        toolkit_name=tool.get('toolkit_name'),
        **{key: settings[key] for key in POLICY_KEYS if settings.get(key) is not None},
    ).get_tools()


class TavilyToolkit(BaseToolkit):
    tools: list[BaseTool] = []

    @staticmethod
    def toolkit_config_schema() -> BaseModel:
        available_tools = TavilyApiWrapper.model_construct().get_available_tools()
        selected_tools = {x['name']: x['args_schema'].schema() for x in available_tools}
        tool_groups = {x['name']: x['group'] for x in available_tools if x.get('group')}
        return create_model(
            name,
            tavily_configuration=(TavilyConfiguration, Field(description="Tavily Configuration",
                                                             json_schema_extra={'configuration_types': ['tavily']})),
            max_results=(int, Field(default=5, ge=1, le=10, description="Search results per query")),
            max_search_depth=(Literal["basic", "advanced"], Field(
                default="basic", description="Deepest search the model may request (advanced costs 2 credits per query)")),
            extract_depth=(Literal["basic", "advanced"], Field(
                default="basic", description="Extraction depth ('advanced' costs double)")),
            max_page_chars=(int, Field(default=20000, ge=1000, le=50000, description="Maximum characters per page")),
            allowed_domains=(List[str], Field(default=[], description="Only these domains (empty = any)")),
            blocked_domains=(List[str], Field(default=[], description="Never these domains")),
            per_run_call_cap=(int, Field(default=10, ge=1, le=100, description="Maximum Tavily calls per run")),
            selected_tools=(List[Literal[tuple(selected_tools)]], Field(default=[], json_schema_extra={'args_schemas': selected_tools, 'tool_groups': tool_groups})),
            __config__=ConfigDict(json_schema_extra={
                'metadata': {
                    "label": "Tavily", "icon_url": "tavily.svg",
                    "categories": ["other"],
                    "extra_categories": ["tavily", "web", "search", "internet", "extract", "crawl"],
                    "tool_spend": TOOL_SPEND,
                }
            })
        )

    @classmethod
    def get_toolkit(cls, selected_tools: list[str] | None = None, toolkit_name: Optional[str] = None, **kwargs):
        selected_tools = selected_tools or []
        credential = kwargs.pop('tavily_configuration', None) or {}
        if hasattr(credential, 'model_dump'):
            credential = credential.model_dump()
        wrapper = TavilyApiWrapper(
            **kwargs,
            api_key=credential.get('api_key') or '',
            configuration_uuid=credential.get('configuration_uuid'),
            configuration_project_id=credential.get('configuration_project_id'),
            configuration_type=credential.get('configuration_type'),
            toolkit_name=toolkit_name,
        )
        tools = []
        for tool in wrapper.get_available_tools():
            if selected_tools and tool["name"] not in selected_tools:
                continue
            description = tool["description"]
            if toolkit_name:
                description = f"Toolkit: {toolkit_name}\n{description}"
            tools.append(BaseAction(
                api_wrapper=wrapper,
                name=tool["name"],
                description=description[:1000],
                args_schema=tool["args_schema"],
                metadata={TOOLKIT_NAME_META: toolkit_name, TOOLKIT_TYPE_META: name, TOOL_NAME_META: tool["name"]}
                if toolkit_name else {TOOL_NAME_META: tool["name"]},
            ))
        return cls(tools=tools)

    def get_tools(self) -> list[BaseTool]:
        return self.tools
