from typing import List, Literal, Optional

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import BaseModel, ConfigDict, Field, create_model

from .api_wrapper import AhaApiWrapper
from ..base.tool import BaseAction
from ..common_tooltips import get_credentials_tooltip
from ...configurations.aha import AhaConfiguration
from ...runtime.utils.constants import (
    TOOL_NAME_META,
    TOOLKIT_NAME_META,
    TOOLKIT_TYPE_META,
)

name = "aha"


def get_tools(tool):
    return AhaToolkit().get_toolkit(
        selected_tools=tool["settings"].get("selected_tools", []),
        aha_configuration=tool["settings"]["aha_configuration"],
        toolkit_name=tool.get("toolkit_name"),
    ).get_tools()


class AhaToolkit(BaseToolkit):
    tools: List[BaseTool] = []

    @staticmethod
    def toolkit_config_schema() -> BaseModel:
        selected_tools = {
            x["name"]: x["args_schema"].schema()
            for x in AhaApiWrapper.model_construct().get_available_tools()
        }
        return create_model(
            name,
            aha_configuration=(
                AhaConfiguration,
                Field(
                    description=get_credentials_tooltip("Aha!"),
                    json_schema_extra={"configuration_types": ["aha"]},
                ),
            ),
            selected_tools=(
                List[Literal[tuple(selected_tools)]] if selected_tools else List[str],
                Field(default=[], json_schema_extra={"args_schemas": selected_tools}),
            ),
            __config__=ConfigDict(
                json_schema_extra={
                    "metadata": {
                        "label": "Aha!",
                        "icon_url": "aha.svg",
                        "categories": ["project management"],
                        "extra_categories": [
                            "aha",
                            "roadmap",
                            "requirements management",
                            "ideas",
                            "product management",
                        ],
                    }
                }
            ),
        )

    @classmethod
    def get_toolkit(
        cls,
        selected_tools: Optional[List[str]] = None,
        toolkit_name: Optional[str] = None,
        **kwargs,
    ):
        if selected_tools is None:
            selected_tools = []

        wrapper_payload = {
            **kwargs,
            **kwargs["aha_configuration"],
        }
        wrapper_payload.pop("aha_configuration", None)

        aha_api_wrapper = AhaApiWrapper(**wrapper_payload)
        available_tools = aha_api_wrapper.get_available_tools()

        tools: List[BaseTool] = []
        for tool in available_tools:
            if selected_tools and tool["name"] not in selected_tools:
                continue
            description = tool["description"]
            if toolkit_name:
                description = f"Toolkit: {toolkit_name}\n{description}"
            description = f"{description}\nAha! instance: {aha_api_wrapper.base_url}"
            description = description[:1000]
            metadata = (
                {
                    TOOLKIT_NAME_META: toolkit_name,
                    TOOLKIT_TYPE_META: name,
                    TOOL_NAME_META: tool["name"],
                }
                if toolkit_name
                else {TOOL_NAME_META: tool["name"]}
            )
            tools.append(
                BaseAction(
                    api_wrapper=aha_api_wrapper,
                    name=tool["name"],
                    description=description,
                    args_schema=tool["args_schema"],
                    metadata=metadata,
                )
            )
        return cls(tools=tools)

    def get_tools(self):
        return self.tools
