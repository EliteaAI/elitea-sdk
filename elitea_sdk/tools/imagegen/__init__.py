"""The `imagegen` toolkit — `generate_image` / `edit_image` through the gateway.

Ported by decision, elitea-platform #864: the legacy `imagegen` plugin
(`legacy/plugins/imagegen/`, `methods/descriptor.py:43-88`) declared one
service provider (`ImageGenServiceProvider`) with one toolkit (`ImageGen`,
config fields `image_generation_model`, `bucket`, `name_prefix`) and tools
`generate_image` / `edit_image`. That toolkit half is what this module ports.
The plugin's OTHER half — the generic `descriptor` / `health` / `invoke` /
`invocations` provider-hub route surface every legacy provider plugin
exposed — is NOT ported: it stays deferred to ADR-0012/P3, which is not yet
Approved, per the #247 umbrella decision recorded 2026-09-08.
"""

from typing import List, Literal, Optional

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import BaseModel, ConfigDict, Field, create_model

from .api_wrapper import ImageGenAPIWrapper
from ..base.tool import BaseAction
from ...runtime.utils.constants import TOOL_NAME_META, TOOLKIT_NAME_META, TOOLKIT_TYPE_META

name = "imagegen"


def get_tools(tool):
    settings = tool["settings"]
    return ImageGenToolkit().get_toolkit(
        elitea=settings["elitea"],
        image_generation_model=settings.get("image_generation_model"),
        bucket=settings.get("bucket"),
        name_prefix=settings.get("name_prefix") or "generated-",
        selected_tools=settings.get("selected_tools", []),
        toolkit_name=tool.get("toolkit_name"),
    ).get_tools()


class ImageGenToolkit(BaseToolkit):
    tools: List[BaseTool] = []

    @staticmethod
    def toolkit_config_schema() -> BaseModel:
        available_tools = ImageGenAPIWrapper.model_construct().get_available_tools()
        selected_tools = {x["name"]: x["args_schema"].schema() for x in available_tools}
        tool_groups = {x["name"]: x["group"] for x in available_tools if x.get("group")}
        return create_model(
            name,
            image_generation_model=(
                str,
                Field(
                    description="The project's image-generation-capable model.",
                    json_schema_extra={"configuration_model": "image_generation"},
                ),
            ),
            bucket=(
                str,
                Field(
                    description="Artifact bucket that generated and edited images are saved to.",
                    pattern=r"^[a-z][a-z0-9-]*$",
                ),
            ),
            name_prefix=(
                Optional[str],
                Field(default="generated-", description="Filename prefix applied to every saved image."),
            ),
            selected_tools=(
                List[Literal[tuple(selected_tools)]],
                Field(default=[], json_schema_extra={"args_schemas": selected_tools, "tool_groups": tool_groups}),
            ),
            __config__=ConfigDict(
                json_schema_extra={
                    "metadata": {
                        "label": "ImageGen",
                        "icon_url": None,
                        "categories": ["media"],
                        "extra_categories": ["image", "generation", "dall-e", "art"],
                        "hidden": False,
                    }
                }
            ),
        )

    @classmethod
    def get_toolkit(
        cls,
        elitea=None,
        image_generation_model: Optional[str] = None,
        bucket: Optional[str] = None,
        name_prefix: str = "generated-",
        selected_tools: Optional[list] = None,
        toolkit_name: Optional[str] = None,
        **kwargs,
    ):
        if selected_tools is None:
            selected_tools = []
        wrapper = ImageGenAPIWrapper(
            elitea=elitea,
            image_generation_model=image_generation_model,
            bucket=bucket,
            name_prefix=name_prefix,
        )
        available_tools = wrapper.get_available_tools()
        tools = []
        for tool in available_tools:
            if selected_tools and tool["name"] not in selected_tools:
                continue
            description = tool["description"]
            if toolkit_name:
                description = f"Toolkit: {toolkit_name}\n{description}"
            description = description[:1000]
            tools.append(
                BaseAction(
                    api_wrapper=wrapper,
                    name=tool["name"],
                    description=description,
                    args_schema=tool["args_schema"],
                    metadata=(
                        {TOOLKIT_NAME_META: toolkit_name, TOOLKIT_TYPE_META: name, TOOL_NAME_META: tool["name"]}
                        if toolkit_name
                        else {TOOL_NAME_META: tool["name"]}
                    ),
                )
            )
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        return self.tools
