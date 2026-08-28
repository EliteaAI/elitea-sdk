import re
import uuid
from logging import getLogger
from typing import Any, Type, Literal, Optional, Union, List, Annotated

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model, ConfigDict, StringConstraints

from ..utils.failure_signals import mcp_is_error, log_shadow_failure

# EmailStr moved to pydantic_extra_types in pydantic v2, use str for simplicity
EmailStr = str

logger = getLogger(__name__)

# Anthropic (and other LLM providers) require tool schema property names to
# match this pattern. MCP servers are free to expose arbitrary property
# names (e.g. "fname[]"), so we sanitize them before building the pydantic
# model that is turned into the JSON schema sent to the LLM. See
# https://github.com/EliteaAI/elitea_issues/issues/6274
PROPERTY_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")
_INVALID_PROPERTY_CHARS = re.compile(r"[^a-zA-Z0-9_.-]")


def sanitize_property_name(name: str) -> str:
    """Sanitize a schema property name so it matches ^[a-zA-Z0-9_.-]{1,64}$.

    Invalid characters (e.g. "[", "]") are stripped. If sanitization yields
    an empty string, a generic fallback name is used. The result is
    truncated to 64 characters.
    """
    if PROPERTY_NAME_PATTERN.match(name):
        return name
    sanitized = _INVALID_PROPERTY_CHARS.sub("", name)
    if not sanitized:
        sanitized = "field"
    return sanitized[:64]


class McpServerTool(BaseTool):
    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None
    return_type: str = "str"
    client: Any
    server: str
    tool_timeout_sec: int = 60

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def create_pydantic_model_from_schema(schema: dict, model_name: str = "ArgsSchema"):
        def parse_type(field: dict, name: str = "Field") -> Any:
            if "allOf" in field:
                merged = {}
                required = set()
                for idx, subschema in enumerate(field["allOf"]):
                    sub_type = parse_type(subschema, f"{name}AllOf{idx}")
                    if hasattr(sub_type, "__fields__"):
                        merged.update({k: (v.outer_type_, v.default) for k, v in sub_type.__fields__.items()})
                        required.update({k for k, v in sub_type.__fields__.items() if v.required})
                if merged:
                    return create_model(f"{name}AllOf", **merged)
                return Any
            if "anyOf" in field or "oneOf" in field:
                key = "anyOf" if "anyOf" in field else "oneOf"
                types = [parse_type(sub, f"{name}{key.capitalize()}{i}") for i, sub in enumerate(field[key])]
                # Check for null type
                if any(sub.get("type") == "null" for sub in field[key]):
                    non_null_types = [parse_type(sub, f"{name}{key.capitalize()}{i}")
                                      for i, sub in enumerate(field[key]) if sub.get("type") != "null"]
                    if len(non_null_types) == 1:
                        return Optional[non_null_types[0]]
                return Union[tuple(types)]
            t = field.get("type")
            if isinstance(t, list):
                if "null" in t:
                    non_null = [x for x in t if x != "null"]
                    if len(non_null) == 1:
                        field = dict(field)
                        field["type"] = non_null[0]
                        return Optional[parse_type(field, name)]
                    return Any
                return Any
            if t == "string":
                if "enum" in field:
                    return Literal[tuple(field["enum"])]
                if field.get("format") == "email":
                    return EmailStr
                if "pattern" in field:
                    return Annotated[str, StringConstraints(pattern=field["pattern"])]
                return str
            if t == "integer":
                return int
            if t == "number":
                return float
            if t == "boolean":
                return bool
            if t == "object":
                # If no properties defined and additionalProperties is true/unset,
                # use dict[str, Any] to preserve arbitrary nested objects
                if not field.get("properties") and field.get("additionalProperties", True):
                    from typing import Dict
                    return Dict[str, Any]
                return McpServerTool.create_pydantic_model_from_schema(field, name.capitalize())
            if t == "array":
                items = field.get("items", {})
                return List[parse_type(items, name + "Item")]
            return Any

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        fields = {}
        # Maps the sanitized (pydantic/schema) field name back to the
        # original property name expected by the MCP server, for any
        # property whose name had to be sanitized.
        property_name_map = {}
        for name, prop in properties.items():
            sanitized_name = sanitize_property_name(name)
            if sanitized_name != name:
                property_name_map[sanitized_name] = name
            typ = parse_type(prop, sanitized_name.capitalize())
            default = prop.get("default", ... if name in required else None)
            field_args = {}
            if "description" in prop:
                field_args["description"] = prop["description"]
            if "format" in prop:
                field_args["format"] = prop["format"]
            fields[sanitized_name] = (typ, Field(default, **field_args))
        model = create_model(model_name, **fields)
        # Attached so McpServerTool._run can translate sanitized argument
        # names back to what the MCP server actually expects.
        model.__property_name_map__ = property_name_map
        return model

    def _run(self, *args, **kwargs):
        # Strip None values — MCP servers reject null for typed optional params
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # Translate sanitized property names (used in the schema shown to
        # the LLM, see create_pydantic_model_from_schema) back to the
        # original names the MCP server expects.
        property_name_map = getattr(self.args_schema, "__property_name_map__", None)
        if property_name_map:
            clean_kwargs = {
                property_name_map.get(k, k): v for k, v in clean_kwargs.items()
            }
        # Use the tool name directly (no prefix extraction needed)
        call_data = {
            "server": self.server,
            "tool_timeout_sec": self.tool_timeout_sec,
            "tool_call_id": str(uuid.uuid4()),
            "params": {
                "name": self.name,
                "arguments": clean_kwargs
            }
        }
        
        result = self.client.mcp_tool_call(call_data)

        # Shadow-mode only: detect isError, never changes the returned value (see #6168)
        if mcp_is_error(result):
            metadata = self.metadata or {}
            log_shadow_failure(
                logger,
                detected_by="mcp_is_error/proxied",
                toolkit_name=metadata.get("toolkit_name"),
                toolkit_type=metadata.get("toolkit_type"),
                toolkit_id=metadata.get("toolkit_id"),
                tool_name=self.name,
                result_len=len(str(result)),
            )

        return result
