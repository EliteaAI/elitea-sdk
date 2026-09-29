import uuid
from logging import getLogger
from typing import Any, Dict, Optional

from langchain_core.tools import BaseTool, ToolException
from langchain_core.tools.base import ArgsSchema
from pydantic import ConfigDict

from .mcp_input_schema import conform_mcp_arguments
from ..utils.failure_signals import mcp_is_error, log_shadow_failure, mcp_error_message

logger = getLogger(__name__)


class McpServerTool(BaseTool):
    name: str
    description: str
    args_schema: Optional[ArgsSchema] = None
    # Sanitized argument name -> the name the MCP server declared, see
    # build_mcp_args_schema.
    property_name_map: Dict[str, str] = {}
    return_type: str = "str"
    client: Any
    server: str
    tool_timeout_sec: int = 60

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _parse_input(self, tool_input, tool_call_id):
        parsed = super()._parse_input(tool_input, tool_call_id)
        if isinstance(self.args_schema, dict) and isinstance(parsed, dict):
            return conform_mcp_arguments(self.args_schema, parsed)
        return parsed

    def _restore_property_names(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return {self.property_name_map.get(name, name): value for name, value in arguments.items()}

    def _run(self, *args, **kwargs):
        # Strip None values — MCP servers reject null for typed optional params
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        clean_kwargs = self._restore_property_names(clean_kwargs)
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

        # An isError result is a failure; raise so it takes the same path as any other
        # tool error instead of being delivered as successful output (#6401).
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
                delivered_as_success=False,
            )
            raise ToolException(mcp_error_message(result))

        return result
