import json

from ..langchain.constants import ELITEA_RS, PRINTER_NODE_RS
from ..utils.utils import clean_string
from langchain_core.tools import BaseTool, ToolException
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from typing import Any, Type, Optional
from pydantic import create_model, model_validator, BaseModel
from pydantic.fields import FieldInfo
from ..langchain.mixedAgentRenderes import convert_message_to_json
from logging import getLogger

logger = getLogger(__name__)

applicationToolSchema = create_model(
    "applicatrionSchema", 
    task = (str, FieldInfo(description="Task for Application")), 
    chat_history = (Optional[list[BaseMessage]], FieldInfo(description="Chat History relevant for Application", default=[]))
)


def formulate_query(kwargs, is_subgraph=False):
    chat_history = []
    if kwargs.get('chat_history'):
        if isinstance(kwargs.get('chat_history')[-1], BaseMessage):
            chat_history = convert_message_to_json(kwargs.get('chat_history')[:])
        elif isinstance(kwargs.get('chat_history')[-1], dict):
            if all([True if message.get('role') and message.get('content') else False for message in kwargs.get('chat_history')]):
                chat_history = kwargs.get('chat_history')[:]
            else:
                for each in kwargs.get('chat_history')[:]:
                    chat_history.append(AIMessage(json.dumps(each)))
        elif isinstance(kwargs.get('chat_history')[-1], str):
            chat_history = []
            for each in kwargs.get('chat_history')[:]:
                chat_history.append(AIMessage(each))
    user_task = kwargs.get('task')
    if not user_task:
        raise ToolException("Task is required to invoke the application. "
                            "Check the provided input (some errors may happen on previous steps).")
    result = {"input": [HumanMessage(content=user_task)] if not is_subgraph else user_task, "chat_history": chat_history}
    for key, value in kwargs.items():
        if key not in ("task", "chat_history"):
            result[key] = value
    return result


def extract_application_response_output(response: Any) -> str:
    """Extract a usable final output string from a nested application response."""

    def normalize_content(content: Any) -> str:
        if content is None:
            return ''

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            has_only_tool_blocks = True

            for block in content:
                if isinstance(block, dict):
                    block_type = block.get('type')
                    if block_type == 'text':
                        text_parts.append(block.get('text', ''))
                        has_only_tool_blocks = False
                    elif 'text' in block and 'type' not in block:
                        text_parts.append(block.get('text', ''))
                        has_only_tool_blocks = False
                    elif block_type in ('tool_use', 'tool_result', 'thinking', 'reasoning'):
                        continue
                    else:
                        text_parts.append(json.dumps(block, ensure_ascii=False))
                        has_only_tool_blocks = False
                elif isinstance(block, str):
                    text_parts.append(block)
                    has_only_tool_blocks = False
                else:
                    text_parts.append(str(block))
                    has_only_tool_blocks = False

            if has_only_tool_blocks and not text_parts:
                return ''
            return ''.join(text_parts)

        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)

        return str(content)

    if isinstance(response, BaseMessage):
        return normalize_content(response.content)

    if isinstance(response, str):
        return normalize_content(response)

    if not isinstance(response, dict):
        return normalize_content(response)

    for key in ('output', ELITEA_RS, PRINTER_NODE_RS):
        normalized = normalize_content(response.get(key))
        if normalized.strip():
            return normalized

    messages = response.get('messages') or []
    for message in reversed(messages):
        if isinstance(message, BaseMessage):
            if isinstance(message, HumanMessage):
                continue
            normalized = normalize_content(message.content)
            if normalized.strip():
                return normalized
        elif isinstance(message, dict):
            role = str(message.get('role') or '').lower()
            if role in ('user', 'human'):
                continue
            normalized = normalize_content(message.get('content'))
            if normalized.strip():
                return normalized

    return ''


class Application(BaseTool):
    name: str
    description: str
    application: Any
    args_schema: Type[BaseModel] = applicationToolSchema
    return_type: str = "str"
    client: Any
    args_runnable: dict = {}
    metadata: dict = {}
    is_subgraph: Optional[bool] = False
    variable_defaults: dict = {}  # Default values for agent variables (from version_details)

    @model_validator(mode='before')
    @classmethod
    def preserve_original_name(cls, data: Any) -> Any:
        """Preserve the original name in metadata before cleaning."""
        if isinstance(data, dict):
            original_name = data.get('name')
            if original_name:
                # Initialize metadata if not present
                if data.get('metadata') is None:
                    data['metadata'] = {}
                # Store original name before cleaning
                data['metadata']['original_name'] = original_name
                # Clean the name
                data['name'] = clean_string(original_name)
        return data

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        """Override default invoke to preserve all fields, not just args_schema"""
        # Handle ToolCall format: {"name": ..., "args": {...}, "id": ..., "type": "tool_call"}
        # LangGraph's ToolNode passes the full ToolCall dict directly; BaseTool.invoke() normally
        # extracts input["args"] via _prep_run_args, but our override bypasses that logic.
        #
        # LangGraph's ToolNode (newer versions) requires tools to return ToolMessage or Command —
        # not a plain str. When called via ToolNode (type=="tool_call"), capture the tool_call_id
        # so we can wrap the result in a ToolMessage before returning.
        tool_call_id = None
        if isinstance(input, dict) and input.get("type") == "tool_call":
            tool_call_id = input.get("id")
            input = input["args"]
        schema_values = self.args_schema(**input).model_dump() if self.args_schema else {}
        extras = {k: v for k, v in input.items() if k not in schema_values}
        all_kwargs = {**kwargs, **extras, **schema_values}
        if config is None:
            config = {}

        # Inject tool metadata into config so it's passed to callbacks.
        # IMPORTANT: copy config and its metadata dict before injecting to avoid mutating
        # the caller's LangGraph run config. LangGraph builds per-tool configs via shallow
        # copies (patch_config/merge_configs), so config['metadata'] is often the same
        # dict object across sequential tool invocations in the same batch. In-place
        # mutation bleeds one tool's metadata into the next tool's on_tool_start event,
        # causing wrong parent_agent_name / agent_type on chips when the parent agent has
        # both agent-type and pipeline-type toolkits.
        if self.metadata:
            config = dict(config)
            config['metadata'] = dict(config.get('metadata') or {})
            # Merge tool metadata into config metadata (config values take precedence)
            for key, value in self.metadata.items():
                if key not in config['metadata']:
                    config['metadata'][key] = value

        # super().invoke() → BaseTool.run() fires on_tool_start/on_tool_end callbacks
        # (agent_tool_start chip event) and passes config to _run() via the RunnableConfig
        # type annotation on _run's `config` parameter.
        # NOTE: since we already unwrapped the ToolCall above, BaseTool sees a plain dict
        # input (no tool_call_id in _prep_run_args), so we must wrap the result ourselves.
        result = super().invoke(all_kwargs, config=config)

        # When invoked from LangGraph's ToolNode, wrap plain str/dict results in a ToolMessage.
        # ToolNode (langgraph >= 0.3) rejects any return type that is not ToolMessage or Command,
        # raising: TypeError: Tool <name> returned unexpected type: <class 'str'>
        if tool_call_id is not None:
            if isinstance(result, ToolMessage):
                # Already a ToolMessage (e.g. is_subgraph path); ensure tool_call_id is set
                if not result.tool_call_id:
                    result.tool_call_id = tool_call_id
                return result
            # Convert str / dict / other to ToolMessage content
            if isinstance(result, str):
                content = result
            elif isinstance(result, dict):
                content = result.get("output", str(result))
            else:
                content = str(result)
            return ToolMessage(
                content=content,
                name=self.name,
                tool_call_id=tool_call_id,
            )

        return result

    def _run(self, *args, config: RunnableConfig = None, **kwargs):
        # `config` is injected by BaseTool.run() because of the RunnableConfig type annotation
        # (via _get_runnable_config_param). It carries the parent's metadata (toolkit_name,
        # agent_type, etc.) that we need to build nested_config, but NOT callbacks/checkpoints
        # — those are already stripped by BaseTool.run() (it passes child_config to _run, not
        # the full parent config) which prevents double-firing and checkpoint corruption.
        invoke_config = config
        # Also consume legacy _invoke_config kwarg in case called directly in tests.
        invoke_config = kwargs.pop('_invoke_config', invoke_config)

        if self.client and self.args_runnable:
            # Recreate new LanggraphAgentRunnable in order to reflect the current input_mapping (it can be dynamic for pipelines).
            # Actually, for pipelines agent toolkits LanggraphAgentRunnable is created (for LLMNode) before pipeline's schema parsing.

            # Merge variable defaults with passed kwargs
            # Defaults are applied first, then overridden by explicitly passed values
            # This ensures variables always have a value (either default or passed)
            merged_vars = dict(self.variable_defaults)  # Start with defaults
            for k, v in kwargs.items():
                # Only override if value is not None (allow explicit override with actual values)
                if v is not None:
                    merged_vars[k] = v
                elif k not in merged_vars:
                    # Keep None values only if no default exists
                    merged_vars[k] = v

            # Build application_variables from merged values
            application_variables = {k: {"name": k, "value": v} for k, v in merged_vars.items()}
            logger.debug(f"[APP_RUN] Variable defaults: {self.variable_defaults}")
            logger.debug(f"[APP_RUN] Merged variables: {list(merged_vars.keys())}")

            self.application = self.client.application(**self.args_runnable, application_variables=application_variables)
        # Forward checkpoint-bearing config to the nested application so child
        # applications participate in the same durable execution tree.
        # Keep callbacks and other non-essential runtime baggage stripped to
        # avoid duplicate events and accidental parent-side config leakage.
        _parent_name = self.metadata.get('original_name') or self.metadata.get('display_name')
        nested_metadata = dict(invoke_config['metadata']) if invoke_config and invoke_config.get('metadata') else {}
        if _parent_name:
            nested_metadata['parent_agent_name'] = _parent_name
        nested_config = {}
        if invoke_config and invoke_config.get('configurable'):
            nested_config['configurable'] = dict(invoke_config['configurable'])
            nested_config['configurable'].pop('selected_tools', None)
            nested_config['configurable'].pop('selected_toolkits', None)
        if nested_metadata:
            nested_config['metadata'] = nested_metadata
        if not nested_config:
            nested_config = None
        response = self.application.invoke(
            formulate_query(kwargs, is_subgraph=self.is_subgraph),
            config=nested_config,
        )
        normalized_output = extract_application_response_output(response)
        if self.return_type == "str":
            return normalized_output
        else:
            return {"messages": [{"role": "assistant", "content": normalized_output}]}
    
