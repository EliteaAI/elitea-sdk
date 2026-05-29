import json

from ..langchain.constants import ELITEA_RS, PRINTER_NODE_RS
from ..models.agent_response import AgentResponse
from ..utils.utils import clean_string
from langchain_core.tools import BaseTool, ToolException
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from typing import Any, Type, Optional
from pydantic import create_model, model_validator, BaseModel
from pydantic.fields import FieldInfo
from logging import getLogger

logger = getLogger(__name__)

applicationToolSchema = create_model(
    "applicatrionSchema", 
    task = (str, FieldInfo(description="Task for Application. Include all context needed by the application.")), 
    chat_history = (Optional[list[BaseMessage]], FieldInfo(description="Deprecated and ignored. Put all application context in task.", default=[]))
)


def formulate_query(kwargs, is_subgraph=False):
    """
    Formulate input for nested application invocation.

    Only passes user-defined business variables to child agent.
    Filters out internal/metadata keys that are specific to parent's execution context.
    """
    user_task = kwargs.get('task')
    if not user_task:
        raise ToolException("Task is required to invoke the application. "
                            "Check the provided input (some errors may happen on previous steps).")
    result = {"input": [HumanMessage(content=user_task)] if not is_subgraph else user_task}

    # Internal/metadata keys that should NOT be passed to child agent:
    # - task, chat_history: handled separately
    # - messages, input, output: graph I/O keys
    # - context_info: parent's summarization metadata
    # - state_types: parent's state schema definition
    # - hitl_decisions, hitl_interrupt: parent's HITL state
    # - thread_id, execution_finished: parent's execution state
    # - ELITEA_RS, PRINTER_NODE_RS: internal output keys
    excluded_keys = {
        "task", "chat_history",
        "messages", "input", "output",
        "context_info", "state_types",
        "hitl_decisions", "hitl_interrupt",
        "thread_id", "execution_finished",
        ELITEA_RS, PRINTER_NODE_RS,
    }

    for key, value in kwargs.items():
        if key not in excluded_keys:
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
            return '\n\n'.join(text_parts)

        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)

        return str(content)

    if isinstance(response, BaseMessage):
        return normalize_content(response.content)

    if isinstance(response, str):
        return normalize_content(response)

    if not isinstance(response, dict):
        return normalize_content(response)

    for key in ('output',):  # only 'output' is unambiguously "this node's result"
        normalized = normalize_content(response.get(key))
        if normalized.strip():
            return normalized

    # Prefer last AI message from messages over state variables like elitea_response
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

    # Only fall back to elitea_response / PRINTER_NODE_RS if messages had nothing useful
    for key in (ELITEA_RS, PRINTER_NODE_RS):
        normalized = normalize_content(response.get(key))
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
        extras = {k: v for k, v in input.items() if k not in schema_values and k != 'chat_history'}
        all_kwargs = {**kwargs, **extras, **schema_values}
        logger.debug(f"[APP_INVOKE] Input keys: {list(input.keys()) if isinstance(input, dict) else 'not a dict'}")
        logger.debug(f"[APP_INVOKE] Schema values: {schema_values}")
        logger.debug(f"[APP_INVOKE] Extras: {extras}")
        logger.debug(f"[APP_INVOKE] All kwargs keys: {list(all_kwargs.keys())}")
        if config is None:
            config = {}

        # IMPORTANT: Pass extras through config's configurable dict.
        # BaseTool._parse_input() validates against args_schema and strips any keys
        # not defined in the schema. Extra variables (like pipeline state variables)
        # get lost before reaching _run() unless preserved here. chat_history is a
        # deprecated reserved input and is intentionally not treated as an extra.
        # By storing them in config['configurable']['_application_extras'], they survive
        # the BaseTool pipeline and can be retrieved in _run().
        if extras:
            config = dict(config)
            config['configurable'] = dict(config.get('configurable') or {})
            config['configurable']['_application_extras'] = extras
            logger.debug(f"[APP_INVOKE] Stored extras in config: {list(extras.keys())}")

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
            # For LLM agent calls, we only need the message content, not state variables
            if isinstance(result, str):
                content = result
            elif isinstance(result, dict):
                # Extract message content from our result dict format
                # _run() returns {"messages": [{"role": "assistant", "content": "..."}], ...}
                messages = result.get("messages", [])
                if messages and isinstance(messages[0], dict):
                    content = messages[0].get("content", "")
                elif result.get("output"):
                    content = result.get("output")
                else:
                    content = str(result)
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
        logger.debug(f"[APP_RUN] kwargs received: {list(kwargs.keys())}")
        logger.debug(f"[APP_RUN] kwargs values: {kwargs}")
        invoke_config = config
        # Also consume legacy _invoke_config kwarg in case called directly in tests.
        invoke_config = kwargs.pop('_invoke_config', invoke_config)

        # Retrieve extras that were passed through config from invoke().
        # BaseTool._parse_input() strips keys not in args_schema, so extras (like pipeline
        # state variables) are stored in config['configurable']['_application_extras'].
        extras_from_invoke = {}
        if invoke_config and invoke_config.get('configurable'):
            extras_from_invoke = invoke_config['configurable'].pop('_application_extras', {})
            logger.debug(f"[APP_RUN] Retrieved extras from config: {list(extras_from_invoke.keys())}")
        # Merge extras into kwargs (extras take precedence as they come from the actual input)
        for k, v in extras_from_invoke.items():
            if k not in kwargs:
                kwargs[k] = v
                logger.debug(f"[APP_RUN] Added extra '{k}' to kwargs")

        if self.client and self.args_runnable:
            # Recreate new LanggraphAgentRunnable in order to reflect the current input_mapping (it can be dynamic for pipelines).
            # Actually, for pipelines agent toolkits LanggraphAgentRunnable is created (for LLMNode) before pipeline's schema parsing.

            # Merge variable defaults with passed kwargs
            # Defaults are applied first, then overridden by explicitly passed values
            # This ensures variables always have a value (either default or passed)
            merged_vars = dict(self.variable_defaults)  # Start with defaults
            for k, v in kwargs.items():
                if k == 'chat_history':
                    continue
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

            # Force is_subgraph=False: the child runs standalone here, so it
            # must be a LangGraphAgentRunnable, not a CompiledStateGraph
            # (checkpointer=True) which langgraph rejects as a root graph. (#5046)
            runnable_args = {**self.args_runnable, 'is_subgraph': False}
            self.application = self.client.application(**runnable_args, application_variables=application_variables)
        # Capture parent's pending intermediate messages BEFORE invoking the
        # child application. The child runs in the same thread/asyncio context
        # and its own __perform_tool_calling overwrites the shared
        # ``_PENDING_TOOL_MESSAGES`` ContextVar (llm.py:2111 resets it to []).
        # If the child triggers a HITL bubble-up, we need the parent's pending
        # — captured here — to merge into ``child_hitl_for_parent`` so the
        # parent's resume restores the parent's preceding tool calls/results
        # (e.g. the first subagent's call + result when the SECOND sequential
        # subagent triggers the interrupt).
        _parent_pending_serialized: list[dict] = []
        try:
            from langchain_core.messages import message_to_dict
            from .llm import _PENDING_TOOL_MESSAGES
            for _m in _PENDING_TOOL_MESSAGES.get([]):
                try:
                    _parent_pending_serialized.append(message_to_dict(_m))
                except Exception:
                    pass
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[APP_RUN] Failed to capture parent pending messages: %s", exc,
            )

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
            parent_configurable = dict(invoke_config['configurable'])
            parent_configurable.pop('selected_tools', None)
            parent_configurable.pop('selected_toolkits', None)
            if self.client and self.args_runnable:
                # Standalone child runs as a root graph with its own checkpointer.
                # Strip ALL parent pregel internals so the child doesn't inherit
                # the parent's checkpoint tree, scratchpad, or task tracking.
                _pregel_keys = [k for k in parent_configurable
                                if k.startswith('__pregel_') or k in (
                                    'checkpoint_id', 'checkpoint_ns', 'checkpoint_map',
                                )]
                for k in _pregel_keys:
                    parent_configurable.pop(k, None)
            parent_thread_id = parent_configurable.get('thread_id')
            if parent_thread_id and self.name:
                parent_configurable['thread_id'] = f"{parent_thread_id}:{self.name}"
            nested_config['configurable'] = parent_configurable
        if nested_metadata:
            nested_config['metadata'] = nested_metadata
        if not nested_config:
            nested_config = None
        try:
            response = self.application.invoke(
                formulate_query(kwargs, is_subgraph=self.is_subgraph),
                config=nested_config,
            )
        except Exception as gb:
            # GraphInterrupt propagation path (typically is_subgraph=True child,
            # or any child whose interrupt() raises directly without being
            # absorbed by an inner pregel). The exception's payload is already
            # in flight to the parent's pregel; mutate its value dict in place
            # so the parent's checkpoint records (a) parent tool identity and
            # (b) the PARENT's intermediate messages — without those, on resume
            # the parent's LLM history is restored from the child's view (or
            # nothing) and the parent re-plans from scratch, re-invoking
            # earlier sequential subagents.
            from langgraph.errors import GraphBubbleUp, GraphInterrupt
            if not isinstance(gb, GraphBubbleUp):
                raise
            if isinstance(gb, GraphInterrupt) and gb.args:
                for interrupts in gb.args:
                    if not isinstance(interrupts, (tuple, list)):
                        continue
                    for it in interrupts:
                        value = getattr(it, 'value', None)
                        if not isinstance(value, dict):
                            continue
                        if value.get('guardrail_type') != 'sensitive_tool':
                            continue
                        value.setdefault('_parent_tool_name', self.name)
                        value.setdefault('_parent_tool_args', {'task': kwargs.get('task', '')})
                        # Always drop the CHILD's pending messages so they can
                        # never leak into the parent checkpoint and pollute the
                        # parent LLM's resume history; attach the parent's
                        # captured pending (if any) in their place.
                        value.pop('_pending_messages', None)
                        if _parent_pending_serialized:
                            value['_pending_messages'] = _parent_pending_serialized
                            logger.info(
                                "[APP_RUN] Augmented bubbled HITL interrupt with %d "
                                "parent intermediate messages (tool=%s)",
                                len(_parent_pending_serialized), self.name,
                            )
            raise

        # HITL bubble-up (dict-bridge): when the child returns hitl_interrupt
        # in its state (e.g. subgraph mode or legacy path), propagate it to
        # the parent graph. On resume, interrupt() returns the user's decision.
        #
        # A single child can pause MULTIPLE times — e.g. it calls two distinct
        # sensitive tools in sequence. Each resume re-invokes the child, which
        # may surface a *new* hitl_interrupt for the next sensitive tool. Loop
        # until the child completes without pausing so every distinct approval
        # dialog is surfaced (the second interrupt was previously swallowed).
        # LangGraph replays the resume values positionally per node task, so
        # each interrupt() call resolves against its own scratchpad slot when
        # this tool re-executes on a later parent resume.
        while isinstance(response, dict) and response.get('hitl_interrupt'):
            child_hitl = response['hitl_interrupt']
            logger.info(
                "[APP_RUN] Child '%s' paused at HITL interrupt (tool=%s), bubbling to parent",
                self.name, child_hitl.get('tool_name', ''),
            )
            # Tag with parent tool identity so the parent's resume handler
            # builds _hitl_resume_context referencing THIS tool (Application),
            # not the child's leaf tool which doesn't exist in the parent graph.
            #
            # Replace the CHILD's _pending_messages (which describe the child's
            # internal state) with the PARENT's pending captured BEFORE the
            # child invocation — only the parent's pending is meaningful when
            # restored into the parent's LLM history on resume. The child
            # preserves its own pending in its own checkpoint for its own
            # resume cycle.
            child_hitl_for_parent = {
                **child_hitl,
                '_parent_tool_name': self.name,
                '_parent_tool_args': {'task': kwargs.get('task', '')},
            }
            # The {**child_hitl} spread copies the CHILD's own _pending_messages.
            # Always drop them so they can't leak into the parent checkpoint;
            # only the parent's pending is meaningful when restored into the
            # parent's LLM history on resume.
            child_hitl_for_parent.pop('_pending_messages', None)
            if _parent_pending_serialized:
                child_hitl_for_parent['_pending_messages'] = _parent_pending_serialized
                logger.info(
                    "[APP_RUN] Captured %d parent intermediate messages for HITL bubble-up",
                    len(_parent_pending_serialized),
                )
            resume_value = interrupt(child_hitl_for_parent)
            # Defensive: a non-dict resume value (LangGraph version drift, test
            # harness, or malformed Command(resume=...) payload) would raise
            # AttributeError on .get() inside the parent pregel loop. Coerce
            # so the bubble-up path degrades to a default-approve instead.
            if not isinstance(resume_value, dict):
                logger.warning(
                    "[APP_RUN] Non-dict resume value for child '%s' (type=%s), "
                    "defaulting to approve",
                    self.name, type(resume_value).__name__,
                )
                resume_value = {}
            logger.info("[APP_RUN] Resuming child '%s' with: %s", self.name, resume_value)
            response = self.application.invoke(
                {
                    "hitl_resume": True,
                    "hitl_action": resume_value.get("action", "approve"),
                    "hitl_value": resume_value.get("value", ""),
                },
                config=nested_config,
            )

        normalized_output = extract_application_response_output(response)

        # Build standardized AgentResponse
        agent_response = AgentResponse(
            output=normalized_output,
            messages=[{"role": "assistant", "content": normalized_output}],
            thread_id=response.get('thread_id') if isinstance(response, dict) else None,
            execution_finished=response.get('execution_finished', True) if isinstance(response, dict) else True,
        )

        # Propagate state variables from child response back to parent.
        # This allows FunctionTool to extract them based on output_variables,
        # enabling child pipeline state to flow back to parent pipeline.
        extra_state = {}
        if isinstance(response, dict):
            # Keys that are internal/output-related and should not be propagated as state
            excluded_keys = {'messages', 'output', 'input', 'chat_history', 'state_types',
                           'thread_id', 'execution_finished',
                           'hitl_decisions', 'hitl_interrupt',
                           ELITEA_RS, PRINTER_NODE_RS}
            for key, value in response.items():
                if key not in excluded_keys:
                    extra_state[key] = value
                    logger.debug(f"[APP_RUN] Propagating state variable '{key}' from child to parent")

        # Convert to dict with extra state variables
        result = {**agent_response.to_dict(), **extra_state}

        # Always return the full result dict with state variables.
        # The invoke() method will handle converting to string for LLM agent calls
        # (when tool_call_id is set), while FunctionTool calls (no tool_call_id)
        # will receive the full dict and can extract state variables.
        return result
    
