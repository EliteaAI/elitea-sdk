import asyncio
import contextvars
import json
import logging
import re
from traceback import format_exc
from typing import Any, Optional, List, Union, Literal, Dict, TYPE_CHECKING, cast
from uuid import NAMESPACE_URL, uuid4, uuid5

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks import dispatch_custom_event
from langgraph.errors import GraphBubbleUp
from langgraph.types import interrupt as _langgraph_interrupt
from pydantic import Field, ValidationError

try:
    from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD as _SCRATCHPAD_KEY
except ImportError:
    _SCRATCHPAD_KEY = '__pregel_scratchpad'

from ..langchain.constants import (
    ELITEA_RS,
    MAX_SKILLS_PER_INVOCATION,
    SKILL_REMINDER_SUFFIX,
    SKILLS_SECTION_ENTRY,
    SKILLS_SECTION_HEADER,
)
from ..langchain.utils import (
    args_match_normalized,
    create_pydantic_model,
    extract_json_content,
    make_anthropic_compatible_schema,
    normalize_null_tool_call_ids,
    propagate_the_input_mapping,
)
from ..exceptions import OutputContinuationExhausted, budget_exceeded_from
from ..toolkits.security import normalize_tool_name, qualified_tool_identity
from ..utils.mcp_oauth import (
    McpAuthorizationRequired,
    build_mcp_auth_decision_result,
)
from .hitl import (
    HITL_CANONICAL_INTERRUPT_ID_KEY,
    HITL_INTERRUPT_ID_KEY,
    HITL_NESTED_INTERRUPT_ID_KEY,
    HITL_TOOL_CALL_ID_KEY,
    HITL_VIA_CALL_ID_KEY,
    PendingHITLEntry,
)
from .skill_tools import (
    LoadSkillTool,
    build_load_skill_tools,
    loaded_skill_names_from_messages,
    render_skill_registry_index,
)
if TYPE_CHECKING:
    from .lazy_tools import ToolRegistry

logger = logging.getLogger(__name__)

SENSITIVE_TOOL_BLOCKED_RESULT_TYPE = 'sensitive_tool_blocked'
STRUCTURED_OUTPUT_PREFILL_PROMPT = "Now produce the structured output based on the information above."
NESTED_OUTPUT_CONTINUATION_LIMIT = 4
NESTED_OUTPUT_CONTINUATION_ANCHOR_MAX_CHARS = 160
NESTED_OUTPUT_CONTINUATION_MIN_OVERLAP_CHARS = 8
NESTED_OUTPUT_CLOSURE_MAX_TOKENS = 128
NESTED_OUTPUT_INVALID_SEAM_RETRY_LIMIT = 1
NESTED_REASONING_ONLY_CONTINUATION_PROMPT = (
    "The previous model attempt reached its output-token limit during internal "
    "reasoning before producing visible output. Produce the visible answer now. "
    "Output only the answer."
)

# ContextVar used by __perform_tool_calling to expose intermediate messages
# accumulated during the current LLMNode execution.  The sensitive-tool guard
# middleware reads this before calling interrupt() so the messages can be
# persisted in the checkpoint and restored on resume.
_PENDING_TOOL_MESSAGES: contextvars.ContextVar[list] = contextvars.ContextVar(
    '_pending_tool_messages', default=[],
)


def _args_match_normalized(args_a: dict, args_b: dict) -> bool:
    """Backwards-compatible alias to :func:`args_match_normalized` in utils.

    Kept for any external callers that imported the leading-underscore name
    from this module. New code should import ``args_match_normalized`` from
    ``elitea_sdk.runtime.langchain.utils`` directly.
    """
    return args_match_normalized(args_a, args_b)


# def _is_thinking_model(llm_client: Any) -> bool:
#     """
#     Check if a model uses extended thinking capability by reading cached metadata.
    
#     Thinking models require special message formatting where assistant messages
#     must start with thinking blocks before tool_use blocks.
    
#     This function reads the `_supports_reasoning` attribute that should be set
#     when the LLM client is created (by checking the model's supports_reasoning field).
    
#     Args:
#         llm_client: LLM client instance with optional _supports_reasoning attribute
        
#     Returns:
#         True if the model is a thinking model, False otherwise
#     """
#     if not llm_client:
#         return False
    
#     # Check if supports_reasoning was cached on the client
#     supports_reasoning = getattr(llm_client, '_supports_reasoning', False)
    
#     if supports_reasoning:
#         model_name = getattr(llm_client, 'model_name', None) or getattr(llm_client, 'model', 'unknown')
#         logger.debug(f"Model '{model_name}' is a thinking/reasoning model (cached from API metadata)")
    
#     return supports_reasoning

JSON_INSTRUCTION_TEMPLATE = (
        "\n\n**IMPORTANT: You MUST respond with ONLY a valid JSON object.**\n\n"
        "Required JSON fields:\n{field_descriptions}\n\n"
        "Example format:\n"
        "{{\n{example_fields}\n}}\n\n"
        "Rules:\n"
        "1. Output ONLY the JSON object - no markdown, no explanations, no extra text\n"
        "2. Ensure all required fields are present\n"
        "3. Use proper JSON syntax with double quotes for strings\n"
        "4. Do not wrap the JSON in code blocks or backticks"
    )

class LLMNode(BaseTool):
    """Enhanced LLM node with chat history and tool binding support"""
    
    # Override BaseTool required fields
    name: str = Field(default='LLMNode', description='Name of the LLM node')
    description: str = Field(default='This is tool node for LLM with chat history and tool support',
                             description='Description of the LLM node')

    # LLM-specific fields
    client: Any = Field(default=None, description='LLM client instance')
    return_type: str = Field(default="str", description='Return type')
    response_key: str = Field(default="messages", description='Response key')
    structured_output_dict: Optional[Dict[str, Any]] = Field(default=None, description='Structured output dictionary')
    output_variables: Optional[List[str]] = Field(default=None, description='Output variables')
    input_mapping: Optional[dict[str, dict]] = Field(default=None, description='Input mapping')
    input_variables: Optional[List[str]] = Field(default=None, description='Input variables')
    structured_output: Optional[bool] = Field(default=False, description='Whether to use structured output')
    available_tools: Optional[List[BaseTool]] = Field(default=None, description='Available tools for binding')
    tool_names: Optional[List[str]] = Field(default=None, description='Specific tool names to filter')
    steps_limit: Optional[int] = Field(default=25, description='Maximum steps for tool execution')
    tool_execution_timeout: Optional[int] = Field(default=900, description='Timeout (seconds) for tool execution. Default is 15 minutes.')

    # Lazy tools mode - reduces token usage by not binding all tools upfront
    lazy_tools_mode: Optional[bool] = Field(
        default=True,
        description='Enable lazy tools mode. When True, only meta-tools (list_toolkits, get_toolkit_tools, invoke_tool) '
                    'are bound to the LLM. The model uses these to discover and invoke any tool from the registry. '
                    'This dramatically reduces token usage for agents with many toolkits (30-100+).'
    )
    tool_registry: Optional[Any] = Field(
        default=None,
        exclude=True,
        description='ToolRegistry instance containing all tools organized by toolkit. '
                    'Required when lazy_tools_mode is True.'
    )
    always_bind_tools: Optional[List[BaseTool]] = Field(
        default=None,
        description='Tools that should always be bound directly to the LLM, even in lazy mode. '
                    'Used for middleware tools like planning that need immediate access. '
                    'These are bound alongside meta-tools, not through the registry.'
    )
    middleware_manager: Optional[Any] = Field(
        default=None,
        exclude=True,
        description='MiddlewareManager instance for before_model/after_model hooks. '
                    'Used for context management like summarization and context editing.'
    )
    child_dispatcher: Optional[Any] = Field(
        default=None,
        exclude=True,
        description='Optional parallel sub-agent dispatch seam (Track 2, issue #4993). '
                    'When present and a turn contains 2+ Application (sub-agent) tool calls, '
                    'the node PARKS by writing child specs to the parallel_tasks state channel '
                    'and returning instead of running children in-process. When None, the '
                    'node falls back to the in-process asyncio.gather fan-out (Track 1).'
    )
    independent_parallel_hitl: bool = Field(
        default=True,
        description='Single-process parallel HITL supervisor. '
                    'When enabled, paused Application children publish immediately '
                    'and accept exact live decisions while siblings keep running.'
    )
    parallel_hitl_max_concurrency: int = Field(
        default=8,
        ge=1,
        le=32,
        description='Maximum concurrently executing in-process Application children '
                    'when the independent parallel HITL supervisor is enabled.'
    )
    _meta_tools: Optional[List[BaseTool]] = None  # Cached meta-tools

    def _prepare_structured_output_params(self) -> dict:
        """
        Prepare structured output parameters from structured_output_dict.

        Expected self.structured_output_dict formats:
          - {"field": "str"} / {"field": "list"} / {"field": "list[dict]"} / {"field": "any"} ...
          - OR {"field": {"type": "...", "description": "...", "default": ...}}  (optional)

        Returns:
            Dict[str, Dict] suitable for create_pydantic_model(...)
        """
        struct_params: dict[str, dict] = {}

        for key, value in (self.structured_output_dict or {}).items():
            # Allow either a plain type string or a dict with details
            if isinstance(value, dict):
                type_str = str(value.get("type") or "any")
                desc = value.get("description", "") or ""
                entry: dict = {"type": type_str, "description": desc}
                if "default" in value:
                    entry["default"] = value["default"]
            else:
                # Ensure we always have a string type
                if isinstance(value, str):
                    type_str = value
                else:
                    # If it's already a type object, convert to string representation
                    type_str = getattr(value, '__name__', 'any')

                entry = {"type": type_str, "description": ""}

            struct_params[key] = entry

        # Add default output field for proper response to user
        struct_params[ELITEA_RS] = {
            "description": "final output to user (summarized output from LLM)",
            "type": "str",
            "default": None,
        }

        return struct_params

    @staticmethod
    def _strip_tool_use_blocks(content: Any) -> Any:
        """Drop ``tool_use`` blocks from Anthropic-shape list content.

        Anthropic returns assistant content as a list of typed blocks
        (``thinking``, ``text``, ``tool_use``). Sending an unmatched
        ``tool_use`` block back to the API triggers the
        "tool_use without tool_result" format error. String content is
        returned unchanged.
        """
        if isinstance(content, list):
            return [
                b for b in content
                if not (isinstance(b, dict) and b.get('type') == 'tool_use')
            ]
        return content

    def _build_clean_messages_for_structured_output(self, new_messages: List) -> List:
        """Return ``new_messages`` shaped for the structured-output follow-up
        call. Two contracts are enforced:

        1. **Tool-use sanitization.** The full tool exchange — all matched
           ``(tool_call → tool_result)`` pairs — is preserved so the
           structured-output call sees the data the model used for its
           synthesis. Only the last ``AIMessage`` is sanitized, and only
           when it carries unmatched ``tool_calls`` / ``tool_use`` blocks
           (the max-iterations exit case). Sending unmatched ``tool_use``
           triggers Anthropic's *"tool_use without tool_result"* error.

        2. **Trailing-user-message invariant.** When the conversation
           ends with an ``AIMessage`` (the synthesis turn), Anthropic's
           gateway treats it as an assistant prefill and rejects it with
           *"This model does not support assistant message prefill. The
           conversation must end with a user message."* Append a short
           ``HumanMessage`` to satisfy the invariant. Schema enforcement
           still happens at the API boundary via
           ``with_structured_output``; the HumanMessage is purely the
           "your turn" signal Anthropic requires.
        """
        if not new_messages:
            return list(new_messages)

        last_ai_index = None
        for i in range(len(new_messages) - 1, -1, -1):
            if isinstance(new_messages[i], AIMessage):
                last_ai_index = i
                break

        if last_ai_index is None:
            return list(new_messages)

        last_msg = new_messages[last_ai_index]
        cleaned_content = self._strip_tool_use_blocks(last_msg.content)
        has_tool_calls = bool(getattr(last_msg, 'tool_calls', None))
        content_changed = cleaned_content is not last_msg.content and cleaned_content != last_msg.content

        if has_tool_calls or content_changed:
            cleaned = AIMessage(
                content=cleaned_content,
                additional_kwargs=dict(getattr(last_msg, 'additional_kwargs', {}) or {}),
                response_metadata=dict(getattr(last_msg, 'response_metadata', {}) or {}),
                id=getattr(last_msg, 'id', None),
            )
            result = list(new_messages[:last_ai_index]) + [cleaned]
        else:
            result = list(new_messages)

        if isinstance(result[-1], AIMessage):
            result.append(HumanMessage(content=STRUCTURED_OUTPUT_PREFILL_PROMPT))
        return result

    def _invoke_with_structured_output(self, llm_client: Any, messages: List, struct_model: Any, config: RunnableConfig):
        """
        Invoke LLM with structured output, handling tool calls if present.

        Returns:
            Tuple of (completion, initial_completion, final_messages)

        Exceptions from the structured-output invocation propagate to the caller,
        which routes them through ``_handle_structured_output_fallback``. There is
        no local recovery path here — the Anthropic schema patch and json_schema
        routing in ``__get_struct_output_model`` keep the supported provider
        matrix functional, so the previous local recovery is dead code.
        """
        initial_completion = llm_client.invoke(messages, config=config)

        if hasattr(initial_completion, 'tool_calls') and initial_completion.tool_calls:
            # Tool-calling branch: run the agentic tool exchange first, then issue
            # the structured-output follow-up against the FULL ``new_messages``
            # history (including the matched tool_call/tool_result pairs). The
            # sanitizer below only touches the last AIMessage if it carries
            # unmatched tool calls — needed for max-iterations exits, harmless
            # otherwise.
            new_messages, _ = self._run_async_in_sync_context(
                self.__perform_tool_calling(initial_completion, messages, llm_client, config)
            )
            clean_messages = self._build_clean_messages_for_structured_output(new_messages)
            completion = self._synthesize_structured(llm_client, clean_messages, struct_model, config)
            return completion, initial_completion, new_messages

        completion = self._synthesize_structured(llm_client, messages, struct_model, config)
        return completion, initial_completion, messages

    def _synthesize_structured(self, llm_client: Any, synth_messages: List, struct_model: Any, config: RunnableConfig) -> Any:
        """Produce a structured completion from ``synth_messages``.

        ``with_structured_output`` makes the provider emit a ``tool_choice`` /
        ``response_format`` / ``json_schema`` transform. Some passthrough proxies
        reject that transform with a 400 (Bedrock: ``tool_choice.type``; Azure:
        ``Unknown parameter: 'tool_choice.function'``). We avoid it two ways:

        - **Proactively** for OpenAI-compatible passthrough clients (Claude via
          LiteLLM) — they always reject, so go straight to the JSON-prompt path.
        - **Reactively** for native clients — try ``with_structured_output`` and,
          if the proxy rejects the transform, fall back to the JSON-prompt path
          rather than leaking a 400 to the UI.

        The JSON-prompt path reuses the same extraction machinery as the
        fallback path: instruct the model to emit a JSON object, then parse it
        from the text response.
        """
        if self._client_is_openai_compatible(self.client):
            return self._structured_via_json_prompt(llm_client, synth_messages, struct_model, config)

        try:
            llm = self.__get_struct_output_model(llm_client, struct_model)
            return llm.invoke(synth_messages, config=config)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            # Fall back on two conditions:
            #   1. Provider 400-rejected the with_structured_output transform.
            #   2. Parser failed on the model's response (e.g. Anthropic extended
            #      thinking disables assistant prefill, so JsonOutputParser sees a
            #      body-only fragment without its leading '{').
            if not (isinstance(exc, OutputParserException)
                    or self._is_structured_transform_rejection(exc)):
                raise
            logger.warning(
                "Structured-output path failed (%s); "
                "retrying via JSON-prompt parsing", type(exc).__name__
            )
            return self._structured_via_json_prompt(llm_client, synth_messages, struct_model, config)

    @staticmethod
    def _is_structured_transform_rejection(exc: Exception) -> bool:
        """True when a provider 400-rejected the ``with_structured_output`` transform.

        Detects the bad-request signatures proxies raise when they don't support
        the ``tool_choice`` / ``response_format`` / ``json_schema`` shape litellm
        derives (e.g. Bedrock ``tool_choice.type``, Azure ``tool_choice.function``).
        """
        msg = str(getattr(exc, 'message', '') or exc).lower()
        is_bad_request = (
            'badrequest' in type(exc).__name__.lower()
            or 'badrequesterror' in msg
            or '400' in msg
            or 'invalid_request_error' in msg
        )
        if not is_bad_request:
            return False
        return any(
            marker in msg for marker in (
                'tool_choice', 'response_format', 'json_schema', 'output_format',
            )
        )

    def _structured_via_json_prompt(self, llm_client: Any, synth_messages: List, struct_model: Any, config: RunnableConfig) -> Any:
        """Prompt for a JSON object and parse it from the text response.

        Provider-agnostic structured-output path: no ``with_structured_output``
        transform, so it works on any proxy that rejects that transform.
        """
        json_instruction = self._build_json_instruction(struct_model)
        prompt_messages = list(synth_messages)
        last = prompt_messages[-1] if prompt_messages else None
        if isinstance(last, HumanMessage) and isinstance(last.content, str):
            prompt_messages[-1] = HumanMessage(content=last.content + json_instruction)
        else:
            prompt_messages.append(HumanMessage(content=json_instruction))

        completion = llm_client.invoke(prompt_messages, config=config)
        extracted = self._extract_structured_from_content(completion, struct_model)
        if extracted is not None:
            return extracted

        content = completion.content if hasattr(completion, 'content') else str(completion)
        if isinstance(content, list):
            content = ''.join(
                b.get('text', '') for b in content
                if isinstance(b, dict) and b.get('type') == 'text'
            )
        return self._create_fallback_completion(str(content).strip(), struct_model)

    def _build_json_instruction(self, struct_model: Any) -> str:
        """
        Build JSON instruction message for fallback handling.

        Args:
            struct_model: Pydantic model with field definitions

        Returns:
            Formatted JSON instruction string
        """
        field_descriptions = []
        for name, field in struct_model.model_fields.items():
            field_type = field.annotation.__name__ if hasattr(field.annotation, '__name__') else str(field.annotation)
            field_desc = field.description or field_type
            field_descriptions.append(f"  - {name} ({field_type}): {field_desc}")

        example_fields = ",\n".join([
            f'  "{k}": <{field.annotation.__name__ if hasattr(field.annotation, "__name__") else "value"}>'
            for k, field in struct_model.model_fields.items()
        ])

        return JSON_INSTRUCTION_TEMPLATE.format(
            field_descriptions="\n".join(field_descriptions),
            example_fields=example_fields
        )

    def _extract_structured_from_content(self, completion: Any, struct_model: Any) -> Any:
        """
        Try to extract structured output from an LLM response's text content.

        Handles models (especially Anthropic) that return valid JSON wrapped in
        markdown code fences as text content instead of using tool calls.

        Returns None if extraction fails.
        """
        try:
            content = completion.content if hasattr(completion, 'content') else str(completion)
            if isinstance(content, list):
                content = ''.join(
                    block.get('text', '') for block in content
                    if isinstance(block, dict) and block.get('type') == 'text'
                )
            content = content.strip()
            if not content:
                return None
            parsed = extract_json_content(content)
            return self._map_parsed_json_to_model(parsed, struct_model)
        except Exception as e:
            logger.debug(f"Content extraction failed: {e}")
            return None

    def _map_parsed_json_to_model(self, parsed: Any, struct_model: Any) -> Any:
        """
        Map parsed JSON (dict or list) to the structured output Pydantic model.

        Handles cases where:
        - parsed is a dict matching the model fields directly
        - parsed is a dict with a single key containing list data for a list field
        - parsed is a list that should map to the first list-type field
        """
        if isinstance(parsed, dict):
            model_fields = set(struct_model.model_fields.keys()) - {ELITEA_RS}
            if model_fields & set(parsed.keys()):
                return struct_model(**parsed)
            # Response has different field names — map by type
            list_fields = [
                k for k, f in struct_model.model_fields.items()
                if k != ELITEA_RS and getattr(f.annotation, '__origin__', None) is list
            ]
            if list_fields:
                for v in parsed.values():
                    if isinstance(v, list):
                        return struct_model(**{list_fields[0]: v})
            return struct_model(**parsed)
        elif isinstance(parsed, list):
            list_fields = [
                k for k, f in struct_model.model_fields.items()
                if k != ELITEA_RS and getattr(f.annotation, '__origin__', None) is list
            ]
            if list_fields:
                return struct_model(**{list_fields[0]: parsed})
        raise ValueError(f"Cannot map parsed JSON to model: {type(parsed)}")

    def _create_fallback_completion(self, content: str, struct_model: Any) -> Any:
        """
        Create a fallback completion object when JSON parsing fails.

        Args:
            content: Plain text content from LLM
            struct_model: Pydantic model to construct

        Returns:
            Pydantic model instance with fallback values
        """
        from pydantic_core import PydanticUndefined
        result_dict = {}
        for k, field in struct_model.model_fields.items():
            if k == ELITEA_RS:
                result_dict[k] = content
            elif field.is_required():
                # Required fields have PydanticUndefined as default - use None instead
                # to avoid serialization errors in LangGraph checkpoints
                result_dict[k] = None
            else:
                # Optional fields: use actual default, but guard against PydanticUndefined
                field_default = field.default
                result_dict[k] = None if field_default is PydanticUndefined else field_default
        return struct_model.model_construct(**result_dict)

    def _handle_structured_output_fallback(self, llm_client: Any, messages: List, struct_model: Any,
                                          config: RunnableConfig, original_error: Exception) -> Any:
        """Recover from a failed structured-output primary path.

        Delegates to ``_structured_via_json_prompt``, the provider-agnostic
        path. It already:
        - handles list-of-blocks content (Anthropic extended thinking)
        - handles code-fenced JSON
        - falls back gracefully via ``_create_fallback_completion`` when the
          model output genuinely cannot be parsed.

        This supersedes the old ``json_mode -> function_calling -> plain LLM``
        cascade, which repeated the same failing strategy (all three re-invoke
        ``with_structured_output`` whose JSON parser had already rejected the
        response) and crashed on list-content ``.strip()`` in the plain-LLM leg.
        """
        logger.warning(
            "Structured-output primary path failed (%s); delegating to JSON-prompt fallback",
            type(original_error).__name__,
        )
        logger.info("Original structured-output error: %s", format_exc())
        return self._structured_via_json_prompt(llm_client, messages, struct_model, config)

    def _format_structured_output_result(self, result: dict, messages: List, initial_completion: Any) -> dict:
        """
        Format structured output result with properly formatted messages.

        Args:
            result: Result dictionary from model_dump()
            messages: Original conversation messages
            initial_completion: Initial completion before tool calls

        Returns:
            Formatted result dictionary with messages
        """
        # Ensure messages are properly formatted
        if result.get('messages') and isinstance(result['messages'], list):
            result['messages'] = [{'role': 'assistant', 'content': '\n'.join(result['messages'])}]
        else:
            # Extract content from initial_completion, handling thinking blocks
            fallback_content = result.get(ELITEA_RS, '')
            if not fallback_content and initial_completion:
                content_parts = self._extract_content_from_completion(initial_completion)
                fallback_content = content_parts.get('text') or ''
                thinking = content_parts.get('thinking')

                # Log thinking if present
                if thinking:
                    logger.debug(f"Thinking content present in structured output: {thinking[:100]}...")

                if not fallback_content:
                    # Final fallback to raw content
                    content = initial_completion.content
                    fallback_content = content if isinstance(content, str) else str(content)

            result['messages'] = self._strip_system_messages(messages + [AIMessage(content=fallback_content)])

        return result

    def get_filtered_tools(self, config: Optional[Any] = None) -> List[BaseTool]:
        """
        Filter available tools based on tool_names list or return meta-tools in lazy mode.

        In lazy_tools_mode (default), returns only meta-tools that allow the model
        to discover and invoke any tool from the registry. This reduces token usage
        from potentially 100k+ tokens to ~2k tokens for agents with many toolkits.

        If dynamic tool selection was performed (selected_tools in config), those
        tools are returned directly instead of meta-tools.

        Always-bind tools (e.g., middleware/planning tools) are included alongside
        meta-tools in lazy mode, giving the model immediate access to these tools.

        Args:
            config: Optional runnable config that may contain selected_tools from
                    dynamic tool selection

        Returns:
            List of filtered tools (or meta-tools + always-bind tools in lazy mode)
        """
        configurable = config.get('configurable', {}) if isinstance(config, dict) and config else {}
        # Rebuilt per call: closes over this turn's attached_skills/invoked_skills.
        skill_tools = build_load_skill_tools(
            configurable.get('attached_skills'), configurable.get('invoked_skills')
        )

        def merge_skill_tools(tools):
            # Anthropic rejects duplicate tool names in the bind list; on a name
            # collision the pre-existing tool wins and load_skill is skipped.
            if not skill_tools:
                return tools
            taken = {getattr(t, 'name', None) for t in tools}
            merged = list(tools)
            for skill_tool in skill_tools:
                if skill_tool.name in taken:
                    logger.warning(
                        "[Skills] Tool name %r already bound by another toolkit — "
                        "skipping the progressive-disclosure skill tool this turn",
                        skill_tool.name,
                    )
                else:
                    merged.append(skill_tool)
            return merged

        # Check for dynamically selected tools from pre-LLM selection
        selected_tools = configurable.get('selected_tools')
        if selected_tools:
            logger.info(f"[DynamicToolSelection] Using {len(selected_tools)} pre-selected tools")
            # Fix for #3290: Always include always_bind_tools (e.g., Planner tools) with
            # dynamically selected tools. Use `or []` to handle None/falsy gracefully.
            # This ensures Planner tools are available even on first message when
            # Smart Tools Selection finds matching toolkits.
            return merge_skill_tools(list(selected_tools) + list(self.always_bind_tools or []))

        # Check if lazy tools mode is enabled and we have a registry
        if self.lazy_tools_mode and self.tool_registry is not None:
            meta_tools = self._get_meta_tools()
            # Include always-bind tools (e.g., planning tools) alongside meta-tools
            if self.always_bind_tools:
                combined_tools = list(meta_tools) + list(self.always_bind_tools)
                logger.info(
                    f"[LazyTools] Binding {len(meta_tools)} meta-tools + "
                    f"{len(self.always_bind_tools)} always-bind tools: "
                    f"{[t.name for t in self.always_bind_tools]}"
                )
                return merge_skill_tools(combined_tools)
            return merge_skill_tools(list(meta_tools))

        # Traditional mode - bind actual tools
        # Fix for #3382: Include always_bind_tools even when lazy mode is disabled
        # This ensures agent/pipeline tools are always available to the LLM
        base_tools = []

        if self.available_tools:
            if not self.tool_names:
                # If no specific tool names provided, use all available tools
                base_tools = list(self.available_tools)
            else:
                # Filter tools by name
                available_tool_names = {tool.name: tool for tool in self.available_tools}
                for tool_name in self.tool_names:
                    if tool_name in available_tool_names:
                        base_tools.append(available_tool_names[tool_name])
                        logger.debug(f"Added tool '{tool_name}' to LLM node")
                    else:
                        logger.warning(f"Tool '{tool_name}' not found in available tools: {list(available_tool_names.keys())}")

        # Always include always_bind_tools (agent/pipeline tools, planning tools)
        # These need direct LLM access regardless of lazy mode status
        if self.always_bind_tools:
            # Avoid duplicates - only add tools not already in base_tools
            existing_names = {t.name for t in base_tools}
            additional_tools = [t for t in self.always_bind_tools if t.name not in existing_names]
            if additional_tools:
                logger.info(
                    f"[DirectBinding] Including {len(additional_tools)} always-bind tools: "
                    f"{[t.name for t in additional_tools]}"
                )
                base_tools.extend(additional_tools)

        return merge_skill_tools(base_tools)

    def _get_meta_tools(self) -> List[BaseTool]:
        """
        Get or create meta-tools for lazy loading.

        Meta-tools are cached on first creation to avoid recreating them
        on every tool access.

        Returns:
            List of meta-tools [list_toolkits, get_toolkit_tools, invoke_tool]
        """
        if self._meta_tools is None:
            from .lazy_tools import create_meta_tools
            meta_tools = create_meta_tools(self.tool_registry)
            # Locally built, so nothing else applied the error contract. Guard is skipped:
            # registry tools are already guarded, guarding invoke_tool too double-prompts.
            if self.middleware_manager is not None:
                meta_tools = [
                    self.middleware_manager.wrap_tool(t, skip_sensitive_guard=True)
                    for t in meta_tools
                ]
            self._meta_tools = meta_tools
            logger.info(
                f"[LazyTools] Created {len(self._meta_tools)} meta-tools for "
                f"{len(self.tool_registry.get_toolkit_names())} toolkits, "
                f"{sum(len(self.tool_registry.get_toolkit_tools(t)) for t in self.tool_registry.get_toolkit_names())} tools"
            )
        return self._meta_tools

    def get_tool_index(self) -> str:
        """
        Generate a compressed tool index for inclusion in system prompt.

        This is only meaningful in lazy_tools_mode when a tool_registry is available.

        Returns:
            Formatted string with toolkit/tool index, or empty string if not applicable
        """
        if self.tool_registry is not None:
            return self.tool_registry.generate_index()
        return ""

    def _inject_tool_index_into_messages(self, messages: List) -> List:
        """
        Inject tool index into the system message for chat-based interactions.

        For lazy tools mode, the model needs to see what tools are available.
        This method finds the first SystemMessage and appends the tool index.

        Args:
            messages: List of messages from state

        Returns:
            Modified messages list with tool index injected into system message
        """
        if not self.tool_registry:
            return messages

        tool_index = self.tool_registry.generate_index()

        # Find and modify the system message
        modified_messages = []
        index_injected = False

        for msg in messages:
            if isinstance(msg, SystemMessage) and not index_injected:
                # Extract plain text from content regardless of whether it arrived as a
                # str or as an Anthropic-style content-block list (cache_control markup).
                # Without this guard, f"{list}\n\n{tool_index}" would stringify a Python
                # list object and corrupt the prompt.
                _existing_text = (
                    msg.content if isinstance(msg.content, str)
                    else next(
                        (b["text"] for b in msg.content if isinstance(b, dict) and b.get("type") == "text"),
                        ""
                    )
                )
                new_text = f"{_existing_text}\n\n{tool_index}"
                # Re-apply cache_control if this is an Anthropic client so that caching
                # is preserved after the tool-index injection.
                modified_messages.append(
                    SystemMessage(content=self._anthropic_system_content(new_text, self.client))
                )
                index_injected = True
                logger.debug("[LazyTools] Injected tool index into existing system message")
            else:
                modified_messages.append(msg)

        # If no system message found, prepend one with just the tool index
        if not index_injected:
            modified_messages.insert(0, SystemMessage(
                content=self._anthropic_system_content(tool_index, self.client)
            ))
            logger.debug("[LazyTools] Added new system message with tool index")

        return modified_messages

    def _get_tool_truncation_suggestions(self, tool_name: Optional[str]) -> str:
        """
        Get context-specific suggestions for how to reduce output from a tool.
        
        First checks if the tool itself provides truncation suggestions via 
        `truncation_suggestions` attribute or `get_truncation_suggestions()` method.
        Falls back to generic suggestions if the tool doesn't provide any.
        
        Args:
            tool_name: Name of the tool that caused the context overflow
            
        Returns:
            Formatted string with numbered suggestions for the specific tool
        """
        suggestions = None
        
        # Try to get suggestions from the tool itself
        if tool_name:
            filtered_tools = self.get_filtered_tools()
            for tool in filtered_tools:
                if tool.name == tool_name:
                    # Check for truncation_suggestions attribute
                    if hasattr(tool, 'truncation_suggestions') and tool.truncation_suggestions:
                        suggestions = tool.truncation_suggestions
                        break
                    # Check for get_truncation_suggestions method
                    elif hasattr(tool, 'get_truncation_suggestions') and callable(tool.get_truncation_suggestions):
                        suggestions = tool.get_truncation_suggestions()
                        break
        
        # Fall back to generic suggestions if tool doesn't provide any
        if not suggestions:
            suggestions = [
                "Check if the tool has parameters to limit output size (e.g., max_items, max_results, max_depth)",
                "Target a more specific path or query instead of broad searches",
                "Break the operation into smaller, focused requests",
            ]
        
        # Format as numbered list
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions))

    @staticmethod
    def _parse_sensitive_tool_blocked_result(tool_result: Any) -> Optional[Dict[str, Any]]:
        if isinstance(tool_result, dict) and tool_result.get('type') == SENSITIVE_TOOL_BLOCKED_RESULT_TYPE:
            return dict(tool_result)

        if isinstance(tool_result, str):
            stripped = tool_result.strip()
            if stripped.startswith('{') and stripped.endswith('}'):
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    return None
                if isinstance(payload, dict) and payload.get('type') == SENSITIVE_TOOL_BLOCKED_RESULT_TYPE:
                    return payload

        return None

    @staticmethod
    def _dispatch_injection_ack(injection_id: str, text: str, config) -> None:
        """Tell the UI this injection landed, and place it in the turn's timeline.

        Carries the text so the indexer can persist a trace-step marker: the
        injection then renders among the tool-call/thinking pins at the point it
        was actually consumed, instead of only inside the user's message bubble
        (which is scrolled away while a turn streams).
        """
        try:
            dispatch_custom_event(
                name="midturn_injection_consumed",
                data={"injection_id": injection_id, "text": text},
                config=config,
            )
        except Exception as e:
            # Non-fatal: the turn-end consumed list is the authoritative signal.
            logger.debug(f"Failed to dispatch injection ack for {injection_id}: {e}")

    @staticmethod
    def _filter_orphaned_tool_calls(messages: List) -> List:
        """Remove AI tool calls that lack matching tool results immediately after.

        Anthropic requires each tool_use block to have a corresponding tool_result
        in the immediately following message(s), before the next assistant message.
        This method filters both the `tool_calls` field and `tool_use` blocks in
        `content` (Anthropic's native format).
        """
        if not messages:
            return messages

        # Single pass: identify AIMessage indices and collect following ToolMessage ids
        # For each AIMessage with tool_calls, gather tool_call_ids from ToolMessages
        # that appear between it and the next AIMessage (or end of list).
        following_tool_ids: dict[int, set[str]] = {}
        current_ai_idx: int | None = None

        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage) and getattr(msg, 'tool_calls', None):
                current_ai_idx = i
                following_tool_ids[i] = set()
            elif isinstance(msg, ToolMessage) and current_ai_idx is not None:
                tc_id = getattr(msg, 'tool_call_id', None)
                if tc_id:
                    following_tool_ids[current_ai_idx].add(tc_id)

        # Early exit if no AIMessages with tool_calls
        if not following_tool_ids:
            return messages

        cleaned_messages: List = []
        filtered_count = 0

        for i, message in enumerate(messages):
            if i not in following_tool_ids:
                cleaned_messages.append(message)
                continue

            # This is an AIMessage with tool_calls - check for orphans
            valid_result_ids = following_tool_ids[i]
            tool_calls = message.tool_calls

            # Build valid tool_calls list and collect orphaned ids in one pass
            valid_tool_calls = []
            orphaned_ids: set[str] = set()
            for tc in tool_calls:
                tc_id = tc.get('id', '') if isinstance(tc, dict) else getattr(tc, 'id', '')
                if tc_id in valid_result_ids:
                    valid_tool_calls.append(tc)
                else:
                    orphaned_ids.add(tc_id)

            # No orphans - keep message as-is
            if not orphaned_ids:
                cleaned_messages.append(message)
                continue

            filtered_count += len(orphaned_ids)

            # Filter tool_use blocks from content if it's a list (Anthropic format)
            content = message.content
            if isinstance(content, list):
                # When no valid tool_calls remain, remove ALL tool_use blocks
                # Otherwise, remove only orphaned tool_use blocks
                if valid_tool_calls:
                    content = [
                        block for block in content
                        if not (isinstance(block, dict) and
                               block.get('type') == 'tool_use' and
                               block.get('id') in orphaned_ids)
                    ]
                else:
                    content = [
                        block for block in content
                        if not (isinstance(block, dict) and block.get('type') == 'tool_use')
                    ]

            # Skip message entirely if no content and no valid tool_calls
            if not valid_tool_calls and not content:
                continue

            # Create filtered message
            try:
                cleaned_messages.append(
                    message.model_copy(update={"tool_calls": valid_tool_calls, "content": content})
                )
            except Exception:
                cleaned_messages.append(AIMessage(content=content, tool_calls=valid_tool_calls))

        if filtered_count > 0:
            logger.info("Filtered %d orphaned tool_calls from message history", filtered_count)
        return cleaned_messages

    def _get_tool_identity(self, tool: BaseTool) -> Dict[str, Optional[str]]:
        metadata = getattr(tool, 'metadata', None) or {}
        toolkit_name = metadata.get('toolkit_name')
        toolkit_type = metadata.get('toolkit_type') or metadata.get('type')
        resolved_tool_name = normalize_tool_name(metadata.get('tool_name') or tool.name)

        if not toolkit_name and self.tool_registry is not None:
            toolkit_name = self.tool_registry.get_toolkit_for_tool(tool.name)

        if not toolkit_type and toolkit_name and self.tool_registry is not None:
            toolkit_type = self.tool_registry.get_toolkit_type(toolkit_name)

        return {
            'tool_name': resolved_tool_name,
            'toolkit_name': toolkit_name,
            'toolkit_type': toolkit_type,
        }

    @staticmethod
    def _build_blocked_tool_guidance(blocked_payload: Dict[str, Any]) -> str:
        # Fallback directive used ONLY when a blocked payload arrives without its
        # own `message` (the sensitive-tool guard is the source of truth and bakes
        # it in). Kept aligned with SensitiveToolGuardMiddleware.BLOCKED_TOOL_MESSAGE:
        # an explicit, imperative continue-instruction that does NOT end on a
        # terminal "stopped" note — weak models (haiku, gpt-5.4-mini) read a
        # terminal ending as "halt" and skip the rest of the workflow.
        action_label = (
            blocked_payload.get('action_label')
            or blocked_payload.get('blocked_tool_name')
            or blocked_payload.get('tool_name')
            or 'the requested action'
        )
        return (
            f"You declined THIS specific call to '{action_label}'; it was not executed. "
            "The block is for THIS invocation only, not the tool itself. "
            "This is NOT a stop signal — do not end your turn or summarize yet. "
            "Do not retry this same call with the same arguments, but DO continue: "
            "if more items remain, call the tool again for the NEXT item now; "
            "otherwise use another available tool to keep making progress. "
            "Only stop and ask the user when nothing remains that can be done without this exact declined call."
        )

    def invoke(
            self,
            state: Union[str, dict],
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
    ) -> dict:
        """
        Invoke the LLM node with proper message handling and tool binding.

        Args:
            state: The current state containing messages and other variables
            config: Optional runnable config
            **kwargs: Additional keyword arguments

        Returns:
            Updated state with LLM response
        """
        middleware_mgr = self.middleware_manager
        middleware_updates = []
        original_state = None

        # Run before_model hooks (may summarize messages)
        if middleware_mgr is not None and isinstance(state, dict):
            original_state = state.copy()
            state, middleware_updates = middleware_mgr.run_before_model(state, config or {})

        # Do LLM invocation
        try:
            result = self._invoke_llm_internal(state, config, middleware_updates)
        except GraphBubbleUp:
            raise
        except OutputContinuationExhausted:
            raise
        except Exception as e:
            # A budget rejection is a policy outcome with no recovery, so it must not
            # become message content the graph carries on with
            budget_error = budget_exceeded_from(e)
            if budget_error is not None:
                raise budget_error from e
            model_info = getattr(self.client, 'model_name', None) or getattr(self.client, 'model', 'unknown')
            logger.error(f"Error in LLM Node: {format_exc()}")
            logger.error(f"Model being used: {model_info}")
            logger.error(f"Error type: {type(e).__name__}")
            result = {"messages": [AIMessage(content=f"Error: {e}")]}

        # Run after_model hooks and add context_info
        if middleware_mgr is not None and isinstance(result, dict) and 'messages' in result:
            final_state = {**(original_state or state), 'messages': result['messages']}
            middleware_mgr.run_after_model(final_state, config or {})
            result['context_info'] = middleware_mgr.get_context_info()

        return result

    def _invoke_llm_internal(
            self,
            state: Union[str, dict],
            config: Optional[RunnableConfig],
            middleware_updates: list,
    ) -> dict:
        """
        Internal LLM invocation logic. Separated to allow automatic after_model hooks.

        Args:
            state: The current state (possibly modified by before_model hooks)
            config: Optional runnable config
            middleware_updates: RemoveMessage ops from before_model hooks

        Returns:
            Result dict with 'messages' key
        """

        func_args = propagate_the_input_mapping(input_mapping=self.input_mapping, input_variables=self.input_variables,
                                                state=state)

        # Check if dynamic tool selection was performed (affects tool index injection)
        configurable = config.get('configurable', {}) if isinstance(config, dict) and config else {}
        has_selected_tools = bool(configurable.get('selected_tools'))
        hitl_ctx = configurable.pop('_hitl_resume_context', None)

        source_node_name = hitl_ctx.get('source_node_name') if hitl_ctx else None
        if source_node_name and source_node_name != self.name:
            logger.info(
                "[HITL] Ignoring resume context from node '%s' in downstream node '%s'",
                source_node_name,
                self.name,
            )
            hitl_ctx = None

        # Guard: only honour the HITL resume context when the tool it
        # references actually belongs to *this* LLM node.  In pipelines
        # the HITL interrupt may have fired inside a preceding Toolkit
        # (FunctionTool) node; that node already executed the tool on
        # resume, but the context lingered in config and would cause this
        # LLM node to fabricate a synthetic tool call for a tool it does
        # not own (see #3966).
        #
        # When toolkit_name is present in the resume context we use
        # qualified identity (toolkit_name + tool_name) so that two
        # different toolkits that expose a tool with the same base name
        # (e.g. jira.create_issue vs github.create_issue) are correctly
        # distinguished.
        if hitl_ctx and hitl_ctx.get('tool_name'):
            ctx_tool = hitl_ctx['tool_name']
            ctx_toolkit = hitl_ctx.get('toolkit_name') or ''
            if ctx_toolkit:
                # Qualified comparison: build qualified identities for
                # every tool this LLM node owns and check membership.
                own_qualified = set()
                for t in (self.available_tools or []):
                    identity = self._get_tool_identity(t)
                    own_qualified.add(
                        qualified_tool_identity(
                            identity['tool_name'],
                            identity.get('toolkit_name'),
                        )
                    )
                ctx_qualified = qualified_tool_identity(ctx_tool, ctx_toolkit)
                if ctx_qualified not in own_qualified:
                    logger.info(
                        "[HITL] Ignoring stale _hitl_resume_context for '%s' "
                        "— not in this LLM node's tools %s",
                        ctx_qualified,
                        sorted(own_qualified) if own_qualified else '(none)',
                    )
                    hitl_ctx = None
            else:
                # Fallback: no toolkit info — use normalized base names so
                # that prefixed/aliased names (e.g. github___tool) still
                # match the base name from the HITL interrupt payload.
                own_tool_names = {normalize_tool_name(t.name) for t in (self.available_tools or [])}
                if normalize_tool_name(ctx_tool) not in own_tool_names:
                    logger.info(
                        "[HITL] Ignoring stale _hitl_resume_context for tool '%s' "
                        "— not in this LLM node's tools %s",
                        ctx_tool,
                        sorted(own_tool_names) if own_tool_names else '(none)',
                    )
                    hitl_ctx = None

        # Set in the system branch (gates the skill registry), reused at bind time.
        prebuilt_filtered_tools = None

        # there are 2 possible flows here: LLM node from pipeline (with prompt and task)
        # or standalone LLM node for chat (with messages only)
        if 'system' in func_args.keys():
            # Flow for LLM node with prompt/task from pipeline
            if func_args.get('system') is None or func_args.get('task') is None:
                raise ToolException(f"LLMNode requires 'system' and 'task' parameters in input mapping. "
                                    f"Actual params: {func_args}")
            # cast to str in case user passes variable different from str
            system_content = str(func_args.get('system'))

            # Inject tool index into system prompt if lazy tools mode is enabled
            # Skip injection if dynamic tool selection provided actual tools
            if self.lazy_tools_mode and self.tool_registry is not None and not has_selected_tools:
                tool_index = self.tool_registry.generate_index()
                system_content = f"{system_content}\n\n{tool_index}"
                logger.debug("[LazyTools] Injected tool index into system prompt")

            # Skill registry (names + descriptions) goes into the CACHED prefix,
            # rendered in fixed (skill_id, name) order so the block stays
            # byte-stable across turns. Advertised only when a LoadSkillTool
            # survived the merge — on a name collision the registry would point
            # the model at the imposter tool.
            skill_registry_advertised = False
            if configurable.get('attached_skills'):
                prebuilt_filtered_tools = self.get_filtered_tools(config=config)
                if any(isinstance(t, LoadSkillTool) for t in prebuilt_filtered_tools):
                    skill_registry = render_skill_registry_index(configurable.get('attached_skills'))
                    if skill_registry:
                        skill_registry_advertised = True
                        system_content = (
                            f"{system_content}\n\n{skill_registry}" if system_content else skill_registry
                        )
                else:
                    logger.warning(
                        "[Skills] load_skill not bound (name collision) — "
                        "suppressing skill registry injection this turn"
                    )

            # Per-turn skills injection. elitea_core resolves the
            # ~skill-name token(s) from THIS user message and threads the resolved bodies
            # through invoke_config["configurable"]["invoked_skills"]. The rendered SKILLS
            # section is kept OUT of the cached static block and passed to
            # _anthropic_system_content as a dynamic suffix: for Anthropic it becomes a
            # separate block AFTER the cache breakpoint, so a skill-invoking turn does not
            # bust the cached prefix (instructions + tool schemas). Empty/absent ⇒ no-op,
            # so behavior is byte-identical when no skill was invoked. The injected text rides
            # the System message and is stripped before checkpoint (_strip_system_messages).
            skills_section = self._build_invoked_skills_section(configurable.get('invoked_skills'))
            if skills_section:
                logger.info("[Skills] Injected per-turn skills section into system prompt")
            # Recency counterweight: the registry sits in the cached prefix, far from
            # the decision point, and loses to transcript anchoring on small models.
            # A one-line reminder in the (already uncached) dynamic suffix puts the
            # instruction last, where it competes with the conversation itself.
            if skill_registry_advertised:
                skills_section = (
                    f"{skills_section}\n\n{SKILL_REMINDER_SUFFIX}" if skills_section
                    else SKILL_REMINDER_SUFFIX
                )

            task_content = func_args.get('task')
            if not isinstance(task_content, (str, list)):
                task_content = str(task_content) if task_content is not None else ""
            _chat_history = list(func_args.get('chat_history', []))
            # When chat_history already ends in a ToolMessage we are RESUMING an
            # in-progress tool loop (the #4993 park/reconcile re-invoke: children
            # settled, their results were appended as ToolMessages, and the graph
            # re-enters this node to synthesize). Re-appending the original task as
            # a trailing HumanMessage makes the model read the conversation as
            # "the user is asking again" — so it re-dispatches the same sub-agents
            # instead of synthesizing, looping forever on Anthropic models (GPT
            # tolerates the duplicate; haiku/sonnet do not). The in-process gather
            # path never hits this because it loops inside __perform_tool_calling
            # on new_messages and never rebuilds the prompt. End on the
            # ToolMessages so the next turn is a pure synthesis turn.
            _resuming_tool_loop = bool(_chat_history) and isinstance(_chat_history[-1], ToolMessage)
            # Omit the system message entirely when content is empty (e.g. the 'bare'
            # persona with no custom instructions and no addon-contributing tools).
            # Sending SystemMessage(content="") is not "no system prompt" — some
            # providers reject or warn on an empty system field — so we drop it.
            _system_msgs = (
                [SystemMessage(content=self._anthropic_system_content(system_content, self.client, skills_section))]
                if system_content else []
            )
            if _resuming_tool_loop:
                messages = [
                    *_system_msgs,
                    *_chat_history,
                ]
            else:
                messages = [
                    *_system_msgs,
                    *_chat_history,
                    HumanMessage(content=task_content),
                ]
                # Remove pre-last item if last two messages are same type and content
                if len(messages) >= 2 and type(messages[-1]) == type(messages[-2]) and messages[-1].content == messages[
                    -2].content:
                    messages.pop(-2)
        else:
            # Flow for chat-based LLM node w/o prompt/task from pipeline but with messages in state
            # verify messages structure
            messages = state.get("messages", []) if isinstance(state, dict) else []
            if messages:
                # Filter out all system messages except the first one to avoid
                # "multiple non-consecutive system messages" error from Anthropic API.
                # In swarm mode, multiple agents may add their system messages to shared state.
                first_system_msg = None
                filtered_messages = []
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        if first_system_msg is None:
                            first_system_msg = msg
                        # Skip subsequent system messages
                    else:
                        filtered_messages.append(msg)
                # Prepend the first system message if found
                if first_system_msg:
                    messages = [first_system_msg] + filtered_messages
                else:
                    messages = filtered_messages

                messages = self._filter_orphaned_tool_calls(messages)

                if not messages:
                    raise ToolException("LLMNode requires 'messages' in state for chat-based interaction")

                # Fresh chat turns must end with a user message.
                # HITL resumes replay a previously reviewed tool call, so the checkpoint
                # may legitimately end in an AI tool call message instead.
                if not hitl_ctx and not isinstance(messages[-1], HumanMessage):
                    raise ToolException("LLMNode requires the last message to be a HumanMessage")

                # Inject tool index into system message if lazy tools mode is enabled
                # Skip injection if dynamic tool selection provided actual tools
                if self.lazy_tools_mode and self.tool_registry is not None and not has_selected_tools:
                    messages = self._inject_tool_index_into_messages(messages)
            else:
                raise ToolException("LLMNode requires 'messages' in state for chat-based interaction")

        # Count of durable base messages the graph re-supplies on every
        # resume (the checkpointed state before this node ran, typically
        # [system, human]).  Captured BEFORE restoring intermediate history so
        # the pending-capture window in __perform_tool_calling extends back
        # across the restored region and carries the FULL cumulative tool
        # history forward to the next interrupt.  Without this, each resume
        # cycle's pending would contain only that cycle's slice and earlier
        # executed-tool results would be shed, causing the LLM to re-plan from
        # scratch and re-invoke already-approved sensitive tools (#5245).
        _durable_base_count = len(messages)

        if hitl_ctx and hitl_ctx.get('pending_messages'):
            from langchain_core.messages.utils import messages_from_dict

            try:
                restored_messages = messages_from_dict(hitl_ctx['pending_messages'])
                messages = list(messages) + list(restored_messages)
                # A delegated-auth checkpoint is restored immediately before
                # its structured ToolMessage is appended below. Filtering here
                # would delete the temporarily-unmatched AI tool call and leave
                # an orphan ToolMessage, rejected by Anthropic and OpenAI alike.
                if not hitl_ctx.get('mcp_auth_payload'):
                    messages = self._filter_orphaned_tool_calls(messages)
                logger.info(
                    "[HITL] Restored %d intermediate messages into LLM node history",
                    len(restored_messages),
                )
            except Exception as exc:
                logger.warning(
                    "[HITL] Failed to restore intermediate messages into LLM node history: %s",
                    exc,
                )

        # Get the LLM client, potentially with tools bound
        llm_client = self.client

        # Bind tools when:
        # 1. Traditional mode: specific tool_names are provided, OR
        # 2. Lazy mode: tool_registry exists (meta-tools will be bound), OR
        # 3. available_tools exist (covers lazy mode auto-disabled case)
        should_bind_tools = (
            len(self.tool_names or []) > 0 or
            (self.lazy_tools_mode and self.tool_registry is not None) or
            bool(self.available_tools) or  # Bind available tools even when lazy mode auto-disabled
            bool(configurable.get('attached_skills'))  # Bind load_skill for progressive disclosure
        )

        if should_bind_tools:
            filtered_tools = (
                prebuilt_filtered_tools if prebuilt_filtered_tools is not None
                else self.get_filtered_tools(config=config)
            )
            if filtered_tools:
                logger.info(f"Binding {len(filtered_tools)} tools to LLM: {[t.name for t in filtered_tools]}")
                llm_client = self.client.bind_tools(filtered_tools)
            else:
                logger.warning("No tools to bind to LLM")

        if self.structured_output and self.output_variables:
            # Handle structured output
            struct_params = self._prepare_structured_output_params()
            struct_model = create_pydantic_model(f"LLMOutput", struct_params)

            try:
                completion, initial_completion, final_messages = self._invoke_with_structured_output(
                    llm_client, messages, struct_model, config
                )
            except (ValueError, ValidationError, OutputParserException) as e:
                # Single recovery point for any structured-output failure.
                completion = self._handle_structured_output_fallback(
                    llm_client, messages, struct_model, config, e
                )
                initial_completion = None
                final_messages = messages

            # Normalize to dict regardless of provider. Anthropic's path
            # passes a JSON-schema dict to ``with_structured_output`` (the
            # ``$defs.JsonValue`` patch lives in ``__get_struct_output_model``),
            # so its runnable yields ``dict`` directly. OpenAI / Azure /
            # Google / extraction-fallback all yield Pydantic instances.
            # Either way, the consumer wants a dict.
            result = completion if isinstance(completion, dict) else completion.model_dump()

            # Anthropic's dict-schema path (see __get_struct_output_model) hands
            # back the raw tool-call arguments with no Pydantic model to filter
            # them through, so any extra key the LLM includes in its JSON output
            # (e.g. because the node's own prompt mentions a differently-named
            # field) passes straight into ``result`` and later gets merged into
            # the graph state, silently overwriting an unrelated pipeline
            # variable that happens to share that name (#6375). Restrict to the
            # node's declared schema fields for every provider so this can't
            # happen even where Pydantic validation was already filtering it.
            allowed_keys = set(struct_model.model_fields.keys())
            result = {k: v for k, v in result.items() if k in allowed_keys}

            result = self._format_structured_output_result(result, final_messages, initial_completion or completion)

            # Prepend middleware updates to messages for checkpoint
            if middleware_updates and 'messages' in result:
                result['messages'] = list(middleware_updates) + result['messages']

            return result

        # Handle regular completion
        #
        # HITL guardrail resume: If a sensitive-tool guard paused execution via
        # interrupt(), LangGraph re-executes this node from scratch. The LLM
        # call is non-deterministic, so re-calling it may produce a completely
        # different response (no tool call -> tool never runs). To avoid this,
        # the graph-level resume path injects `_hitl_resume_context` into the
        # config. When present, we skip the LLM call and build a synthetic
        # AIMessage with the reviewed tool call so the normal
        # __perform_tool_calling loop can execute it. The guard will then
        # resolve the resume action consistently: approve executes the tool,
        # reject returns a blocked-tool result and gives the LLM another turn.
        if hitl_ctx and hitl_ctx.get('mcp_auth_payload'):
            # ---- Delegated toolkit authorization resume (#6072) ----
            # The tool already reached the exact authorization boundary before
            # the checkpoint was written.  Do not ask the LLM to recreate that
            # call (which can select a different tool or, for a nested agent,
            # create a different child invocation).  Consume the Command resume
            # here and close the original leaf tool_call with a structured
            # ToolMessage, then let the same LLM node continue from its restored
            # intermediate history.
            auth_payload = dict(hitl_ctx['mcp_auth_payload'])
            scratchpad = configurable.get(_SCRATCHPAD_KEY)
            n_prior = (
                len(scratchpad.resume)
                if scratchpad
                and hasattr(scratchpad, 'resume')
                and scratchpad.resume
                else 0
            )
            for _i in range(n_prior):
                try:
                    _langgraph_interrupt({'__replay_consumer__': True})
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "[MCP_AUTH] Replay consumer #%d raised %s — stopping "
                        "before authorization resume",
                        _i, exc,
                    )
                    break

            resume_value = _langgraph_interrupt(auth_payload)
            action = (
                str((resume_value or {}).get('action') or 'skip').strip().lower()
                if isinstance(resume_value, dict)
                else 'skip'
            )
            if action not in {'authorize', 'skip'}:
                action = 'skip'
            tool_call_id = (
                auth_payload.get('tool_call_id') or 'mcp_auth_resume_call'
            )
            completion = next(
                (
                    message for message in reversed(messages)
                    if isinstance(message, AIMessage)
                    and any(
                        str(call.get('id') or '') == tool_call_id
                        for call in (message.tool_calls or [])
                        if isinstance(call, dict)
                    )
                ),
                None,
            )
            if completion is None:
                # Defensive fallback for an old/incomplete checkpoint. New
                # checkpoints always carry the original AIMessage in
                # _pending_messages, preserving provider-specific content.
                completion = AIMessage(
                    content='',
                    tool_calls=[{
                        'name': auth_payload.get('_tool_call_name') or 'mcp_auth_control',
                        'args': auth_payload.get('tool_args_raw') or {},
                        'id': tool_call_id,
                    }],
                )
                messages = list(messages) + [completion]
            decision_result = self._mcp_auth_decision_message(
                auth_payload, action,
            )
            messages = list(messages) + [ToolMessage(
                content=decision_result,
                tool_call_id=tool_call_id,
            )]
            # The authorization boundary is completed synthetically on resume:
            # the original tool call must not execute again, but its structured
            # result is still a real step in the leaf's history.  Emit an
            # explicit completion event so the worker persists the same
            # LLM -> tool -> LLM timeline that ordinary tools produce.  Without
            # this, resolved auth cards disappear and the child accordion looks
            # as though the guarded call never happened.
            dispatch_custom_event(
                name="mcp_auth_decision",
                data={
                    "tool_name": (
                        auth_payload.get("_tool_call_name")
                        or "mcp_auth_control"
                    ),
                    "tool_call_id": tool_call_id,
                    "toolkit_name": auth_payload.get("toolkit_name") or "",
                    "toolkit_type": auth_payload.get("toolkit_type") or "",
                    "tool_output": decision_result,
                    "action": action,
                },
                config=config,
            )
        elif hitl_ctx and hitl_ctx.get('parallel_calls'):
            # ---- Parallel sub-agent resume (issue #4993) ----
            # The original turn fanned out 2+ Application calls and the parent
            # paused on ONE aggregated interrupt (not the single-tool guard).
            # Rebuild the original AIMessage carrying ALL N tool_calls so
            # __perform_tool_calling re-enters the fan-out: completed siblings
            # are skipped (their ToolMessages were restored above), and each
            # paused child is resumed from its own checkpoint via the matching
            # hitl_decisions entry.
            #
            # Multi-round parallel HITL: a resumed child whose LLM picks a new
            # sensitive tool re-pauses, and _run_parallel_application_calls
            # re-issues a fresh parent-level interrupt(aggregate). That aggregate
            # is the FIRST interrupt() of THIS re-execution, so without help it
            # consumes a still-pending resume value and RETURNS it instead of
            # raising — swallowing the divergent child's new pause.
            #
            # LangGraph (1.x) delivers resume values two ways (see
            # langgraph/pregel/_algo.py::_scratchpad + langgraph/types.py::interrupt):
            #   * the scalar Command(resume=X) of THIS cycle arrives as the
            #     "null resume" (one per cycle); the first interrupt() consumes it.
            #   * values consumed by interrupt() in PRIOR cycles are persisted as
            #     task-specific positional `scratchpad.resume` entries and replayed
            #     by index on every later re-execution.
            # So the count of pending values the aggregate would otherwise eat is
            # len(scratchpad.resume) (positional, prior rounds) + 1 (this cycle's
            # null). Consume them all here so the aggregate interrupt() lands past
            # them and actually raises. Child decisions ride the SEPARATE
            # invocation-scoped `hitl_decisions` resume context, so consuming the
            # parent's resume values never robs a child of its answer.
            scratchpad = configurable.get(_SCRATCHPAD_KEY)
            n_positional = (
                len(scratchpad.resume)
                if scratchpad is not None
                and getattr(scratchpad, 'resume', None)
                else 0
            )
            has_null = False
            if scratchpad is not None and hasattr(scratchpad, 'get_null_resume'):
                try:
                    has_null = scratchpad.get_null_resume(False) is not None
                except Exception:  # pragma: no cover - defensive
                    has_null = False
            # A true Application fan-out resumes its children from the
            # invocation-scoped decision list; its next aggregate interrupt
            # must therefore advance past the current Command null resume too.
            # A sequential A -> B bridge for a raw aggregate bubbled from inside
            # B is different (even if the saved assistant turn also contained a
            # regular tool). Application._run
            # must consume the current Command value itself and forward the
            # complete decision list to B. Eating that null here strands B at
            # its old checkpoint and makes the leaves replay one by one.
            consume_current_null = not hitl_ctx.get(
                'sequential_application_bridge', False,
            )
            n_prior = n_positional + (
                1 if has_null and consume_current_null else 0
            )
            if n_prior:
                logger.info(
                    "[HITL] Consuming %d pending parent resume value(s) before "
                    "parallel sub-agent re-fanout (multi-round): %d positional "
                    "+ %d null",
                    n_prior, n_positional,
                    1 if has_null and consume_current_null else 0,
                )
                for _i in range(n_prior):
                    try:
                        _langgraph_interrupt({'__replay_consumer__': True})
                    except Exception as exc:
                        logger.warning(
                            "[HITL] Parallel replay consumer #%d raised %s — "
                            "stopping replay consumption early",
                            _i, exc,
                        )
                        break
            completion = None
            original_ai = hitl_ctx.get('original_ai_message')
            if isinstance(original_ai, dict):
                try:
                    from langchain_core.messages.utils import messages_from_dict
                    restored = messages_from_dict([original_ai])
                    if restored and isinstance(restored[0], AIMessage):
                        completion = restored[0]
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "[HITL] Failed to deserialize original AIMessage for "
                        "parallel resume: %s", exc,
                    )
            if completion is None:
                completion = AIMessage(
                    content=hitl_ctx.get('content', ''),
                    tool_calls=list(hitl_ctx['parallel_calls']),
                )
        elif hitl_ctx and hitl_ctx.get('tool_name'):
            # ---- Consume stale interrupt replay values ----
            # LangGraph replays ALL previously consumed interrupt/resume
            # values from prior resumes of this task (node execution).
            # Each interrupt() call returns the stored value at its
            # positional index.  Because the synthetic AIMessage below
            # contains ONLY the current HITL tool, the guard's
            # interrupt() would land at index 0 and receive a stale
            # value from an earlier resume instead of the current one.
            # Fix: advance the interrupt counter past the stale entries
            # so the guard's interrupt() gets the correct (current)
            # resume value.
            scratchpad = configurable.get(_SCRATCHPAD_KEY)
            n_prior = (
                len(scratchpad.resume)
                if scratchpad
                and hasattr(scratchpad, 'resume')
                and scratchpad.resume
                else 0
            )
            if n_prior:
                logger.info(
                    "[HITL] Consuming %d stale interrupt replay value(s) "
                    "before sensitive-tool resume",
                    n_prior,
                )
                for _i in range(n_prior):
                    try:
                        _langgraph_interrupt({'__replay_consumer__': True})
                    except Exception as exc:
                        logger.warning(
                            "[HITL] Replay consumer #%d raised %s — stopping "
                            "replay consumption early (may indicate misaligned "
                            "checkpoint state)",
                            _i, exc,
                        )
                        break

            # Create synthetic AIMessage with the reviewed tool call.
            #
            # Anthropic tool-calling turns can carry provider-specific content
            # blocks (for example thinking, redacted_thinking, or later text
            # blocks preceding tool_use). Replacing the original tool-calling
            # AIMessage with ``content=''`` strips that context and can cause
            # resumed runs to lose continuity. The graph-level resume handler
            # captures the original AIMessage into
            # ``hitl_ctx['original_ai_message']``. When its tool_call matches
            # the resumed tool, reuse it as the completion so the full original
            # assistant message shape survives the resume. Only fall back to the
            # empty synthetic AIMessage when the original cannot be reused.
            completion = self._build_resume_completion(hitl_ctx, messages)
            if completion is None:
                # Fallback: preserve content from the original AIMessage when
                # available.  For Anthropic thinking models, the original
                # AIMessage carries thinking/redacted_thinking blocks that MUST
                # be present in the content for the follow-up LLM call to
                # succeed.  Using content='' causes the API to reject the
                # request or the LLM to lose context and re-invoke all tools.
                resume_content = self._extract_original_content_for_resume(hitl_ctx)
                completion = AIMessage(
                    content=resume_content,
                    tool_calls=[{
                        'name': hitl_ctx['tool_name'],
                        'args': hitl_ctx.get('tool_args', {}),
                        'id': hitl_ctx.get('tool_call_id', 'hitl_resume_call'),
                    }],
                )
        else:
            completion = llm_client.invoke(messages, config=config)
        completion = self._continue_nested_output(
            messages=messages,
            completion=completion,
            config=config,
        )
        logger.info(f"Initial completion: {completion}")

        # Handle both tool-calling and regular responses
        if hasattr(completion, 'tool_calls') and completion.tool_calls:
            # Handle iterative tool-calling and execution
            # The invocation-scoped resume context carries checkpoint-hydrated
            # private routing fields. Prefer it over the durable audit channel so
            # `_via_call_id` / `_nested_interrupt_id` never need to be persisted
            # in graph state or surfaced with ordinary state values.
            hitl_decisions = (
                hitl_ctx.get('hitl_decisions')
                if hitl_ctx and isinstance(hitl_ctx.get('hitl_decisions'), list)
                else state.get('hitl_decisions') if isinstance(state, dict) else None
            )

            # __perform_tool_calling deduplicates the completion against
            # `messages` internally (multi-tool sibling HITL resume case),
            # so we can pass the full `messages` here unconditionally.
            #
            # parked_holder is a mutable hand-off for the Track 2 (#4993)
            # park-by-returning path. It is passed by reference (NOT a contextvar)
            # so the parked signal survives even when _run_async_in_sync_context
            # runs the coroutine in a worker thread with copy_context() — the
            # thread mutates the same dict object the caller holds.
            parked_holder: Dict[str, Any] = {}
            new_messages, current_completion = self._run_async_in_sync_context(
                self.__perform_tool_calling(
                    completion, messages, llm_client, config,
                    hitl_decisions=hitl_decisions,
                    pending_hitl_entries=(
                        hitl_ctx.get('parallel_pending')
                        if hitl_ctx and isinstance(hitl_ctx.get('parallel_pending'), list)
                        else None
                    ),
                    pending_capture_base=_durable_base_count,
                    parked_holder=parked_holder,
                )
            )

            output_msgs = {"messages": self._prepare_output_messages(new_messages, middleware_updates)}
            if parked_holder.get('parked'):
                # Parallel fan-out parked for durable dispatch. Write the child
                # specs into the parallel_tasks state channel so they survive the
                # checkpoint; the LangGraphAgentRunnable reads this back and emits
                # the parked result shape (execution_finished=False). The fresh
                # parallel_reconcile invocation later reads each child's own
                # checkpoint to assemble ToolMessages and continue the loop.
                output_msgs['parallel_tasks'] = {
                    'parked': True,
                    'dispatch_epoch': parked_holder.get('dispatch_epoch'),
                    'children': parked_holder.get('children', {}),
                }
                return output_msgs
            if self.output_variables:
                if self.output_variables[0] == 'messages':
                    return output_msgs
                # Extract content properly from thinking-enabled responses
                if current_completion:
                    content_parts = self._extract_content_from_completion(current_completion)
                    text_content = content_parts.get('text')
                    thinking = content_parts.get('thinking')

                    # Dispatch thinking event if present
                    if thinking:
                        try:
                            model_name = getattr(llm_client, 'model_name', None) or getattr(llm_client, 'model', 'LLM')
                            dispatch_custom_event(
                                name="thinking_step",
                                data={
                                    "message": thinking,
                                    "tool_name": f"LLM ({model_name})",
                                    "toolkit": "reasoning",
                                },
                                config=config,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to dispatch thinking event: {e}")

                    if text_content:
                        output_msgs[self.output_variables[0]] = text_content
                    else:
                        # Fallback to raw content
                        content = current_completion.content
                        output_msgs[self.output_variables[0]] = content if isinstance(content, str) else str(content)
                else:
                    output_msgs[self.output_variables[0]] = None

            return output_msgs

        # Regular text response - handle both simple strings and thinking-enabled responses
        content_parts = self._extract_content_from_completion(completion)
        thinking = content_parts.get('thinking')
        text_content = content_parts.get('text') or ''

        # Fallback to string representation if no content extracted
        if not text_content:
            if hasattr(completion, 'content'):
                content = completion.content
                text_content = content.strip() if isinstance(content, str) else str(content)
            else:
                text_content = str(completion)

        # Dispatch thinking step event to chat if present
        if thinking:
            logger.info(f"Model thinking: {thinking[:200]}..." if len(thinking) > 200 else f"Model thinking: {thinking}")

            # Dispatch custom event for thinking step to be displayed in chat
            try:
                model_name = getattr(llm_client, 'model_name', None) or getattr(llm_client, 'model', 'LLM')
                dispatch_custom_event(
                    name="thinking_step",
                    data={
                        "message": thinking,
                        "tool_name": f"LLM ({model_name})",
                        "toolkit": "reasoning",
                    },
                    config=config,
                )
            except Exception as e:
                logger.warning(f"Failed to dispatch thinking event: {e}")

        # Build the AI message with both thinking and text
        # Store thinking in additional_kwargs for potential future use
        ai_message_kwargs = {'content': text_content}
        if thinking:
            ai_message_kwargs['additional_kwargs'] = {'thinking': thinking}
        ai_message = AIMessage(**ai_message_kwargs)

        # Try to extract JSON if output variables are specified (but exclude 'messages' which is handled separately)
        json_output_vars = [var for var in (self.output_variables or []) if var != 'messages']
        if json_output_vars:
            # set response to be the first output variable for non-structured output
            response_data = {json_output_vars[0]: text_content}
            new_messages = messages + [ai_message]
            response_data['messages'] = self._prepare_output_messages(new_messages, middleware_updates)
            return response_data

        # Simple text response (either no output variables or JSON parsing failed)
        new_messages = messages + [ai_message]
        return {"messages": self._prepare_output_messages(new_messages, middleware_updates)}

    @staticmethod
    def _build_invoked_skills_section(invoked_skills: Any) -> str:
        if not invoked_skills or not isinstance(invoked_skills, list):
            return ""

        entries = []
        for skill in invoked_skills:
            if not isinstance(skill, dict):
                continue
            name = skill.get('name')
            instructions = skill.get('instructions')
            if not name or not str(name).strip():
                continue
            if not instructions or not str(instructions).strip():
                continue
            entries.append(SKILLS_SECTION_ENTRY.format(name=name, instructions=instructions))
            if len(entries) >= MAX_SKILLS_PER_INVOCATION:
                break

        if not entries:
            return ""

        return "{header}\n\n{body}".format(
            header=SKILLS_SECTION_HEADER,
            body="\n\n".join(entries),
        )

    @staticmethod
    def _strip_system_messages(messages: list) -> list:
        """Strip SystemMessage objects from a message list before returning to graph state.

        The LLMNode constructs SystemMessage on-the-fly from its input_mapping['system']
        for each invocation. Storing SystemMessages in the graph state would cause them
        to accumulate in checkpoints, leading to "multiple non-consecutive system messages"
        errors on subsequent turns (especially with Anthropic models).

        Args:
            messages: List of messages to process

        Returns:
            Filtered message list without SystemMessage objects
        """
        return [m for m in messages if not isinstance(m, SystemMessage)]

    @staticmethod
    def _prepare_output_messages(messages: list, middleware_updates: list = None) -> list:
        """Prepare messages for output, stripping system messages and prepending middleware updates.

        Args:
            messages: List of messages to process
            middleware_updates: Optional list of RemoveMessage operations from middleware.
                               These are prepended so LangGraph's reducer processes deletions
                               before adding new messages (e.g., for summarization).

        Returns:
            Filtered message list with RemoveMessage ops prepended
        """
        filtered = [m for m in messages if not isinstance(m, SystemMessage)]
        if middleware_updates:
            return list(middleware_updates) + filtered
        return filtered

    def _run(self, *args, **kwargs):
        # Legacy support for old interface
        return self.invoke(kwargs, **kwargs)

    @staticmethod
    def _tool_call_already_completed(tool_call_id: str, messages: list) -> bool:
        """Return True if `messages` already contains a ToolMessage for ``tool_call_id``.

        Used by ``__perform_tool_calling`` to skip tool_calls whose results
        survived a HITL round-trip (see issue #4333). Without this, multi-tool
        sibling cases re-execute non-sensitive tools every time the user
        approves a sensitive sibling.
        """
        if not tool_call_id:
            return False
        from langchain_core.messages import ToolMessage
        for msg in messages:
            if isinstance(msg, ToolMessage) and getattr(msg, 'tool_call_id', None) == tool_call_id:
                return True
        return False

    def _resolve_tool_to_execute(self, tool_name, config):
        """Resolve a tool name to a BaseTool using the sequential loop's lookup chain.

        Order: filtered tools (dynamic selection aware) → available_tools →
        tool_registry. Returns None when the name cannot be resolved. Extracted
        so the parallel fan-out partition and the sequential loop resolve tools
        identically.
        """
        for tool in self.get_filtered_tools(config=config):
            if tool.name == tool_name:
                return tool
        for tool in (self.available_tools or []):
            if tool.name == tool_name:
                logger.info("Resolved tool '%s' via available_tools fallback", tool_name)
                return tool
        if self.tool_registry is not None:
            registry_tool = self.tool_registry.get_tool_by_name(tool_name)
            if registry_tool is not None:
                logger.info("Resolved tool '%s' via tool_registry fallback", tool_name)
                return registry_tool
        return None

    @staticmethod
    def _append_completion_dedup(messages: list, completion: AIMessage) -> list:
        """Append ``completion`` to ``messages`` unless it's already there by identity.

        ``_build_resume_completion`` may return an AIMessage object that is
        already present in the restored ``messages`` list (the multi-tool
        sibling HITL resume case where the original AI sits between two
        ToolMessages). Identity match is sufficient and safe — appending the
        same object twice would corrupt the conversation, while distinct
        AIMessage instances (e.g., a fresh deserialization with the same
        tool_calls) must still be appended.
        """
        for msg in messages:
            if msg is completion:
                logger.info(
                    "[HITL] Skipping duplicate AIMessage append "
                    "(completion already present by identity)."
                )
                return messages
        messages.append(completion)
        return messages

    @staticmethod
    def _build_resume_completion(hitl_ctx: dict, messages: list) -> Optional[AIMessage]:
        """Reuse the original tool-calling AIMessage as the resume completion.

        Anthropic tool-calling turns can carry list-shaped ``content`` with
        provider-specific blocks such as ``thinking``, ``redacted_thinking``,
        and plain ``text`` immediately before ``tool_use``. Replacing that
        original AIMessage with a synthetic ``AIMessage(content='',
        tool_calls=[...])`` strips the original assistant message shape and can
        make resumed runs lose continuity.

        This helper deserializes ``hitl_ctx['original_ai_message']`` (captured at
        interrupt time by the graph-level resume handler) and returns it when:
            * a tool_call on the message matches the approved tool name + args, AND
            * the original AIMessage carries meaningful assistant content that
            would otherwise be lost (structured list content, or a non-empty
            string), AND
            * the same message is not already present in ``messages`` (which would
            duplicate it for the multi-tool sibling case where ``_trim`` keeps
            the AI in the restored history).

        It also rewrites ``hitl_ctx['tool_call_id']`` to the original tool_call id
        so the downstream ``ToolMessage`` uses a matching id.

        Returns ``None`` to indicate that the caller should fall back to the
        empty synthetic AIMessage (current behavior).
        """
        if not isinstance(hitl_ctx, dict):
            return None
        original_dict = hitl_ctx.get('original_ai_message')
        if not isinstance(original_dict, dict):
            return None

        try:
            from langchain_core.messages.utils import messages_from_dict
            restored = messages_from_dict([original_dict])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[HITL] Failed to deserialize original AIMessage for resume: %s", exc,
            )
            return None
        if not restored or not isinstance(restored[0], AIMessage):
            return None

        original_ai: AIMessage = restored[0]
        # Always reuse the original AIMessage when a tool_call matches.  Even
        # for non-thinking models with empty ``content``, the original carries
        # the canonical tool_call ids that downstream sibling-skip logic
        # (``_tool_call_already_completed``) relies on. Replacing it with a
        # synthetic AIMessage that uses fresh UUIDs makes those ids drift, so
        # if the LLM re-emits the original batch on the next iteration, the
        # already-completed siblings cannot be matched and re-execute.
        # (Issue #4333.)

        target_tool = hitl_ctx.get('tool_name', '')
        target_args = hitl_ctx.get('tool_args', {}) or {}
        matching_tc = None
        for tc in (original_ai.tool_calls or []):
            tc_name = tc.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '')
            tc_args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
            if tc_name == target_tool and args_match_normalized(tc_args, target_args):
                matching_tc = tc
                break
        if matching_tc is None:
            # Fallback: match by tool name only when there's exactly one
            # tool_call with that name.  After JSON round-trip through the
            # checkpoint, nested args can diverge (int vs float, key order).
            name_matches = [
                tc for tc in (original_ai.tool_calls or [])
                if (tc.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '')) == target_tool
            ]
            if len(name_matches) == 1:
                matching_tc = name_matches[0]
                logger.info(
                    "[HITL] Matched original AIMessage tool_call by name-only "
                    "fallback (tool=%s) — args comparison failed after JSON round-trip",
                    target_tool,
                )
            else:
                return None

        original_tc_id = (
            matching_tc.get('id', '') if isinstance(matching_tc, dict)
            else getattr(matching_tc, 'id', '')
        )

        # If an AIMessage carrying the same tool_call id is already present
        # in the restored history (multi-tool sibling case), reuse it as the
        # completion. ``_append_completion_dedup`` in ``__perform_tool_calling``
        # will skip the duplicate append, and ``_tool_call_already_completed``
        # will skip re-executing siblings whose ToolMessage is already there.
        # See issue #4333.
        if original_tc_id:
            for m in messages:
                if isinstance(m, AIMessage):
                    for tc in (m.tool_calls or []):
                        existing_id = (
                            tc.get('id', '') if isinstance(tc, dict)
                            else getattr(tc, 'id', '')
                        )
                        if existing_id and existing_id == original_tc_id:
                            hitl_ctx['tool_call_id'] = original_tc_id
                            logger.info(
                                "[HITL] Original AIMessage already present in "
                                "restored history (tool_call_id=%s); reusing it "
                                "as the resume completion.",
                                original_tc_id,
                            )
                            return m

        if original_tc_id:
            hitl_ctx['tool_call_id'] = original_tc_id

        content = original_ai.content
        if isinstance(content, list):
            content_kinds = []
            for block in content:
                if isinstance(block, dict):
                    content_kinds.append(str(block.get('type', '<missing>')))
                elif hasattr(block, 'type'):
                    content_kinds.append(str(getattr(block, 'type', '<missing>')))
                else:
                    content_kinds.append(type(block).__name__)
            content_shape = '[' + ','.join(content_kinds) + ']'
        else:
            content_shape = type(content).__name__

        # Filter tool_calls to prevent orphaned tool_call_ids. The API requires
        # that ToolMessages FOLLOW their AIMessage. Tool_calls are kept if:
        # 1. They are at or after the resumed tool's position (will execute and
        #    get ToolMessages AFTER this AIMessage), OR
        # 2. They are before the resumed position AND already have ToolMessages
        #    in the conversation history (already completed).
        #
        # This prevents the "tool_call_ids did not have response messages" error
        # when the original AIMessage had multiple tool_calls but not all executed
        # before the interrupt.

        # Find the index of the resumed tool_call
        resumed_idx = None
        for i, tc in enumerate(original_ai.tool_calls or []):
            tc_id = tc.get('id', '') if isinstance(tc, dict) else getattr(tc, 'id', '')
            if tc_id == original_tc_id:
                resumed_idx = i
                break

        if resumed_idx is not None:
            # Collect tool_call_ids that already have ToolMessages
            tool_result_ids = {
                getattr(m, 'tool_call_id', None)
                for m in messages
                if isinstance(m, ToolMessage) and getattr(m, 'tool_call_id', None)
            }

            filtered_tool_calls = []
            for i, tc in enumerate(original_ai.tool_calls or []):
                tc_id = tc.get('id', '') if isinstance(tc, dict) else getattr(tc, 'id', '')
                if i >= resumed_idx:
                    # This tool_call will execute (at or after resumed position), keep it
                    filtered_tool_calls.append(tc)
                elif tc_id in tool_result_ids:
                    # This tool_call already has a ToolMessage, keep it
                    filtered_tool_calls.append(tc)
                # else: tool_call before resumed position without ToolMessage - filter out

            if len(filtered_tool_calls) != len(original_ai.tool_calls or []):
                logger.info(
                    "[HITL] Filtered original AIMessage tool_calls from %d to %d "
                    "(keeping completed siblings + resumed tool and later siblings)",
                    len(original_ai.tool_calls or []),
                    len(filtered_tool_calls),
                )
                try:
                    original_ai = original_ai.model_copy(update={"tool_calls": filtered_tool_calls})
                except Exception:
                    original_ai = AIMessage(content=original_ai.content, tool_calls=filtered_tool_calls)

        logger.info(
            "[HITL] Reusing original AIMessage as resume completion "
            "(tool=%s, tool_call_id=%s, content_shape=%s) — preserves the "
            "original assistant message across resume",
            target_tool,
            original_tc_id or '<missing>',
            content_shape,
        )
        return original_ai

    @staticmethod
    def _extract_original_content_for_resume(hitl_ctx: dict) -> Any:
        """Extract content from original_ai_message for fallback synthetic AIMessage.

        When ``_build_resume_completion`` returns None (e.g., args mismatch after
        JSON round-trip), we still want to preserve the original AIMessage's
        content in the synthetic completion.  For Anthropic thinking models, the
        content is a list carrying ``thinking``/``redacted_thinking`` blocks that
        MUST be present for the follow-up LLM call to succeed.

        Without this, the synthetic AIMessage gets ``content=''`` which causes:
        - Anthropic API to reject the request with a thinking-block format error, OR
        - The LLM to lose context and re-plan from scratch (re-invoking all tools)

        Returns:
            The original content (list for thinking models, '' otherwise).
        """
        original_dict = hitl_ctx.get('original_ai_message') if isinstance(hitl_ctx, dict) else None
        if not isinstance(original_dict, dict):
            return ''
        data = original_dict.get('data')
        if isinstance(data, dict):
            content = data.get('content', '')
        else:
            content = original_dict.get('content', '')
        # Only preserve list content (which indicates structured blocks like
        # thinking/redacted_thinking/text). Simple string content doesn't need
        # special handling since content='' works fine for non-thinking models.
        if isinstance(content, list) and content:
            logger.info(
                "[HITL] Preserving original AIMessage content (%d blocks) in "
                "synthetic completion for thinking-model compatibility",
                len(content),
            )
            return content
        return ''

    @staticmethod
    def _extract_content_from_completion(completion) -> dict:
        """Extract thinking and text content from LLM completion.
        
        Handles Anthropic's extended thinking format where content is a list
        of blocks with types: 'thinking' and 'text'.
        
        Args:
            completion: LLM completion object with content attribute
            
        Returns:
            dict with 'thinking' and 'text' keys
        """
        result = {'thinking': None, 'text': None}
        
        if not hasattr(completion, 'content'):
            return result
            
        content = completion.content
        
        # Handle list of content blocks (Anthropic extended thinking format)
        if isinstance(content, list):
            thinking_blocks = []
            text_blocks = []
            
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get('type', '')
                    if block_type == 'thinking':
                        thinking_blocks.append(block.get('thinking', ''))
                    elif block_type == 'text':
                        text_blocks.append(block.get('text', ''))
                elif hasattr(block, 'type'):
                    # Handle object format
                    if block.type == 'thinking':
                        thinking_blocks.append(getattr(block, 'thinking', ''))
                    elif block.type == 'text':
                        text_blocks.append(getattr(block, 'text', ''))
                elif isinstance(block, str) and block:
                    # Some providers/thinking modes (e.g. Anthropic adaptive
                    # thinking) emit the final answer as a bare string list
                    # item instead of a {'type': 'text', ...} dict. Without
                    # this branch that text is silently dropped.
                    text_blocks.append(block)

            if thinking_blocks:
                result['thinking'] = '\n\n'.join(thinking_blocks)
            if text_blocks:
                result['text'] = '\n\n'.join(text_blocks)
        
        # Handle simple string content
        elif isinstance(content, str):
            result['text'] = content
        
        return result

    @staticmethod
    def _completion_stop_reason(completion: Any) -> str:
        """Return a normalized provider stop reason from an AI message."""
        metadata = getattr(completion, 'response_metadata', None) or {}
        if not isinstance(metadata, dict):
            return ''

        reason = metadata.get('stop_reason') or metadata.get('finish_reason')
        if not reason and metadata.get('status') == 'incomplete':
            incomplete_details = metadata.get('incomplete_details') or {}
            if isinstance(incomplete_details, dict):
                reason = incomplete_details.get('reason')

        return str(reason or '').lower()

    @classmethod
    def _completion_finished_by_length(cls, completion: Any) -> bool:
        """Normalize provider-specific output-limit metadata on an AI message."""

        return cls._completion_stop_reason(completion) in {
            'length',
            'max_tokens',
            'max_output_tokens',
            'stop_reason_max_tokens',
        }

    @staticmethod
    def _is_nested_execution(config: Optional[RunnableConfig]) -> bool:
        metadata = config.get('metadata', {}) if isinstance(config, dict) else {}
        return isinstance(metadata, dict) and bool(
            metadata.get('parent_agent_path')
            or metadata.get('parent_agent_name')
            or metadata.get('langgraph_node')
        )

    @staticmethod
    def _merge_continuation_text(existing: str, incoming: str) -> str:
        """Join continuation text, including output-limit cuts inside a token."""
        continuation = incoming
        max_overlap = min(len(existing), len(continuation), 150)
        for overlap in range(max_overlap, 3, -1):
            suffix = existing[-overlap:]
            if suffix != continuation[:overlap] or not any(char.isalnum() for char in suffix):
                continue
            starts_at_boundary = (
                overlap == len(existing)
                or not (existing[-overlap - 1].isalnum() and suffix[0].isalnum())
            )
            ends_at_boundary = (
                overlap == len(continuation)
                or not (suffix[-1].isalnum() and continuation[overlap].isalnum())
            )
            if starts_at_boundary and ends_at_boundary:
                continuation = continuation[overlap:]
                break

        if not existing:
            return continuation
        if not continuation:
            return existing
        stripped = continuation.lstrip(' \t')
        first_word = stripped.split(' ', 1)[0]
        numbered_item = (
            len(first_word) > 1
            and first_word[-1] in '.)'
            and first_word[:-1].isdigit()
        )
        block_start = (
            numbered_item
            or stripped.startswith(('- ', '* ', '+ ', '#', '```'))
        )
        separator = (
            '\n'
            if block_start
            and not existing[-1].isspace()
            and not continuation[0].isspace()
            else ''
        )
        return f'{existing}{separator}{continuation}'

    @staticmethod
    def _continuation_anchor(existing: str) -> str:
        """Return a bounded, visible suffix used to prove the merge seam."""
        return existing.rstrip()[-NESTED_OUTPUT_CONTINUATION_ANCHOR_MAX_CHARS:]

    @staticmethod
    def _requested_output_word_target(messages: List) -> Optional[int]:
        """Return a confidently stated output word target from the latest request.

        A hyphenated count is accepted only after an output-creation verb. A
        plain ``N words`` count additionally needs either that verb or a length
        qualifier. This deliberately ignores source descriptions such as
        ``review this 1200-word story``.
        """
        user_text = next(
            (
                message.content
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
                and isinstance(message.content, str)
            ),
            '',
        )
        if not user_text:
            return None

        creation_verb = r'(?:write|draft|compose|create|generate|produce)'
        count = r'(?P<count>[1-9]\d{1,5}(?:,\d{3})*)'
        qualified_count = re.search(
            rf'(?is)\b(?:in\s+)?(?:about|approximately|around|roughly|exactly|'
            rf'at\s+least|no\s+more\s+than|up\s+to|under|within)\s+{count}'
            rf'\s+words?\b',
            user_text,
        )
        match = qualified_count or re.search(
            rf'(?is)\b{creation_verb}\b[^.\n]{{0,100}}?\b{count}'
            rf'\s*(?:-\s*|\s+)words?\b',
            user_text,
        )
        if not match:
            return None

        return int(match.group('count').replace(',', ''))

    @staticmethod
    def _ends_at_sentence_boundary(text: str) -> bool:
        """Return whether an explicit prose budget ended at a safe stop point."""
        stripped = text.rstrip().rstrip('"\'”’)]}')
        return bool(stripped) and stripped[-1] in '.?!'

    def _continuation_max_tokens(
        self,
        *,
        remaining_words: int,
        anchor: str,
        closure_only: bool,
    ) -> Optional[int]:
        """Reduce, but never increase, a configured per-call output limit."""
        if closure_only:
            desired_limit = NESTED_OUTPUT_CLOSURE_MAX_TOKENS
        else:
            # English prose is normally about 1.3 tokens per word. The 3/2
            # multiplier plus the echoed seam leaves modest tokenizer variance.
            desired_limit = max(1, (remaining_words * 3 + 1) // 2)
            desired_limit += max(8, len(anchor) // 4)

        configured_limit = None
        for field_name in ('max_tokens', 'max_completion_tokens'):
            value = getattr(self.client, field_name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                configured_limit = value
                break

        if configured_limit is None:
            # Default mode intentionally has no Elitea-defined provider cap.
            # Keep that contract and rely on the semantic budget prompt.
            return None
        return min(configured_limit, desired_limit)

    @classmethod
    def _build_output_continuation_prompt(
        cls,
        accumulated_text: str,
        *,
        word_target: Optional[int] = None,
        closure_only: bool = False,
        seam_retry: bool = False,
    ) -> tuple[str, str]:
        """Build a compact continuation contract for a fresh model request."""
        anchor = cls._continuation_anchor(accumulated_text)
        word_count = len(accumulated_text.split())
        char_count = len(accumulated_text)
        scope_contract = ''
        if word_target is not None:
            remaining_words = max(0, word_target - word_count)
            if closure_only:
                scope_contract = (
                    f"The original {word_target}-word budget is already satisfied. "
                    "CLOSURE-ONLY MODE: add only what is required to finish the "
                    "cut-off sentence and close the answer immediately. Do not "
                    "start another paragraph, scene, section, example, or topic.\n\n"
                )
            else:
                scope_contract = (
                    f"The original request sets one total target of {word_target} "
                    f"words. The accepted output has {word_count} words, so "
                    f"approximately {remaining_words} words remain for the entire "
                    "answer, across this and any later continuation. Complete the "
                    "answer within that remaining budget; it is not a fresh budget "
                    "for this call.\n\n"
                )
        seam_contract = ''
        if seam_retry:
            seam_contract = (
                "BOUNDARY REPAIR: the prior continuation was rejected because it "
                "replaced or skipped accepted boundary text. Do not revise, remove, "
                "or reinterpret the final word. First copy the exact anchor with "
                "identical case and punctuation, then append new characters.\n\n"
            )
        prompt = (
            "The previous assistant output reached its output-token limit. "
            "The original task and conversation above remain authoritative. "
            "Silently re-read them together with the accepted assistant output, "
            "then complete only the missing portion of that same answer. Preserve "
            "its existing semantic scope and structure; do not restart, repeat "
            "covered material, introduce optional new topics, or make the answer "
            "longer merely because this is another request. Stop as soon as the "
            "original task is coherently satisfied.\n\n"
            f"{scope_contract}"
            f"{seam_contract}"
            f"Accepted output progress: {word_count} words, {char_count} characters. "
            "These counts are progress information, not a new requested target.\n\n"
            "Your response must begin character-for-character with the exact anchor "
            "below, immediately followed by the missing continuation. Return only "
            "that anchor plus the continuation; do not explain this protocol.\n"
            f"Exact anchor: {json.dumps(anchor, ensure_ascii=False)}"
        )
        return prompt, anchor

    @staticmethod
    def _merge_anchored_continuation(
        existing: str,
        incoming: str,
        anchor: str,
    ) -> Optional[str]:
        """Merge only when ``incoming`` proves an exact accepted-output seam."""
        if not anchor:
            return None

        overlap = len(anchor) if incoming.startswith(anchor) else 0
        if not overlap:
            existing_without_trailing_space = existing.rstrip()
            max_overlap = min(
                len(existing_without_trailing_space),
                len(incoming),
                NESTED_OUTPUT_CONTINUATION_ANCHOR_MAX_CHARS,
            )
            for candidate in range(
                max_overlap,
                NESTED_OUTPUT_CONTINUATION_MIN_OVERLAP_CHARS - 1,
                -1,
            ):
                suffix = existing_without_trailing_space[-candidate:]
                if suffix != incoming[:candidate]:
                    continue
                starts_at_boundary = (
                    candidate == len(existing_without_trailing_space)
                    or not (
                        existing_without_trailing_space[-candidate - 1].isalnum()
                        and suffix[0].isalnum()
                    )
                )
                ends_at_boundary = (
                    candidate == len(incoming)
                    or not (suffix[-1].isalnum() and incoming[candidate].isalnum())
                )
                if starts_at_boundary and ends_at_boundary:
                    overlap = candidate
                    break

            if not overlap:
                final_line = existing_without_trailing_space.rsplit('\n', 1)[-1]
                if (
                    len(final_line) >= 4
                    and sum(character.isalnum() for character in final_line) >= 4
                    and incoming.startswith(final_line)
                ):
                    overlap = len(final_line)

        if not overlap:
            return None

        continuation = incoming[overlap:]
        if not continuation:
            return existing

        base = existing.rstrip()
        removed_whitespace = existing[len(base):]
        if removed_whitespace and not continuation[0].isspace():
            continuation = f'{removed_whitespace}{continuation}'
        return f'{base}{continuation}'

    def _continue_nested_output(
        self,
        *,
        messages: List,
        completion: Any,
        config: Optional[RunnableConfig],
    ) -> Any:
        """Finish truncated graph-node output before returning it to the graph."""
        if (
            not self._is_nested_execution(config)
            or not self._completion_finished_by_length(completion)
        ):
            return completion

        current_completion = completion
        current_text = self._extract_content_from_completion(completion).get('text') or ''
        accumulated_text = current_text
        word_target = self._requested_output_word_target(messages)
        invalid_seam_retries = 0
        retrying_invalid_seam = False

        if (
            word_target is not None
            and len(accumulated_text.split()) >= word_target
            and self._ends_at_sentence_boundary(accumulated_text)
        ):
            logger.info(
                "[NESTED_CONTINUE] Explicit word target reached at a safe boundary "
                "without another request (target=%d, actual=%d)",
                word_target,
                len(accumulated_text.split()),
            )
            return completion

        for continuation_round in range(1, NESTED_OUTPUT_CONTINUATION_LIMIT + 1):
            accumulated_word_count = len(accumulated_text.split())
            closure_only = (
                word_target is not None and accumulated_word_count >= word_target
            )
            continuation_max_tokens = None
            if accumulated_text.strip():
                prompt, anchor = self._build_output_continuation_prompt(
                    accumulated_text,
                    word_target=word_target,
                    closure_only=closure_only,
                    seam_retry=retrying_invalid_seam,
                )
                if word_target is not None:
                    continuation_max_tokens = self._continuation_max_tokens(
                        remaining_words=max(0, word_target - accumulated_word_count),
                        anchor=anchor,
                        closure_only=closure_only,
                    )
                continuation_messages = [
                    *messages,
                    AIMessage(content=accumulated_text),
                    HumanMessage(content=prompt),
                ]
            else:
                anchor = ''
                continuation_messages = [
                    *messages,
                    HumanMessage(content=NESTED_REASONING_ONLY_CONTINUATION_PROMPT),
                ]
            logger.info(
                "[NESTED_CONTINUE] Completing truncated leaf output (round=%d/%d)",
                continuation_round,
                NESTED_OUTPUT_CONTINUATION_LIMIT,
            )
            try:
                invoke_kwargs = {'config': config}
                if continuation_max_tokens is not None:
                    invoke_kwargs['max_tokens'] = continuation_max_tokens
                current_completion = self.client.invoke(
                    continuation_messages,
                    **invoke_kwargs,
                )
            except (GraphBubbleUp, McpAuthorizationRequired, OutputContinuationExhausted):
                raise
            except Exception as exc:
                budget_error = budget_exceeded_from(exc)
                if budget_error is not None:
                    raise budget_error from exc
                raise OutputContinuationExhausted(
                    attempts=continuation_round,
                    partial_output=accumulated_text,
                    stop_reason=self._completion_stop_reason(current_completion),
                    failure_reason='provider_error',
                ) from exc
            current_text = (
                self._extract_content_from_completion(current_completion).get('text') or ''
            )
            if current_text:
                merged_text = (
                    self._merge_anchored_continuation(
                        accumulated_text,
                        current_text,
                        anchor,
                    )
                    if anchor
                    else current_text
                )
                if merged_text is None:
                    if invalid_seam_retries < NESTED_OUTPUT_INVALID_SEAM_RETRY_LIMIT:
                        invalid_seam_retries += 1
                        retrying_invalid_seam = True
                        logger.warning(
                            "[NESTED_CONTINUE] Rejected an unverified output seam; "
                            "retrying once with the strict boundary contract "
                            "(round=%d)",
                            continuation_round,
                        )
                        continue
                    raise OutputContinuationExhausted(
                        attempts=continuation_round,
                        partial_output=accumulated_text,
                        stop_reason=self._completion_stop_reason(current_completion),
                        failure_reason='invalid_continuation',
                    )
                retrying_invalid_seam = False
                if merged_text == accumulated_text:
                    raise OutputContinuationExhausted(
                        attempts=continuation_round,
                        partial_output=accumulated_text,
                        stop_reason=self._completion_stop_reason(current_completion),
                        failure_reason='no_progress',
                    )
                accumulated_text = merged_text
            else:
                raise OutputContinuationExhausted(
                    attempts=continuation_round,
                    partial_output=accumulated_text,
                    stop_reason=self._completion_stop_reason(current_completion),
                    failure_reason='no_progress',
                )

            if (
                word_target is not None
                and len(accumulated_text.split()) >= word_target
                and self._ends_at_sentence_boundary(accumulated_text)
            ):
                logger.info(
                    "[NESTED_CONTINUE] Explicit word target reached at a safe boundary "
                    "(round=%d, target=%d, actual=%d)",
                    continuation_round,
                    word_target,
                    len(accumulated_text.split()),
                )
                break

            if closure_only and self._completion_finished_by_length(current_completion):
                raise OutputContinuationExhausted(
                    attempts=continuation_round,
                    partial_output=accumulated_text,
                    stop_reason=self._completion_stop_reason(current_completion),
                    failure_reason='attempt_limit',
                )

            if not self._completion_finished_by_length(current_completion):
                break
        else:
            logger.warning(
                "[NESTED_CONTINUE] Leaf output remained truncated after %d continuation rounds",
                NESTED_OUTPUT_CONTINUATION_LIMIT,
            )
            raise OutputContinuationExhausted(
                attempts=NESTED_OUTPUT_CONTINUATION_LIMIT,
                partial_output=accumulated_text,
                stop_reason=self._completion_stop_reason(current_completion),
                failure_reason='attempt_limit',
            )

        if not accumulated_text:
            return current_completion
        if hasattr(current_completion, 'model_copy'):
            return current_completion.model_copy(update={
                'content': accumulated_text,
                'tool_calls': [],
            })
        return AIMessage(
            content=accumulated_text,
            response_metadata=dict(
                getattr(current_completion, 'response_metadata', None) or {}
            ),
        )
    
    def _run_async_in_sync_context(self, coro):
        """Run async coroutine from sync context.

        For MCP tools with persistent sessions, we reuse the same event loop
        that was used to create the MCP client and sessions (set by CLI).

        When called from within a running event loop (e.g., nested LLM nodes),
        we need to handle this carefully to avoid "event loop already running" errors.

        This method handles three scenarios:
        1. Called from async context (event loop running) - creates new thread with new loop
        2. Called from sync context with persistent loop - reuses persistent loop
        3. Called from sync context without loop - creates new persistent loop
        """
        import contextvars
        import threading

        # Check if there's a running loop
        try:
            running_loop = asyncio.get_running_loop()
            loop_is_running = True
            logger.debug(f"Detected running event loop (id: {id(running_loop)}), executing tool calls in separate thread")
        except RuntimeError:
            loop_is_running = False

        # Scenario 1: Loop is currently running - MUST use thread
        if loop_is_running:
            result_container = []
            exception_container = []

            # Capture the current context (including LangGraph's
            # var_child_runnable_config) so interrupt() works inside the
            # spawned thread. Without this, ContextVars are not inherited
            # by plain threading.Thread targets.
            parent_ctx = contextvars.copy_context()

            # Try to capture Streamlit context from current thread for propagation
            streamlit_ctx = None
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
                streamlit_ctx = get_script_run_ctx()
                if streamlit_ctx:
                    logger.debug("Captured Streamlit context for propagation to worker thread")
            except (ImportError, Exception) as e:
                logger.debug(f"Streamlit context not available or failed to capture: {e}")

            def run_in_thread():
                """Run coroutine in a new thread with its own event loop,
                inheriting the parent's ContextVars."""
                def _inner():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        result = new_loop.run_until_complete(coro)
                        result_container.append(result)
                    except GraphBubbleUp as gb:
                        exception_container.append(gb)
                    except Exception as e:
                        logger.debug(f"Exception in async thread: {e}")
                        exception_container.append(e)
                    finally:
                        new_loop.close()
                        asyncio.set_event_loop(None)
                parent_ctx.run(_inner)

            thread = threading.Thread(target=run_in_thread, daemon=False)

            # Propagate Streamlit context to the worker thread if available
            if streamlit_ctx is not None:
                try:
                    add_script_run_ctx(thread, streamlit_ctx)
                    logger.debug("Successfully propagated Streamlit context to worker thread")
                except Exception as e:
                    logger.warning(f"Failed to propagate Streamlit context to worker thread: {e}")

            thread.start()
            thread.join(timeout=self.tool_execution_timeout)  # 15 minute timeout for safety

            if thread.is_alive():
                logger.error("Async operation timed out after 5 minutes")
                raise TimeoutError("Async operation in thread timed out")

            # Re-raise exception if one occurred
            if exception_container:
                raise exception_container[0]

            return result_container[0] if result_container else None

        # Scenario 2 & 3: No loop running - use or create persistent loop
        else:
            # Get or create persistent loop
            if not hasattr(self.__class__, '_persistent_loop') or \
               self.__class__._persistent_loop is None or \
               self.__class__._persistent_loop.is_closed():
                self.__class__._persistent_loop = asyncio.new_event_loop()
                logger.debug("Created persistent event loop for async tools")

            loop = self.__class__._persistent_loop

            # Double-check the loop is not running (safety check)
            if loop.is_running():
                logger.debug("Persistent loop is unexpectedly running, using thread execution")

                result_container = []
                exception_container = []
                parent_ctx = contextvars.copy_context()

                # Try to capture Streamlit context from current thread for propagation
                streamlit_ctx = None
                try:
                    from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
                    streamlit_ctx = get_script_run_ctx()
                    if streamlit_ctx:
                        logger.debug("Captured Streamlit context for propagation to worker thread")
                except (ImportError, Exception) as e:
                    logger.debug(f"Streamlit context not available or failed to capture: {e}")

                def run_in_thread():
                    """Run coroutine in a new thread with its own event loop,
                    inheriting the parent's ContextVars."""
                    def _inner():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            result = new_loop.run_until_complete(coro)
                            result_container.append(result)
                        except GraphBubbleUp as gb:
                            exception_container.append(gb)
                        except Exception as ex:
                            logger.debug(f"Exception in async thread: {ex}")
                            exception_container.append(ex)
                        finally:
                            new_loop.close()
                            asyncio.set_event_loop(None)
                    parent_ctx.run(_inner)

                thread = threading.Thread(target=run_in_thread, daemon=False)

                # Propagate Streamlit context to the worker thread if available
                if streamlit_ctx is not None:
                    try:
                        add_script_run_ctx(thread, streamlit_ctx)
                        logger.debug("Successfully propagated Streamlit context to worker thread")
                    except Exception as e:
                        logger.warning(f"Failed to propagate Streamlit context to worker thread: {e}")

                thread.start()
                thread.join(timeout=self.tool_execution_timeout)

                if thread.is_alive():
                    logger.error("Async operation timed out after 15 minutes")
                    raise TimeoutError("Async operation in thread timed out")

                if exception_container:
                    raise exception_container[0]

                return result_container[0] if result_container else None
            else:
                # Loop exists but not running - safe to use run_until_complete
                logger.debug(f"Using persistent loop (id: {id(loop)}) with run_until_complete")
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)

    async def _arun(self, *args, **kwargs):
        # Legacy async support
        return self.invoke(kwargs, **kwargs)

    def _collect_parallel_application_specs(
        self, tool_calls, messages, config,
        hitl_decisions=None,
    ):
        """Return per-call specs when this turn is a pure multi-Application batch.

        A turn qualifies for parallel fan-out (issue #4993) only when, after
        skipping tool_calls that already completed across a HITL round-trip, it
        contains 2+ Application (sub-agent) calls and NO regular tool calls.
        Mixed batches keep the sequential path (returns None). Each returned
        spec is ``(tool_name, tool_args, tool_call_id, application_tool)``.

        Resume exception: when one child completed and another paused, the
        resume turn has only the single paused child left (the completed
        sibling's ToolMessage was restored and is skipped). That lone child was
        checkpointed under the parallel-suffixed thread_id, so it MUST stay on
        the parallel path to resume from its own checkpoint. We detect this by
        matching a remaining call against the resume ``hitl_decisions`` and allow
        a 1-spec parallel batch in that case.
        """
        from .application import Application
        if not tool_calls or len(tool_calls) < 2:
            return None
        decision_ids = {
            d.get(HITL_VIA_CALL_ID_KEY) or d.get(HITL_TOOL_CALL_ID_KEY)
            for d in (hitl_decisions or [])
            if isinstance(d, dict)
            and (d.get(HITL_VIA_CALL_ID_KEY) or d.get(HITL_TOOL_CALL_ID_KEY))
        }
        specs = []
        for tool_call in tool_calls:
            tool_name = tool_call.get('name', '') if isinstance(tool_call, dict) else getattr(tool_call, 'name', '')
            tool_args = tool_call.get('args', {}) if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
            tool_call_id = tool_call.get('id', '') if isinstance(tool_call, dict) else getattr(tool_call, 'id', '')
            # Already-completed siblings (results restored across a HITL resume)
            # neither count toward the batch nor re-execute.
            if tool_call_id and self._tool_call_already_completed(tool_call_id, messages):
                continue
            tool = self._resolve_tool_to_execute(tool_name, config)
            if not isinstance(tool, Application):
                return None  # a non-Application call → sequential path
            specs.append((tool_name, tool_args, tool_call_id, tool))
        if not specs:
            return None
        # Parallel resume of the remaining paused child(ren): keep on the
        # parallel path even with a single spec so the suffixed thread_id matches.
        if any(s[2] in decision_ids for s in specs):
            return specs
        return specs if len(specs) >= 2 else None

    def _build_parallel_dispatch_specs(self, app_specs, config):
        """Turn gather specs into durable-dispatch child specs (Track 2, #4993).

        Used when a ``child_dispatcher`` seam is present: instead of running the
        sub-agents in-process via ``asyncio.gather``, the parent PARKS and hands
        these specs to pylon_main, which launches each child as an independent
        durable ``indexer_agent`` task. Each spec is a plain JSON-serialisable
        dict (it must survive the checkpoint channel + an RPC round-trip), so it
        carries NO live tool object — only the identity pylon_main needs to spawn
        the child and the SDK needs to read its checkpoint back on reconcile.

        The durable ``child_thread_id`` includes the persisted dispatch epoch so
        a provider that reuses tool-call ids on a later turn cannot reopen an old
        child checkpoint.
        """
        configurable = config.get('configurable', {}) if isinstance(config, dict) else {}
        parent_thread_id = configurable.get('thread_id')
        metadata = config.get('metadata', {}) if isinstance(config, dict) else {}
        call_ids = [str(spec[2] or '') for spec in app_specs]
        if any(not call_id for call_id in call_ids) or len(set(call_ids)) != len(call_ids):
            logger.warning(
                "[PARALLEL] durable dispatch requires unique non-empty tool-call ids; "
                "falling back to in-process gather",
            )
            return None
        # One random generation per fan-out, persisted in parallel_tasks. It is
        # stable for retries of this park but distinct across fresh turns even if
        # a provider reuses the same tool-call ids on the same parent thread.
        dispatch_epoch = f"dispatch_{uuid4().hex}"
        inherited_path = metadata.get('parent_agent_path')
        inherited_path = list(inherited_path) if isinstance(inherited_path, list) else []
        specs = {}
        for index, (tool_name, tool_args, tool_call_id, tool) in enumerate(app_specs):
            app_name = getattr(tool, 'name', None) or tool_name
            child_thread_id = (
                f"{parent_thread_id}:{dispatch_epoch}:{app_name}:{tool_call_id}"
                if parent_thread_id else None
            )
            # Display label for the UI card, mirroring the gather aggregate's
            # metadata precedence (original_name → display_name → name).
            meta = getattr(tool, 'metadata', None) or {}
            display_name = (
                meta.get('original_name')
                or meta.get('display_name')
                or app_name
            )
            # Child identity for pylon_main to launch a standalone indexer_agent
            # task. ``args_runnable`` is how the in-process path recreates the
            # child (toolkits/application.py:187); the id/version + already-fetched
            # version_details make the spec self-sufficient so pylon_main need not
            # re-resolve the sub-agent. version_details is a plain dict (JSON-safe);
            # the live ``llm``/``memory`` objects in args_runnable are intentionally
            # NOT carried — the child re-resolves those from the parent's payload.
            runnable = getattr(tool, 'args_runnable', None) or {}
            specs[tool_call_id] = self._jsonsafe_spec({
                'tool_call_id': tool_call_id,
                'parent_agent_call_id': tool_call_id,
                'parent_agent_path': inherited_path,
                'dispatch_epoch': dispatch_epoch,
                'dispatch_id': f'{dispatch_epoch}:{tool_call_id}',
                'name': app_name,
                'display_name': display_name,
                'input': tool_args,
                'child_thread_id': child_thread_id,
                'index': index,
                'sibling_ordinal': index + 1,
                'application_id': runnable.get('application_id'),
                'application_version_id': runnable.get('application_version_id'),
                'version_details': runnable.get('version_details'),
                'variable_defaults': getattr(tool, 'variable_defaults', None) or {},
            })
        return specs

    @staticmethod
    def _jsonsafe_spec(spec: dict) -> dict:
        """Deep-coerce a dispatch spec to plain JSON, dropping non-serialisable leaves.

        The spec is written to the ``parallel_tasks`` checkpoint channel and then
        RPC'd to pylon_main, so it MUST be msgpack/JSON-safe. ``input`` (LLM
        tool_args, whose schema allows ``chat_history: list[BaseMessage]``) and
        the nested ``version_details`` dict can carry live objects (BaseMessage,
        an ``EliteAClient`` reference). A non-serialisable leaf becomes ``None``;
        dict/list structure and JSON scalars — everything reconcile reads back
        (``version_details.llm_settings``/``meta``/``variables``) — are preserved.
        """
        def _coerce(value):
            if isinstance(value, dict):
                return {k: _coerce(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_coerce(v) for v in value]
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return None
        return _coerce(spec)

    @staticmethod
    def _parallel_interrupt_id(
        container_call_id: str, leaf_call_id: str, _ordinal: int | None,
    ) -> str:
        """Return a stable public id unique within a flattened HITL aggregate.

        Leaf tool-call ids are only graph-local and can legitimately repeat in
        sibling sub-orchestrators. Scope them by the immediate container call,
        then expose only the opaque UUID-derived value to clients.
        """
        # The immediate container scopes graph-local leaf identities. Do not
        # include the pending-list ordinal: after one sibling resolves, the
        # surviving card is re-emitted at a different position and must retain
        # the public id already shown to the client.
        route = f"{container_call_id}\x1f{leaf_call_id}"
        return f"hitl_{uuid5(NAMESPACE_URL, route).hex}"

    def _build_mcp_auth_interrupt(
        self,
        exc: McpAuthorizationRequired,
        tool_to_execute: BaseTool,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
        config: Optional[RunnableConfig],
    ) -> dict:
        """Build the checkpoint-safe delegated-authorization interrupt payload."""
        from langchain_core.messages import message_to_dict

        auth_metadata = exc.to_dict()
        identity = self._get_tool_identity(tool_to_execute)
        resolved_tool_name = (
            auth_metadata.get('tool_name')
            or identity.get('tool_name')
            or tool_name
        )
        toolkit_name = (
            auth_metadata.get('toolkit_name')
            or identity.get('toolkit_name')
            or ''
        )
        toolkit_type = (
            auth_metadata.get('toolkit_type')
            or identity.get('toolkit_type')
            or ''
        )
        configurable = (
            config.get('configurable', {})
            if isinstance(config, dict)
            else {}
        )

        serialized_pending = []
        for message in _PENDING_TOOL_MESSAGES.get([]):
            try:
                serialized_pending.append(message_to_dict(message))
            except Exception:  # pragma: no cover - defensive serialization seam
                continue

        payload = {
            'type': 'hitl',
            'interrupt_id': f'mcp_auth_{uuid4().hex}',
            'guardrail_type': 'mcp_auth',
            'node_name': 'mcp_auth_guard',
            'message': str(exc),
            'available_actions': ['authorize', 'skip'],
            'tool_name': resolved_tool_name,
            'toolkit_name': toolkit_name,
            'toolkit_type': toolkit_type,
            'tool_call_id': tool_call_id,
            # Checkpoint-private invoked name. The public tool_name may be the
            # resource operation exposed by the authorization exception.
            '_tool_call_name': tool_name,
            # Arguments are checkpoint-private. Authorization cards never need
            # request bodies, which may contain credentials or user content.
            'tool_args': {},
            'tool_args_raw': dict(tool_args or {}),
            'server_url': auth_metadata.get('server_url'),
            'resource_metadata_url': auth_metadata.get('resource_metadata_url'),
            'www_authenticate': auth_metadata.get('www_authenticate'),
            'resource_metadata': auth_metadata.get('resource_metadata'),
            'authorization_servers': auth_metadata.get('authorization_servers'),
            'status': auth_metadata.get('status'),
            'thread_id': configurable.get('thread_id'),
            'checkpoint_ns': configurable.get('checkpoint_ns') or '',
            '_source_node_name': self.name,
        }
        if serialized_pending:
            payload['_pending_messages'] = serialized_pending
        return payload

    @staticmethod
    def _mcp_auth_decision_message(interrupt_payload: dict, action: str) -> str:
        authorized = action == 'authorize'
        return build_mcp_auth_decision_result(
            status='authorized' if authorized else 'declined',
            server_url=interrupt_payload.get('server_url') or '',
            tool_name=interrupt_payload.get('tool_name') or '',
            toolkit_type=interrupt_payload.get('toolkit_type') or '',
            message=(
                'Authorization completed for this toolkit call only. Continue '
                'the original task from its next unfinished step.'
                if authorized else
                'The user skipped authorization for this toolkit call only. '
                'Continue the original task from its next unfinished step.'
            ),
            next_step=(
                'Retry the guarded operation using the authorized toolkit, then '
                'execute every remaining step required by the original task. Do '
                'not repeat work already completed before authorization.'
                if authorized else
                'Do not request this toolkit again during the current run. '
                'Execute every remaining step that does not require it, including '
                'all other required tool calls. Do not finish early solely because '
                'authorization was skipped, and do not repeat completed work.'
            ),
            denial_reason=None if authorized else 'User skipped authorization for this run.',
            resource_metadata_url=interrupt_payload.get('resource_metadata_url'),
            www_authenticate=interrupt_payload.get('www_authenticate'),
            resource_metadata=interrupt_payload.get('resource_metadata'),
        )

    async def _run_parallel_application_calls(
        self, app_specs, new_messages, config, hitl_decisions=None,
        pending_capture_start=0, pending_hitl_entries=None,
    ):
        """Execute multiple Application (sub-agent) tool calls concurrently.

        Children run in worker threads so their blocking sub-graph invocations
        overlap (elapsed ≈ max, not sum).  The default supervised path uses
        bounded tasks plus ``asyncio.wait(FIRST_COMPLETED)`` so a paused child
        can be decided and resumed while runnable siblings continue. The
        aggregate ``asyncio.gather`` path remains available as a compatibility
        opt-out.
        ``contextvars.copy_context()`` is captured per child so each runs in an
        isolated context (its own ``_PENDING_TOOL_MESSAGES`` slot). Per #5245
        there is no shared approval set — every sensitive call interrupts on its
        own, so each paused child surfaces its own approval independently.

        A child that pauses for sensitive-tool approval returns a deferred
        sentinel (it must NOT call ``interrupt()`` inside the executor thread —
        the raised GraphInterrupt would be captured by gather and the pause
        lost). All paused children are aggregated into ONE parent-level
        ``interrupt()`` carrying ``guardrail_type='parallel_sensitive_tools'`` so
        the UI surfaces N stacked approval cards and a single resume call routes
        each decision back to the correct child via its ``tool_call_id``.
        Completed children's ``ToolMessage``s are appended to ``new_messages`` in
        tool_call order regardless of whether siblings paused. See issue #4993.

        When no runnable child remains, both modes raise the same durable root
        aggregate interrupt.  That checkpoint is the fallback for process exit,
        reload, Stop, and lost live ownership.
        """
        from langchain_core.messages import ToolMessage, message_to_dict

        # Map prior decisions (this turn's resume) by the PARENT Application
        # tool_call_id so each paused child resumes from its own checkpoint.
        #
        # #5778 depth-3: a decision may instead target a GRANDCHILD (a leaf
        # paused two levels down, under one of THIS level's containers). Such
        # a decision's own tool_call_id is the leaf's id (never one of THIS
        # level's immediate children), but it carries `_via_call_id` pointing
        # at the immediate container it must be routed through. Group those
        # separately so each container gets its OWN sub-list of grandchild
        # decisions instead of being silently skipped (the original bug: the
        # root's decisions_by_id could never match a container id when every
        # decision was keyed by a leaf id).
        decisions_by_id = {}
        grandchild_decisions_by_via_id: Dict[str, list] = {}
        for decision in (hitl_decisions or []):
            tcid = decision.get(HITL_TOOL_CALL_ID_KEY)
            via_id = decision.get(HITL_VIA_CALL_ID_KEY)
            if via_id:
                grandchild_decisions_by_via_id.setdefault(via_id, []).append(decision)
            elif tcid:
                decisions_by_id[tcid] = decision

        # A partial root resume replays the complete gather node. Do not invoke
        # children that are still paused and did not receive this decision: a
        # normal Application invocation against their interrupted checkpoint
        # starts/replans the child and can generate fresh nested tool-call ids.
        # Reconstruct their prior deferred sentinels from the root checkpoint
        # instead. Selected children still resume normally below.
        prior_direct_by_id: Dict[str, list] = {}
        prior_nested_by_via_id: Dict[str, list] = {}
        for raw_entry in (pending_hitl_entries or []):
            if not isinstance(raw_entry, dict):
                continue
            via_id = raw_entry.get(HITL_VIA_CALL_ID_KEY)
            tool_call_id = raw_entry.get(HITL_TOOL_CALL_ID_KEY)
            if via_id:
                prior_nested_by_via_id.setdefault(via_id, []).append(raw_entry)
            elif tool_call_id:
                prior_direct_by_id.setdefault(tool_call_id, []).append(raw_entry)

        def _restore_child_entry(raw_entry: dict) -> dict:
            entry = dict(raw_entry)
            entry.pop(HITL_VIA_CALL_ID_KEY, None)
            nested_interrupt_id = entry.pop(HITL_NESTED_INTERRUPT_ID_KEY, None)
            if nested_interrupt_id:
                entry[HITL_INTERRUPT_ID_KEY] = nested_interrupt_id
                # The canonical marker applies to the public id at THIS
                # supervisor tier. Once we descend to the child-owned id, the
                # child must deterministically project it again. Retaining the
                # marker here would incorrectly expose the private nested id as
                # a new public card after any sibling resumes.
                entry.pop(HITL_CANONICAL_INTERRUPT_ID_KEY, None)
            return entry

        def _prior_deferred_interrupt(tool_call_id: str):
            nested_entries = prior_nested_by_via_id.get(tool_call_id) or []
            if nested_entries:
                pending = [_restore_child_entry(entry) for entry in nested_entries]
                guardrail_types = {
                    entry.get('guardrail_type') for entry in pending
                    if entry.get('guardrail_type')
                }
                if guardrail_types == {'mcp_auth'}:
                    guardrail_type = 'parallel_mcp_auth'
                elif guardrail_types == {'sensitive_tool'}:
                    guardrail_type = 'parallel_sensitive_tools'
                else:
                    guardrail_type = 'parallel_guardrails'
                return {
                    'type': 'hitl',
                    'guardrail_type': guardrail_type,
                    'message': pending[0].get(
                        'message', 'Multiple actions require your review before continuing.',
                    ),
                    'pending': pending,
                }
            direct_entries = prior_direct_by_id.get(tool_call_id) or []
            if direct_entries:
                return _restore_child_entry(direct_entries[0])
            return None

        loop = asyncio.get_running_loop()

        async def _run_one(sibling_ordinal, spec):
            tool_name, tool_args, tool_call_id, tool = spec
            envelope = {"type": "tool_call", "id": tool_call_id, "args": tool_args, "name": tool_name}
            child_config = dict(config)
            child_config['configurable'] = dict(config.get('configurable', {}))
            child_config['configurable']['__independent_parallel_hitl__'] = (
                self.independent_parallel_hitl
            )
            child_config['configurable']['__parallel_hitl_max_concurrency__'] = (
                self.parallel_hitl_max_concurrency
            )
            child_config['metadata'] = dict(config.get('metadata', {}))
            child_config['metadata']['sibling_ordinal'] = sibling_ordinal
            decision = decisions_by_id.get(tool_call_id)
            grandchild_decisions = grandchild_decisions_by_via_id.get(tool_call_id)
            prior_interrupt = _prior_deferred_interrupt(tool_call_id)
            if decision is None and not grandchild_decisions and prior_interrupt:
                logger.info(
                    "[HITL] Carrying forward untouched paused child '%s' "
                    "(tool_call_id=%s) without re-invocation",
                    tool_name, tool_call_id,
                )
                return {
                    '__hitl_deferred__': True,
                    'hitl_interrupt': prior_interrupt,
                    'tool_call_id': tool_call_id,
                }
            if decision is not None:
                # Resume this child from its checkpoint with the user's decision
                # (the derived child thread_id is keyed by this tool_call_id).
                child_config['configurable']['__hitl_parallel_resume__'] = {
                    'action': decision.get('action', 'approve'),
                    'value': decision.get('value', decision.get('user_feedback', '')),
                    'guardrail_type': decision.get('guardrail_type'),
                    # The root aggregate owns a public interrupt id, while the
                    # child checkpoint owns the nested id. Forward the latter
                    # so checkpoint drift cannot apply an MCP decision to a
                    # newer sensitive-tool pause (or vice versa).
                    'interrupt_id': (
                        decision.get(HITL_NESTED_INTERRUPT_ID_KEY)
                        or decision.get(HITL_INTERRUPT_ID_KEY)
                    ),
                }
                # A supervised parallel child stays alive inside the original
                # worker process. Unlike an ordinary checkpoint continuation,
                # its Application instance is not reconstructed by the worker
                # after OAuth, so its args_runnable still contains the pre-auth
                # token snapshot. Core sends newly issued tokens only on the
                # internal decision transport; pass them as an invocation-local
                # control that Application consumes before rebuilding the child.
                # Never place this value in interrupt/checkpoint payloads.
                if decision.get('_mcp_tokens') is not None:
                    child_config['configurable']['__live_mcp_tokens__'] = (
                        decision['_mcp_tokens']
                    )
            elif grandchild_decisions:
                # This child is itself a container: its OWN prior pause was a
                # nested `parallel_sensitive_tools` aggregate (issue #5778). Pass
                # the whole sub-list through so Application._run can resume the
                # container's graph with `hitl_decisions` (list) rather than a
                # single action/value pair — the container's own LLMNode then
                # re-runs ITS `_run_parallel_application_calls` and routes each
                # decision to the correct leaf via the existing single-level
                # `decisions_by_id` machinery one level down.
                #
                # CONSUME one hop of the routing chain: `_via_call_id` pointed
                # this decision at THIS container (call_id == tool_call_id here).
                # One level down, the decision's own `tool_call_id` (the leaf id)
                # IS a direct child, so it must land in the container's
                # `decisions_by_id`, not be re-bucketed as a grandchild. Drop the
                # now-consumed `_via_call_id` so the child level routes it as a
                # direct decision. (If depth-4+ is ever supported, this becomes a
                # stack pop instead of a single strip.)
                _forwarded = []
                for _d in grandchild_decisions:
                    _d2 = dict(_d)
                    _d2.pop(HITL_VIA_CALL_ID_KEY, None)
                    # The root aggregate exposes a root-scoped public id, while
                    # this container's own checkpoint stores its original
                    # child-scoped id. Restore that inner public id as the route
                    # hop is consumed so the child's checkpoint-authoritative
                    # hydration can validate the decision.
                    _nested_interrupt_id = _d2.pop(
                        HITL_NESTED_INTERRUPT_ID_KEY, None,
                    )
                    if _nested_interrupt_id:
                        _d2[HITL_INTERRUPT_ID_KEY] = _nested_interrupt_id
                    _forwarded.append(_d2)
                child_config['configurable']['__hitl_parallel_resume__'] = {
                    'decisions': _forwarded,
                }
            # Deferred mode must stay sticky ACROSS resume, not just on the fresh
            # run. A resumed child whose LLM picks a DIFFERENT sensitive tool on
            # its next turn would otherwise call interrupt() inside this executor
            # thread — where asyncio.gather captures the GraphInterrupt and the
            # pause is lost. Keeping it on means a post-resume pause RETURNS a
            # sentinel and re-aggregates into a fresh parent interrupt (multi-round
            # parallel HITL — issue #4993 follow-up).
            child_config['configurable']['__hitl_deferred_mode__'] = True
            child_config['configurable']['__hitl_parallel_call_id__'] = tool_call_id
            ctx = contextvars.copy_context()
            return await loop.run_in_executor(
                None, lambda c=ctx, t=tool, e=envelope, cc=child_config: c.run(t.invoke, e, config=cc),
            )

        def _build_pending_payload(deferred_items):
            """Project deferred child pauses into stable root-scoped cards."""
            payload: list[PendingHITLEntry] = []
            for spec, sentinel in deferred_items:
                _tn, _ta, tool_call_id, _tool = spec
                nested_aggregate = sentinel.get('hitl_interrupt') or {}
                nested_pending = nested_aggregate.get('pending')
                if isinstance(nested_pending, list) and nested_pending:
                    for leaf_index, leaf_entry in enumerate(nested_pending):
                        if not isinstance(leaf_entry, dict):
                            continue
                        flat_entry = cast(PendingHITLEntry, dict(leaf_entry))
                        flat_entry[HITL_VIA_CALL_ID_KEY] = tool_call_id
                        canonical_id = (
                            self.independent_parallel_hitl
                            and bool(flat_entry.get(
                                HITL_CANONICAL_INTERRUPT_ID_KEY
                            ))
                        )
                        if flat_entry.get(HITL_INTERRUPT_ID_KEY):
                            flat_entry[HITL_NESTED_INTERRUPT_ID_KEY] = flat_entry[
                                HITL_INTERRUPT_ID_KEY
                            ]
                        nested_interrupt_id = flat_entry.get(
                            HITL_NESTED_INTERRUPT_ID_KEY
                        )
                        if not canonical_id:
                            flat_entry[HITL_INTERRUPT_ID_KEY] = self._parallel_interrupt_id(
                                tool_call_id,
                                str(
                                    nested_interrupt_id
                                    or flat_entry.get(HITL_TOOL_CALL_ID_KEY)
                                    or flat_entry.get('tool_name')
                                    or ''
                                ),
                                None if nested_interrupt_id else leaf_index,
                            )
                        # Once an interrupt has been projected by the innermost
                        # active supervisor, every outer supervisor must reuse
                        # that public id. Re-scoping it at each Application
                        # boundary renders duplicate cards for the same leaf and
                        # makes the earlier card impossible for the outer
                        # checkpoint to hydrate.
                        if self.independent_parallel_hitl:
                            flat_entry[HITL_CANONICAL_INTERRUPT_ID_KEY] = True
                        flat_entry.pop('_pending_messages', None)
                        payload.append(flat_entry)
                    continue

                entry = cast(PendingHITLEntry, dict(nested_aggregate))
                entry[HITL_TOOL_CALL_ID_KEY] = tool_call_id
                nested_interrupt_id = nested_aggregate.get(HITL_INTERRUPT_ID_KEY)
                interrupt_route_seed = str(
                    nested_interrupt_id
                    or ':'.join(filter(None, (
                        tool_call_id,
                        nested_aggregate.get('tool_name'),
                        nested_aggregate.get('message'),
                    )))
                )
                entry[HITL_NESTED_INTERRUPT_ID_KEY] = interrupt_route_seed
                canonical_id = (
                    self.independent_parallel_hitl
                    and bool(nested_aggregate.get(
                        HITL_CANONICAL_INTERRUPT_ID_KEY
                    ))
                )
                if not canonical_id:
                    entry[HITL_INTERRUPT_ID_KEY] = self._parallel_interrupt_id(
                        tool_call_id,
                        interrupt_route_seed,
                        None if nested_interrupt_id else len(payload),
                    )
                if self.independent_parallel_hitl:
                    entry[HITL_CANONICAL_INTERRUPT_ID_KEY] = True
                app_meta = getattr(_tool, 'metadata', None) or {}
                sub_agent_name = (
                    app_meta.get('original_name')
                    or app_meta.get('display_name')
                    or getattr(_tool, 'name', '')
                    or _tn
                )
                if sub_agent_name:
                    entry['parent_agent_name'] = sub_agent_name
                entry.pop('_pending_messages', None)
                payload.append(entry)
            return payload

        async def _run_supervised():
            """Settle/resume children independently while retaining one process."""
            from .. import _parallel_hitl_registry as decision_registry

            configurable = (config or {}).get('configurable', {})
            # ``thread_id`` is rewritten at every Application boundary so each
            # nested graph owns a distinct checkpoint.  It is therefore not
            # the worker mailbox address.  The worker stamps its canonical
            # root id separately and every nested supervisor routes through it.
            # Otherwise a durable fallback can reconstruct the root runnable
            # against a child checkpoint and resume the orchestrator instead
            # of the interrupted caller (#6264).
            root_thread_id = str(
                configurable.get('__parallel_hitl_root_thread_id__')
                or configurable.get('thread_id')
                or ''
            )
            if not root_thread_id:
                return await asyncio.gather(
                    *[
                        _run_one(index + 1, spec)
                        for index, spec in enumerate(app_specs)
                    ],
                    return_exceptions=True,
                )

            semaphore = asyncio.Semaphore(self.parallel_hitl_max_concurrency)

            async def _bounded(index):
                async with semaphore:
                    try:
                        return await _run_one(index + 1, app_specs[index])
                    except BaseException as exc:  # preserve gather return_exceptions parity
                        return exc

            results = [None] * len(app_specs)
            running = {
                asyncio.create_task(_bounded(index)): index
                for index in range(len(app_specs))
            }
            paused_entries: Dict[int, list] = {}
            supervisor_id = decision_registry.attach(root_thread_id, loop)
            decision_waiter = None
            for spec in app_specs:
                dispatch_custom_event(
                    'parallel_hitl_state',
                    {
                        'state': 'running',
                        'root_thread_id': root_thread_id,
                        'tool_call_id': spec[2],
                    },
                    config=config,
                )

            def _public_entry(entry):
                public = dict(entry)
                for key in (
                    HITL_CANONICAL_INTERRUPT_ID_KEY,
                    HITL_NESTED_INTERRUPT_ID_KEY,
                    HITL_VIA_CALL_ID_KEY,
                    'tool_args_raw',
                    '_pending_messages',
                ):
                    public.pop(key, None)
                public['root_thread_id'] = root_thread_id
                public['resume_strategy'] = 'supervised_child'
                return public

            def _publish_pause(index, result):
                entries = _build_pending_payload([(app_specs[index], result)])
                paused_entries[index] = entries
                decision_registry.advertise(
                    root_thread_id,
                    supervisor_id,
                    [
                        str(entry.get(HITL_INTERRUPT_ID_KEY) or '')
                        for entry in entries
                    ],
                )
                public_entries = [_public_entry(entry) for entry in entries]
                if not public_entries:
                    return
                dispatch_custom_event(
                    'parallel_hitl_state',
                    {
                        'state': 'paused',
                        'root_thread_id': root_thread_id,
                        'interrupt_id': public_entries[0].get(
                            HITL_INTERRUPT_ID_KEY
                        ),
                        'tool_call_id': app_specs[index][2],
                    },
                    config=config,
                )
                dispatch_custom_event(
                    'parallel_hitl_interrupt',
                    {
                        'root_thread_id': root_thread_id,
                        'thread_id': root_thread_id,
                        'resume_strategy': 'supervised_child',
                        'hitl_interrupt': public_entries[0],
                        'hitl_interrupts': public_entries,
                        'message': public_entries[0].get(
                            'message', 'Awaiting human review...',
                        ),
                    },
                    config=config,
                )

            def _route_decisions(decisions):
                routed: Dict[int, list] = {}
                for raw_decision in decisions:
                    if not isinstance(raw_decision, dict):
                        continue
                    interrupt_id = raw_decision.get(HITL_INTERRUPT_ID_KEY)
                    target = None
                    target_entry = None
                    for index, entries in paused_entries.items():
                        target_entry = next(
                            (
                                entry for entry in entries
                                if entry.get(HITL_INTERRUPT_ID_KEY) == interrupt_id
                            ),
                            None,
                        )
                        if target_entry is not None:
                            target = index
                            break
                    if target is None or target_entry is None:
                        logger.info(
                            '[HITL] Ignoring stale supervised decision interrupt_id=%s',
                            interrupt_id,
                        )
                        continue

                    routed.setdefault(target, []).append(
                        (dict(raw_decision), target_entry)
                    )

                for target, target_decisions in routed.items():
                    routed_by_via: Dict[str, list] = {}
                    direct_decision = None
                    for raw_decision, target_entry in target_decisions:
                        decision = dict(raw_decision)
                        decision[HITL_TOOL_CALL_ID_KEY] = target_entry.get(
                            HITL_TOOL_CALL_ID_KEY
                        )
                        nested_id = target_entry.get(HITL_NESTED_INTERRUPT_ID_KEY)
                        if nested_id:
                            decision[HITL_NESTED_INTERRUPT_ID_KEY] = nested_id
                        via_id = target_entry.get(HITL_VIA_CALL_ID_KEY)
                        if via_id:
                            decision[HITL_VIA_CALL_ID_KEY] = via_id
                            routed_by_via.setdefault(via_id, []).append(decision)
                        else:
                            direct_decision = decision

                    for via_id, via_decisions in routed_by_via.items():
                        grandchild_decisions_by_via_id[via_id] = via_decisions
                    if direct_decision is not None:
                        decisions_by_id[str(
                            direct_decision.get(HITL_TOOL_CALL_ID_KEY) or ''
                        )] = direct_decision

                    # One paused container can expose multiple nested leaf
                    # cards. Reconstruct it once with every decision that was
                    # committed in this drain; launching once per card would
                    # discard later decisions and replay the same container.
                    owned_entries = paused_entries.get(target) or []
                    decision_registry.withdraw(
                        root_thread_id,
                        supervisor_id,
                        [
                            str(entry.get(HITL_INTERRUPT_ID_KEY) or '')
                            for entry in owned_entries
                        ],
                    )
                    paused_entries.pop(target, None)
                    results[target] = None
                    task = asyncio.create_task(_bounded(target))
                    running[task] = target
                    for raw_decision, target_entry in target_decisions:
                        dispatch_custom_event(
                            'parallel_hitl_state',
                            {
                                'state': 'resuming',
                                'root_thread_id': root_thread_id,
                                'interrupt_id': raw_decision.get(
                                    HITL_INTERRUPT_ID_KEY
                                ),
                                'tool_call_id': target_entry.get(
                                    HITL_TOOL_CALL_ID_KEY
                                ),
                            },
                            config=config,
                        )

                return bool(routed)

            try:
                while running or paused_entries:
                    # Worker teardown removes the live mailbox and wakes this
                    # coroutine.  Let siblings that are still running settle
                    # before building the durable aggregate; breaking on the
                    # first paused child made the visible card set depend on
                    # asyncio scheduling.  With no live mailbox, wait only for
                    # running children, never for an unreachable decision.
                    mailbox_active = decision_registry.is_active(root_thread_id)
                    if paused_entries and not mailbox_active and not running:
                        break
                    queued = decision_registry.drain(
                        root_thread_id, supervisor_id,
                    )
                    if queued and _route_decisions(queued):
                        continue

                    wait_set = set(running)
                    if paused_entries and mailbox_active:
                        # A supervisor is a coroutine inside the worker's existing
                        # process, not a parked process per child.  Keep it alive
                        # even when EVERY child is paused so a later decision can
                        # resume the exact owning leaf without replaying the root
                        # tool call.  Detaching here used to make every click miss
                        # live ownership and fall back through the durable root
                        # checkpoint, which rebuilt the outer hierarchy and lost
                        # the independent-child UX.  The durable aggregate below
                        # is still the crash/restart fallback: if this coroutine is
                        # cancelled, the persisted child checkpoints and roster
                        # reconstruct the same pending leaves.
                        decision_waiter = asyncio.create_task(
                            decision_registry.wait(root_thread_id, supervisor_id)
                        )
                        wait_set.add(decision_waiter)
                    done, _ = await asyncio.wait(
                        wait_set, return_when=asyncio.FIRST_COMPLETED,
                    )

                    if decision_waiter is not None and decision_waiter in done:
                        decisions = decision_waiter.result()
                        decision_waiter = None
                        _route_decisions(decisions)

                    for task in done:
                        if task not in running:
                            continue
                        index = running.pop(task)
                        result = task.result()
                        results[index] = result
                        if isinstance(result, dict) and result.get('__hitl_deferred__'):
                            _publish_pause(index, result)
                        else:
                            paused_entries.pop(index, None)
                            dispatch_custom_event(
                                'parallel_hitl_state',
                                {
                                    'state': (
                                        'failed'
                                        if isinstance(result, BaseException)
                                        else 'completed'
                                    ),
                                    'root_thread_id': root_thread_id,
                                    'tool_call_id': app_specs[index][2],
                                },
                                config=config,
                            )

                    if decision_waiter is not None:
                        decision_waiter.cancel()
                        await asyncio.gather(decision_waiter, return_exceptions=True)
                        decision_waiter = None
            finally:
                if decision_waiter is not None:
                    decision_waiter.cancel()
                decision_registry.detach(root_thread_id, supervisor_id)

            for index in paused_entries:
                # ``results`` already contains the latest deferred sentinel.
                if results[index] is None:
                    logger.error(
                        '[HITL] Supervised child %s lost its deferred state', index,
                    )
            return results

        # The supervisor removes only the all-child settlement barrier; both
        # modes build the same durable root interrupt below. The legacy
        # aggregate path remains available as an explicit compatibility opt-out.
        if self.independent_parallel_hitl:
            results = await _run_supervised()
        else:
            results = await asyncio.gather(
                *[
                    _run_one(index + 1, spec)
                    for index, spec in enumerate(app_specs)
                ],
                return_exceptions=True,
            )

        pending_deferred = []
        for spec, result in zip(app_specs, results):
            tool_name, _tool_args, tool_call_id, _tool = spec
            if isinstance(result, GraphBubbleUp):
                raise result
            if isinstance(result, BaseException):
                if isinstance(result, OutputContinuationExhausted):
                    raise result
                # Never hand a budget rejection back as tool output: the parent model
                # would reason about it as data and may retry or paraphrase it
                budget_error = budget_exceeded_from(result)
                if budget_error is not None:
                    raise budget_error from result
                logger.debug("Parallel sub-agent '%s' failed: %s", tool_name, result)
                new_messages.append(ToolMessage(
                    content=f"Error executing {tool_name}: {result}",
                    tool_call_id=tool_call_id,
                    status="error",
                ))
                continue
            if isinstance(result, dict) and result.get('__hitl_deferred__'):
                pending_deferred.append((spec, result))
                continue
            if isinstance(result, ToolMessage):
                if not result.tool_call_id:
                    result.tool_call_id = tool_call_id
                new_messages.append(result)
            else:
                new_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))

        if not pending_deferred:
            return

        # Build ONE aggregated interrupt for all paused children. Each entry is
        # the single-shape sensitive-tool payload, re-keyed to the PARENT
        # Application tool_call_id so the resume decision routes back here.
        #
        # #5778 depth-3: a paused child can ITSELF be a container whose own
        # pause was already a `parallel_sensitive_tools` aggregate (its own
        # leaves paused underneath it). Flattening that nested aggregate to a
        # single entry keyed by the container's tool_call_id (old behaviour)
        # would overwrite the leaf's own tool_call_id and permanently sever the
        # UI-displayed card's id from the id the resume decision must target —
        # decisions_by_id (below, on this container's NEXT resume) would then
        # never match, and the leaf's pause could never be answered. Instead,
        # surface each nested leaf entry as its OWN top-level pending item,
        # preserving its original tool_call_id and stamping `_via_call_id` with
        # THIS level's container id so a later resume can be regrouped and
        # routed back down (see the grandchild_decisions_by_via_id map above).
        pending_payload = _build_pending_payload(pending_deferred)

        # Expose + embed the parent's intermediate messages (completed siblings
        # and the AIMessage that owns all N tool_calls) so the resume restores
        # them and the LLMNode skips the finished siblings. Mirrors the
        # single-tool guard's _pending_messages contract (PR #199).
        intermediate = list(new_messages[pending_capture_start:])
        _PENDING_TOOL_MESSAGES.set(intermediate)
        pending_serialized = []
        for _m in intermediate:
            try:
                pending_serialized.append(message_to_dict(_m))
            except Exception:  # pragma: no cover - defensive
                pass

        guardrail_types = {
            item.get('guardrail_type') for item in pending_payload
            if item.get('guardrail_type')
        }
        if guardrail_types == {'mcp_auth'}:
            aggregate_guardrail = 'parallel_mcp_auth'
        elif guardrail_types == {'sensitive_tool'}:
            aggregate_guardrail = 'parallel_sensitive_tools'
        else:
            aggregate_guardrail = 'parallel_guardrails'
        aggregate = {
            'type': 'hitl',
            'guardrail_type': aggregate_guardrail,
            'message': pending_payload[0].get(
                'message', 'Multiple actions require your review before continuing.',
            ),
            'pending': pending_payload,
            '_pending_messages': pending_serialized,
        }
        logger.info(
            "[HITL] Aggregating %d paused parallel sub-agent(s) into one interrupt",
            len(pending_payload),
        )
        _langgraph_interrupt(aggregate)  # raises GraphInterrupt → parent pregel

    async def __perform_tool_calling(
        self, completion, messages, llm_client, config, hitl_decisions=None,
        pending_hitl_entries=None, pending_capture_base=None, parked_holder=None,
    ):
        # Several historical branches import ToolMessage locally.  Bind it
        # before any awaited tool invocation so an exception raised before one
        # of those imports can still be converted into a guardrail result.
        from langchain_core.messages import ToolMessage

        # Handle iterative tool-calling and execution
        logger.info(f"__perform_tool_calling called with {len(completion.tool_calls) if hasattr(completion, 'tool_calls') else 0} tool calls")
        normalize_null_tool_call_ids(completion)

        # Per-call independent approval (issue #5303): a sensitive tool the user
        # rejects is NOT excluded from future turns. The blocked tool stays bound
        # and the loop re-invokes with the full toolset; the only steer is the
        # invocation-scoped guidance line inside the blocked ToolMessage. If the
        # user rejects create_file call #1 they can still approve create_file #2.
        new_messages = self._append_completion_dedup(list(messages), completion)
        iteration = 0

        # Track the number of input messages so we can compute intermediate
        # messages produced during this execution (for HITL checkpoint restore).
        _input_msg_count = len(messages)

        # Index of the tool-calling AIMessage in ``new_messages``.  When the
        # completion was deduplicated against ``messages`` (multi-tool sibling
        # HITL resume case), this points to the existing AIMessage so the
        # captured pending_messages always include the AIMessage that owns the
        # tool_calls — without it, the restored ToolMessages would be orphaned
        # and stripped by ``_filter_orphaned_tool_calls`` on the next resume.
        try:
            _completion_index = new_messages.index(completion)
        except ValueError:
            _completion_index = _input_msg_count
        _pending_capture_start = min(_completion_index, _input_msg_count)

        # On an HITL resume, ``messages`` already has prior-cycle tool history
        # appended (restored from the previous interrupt's pending_messages),
        # so ``_input_msg_count`` sits PAST that region.  Anchor the capture
        # window at the durable checkpoint base instead, so the pending we hand
        # to the next interrupt is the FULL cumulative history — not just this
        # cycle's slice.  Otherwise earlier executed-tool results are shed on
        # each resume and the LLM re-plans from scratch, re-invoking
        # already-approved sensitive tools (#5245).
        if pending_capture_base is not None:
            _pending_capture_start = min(_pending_capture_start, pending_capture_base)

        # Reset the pending-messages contextvar at the start of each execution.
        _PENDING_TOOL_MESSAGES.set([])

        # Local, per-turn step budget. Mid-turn injections may bump it (bounded);
        # never mutate self.steps_limit, which is shared across turns.
        _injection_thread_id = (config or {}).get('configurable', {}).get('thread_id')
        effective_limit = self.steps_limit
        _injection_budget_max = self.steps_limit * 2

        # Continue executing tools until no more tool calls or max iterations reached
        current_completion = completion
        while (hasattr(current_completion, 'tool_calls') and
               current_completion.tool_calls and
               iteration < effective_limit):

            iteration += 1
            logger.info(f"Tool execution iteration {iteration}/{effective_limit}")

            # Execute each tool call in the current completion
            tool_calls = current_completion.tool_calls if hasattr(current_completion.tool_calls,
                                                                  '__iter__') else []

            # ── Parallel sub-agent fan-out (issue #4993) ────────────────────
            # When the assistant turn contains 2+ Application (sub-agent) tool
            # calls and NOTHING else, run them concurrently. Mixed batches
            # (Application + a regular toolkit call) keep the sequential path.
            # Parallelism is LLM-driven — steered by TASK_DELEGATION_ADDON and
            # the sub-agent tool descriptions — not a feature flag.
            app_specs = self._collect_parallel_application_specs(
                tool_calls, new_messages, config,
                hitl_decisions=hitl_decisions,
            )
            if app_specs is not None:
                # ── Track 2 (#4993): durable park-by-returning ──────────────
                # When a child_dispatcher seam is injected, do NOT run the
                # children in-process. Build their dispatch specs, hand them to
                # the caller via parked_holder, and RETURN immediately. The
                # parent's AIMessage (with N dangling tool_calls) stays in
                # new_messages; the LangGraph node ends, its task goes terminal,
                # and pylon_main launches each child as an independent durable
                # task. We must NOT fall through to the LLM re-invoke below —
                # there are no ToolMessages yet, so re-invoking would error on a
                # dangling-tool-call AIMessage. dispatcher None → fall through to
                # today's in-process gather (Track 1 baseline, CLI/tests intact).
                if self.child_dispatcher is not None and parked_holder is not None:
                    children = self._build_parallel_dispatch_specs(app_specs, config)
                    if children is not None:
                        parked_holder['parked'] = True
                        parked_holder['children'] = children
                        parked_holder['dispatch_epoch'] = next(
                            (spec.get('dispatch_epoch') for spec in children.values()),
                            None,
                        )
                        logger.info(
                            "[PARALLEL] child_dispatcher present — parking %d sub-agent(s) "
                            "for durable dispatch instead of in-process gather", len(children),
                        )
                        _PENDING_TOOL_MESSAGES.set([])
                        return new_messages, current_completion
                try:
                    await self._run_parallel_application_calls(
                        app_specs, new_messages, config,
                        hitl_decisions=hitl_decisions,
                        pending_capture_start=_pending_capture_start,
                        pending_hitl_entries=pending_hitl_entries,
                    )
                except GraphBubbleUp:
                    # The aggregated parallel interrupt() raised — mirror the
                    # sequential path's cleanup before propagating to the executor.
                    _PENDING_TOOL_MESSAGES.set([])
                    raise
                tool_calls = []  # handled in bulk; skip the sequential loop below

            for tool_call in tool_calls:
                tool_name = tool_call.get('name', '') if isinstance(tool_call, dict) else getattr(tool_call,
                                                                                                  'name',
                                                                                                  '')
                tool_args = tool_call.get('args', {}) if isinstance(tool_call, dict) else getattr(tool_call,
                                                                                                  'args',
                                                                                                  {})
                tool_call_id = tool_call.get('id', '') if isinstance(tool_call, dict) else getattr(
                    tool_call, 'id', '')

                # HITL resume safety: skip any tool_call whose id already has a
                # corresponding ToolMessage in the conversation history. This
                # prevents already-completed sibling tools from being re-executed
                # when the original AIMessage (with multiple tool_calls) is
                # reused as the resume completion. See issue #4333.
                if tool_call_id and self._tool_call_already_completed(
                    tool_call_id, new_messages,
                ):
                    logger.info(
                        "[HITL] Skipping tool_call '%s' (id=%s) — ToolMessage "
                        "already present in history (sibling already completed).",
                        tool_name, tool_call_id,
                    )
                    continue

                # Resolve the tool via the shared lookup chain (filtered →
                # available_tools → tool_registry). Extracted so the parallel
                # fan-out partition (#4993) and the sequential loop resolve
                # tools identically.
                tool_to_execute = self._resolve_tool_to_execute(tool_name, config)

                # Seed load_skill with bodies already in context so a re-load
                # answers "already active" (#5698); semantics documented on
                # loaded_skill_names_from_messages.
                if isinstance(tool_to_execute, LoadSkillTool):
                    tool_to_execute.mark_already_loaded(
                        loaded_skill_names_from_messages(new_messages)
                    )

                if tool_to_execute:
                    try:
                        logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")

                        # Expose accumulated intermediate messages BEFORE invoking
                        # the tool.  If the tool triggers a sensitive-tool interrupt,
                        # the guard reads this contextvar so the messages survive the
                        # checkpoint and can be restored on resume.
                        _PENDING_TOOL_MESSAGES.set(list(new_messages[_pending_capture_start:]))

                        # Application reads tool_call_id from a ToolCall envelope and
                        # returns a collapsed ToolMessage; other tools get raw args so
                        # langchain's BaseTool.invoke does not auto-wrap their result
                        # (which would defeat blocked-payload detection below).
                        from .application import Application
                        is_application = isinstance(tool_to_execute, Application)

                        tool_result = None
                        if is_application:
                            # Application overrides invoke() (not ainvoke) to read tool_call_id
                            # from a ToolCall envelope and collapse AgentResponse → output string.
                            # Routing to ainvoke would hit BaseTool.ainvoke instead, which auto-
                            # wraps the dict result and defeats the collapse.
                            tool_call_envelope = {
                                "type": "tool_call",
                                "id": tool_call_id,
                                "args": tool_args,
                                "name": tool_name,
                            }
                            # A single container Application may itself fan out
                            # into parallel leaves.  Stamp the supervisor contract
                            # on every Application boundary, not only on a 2+
                            # sibling batch at this level, so orchestrator ->
                            # container -> leaf hierarchies do not silently fall
                            # back to the legacy gather barrier.
                            application_config = dict(config or {})
                            application_config['configurable'] = dict(
                                application_config.get('configurable') or {}
                            )
                            application_config['configurable'][
                                '__independent_parallel_hitl__'
                            ] = self.independent_parallel_hitl
                            application_config['configurable'][
                                '__parallel_hitl_max_concurrency__'
                            ] = self.parallel_hitl_max_concurrency
                            tool_result = tool_to_execute.invoke(
                                tool_call_envelope, config=application_config,
                            )
                        elif hasattr(tool_to_execute, 'ainvoke'):
                            try:
                                tool_result = await tool_to_execute.ainvoke(tool_args, config=config)
                            except (NotImplementedError, AttributeError):
                                logger.debug(f"Tool '{tool_name}' ainvoke failed, falling back to sync invoke")
                                tool_result = tool_to_execute.invoke(tool_args, config=config)
                        else:
                            # Sync-only tool
                            tool_result = tool_to_execute.invoke(tool_args, config=config)

                        # Create tool message with result - preserve structured content
                        from langchain_core.messages import ToolMessage

                        # Short-circuit: Application returned an already-formed ToolMessage
                        # with collapsed content (output string, not stringified dict).
                        if isinstance(tool_result, ToolMessage):
                            if not tool_result.tool_call_id:
                                tool_result.tool_call_id = tool_call_id
                            new_messages.append(tool_result)
                            continue

                        blocked_payload = self._parse_sensitive_tool_blocked_result(tool_result)
                        if blocked_payload is not None:
                            # User declined this sensitive call. The blocked tool
                            # stays bound (per-call independent approval, #5303).
                            # The guard already produces a SLIM structured payload
                            # (type + tool/toolkit identities + denial_reason + a
                            # single `message` directive). Pass it through verbatim
                            # so the model input is identical to the tool-trace the
                            # user sees — one source of truth. The directive in
                            # `message` is what steers continuation; the slim
                            # structure avoids the field bloat that tripped weak
                            # models (haiku, gpt-5.4-mini). If a payload somehow
                            # lacks `message`, synthesize a fallback directive.
                            if not blocked_payload.get('message'):
                                blocked_payload = {
                                    **blocked_payload,
                                    'message': self._build_blocked_tool_guidance(blocked_payload),
                                }
                            # status stays 'success': a declined sensitive action is not
                            # a tool failure, and status cannot carry ToolOutcome BLOCKED.
                            tool_message = ToolMessage(
                                content=json.dumps(
                                    blocked_payload,
                                    ensure_ascii=True,
                                    separators=(',', ':'),
                                ),
                                tool_call_id=tool_call_id,
                            )
                            new_messages.append(tool_message)
                            continue

                        # Check if tool_result is structured content (list of dicts)
                        # Only use the structured fast-path when every item has an
                        # LLM-standard content block type AND no bytes values are
                        # present (bytes are not JSON-serializable and would cause
                        # a 400 from the LLM API).
                        _STANDARD_CONTENT_TYPES = {"text", "image", "image_url", "document", "search_result"}

                        def _is_llm_safe_content_block(item: dict) -> bool:
                            if not isinstance(item, dict):
                                return False
                            if item.get('type') not in _STANDARD_CONTENT_TYPES:
                                return False
                            return not any(isinstance(v, bytes) for v in item.values())

                        if isinstance(tool_result, list) and tool_result and all(
                                _is_llm_safe_content_block(item) for item in tool_result
                        ):
                            # Use structured content directly for multimodal support
                            tool_message = ToolMessage(
                                content=tool_result,
                                tool_call_id=tool_call_id
                            )
                        else:
                            # Fallback to string conversion for other tool results
                            tool_message = ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call_id
                            )
                        new_messages.append(tool_message)

                    except GraphBubbleUp:
                        # GraphInterrupt (from interrupt()) and other graph-level
                        # signals must propagate to the graph executor.
                        _PENDING_TOOL_MESSAGES.set([])
                        raise
                    except McpAuthorizationRequired as exc:
                        # Delegated toolkit authorization is a durable guardrail,
                        # not an ordinary tool failure. Interrupt at the exact LLM
                        # tool-call boundary while the leaf call id, pending
                        # messages and child checkpoint are still available.
                        auth_interrupt = self._build_mcp_auth_interrupt(
                            exc,
                            tool_to_execute,
                            tool_name,
                            tool_args,
                            tool_call_id,
                            config,
                        )
                        resume_value = _langgraph_interrupt(auth_interrupt)
                        action = (
                            str((resume_value or {}).get('action') or 'skip')
                            .strip().lower()
                            if isinstance(resume_value, dict)
                            else 'skip'
                        )
                        if action not in {'authorize', 'skip'}:
                            action = 'skip'
                        new_messages.append(ToolMessage(
                            content=self._mcp_auth_decision_message(
                                auth_interrupt, action,
                            ),
                            tool_call_id=tool_call_id,
                        ))
                        continue
                    except Exception as e:
                        if isinstance(e, OutputContinuationExhausted):
                            _PENDING_TOOL_MESSAGES.set([])
                            raise
                        # Same reasoning as the MCP clause above: swallowing a budget
                        # rejection here hides it from the user entirely
                        budget_error = budget_exceeded_from(e)
                        if budget_error is not None:
                            _PENDING_TOOL_MESSAGES.set([])
                            raise budget_error from e
                        import traceback
                        error_details = traceback.format_exc()
                        # Use debug level to avoid duplicate output when CLI callbacks are active
                        logger.debug(f"Error executing tool '{tool_name}': {e}\n{error_details}")
                        # Create error tool message
                        from langchain_core.messages import ToolMessage
                        tool_message = ToolMessage(
                            content=f"Error executing {tool_name}: {str(e)}",
                            tool_call_id=tool_call_id,
                            status="error",
                        )
                        new_messages.append(tool_message)
                else:
                    logger.warning(f"Tool '{tool_name}' not found in available tools")
                    # Create error tool message for missing tool
                    from langchain_core.messages import ToolMessage
                    tool_message = ToolMessage(
                        content=f"Tool '{tool_name}' not available",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                    new_messages.append(tool_message)

            # ── Mid-turn user input injection drain ─────────────────────────
            # Fold any messages the user sent while this turn was running into
            # the next invoke, AFTER the tool results so no tool pair is
            # orphaned. Bump the local budget (bounded) so the interjection can
            # actually be acted on.
            if _injection_thread_id:
                try:
                    from .. import _injection_registry as _inj_reg
                    _injected = _inj_reg.drain_items(_injection_thread_id)
                except Exception:
                    _injected = []
                if _injected:
                    from langchain_core.messages import HumanMessage
                    for _inj_id, _text in _injected:
                        new_messages.append(HumanMessage(
                            content=f"[user interjected mid-task]: {_text}"))
                        if effective_limit < _injection_budget_max:
                            effective_limit += 1
                        _inj_reg.mark_consumed(_injection_thread_id, _inj_id)
                        self._dispatch_injection_ack(_inj_id, _text, config)
                    logger.info(
                        "[INJECTION] folded %d mid-turn message(s) for thread %s; "
                        "effective_limit now %d",
                        len(_injected), _injection_thread_id, effective_limit,
                    )

            # Call LLM again with tool results to get next response
            try:
                sanitized_messages = self._filter_orphaned_tool_calls(new_messages)
                if len(sanitized_messages) != len(new_messages):
                    logger.info(
                        "Filtered %s orphaned tool-call message(s) before follow-up LLM invoke",
                        len(new_messages) - len(sanitized_messages),
                    )
                new_messages = sanitized_messages

                # Re-invoke with the SAME full toolset — including any sensitive
                # tool the user just declined. The block is invocation-scoped
                # (per-call independent approval, #5303), so the tool stays bound
                # and the model can call it again for a different item if needed.
                # The invocation-scoped guidance carried inside the blocked
                # ToolMessage tells the model the call was declined and to
                # continue the remaining work; no forced rebinding or nudge turn.
                current_completion = llm_client.invoke(new_messages, config=config)
                current_completion = self._continue_nested_output(
                    messages=new_messages,
                    completion=current_completion,
                    config=config,
                )
                normalize_null_tool_call_ids(current_completion)
                new_messages.append(current_completion)

                # Check if we still have tool calls
                if hasattr(current_completion, 'tool_calls') and current_completion.tool_calls:
                    logger.info(f"LLM requested {len(current_completion.tool_calls)} more tool calls")
                else:
                    logger.info("LLM completed without requesting more tools")
                    break

            except GraphBubbleUp:
                # Preserve GraphInterrupt and related graph-level signals raised
                # anywhere in the tool iteration, including async-to-sync fallback.
                _PENDING_TOOL_MESSAGES.set([])
                raise
            except Exception as e:
                if isinstance(e, OutputContinuationExhausted):
                    _PENDING_TOOL_MESSAGES.set([])
                    raise
                # Checked before the string-matching classification below, which has no
                # budget bucket and would fall through to the generic append-and-break
                budget_error = budget_exceeded_from(e)
                if budget_error is not None:
                    _PENDING_TOOL_MESSAGES.set([])
                    raise budget_error from e
                error_str = str(e).lower()
                
                # Check for thinking model message format errors
                is_thinking_format_error = any(indicator in error_str for indicator in [
                    'expected `thinking`',
                    'expected `redacted_thinking`',
                    'thinking block',
                    'must start with a thinking block',
                    'when `thinking` is enabled'
                ])
                
                # Check for non-recoverable errors that should fail immediately
                # These indicate configuration or permission issues, not content size issues
                is_non_recoverable = any(indicator in error_str for indicator in [
                    'model identifier is invalid',
                    'authentication',
                    'unauthorized',
                    'access denied',
                    'permission denied',
                    'invalid credentials',
                    'api key',
                    'quota exceeded',
                    'rate limit'
                ])
                
                # Check for context window / token limit errors
                is_context_error = any(indicator in error_str for indicator in [
                    'context window', 'context_window', 'token limit', 'too long',
                    'maximum context length', 'input is too long', 'exceeds the limit',
                    'contextwindowexceedederror', 'max_tokens', 'content too large'
                ])
                
                # Check for Bedrock/Claude output limit errors (recoverable by truncation)
                is_output_limit_error = any(indicator in error_str for indicator in [
                    'output token',
                    'response too large',
                    'max_tokens_to_sample',
                    'output_token_limit',
                    'output exceeds'
                ])
                
                # Handle thinking model format errors
                if is_thinking_format_error:
                    model_info = getattr(llm_client, 'model_name', None) or getattr(llm_client, 'model', 'unknown')
                    logger.error(f"Thinking model message format error during tool execution iteration {iteration}")
                    logger.error(f"Model: {model_info}")
                    logger.error(f"Error details: {e}")
                    
                    error_msg = (
                        f"⚠️ THINKING MODEL FORMAT ERROR\n\n"
                        f"The model '{model_info}' uses extended thinking and requires specific message formatting.\n\n"
                        f"**Issue**: When 'thinking' is enabled, assistant messages must start with thinking blocks "
                        f"before any tool_use blocks. This framework cannot preserve thinking_blocks during iterative "
                        f"tool execution.\n\n"
                        f"**Root Cause**: Anthropic's Messages API is stateless - clients must manually preserve and "
                        f"resend thinking_blocks with every tool response. LangChain's message abstraction doesn't "
                        f"include thinking_blocks, so they are lost between turns.\n\n"
                        f"**Solutions**:\n"
                        f"1. **Recommended**: Use non-thinking model variants:\n"
                        f"   - claude-3-5-sonnet-20241022-v2:0 (instead of thinking variants)\n"
                        f"   - anthropic.claude-3-5-sonnet-20241022-v2:0 (Bedrock)\n"
                        f"2. Disable extended thinking: Set reasoning_effort=None or remove thinking config\n"
                        f"3. Use LiteLLM directly with modify_params=True (handles thinking_blocks automatically)\n"
                        f"4. Avoid tool calling with thinking models (use for reasoning tasks only)\n\n"
                        f"**Technical Context**: {str(e)}\n\n"
                        f"References:\n"
                        f"- https://docs.claude.com/en/docs/build-with-claude/extended-thinking\n"
                        f"- https://docs.litellm.ai/docs/reasoning_content (See 'Tool Calling with thinking' section)"
                    )
                    new_messages.append(AIMessage(content=error_msg))
                    raise ValueError(error_msg)
                
                # Handle non-recoverable errors immediately
                if is_non_recoverable:
                    # Enhanced error logging with model information for better diagnostics
                    model_info = getattr(llm_client, 'model_name', None) or getattr(llm_client, 'model', 'unknown')
                    logger.error(f"Non-recoverable error during tool execution iteration {iteration}")
                    logger.error(f"Model: {model_info}")
                    logger.error(f"Error details: {e}")
                    logger.error(f"Error type: {type(e).__name__}")
                    
                    # Provide detailed error message for debugging
                    error_details = []
                    error_details.append(f"Model configuration error: {str(e)}")
                    error_details.append(f"Model identifier: {model_info}")
                    
                    # Check for common Bedrock model ID issues
                    if 'model identifier is invalid' in error_str:
                        error_details.append("\nPossible causes:")
                        error_details.append("1. Model not available in the configured AWS region")
                        error_details.append("2. Model not enabled in your AWS Bedrock account")
                        error_details.append("3. LiteLLM model group prefix not stripped (check for prefixes like '1_')")
                        error_details.append("4. Incorrect model version or typo in model name")
                        error_details.append("\nPlease verify:")
                        error_details.append("- AWS Bedrock console shows this model as available")
                        error_details.append("- LiteLLM router configuration is correct")
                        error_details.append("- Model ID doesn't contain unexpected prefixes")
                    
                    error_msg = "\n".join(error_details)
                    new_messages.append(AIMessage(content=error_msg))
                    break
                
                if is_context_error or is_output_limit_error:
                    error_type = "output limit" if is_output_limit_error else "context window"
                    logger.warning(f"{error_type.title()} exceeded during tool execution iteration {iteration}")
                    
                    # Find the last tool message and its associated tool name
                    last_tool_msg_idx = None
                    last_tool_name = None
                    last_tool_call_id = None
                    
                    # First, find the last tool message
                    for i in range(len(new_messages) - 1, -1, -1):
                        msg = new_messages[i]
                        if hasattr(msg, 'tool_call_id') or (hasattr(msg, 'type') and getattr(msg, 'type', None) == 'tool'):
                            last_tool_msg_idx = i
                            last_tool_call_id = getattr(msg, 'tool_call_id', None)
                            break
                    
                    # Find the tool name from the AIMessage that requested this tool call
                    if last_tool_call_id:
                        for i in range(last_tool_msg_idx - 1, -1, -1):
                            msg = new_messages[i]
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    tc_id = tc.get('id', '') if isinstance(tc, dict) else getattr(tc, 'id', '')
                                    if tc_id == last_tool_call_id:
                                        last_tool_name = tc.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '')
                                        break
                                if last_tool_name:
                                    break
                    
                    # Build dynamic suggestion based on the tool that caused the overflow
                    tool_suggestions = self._get_tool_truncation_suggestions(last_tool_name)
                    
                    # Truncate the problematic tool result if found
                    if last_tool_msg_idx is not None:
                        from langchain_core.messages import ToolMessage
                        original_msg = new_messages[last_tool_msg_idx]
                        tool_call_id = getattr(original_msg, 'tool_call_id', 'unknown')
                        
                        # Build error-specific guidance
                        if is_output_limit_error:
                            truncated_content = (
                                f"⚠️ MODEL OUTPUT LIMIT EXCEEDED\n\n"
                                f"The tool '{last_tool_name or 'unknown'}' returned data, but the model's response was too large.\n\n"
                                f"IMPORTANT: You must provide a SMALLER, more focused response.\n"
                                f"- Break down your response into smaller chunks\n"
                                f"- Summarize instead of listing everything\n"
                                f"- Focus on the most relevant information first\n"
                                f"- If listing items, show only top 5-10 most important\n\n"
                                f"Tool-specific tips:\n{tool_suggestions}\n\n"
                                f"Please retry with a more concise response."
                            )
                        else:
                            truncated_content = (
                                f"⚠️ TOOL OUTPUT TRUNCATED - Context window exceeded\n\n"
                                f"The tool '{last_tool_name or 'unknown'}' returned too much data for the model's context window.\n\n"
                                f"To fix this:\n{tool_suggestions}\n\n"
                                f"Please retry with more restrictive parameters."
                            )
                        
                        # status stays 'success': the call succeeded, and status cannot
                        # carry ToolOutcome TRUNCATED.
                        truncated_msg = ToolMessage(
                            content=truncated_content,
                            tool_call_id=tool_call_id
                        )
                        new_messages[last_tool_msg_idx] = truncated_msg
                        
                        logger.info(f"Truncated large tool result from '{last_tool_name}' and retrying LLM call")

                        # CRITICAL FIX: Call LLM again with truncated message to get fresh completion
                        # This prevents duplicate tool_call_ids that occur when we continue with
                        # the same current_completion that still has the original tool_calls
                        try:
                            current_completion = llm_client.invoke(new_messages, config=config)
                            current_completion = self._continue_nested_output(
                                messages=new_messages,
                                completion=current_completion,
                                config=config,
                            )
                            normalize_null_tool_call_ids(current_completion)
                            new_messages.append(current_completion)

                            # Continue to process any new tool calls in the fresh completion
                            if hasattr(current_completion, 'tool_calls') and current_completion.tool_calls:
                                logger.info(f"LLM requested {len(current_completion.tool_calls)} more tool calls after truncation")
                                continue
                            else:
                                logger.info("LLM completed after truncation without requesting more tools")
                                break
                        except Exception as retry_error:
                            if isinstance(retry_error, OutputContinuationExhausted):
                                _PENDING_TOOL_MESSAGES.set([])
                                raise
                            logger.error(f"Error retrying LLM after truncation: {retry_error}")
                            error_msg = f"Failed to retry after truncation: {str(retry_error)}"
                            new_messages.append(AIMessage(content=error_msg))
                            break
                    else:
                        # Couldn't find tool message, add error and break
                        if is_output_limit_error:
                            error_msg = (
                                "Model output limit exceeded. Please provide a more concise response. "
                                "Break down your answer into smaller parts and summarize where possible."
                            )
                        else:
                            error_msg = (
                                "Context window exceeded. The conversation or tool results are too large. "
                                "Try using tools with smaller output limits (e.g., max_items, max_depth parameters)."
                            )
                        new_messages.append(AIMessage(content=error_msg))
                        break
                else:
                    logger.error(f"Error in LLM call during iteration {iteration}: {e}")
                    # Add error message and break the loop
                    error_msg = f"Error processing tool results in iteration {iteration}: {str(e)}"
                    new_messages.append(AIMessage(content=error_msg))
                    break

        # Handle max iterations (against the local budget, which injections may have bumped)
        if iteration >= effective_limit:
            logger.warning(f"Reached maximum iterations ({effective_limit}) for tool execution")
            
            # CRITICAL: Check if the last message is an AIMessage with pending tool_calls
            # that were not processed. If so, we need to add placeholder ToolMessages to prevent
            # the "assistant message with 'tool_calls' must be followed by tool messages" error
            # when the conversation continues.
            if new_messages:
                last_msg = new_messages[-1]
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    from langchain_core.messages import ToolMessage
                    pending_tool_calls = last_msg.tool_calls if hasattr(last_msg.tool_calls, '__iter__') else []
                    
                    # Check which tool_call_ids already have responses
                    existing_tool_call_ids = set()
                    for msg in new_messages:
                        if hasattr(msg, 'tool_call_id'):
                            existing_tool_call_ids.add(msg.tool_call_id)
                    
                    # Add placeholder responses for any tool calls without responses
                    for tool_call in pending_tool_calls:
                        tool_call_id = tool_call.get('id', '') if isinstance(tool_call, dict) else getattr(tool_call, 'id', '')
                        tool_name = tool_call.get('name', '') if isinstance(tool_call, dict) else getattr(tool_call, 'name', '')
                        
                        if tool_call_id and tool_call_id not in existing_tool_call_ids:
                            logger.info(f"Adding placeholder ToolMessage for interrupted tool call: {tool_name} ({tool_call_id})")
                            placeholder_msg = ToolMessage(
                                content=f"[Tool execution interrupted - step limit ({effective_limit}) reached before {tool_name} could be executed]",
                                tool_call_id=tool_call_id,
                                status="error",
                            )
                            new_messages.append(placeholder_msg)
            
            # Add warning message - CLI or calling code can detect this and prompt user
            warning_msg = f"Maximum tool execution iterations ({effective_limit}) reached. Stopping tool execution."
            new_messages.append(AIMessage(content=warning_msg))
        else:
            logger.info(f"Tool execution completed after {iteration} iterations")

        # Clear the pending-messages contextvar on normal completion.
        _PENDING_TOOL_MESSAGES.set([])
        return new_messages, current_completion

    # -----------------------------------------------------------------------
    # Anthropic thinking-mode detection
    # -----------------------------------------------------------------------

    @staticmethod
    def _anthropic_candidates(client: Any) -> list:
        """Return *client* plus its ``.bound`` if present.

        ``llm_client`` may be a tool-bound ``RunnableBinding`` (produced
        by ``bind_tools``) wrapping the real ChatAnthropic model, or the
        base ChatAnthropic directly. Both detection helpers below need to
        check both layers — this returns the list to walk.
        """
        try:
            bound = getattr(client, 'bound', None)
            return [client, bound] if bound is not None else [client]
        except Exception:  # pragma: no cover — defensive
            return [client]

    @staticmethod
    def _is_anthropic_client(client: Any) -> bool:
        """Return True when *client* is (or wraps) any langchain-anthropic
        ``ChatAnthropic`` — thinking or non-thinking.

        Used to decide whether the structured-output schema needs the
        ``$defs.JsonValue`` patch applied (Anthropic's ``transform_schema``
        rejects the empty def Pydantic emits for ``JsonValue``).
        """
        for candidate in LLMNode._anthropic_candidates(client):
            module = getattr(type(candidate), '__module__', '') or ''
            if 'langchain_anthropic' in module:
                return True
        return False

    @staticmethod
    def _client_is_openai_compatible(client: Any) -> bool:
        """Return True when *client* is (or wraps) an OpenAI-compatible
        passthrough client — e.g. Claude served via a LiteLLM
        ``/chat/completions`` endpoint as a ``ChatOpenAI``.

        The signal is stamped on the client at build time in
        ``EliteAClient.get_llm`` (``_elitea_openai_compatible``). Such backends
        reject the parallel_tool_calls / json_schema / output_format transforms
        litellm derives for Bedrock, so block-continuation and structured-output
        routing avoid those transforms for these clients.
        """
        for candidate in LLMNode._anthropic_candidates(client):
            if getattr(candidate, '_elitea_openai_compatible', False):
                return True
        return False

    @staticmethod
    def _is_anthropic_thinking_client(client: Any) -> bool:
        """Return True when *client* is (or wraps) a langchain-anthropic
        ChatAnthropic with thinking enabled (type "enabled" or "adaptive").
        """
        for candidate in LLMNode._anthropic_candidates(client):
            module = getattr(type(candidate), '__module__', '') or ''
            if 'langchain_anthropic' not in module:
                continue
            thinking = getattr(candidate, 'thinking', None)
            if isinstance(thinking, dict) and thinking.get('type') in ('enabled', 'adaptive'):
                return True
        return False

    @staticmethod
    def _anthropic_system_content(text: str, client: Any, dynamic_suffix: str = "") -> Any:
        """Return the SystemMessage content value appropriate for *client*.

        For Anthropic clients: a content-block list with a cache_control breakpoint
        so that langchain-anthropic 1.4.1+ forwards it to the Anthropic API and the
        system prompt is eligible for prompt caching.

        For all other clients: the plain string, unchanged — no behavior change.

        Args:
            text: The resolved system prompt text.
            client: The raw LLM client (NOT a bound-tools wrapper).
            dynamic_suffix: Optional per-turn content (e.g. invoked-skill guidance)
                that changes between turns. For Anthropic it is emitted as a SEPARATE
                block placed AFTER the cache breakpoint, so it does NOT invalidate the
                cached static prefix (instructions + tool schemas) on turns where it
                changes. For other clients it is concatenated onto the text.
        """
        if LLMNode._is_anthropic_client(client) and text:
            blocks = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
            if dynamic_suffix:
                # No cache_control: this block sits after the breakpoint and is re-priced
                # each turn, which is correct since its content varies per turn anyway.
                blocks.append({"type": "text", "text": dynamic_suffix})
            return blocks
        if dynamic_suffix:
            return f"{text}\n\n{dynamic_suffix}" if text else dynamic_suffix
        return text

    def __get_struct_output_model(
        self,
        llm_client: Any,
        pydantic_model: Any,
        method: Literal["function_calling", "json_mode", "json_schema"] = "function_calling",
    ) -> Any:
        """Return a structured-output runnable bound to ``pydantic_model``.

        Two provider-specific divergences are encoded here:

        1. **Anthropic schema patch.** ``parse_pydantic_type`` emits
           Pydantic's ``JsonValue`` for the ``"list"`` / ``"any"`` types,
           which OpenAI accepts (including the reasoning models — they
           hallucinate ``list[list[str]]`` under tighter element unions).
           Anthropic's ``transform_schema``, however, rejects the empty
           ``$defs.JsonValue`` Pydantic emits. For Anthropic clients we
           replace ``$defs.JsonValue`` with the canonical recursive
           concrete union via ``make_anthropic_compatible_schema`` and
           pass the resulting **dict** to ``with_structured_output`` —
           OpenAI / Azure / Google / etc. continue to receive the
           Pydantic class unchanged.

        2. **Thinking-mode method override** (issue #4890). For Anthropic
           with ``thinking={"type": "enabled"}`` and the default
           ``function_calling`` request, we force ``method='json_schema'``
           because ``function_calling`` routes through
           ``_raise_if_no_tool_calls`` which raises after the
           tool-calling exchange resolves to a plain synthesis turn.
           ``json_schema`` uses Anthropic's native ``output_format`` API
           parameter, which is compatible with thinking and does NOT go
           through ``_raise_if_no_tool_calls``.

        For non-Anthropic providers the ``method`` is forwarded unchanged
        and the Pydantic class is passed directly.

        Heterogeneous return: the Anthropic branch returns a runnable
        that yields ``dict`` (it received a dict schema); other providers
        yield Pydantic instances. Callers normalize with a one-line
        ``isinstance`` check.
        """
        if self._is_anthropic_client(self.client):
            schema_dict = make_anthropic_compatible_schema(pydantic_model)
            if method == "function_calling" and self._is_anthropic_thinking_client(self.client):
                return llm_client.with_structured_output(schema_dict, method='json_schema')
            return llm_client.with_structured_output(schema_dict, method=method)
        return llm_client.with_structured_output(pydantic_model, method=method)
