import asyncio
import contextvars
import json
import logging
from traceback import format_exc
from typing import Any, Optional, List, Union, Literal, Dict, TYPE_CHECKING

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

from ..langchain.constants import ELITEA_RS, SKILLS_SECTION_HEADER, SKILLS_SECTION_ENTRY, MAX_SKILLS_PER_INVOCATION
from ..langchain.utils import (
    args_match_normalized,
    create_pydantic_model,
    extract_json_content,
    make_anthropic_compatible_schema,
    propagate_the_input_mapping,
)
from ..toolkits.security import normalize_tool_name, qualified_tool_identity
from ..utils.mcp_oauth import McpAuthorizationRequired
if TYPE_CHECKING:
    from .lazy_tools import ToolRegistry

logger = logging.getLogger(__name__)

SENSITIVE_TOOL_BLOCKED_RESULT_TYPE = 'sensitive_tool_blocked'
STRUCTURED_OUTPUT_PREFILL_PROMPT = "Now produce the structured output based on the information above."

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
        logger.debug("Original structured-output error: %s", format_exc())
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
        # Check for dynamically selected tools from pre-LLM selection
        if config is not None:
            configurable = config.get('configurable', {}) if isinstance(config, dict) else {}
            selected_tools = configurable.get('selected_tools')
            if selected_tools:
                logger.info(f"[DynamicToolSelection] Using {len(selected_tools)} pre-selected tools")
                # Fix for #3290: Always include always_bind_tools (e.g., Planner tools) with
                # dynamically selected tools. Use `or []` to handle None/falsy gracefully.
                # This ensures Planner tools are available even on first message when
                # Smart Tools Selection finds matching toolkits.
                return list(selected_tools) + list(self.always_bind_tools or [])

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
                return combined_tools
            return meta_tools

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

        return base_tools

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
            self._meta_tools = create_meta_tools(self.tool_registry)
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
        except Exception as e:
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
            bool(self.available_tools)  # Bind available tools even when lazy mode auto-disabled
        )

        if should_bind_tools:
            filtered_tools = self.get_filtered_tools(config=config)
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
        if hitl_ctx and hitl_ctx.get('parallel_calls'):
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
            # `hitl_decisions` state channel, so consuming the parent's resume
            # values never robs a child of its answer.
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
            n_prior = n_positional + (1 if has_null else 0)
            if n_prior:
                logger.info(
                    "[HITL] Consuming %d pending parent resume value(s) before "
                    "parallel sub-agent re-fanout (multi-round): %d positional "
                    "+ %d null",
                    n_prior, n_positional, 1 if has_null else 0,
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
        logger.info(f"Initial completion: {completion}")

        # Handle both tool-calling and regular responses
        if hasattr(completion, 'tool_calls') and completion.tool_calls:
            # Handle iterative tool-calling and execution
            hitl_decisions = state.get('hitl_decisions') if isinstance(state, dict) else None

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
            
            if thinking_blocks:
                result['thinking'] = '\n\n'.join(thinking_blocks)
            if text_blocks:
                result['text'] = '\n\n'.join(text_blocks)
        
        # Handle simple string content
        elif isinstance(content, str):
            result['text'] = content
        
        return result
    
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
            d.get('tool_call_id')
            for d in (hitl_decisions or [])
            if isinstance(d, dict) and d.get('tool_call_id')
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

        The derived ``child_thread_id`` MUST match the in-process scheme in
        ``application.py`` (``f"{parent}:{name}:{call_id}"``) so the reconcile
        pass reads each child from the exact checkpoint the child wrote.
        """
        configurable = config.get('configurable', {}) if isinstance(config, dict) else {}
        parent_thread_id = configurable.get('thread_id')
        specs = {}
        for index, (tool_name, tool_args, tool_call_id, tool) in enumerate(app_specs):
            app_name = getattr(tool, 'name', None) or tool_name
            child_thread_id = (
                f"{parent_thread_id}:{app_name}:{tool_call_id}"
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
                'name': app_name,
                'display_name': display_name,
                'input': tool_args,
                'child_thread_id': child_thread_id,
                'index': index,
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

    async def _run_parallel_application_calls(
        self, app_specs, new_messages, config, hitl_decisions=None,
        pending_capture_start=0,
    ):
        """Execute multiple Application (sub-agent) tool calls concurrently.

        Children run in worker threads under ``asyncio.gather`` so their
        (blocking) sub-graph invocations overlap (elapsed ≈ max, not sum).
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

        All children are awaited to settlement before the aggregate interrupt is
        raised: ``interrupt()`` checkpoints and replays the WHOLE node on resume,
        so a sibling cannot be left running across a human pause on a stateless
        server — abandoning + re-running it desynchronises the resume-value
        replay cadence and re-fires the same interrupt in a loop.
        """
        from langchain_core.messages import ToolMessage, message_to_dict

        # Map prior decisions (this turn's resume) by the PARENT Application
        # tool_call_id so each paused child resumes from its own checkpoint.
        decisions_by_id = {}
        for decision in (hitl_decisions or []):
            tcid = decision.get('tool_call_id')
            if tcid:
                decisions_by_id[tcid] = decision

        loop = asyncio.get_running_loop()

        async def _run_one(spec):
            tool_name, tool_args, tool_call_id, tool = spec
            envelope = {"type": "tool_call", "id": tool_call_id, "args": tool_args, "name": tool_name}
            child_config = dict(config)
            child_config['configurable'] = dict(config.get('configurable', {}))
            decision = decisions_by_id.get(tool_call_id)
            if decision is not None:
                # Resume this child from its checkpoint with the user's decision
                # (the derived child thread_id is keyed by this tool_call_id).
                child_config['configurable']['__hitl_parallel_resume__'] = {
                    'action': decision.get('action', 'approve'),
                    'value': decision.get('value', decision.get('user_feedback', '')),
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

        # Run every child concurrently and wait for ALL to settle. A child that
        # pauses for sensitive-tool approval returns a deferred sentinel (it must
        # NOT call interrupt() inside the executor thread — gather would capture
        # the GraphInterrupt and lose the pause). Completed siblings produce
        # ToolMessages; paused siblings are aggregated into ONE parent interrupt
        # below. return_exceptions keeps one child's failure from cancelling the
        # rest (per-child isolation, issue #4993).
        results = await asyncio.gather(
            *[_run_one(spec) for spec in app_specs], return_exceptions=True,
        )

        pending_deferred = []
        for spec, result in zip(app_specs, results):
            tool_name, _tool_args, tool_call_id, _tool = spec
            if isinstance(result, GraphBubbleUp):
                raise result
            if isinstance(result, BaseException):
                logger.debug("Parallel sub-agent '%s' failed: %s", tool_name, result)
                new_messages.append(ToolMessage(
                    content=f"Error executing {tool_name}: {result}",
                    tool_call_id=tool_call_id,
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
        pending_payload = []
        for spec, sentinel in pending_deferred:
            _tn, _ta, tool_call_id, _tool = spec
            entry = dict(sentinel.get('hitl_interrupt') or {})
            entry['tool_call_id'] = tool_call_id
            # Label the paused card with the sub-agent it originated from so the
            # UI can group N stacked approvals by sub-agent name (issue #4993).
            # Falls back to the call name when metadata is absent.
            sub_agent_name = ''
            _app_meta = getattr(_tool, 'metadata', None) or {}
            sub_agent_name = (
                _app_meta.get('original_name')
                or _app_meta.get('display_name')
                or getattr(_tool, 'name', '')
                or _tn
            )
            if sub_agent_name:
                entry['parent_agent_name'] = sub_agent_name
            # Per-entry pending messages would duplicate the parent-level set
            # carried on the aggregate; drop them to keep the payload lean.
            entry.pop('_pending_messages', None)
            pending_payload.append(entry)

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

        aggregate = {
            'type': 'hitl',
            'guardrail_type': 'parallel_sensitive_tools',
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

    async def __perform_tool_calling(self, completion, messages, llm_client, config, hitl_decisions=None,
                                     pending_capture_base=None, parked_holder=None):
        # Handle iterative tool-calling and execution
        logger.info(f"__perform_tool_calling called with {len(completion.tool_calls) if hasattr(completion, 'tool_calls') else 0} tool calls")

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

        # Continue executing tools until no more tool calls or max iterations reached
        current_completion = completion
        while (hasattr(current_completion, 'tool_calls') and
               current_completion.tool_calls and
               iteration < self.steps_limit):

            iteration += 1
            logger.info(f"Tool execution iteration {iteration}/{self.steps_limit}")

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
                    parked_holder['parked'] = True
                    parked_holder['children'] = children
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
                            tool_result = tool_to_execute.invoke(tool_call_envelope, config=config)
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
                    except McpAuthorizationRequired:
                        # Re-raise so the parent agent's on_tool_error callback
                        # can emit the mcp_authorization_required socket event,
                        # which triggers the Login button in the Chat UI.
                        # Without this, the exception is swallowed into a ToolMessage
                        # and the nested agent silently fails to show the login prompt.
                        raise
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        # Use debug level to avoid duplicate output when CLI callbacks are active
                        logger.debug(f"Error executing tool '{tool_name}': {e}\n{error_details}")
                        # Create error tool message
                        from langchain_core.messages import ToolMessage
                        tool_message = ToolMessage(
                            content=f"Error executing {tool_name}: {str(e)}",
                            tool_call_id=tool_call_id
                        )
                        new_messages.append(tool_message)
                else:
                    logger.warning(f"Tool '{tool_name}' not found in available tools")
                    # Create error tool message for missing tool
                    from langchain_core.messages import ToolMessage
                    tool_message = ToolMessage(
                        content=f"Tool '{tool_name}' not available",
                        tool_call_id=tool_call_id
                    )
                    new_messages.append(tool_message)

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
                            new_messages.append(current_completion)
                            
                            # Continue to process any new tool calls in the fresh completion
                            if hasattr(current_completion, 'tool_calls') and current_completion.tool_calls:
                                logger.info(f"LLM requested {len(current_completion.tool_calls)} more tool calls after truncation")
                                continue
                            else:
                                logger.info("LLM completed after truncation without requesting more tools")
                                break
                        except Exception as retry_error:
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

        # Handle max iterations
        if iteration >= self.steps_limit:
            logger.warning(f"Reached maximum iterations ({self.steps_limit}) for tool execution")
            
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
                                content=f"[Tool execution interrupted - step limit ({self.steps_limit}) reached before {tool_name} could be executed]",
                                tool_call_id=tool_call_id
                            )
                            new_messages.append(placeholder_msg)
            
            # Add warning message - CLI or calling code can detect this and prompt user
            warning_msg = f"Maximum tool execution iterations ({self.steps_limit}) reached. Stopping tool execution."
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
