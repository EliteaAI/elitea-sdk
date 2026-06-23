"""Sensitive tool authorization guard middleware."""

import inspect
import json
import logging
import re
import types
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages.base import message_to_dict
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt

from .base import Middleware

logger = logging.getLogger(__name__)

from ..tools.application import Application
from ..toolkits.security import (
    find_sensitive_tool_match,
    get_sensitive_tool_policy,
    has_sensitive_tools_config,
    normalize_tool_name,
)


def normalize_tool_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Normalize tool arguments into a single payload."""
    if kwargs:
        return kwargs
    if len(args) == 1:
        return args[0]
    if not args:
        return {}
    return list(args)


class SensitiveToolGuardMiddleware(Middleware):
    """Pause execution before running configured sensitive tools."""

    # Marks this middleware as the sensitive-action guard so trusted contexts
    # (e.g. pipeline Code nodes running static, editor-authored code — issue #5348)
    # can opt out of it via MiddlewareManager.wrap_tool(skip_sensitive_guard=True)
    # while still receiving every other middleware (e.g. exception handling).
    GUARDS_SENSITIVE_ACTIONS = True

    BLOCKED_TOOL_RESULT_TYPE = 'sensitive_tool_blocked'

    # Default reason when the user rejected without typing a note.
    BLOCKED_TOOL_DEFAULT_REASON = 'denied by user'

    # Resume actions that mean "block this call but carry a free-text note"
    # (issue #5318). They map onto the reject execution path; the note rides
    # `value` into the blocked-tool result's `denial_reason` so the model reads
    # the user's reason and adapts its next step. `reject_with_comment` is an
    # accepted alias so a caller can use either name.
    BLOCK_WITH_COMMENT_ACTIONS = frozenset({'block_with_comment', 'reject_with_comment'})

    # Defensive cap on the user's note. A pathological multi-KB comment would
    # bloat every blocked-tool result and the checkpoint; the model only needs
    # the gist of the reason. Keep the head.
    MAX_COMMENT_LEN = 2000

    # `message` is the single natural-language directive a model reads off the
    # blocked result (there is no duplicate `guidance` field any more). The text
    # must NOT end on a terminal "stopped / not executed" note — that reads as
    # "halt" to a weak model (haiku, gpt-5.4-mini) and made them skip the rest
    # of the workflow. Lead with the block fact, then an explicit, imperative
    # continue-instruction. The slim structured fields around it (tool name,
    # toolkit, denial_reason) give the model machine-readable identities without
    # the field bloat that was tripping weak models.
    BLOCKED_TOOL_MESSAGE = (
        "You declined THIS specific call to '{action_label}'; it was not executed. "
        "The block is for THIS invocation only, not the tool itself. "
        "This is NOT a stop signal — do not end your turn or summarize yet. "
        "Do not retry this same call with the same arguments, but DO continue: "
        "if more items remain, call the tool again for the NEXT item now; "
        "otherwise use another available tool to keep making progress. "
        "Only stop and ask the user when nothing remains that can be done without this exact declined call."
    )

    def __init__(
        self,
        conversation_id: Optional[str] = None,
        callbacks: Optional[Dict[str, Callable]] = None,
        auto_approve: bool = False,
        **kwargs,
    ):
        super().__init__(conversation_id=conversation_id, callbacks=callbacks, **kwargs)
        self._wrapped_tools_cache: Dict[int, BaseTool] = {}
        self._auto_approve = auto_approve

    def get_tools(self) -> List[BaseTool]:
        return []

    def get_system_prompt(self) -> str:
        return ''

    @classmethod
    def _build_blocked_tool_message(cls, action_label: str) -> str:
        return cls.BLOCKED_TOOL_MESSAGE.format(action_label=action_label)

    @classmethod
    def _build_blocked_tool_result_payload(
        cls,
        *,
        action_label: str,
        tool_name: str,
        toolkit_name: Optional[str] = None,
        toolkit_type: Optional[str] = None,
        user_feedback: str = '',
    ) -> dict[str, Any]:
        # Slim, structured shape. Only `type` (detection), the tool/toolkit
        # identities (pipeline-block message), `denial_reason`, and a single
        # `message` directive. No duplicate `guidance`; no unread status/
        # retry_allowed/equivalent_action fields — those were field bloat that
        # weak models choked on. `denial_reason` carries the user's typed note
        # when given, else a default.
        payload: dict[str, Any] = {
            'type': cls.BLOCKED_TOOL_RESULT_TYPE,
            'blocked_tool_name': tool_name,
            'denial_reason': user_feedback.strip() or cls.BLOCKED_TOOL_DEFAULT_REASON,
            'message': cls._build_blocked_tool_message(action_label),
        }
        if toolkit_name:
            payload['blocked_toolkit_name'] = toolkit_name
        if toolkit_type:
            payload['blocked_toolkit_type'] = toolkit_type
        return payload

    @classmethod
    def _build_blocked_tool_result(
        cls,
        *,
        action_label: str,
        tool_name: str,
        toolkit_name: Optional[str] = None,
        toolkit_type: Optional[str] = None,
        user_feedback: str = '',
    ) -> str:
        return json.dumps(
            cls._build_blocked_tool_result_payload(
                action_label=action_label,
                tool_name=tool_name,
                toolkit_name=toolkit_name,
                toolkit_type=toolkit_type,
                user_feedback=user_feedback,
            ),
            ensure_ascii=True,
            separators=(',', ':'),
        )

    @staticmethod
    def _get_tool_metadata_value(tool: BaseTool, *keys: str) -> Optional[str]:
        metadata = getattr(tool, 'metadata', None) or {}
        for key in keys:
            value = metadata.get(key)
            if value:
                return str(value)
        return None

    @staticmethod
    def _get_tool_function(tool: BaseTool) -> Callable:
        if hasattr(tool, 'func') and callable(tool.func):
            return tool.func
        if hasattr(tool, '_run') and callable(tool._run):
            return tool._run
        if callable(tool):
            return tool
        raise ValueError(f"Cannot extract callable from tool '{tool.name}'")

    @staticmethod
    def _get_async_tool_function(tool: BaseTool) -> Optional[Callable]:
        if hasattr(tool, 'coroutine') and callable(tool.coroutine):
            return tool.coroutine
        if tool.__class__._arun is not BaseTool._arun and hasattr(tool, '_arun') and callable(tool._arun):
            return tool._arun
        return None

    @staticmethod
    def _mask_sensitive_tool_args(value: Any, field_name: Optional[str] = None) -> Any:
        sensitive_markers = (
            'password',
            'secret',
            'token',
            'api_key',
            'apikey',
            'authorization',
            'credential',
            'private_key',
            'access_key',
            'client_secret',
            'refresh_token',
            'session_token',
            'cookie',
            'pat',
        )

        normalized_field_name = str(field_name or '').strip().lower()
        if normalized_field_name and any(
            re.search(rf'(?:^|[\W_]){re.escape(marker)}(?:$|[\W_])', normalized_field_name)
            for marker in sensitive_markers
        ):
            return '***'

        if isinstance(value, dict):
            return {
                key: SensitiveToolGuardMiddleware._mask_sensitive_tool_args(item, key)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [SensitiveToolGuardMiddleware._mask_sensitive_tool_args(item, field_name) for item in value]
        if isinstance(value, tuple):
            return tuple(SensitiveToolGuardMiddleware._mask_sensitive_tool_args(item, field_name) for item in value)

        return value

    def _build_sensitive_tool_context(self, tool_to_execute: BaseTool, tool_input: Any) -> Optional[dict[str, Any]]:
        toolkit_name = self._get_tool_metadata_value(tool_to_execute, 'toolkit_name')
        toolkit_type = self._get_tool_metadata_value(tool_to_execute, 'toolkit_type', 'type')
        resolved_tool_name = normalize_tool_name(
            self._get_tool_metadata_value(tool_to_execute, 'tool_name') or tool_to_execute.name
        )
        display_args: Any = tool_input

        if tool_to_execute.name == 'invoke_tool' and isinstance(tool_input, dict):
            toolkit_name = str(tool_input.get('toolkit') or '') or toolkit_name
            resolved_tool_name = normalize_tool_name(str(tool_input.get('tool') or '') or resolved_tool_name)
            display_args = tool_input.get('arguments', {}) or {}
            registry = getattr(tool_to_execute, 'registry', None)
            if registry and toolkit_name:
                toolkit_type = registry.get_toolkit_type(toolkit_name) or toolkit_type

        if not resolved_tool_name:
            return None

        toolkit_label = toolkit_name or toolkit_type or 'this toolkit'
        policy = get_sensitive_tool_policy(
            tool_name=resolved_tool_name,
            toolkit_identifiers=[toolkit_type, toolkit_name],
            toolkit_label=toolkit_label,
        )
        if not policy:
            return None

        return {
            'tool_name': resolved_tool_name,
            'toolkit_name': toolkit_name,
            'toolkit_type': toolkit_type,
            'action_label': policy['action_name'],
            'policy_message': policy['policy_message'],
            'tool_args': self._mask_sensitive_tool_args(display_args),
            'tool_args_raw': display_args,
        }

    def _review_sensitive_tool_call(self, sensitive_tool_context: dict[str, Any]) -> dict[str, str]:
        # Auto-approve when the project secret allows it (API-only flows).
        if self._auto_approve:
            logger.info(
                "[HITL] Auto-approving '%s' (auto_approve_sensitive_actions enabled)",
                sensitive_tool_context['tool_name'],
            )
            return {'action': 'approve', 'value': ''}

        # Capture intermediate messages accumulated by __perform_tool_calling
        # before the interrupt.  These will be stored in the checkpoint and
        # injected back into graph state on HITL resume so the LLM retains
        # awareness of all previously completed tool calls.
        from ..tools.llm import _PENDING_TOOL_MESSAGES
        pending_msgs = _PENDING_TOOL_MESSAGES.get([])
        serialized_pending: list[dict] = []
        if pending_msgs:
            for msg in pending_msgs:
                try:
                    serialized_pending.append(message_to_dict(msg))
                except Exception:
                    pass
            if serialized_pending:
                logger.info(
                    "[HITL] Captured %d intermediate messages for checkpoint restore",
                    len(serialized_pending),
                )

        interrupt_payload = {
            'type': 'hitl',
            'guardrail_type': 'sensitive_tool',
            'node_name': 'sensitive_tool_guard',
            'message': sensitive_tool_context['policy_message'],
            'available_actions': ['approve', 'reject', 'block_with_comment'],
            'routes': {},
            'tool_name': sensitive_tool_context['tool_name'],
            'toolkit_name': sensitive_tool_context['toolkit_name'],
            'toolkit_type': sensitive_tool_context['toolkit_type'],
            'action_label': sensitive_tool_context['action_label'],
            'tool_args': sensitive_tool_context['tool_args'],
            'tool_args_raw': sensitive_tool_context.get('tool_args_raw', sensitive_tool_context['tool_args']),
            'policy_message': sensitive_tool_context['policy_message'],
        }

        # Store pending messages in interrupt payload (internal only). This field
        # is explicitly filtered/stripped by downstream interrupt/UI handling
        # logic to ensure it is never exposed to end users.
        if serialized_pending:
            interrupt_payload['_pending_messages'] = serialized_pending

        resume_value = interrupt(interrupt_payload)
        if not isinstance(resume_value, dict):
            # Non-dict resume payloads are treated as approve.  Each sensitive
            # tool call is reviewed independently — approving one invocation
            # never carries over to a later call of the same tool (issue
            # #5245: prompt on EVERY sensitive invocation).
            return {'action': 'approve', 'value': str(resume_value or '')}

        action = str(resume_value.get('action', 'approve')).strip().lower()
        value = str(resume_value.get('value', '') or '')

        # 'block_with_comment' (issue #5318) is a reject that carries a free-text
        # note. Map it onto the reject execution path so the tool is blocked; the
        # note rides `value` -> the blocked-tool result's `denial_reason`, the
        # AI-native signal the model reads to decide its next step. Any other
        # unrecognized action falls back to approve (the historical default for
        # malformed/legacy resume payloads).
        if action in self.BLOCK_WITH_COMMENT_ACTIONS:
            action = 'reject'
        elif action not in {'approve', 'reject'}:
            action = 'approve'

        if len(value) > self.MAX_COMMENT_LEN:
            value = value[:self.MAX_COMMENT_LEN]

        return {
            'action': action,
            'value': value,
        }

    def _could_be_sensitive(self, tool: BaseTool) -> bool:
        """Quick check whether a tool could match any sensitive tools config entry."""
        tool_name = normalize_tool_name(self._get_tool_metadata_value(tool, 'tool_name') or tool.name)
        toolkit_name = self._get_tool_metadata_value(tool, 'toolkit_name')
        toolkit_type = self._get_tool_metadata_value(tool, 'toolkit_type', 'type')
        identifiers = [i for i in [toolkit_type, toolkit_name] if i]
        if find_sensitive_tool_match(tool_name, identifiers) is not None:
            return True
        # invoke_tool (lazy mode) can invoke any tool dynamically
        if tool.name == 'invoke_tool':
            return True
        return False

    @staticmethod
    def _rebind_invoke_after_copy(copied: BaseTool) -> None:
        """Re-bind any instance-level ``invoke`` method to the new copy.

        ``_patch_tool_invoke`` uses ``types.MethodType`` to bind a metadata-
        forwarding wrapper to a specific tool instance.  ``model_copy()``
        preserves this bound method in ``__dict__``, but it remains bound to
        the *original* instance — so ``invoke()`` on the copy routes execution
        back to the original, bypassing ``_run``/``func`` patches on the copy.
        """
        if 'invoke' in copied.__dict__:
            stale = copied.__dict__['invoke']
            if callable(stale) and hasattr(stale, '__func__'):
                object.__setattr__(
                    copied, 'invoke',
                    types.MethodType(stale.__func__, copied),
                )

    def wrap_tool(self, tool: BaseTool) -> BaseTool:
        if not has_sensitive_tools_config():
            return tool

        if not self._could_be_sensitive(tool):
            return tool

        cache_key = id(tool)
        if cache_key in self._wrapped_tools_cache:
            return self._wrapped_tools_cache[cache_key]

        # Use model_copy to preserve the exact tool type, class, and all attributes.
        # Only the execution functions (_run/_arun or func/coroutine) are replaced.
        guard = self
        original_tool = tool

        if isinstance(tool, StructuredTool) and hasattr(tool, 'func'):
            copied = tool.model_copy()
            self._rebind_invoke_after_copy(copied)
            original_func = tool.func
            original_coroutine = getattr(tool, 'coroutine', None)

            def guarded_func(*args: Any, **kwargs: Any) -> Any:
                tool_input = normalize_tool_input(args, kwargs)
                ctx = guard._build_sensitive_tool_context(original_tool, tool_input)
                if ctx:
                    review = guard._review_sensitive_tool_call(ctx)
                    if review['action'] == 'reject':
                        return guard._build_blocked_tool_result(
                            action_label=ctx['action_label'],
                            tool_name=ctx['tool_name'],
                            toolkit_name=ctx['toolkit_name'],
                            toolkit_type=ctx['toolkit_type'],
                            user_feedback=review['value'],
                        )
                return original_func(*args, **kwargs)

            copied.func = guarded_func

            if original_coroutine is not None:
                async def guarded_coroutine(*args: Any, **kwargs: Any) -> Any:
                    tool_input = normalize_tool_input(args, kwargs)
                    ctx = guard._build_sensitive_tool_context(original_tool, tool_input)
                    if ctx:
                        review = guard._review_sensitive_tool_call(ctx)
                        if review['action'] == 'reject':
                            return guard._build_blocked_tool_result(
                                action_label=ctx['action_label'],
                                tool_name=ctx['tool_name'],
                                toolkit_name=ctx['toolkit_name'],
                                toolkit_type=ctx['toolkit_type'],
                                user_feedback=review['value'],
                            )
                    return await original_coroutine(*args, **kwargs)

                copied.coroutine = guarded_coroutine

            self._wrapped_tools_cache[cache_key] = copied
            return copied

        if isinstance(tool, Application):
            copied = tool.model_copy()
            self._rebind_invoke_after_copy(copied)
            original_invoke = tool.invoke
            original_ainvoke = tool.ainvoke

            def guarded_invoke(input: Any, config: Any = None, **kwargs: Any) -> Any:
                ctx = guard._build_sensitive_tool_context(original_tool, input)
                if ctx:
                    review = guard._review_sensitive_tool_call(ctx)
                    if review['action'] == 'reject':
                        return guard._build_blocked_tool_result(
                            action_label=ctx['action_label'],
                            tool_name=ctx['tool_name'],
                            toolkit_name=ctx['toolkit_name'],
                            toolkit_type=ctx['toolkit_type'],
                            user_feedback=review['value'],
                        )
                return original_invoke(input, config=config, **kwargs)

            async def guarded_ainvoke(input: Any, config: Any = None, **kwargs: Any) -> Any:
                ctx = guard._build_sensitive_tool_context(original_tool, input)
                if ctx:
                    review = guard._review_sensitive_tool_call(ctx)
                    if review['action'] == 'reject':
                        return guard._build_blocked_tool_result(
                            action_label=ctx['action_label'],
                            tool_name=ctx['tool_name'],
                            toolkit_name=ctx['toolkit_name'],
                            toolkit_type=ctx['toolkit_type'],
                            user_feedback=review['value'],
                        )
                return await original_ainvoke(input, config=config, **kwargs)

            copied.invoke = guarded_invoke
            copied.ainvoke = guarded_ainvoke
            self._wrapped_tools_cache[cache_key] = copied
            return copied

        # Generic BaseTool subclass (e.g. BaseAction): patch _run/_arun on a copy
        copied = tool.model_copy()
        self._rebind_invoke_after_copy(copied)
        original_run = tool._run
        original_async_func = self._get_async_tool_function(tool)
        run_accepts_run_manager = "run_manager" in inspect.signature(original_run).parameters

        def guarded_run(*args: Any, run_manager: Any = None, **kwargs: Any) -> Any:
            tool_input = normalize_tool_input(args, kwargs)
            ctx = guard._build_sensitive_tool_context(original_tool, tool_input)
            if ctx:
                review = guard._review_sensitive_tool_call(ctx)
                if review['action'] == 'reject':
                    return guard._build_blocked_tool_result(
                        action_label=ctx['action_label'],
                        tool_name=ctx['tool_name'],
                        toolkit_name=ctx['toolkit_name'],
                        toolkit_type=ctx['toolkit_type'],
                        user_feedback=review['value'],
                    )
            if run_accepts_run_manager:
                return original_run(*args, run_manager=run_manager, **kwargs)
            return original_run(*args, **kwargs)

        # Set as instance attribute to shadow the class method
        copied._run = guarded_run

        if original_async_func is not None:
            arun_accepts_run_manager = "run_manager" in inspect.signature(original_async_func).parameters

            async def guarded_arun(*args: Any, run_manager: Any = None, **kwargs: Any) -> Any:
                tool_input = normalize_tool_input(args, kwargs)
                ctx = guard._build_sensitive_tool_context(original_tool, tool_input)
                if ctx:
                    review = guard._review_sensitive_tool_call(ctx)
                    if review['action'] == 'reject':
                        return guard._build_blocked_tool_result(
                            action_label=ctx['action_label'],
                            tool_name=ctx['tool_name'],
                            toolkit_name=ctx['toolkit_name'],
                            toolkit_type=ctx['toolkit_type'],
                            user_feedback=review['value'],
                        )
                if arun_accepts_run_manager:
                    return await original_async_func(*args, run_manager=run_manager, **kwargs)
                return await original_async_func(*args, **kwargs)

            copied._arun = guarded_arun

        self._wrapped_tools_cache[cache_key] = copied
        return copied
