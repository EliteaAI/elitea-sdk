"""Bound an oversized tool result before it reaches state / events / DB / LLM context.

One entry point (``bound_tool_result``) shared by the middleware wrapper, which
covers every wrapped tool on both the agent and pipeline paths, and by the tool
nodes, which backstop the tools the middleware never wraps. Idempotent, so both
layers together produce exactly one truncation marker (#6140).
"""
import logging
from typing import Any, Optional, Tuple

from .tool_outcome import ToolOutcome, ToolResultStatus, record_outcome
from .utils.trace_limits import (
    cap_tool_result_structure,
    cap_tool_result_text,
    estimate_chars,
    measure_ceiling,
    resolve_tool_result_limit,
    structure_is_bounded,
    text_is_bounded,
    tool_result_bounding_enabled,
)

logger = logging.getLogger(__name__)

# Deferred human-in-the-loop marker: a paused child returns this during parallel
# fan-out so the parent can raise one combined interrupt. Touching it loses the pause.
HITL_DEFERRED_KEY = '__hitl_deferred__'

# Matched by name so this module stays importable without langgraph, and so a
# ToolMessage/Command from any version is recognised.
_PASSTHROUGH_TYPE_NAMES = frozenset({'ToolMessage', 'Command', 'AIMessage', 'BaseMessage'})


def _is_control_flow(value: Any) -> bool:
    """Control flow, not payload - never inspect the content of these."""
    if isinstance(value, BaseException):
        return True
    if isinstance(value, dict) and value.get(HITL_DEFERRED_KEY):
        return True
    for klass in type(value).__mro__:
        if klass.__name__ in _PASSTHROUGH_TYPE_NAMES:
            return True
    return False


def bound_tool_result(
    value: Any,
    tool_name: Any = None,
    toolkit_type: Any = None,
) -> Tuple[Any, Optional[int]]:
    """Return ``(value, original_chars)``; ``original_chars`` is None when untouched."""
    if _is_control_flow(value):
        return value, None

    limit = resolve_tool_result_limit(toolkit_type)
    enabled = tool_result_bounding_enabled()

    if isinstance(value, tuple) and len(value) == 2:
        # content_and_artifact tool: bound the model-facing half only. The artifact
        # half is the raw payload heading for storage, which is where big data belongs.
        content, artifact = value
        bounded, original = bound_tool_result(content, tool_name, toolkit_type)
        return (bounded, artifact), original

    if structure_is_bounded(value) or text_is_bounded(value):
        return value, None

    size = estimate_chars(value, ceiling=limit)
    if size <= limit:
        return value, None

    if not enabled:
        # Off means off - but report what would have been cut, so the flag can be
        # tuned (and its impact judged) without enforcing anything.
        logger.info(
            "Tool result over limit but truncation disabled: tool=%s toolkit=%s "
            "chars=%s limit=%s", tool_name, toolkit_type, size, limit,
        )
        return value, None

    # Size for the marker, ceiling-bounded: an exact figure would cost a full walk of
    # a payload that may be hundreds of megabytes, which is the CPU-bound stall this
    # guard exists to prevent. Reported as a floor instead ("over N chars").
    original = estimate_chars(value, ceiling=measure_ceiling(limit))
    if isinstance(value, str):
        bounded = cap_tool_result_text(value, limit, tool_name, original)
    elif isinstance(value, (dict, list)):
        bounded = cap_tool_result_structure(value, limit, tool_name, original)
    else:
        # Not a shape we can cut without changing its type; leave it and say so.
        logger.warning(
            "Tool result over limit but not truncatable: tool=%s type=%s chars=%s",
            tool_name, type(value).__name__, original,
        )
        return value, None

    logger.info(
        "Tool result truncated: tool=%s toolkit=%s chars>=%s limit=%s",
        tool_name, toolkit_type, original, limit,
    )
    return bounded, original


def bound_and_record(
    value: Any,
    tool_name: Any = None,
    toolkit_type: Any = None,
) -> Any:
    """``bound_tool_result`` plus the #6164 envelope, fail-open on any error.

    A guard that can raise is worse than no guard: this runs on every tool call.
    """
    try:
        bounded, original = bound_tool_result(value, tool_name, toolkit_type)
    except Exception:  # pylint: disable=W0703
        logger.exception("Bounding tool result failed; passing it through unchanged")
        return value
    if original is None:
        return bounded
    record_outcome(ToolOutcome(
        status=ToolResultStatus.TRUNCATED,
        message=(
            f"Tool result was truncated: {original} chars exceeded the "
            f"{resolve_tool_result_limit(toolkit_type)}-char limit."
        ),
        tool_name=str(tool_name) if tool_name else None,
        toolkit_type=str(toolkit_type) if toolkit_type else None,
        truncated=True,
        original_size=original,
    ))
    return bounded


def toolkit_type_of(tool: Any) -> Optional[str]:
    metadata = getattr(tool, 'metadata', None) or {}
    value = metadata.get('toolkit_type')
    return value if isinstance(value, str) else None
