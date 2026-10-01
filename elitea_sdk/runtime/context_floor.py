"""Always-on context floor for the tool-calling loop (#5915).

``SummarizationMiddleware`` runs once per node invocation, *before* the tool
loop starts, and deliberately bails out on tool-related state to keep every
``tool_call``/``tool_result`` pair intact. So nothing guards the one place where
context actually grows: the N tool results appended *inside* a single turn. When
their combined size crosses the model's window, the next in-loop model call is
rejected and the turn is lost — the guard never had a chance to run.

This module is that guard: a deterministic, LLM-free compaction of the OLDEST
tool results of the current turn, applied before each follow-up model call. It
is intentionally NOT gated on any context-management setting. Bounding the size
of a request the process builds, serializes and ships is a safety floor, not a
context-quality preference — the toggles in ``ContextStrategyModal`` decide
whether context is *summarized*, not whether a turn is allowed to crash.

Pairing is preserved by construction: a ``ToolMessage`` is only ever emptied of
content, never removed, so its ``tool_call_id`` keeps answering its
``tool_call``. Content is replaced in place so message identity (and therefore
the ``_PENDING_TOOL_MESSAGES`` HITL bookkeeping that holds these objects) stays
valid.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

# Ceiling when the model's window is unknown. Any smaller fixed number is wrong
# for some model: 200k drops results a 1M-window model accepts, yet still misses
# a 128k overflow. Without a window the floor only bounds the resource cost of a
# runaway turn; a real overflow is left to the provider's rejection, which the
# reactive retry in ``LLMNode`` recovers from.
UNKNOWN_WINDOW_CEILING_TOKENS = 1_000_000
# A misconfigured tiny window must not turn the floor into a shredder that
# empties every tool result of every turn.
MIN_FLOOR_TOKENS = 16_000
# The most recent tool results are what the model is actually reasoning about,
# so they are compacted last — and only when nothing older is left to free.
KEEP_RECENT_TOOL_RESULTS = 2
# Present in a compacted result: makes compaction idempotent and greppable.
COMPACTED_SENTINEL = '[tool result dropped: context floor]'
# Model attributes that may carry a real input window. ``_elitea_context_window``
# is for the platform to stamp from an admin-set model ``context_window``.
_WINDOW_ATTRS = ('_elitea_context_window', 'max_input_tokens', 'context_window')
# Substring fallback for proxies and SDKs that raise a generic error.
_CONTEXT_OVERFLOW_PHRASES = (
    'context window', 'context_window', 'token limit', 'too long',
    'maximum context length', 'input is too long', 'exceeds the limit',
    'contextwindowexceedederror', 'max_tokens', 'content too large',
)


@dataclass
class Compaction:
    """What one compaction pass freed."""

    compacted: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def applied(self) -> bool:
        return self.compacted > 0

    @property
    def tokens_freed(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


def count_context_tokens(messages) -> int:
    """Estimated tokens for ``messages``, via the context manager's own counter.

    Uses ``_count_tokens_image_aware`` — the same counter the summarization
    trigger and cutoff loop use — so the floor and the middleware never disagree
    about how big a conversation is. Falls back to a character heuristic if the
    middleware cannot be imported, since accounting must never break a turn.
    """
    if not messages:
        return 0
    try:
        from .middleware.summarization.middleware import _count_tokens_image_aware
        return _count_tokens_image_aware(messages)
    except Exception:  # noqa: BLE001 - accounting must never break a turn
        return _count_tokens_fallback(messages)


def _count_tokens_fallback(messages) -> int:
    total = 0
    for message in messages:
        content = getattr(message, 'content', message)
        text = content if isinstance(content, str) else repr(content)
        total += math.ceil(len(text) / 4.0) + 3
    return total


def resolve_floor_tokens(llm_client: Any = None) -> int:
    """The token ceiling the tool loop is not allowed to build past.

    The model's real input window minus the output it may generate, when the
    model reports one; otherwise ``UNKNOWN_WINDOW_CEILING_TOKENS``.
    """
    model = getattr(llm_client, 'bound', llm_client)
    window = _first_positive_attr(model, _WINDOW_ATTRS) or _profile_tokens(model, 'max_input_tokens')
    if window is None:
        return UNKNOWN_WINDOW_CEILING_TOKENS
    reserved_output = _first_positive_attr(model, ('max_tokens',)) or 0
    return max(MIN_FLOOR_TOKENS, window - reserved_output)


def _first_positive_attr(obj: Any, names) -> Optional[int]:
    if obj is None:
        return None
    for name in names:
        try:
            value = int(getattr(obj, name, None) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _profile_tokens(model: Any, key: str) -> Optional[int]:
    """``BaseChatModel.profile`` is keyed by model name, so it is empty for proxy aliases."""
    profile = getattr(model, 'profile', None)
    if not isinstance(profile, dict):
        return None
    try:
        value = int(profile.get(key) or 0)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def compact_tool_results(
    messages: List[Any],
    *,
    floor_tokens: int,
    start: int = 0,
    keep_recent: int = KEEP_RECENT_TOOL_RESULTS,
    counter: Optional[Callable[[List[Any]], int]] = None,
    require_fit: bool = True,
) -> Compaction:
    """Free tool-result content, oldest first, until ``messages`` fits the floor.

    Args:
        messages: the live message list; compacted entries are edited in place.
        floor_tokens: token ceiling to get under.
        start: index the current turn begins at — nothing before it is touched,
            so prior turns and restored HITL history stay byte-identical.
        keep_recent: how many of the newest tool results to leave alone unless
            freeing everything older still is not enough.
        counter: token counter for a message list; the summarization trigger's
            own counter when one is active, so the two never disagree.
        require_fit: skip compaction when even compacting every result of this
            turn cannot get under ``floor_tokens``.

    Returns:
        A ``Compaction`` describing the pass (``applied`` False when the list
        already fitted, which is the common case).
    """
    counter = counter or count_context_tokens
    total = counter(messages)
    result = Compaction(tokens_before=total, tokens_after=total)
    if total <= floor_tokens:
        return result

    # A note costs ~85 tokens, so compacting a small result grows the request.
    savings = {
        index: _compaction_saving(messages[index], counter)
        for index in compactable_indexes(messages, start)
    }
    candidates = [index for index, saving in savings.items() if saving > 0]
    if not candidates:
        return result
    if require_fit and total - sum(savings[index] for index in candidates) > floor_tokens:
        # History this pass may not touch is over the floor on its own. Dropping
        # this turn's results cannot make the request fit; it only loses them
        # and sends the model back to re-call the same tools every step.
        logger.warning(
            "[CONTEXT FLOOR] %d tokens cannot fit %d by compacting this turn's "
            "tool results; sending unchanged", total, floor_tokens,
        )
        return result

    # Oldest first, and only reach into the recent tail once everything older is
    # already gone — the newest results are what the next model call reasons over.
    ordered = candidates[:-keep_recent] if keep_recent else list(candidates)
    ordered += candidates[len(ordered):]

    for index in ordered:
        if total <= floor_tokens:
            break
        _compact_in_place(messages[index], counter)
        total -= savings[index]
        result.compacted += 1

    result.tokens_after = total
    if result.applied:
        logger.warning(
            "[CONTEXT FLOOR] dropped %d older tool result(s) to fit %d tokens "
            "(%d -> ~%d); the model is told each one was dropped",
            result.compacted, floor_tokens, result.tokens_before, result.tokens_after,
        )
    return result


def _compaction_saving(message: Any, counter: Callable[[List[Any]], int]) -> int:
    original = counter([message])
    note = ToolMessage(content=_compaction_note(message, original), tool_call_id='saving')
    return original - counter([note])


def compactable_indexes(messages: List[Any], start: int) -> List[int]:
    """Indexes of this turn's tool results that still hold compactable content."""
    indexes = []
    for index in range(max(0, start), len(messages)):
        message = messages[index]
        if not _is_tool_message(message):
            continue
        content = getattr(message, 'content', None)
        if isinstance(content, str) and COMPACTED_SENTINEL in content:
            continue
        indexes.append(index)
    return indexes


def _is_tool_message(message: Any) -> bool:
    return (
        getattr(message, 'tool_call_id', None) is not None
        or getattr(message, 'type', None) == 'tool'
    )


def _compaction_note(message: Any, original_tokens: int) -> str:
    tool_name = getattr(message, 'name', None) or 'unknown'
    return (
        f"⚠️ {COMPACTED_SENTINEL}\n\n"
        f"The result of '{tool_name}' (~{original_tokens} tokens) was dropped to keep this "
        f"turn inside the model's context window. It is GONE, not summarized — do not "
        f"answer as if you had read it.\n\n"
        f"If you still need this data, call the tool again with narrower parameters "
        f"(a smaller range, a filter, fewer items) so the result fits."
    )


def _compact_in_place(message: Any, counter: Callable[[List[Any]], int] = None) -> None:
    """Replace a tool result with a note the model can act on.

    Edited in place: the object stays the one the tool loop and the pending-HITL
    contextvar already hold, and ``tool_call_id``/``status``/``artifact`` keep
    answering the tool call that produced it.
    """
    note = _compaction_note(message, (counter or count_context_tokens)([message]))
    try:
        message.content = note
    except Exception as exc:  # noqa: BLE001 - never break a turn over a note
        logger.debug("Could not compact tool result for '%s': %s", getattr(message, 'name', None), exc)


def shrink_after_overflow(
    messages: List[Any],
    *,
    start: int = 0,
    ratio: float = 0.5,
    counter: Optional[Callable[[List[Any]], int]] = None,
) -> Compaction:
    """Free tool-result content after the provider rejected the request again.

    The provider has proven the request too big, so every tool result of the
    current turn is fair game (``keep_recent=0``) and the pass frees what it can
    even when ``ratio`` is out of reach (``require_fit=False``).
    """
    counter = counter or count_context_tokens
    target = int(counter(messages) * max(0.1, min(ratio, 0.9)))
    return compact_tool_results(
        messages, floor_tokens=target, start=start, keep_recent=0,
        counter=counter, require_fit=False,
    )


def is_context_overflow(error: BaseException) -> bool:
    """True when ``error`` is the provider refusing an over-long request.

    Typed first — providers that ship a real exception class for this are not
    ambiguous — with the message-substring pass kept as the fallback for the
    proxies and SDKs that only raise a generic error with prose inside.
    """
    for name in ('ContextWindowExceededError', 'ContextWindowExceeded'):
        for module in ('litellm.exceptions', 'litellm'):
            try:
                exc_type = getattr(__import__(module, fromlist=[name]), name, None)
            except Exception:  # noqa: BLE001 - optional dependency
                continue
            if isinstance(exc_type, type) and isinstance(error, exc_type):
                return True
    message = str(error).lower()
    return any(phrase in message for phrase in _CONTEXT_OVERFLOW_PHRASES)
