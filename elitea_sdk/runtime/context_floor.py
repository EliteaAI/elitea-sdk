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
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Floor used when neither the model nor the conversation tells us anything
# better. Generous on purpose: this must never engage during ordinary work, only
# on the runaway fan-out the issue describes.
DEFAULT_FLOOR_TOKENS = 200_000
# A misconfigured tiny ``max_context_tokens`` must not turn the floor into a
# shredder that empties every tool result of every turn.
MIN_FLOOR_TOKENS = 16_000
# Headroom over the conversation's configured ``max_context_tokens`` when that is
# all we know about the window. Without it the floor would engage at exactly the
# summarization trigger, and since the floor runs inside the tool loop — before
# the next ``before_model`` — it would shrink the history back under the trigger
# and the summarizer would never fire again. The floor must sit BENEATH Tier 2,
# catching only what a single turn blows past, never racing it.
FLOOR_TRIGGER_HEADROOM = 3.0
# The most recent tool results are what the model is actually reasoning about,
# so they are compacted last — and only when nothing older is left to free.
KEEP_RECENT_TOOL_RESULTS = 2
# Present in a compacted result: makes compaction idempotent and greppable.
COMPACTED_SENTINEL = '[tool result dropped: context floor]'
# Model client attributes that may carry a real input window.
_WINDOW_ATTRS = ('max_input_tokens', 'context_window', 'max_context_tokens')


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


def resolve_floor_tokens(llm_client: Any = None, middleware_manager: Any = None) -> int:
    """The token ceiling the tool loop is not allowed to build past.

    Prefers a real input window advertised by the model client. Failing that it
    derives a ceiling from the conversation's ``max_context_tokens`` (read off the
    summarization trigger) plus ``FLOOR_TRIGGER_HEADROOM`` — a size hint, never an
    on/off gate and never the threshold itself. Otherwise the default.
    """
    window = _first_positive_attr(llm_client, _WINDOW_ATTRS)
    if window is not None:
        # A window the model actually advertises IS the hard ceiling.
        return max(MIN_FLOOR_TOKENS, int(window))

    trigger = _trigger_tokens(middleware_manager)
    if trigger is None:
        return DEFAULT_FLOOR_TOKENS
    # A summarization trigger is a preference, not a ceiling: leave room above it
    # so the configured summarization stays the thing that normally reclaims
    # context, and cap at the default so a large trigger cannot lift the floor
    # past what any real window is likely to be.
    return max(MIN_FLOOR_TOKENS, min(DEFAULT_FLOOR_TOKENS, int(trigger * FLOOR_TRIGGER_HEADROOM)))


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


def _trigger_tokens(middleware_manager: Any) -> Optional[int]:
    """``max_context_tokens`` as configured for this conversation, if any."""
    for middleware in getattr(middleware_manager, '_middleware', None) or []:
        trigger = getattr(middleware, 'trigger', None)
        for entry in (trigger if isinstance(trigger, list) else [trigger]):
            if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == 'tokens':
                try:
                    tokens = int(entry[1])
                except (TypeError, ValueError):
                    continue
                if tokens > 0:
                    return tokens
    return None


def compact_tool_results(
    messages: List[Any],
    *,
    floor_tokens: int,
    start: int = 0,
    keep_recent: int = KEEP_RECENT_TOOL_RESULTS,
) -> Compaction:
    """Free tool-result content, oldest first, until ``messages`` fits the floor.

    Args:
        messages: the live message list; compacted entries are edited in place.
        floor_tokens: token ceiling to get under.
        start: index the current turn begins at — nothing before it is touched,
            so prior turns and restored HITL history stay byte-identical.
        keep_recent: how many of the newest tool results to leave alone unless
            freeing everything older still is not enough.

    Returns:
        A ``Compaction`` describing the pass (``applied`` False when the list
        already fitted, which is the common case).
    """
    total = count_context_tokens(messages)
    result = Compaction(tokens_before=total, tokens_after=total)
    if total <= floor_tokens:
        return result

    candidates = _compactable_indexes(messages, start)
    if not candidates:
        return result

    # Oldest first, and only reach into the recent tail once everything older is
    # already gone — the newest results are what the next model call reasons over.
    ordered = candidates[:-keep_recent] if keep_recent else list(candidates)
    ordered += candidates[len(ordered):]

    for index in ordered:
        if total <= floor_tokens:
            break
        message = messages[index]
        before = count_context_tokens([message])
        _compact_in_place(message)
        total -= max(0, before - count_context_tokens([message]))
        result.compacted += 1

    result.tokens_after = total
    if result.applied:
        logger.warning(
            "[CONTEXT FLOOR] dropped %d older tool result(s) to fit %d tokens "
            "(%d -> ~%d); the model is told each one was dropped",
            result.compacted, floor_tokens, result.tokens_before, result.tokens_after,
        )
    return result


def _compactable_indexes(messages: List[Any], start: int) -> List[int]:
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


def _compact_in_place(message: Any) -> None:
    """Replace a tool result with a note the model can act on.

    Edited in place: the object stays the one the tool loop and the pending-HITL
    contextvar already hold, and ``tool_call_id``/``status``/``artifact`` keep
    answering the tool call that produced it.
    """
    tool_name = getattr(message, 'name', None) or 'unknown'
    original = count_context_tokens([message])
    note = (
        f"⚠️ {COMPACTED_SENTINEL}\n\n"
        f"The result of '{tool_name}' (~{original} tokens) was dropped to keep this "
        f"turn inside the model's context window. It is GONE, not summarized — do not "
        f"answer as if you had read it.\n\n"
        f"If you still need this data, call the tool again with narrower parameters "
        f"(a smaller range, a filter, fewer items) so the result fits."
    )
    try:
        message.content = note
    except Exception as exc:  # noqa: BLE001 - never break a turn over a note
        logger.debug("Could not compact tool result for '%s': %s", tool_name, exc)


def shrink_after_overflow(messages: List[Any], *, start: int = 0, ratio: float = 0.5) -> Compaction:
    """Free tool-result content after the provider already rejected the request.

    The floor works off an estimate, so a provider can still say no — the model's
    real window may be smaller than we assumed, or its tokenizer denser than our
    heuristic. At that point the estimate has been proven wrong, and the only
    useful move is to make the request decisively smaller rather than shave the
    single newest result and retry into the same wall.

    Every tool result of the current turn is fair game here (``keep_recent=0``):
    the retry is worth more than any one of them, and each compacted result still
    tells the model what it lost and how to fetch it again.
    """
    target = int(count_context_tokens(messages) * max(0.1, min(ratio, 0.9)))
    return compact_tool_results(messages, floor_tokens=target, start=start, keep_recent=0)


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
    return False
