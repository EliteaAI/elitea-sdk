"""Typed outcome envelope for a single tool call, plus the one tool-error classifier.

Consumers (the LLM, the ToolMessage on the wire, a pipeline node's state) read these
fields instead of each re-deriving them by substring-matching an error string.
"""
import contextvars
from contextlib import contextmanager
from enum import Enum
from typing import Iterator, List, Optional

from pydantic import BaseModel


class ToolResultStatus(str, Enum):
    """str-valued so the enum survives safe_serialize and JSON without conversion."""

    SUCCESS = "success"
    ERROR = "error"
    TRUNCATED = "truncated"
    BLOCKED = "blocked"


class ToolErrorClass(str, Enum):
    INFRASTRUCTURE = "infrastructure"
    INPUT = "input"
    TOOL_INTERNAL = "tool_internal"
    POLICY = "policy"


class ToolOutcome(BaseModel):
    """What happened to one tool call. Only status and message are ever required."""

    status: ToolResultStatus
    message: str
    tool_name: Optional[str] = None
    error_class: Optional[ToolErrorClass] = None
    retriable: bool = False
    retry_after: Optional[float] = None
    exception_type: Optional[str] = None
    toolkit_type: Optional[str] = None
    truncated: bool = False
    original_size: Optional[int] = None


# Channel from the middleware that builds an outcome back to the caller that needs it.
# Caller-owned so a stale read is impossible: no sink installed, nothing recorded.
_OUTCOME_SINK: contextvars.ContextVar[Optional[List[ToolOutcome]]] = contextvars.ContextVar(
    '_tool_outcome_sink', default=None,
)


@contextmanager
def outcome_sink() -> Iterator[List[ToolOutcome]]:
    """Collect the outcomes recorded while running the block."""
    sink: List[ToolOutcome] = []
    token = _OUTCOME_SINK.set(sink)
    try:
        yield sink
    finally:
        _OUTCOME_SINK.reset(token)


def record_outcome(outcome: ToolOutcome) -> None:
    """No-op unless a caller installed a sink, which the agent tool loop never does."""
    sink = _OUTCOME_SINK.get()
    if sink is not None:
        sink.append(outcome)


# Matched by isinstance. ValueError covers pydantic ValidationError, which subclasses it.
_INFRASTRUCTURE_TYPES = (TimeoutError, ConnectionError)
_INPUT_TYPES = (ValueError, TypeError)
_TOOL_INTERNAL_TYPES = (AttributeError, KeyError, IndexError, NotImplementedError)

# Matched by class name across the MRO, so httpx/openai/requests need not be importable here.
_INFRASTRUCTURE_NAMES = frozenset({
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
    "ReadError", "RemoteProtocolError", "ChunkedEncodingError", "SSLError", "ProxyError",
    "Timeout", "TimeoutException", "ConnectionError",
    "APIConnectionError", "APITimeoutError", "InternalServerError",
    "ServiceUnavailable", "GatewayTimeout", "BadGateway",
    # Transient and retrying does help, so these belong here rather than in POLICY.
    # RateLimitExceededException carries status 403, so the name must beat _POLICY_STATUSES.
    "RateLimitError", "TooManyRequests", "RateLimitExceededException",
})
_POLICY_NAMES = frozenset({
    "AuthenticationError", "PermissionDeniedError", "PermissionError",
    "Unauthorized", "Forbidden",
})

# Read off the exception when it carries one. Covers a whole family of API errors per
# entry, which per-library class names cannot: PyGithub raised BadCredentialsException
# (401) and went unclassified until this existed.
_STATUS_ATTRS = ("status", "status_code")
_POLICY_STATUSES = frozenset({401, 403})
# 429 spans a rate-limit window (waiting resolves it) and an exhausted plan quota (waiting
# does not); HTTP cannot tell them apart, so the common case wins and retriable stays coarse.
_INFRASTRUCTURE_STATUSES = frozenset({429, 500, 502, 503, 504})

# Last resort, and deliberately narrow: no bare status codes, which appear in ordinary
# payloads (a Jira key like PROJ-400, a byte count) far more often than in real signals.
_INFRASTRUCTURE_PHRASES = (
    "timed out", "timeout", "connection refused", "connection reset",
    "connection aborted", "connection error", "temporarily unavailable",
    "service unavailable", "rate limit", "too many requests",
)
_POLICY_PHRASES = (
    "unauthorized", "forbidden", "permission denied", "access denied",
    "authentication failed", "invalid credentials", "not authorized",
)

_MAX_CHAIN_DEPTH = 3


def _attr(obj: object, name: str) -> object:
    """Guarded because these are arbitrary third-party objects: a property that raises
    must not turn a handled tool error into a crash on the error path."""
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _status_of(exc: BaseException) -> Optional[int]:
    """An HTTP status carried on the exception, or on the response it wrapped."""
    candidates = []
    for holder in (exc, _attr(exc, "response")):
        candidates.extend(_attr(holder, attr) for attr in _STATUS_ATTRS)
    for value in candidates:
        # bool is an int subclass, and a `status=True` flag is not a status code.
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def _class_from_status(exc: BaseException) -> Optional[ToolErrorClass]:
    status = _status_of(exc)
    if status in _POLICY_STATUSES:
        return ToolErrorClass.POLICY
    if status in _INFRASTRUCTURE_STATUSES:
        return ToolErrorClass.INFRASTRUCTURE
    # 400/404/422 are deliberately absent: a 404 is as often a permissions artifact
    # (GitHub hides private repos behind it) as it is a bad argument.
    return None


def _class_from_exception(exc: BaseException) -> Optional[ToolErrorClass]:
    mro_names = {cls.__name__ for cls in type(exc).__mro__}
    if mro_names & _INFRASTRUCTURE_NAMES:
        return ToolErrorClass.INFRASTRUCTURE
    if mro_names & _POLICY_NAMES:
        return ToolErrorClass.POLICY
    from_status = _class_from_status(exc)
    if from_status is not None:
        return from_status
    if isinstance(exc, _INFRASTRUCTURE_TYPES):
        return ToolErrorClass.INFRASTRUCTURE
    if isinstance(exc, _TOOL_INTERNAL_TYPES):
        return ToolErrorClass.TOOL_INTERNAL
    if isinstance(exc, _INPUT_TYPES):
        return ToolErrorClass.INPUT
    return None


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """The exception, then the causes behind it, nearest first and depth-capped.

    Toolkits overwhelmingly `raise ToolException(f"... {e}")` inside an except block, which
    loses the type from the string but leaves __context__ set even without `from e`.
    """
    current, depth = exc, 0
    while current is not None and depth <= _MAX_CHAIN_DEPTH:
        yield current
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__suppress_context__:
            # `raise X from None`: the raiser judged the context misleading, and reading it
            # anyway would contradict them — often publishing retriable on a caller error.
            current = None
        else:
            current = current.__context__
        depth += 1


def classify_tool_error(exc: BaseException) -> Optional[ToolErrorClass]:
    """Classify a tool failure, or return None when nothing reliable says what it was.

    Shaped after runtime.exceptions.budget_exceeded_from: structured signal first,
    substring matching only as a fallback, and None rather than a guess — an
    unclassified error is strictly better than a wrong label a consumer branches on.
    """
    for link in _chain(exc):
        from_exception = _class_from_exception(link)
        if from_exception is not None:
            return from_exception

    text = str(exc).lower()
    if any(phrase in text for phrase in _POLICY_PHRASES):
        return ToolErrorClass.POLICY
    if any(phrase in text for phrase in _INFRASTRUCTURE_PHRASES):
        return ToolErrorClass.INFRASTRUCTURE
    return None


def retriable_for(error_class: Optional[ToolErrorClass]) -> bool:
    """Whether retrying identical input could ever succeed — not whether to retry now.
    Nothing in the SDK acts on this; it is published for consumers outside the epic."""
    return error_class is ToolErrorClass.INFRASTRUCTURE


# Kept identical to provider_worker/utils/failure_signals.py's PROVIDER_CATEGORY_CLASSES
# (see the shared-vocabulary test) — one vocabulary, no cross-plugin import.
_PROVIDER_CATEGORY_CLASSES = {
    "timeout": ToolErrorClass.INFRASTRUCTURE,
    "timeout_error": ToolErrorClass.INFRASTRUCTURE,
    "service_busy": ToolErrorClass.INFRASTRUCTURE,
    "rate_limit": ToolErrorClass.INFRASTRUCTURE,
    "out_of_memory": ToolErrorClass.INFRASTRUCTURE,
    "killed": ToolErrorClass.INFRASTRUCTURE,
    "terminated": ToolErrorClass.INFRASTRUCTURE,
    "deadline_exceeded": ToolErrorClass.INFRASTRUCTURE,
    "backoff_limit_exceeded": ToolErrorClass.INFRASTRUCTURE,
    "scheduling_failed": ToolErrorClass.INFRASTRUCTURE,
    "platform_upload_failed": ToolErrorClass.INFRASTRUCTURE,
    "artifact_error": ToolErrorClass.INFRASTRUCTURE,
    "invalid_input": ToolErrorClass.INPUT,
    "input_error": ToolErrorClass.INPUT,
    "resource_not_found": ToolErrorClass.INPUT,
    "branch_not_found": ToolErrorClass.INPUT,
    "repository_not_found": ToolErrorClass.INPUT,
    "empty_repository": ToolErrorClass.INPUT,
    "runtime_error": ToolErrorClass.TOOL_INTERNAL,
    "training_failed": ToolErrorClass.TOOL_INTERNAL,
    "inference_failed": ToolErrorClass.TOOL_INTERNAL,
    "indexing_failed": ToolErrorClass.TOOL_INTERNAL,
    "authentication_error": ToolErrorClass.POLICY,
}


def classify_provider_error_category(category: Optional[str]) -> Optional[ToolErrorClass]:
    """Map a provider's raw `error_category` string to a `ToolErrorClass`, or None.

    Same contract as `classify_tool_error`: unmapped or absent categories return
    None rather than a guess.
    """
    if category is None:
        return None
    return _PROVIDER_CATEGORY_CLASSES.get(category)

