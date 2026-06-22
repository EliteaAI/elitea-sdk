"""
Reusable retry utilities using tenacity library.

This module provides standardized retry logic for different error types:
- Server errors (5xx, 429 rate limits)
- Volume/network errors (timeouts, connection issues)
- LLM errors (transient API failures)

Usage:
    from elitea_sdk.tools.utils.retry import (
        is_server_error_retriable,
        is_volume_error_retriable,
        is_llm_error_retriable,
        retry_on_server_error,
        retry_on_llm_error,
    )

    @retry_on_server_error(max_attempts=3)
    def my_api_call():
        ...
"""
import logging
from typing import Tuple

from tenacity import (
    retry,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
    retry_if_exception,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


def is_server_error_retriable(exception: BaseException) -> bool:
    """
    Check if exception is a retriable server error (5xx, 429).

    Handles:
    - httpx.RemoteProtocolError: Connection closed prematurely
    - httpx.HTTPStatusError: HTTP errors with 5xx/429 status
    - openai.APIStatusError: OpenAI API errors with 5xx/429 status
    - openai.APIConnectionError: Connection issues
    - String-based detection for wrapped exceptions

    Args:
        exception: The exception to check

    Returns:
        True if the error is retriable, False otherwise
    """
    try:
        import httpx
        if isinstance(exception, httpx.RemoteProtocolError):
            return True
        if isinstance(exception, httpx.HTTPStatusError):
            status = exception.response.status_code
            return 500 <= status < 600 or status == 429
    except ImportError:
        pass

    try:
        import openai
        if isinstance(exception, openai.APIConnectionError):
            return True
        if isinstance(exception, openai.APIStatusError):
            return 500 <= exception.status_code < 600 or exception.status_code == 429
    except ImportError:
        pass

    error_str = str(exception).lower()
    if any(f"{code}" in str(exception) for code in [429, 500, 502, 503, 504]):
        return True
    if "rate limit" in error_str or "too many requests" in error_str:
        return True
    if "internal" in error_str and "error" in error_str:
        return True
    if "service unavailable" in error_str:
        return True
    if "gateway" in error_str and ("timeout" in error_str or "bad" in error_str):
        return True

    return False


def is_volume_error_retriable(exception: BaseException) -> bool:
    """
    Check if error is volume/timeout/network related.

    These errors indicate the request was too large, took too long,
    or failed due to network issues. Used by bisection strategy to
    split large requests into smaller chunks.

    Args:
        exception: The exception to check

    Returns:
        True if the error is volume/network related, False otherwise
    """
    error_str = str(exception).lower()

    volume_patterns = ["timeout", "too large", "request entity too large", "payload too large"]
    if any(p in error_str for p in volume_patterns):
        return True

    network_patterns = [
        "response ended prematurely",
        "chunkedencodingerror",
        "connection reset",
        "connection aborted",
        "connection refused",
        "broken pipe",
        "protocol error",
        "incomplete read",
        "remote disconnected",
    ]
    if any(p in error_str for p in network_patterns):
        return True

    if "400" in str(exception) and ("size" in error_str or "large" in error_str):
        return True
    if "400" in str(exception):
        return True

    return False


def is_llm_error_retriable(exception: BaseException) -> bool:
    """
    Check if LLM error is retriable (5xx, 429, timeout).

    Args:
        exception: The exception to check

    Returns:
        True if the LLM error is retriable, False otherwise
    """
    error_str = str(exception).lower()

    if any(f"{code}" in str(exception) for code in [429, 500, 502, 503, 504]):
        return True
    if "internalservererror" in error_str:
        return True
    if "rate limit" in error_str or "too many requests" in error_str:
        return True
    if "timeout" in error_str or "timed out" in error_str:
        return True

    return False


def retry_on_server_error(
    max_attempts: int = 4,
    wait_seconds: Tuple[int, ...] = (1, 4, 10, 30),
    log: logging.Logger = None,
):
    """
    Decorator factory for retrying on server errors (5xx, 429).

    Args:
        max_attempts: Maximum number of attempts (including first try)
        wait_seconds: Tuple of wait times between retries
        log: Logger for retry messages (defaults to module logger)

    Returns:
        A tenacity retry decorator

    Usage:
        @retry_on_server_error(max_attempts=3)
        def my_api_call():
            ...
    """
    _log = log or logger
    wait_times = wait_seconds[: max_attempts - 1] if wait_seconds else (1,)
    return retry(
        retry=retry_if_exception(is_server_error_retriable),
        stop=stop_after_attempt(max_attempts),
        wait=wait_chain(*[wait_fixed(s) for s in wait_times]),
        before_sleep=before_sleep_log(_log, logging.WARNING),
        reraise=True,
    )


def retry_on_llm_error(
    max_attempts: int = 3,
    wait_seconds: Tuple[int, ...] = (1, 4, 10, 30),
    log: logging.Logger = None,
):
    """
    Decorator factory for retrying LLM calls on transient errors.

    Args:
        max_attempts: Maximum number of attempts (including first try)
        wait_seconds: Tuple of wait times between retries
        log: Logger for retry messages (defaults to module logger)

    Returns:
        A tenacity retry decorator

    Usage:
        @retry_on_llm_error(max_attempts=3)
        def my_llm_call():
            ...
    """
    _log = log or logger
    wait_times = wait_seconds[: max_attempts - 1] if wait_seconds else (1,)
    return retry(
        retry=retry_if_exception(is_llm_error_retriable),
        stop=stop_after_attempt(max_attempts),
        wait=wait_chain(*[wait_fixed(s) for s in wait_times]),
        before_sleep=before_sleep_log(_log, logging.WARNING),
        reraise=True,
    )


def retry_on_volume_error(
    max_attempts: int = 3,
    wait_seconds: Tuple[int, ...] = (1, 4, 10, 30),
    log: logging.Logger = None,
):
    """
    Decorator factory for retrying on volume/network errors.

    Args:
        max_attempts: Maximum number of attempts (including first try)
        wait_seconds: Tuple of wait times between retries
        log: Logger for retry messages (defaults to module logger)

    Returns:
        A tenacity retry decorator
    """
    _log = log or logger
    wait_times = wait_seconds[: max_attempts - 1] if wait_seconds else (1,)
    return retry(
        retry=retry_if_exception(is_volume_error_retriable),
        stop=stop_after_attempt(max_attempts),
        wait=wait_chain(*[wait_fixed(s) for s in wait_times]),
        before_sleep=before_sleep_log(_log, logging.WARNING),
        reraise=True,
    )
