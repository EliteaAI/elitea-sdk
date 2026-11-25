"""Retry of throttled Microsoft Graph calls, shared by the SharePoint wrappers.

It sits apart from ``library_metadata`` and ``graph_wrapper``, which import each other's
neighbours, so every wrapper can import it at module level.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable

import requests

logger = logging.getLogger(__name__)

MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 1.0
MAX_RETRY_AFTER_SECONDS = 60.0
RETRYABLE_STATUS = frozenset({429, 503, 504})
TRANSIENT_ERRORS = (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError)


def _header_seconds(header: str) -> float | None:
    """The wait of a Retry-After header in seconds, or None when it is not a number.

    The HTTP-date form is not supported; the caller then falls back to the backoff.
    """
    try:
        return float(header)
    except ValueError:
        return None


def _retry_after_seconds(resp: requests.Response, attempt: int) -> float:
    header = (resp.headers or {}).get("Retry-After")
    seconds = _header_seconds(header) if header else None
    if seconds is not None:
        return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)
    return min(BACKOFF_BASE_SECONDS * (2 ** attempt), MAX_RETRY_AFTER_SECONDS)


def send_with_retry(send: Callable[[], requests.Response],
                    sleep: Callable[[float], None] | None = None) -> requests.Response:
    """Call ``send`` until it answers with something other than a throttling status.

    A timeout or a dropped connection is retried the same way. When the attempts
    run out, a transient error is raised and a throttling answer is returned, so
    the caller decides what a final 429 or 503 means.
    """
    sleep = sleep or time.sleep
    attempt = 0
    while True:
        try:
            resp = send()
        except TRANSIENT_ERRORS:
            if attempt >= MAX_RETRIES:
                raise
            sleep(min(BACKOFF_BASE_SECONDS * (2 ** attempt), MAX_RETRY_AFTER_SECONDS))
            attempt += 1
            continue
        if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
            delay = _retry_after_seconds(resp, attempt)
            logger.info("[SP-library] Graph answered %s; retrying in %.1fs (attempt %d)",
                        resp.status_code, delay, attempt + 1)
            # a streamed answer holds its connection until it is closed
            close = getattr(resp, "close", None)
            if callable(close):
                close()
            sleep(delay)
            attempt += 1
            continue
        return resp
