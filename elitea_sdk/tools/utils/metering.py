"""Tool Spend metering (#6980): wrap each paid provider attempt in ``metered_call`` (check → call → record)."""
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Optional

import requests
from langchain_core.tools import ToolException

logger = logging.getLogger(__name__)

CODE_LIMIT = "tool_spend_limit_reached"
CODE_DISABLED = "tool_spend_feature_disabled"
CODE_UNAVAILABLE = "tool_spend_unavailable"

EXEMPT_TTL_SECONDS = 600
RECORD_ATTEMPTS = 3
RECORD_BACKOFF_SECONDS = (0.25, 0.75)

_exempt_lock = threading.Lock()
_exempt_until: dict = {}
_sleep = time.sleep


class ToolSpendDenied(ToolException):
    """The platform refused a paid tool call; the run continues, only this tool is refused."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class MeteredCall:
    """Per-block state the caller fills in: units consumed and extra counters for the ledger."""

    def __init__(self):
        self.units: float = 0
        self.ok: bool = True
        self.meta: dict = {}
        self.permit: Optional[str] = None
        self.exempt: bool = False


def _credential_ids(credential: Any):
    get = credential.get if isinstance(credential, dict) else lambda key: getattr(credential, key, None)
    return get("configuration_uuid"), get("configuration_project_id")


def _remember_exempt(uuid: str, exempt: bool):
    with _exempt_lock:
        if exempt:
            _exempt_until[uuid] = time.monotonic() + EXEMPT_TTL_SECONDS
        else:
            _exempt_until.pop(uuid, None)


def _known_exempt(uuid: str) -> bool:
    with _exempt_lock:
        return _exempt_until.get(uuid, 0) > time.monotonic()


def _unreachable(error: Exception) -> bool:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        # 404: an older platform without the endpoint; 5xx/503: down or not ready
        return error.response.status_code == 404 or error.response.status_code >= 500
    return isinstance(error, (requests.ConnectionError, requests.Timeout, ValueError))


def _label(feature: str) -> str:
    return feature.replace("_", " ").capitalize()


def _check(elitea, payload: dict, uuid: str, feature: str) -> dict:
    try:
        return elitea.check_tool_spend(payload)
    except Exception as error:  # pylint: disable=W0703
        if _unreachable(error):
            if _known_exempt(uuid):
                # Own credential: the user's money, so a missed record is acceptable
                logger.warning("Tool spend check unreachable, allowing cached exempt credential: %s", error)
                return {"allowed": True, "exempt": True, "permit": None}
            logger.warning("Tool spend check unreachable, denying: %s", error)
        else:
            logger.warning("Tool spend check rejected the request: %s", error)
        raise ToolSpendDenied(CODE_UNAVAILABLE, f"{_label(feature)} is temporarily unavailable.") from error


def _record(elitea, payload: dict):
    for attempt in range(RECORD_ATTEMPTS):
        try:
            elitea.record_tool_spend(payload)
            return
        except Exception as error:  # pylint: disable=W0703
            if attempt + 1 >= RECORD_ATTEMPTS:
                logger.warning("Tool spend record failed after %d attempts: %s", RECORD_ATTEMPTS, error)
                return
            _sleep(RECORD_BACKOFF_SECONDS[min(attempt, len(RECORD_BACKOFF_SECONDS) - 1)])


@contextmanager
def metered_call(elitea, feature: str, backend: str, op: str, credential: Any,
                 plan: Optional[dict] = None, tool_name: Optional[str] = None):
    """Check before, record after (also on failure). No-op without a platform client (CLI)."""
    call = MeteredCall()
    if elitea is None or not hasattr(elitea, "check_tool_spend"):
        yield call
        return
    #
    uuid, owner = _credential_ids(credential)
    base = {"feature": feature, "backend": backend, "op": op,
            "configuration_uuid": uuid, "configuration_project_id": owner}
    verdict = _check(elitea, dict(base, plan=plan or {}), uuid, feature)
    if not verdict.get("allowed"):
        raise ToolSpendDenied(verdict.get("code") or CODE_UNAVAILABLE,
                              verdict.get("message") or f"{_label(feature)} is not available.")
    call.permit, call.exempt = verdict.get("permit"), bool(verdict.get("exempt"))
    if uuid:
        _remember_exempt(uuid, call.exempt)
    #
    started = time.monotonic()
    try:
        yield call
    except BaseException:
        call.ok = False
        raise
    finally:
        if call.permit:
            _record(elitea, dict(
                base, permit=call.permit, units=call.units, ok=call.ok, tool_name=tool_name,
                duration_ms=int((time.monotonic() - started) * 1000), meta=call.meta,
            ))
