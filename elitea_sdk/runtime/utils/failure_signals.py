"""Pure helpers for detecting MCP tool-call failures and the shared shadow-log line.

Stdlib-only. Counterpart to provider_worker/utils/failure_signals.py — same log
contract (see #6168), different detection mechanism (isError vs status/error_category).
"""
import json
from typing import Any

_SHADOW_FIELDS = (
    "detected_by", "would_be_error_class", "provider_name", "toolkit_name",
    "toolkit_type", "toolkit_id", "tool_name", "error_category", "error_type",
    "invocation_id", "project_id", "user_id", "result_len",
)


def mcp_is_error(result: Any) -> bool:
    """True if an MCP tool result signals isError, across the shapes seen in the wild.

    Tolerates a dict, an object exposing `.isError`, and a JSON-RPC envelope where
    `isError` sits at the top level or nested under `result`.
    """
    if result is None:
        return False
    if isinstance(result, dict):
        if result.get("isError"):
            return True
        nested = result.get("result")
        return bool(isinstance(nested, dict) and nested.get("isError"))
    return bool(getattr(result, "isError", False))


def mcp_error_message(result: Any) -> str:
    """Render an MCP failure result as the same text the success path would return (#6401)."""
    content = result.get("content") if isinstance(result, dict) else getattr(result, "content", None)
    if isinstance(content, list) and content:
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("data", item))))
            else:
                text = getattr(item, "text", None)
                data = getattr(item, "data", None)
                parts.append(str(text if text is not None else data if data is not None else item))
        return "\n".join(parts)
    text = result.get("text") if isinstance(result, dict) else getattr(result, "text", None)
    if text:
        return str(text)
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    return str(result)


SHADOW_LOGGED_ATTR = "_elitea_shadow_logged"


def mark_shadow_logged(exc):
    """Tag an exception whose failure was already shadow-logged, so a re-raise does not log it twice."""
    setattr(exc, SHADOW_LOGGED_ATTR, True)
    return exc


def is_shadow_logged(exc) -> bool:
    return bool(getattr(exc, SHADOW_LOGGED_ATTR, False))


def log_shadow_failure(logger, **fields) -> None:
    """Emit one TOOL_FAILURE_SHADOW line. Never logs a payload — only `result_len`.

    Missing fields are filled with None so every emitted line has the same key set
    across both the provider and MCP detection sites.
    """
    payload = {key: fields.get(key) for key in _SHADOW_FIELDS}
    # False once the site enforces the signal by raising instead of returning it (#6401).
    payload["delivered_as_success"] = bool(fields.get("delivered_as_success", True))
    logger.warning("TOOL_FAILURE_SHADOW %s", json.dumps(payload))
