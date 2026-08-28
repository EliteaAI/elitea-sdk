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


def log_shadow_failure(logger, **fields) -> None:
    """Emit one TOOL_FAILURE_SHADOW line. Never logs a payload — only `result_len`.

    Missing fields are filled with None so every emitted line has the same key set
    across both the provider and MCP detection sites.
    """
    payload = {key: fields.get(key) for key in _SHADOW_FIELDS}
    payload["delivered_as_success"] = True
    logger.warning("TOOL_FAILURE_SHADOW %s", json.dumps(payload))
