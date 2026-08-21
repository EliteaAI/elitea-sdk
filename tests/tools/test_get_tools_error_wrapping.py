"""Regression tests for get_tools' generic toolkit-loader error handling
(elitea_sdk/tools/__init__.py, formerly line 290).

`except Exception as e: if isinstance(e, ToolException): raise` here is the
legitimate pattern — it re-raises an already-raised ToolException unchanged,
and wraps any other exception into a new ToolException. No conversion is
needed; this only closes the missing-coverage gap the plan flagged.
"""
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools import AVAILABLE_TOOLS, get_tools


def _tool_config(tool_type="fake_toolkit"):
    return {"type": tool_type, "settings": {"selected_tools": []}}


def test_tool_exception_from_producer_is_reraised_unchanged(monkeypatch):
    original_error = ToolException("Auth token expired")
    monkeypatch.setitem(
        AVAILABLE_TOOLS, "fake_toolkit", {"get_tools": MagicMock(side_effect=original_error)}
    )

    with pytest.raises(ToolException) as excinfo:
        get_tools([_tool_config()], elitea=None, llm=None)

    assert excinfo.value is original_error


def test_generic_exception_from_producer_is_wrapped(monkeypatch):
    monkeypatch.setitem(
        AVAILABLE_TOOLS,
        "fake_toolkit",
        {"get_tools": MagicMock(side_effect=RuntimeError("connection reset"))},
    )

    with pytest.raises(ToolException, match="Error getting tools for fake_toolkit"):
        get_tools([_tool_config()], elitea=None, llm=None)
