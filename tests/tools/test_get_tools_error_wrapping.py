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


def test_ado_repos_error_is_not_swallowed(monkeypatch):
    original_error = ToolException(
        "Failed to connect to Azure DevOps: TF401019: The Git repository does not exist"
    )
    monkeypatch.setitem(
        AVAILABLE_TOOLS, "ado_repos", {"get_tools": MagicMock(side_effect=original_error)}
    )

    with pytest.raises(ToolException) as excinfo:
        get_tools([_tool_config("ado_repos")], elitea=None, llm=None)

    assert excinfo.value is original_error


def test_ado_repos_generic_exception_is_wrapped(monkeypatch):
    monkeypatch.setitem(
        AVAILABLE_TOOLS,
        "ado_repos",
        {"get_tools": MagicMock(side_effect=TypeError("argument after ** must be a mapping"))},
    )

    with pytest.raises(ToolException, match="Error getting tools for ado_repos"):
        get_tools([_tool_config("ado_repos")], elitea=None, llm=None)


def test_azure_devops_repos_alias_routes_to_ado_repos(monkeypatch):
    original_error = ToolException("Auth token expired")
    producer = MagicMock(side_effect=original_error)
    monkeypatch.setitem(AVAILABLE_TOOLS, "ado_repos", {"get_tools": producer})

    with pytest.raises(ToolException) as excinfo:
        get_tools([_tool_config("azure_devops_repos")], elitea=None, llm=None)

    assert excinfo.value is original_error
    producer.assert_called_once()


def test_generic_exception_from_producer_is_wrapped(monkeypatch):
    monkeypatch.setitem(
        AVAILABLE_TOOLS,
        "fake_toolkit",
        {"get_tools": MagicMock(side_effect=RuntimeError("connection reset"))},
    )

    with pytest.raises(ToolException, match="Error getting tools for fake_toolkit"):
        get_tools([_tool_config()], elitea=None, llm=None)
