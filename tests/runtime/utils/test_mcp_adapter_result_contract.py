"""Compatibility tests for MCP results returned through ``UnifiedMcpClient``."""

import asyncio
import builtins
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent
from langchain_mcp_adapters import tools as mcp_tools

from elitea_sdk.runtime.utils.mcp_adapter import UnifiedMcpClient


def _configure_legacy_tool_loader(monkeypatch, client, result):
    """Let this contract test exercise both the old and the new implementation."""

    tool = SimpleNamespace(
        name="create_issue",
        ainvoke=AsyncMock(return_value=result),
    )
    monkeypatch.setattr(
        mcp_tools,
        "load_mcp_tools",
        AsyncMock(return_value=[tool]),
    )
    client._client = SimpleNamespace(connections={client._server_name: {}})


def test_call_tool_preserves_mcp_result_shape_and_strips_none_arguments(
    monkeypatch,
):
    """Structured content and error state must survive the compatibility layer."""

    expected_result = CallToolResult(
        content=[TextContent(type="text", text="validation failed")],
        structuredContent={"field": "project", "reason": "required"},
        isError=True,
    )
    session = SimpleNamespace(call_tool=AsyncMock(return_value=expected_result))
    client = UnifiedMcpClient(url="https://mcp.example.test/mcp")
    client._session = session
    _configure_legacy_tool_loader(monkeypatch, client, expected_result)

    result = asyncio.run(
        client.call_tool(
            "create_issue",
            {"project": None, "summary": "Missing project"},
        )
    )

    session.call_tool.assert_awaited_once_with(
        "create_issue",
        {"summary": "Missing project"},
    )
    assert result == {
        "content": [{"type": "text", "text": "validation failed"}],
        "structuredContent": {"field": "project", "reason": "required"},
        "isError": True,
    }


def test_call_tool_accepts_mapping_results_from_compatible_sessions(monkeypatch):
    """Custom MCP-compatible sessions may already return JSON-ready mappings."""

    expected_result = {
        "content": [{"type": "text", "text": "ok"}],
        "structuredContent": {"issue": 42},
    }
    session = SimpleNamespace(call_tool=AsyncMock(return_value=expected_result))
    client = UnifiedMcpClient(url="https://mcp.example.test/mcp")
    client._session = session
    _configure_legacy_tool_loader(monkeypatch, client, expected_result)

    result = asyncio.run(client.call_tool("create_issue", {"summary": "Example"}))

    assert result == expected_result


def test_connect_preserves_transitive_adapter_import_error(monkeypatch):
    """An incompatible adapter dependency must not be reported as an absent adapter."""

    original_import = builtins.__import__

    def fail_adapter_import(name, *args, **kwargs):
        if name == "langchain_mcp_adapters.client":
            raise ImportError("cannot import name 'sentinel' from 'typing_extensions'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_adapter_import)
    client = UnifiedMcpClient(url="https://mcp.example.test/mcp")

    with pytest.raises(ImportError, match="typing_extensions"):
        asyncio.run(client._connect())


def test_connect_reports_missing_adapter_package(monkeypatch):
    original_import = builtins.__import__

    def fail_adapter_import(name, *args, **kwargs):
        if name == "langchain_mcp_adapters.client":
            error = ModuleNotFoundError("No module named 'langchain_mcp_adapters'")
            error.name = "langchain_mcp_adapters"
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_adapter_import)
    client = UnifiedMcpClient(url="https://mcp.example.test/mcp")

    with pytest.raises(ImportError, match="langchain-mcp-adapters is required"):
        asyncio.run(client._connect())
