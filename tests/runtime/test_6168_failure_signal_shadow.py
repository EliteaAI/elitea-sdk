"""Unit tests for #6168 shadow-mode failure detection (SDK side).

Covers the pure helpers (mcp_is_error, classify_provider_error_category), the
shared-vocabulary contract with provider_worker's table, and that each of the
three MCP call sites still returns exactly what it returned before when
isError is true — shadow mode must never change a tool's return value.
"""
import asyncio
import importlib.util
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elitea_sdk.runtime.tool_outcome import ToolErrorClass, classify_provider_error_category
from elitea_sdk.runtime.utils.failure_signals import mcp_is_error, log_shadow_failure
from elitea_sdk.runtime.tools.mcp_server_tool import McpServerTool
from elitea_sdk.runtime.tools.mcp_remote_tool import McpRemoteTool

_PROVIDER_WORKER_FAILURE_SIGNALS = (
    Path(__file__).resolve().parents[3]
    / "centry" / "pylon_indexer" / "plugins" / "provider_worker" / "utils" / "failure_signals.py"
)


class TestMcpIsError:
    @pytest.mark.parametrize("result,expected", [
        ({"isError": True}, True),
        ({"isError": False}, False),
        ({"content": []}, False),
        ({}, False),
        (None, False),
        ({"result": {"isError": True}}, True),
        ({"result": {"isError": False}}, False),
        ({"result": "not a dict"}, False),
        ("plain string result", False),
        (42, False),
    ])
    def test_dict_and_scalar_shapes(self, result, expected):
        assert mcp_is_error(result) is expected

    def test_object_with_is_error_attr_true(self):
        obj = MagicMock()
        obj.isError = True
        assert mcp_is_error(obj) is True

    def test_object_with_is_error_attr_false(self):
        obj = MagicMock()
        obj.isError = False
        assert mcp_is_error(obj) is False

    def test_object_without_is_error_attr(self):
        assert mcp_is_error(object()) is False


class TestClassifyProviderErrorCategory:
    @pytest.mark.parametrize("category,expected", [
        ("timeout", ToolErrorClass.INFRASTRUCTURE),
        ("timeout_error", ToolErrorClass.INFRASTRUCTURE),
        ("service_busy", ToolErrorClass.INFRASTRUCTURE),
        ("rate_limit", ToolErrorClass.INFRASTRUCTURE),
        ("out_of_memory", ToolErrorClass.INFRASTRUCTURE),
        ("killed", ToolErrorClass.INFRASTRUCTURE),
        ("terminated", ToolErrorClass.INFRASTRUCTURE),
        ("deadline_exceeded", ToolErrorClass.INFRASTRUCTURE),
        ("backoff_limit_exceeded", ToolErrorClass.INFRASTRUCTURE),
        ("scheduling_failed", ToolErrorClass.INFRASTRUCTURE),
        ("platform_upload_failed", ToolErrorClass.INFRASTRUCTURE),
        ("artifact_error", ToolErrorClass.INFRASTRUCTURE),
        ("invalid_input", ToolErrorClass.INPUT),
        ("input_error", ToolErrorClass.INPUT),
        ("resource_not_found", ToolErrorClass.INPUT),
        ("branch_not_found", ToolErrorClass.INPUT),
        ("repository_not_found", ToolErrorClass.INPUT),
        ("empty_repository", ToolErrorClass.INPUT),
        ("runtime_error", ToolErrorClass.TOOL_INTERNAL),
        ("training_failed", ToolErrorClass.TOOL_INTERNAL),
        ("inference_failed", ToolErrorClass.TOOL_INTERNAL),
        ("indexing_failed", ToolErrorClass.TOOL_INTERNAL),
        ("authentication_error", ToolErrorClass.POLICY),
    ])
    def test_full_vocabulary(self, category, expected):
        assert classify_provider_error_category(category) is expected

    @pytest.mark.parametrize("category", [None, "", "unknown_error", "totally_made_up"])
    def test_unmapped_or_absent_returns_none(self, category):
        assert classify_provider_error_category(category) is None


class TestSharedVocabulary:
    """The SDK and provider_worker tables must describe the same categories.

    provider_worker/utils/failure_signals.py is stdlib-only, so it can be loaded
    by path from the SDK's own venv without any provider_worker dependency.
    """

    def _load_provider_table(self):
        # Sibling repo, not an SDK dependency: absent in a standalone SDK checkout
        # (e.g. CI), so skip rather than fail when it isn't there.
        if not _PROVIDER_WORKER_FAILURE_SIGNALS.exists():
            pytest.skip(f"provider_worker checkout not found at {_PROVIDER_WORKER_FAILURE_SIGNALS}")
        spec = importlib.util.spec_from_file_location(
            "provider_worker_failure_signals", _PROVIDER_WORKER_FAILURE_SIGNALS
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.PROVIDER_CATEGORY_CLASSES, module.TERMINAL_STATUSES, module.FAILURE_STATUSES

    def test_category_tables_are_identical(self):
        from elitea_sdk.runtime.tool_outcome import _PROVIDER_CATEGORY_CLASSES as sdk_table

        provider_table, _, _ = self._load_provider_table()
        sdk_as_strings = {k: v.value for k, v in sdk_table.items()}
        assert sdk_as_strings == provider_table

    def test_terminal_and_failure_statuses_include_failed(self):
        _, terminal, failure = self._load_provider_table()
        assert "Failed" in terminal
        assert "Failed" in failure
        assert "Error" in failure
        assert "Completed" in terminal
        assert "Completed" not in failure


class TestLogShadowFailure:
    def test_emits_stable_field_set_and_never_null_delivered_as_success(self, caplog):
        logger = logging.getLogger("test_6168_shadow")
        with caplog.at_level(logging.WARNING, logger="test_6168_shadow"):
            log_shadow_failure(logger, detected_by="mcp_is_error/remote", tool_name="foo", result_len=5)

        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert message.startswith("TOOL_FAILURE_SHADOW ")
        payload = json.loads(message[len("TOOL_FAILURE_SHADOW "):])
        assert payload["detected_by"] == "mcp_is_error/remote"
        assert payload["tool_name"] == "foo"
        assert payload["result_len"] == 5
        assert payload["delivered_as_success"] is True
        assert payload["provider_name"] is None
        assert payload["invocation_id"] is None

    def test_never_logs_a_payload_body(self, caplog):
        logger = logging.getLogger("test_6168_shadow_payload")
        with caplog.at_level(logging.WARNING, logger="test_6168_shadow_payload"):
            log_shadow_failure(logger, detected_by="mcp_is_error/stdio", tool_name="t", result_len=999999)

        message = caplog.records[0].getMessage()
        assert "999999" not in message.split("result_len")[0]
        payload = json.loads(message[len("TOOL_FAILURE_SHADOW "):])
        assert set(payload.keys()) == {
            "detected_by", "would_be_error_class", "provider_name", "toolkit_name",
            "toolkit_type", "toolkit_id", "tool_name", "error_category", "error_type",
            "invocation_id", "project_id", "user_id", "result_len", "delivered_as_success",
        }


class TestMcpServerToolProxiedSiteUnchanged:
    """Site 2 (platform-proxied): McpServerTool._run must return client.mcp_tool_call's
    result unmodified, whether or not it signals isError."""

    def _make_tool(self, client):
        return McpServerTool(
            name="do_thing",
            description="test tool",
            client=client,
            server="test-server",
            metadata={"toolkit_name": "jira", "toolkit_type": "mcp", "toolkit_id": 42},
        )

    def test_success_result_passthrough(self, caplog):
        client = MagicMock()
        client.mcp_tool_call.return_value = {"content": [{"type": "text", "text": "ok"}]}
        tool = self._make_tool(client)

        with caplog.at_level(logging.WARNING):
            result = tool._run(query="x")

        assert result == {"content": [{"type": "text", "text": "ok"}]}
        assert not any("TOOL_FAILURE_SHADOW" in r.message for r in caplog.records)

    def test_error_result_passthrough_and_logged(self, caplog):
        client = MagicMock()
        client.mcp_tool_call.return_value = {"isError": True, "content": [{"type": "text", "text": "boom"}]}
        tool = self._make_tool(client)

        with caplog.at_level(logging.WARNING):
            result = tool._run(query="x")

        assert result == {"isError": True, "content": [{"type": "text", "text": "boom"}]}
        shadow_lines = [r.getMessage() for r in caplog.records if "TOOL_FAILURE_SHADOW" in r.getMessage()]
        assert len(shadow_lines) == 1
        payload = json.loads(shadow_lines[0][len("TOOL_FAILURE_SHADOW "):])
        assert payload["detected_by"] == "mcp_is_error/proxied"
        assert payload["toolkit_name"] == "jira"
        assert payload["toolkit_type"] == "mcp"
        assert payload["toolkit_id"] == 42
        assert payload["tool_name"] == "do_thing"


class TestMcpRemoteToolRemoteSiteUnchanged:
    """Site 3 (remote HTTP/SSE): _execute_remote_tool's formatted return value must be
    unaffected by isError, and the exception-swallowing branch keeps its old message."""

    def _make_tool(self):
        return McpRemoteTool(
            name="do_thing",
            description="test tool",
            client=MagicMock(),
            server="unused",
            server_url="https://example.invalid/mcp",
            session_id="sess-1",
            original_tool_name="do_thing",
            metadata={"toolkit_name": "github", "toolkit_type": "mcp", "toolkit_id": 7},
        )

    def _run_with_mocked_client(self, tool, call_tool_return):
        mock_client = MagicMock()
        mock_client.initialize = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=call_tool_return)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("elitea_sdk.runtime.tools.mcp_remote_tool.McpClient", return_value=mock_client):
            return asyncio.run(tool._execute_remote_tool({}))

    def test_success_result_passthrough(self, caplog):
        tool = self._make_tool()
        with caplog.at_level(logging.WARNING):
            result = self._run_with_mocked_client(tool, {"content": [{"type": "text", "text": "ok"}]})

        assert result == "ok"
        assert not any("TOOL_FAILURE_SHADOW" in r.message for r in caplog.records)

    def test_error_result_passthrough_and_logged(self, caplog):
        tool = self._make_tool()
        with caplog.at_level(logging.WARNING):
            result = self._run_with_mocked_client(
                tool, {"isError": True, "content": [{"type": "text", "text": "boom"}]}
            )

        assert result == "boom"
        shadow_lines = [r.getMessage() for r in caplog.records if "TOOL_FAILURE_SHADOW" in r.getMessage()]
        assert len(shadow_lines) == 1
        payload = json.loads(shadow_lines[0][len("TOOL_FAILURE_SHADOW "):])
        assert payload["detected_by"] == "mcp_is_error/remote"
        assert payload["toolkit_name"] == "github"
        assert payload["toolkit_id"] == 7

    def test_exception_swallowed_branch_message_and_shadow_log_unchanged(self, caplog):
        tool = self._make_tool()

        with patch.object(McpRemoteTool, "_run_in_new_loop", side_effect=RuntimeError("connection refused")):
            with caplog.at_level(logging.WARNING):
                result = tool._run()

        assert result == "Error executing tool: connection refused"
        shadow_lines = [r.getMessage() for r in caplog.records if "TOOL_FAILURE_SHADOW" in r.getMessage()]
        assert len(shadow_lines) == 1
        payload = json.loads(shadow_lines[0][len("TOOL_FAILURE_SHADOW "):])
        assert payload["detected_by"] == "mcp_exception_swallowed"
        assert payload["toolkit_name"] == "github"


class TestMcpConfigStdioSiteUnchanged:
    """Site 1 (stdio): the local-config tool_func must format the CallToolResult the
    same way regardless of isError, and log a shadow line only when it is set."""

    def _make_result(self, is_error, text):
        result = MagicMock()
        result.isError = is_error
        content_item = MagicMock()
        content_item.text = text
        del content_item.data
        result.content = [content_item]
        return result

    def _call_tool_func(self, call_tool_return):
        from elitea_sdk.runtime.toolkits.mcp_config import _create_stdio_tool_func

        tool_func = _create_stdio_tool_func("do_thing", "playwright", {"command": "npx", "args": []})

        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=call_tool_return)
        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_stdio_cm = MagicMock()
        mock_stdio_cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("mcp.client.stdio.stdio_client", return_value=mock_stdio_cm), \
             patch("mcp.ClientSession", return_value=mock_session_cm):
            return tool_func()

    def test_success_result_passthrough(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = self._call_tool_func(self._make_result(False, "ok"))

        assert result == "ok"
        assert not any("TOOL_FAILURE_SHADOW" in r.message for r in caplog.records)

    def test_error_result_passthrough_and_logged(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = self._call_tool_func(self._make_result(True, "boom"))

        assert result == "boom"
        shadow_lines = [r.getMessage() for r in caplog.records if "TOOL_FAILURE_SHADOW" in r.getMessage()]
        assert len(shadow_lines) == 1
        payload = json.loads(shadow_lines[0][len("TOOL_FAILURE_SHADOW "):])
        assert payload["detected_by"] == "mcp_is_error/stdio"
        assert payload["toolkit_name"] == "playwright"
        assert payload["tool_name"] == "do_thing"
