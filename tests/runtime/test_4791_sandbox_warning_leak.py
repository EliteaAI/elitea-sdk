"""Tests for GitHub issue #4791: sandbox shows version-mismatch warnings on success.

Verifies that stderr content (e.g. RequestsDependencyWarning) is suppressed from
the user-facing result when execution succeeds (status == 'success'), and that
genuine errors still surface when status == 'error'.

Covers all three fix sites:
  - PyodideSandboxTool._arun  (sandbox.py — _run delegates to _arun)
  - PyodideSandboxTool._run   (pyodide_sandbox.py, sync path)
  - PyodideSandboxTool._arun  (pyodide_sandbox.py, async path)
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# sandbox.py tool
from elitea_sdk.runtime.tools.sandbox import PyodideSandboxTool as SandboxPyodideSandboxTool

# pyodide_sandbox.py tool (same exported name, different module)
from elitea_sdk.runtime.langchain.pyodide_sandbox import (
    CodeExecutionResult,
    PyodideSandboxTool as LangchainPyodideSandboxTool,
    SyncPyodideSandbox,
    PyodideSandbox,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WARNING_STDERR = (
    "RequestsDependencyWarning: urllib3 (1.26.14) or chardet (5.1.0) doesn't match "
    "a supported version!"
)
SUCCESS_STDOUT = "ok"


def _make_result(*, status, stdout=None, stderr=None, result=None):
    return CodeExecutionResult(
        status=status,
        stdout=stdout,
        stderr=stderr,
        result=result,
        execution_time=0.1,
        session_bytes=None,
        session_metadata=None,
    )


# ---------------------------------------------------------------------------
# Site 1: sandbox.py PyodideSandboxTool._arun
# (The _run method delegates straight to _arun via asyncio.run)
# ---------------------------------------------------------------------------

class TestSandboxPyToolWarningLeak:
    """Covers the fix in elitea_sdk/runtime/tools/sandbox.py."""

    def _make_tool(self):
        """Construct via model_construct to skip Deno/Pydantic validation."""
        tool = SandboxPyodideSandboxTool.model_construct(
            stateful=False,
            timeout_seconds=30,
            session_bytes=None,
            session_metadata=None,
        )
        # Attach a mock async sandbox
        mock_sandbox = AsyncMock()
        object.__setattr__(tool, '_sandbox', mock_sandbox)
        return tool

    def _run_arun(self, tool, fake_result):
        """Patch initialization and execute _arun with a fake result."""
        tool._sandbox.execute = AsyncMock(return_value=fake_result)
        with patch.object(
            type(tool), '_prepare_pyodide_input', return_value="<code>", create=True
        ), patch.object(
            type(tool), '_initialize_sandbox', return_value=None, create=True
        ):
            return asyncio.get_event_loop().run_until_complete(
                tool._arun("<code>")
            )

    def test_success_with_warning_stderr_has_no_error_key(self):
        """On success, stderr warnings must NOT appear in result_dict['error']."""
        tool = self._make_tool()
        fake_result = _make_result(
            status="success",
            stdout=SUCCESS_STDOUT,
            stderr=WARNING_STDERR,
        )
        result_dict = self._run_arun(tool, fake_result)

        assert "error" not in result_dict, (
            f"'error' key must be absent on success, got: {result_dict}"
        )
        assert result_dict.get("output") == SUCCESS_STDOUT

    def test_error_with_stderr_populates_error_key(self):
        """On failure, stderr must still appear in result_dict['error']."""
        tool = self._make_tool()
        fake_result = _make_result(
            status="error",
            stdout=None,
            stderr="NameError: name 'x' is not defined",
        )
        result_dict = self._run_arun(tool, fake_result)

        assert result_dict.get("error") == "NameError: name 'x' is not defined"
        assert result_dict.get("status") == "Execution failed"

    def test_success_no_stderr_has_no_error_key(self):
        """Baseline: clean success has neither error key nor output when stdout empty."""
        tool = self._make_tool()
        fake_result = _make_result(
            status="success",
            stdout=SUCCESS_STDOUT,
            stderr=None,
        )
        result_dict = self._run_arun(tool, fake_result)

        assert "error" not in result_dict
        assert result_dict.get("output") == SUCCESS_STDOUT


# ---------------------------------------------------------------------------
# Site 2: pyodide_sandbox.py PyodideSandboxTool._run (sync)
# ---------------------------------------------------------------------------

class TestLangchainSandboxSyncWarningLeak:
    """Covers the fix in elitea_sdk/runtime/langchain/pyodide_sandbox.py _run."""

    def _make_tool(self, fake_result):
        """Build a non-stateful tool with a mocked sync sandbox."""
        mock_sync = MagicMock(spec=SyncPyodideSandbox)
        mock_sync.execute.return_value = fake_result

        tool = LangchainPyodideSandboxTool.model_construct(
            name="python_code_sandbox",
            stateful=False,
            timeout_seconds=30,
        )
        object.__setattr__(tool, '_sync_sandbox', mock_sync)
        return tool

    def test_success_with_warning_stderr_returns_stdout(self):
        """On success, _run must return stdout, not the warning string."""
        fake_result = _make_result(
            status="success",
            stdout=SUCCESS_STDOUT,
            stderr=WARNING_STDERR,
        )
        tool = self._make_tool(fake_result)
        result = tool._run("<code>")

        assert result == SUCCESS_STDOUT, (
            f"Expected stdout on success, got: {result!r}"
        )
        assert "RequestsDependencyWarning" not in (result or "")

    def test_error_with_stderr_returns_error_string(self):
        """On failure, _run must return 'Error during execution: ...'."""
        error_msg = "SyntaxError: invalid syntax"
        fake_result = _make_result(
            status="error",
            stdout=None,
            stderr=error_msg,
        )
        tool = self._make_tool(fake_result)
        result = tool._run("<broken code>")

        assert "Error during execution" in result
        assert error_msg in result

    def test_success_no_stderr_returns_stdout(self):
        """Baseline: clean success without stderr still returns stdout."""
        fake_result = _make_result(
            status="success",
            stdout=SUCCESS_STDOUT,
            stderr=None,
        )
        tool = self._make_tool(fake_result)
        result = tool._run("<code>")

        assert result == SUCCESS_STDOUT


# ---------------------------------------------------------------------------
# Site 3: pyodide_sandbox.py PyodideSandboxTool._arun (async)
# ---------------------------------------------------------------------------

class TestLangchainSandboxAsyncWarningLeak:
    """Covers the fix in elitea_sdk/runtime/langchain/pyodide_sandbox.py _arun."""

    def _make_tool(self, fake_result):
        """Build a non-stateful tool with a mocked async sandbox."""
        mock_async = AsyncMock(spec=PyodideSandbox)
        mock_async.execute = AsyncMock(return_value=fake_result)

        tool = LangchainPyodideSandboxTool.model_construct(
            name="python_code_sandbox",
            stateful=False,
            timeout_seconds=30,
        )
        object.__setattr__(tool, '_sandbox', mock_async)
        return tool

    def test_async_success_with_warning_stderr_returns_stdout(self):
        """Async path: on success, _arun must return stdout."""
        fake_result = _make_result(
            status="success",
            stdout=SUCCESS_STDOUT,
            stderr=WARNING_STDERR,
        )
        tool = self._make_tool(fake_result)
        result = asyncio.get_event_loop().run_until_complete(
            tool._arun("<code>")
        )

        assert result == SUCCESS_STDOUT, (
            f"Expected stdout on success, got: {result!r}"
        )
        assert "RequestsDependencyWarning" not in (result or "")

    def test_async_error_with_stderr_returns_error_string(self):
        """Async path: on failure, _arun must return 'Error during execution: ...'."""
        error_msg = "RuntimeError: boom"
        fake_result = _make_result(
            status="error",
            stdout=None,
            stderr=error_msg,
        )
        tool = self._make_tool(fake_result)
        result = asyncio.get_event_loop().run_until_complete(
            tool._arun("<code>")
        )

        assert "Error during execution" in result
        assert error_msg in result

    def test_async_success_no_stderr_returns_stdout(self):
        """Async baseline: clean success returns stdout."""
        fake_result = _make_result(
            status="success",
            stdout=SUCCESS_STDOUT,
            stderr=None,
        )
        tool = self._make_tool(fake_result)
        result = asyncio.get_event_loop().run_until_complete(
            tool._arun("<code>")
        )

        assert result == SUCCESS_STDOUT
