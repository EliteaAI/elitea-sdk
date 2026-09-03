"""Tests for GitHub issue #6173: sandbox admission refusals must raise, not return.

Covers the gates (concurrency/memory), the typed timeout signal, the remote backend,
the pipeline path (FunctionTool never corrupts output_variables), the agent path
(swarm_handle_tool_errors degrades gracefully instead of aborting the turn), and the
middleware bypass (both sync and async wrappers propagate the raise).
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool

from elitea_sdk.runtime.exceptions import SandboxAdmissionRefused
from elitea_sdk.runtime.tool_outcome import ToolErrorClass, classify_tool_error, retriable_for
from elitea_sdk.runtime.tools.sandbox import PyodideSandboxTool
from elitea_sdk.runtime.langchain.pyodide_sandbox import CodeExecutionResult
from elitea_sdk.runtime.tools.function import FunctionTool, TOOL_OUTCOMES_KEY, LAST_TOOL_OUTCOME_KEY
from elitea_sdk.runtime.middleware.tool_exception_handler import (
    ToolExceptionHandlerMiddleware,
    swarm_handle_tool_errors,
)
from elitea_sdk.runtime.middleware.strategies import LoggingStrategy


def _make_result(*, status="success", stdout=None, stderr=None, result=None, timed_out=False,
                  infra_category=None):
    return CodeExecutionResult(
        status=status,
        stdout=stdout,
        stderr=stderr,
        result=result,
        execution_time=0.1,
        session_bytes=None,
        session_metadata=None,
        timed_out=timed_out,
        infra_category=infra_category,
    )


def _make_tool():
    """Construct via model_construct to skip Deno/Pydantic validation."""
    tool = PyodideSandboxTool.model_construct(
        stateful=False,
        timeout_seconds=30,
        session_bytes=None,
        session_metadata=None,
    )
    object.__setattr__(tool, '_sandbox', AsyncMock())
    return tool


def _run_coro(coro):
    """asyncio.run() unsets the thread's current loop on exit, breaking later
    get_event_loop() calls in other test files - use a loop we leave installed."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _run_arun(tool, fake_result=None, side_effect=None):
    if side_effect is not None:
        tool._sandbox.execute = AsyncMock(side_effect=side_effect)
    else:
        tool._sandbox.execute = AsyncMock(return_value=fake_result)
    with patch.object(
        type(tool), '_prepare_pyodide_input', return_value="<code>", create=True
    ), patch.object(
        type(tool), '_initialize_sandbox', return_value=None, create=True
    ):
        return _run_coro(tool._arun("<code>"))


class TestGatesRaise:
    def test_concurrency_gate_raises(self):
        tool = _make_tool()
        with patch('elitea_sdk.runtime.tools.sandbox._count_deno_processes', return_value=16), \
             patch.object(type(tool), '_sandbox_limits', {
                 "max_concurrent": 16, "memory_pressure_pct": 0,
                 "timeout_seconds": 30, "wasm_max_mem_mb": 512,
             }, create=True):
            with pytest.raises(SandboxAdmissionRefused) as exc_info:
                _run_arun(tool)
        assert exc_info.value.provider_error_category == "service_busy"
        assert "Sandbox busy" in str(exc_info.value)
        assert exc_info.value.retry_after == 5.0

    def test_memory_gate_raises(self):
        tool = _make_tool()
        with patch('elitea_sdk.runtime.tools.sandbox._count_deno_processes', return_value=0), \
             patch('elitea_sdk.runtime.tools.sandbox._cgroup_memory_pressure_pct', return_value=90.0), \
             patch.object(type(tool), '_sandbox_limits', {
                 "max_concurrent": 0, "memory_pressure_pct": 85,
                 "timeout_seconds": 30, "wasm_max_mem_mb": 512,
             }, create=True):
            with pytest.raises(SandboxAdmissionRefused) as exc_info:
                _run_arun(tool)
        assert exc_info.value.provider_error_category == "out_of_memory"
        assert exc_info.value.retry_after == 10.0

    def test_sync_run_with_gate_tripped_raises_not_returns_string(self):
        """Regression for the easiest way to ship this fix broken: the sync _run catch-all."""
        tool = _make_tool()
        with patch('elitea_sdk.runtime.tools.sandbox._count_deno_processes', return_value=16), \
             patch.object(type(tool), '_sandbox_limits', {
                 "max_concurrent": 16, "memory_pressure_pct": 0,
                 "timeout_seconds": 30, "wasm_max_mem_mb": 512,
             }, create=True), \
             patch.object(type(tool), '_prepare_pyodide_input', return_value="<code>", create=True), \
             patch.object(type(tool), '_initialize_sandbox', return_value=None, create=True):
            tool._sandbox.execute = AsyncMock()
            with pytest.raises(SandboxAdmissionRefused):
                tool._run("<code>")


class TestTimeoutRaises:
    def test_timed_out_result_raises(self):
        tool = _make_tool()
        fake_result = _make_result(status="error", stderr="Execution timed out after 30 seconds",
                                    timed_out=True)
        with pytest.raises(SandboxAdmissionRefused) as exc_info:
            _run_arun(tool, fake_result)
        assert exc_info.value.provider_error_category == "timeout"

    def test_remote_infra_category_raises(self):
        tool = _make_tool()
        fake_result = _make_result(status="error", stderr="Sandbox service unavailable",
                                    infra_category="service_busy")
        with pytest.raises(SandboxAdmissionRefused) as exc_info:
            _run_arun(tool, fake_result)
        assert exc_info.value.provider_error_category == "service_busy"


class TestGenuineErrorRegression:
    def test_user_code_error_still_returns_dict(self):
        tool = _make_tool()
        fake_result = _make_result(status="error", stderr="NameError: name 'x' is not defined")
        result_dict = _run_arun(tool, fake_result)
        assert result_dict.get("error") == "NameError: name 'x' is not defined"
        assert result_dict.get("status") == "Execution failed"


class TestClassification:
    @pytest.mark.parametrize("category", ["service_busy", "out_of_memory", "timeout"])
    def test_classifies_as_infrastructure_and_retriable(self, category):
        exc = SandboxAdmissionRefused("boom", category)
        error_class = classify_tool_error(exc)
        assert error_class is ToolErrorClass.INFRASTRUCTURE
        assert retriable_for(error_class) is True


class TestPipelineNonCorruption:
    def test_function_tool_never_writes_output_variables_on_refusal(self):
        def _boom(code: str) -> str:
            raise SandboxAdmissionRefused("Sandbox busy: retry shortly", "service_busy")

        stub_tool = StructuredTool.from_function(func=_boom, name="pyodide_sandbox", description="stub")
        stub_tool.metadata = {}

        node = FunctionTool.model_construct(
            name="code_node",
            tool=stub_tool,
            output_variables=["result_var"],
            input_variables=["code"],
            input_mapping={"code": {"type": "fixed", "value": "code"}},
            structured_output=False,
            debug=False,
        )

        result = node.invoke({"code": "1 + 1"})

        assert "messages" in result
        assert "result_var" not in result
        assert result[LAST_TOOL_OUTCOME_KEY]["status"] == "error"
        assert result[LAST_TOOL_OUTCOME_KEY]["error_class"] == "infrastructure"
        assert result[TOOL_OUTCOMES_KEY]["code_node"]["status"] == "error"


class TestAgentPathGracefulDegradation:
    def test_swarm_handle_tool_errors_returns_message_not_raises(self):
        exc = SandboxAdmissionRefused("Sandbox busy: retry shortly", "service_busy")
        message = swarm_handle_tool_errors(exc)
        assert "Sandbox busy" in message


class TestMiddlewareBypass:
    def _middleware(self):
        return ToolExceptionHandlerMiddleware(strategies=[LoggingStrategy()])

    def test_sync_wrapper_propagates_raise(self):
        middleware = self._middleware()

        def boom(*args, **kwargs):
            raise SandboxAdmissionRefused("Sandbox busy: retry shortly", "service_busy")

        tool = MagicMock()
        tool.name = "pyodide_sandbox"
        tool.response_format = "content"
        wrapped = middleware._sync_wrapper(tool, boom)

        with pytest.raises(SandboxAdmissionRefused):
            wrapped(code="1 + 1")

    def test_async_wrapper_propagates_raise(self):
        middleware = self._middleware()

        async def boom(*args, **kwargs):
            raise SandboxAdmissionRefused("Sandbox busy: retry shortly", "service_busy")

        tool = MagicMock()
        tool.name = "pyodide_sandbox"
        tool.response_format = "content"
        wrapped = middleware._async_wrapper(tool, boom)

        with pytest.raises(SandboxAdmissionRefused):
            _run_coro(wrapped(code="1 + 1"))
