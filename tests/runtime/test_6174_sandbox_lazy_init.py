"""Tests for GitHub issue #6174: the sandbox's Deno probe must be lazy.

Before the fix, PyodideSandboxTool.__init__ eagerly called _initialize_sandbox(),
which spawns a `deno --version` subprocess (matched by the concurrency gate's
`pgrep -c deno`). This meant every tool *construction* — even one never actually
invoked — paid the probe cost and could inflate the gate's count. The fix removes
the eager call so the existing `if self._sandbox is None` guards in `_run`/`_arun`
become the sole (lazy) trigger, on first real execution only.

These tests construct the tool via its real __init__ (unlike test_6173's suite,
which uses model_construct() and therefore never exercises this code path at all)
to prove construction stays lazy and first-use still initializes correctly.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from elitea_sdk.runtime.exceptions import SandboxAdmissionRefused
from elitea_sdk.runtime.tools.sandbox import PyodideSandboxTool, StatefulPyodideSandboxTool
from elitea_sdk.runtime.langchain.pyodide_sandbox import CodeExecutionResult


def _make_result(*, status="success", stdout=None, stderr=None, result=None):
    return CodeExecutionResult(
        status=status,
        stdout=stdout,
        stderr=stderr,
        result=result,
        execution_time=0.1,
        session_bytes=None,
        session_metadata=None,
    )


def _run_coro(coro):
    """asyncio.run() unsets the thread's current loop on exit, breaking later
    get_event_loop() calls in other test files - use a loop we leave installed."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class TestConstructionIsLazy:
    def test_init_does_not_call_deno_probe(self):
        with patch('elitea_sdk.runtime.tools.sandbox._is_deno_available') as mock_probe:
            tool = PyodideSandboxTool()
        mock_probe.assert_not_called()
        assert tool._sandbox is None

    def test_idle_binds_never_probe_regardless_of_count(self):
        """Regression for the 27-35% of tools that get bound but never invoked."""
        with patch('elitea_sdk.runtime.tools.sandbox._is_deno_available') as mock_probe:
            for _ in range(5):
                PyodideSandboxTool()
        mock_probe.assert_not_called()

    def test_stateful_variant_construction_is_also_lazy(self):
        with patch('elitea_sdk.runtime.tools.sandbox._is_deno_available') as mock_probe:
            tool = StatefulPyodideSandboxTool()
        mock_probe.assert_not_called()
        assert tool._sandbox is None
        assert tool.stateful is True


class TestFirstExecutionInitializes:
    def test_first_arun_call_triggers_init(self):
        tool = PyodideSandboxTool()
        assert tool._sandbox is None
        tool._sandbox_limits = {
            "max_concurrent": 16, "memory_pressure_pct": 0,
            "timeout_seconds": 30, "wasm_max_mem_mb": 512,
        }

        fake_sandbox = AsyncMock()
        fake_sandbox.execute = AsyncMock(return_value=_make_result(stdout="2"))

        def _fake_init():
            tool._sandbox = fake_sandbox

        with patch.object(type(tool), '_initialize_sandbox', side_effect=_fake_init, create=True) as mock_init, \
             patch.object(type(tool), '_prepare_pyodide_input', return_value="<code>", create=True), \
             patch('elitea_sdk.runtime.tools.sandbox._count_deno_processes', return_value=0):
            result = _run_coro(tool._arun("1 + 1"))

        mock_init.assert_called_once()
        assert tool._sandbox is fake_sandbox
        assert result["output"] == "2"

    def test_gate_checked_before_lazy_init_on_first_execution(self):
        """The tool's own first-execution probe must not be able to inflate its
        own gate check - locks in that the gate check (:441-457) runs before the
        lazy init call (:474-475)."""
        tool = PyodideSandboxTool()
        tool._sandbox_limits = {
            "max_concurrent": 16, "memory_pressure_pct": 0,
            "timeout_seconds": 30, "wasm_max_mem_mb": 512,
        }
        call_order = []

        def _fake_count(*args, **kwargs):
            call_order.append("gate")
            return 0

        def _fake_init():
            call_order.append("init")
            tool._sandbox = AsyncMock()
            tool._sandbox.execute = AsyncMock(return_value=_make_result(stdout="ok"))

        with patch('elitea_sdk.runtime.tools.sandbox._count_deno_processes', side_effect=_fake_count), \
             patch.object(type(tool), '_initialize_sandbox', side_effect=_fake_init, create=True), \
             patch.object(type(tool), '_prepare_pyodide_input', return_value="<code>", create=True):
            _run_coro(tool._arun("1 + 1"))

        assert call_order == ["gate", "init"]


class TestDenoUnavailableAfterConstruction:
    def test_sync_run_returns_friendly_message_not_raise(self):
        """Construction no longer fails hard when Deno is missing - the same
        RuntimeError now surfaces on first use instead, via the existing catch."""
        tool = PyodideSandboxTool()
        with patch('elitea_sdk.runtime.tools.sandbox._is_deno_available', return_value=False):
            result = tool._run("1 + 1")
        assert result.startswith("❌")
        assert "Deno" in result

    def test_arun_returns_error_dict_not_raise(self):
        tool = PyodideSandboxTool()
        with patch('elitea_sdk.runtime.tools.sandbox._is_deno_available', return_value=False), \
             patch('elitea_sdk.runtime.tools.sandbox._count_deno_processes', return_value=0):
            result = _run_coro(tool._arun("1 + 1"))
        assert "Deno is required" in result["error"]


class TestGateStillEnforcedAfterLazyInit:
    def test_genuinely_concurrent_execution_still_trips_gate(self):
        """The gate must keep protecting real load after this fix - it's only
        idle binds that should stop being counted, not real concurrency."""
        tool = PyodideSandboxTool()
        tool._sandbox_limits = {
            "max_concurrent": 4, "memory_pressure_pct": 0,
            "timeout_seconds": 30, "wasm_max_mem_mb": 512,
        }

        def _fake_init():
            tool._sandbox = AsyncMock()
            tool._sandbox.execute = AsyncMock(return_value=_make_result(stdout="ok"))

        with patch.object(type(tool), '_initialize_sandbox', side_effect=_fake_init, create=True), \
             patch.object(type(tool), '_prepare_pyodide_input', return_value="<code>", create=True), \
             patch('elitea_sdk.runtime.tools.sandbox._count_deno_processes', return_value=0):
            _run_coro(tool._arun("1 + 1"))  # first call: lazily initializes, succeeds
        assert tool._sandbox is not None

        with patch.object(type(tool), '_prepare_pyodide_input', return_value="<code>", create=True), \
             patch('elitea_sdk.runtime.tools.sandbox._count_deno_processes', return_value=4):
            with pytest.raises(SandboxAdmissionRefused) as exc_info:
                _run_coro(tool._arun("1 + 1"))
        assert exc_info.value.provider_error_category == "service_busy"
