"""Regression coverage for the MCP discovery hang incident.

Locks down the three behaviors requested in review of the discovery-hang fix:

1. Pre-flight fast-fail: an SSO/forward-auth proxy that answers an
   unauthenticated MCP POST with a 3xx redirect or a 200 ``text/html`` login
   page is surfaced as a ``ValueError`` instead of being followed into a page
   the real MCP client cannot parse (which used to hang ``initialize()``).

2. A stalled session ``__aenter__`` (transport connect / handshake that never
   resolves) is bounded by ``asyncio.wait_for`` in ``_connect`` and raises
   ``TimeoutError`` within roughly the configured timeout rather than hanging.

3. The synchronous discovery backstop, run repeatedly against a failing worker,
   drains and closes each worker loop so no pending asyncio tasks or
   "Task was destroyed but it is pending" warnings are left behind.

Tests drive coroutines manually: this repo has no pytest-asyncio.
"""

import asyncio
import gc
import logging
import time

import pytest

from elitea_sdk.runtime.utils.mcp_adapter import UnifiedMcpClient
from elitea_sdk.runtime.toolkits import mcp as mcp_toolkit
from elitea_sdk.runtime.toolkits.mcp import McpToolkit, _drain_and_close_loop
from elitea_sdk.runtime.models.mcp_models import McpConnectionConfig


def _run(coro):
    """Run a coroutine on a fresh event loop, draining it afterwards."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        _drain_and_close_loop(loop)


# ---------------------------------------------------------------------------
# Fake aiohttp plumbing for the pre-flight check
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status, headers):
        self.status = status
        self.headers = headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakePost:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeClientSession:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, *args, **kwargs):
        return _FakePost(self._response)


def _patch_aiohttp_response(monkeypatch, response):
    import aiohttp

    def _factory(*args, **kwargs):
        return _FakeClientSession(response)

    monkeypatch.setattr(aiohttp, "ClientSession", _factory)


# ---------------------------------------------------------------------------
# 1. Pre-flight fast-fail on redirect / HTML login page
# ---------------------------------------------------------------------------


def test_preflight_fast_fails_on_sso_redirect(monkeypatch):
    """A 302 to a login page must fail fast, not be followed into an HTML page."""
    _patch_aiohttp_response(
        monkeypatch,
        _FakeResponse(302, {"Location": "https://login.example.test/sso"}),
    )
    client = UnifiedMcpClient(url="https://mcp.example.test/mcp")

    with pytest.raises(ValueError, match="redirected"):
        _run(client._preflight_auth_check())


def test_preflight_fast_fails_on_html_login_page(monkeypatch):
    """A 200 that serves text/html (login proxy) must fail fast, not hang."""
    _patch_aiohttp_response(
        monkeypatch,
        _FakeResponse(200, {"Content-Type": "text/html; charset=utf-8"}),
    )
    client = UnifiedMcpClient(url="https://mcp.example.test/mcp")

    with pytest.raises(ValueError, match="HTML"):
        _run(client._preflight_auth_check())


# ---------------------------------------------------------------------------
# 2. Stalled session entry is bounded by the configured timeout
# ---------------------------------------------------------------------------


class _StallingSessionContext:
    """Session context whose __aenter__ never resolves on its own."""

    def __init__(self):
        self.exited = False

    async def __aenter__(self):
        await asyncio.sleep(3600)

    async def __aexit__(self, *exc):
        self.exited = True
        return False


class _StallingMultiClient:
    def __init__(self, config):
        self.context = _StallingSessionContext()

    def session(self, name):
        return self.context


def test_stalled_session_entry_returns_within_configured_bound(monkeypatch):
    """A handshake that never resolves must raise TimeoutError near `timeout`."""
    import langchain_mcp_adapters.client as lc_client

    monkeypatch.setattr(lc_client, "MultiServerMCPClient", _StallingMultiClient)

    async def _noop_preflight(self):
        return None

    monkeypatch.setattr(UnifiedMcpClient, "_preflight_auth_check", _noop_preflight)

    client = UnifiedMcpClient(url="https://mcp.example.test/mcp")
    client.timeout = 0.3  # keep the test fast; still exercises wait_for bound

    start = time.monotonic()
    with pytest.raises(TimeoutError, match="Timed out"):
        _run(client._connect())
    elapsed = time.monotonic() - start

    # Must return shortly after the configured bound, not hang indefinitely.
    assert elapsed < 5.0, f"connect took {elapsed:.2f}s, expected ~{client.timeout}s"
    # The half-opened session context is torn down on the timeout path.
    assert client._session_context is None


# ---------------------------------------------------------------------------
# 3a. The loop-drain helper cancels pending tasks and closes the loop
# ---------------------------------------------------------------------------


def test_drain_and_close_loop_cancels_pending_and_closes():
    loop = asyncio.new_event_loop()

    async def _never():
        await asyncio.sleep(3600)

    # Schedule a task, let it start, then leave it pending.
    task = loop.create_task(_never())
    loop.run_until_complete(asyncio.sleep(0))
    assert not task.done()

    _drain_and_close_loop(loop)

    assert loop.is_closed()
    assert task.cancelled() or task.done()


# ---------------------------------------------------------------------------
# 3b. Repeated synchronous failures leave no pending tasks / growth warnings
# ---------------------------------------------------------------------------


class _WarningCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def test_repeated_failures_leave_no_pending_tasks(monkeypatch):
    """The threaded backstop must drain each worker loop across repeated failures."""

    async def _failing_discover(cls, **kwargs):
        # Fail fast, but only after yielding so a real task exists on the
        # worker loop (mirrors an async connect that errors mid-flight).
        await asyncio.sleep(0)
        raise RuntimeError("simulated discovery failure")

    monkeypatch.setattr(
        McpToolkit, "_discover_tools_async", classmethod(_failing_discover)
    )

    connection_config = McpConnectionConfig(
        url="https://mcp.example.test/mcp",
        session_id="test-session",
    )

    # Capture asyncio's "Task was destroyed but it is pending" and any
    # loop-level exception-handler output on the root logger.
    capture = _WarningCapture()
    root_logger = logging.getLogger()
    asyncio_logger = logging.getLogger("asyncio")
    root_logger.addHandler(capture)
    asyncio_logger.addHandler(capture)

    async def _driver():
        # Running inside a live loop forces the ThreadPoolExecutor worker path.
        results = []
        for _ in range(5):
            try:
                McpToolkit._discover_tools_sync(
                    toolkit_name="stub",
                    toolkit_type="mcp",
                    connection_config=connection_config,
                    timeout=1,
                    ssl_verify=True,
                    oauth_token_injected=False,
                )
                results.append("ok")
            except RuntimeError:
                results.append("raised")
        return results

    try:
        results = _run(_driver())
    finally:
        root_logger.removeHandler(capture)
        asyncio_logger.removeHandler(capture)

    # Every iteration surfaced the worker failure (backstop did not swallow it).
    assert results == ["raised"] * 5

    # Give any un-drained tasks a chance to be GC'd and complain, then assert
    # the drain left nothing pending behind.
    gc.collect()
    pending_warnings = [
        msg for msg in capture.records
        if "was destroyed but it is pending" in msg
    ]
    assert not pending_warnings, f"leaked pending tasks: {pending_warnings}"
