"""Tests for RemoteSandbox HTTP client."""

import base64
import json
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from elitea_sdk.runtime.langchain.remote_sandbox import (
    RemoteSandbox,
    _MAX_RESPONSE_BYTES,
    _MAX_TIMEOUT_SECONDS,
)


@pytest.fixture
def sandbox():
    sb = RemoteSandbox(
        url="http://pylon_sandbox:8080/execute",
        auth_token="test-token",
        tenant_id="tenant-1",
    )
    yield sb


class MockResponse:
    def __init__(self, status, json_data=None, text_data="", raise_on_json=False):
        self.status = status
        self._json = json_data
        self._text = text_data
        self._raise_on_json = raise_on_json

    async def json(self):
        if self._raise_on_json:
            raise aiohttp.ContentTypeError(
                message=MagicMock(), history=(), request_info=MagicMock()
            )
        return self._json

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


from unittest.mock import MagicMock


class MockSession:
    def __init__(self, response):
        self._response = response
        self._last_json = None
        self._last_headers = None
        self.closed = False

    def post(self, url, json=None, headers=None):
        self._last_url = url
        self._last_json = json
        self._last_headers = headers
        return self._response

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_successful_execution(sandbox):
    response = MockResponse(200, {
        "success": True,
        "stdout": "hello\n",
        "stderr": None,
        "result": 42,
        "session_bytes": None,
        "session_metadata": None,
    })
    session = MockSession(response)
    sandbox._session = session

    result = await sandbox.execute("print('hello')", timeout_seconds=30)

    assert result.status == "success"
    assert result.stdout == "hello\n"
    assert "42" in str(result.result)
    assert result.execution_time > 0


@pytest.mark.asyncio
async def test_session_bytes_roundtrip(sandbox):
    raw_session = b"\x01\x02\x03\x04"
    encoded = base64.b64encode(raw_session).decode("ascii")

    response = MockResponse(200, {
        "success": True,
        "stdout": None,
        "stderr": None,
        "result": None,
        "session_bytes": encoded,
        "session_metadata": {"packages": ["numpy"]},
    })
    session = MockSession(response)
    sandbox._session = session

    result = await sandbox.execute(
        "x=1",
        session_bytes=raw_session,
        session_metadata={"packages": []},
    )

    assert result.session_bytes == raw_session
    assert result.session_metadata == {"packages": ["numpy"]}

    sent_body = session._last_json
    assert sent_body["session_bytes"] == encoded
    assert sent_body["session_metadata"] == {"packages": []}


@pytest.mark.asyncio
async def test_auth_failure(sandbox):
    response = MockResponse(401)
    session = MockSession(response)
    sandbox._session = session

    with pytest.raises(ValueError, match="Sandbox auth token invalid"):
        await sandbox.execute("x=1")


@pytest.mark.asyncio
async def test_service_unavailable_503_json(sandbox):
    response = MockResponse(503, {"stderr": "Too many concurrent executions, retry shortly"})
    session = MockSession(response)
    sandbox._session = session

    result = await sandbox.execute("x=1")

    assert result.status == "error"
    assert "retry shortly" in result.stderr


@pytest.mark.asyncio
async def test_service_unavailable_503_non_json(sandbox):
    response = MockResponse(503, raise_on_json=True)
    session = MockSession(response)
    sandbox._session = session

    result = await sandbox.execute("x=1")

    assert result.status == "error"
    assert "unavailable" in result.stderr.lower()


@pytest.mark.asyncio
async def test_execution_failure(sandbox):
    response = MockResponse(200, {
        "success": False,
        "stdout": None,
        "stderr": "NameError: name 'foo' is not defined",
        "result": None,
        "session_bytes": None,
        "session_metadata": None,
    })
    session = MockSession(response)
    sandbox._session = session

    result = await sandbox.execute("foo()")

    assert result.status == "error"
    assert "NameError" in result.stderr


@pytest.mark.asyncio
async def test_correct_headers_sent(sandbox):
    response = MockResponse(200, {"success": True})
    session = MockSession(response)
    sandbox._session = session

    await sandbox.execute("x=1", timeout_seconds=30)

    assert session._last_headers["X-Sandbox-Token"] == "test-token"
    assert session._last_headers["X-Tenant-ID"] == "tenant-1"


@pytest.mark.asyncio
async def test_timeout_capped_at_max(sandbox):
    response = MockResponse(200, {"success": True})
    session = MockSession(response)
    sandbox._session = session

    await sandbox.execute("x=1", timeout_seconds=120)

    assert session._last_json["timeout_seconds"] == int(_MAX_TIMEOUT_SECONDS)


@pytest.mark.asyncio
async def test_session_reused_across_calls(sandbox):
    response = MockResponse(200, {"success": True})
    session = MockSession(response)
    sandbox._session = session

    await sandbox.execute("x=1")
    await sandbox.execute("x=2")

    assert sandbox._session is session
    assert not session.closed
