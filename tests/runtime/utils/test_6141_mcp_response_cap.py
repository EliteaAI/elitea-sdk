"""Wire-level cap on MCP HTTP response bodies (#6141).

Two tiers, mirroring how the guard can fail:

* Tier 1 drives our counting stream directly through ``httpx.MockTransport``. It proves
  the abort works on both body paths and that peak allocation stays flat no matter how
  much the "server" claims to send. No sockets, so it is fast and deterministic.
* Tier 2 drives the *real* ``mcp`` streamable-http transport over a mock HTTP layer, to
  prove the two integration facts a unit test cannot: the abort surfaces as a size error
  rather than the session read timeout, and the oversized body is fetched exactly ONCE
  (mcp retries stream failures with Last-Event-ID, so a plain Exception would re-fetch).

Sizing cases use the catalogue sizes measured in docs/issues/6141_mcp_wire_cap_research.md.
Tests drive coroutines manually: this repo has no pytest-asyncio.
"""

import asyncio
import json

import httpx
import pytest

from elitea_sdk.runtime.utils import trace_limits
from elitea_sdk.runtime.utils.mcp_response_limit import (
    MCP_RESPONSE_MAX_BYTES,
    BoundedAsyncClient,
    McpResponseTooLarge,
    McpResponseTooLargeError,
    SizeTrip,
    build_httpx_client_factory,
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


CHUNK = b'x' * 65536


class _BodyStream(httpx.AsyncByteStream):
    """Serves a fixed body in chunks. A Response built with ``content=`` sets _content
    and aread() short-circuits without touching the stream, so the counter would never
    run - every sizing test must stream."""

    def __init__(self, body, chunk=65536):
        self._body = body
        self._chunk = chunk

    async def __aiter__(self):
        for i in range(0, len(self._body), self._chunk):
            yield self._body[i:i + self._chunk]

    async def aclose(self):
        return None


class _EndlessStream(httpx.AsyncByteStream):
    """A server that never stops sending. ``sent`` records what we actually allocated."""

    def __init__(self, hard_stop_chunks=2048):
        self.sent = 0
        self._hard_stop = hard_stop_chunks

    async def __aiter__(self):
        for _ in range(self._hard_stop):
            self.sent += len(CHUNK)
            yield CHUNK
        raise AssertionError('stream was not aborted - the cap did not fire')

    async def aclose(self):
        return None


def _client(limit, trip=None, handler=None, stream=None, content_type='application/json'):
    def _default_handler(request):
        return httpx.Response(200, headers={'content-type': content_type}, stream=stream)

    return BoundedAsyncClient(
        response_limit_bytes=limit,
        size_trip=trip or SizeTrip(),
        transport=httpx.MockTransport(handler or _default_handler),
    )


# ---------------------------------------------------------------------------
# Tier 1 - the counting stream
# ---------------------------------------------------------------------------


def test_oversized_json_body_is_aborted_and_allocation_stays_bounded():
    """The aread() path: streamable_http reads plain JSON replies this way."""
    limit = 256 * 1024
    stream = _EndlessStream()
    trip = SizeTrip()

    async def go():
        async with _client(limit, trip, stream=stream) as client:
            async with client.stream('GET', 'https://mcp.example.com/mcp') as response:
                await response.aread()

    with pytest.raises(McpResponseTooLarge):
        _run(go())

    assert trip.tripped
    assert trip.limit == limit
    # The whole point: we stopped one chunk past the cap instead of buffering the
    # gigabytes the server was willing to send.
    assert stream.sent <= limit + len(CHUNK)


def test_oversized_sse_body_is_aborted():
    """The aiter_sse() path drains the same stream, so one counter covers both."""
    limit = 128 * 1024
    stream = _EndlessStream()

    async def go():
        async with _client(limit, stream=stream, content_type='text/event-stream') as client:
            async with client.stream('GET', 'https://mcp.example.com/mcp') as response:
                async for _ in response.aiter_bytes():
                    pass

    with pytest.raises(McpResponseTooLarge):
        _run(go())
    assert stream.sent <= limit + len(CHUNK)


def test_under_limit_body_is_byte_identical():
    body = b'{"jsonrpc":"2.0","id":1,"result":{}}'

    async def go():
        async with _client(1024 * 1024, stream=_BodyStream(body)) as client:
            return await client.get('https://mcp.example.com/mcp')

    assert _run(go()).content == body


@pytest.mark.parametrize('size,label', [
    (76_259, 'notion 24 tools'),
    (253_989, 'github 89 tools'),
    (21_900_000, 'largest catalogue measured in production'),
])
def test_measured_real_catalogues_pass_the_default_cap(size, label):
    """Regression against tuning the default too tight: every tools/list size actually
    observed in the wild must still get through."""
    body = b'z' * size

    async def go():
        async with _client(MCP_RESPONSE_MAX_BYTES, stream=_BodyStream(body)) as client:
            return await client.get('https://mcp.example.com/mcp')

    assert len(_run(go()).content) == size, label


def test_abort_signal_escapes_mcp_reconnect_handlers():
    """mcp catches bare Exception around stream reads and then retries with
    Last-Event-ID. A cap that raised Exception would re-fetch the oversized body."""
    assert issubclass(McpResponseTooLarge, BaseException)
    assert not issubclass(McpResponseTooLarge, Exception)


def test_near_cap_response_is_logged_but_not_aborted(caplog):
    limit = 256 * 1024
    body = b'y' * int(limit * 0.85)

    async def go():
        async with _client(limit, stream=_BodyStream(body)) as client:
            return await client.get('https://mcp.example.com/mcp')

    with caplog.at_level('INFO'):
        assert len(_run(go()).content) == len(body)
    assert any('% of the byte cap' in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Tier 1 - the factory
# ---------------------------------------------------------------------------


def test_factory_disables_compression_so_the_cap_counts_payload_bytes():
    """A wire cap on compressed bytes is not a memory bound: measured MCP catalogues
    decode at up to ~19x, and the formats permit far worse."""
    client = build_httpx_client_factory(True, 1024, SizeTrip())()
    assert client.headers['accept-encoding'] == 'identity'


def test_factory_respects_a_caller_supplied_accept_encoding():
    client = build_httpx_client_factory(True, 1024, SizeTrip())(
        headers={'Accept-Encoding': 'gzip'},
    )
    assert client.headers['accept-encoding'] == 'gzip'


def test_factory_without_a_limit_is_a_plain_unbounded_client():
    """Flag off must be today's behaviour exactly - no wrapper, no header rewrite."""
    client = build_httpx_client_factory(True, None, SizeTrip())()
    assert type(client) is httpx.AsyncClient
    assert 'identity' not in client.headers.get('accept-encoding', '')


def test_factory_preserves_ssl_bypass_and_upstream_defaults():
    insecure = build_httpx_client_factory(False, 1024, SizeTrip())()
    assert insecure.follow_redirects is True
    pool = insecure._transport._pool
    assert pool._ssl_context.verify_mode.name == 'CERT_NONE'


def test_factory_still_honours_env_proxies(monkeypatch):
    """The regression this design exists to avoid: httpx only reads HTTP(S)_PROXY when
    ``transport is None`` (_client.py:1399), so bounding via a custom transport would
    silently break every deployment behind an egress proxy."""
    monkeypatch.setenv('HTTPS_PROXY', 'http://proxy.internal:3128')
    client = build_httpx_client_factory(True, 1024, SizeTrip())()
    assert any(mount is not None for mount in client._mounts.values())


# ---------------------------------------------------------------------------
# Tier 1 - error surfacing
# ---------------------------------------------------------------------------


def test_trip_message_names_the_limit_and_leaks_no_credentials():
    trip = SizeTrip()
    trip.record(1024, 2048, 'https://mcp.example.com')
    message = trip.message('search_issues')
    assert '1024' in message and 'search_issues' in message
    assert 'token' not in message and '?' not in message


def test_client_converts_a_trip_into_a_readable_error():
    from elitea_sdk.runtime.utils.mcp_adapter import UnifiedMcpClient

    client = UnifiedMcpClient(url='https://mcp.example.com/mcp', tool_name='search_issues')
    client._size_trip.record(MCP_RESPONSE_MAX_BYTES, MCP_RESPONSE_MAX_BYTES + 1, 'https://mcp.example.com')
    with pytest.raises(McpResponseTooLargeError) as excinfo:
        client._raise_if_size_tripped()
    assert 'search_issues' in str(excinfo.value)


def test_mcp_cap_defaults_to_off():
    try:
        trace_limits.configure_tool_result_limits(enabled=True)
        assert trace_limits.resolve_mcp_response_limit() is None
    finally:
        trace_limits.configure_tool_result_limits(enabled=True)


def test_either_flag_off_resolves_to_no_cap():
    try:
        # Master kill switch.
        trace_limits.configure_tool_result_limits(enabled=False)
        assert trace_limits.resolve_mcp_response_limit() is None
        # Narrow switch: char truncation stays on, only the wire cap goes away.
        trace_limits.configure_tool_result_limits(enabled=True, mcp_cap_enabled=False)
        assert trace_limits.resolve_mcp_response_limit() is None
        assert trace_limits.tool_result_bounding_enabled() is True
        trace_limits.configure_tool_result_limits(enabled=True, mcp_cap_enabled=True, mcp_response_bytes=123)
        assert trace_limits.resolve_mcp_response_limit() == 123
        # An unreadable value must fall back to the default, never to "no cap".
        trace_limits.configure_tool_result_limits(enabled=True, mcp_cap_enabled=True, mcp_response_bytes='nonsense')
        assert trace_limits.resolve_mcp_response_limit() == MCP_RESPONSE_MAX_BYTES
    finally:
        trace_limits.configure_tool_result_limits(enabled=True)


# ---------------------------------------------------------------------------
# Tier 2 - the real mcp streamable-http transport
# ---------------------------------------------------------------------------


class _MockMcpServer:
    """Speaks just enough streamable-http MCP to get to a tool call, then answers it
    with an SSE stream that never ends."""

    def __init__(self):
        self.tool_call_requests = 0
        self.oversized = _EndlessStream()

    def handler(self, request):
        if request.method == 'GET':
            return httpx.Response(405)  # decline the server-initiated stream

        payload = json.loads(request.content)
        method = payload.get('method')

        if method == 'initialize':
            result = {
                'jsonrpc': '2.0',
                'id': payload['id'],
                'result': {
                    'protocolVersion': payload['params']['protocolVersion'],
                    'capabilities': {'tools': {}},
                    'serverInfo': {'name': 'mock', 'version': '1.0'},
                },
            }
            return httpx.Response(
                200,
                headers={'content-type': 'application/json', 'mcp-session-id': 'sess-1'},
                stream=_BodyStream(json.dumps(result).encode()),
            )

        if method == 'tools/call':
            self.tool_call_requests += 1
            return httpx.Response(
                200,
                headers={'content-type': 'text/event-stream'},
                stream=self.oversized,
            )

        return httpx.Response(202)  # notifications


def test_real_transport_surfaces_size_error_and_fetches_the_body_once(monkeypatch):
    from elitea_sdk.runtime.utils import mcp_adapter

    limit = 256 * 1024
    server = _MockMcpServer()

    def fake_factory(ssl_verify, byte_limit, trip):
        def factory(headers=None, timeout=None, auth=None):
            return BoundedAsyncClient(
                response_limit_bytes=limit,
                size_trip=trip,
                transport=httpx.MockTransport(server.handler),
                headers=headers,
                timeout=timeout,
                auth=auth,
                follow_redirects=True,
            )
        return factory

    monkeypatch.setattr(mcp_adapter, 'build_httpx_client_factory', fake_factory)

    async def go():
        client = mcp_adapter.UnifiedMcpClient(
            url='https://mcp.example.com/mcp',
            transport='streamable_http',
            timeout=10,
            tool_name='big_tool',
        )
        # The pre-flight check uses aiohttp against the real network; not under test here.
        monkeypatch.setattr(client, '_preflight_auth_check', lambda: asyncio.sleep(0))
        async with client:
            await client.call_tool('big_tool', {})

    with pytest.raises(McpResponseTooLargeError) as excinfo:
        _run(go())

    assert 'big_tool' in str(excinfo.value)
    # mcp retries a failed SSE read with Last-Event-ID. If the abort were an ordinary
    # Exception we would re-download the oversized body up to three times.
    assert server.tool_call_requests == 1
    assert server.oversized.sent <= limit + len(CHUNK)
