"""Wire-level cap on MCP HTTP response bodies: bytes past the cap are never read (#6141)."""

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Deliberately far above the #6140 character cap: this one protects process RAM,
# not the LLM context, and tool-discovery catalogues are legitimately large.
MCP_RESPONSE_MAX_BYTES = 50 * 1024 * 1024


class McpResponseTooLarge(BaseException):
    # BaseException on purpose: mcp catches Exception around stream reads and retries with
    # Last-Event-ID (streamable_http.py:428), which would re-fetch the oversized body.

    def __init__(self, limit: int, received: int, origin: str = ''):
        super().__init__(f'MCP response exceeded {limit} bytes (aborted at {received})')
        self.limit = limit
        self.received = received
        self.origin = origin


class McpResponseTooLargeError(Exception):
    """Surfaced to the caller once the aborted call fails."""


class SizeTrip:
    # Instance state, not a ContextVar: the sync path runs in a fresh thread loop where a
    # ContextVar set in the transport would not be visible.

    __slots__ = ('limit', 'received', 'origin', 'tripped')

    def __init__(self):
        self.tripped = False
        self.limit = 0
        self.received = 0
        self.origin = ''

    def record(self, limit: int, received: int, origin: str) -> None:
        if self.tripped:
            return
        self.tripped = True
        self.limit = limit
        self.received = received
        self.origin = origin

    def message(self, tool_name: Any = None) -> str:
        target = f" while calling '{tool_name}'" if tool_name else ''
        # Origin only, never the full URL: MCP URLs can carry tokens in the query.
        return (
            f'The MCP server returned a response larger than the {self.limit} byte limit'
            f'{target}; the read was aborted after {self.received} bytes and no result is '
            'available. Ask the server for less data (narrow filters, request fewer items, '
            'or use pagination) - do not retry the same call unchanged.'
        )


# Fractions of the cap at which a healthy response is still logged, so the default can
# be reviewed against real traffic instead of waiting for the first failure.
_NEAR_CAP_MARKS = (0.5, 0.8)


class _CountingStream(httpx.AsyncByteStream):
    # Per HTTP response, which matches every current MCP operation (one POST per JSON-RPC
    # message); rev 2026-07-28 subscriptions/listen would need per-message counting.

    def __init__(self, stream: Any, limit: int, trip: SizeTrip, origin: str, encoding: str = ''):
        self._stream = stream
        self._limit = limit
        self._trip = trip
        self._origin = origin
        self._encoding = encoding or 'identity'
        self._next_mark = 0

    async def __aiter__(self):
        total = 0
        async for chunk in self._stream:
            total += len(chunk)
            if total > self._limit:
                self._trip.record(self._limit, total, self._origin)
                logger.error(
                    'MCP response exceeded byte cap: origin=%s limit=%s received=%s encoding=%s',
                    self._origin, self._limit, total, self._encoding,
                )
                raise McpResponseTooLarge(self._limit, total, self._origin)
            self._log_near_cap(total)
            yield chunk

    def _log_near_cap(self, total: int) -> None:
        while self._next_mark < len(_NEAR_CAP_MARKS):
            mark = _NEAR_CAP_MARKS[self._next_mark]
            if total < self._limit * mark:
                return
            self._next_mark += 1
            logger.info(
                'MCP response passed %d%% of the byte cap: origin=%s received=%s limit=%s encoding=%s',
                int(mark * 100), self._origin, total, self._limit, self._encoding,
            )

    async def aclose(self) -> None:
        aclose = getattr(self._stream, 'aclose', None)
        if aclose is not None:
            await aclose()


class BoundedAsyncClient(httpx.AsyncClient):
    # Wraps response.stream, not the transport: httpx only honours *_PROXY env vars when
    # transport is None (_client.py:1399), so a custom transport breaks proxied deployments.

    def __init__(self, *args, response_limit_bytes: int, size_trip: SizeTrip, **kwargs):
        super().__init__(*args, **kwargs)
        self._response_limit_bytes = response_limit_bytes
        self._size_trip = size_trip

    async def _send_single_request(self, request: httpx.Request) -> httpx.Response:
        response = await super()._send_single_request(request)
        response.stream = _CountingStream(
            response.stream,
            self._response_limit_bytes,
            self._size_trip,
            _origin(request.url),
            response.headers.get('content-encoding', ''),
        )
        return response


def _origin(url) -> str:
    """scheme://host[:port] only - a full MCP URL can carry a token in its query."""
    port = f':{url.port}' if url.port else ''
    return f'{url.scheme}://{url.host}{port}'


def build_httpx_client_factory(ssl_verify: bool, limit: Optional[int], trip: SizeTrip):
    """Factory matching mcp's McpHttpClientFactory; limit=None means no bounding at all."""

    def factory(headers=None, timeout=None, auth=None):
        kwargs = {
            'headers': _with_identity_encoding(headers, limit),
            'auth': auth,
            # Mirrors mcp.shared._httpx_utils.create_mcp_http_client defaults.
            'follow_redirects': True,
            'timeout': timeout if timeout is not None else httpx.Timeout(30.0),
        }
        if not ssl_verify:
            kwargs['verify'] = False
        if limit is None:
            return httpx.AsyncClient(**kwargs)
        return BoundedAsyncClient(response_limit_bytes=limit, size_trip=trip, **kwargs)

    return factory


def _with_identity_encoding(headers, limit):
    # Compression off so the cap counts payload bytes: httpx decompresses above the point
    # where we count, so a gzip bomb would pass the cap and still expand to GBs.
    if limit is None:
        return headers
    merged = dict(headers or {})
    if not any(key.lower() == 'accept-encoding' for key in merged):
        merged['accept-encoding'] = 'identity'
    return merged
