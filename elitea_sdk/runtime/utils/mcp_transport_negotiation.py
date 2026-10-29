import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from httpx_sse import EventSource

from .mcp_oauth import (
    GITHUB_BAD_TOKEN_MESSAGE,
    McpEndpointError,
    html_login_page_message,
    login_redirect_message,
    retired_endpoint_message,
    sso_redirect_message,
)
from .mcp_response_limit import SizeTrip, build_httpx_client_factory

logger = logging.getLogger(__name__)

STREAMABLE_HTTP = "streamable_http"
LEGACY_SSE = "sse"
AUTO = "auto"
PROBE_TIMEOUT_SECONDS = 10.0
LEGACY_SSE_PROBE_TIMEOUT_SECONDS = 5.0
MAX_SAME_ORIGIN_REDIRECTS = 5
RESUBMITTABLE_REDIRECTS = frozenset({301, 302, 307, 308})
STATUSES_WITH_OWN_MEANING = frozenset({401, 403, 429})
GONE_STATUSES = frozenset({404, 405, 410})
DEFAULT_PORTS = {"http": 80, "https": 443}
LEGACY_SSE_SUFFIX = "/sse"
CURRENT_TRANSPORT_SUFFIX = "/mcp"
PROBE_PROTOCOL_VERSION = "2024-11-05"
LOGIN_PATH_SEGMENTS = frozenset({"login", "forward-auth"})
AUTHORIZATION_REQUEST_PARAMS = frozenset({"client_id", "response_type"})
EDGE_REQUEST_ID_HEADERS = ("cf-ray", "x-request-id")
MAX_REQUEST_ID_LENGTH = 128

UnauthorizedHandler = Callable[[httpx.Response], Awaitable[None]]


@dataclass(frozen=True)
class TransportDecision:
    transport: str
    url: str


@dataclass(frozen=True)
class LegacySseProbe:
    endpoint_found: bool
    response: Optional[httpx.Response]

    @property
    def status(self) -> Optional[int]:
        return self.response.status_code if self.response is not None else None


NO_LEGACY_SSE = LegacySseProbe(endpoint_found=False, response=None)


def effective_port(parsed) -> Optional[int]:
    return parsed.port or DEFAULT_PORTS.get(parsed.scheme)


def is_same_origin_redirect(source: str, target: str) -> bool:
    origin, destination = urlparse(source), urlparse(target)
    if (origin.hostname or "").lower() != (destination.hostname or "").lower():
        return False
    if origin.scheme == destination.scheme:
        return effective_port(origin) == effective_port(destination)
    return origin.scheme == "http" and destination.scheme == "https"


def is_https_dropped_by_a_proxy(source: str, target: str) -> bool:
    origin, destination = urlparse(source), urlparse(target)
    return (
        origin.scheme == "https"
        and destination.scheme == "http"
        and destination.port is None
        and (origin.hostname or "").lower() == (destination.hostname or "").lower()
    )


def keep_https(source: str, target: str) -> str:
    if not is_https_dropped_by_a_proxy(source, target):
        return target
    origin = urlparse(source)
    return urlparse(target)._replace(scheme=origin.scheme, netloc=origin.netloc).geturl()


def is_login_redirect(target: str) -> bool:
    """A redirect into a login or authorization page: a ``login`` or ``forward-auth`` path segment, or an
    OAuth/OIDC authorization request (``client_id`` and ``response_type`` in the query). Such routes are
    GET-only, so re-POSTing the probe into them only ends in a misleading 405 (#7278)."""
    parsed = urlparse(target)
    segments = {segment.lower() for segment in parsed.path.split("/")}
    query = set(parse_qs(parsed.query, keep_blank_values=True))
    return bool(segments & LOGIN_PATH_SEGMENTS) or AUTHORIZATION_REQUEST_PARAMS <= query


def display_url(url: str) -> str:
    """``url`` for an error message: no credentials, query or fragment."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    host = f"[{host}]" if ":" in host else host
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return parsed._replace(netloc=netloc, params="", query="", fragment="").geturl()


def edge_request_id(responses) -> Optional[str]:
    for response in reversed(responses):
        for name in EDGE_REQUEST_ID_HEADERS:
            value = response.headers.get(name, "")
            if value and value.isprintable():
                return f"{name}: {value[:MAX_REQUEST_ID_LENGTH]}"
    return None


def describe_exchange(method: str, url: str, responses) -> str:
    """The request and the status of each hop, plus the edge's request id; never a body or a header value
    other than that id."""
    statuses = " -> ".join(str(response.status_code) for response in responses)
    request_id = edge_request_id(responses)
    return f"{method} {display_url(url)} (statuses: {statuses}{'; ' + request_id if request_id else ''})"


def shares_scheme_and_host(url: str, other: str) -> bool:
    connection, endpoint = urlparse(url), urlparse(other)
    return connection.scheme == endpoint.scheme and connection.netloc == endpoint.netloc


def is_html(response: httpx.Response) -> bool:
    return "text/html" in response.headers.get("content-type", "").lower()


def is_event_stream(response: httpx.Response) -> bool:
    return "text/event-stream" in response.headers.get("content-type", "").lower()


def current_transport_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if not path.endswith(LEGACY_SSE_SUFFIX):
        return None
    new_path = path[: -len(LEGACY_SSE_SUFFIX)] + CURRENT_TRANSPORT_SUFFIX
    return parsed._replace(path=new_path, query="", fragment="").geturl()


def transport_from_url_suffix(url: str) -> str:
    return LEGACY_SSE if url.rstrip("/").endswith(LEGACY_SSE_SUFFIX) else STREAMABLE_HTTP


def is_legacy_sse_candidate(status: int) -> bool:
    return 400 <= status < 500 and status not in STATUSES_WITH_OWN_MEANING


def initialize_request() -> Dict:
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "initialize",
        "params": {
            "protocolVersion": PROBE_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "ELITEA MCP Client", "version": "1.0.0"},
        },
    }


def raise_for_unsupported_status(status: int, url: str, exchange: str) -> None:
    if status == 404:
        raise ValueError(f"MCP server endpoint not found (404). Please verify the server URL is correct: {url}")
    if status == 403:
        raise ValueError(
            "Access forbidden (403). Your credentials are valid but you don't have "
            "permission to access this MCP server. Please check with your administrator."
        )
    if status == 500:
        raise ValueError(
            "The MCP server encountered an internal error (500). "
            "Please try again later or contact the server administrator."
        )
    if status in (502, 503):
        raise ValueError(
            f"The MCP server is unavailable ({status}). "
            "It may be down for maintenance. Please try again later."
        )
    if status >= 400:
        # Shown verbatim, except where the user-facing mapper has its own guidance (401, 429)
        error = ValueError if status in STATUSES_WITH_OWN_MEANING else McpEndpointError
        raise error(
            f"The MCP server returned an error ({status}) to {exchange}. "
            "Please verify the server URL and try again."
        )


class TransportNegotiator:
    def __init__(
        self,
        url: str,
        headers: Dict[str, str],
        ssl_verify: bool,
        configured_auth: bool,
        requested_transport: str,
        on_unauthorized: UnauthorizedHandler,
    ):
        self.url = url
        self.headers = headers
        self.ssl_verify = ssl_verify
        self.configured_auth = configured_auth
        self.requested_transport = requested_transport
        self.on_unauthorized = on_unauthorized

    async def negotiate(self) -> Optional[TransportDecision]:
        async with self._probe_client() as client:
            if self.requested_transport == LEGACY_SSE:
                return await self._confirm_legacy_sse(client)
            url, responses = await asyncio.wait_for(self._post_initialize(client), timeout=PROBE_TIMEOUT_SECONDS)
            return await self._decide(client, url, responses)

    def _probe_client(self) -> httpx.AsyncClient:
        factory = build_httpx_client_factory(self.ssl_verify, None, SizeTrip())
        return factory(timeout=httpx.Timeout(PROBE_TIMEOUT_SECONDS))

    async def _post_initialize(self, client: httpx.AsyncClient):
        url, responses = self.url, []
        for _ in range(MAX_SAME_ORIGIN_REDIRECTS + 1):
            response = await self._send_initialize(client, url)
            responses.append(response)
            if not response.is_redirect:
                return url, responses
            location = keep_https(url, urljoin(url, response.headers.get("location", "")))
            if response.status_code not in RESUBMITTABLE_REDIRECTS or not is_same_origin_redirect(url, location):
                raise McpEndpointError(sso_redirect_message(response.status_code, location))
            if is_login_redirect(location):
                raise McpEndpointError(login_redirect_message(display_url(location),
                                                              describe_exchange("POST", url, responses)))
            logger.info(f"[MCP negotiation] Following same-origin redirect {response.status_code} {url} -> {location}")
            url = location
        raise McpEndpointError(sso_redirect_message(response.status_code, url))

    async def _send_initialize(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.headers,
        }
        async with client.stream("POST", url, json=initialize_request(), headers=headers,
                                 follow_redirects=False) as response:
            return response

    async def _decide(self, client: httpx.AsyncClient, url: str, responses) -> TransportDecision:
        response = responses[-1]
        status = response.status_code
        if response.is_success:
            if is_html(response):
                raise McpEndpointError(html_login_page_message())
            return TransportDecision(STREAMABLE_HTTP, url)
        if status == 401:
            await self.on_unauthorized(response)
        if self.requested_transport == AUTO and is_legacy_sse_candidate(status):
            legacy = await self._probe_legacy_sse(client, url)
            if legacy.endpoint_found:
                return TransportDecision(LEGACY_SSE, url)
            if status == 400 and self.configured_auth:
                raise ValueError(GITHUB_BAD_TOKEN_MESSAGE)
            if status in GONE_STATUSES and legacy.status == 401:
                await self.on_unauthorized(legacy.response)
            if status in GONE_STATUSES and legacy.status in GONE_STATUSES:
                raise McpEndpointError(retired_endpoint_message(url, current_transport_url(url)))
        elif status == 400 and self.configured_auth:
            raise ValueError(GITHUB_BAD_TOKEN_MESSAGE)
        raise_for_unsupported_status(status, url, describe_exchange("POST", url, responses))
        return TransportDecision(STREAMABLE_HTTP, url)

    async def _confirm_legacy_sse(self, client: httpx.AsyncClient) -> Optional[TransportDecision]:
        legacy = await self._probe_legacy_sse(client, self.url)
        if legacy.status == 401:
            await self.on_unauthorized(legacy.response)
        return TransportDecision(LEGACY_SSE, self.url) if legacy.endpoint_found else None

    async def _probe_legacy_sse(self, client: httpx.AsyncClient, url: str) -> LegacySseProbe:
        try:
            return await asyncio.wait_for(self._read_first_event(client, url),
                                          timeout=LEGACY_SSE_PROBE_TIMEOUT_SECONDS)
        except (httpx.TransportError, asyncio.TimeoutError) as error:
            logger.info(f"[MCP negotiation] Legacy SSE probe of {url} gave no endpoint ({type(error).__name__})")
            return NO_LEGACY_SSE

    async def _read_first_event(self, client: httpx.AsyncClient, url: str) -> LegacySseProbe:
        headers = {**self.headers, "Accept": "text/event-stream"}
        async with client.stream("GET", url, headers=headers, follow_redirects=False) as response:
            if response.status_code != 200 or not is_event_stream(response):
                return LegacySseProbe(endpoint_found=False, response=response)
            async for event in EventSource(response).aiter_sse():
                return LegacySseProbe(
                    endpoint_found=event.event == "endpoint"
                    and shares_scheme_and_host(url, urljoin(url, event.data)),
                    response=response,
                )
            return LegacySseProbe(endpoint_found=False, response=response)
