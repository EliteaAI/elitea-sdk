"""
Unified MCP Adapter using langchain-mcp-adapters.

This adapter provides a compatibility layer between the old custom McpClient
interface and the official langchain-mcp-adapters implementation.

Phase 2 of MCP consolidation plan - provides backward compatibility while
migrating from custom McpClient/McpSseClient to langchain-mcp-adapters.

Usage:
    # Old way (custom client):
    from ..utils.mcp_client import McpClient
    client = McpClient(url=url, headers=headers)

    # New way (unified adapter):
    from ..utils.mcp_adapter import UnifiedMcpClient
    client = UnifiedMcpClient(url=url, headers=headers)

    # Both have same interface!
"""

import asyncio
import logging
import uuid
from datetime import timedelta

import httpx

from .mcp_discovery_cache import retire_cached_discovery
from .mcp_oauth import (
    GITHUB_BAD_TOKEN_MESSAGE,
    INVALID_CONFIGURED_CREDENTIALS_MESSAGE,
    McpAuthorizationRequired,
)
from .mcp_response_limit import (
    McpResponseTooLargeError,
    SizeTrip,
    build_httpx_client_factory,
)
from .mcp_transport_negotiation import (
    AUTO,
    LEGACY_SSE,
    STREAMABLE_HTTP,
    TransportDecision,
    TransportNegotiator,
    shares_scheme_and_host,
    transport_from_url_suffix,
)
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NEGOTIABLE_TRANSPORTS = frozenset({AUTO, LEGACY_SSE, STREAMABLE_HTTP, "http"})
METADATA_TIMEOUT_SECONDS = 30


def find_unauthorized_response(error: BaseException) -> Optional[httpx.Response]:
    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 401:
        return error.response
    for inner in getattr(error, "exceptions", None) or ():
        found = find_unauthorized_response(inner)
        if found is not None:
            return found
    return None


def _resolve_response_limit():
    """Byte cap for MCP responses, or None when disabled. A lookup that cannot be
    read must not crash the connection - it degrades to today's unbounded behaviour."""
    try:
        from .trace_limits import resolve_mcp_response_limit  # pylint: disable=C0415
        return resolve_mcp_response_limit()
    except Exception:  # pylint: disable=W0718
        logger.exception("[Unified MCP] Could not resolve response byte cap - not bounding")
        return None


class UnifiedMcpClient:
    """
    Unified MCP client using langchain-mcp-adapters under the hood.

    Provides compatibility with the existing McpClient interface while
    using the official langchain-mcp-adapters implementation internally.

    This allows incremental migration - code can switch from McpClient
    to UnifiedMcpClient without changing the interface.
    """

    def __init__(
        self,
        url: str,
        session_id: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 300,
        transport: str = "auto",
        ssl_verify: bool = True,
        configured_auth: bool = False,
        tool_name: Optional[str] = None,
        toolkit_type: Optional[str] = None,
        toolkit_name: Optional[str] = None,
    ):
        """
        Initialize the unified MCP client.

        Args:
            url: MCP server URL
            session_id: Session ID (for compatibility, may not be used by adapter)
            headers: HTTP headers (e.g., Authorization)
            timeout: Request timeout in seconds
            transport: Transport type - "auto", "sse", "streamable_http", "stdio"
            ssl_verify: Whether to verify SSL certificates (default: True)
            configured_auth: True when the Authorization header came from the toolkit's
                database configuration (not an OAuth token acquired at runtime). When True,
                a 401 raises ValueError asking the user to fix their credentials instead of
                triggering the OAuth flow.
            tool_name: Optional tool/display name used in auth-required payloads.
            toolkit_type: Optional toolkit type (e.g., "mcp", "mcp_github") used
                in auth-required payloads.
            toolkit_name: Optional toolkit display/config name used in auth-required
                payloads.
        """
        self.url = url
        self.session_id = session_id or str(uuid.uuid4())
        self.headers = headers or {}
        self.timeout = timeout
        self.transport = transport
        self.ssl_verify = ssl_verify
        self.configured_auth = configured_auth
        self.tool_name = tool_name
        self.toolkit_type = toolkit_type
        self.toolkit_name = toolkit_name

        # Internal state
        self._client = None
        self._session = None
        self._session_context = None
        self._server_name = f"mcp_server_{self.session_id[:8]}"
        # Records a wire-level size abort so the resulting failure can be reported as
        # "response too large" instead of the generic read timeout mcp produces (#6141).
        self._size_trip = SizeTrip()
        self._initialized = False
        self._detected_transport = None
        self._resolved_url = None

        logger.info(f"[Unified MCP] Created client for {url} (transport={transport}, ssl_verify={ssl_verify})")

    async def __aenter__(self):
        """Async context manager entry."""
        await self._connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _connect(self):
        """Establish connection using langchain-mcp-adapters."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ModuleNotFoundError as exc:
            if exc.name != 'langchain_mcp_adapters':
                raise
            raise ImportError(
                "langchain-mcp-adapters is required. "
                "Install with: pip install langchain-mcp-adapters"
            ) from exc

        # Validate URL scheme before attempting any connection.
        # aiohttp raises InvalidURL(url) with just the URL string as the message,
        # which would surface as a cryptic error like "ttttt" instead of a helpful message.
        # Use urlparse so we can distinguish "no scheme" from "unsupported scheme" and
        # handle case-insensitive schemes (HTTP:// is valid per RFC 3986).
        from urllib.parse import urlparse as _urlparse
        _supported_schemes = {'http', 'https'}
        _parsed = _urlparse(self.url)
        if not _parsed.scheme:
            raise ValueError(
                f"Invalid MCP server URL '{self.url}': URL must include a scheme "
                f"(e.g., 'https://{self.url}'). "
                "Please check the URL in your toolkit settings."
            )
        if _parsed.scheme.lower() not in _supported_schemes:
            raise ValueError(
                f"Unsupported URL scheme '{_parsed.scheme}://' in MCP server URL '{self.url}'. "
                f"Supported schemes: {', '.join(sorted(_supported_schemes))}. "
                "Please check the URL in your toolkit settings."
            )

        decision = await self._preflight_auth_check() if self.transport in NEGOTIABLE_TRANSPORTS else None
        detected_transport = decision.transport if decision else self._detect_transport()
        self._resolved_url = decision.url if decision else self.url

        # Build server config for langchain-mcp-adapters
        # Note: SSL verification is handled via httpx_client_factory in _build_server_config
        server_config = self._build_server_config(detected_transport)

        logger.debug(f"[Unified MCP] Connecting to server: {self._server_name} (transport={detected_transport})")

        # Create MultiServerMCPClient with single server
        self._client = MultiServerMCPClient({self._server_name: server_config})

        # Create persistent session.
        # session() auto-initializes the MCP session inside __aenter__ (it calls
        # ClientSession.initialize()). Wrap it in wait_for as a hard outer bound so a
        # stalled TCP connect or an initialize() that never resolves cannot hang the
        # worker thread indefinitely. The per-request session read timeout set in
        # _build_server_config bounds each MCP request; this additionally bounds the
        # transport connect + full handshake.
        self._session_context = self._client.session(self._server_name)
        try:
            self._session = await asyncio.wait_for(
                self._session_context.__aenter__(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError as e:
            # Tear down the half-opened session context so we don't leak the
            # transport/task group.
            try:
                await self._session_context.__aexit__(type(e), e, e.__traceback__)
            except BaseException:
                pass
            self._session_context = None
            self._raise_if_size_tripped()
            raise TimeoutError(
                f"Timed out after {self.timeout}s connecting to MCP server "
                f"'{self._server_name}' at {self.url}. The server may be unreachable "
                "or behind a login proxy that never completes the MCP handshake."
            ) from e
        except BaseException as e:
            self._raise_if_size_tripped()
            unauthorized = find_unauthorized_response(e)
            if unauthorized is not None and not self.configured_auth:
                await self._handle_401_response(unauthorized)
            # langchain-mcp-adapters uses asyncio.TaskGroup internally, which wraps
            # exceptions in ExceptionGroup. Unwrap it so callers see the real error.
            if hasattr(e, 'exceptions') and e.exceptions:
                inner = e.exceptions[0]
                inner_str = str(inner).lower()
                # If auth was configured in DB and the error looks like 401/auth failure,
                # raise a clear ValueError instead of the raw wrapped exception.
                if self.configured_auth and any(
                    kw in inner_str for kw in ['401', 'unauthorized', 'forbidden', '403', 'authentication']
                ):
                    raise ValueError(INVALID_CONFIGURED_CREDENTIALS_MESSAGE) from inner
                # Some servers (e.g. GitHub Copilot) return 400 for an invalid/malformed token.
                if self.configured_auth and '400' in inner_str and 'bad request' in inner_str:
                    raise ValueError(GITHUB_BAD_TOKEN_MESSAGE) from inner
                raise inner from e
            raise

        self._detected_transport = detected_transport
        logger.info(f"[Unified MCP] Connected using {detected_transport} transport")

    def _detect_transport(self) -> str:
        """
        Detect transport type from URL and configuration.

        Mimics the logic from custom McpClient._auto_detect_and_connect.
        """
        if self.transport != AUTO:
            return self.transport
        return transport_from_url_suffix(self.url)

    def _build_server_config(self, transport: str) -> Dict[str, Any]:
        """
        Build server configuration for langchain-mcp-adapters.

        Converts our interface to langchain-mcp-adapters format.
        """
        config = {
            'transport': transport,
        }

        if transport in ['streamable_http', 'sse', 'http']:
            config['url'] = self._resolved_url or self.url
            if self.headers:
                config['headers'] = self.headers
            # Bound the MCP session's request/response wait. mcp's ClientSession
            # defaults read_timeout_seconds=None, so send_request() uses
            # anyio.fail_after(None) and initialize()/list_tools() can wait forever
            # if the server accepts the connection but never returns a valid MCP
            # response (e.g. an SSO proxy answers with a 200 HTML login page whose
            # unexpected-content-type error is dropped instead of routed to the
            # pending request). This makes every MCP request time-bounded.
            config['session_kwargs'] = {
                'read_timeout_seconds': timedelta(seconds=self.timeout)
            }
            # Bound transport-level HTTP/idle-stream reads too. SSEConnection takes
            # plain floats; StreamableHttpConnection (and the 'http' alias) take timedelta.
            if transport == 'sse':
                config['timeout'] = float(self.timeout)
                config['sse_read_timeout'] = float(self.timeout)
            else:
                config['timeout'] = timedelta(seconds=self.timeout)
                config['sse_read_timeout'] = timedelta(seconds=self.timeout)
            # Supplied unconditionally (#6141): the factory carries both the SSL bypass
            # and the response byte cap. resolve_mcp_response_limit() returns None when
            # truncation is disabled, which yields a plain unbounded client.
            byte_limit = _resolve_response_limit()
            config['httpx_client_factory'] = build_httpx_client_factory(
                self.ssl_verify, byte_limit, self._size_trip,
            )
            if not self.ssl_verify:
                logger.warning("[Unified MCP] Using custom httpx client with SSL verification disabled")
        elif transport == 'stdio':
            # Not typically used via this adapter, but support it
            raise ValueError("stdio transport not supported via UnifiedMcpClient URL interface")

        return config

    def _raise_if_size_tripped(self) -> None:
        """Turn a wire-level size abort into a clear error for the model.

        The abort unwinds mcp's transport without answering the pending request, so
        without this the caller only ever sees the session read timeout.
        """
        if self._size_trip.tripped:
            raise McpResponseTooLargeError(self._size_trip.message(self.tool_name))

    async def _preflight_auth_check(self) -> Optional[TransportDecision]:
        negotiator = TransportNegotiator(
            url=self.url,
            headers=self.headers,
            ssl_verify=self.ssl_verify,
            configured_auth=self.configured_auth,
            requested_transport=self.transport,
            on_unauthorized=self._handle_401_response,
        )
        try:
            return await negotiator.negotiate()
        except (McpAuthorizationRequired, ValueError):
            raise
        except Exception as error:  # pylint: disable=W0718
            logger.warning(f"[Unified MCP] Transport negotiation for {self.url} failed "
                           f"({type(error).__name__}: {error}); falling back to the URL suffix")
            return None

    def _metadata_client(self, metadata_url: str) -> httpx.AsyncClient:
        verify_certificate = self.ssl_verify or not shares_scheme_and_host(self.url, metadata_url)
        factory = build_httpx_client_factory(verify_certificate, None, SizeTrip())
        client = factory(timeout=httpx.Timeout(METADATA_TIMEOUT_SECONDS))
        if not verify_certificate:
            client.event_hooks["request"].append(self._refuse_unverified_hop_off_the_configured_host)
        return client

    async def _refuse_unverified_hop_off_the_configured_host(self, request: httpx.Request) -> None:
        if not shares_scheme_and_host(self.url, str(request.url)):
            raise httpx.RequestError(
                f"Refusing to leave {self.url} without certificate verification: {request.url}",
                request=request,
            )

    async def _handle_401_response(self, response):
        """
        Handle 401 Unauthorized response by extracting OAuth metadata and raising exception.

        Args:
            response: HTTP response with 401 status (anything exposing ``headers``)

        Raises:
            ValueError: When configured_auth=True (DB-configured credentials are invalid)
            McpAuthorizationRequired: When configured_auth=False (OAuth flow needed)
        """
        from ..utils.mcp_oauth import (
            McpAuthorizationRequired,
            canonical_resource,
            extract_resource_metadata_url,
            extract_authorization_uri,
            fetch_resource_metadata_async,
            infer_authorization_servers_from_realm,
            fetch_oauth_authorization_server_metadata
        )

        auth_header = response.headers.get('WWW-Authenticate', '')
        # The tool list cached under this credential is dead with it; the next run must
        # discover live so a bad token is met at construction, where the deferred-auth path
        # lives, instead of at a tool call. Other credentials' entries for the server stay:
        # a first-login challenge from a user with no token is normal traffic, not an error.
        retire_cached_discovery(self.url, self.headers, self.ssl_verify)

        # If Authorization was configured in the toolkit's database settings, the credentials
        # are wrong — never start an OAuth flow, just report the error so the user can fix them.
        if self.configured_auth:
            # Default message; try to extract a more specific description from WWW-Authenticate
            error_desc = "Authorization credentials are invalid or insufficient"
            auth_lower = auth_header.lower()

            invalid_token_indicators = [
                'error="invalid_token"',
                'error=invalid_token',
                'token is not authorized',
                'token is expired',
                'token has expired',
                'invalid access token',
                'access token expired',
                'token invalid',
            ]

            if any(indicator in auth_lower for indicator in invalid_token_indicators):
                error_desc = "Token is invalid or has expired"

            if 'error_description="' in auth_header:
                start = auth_header.index('error_description="') + len('error_description="')
                end = auth_header.index('"', start)
                error_desc = auth_header[start:end]
            elif 'error_description=' in auth_header:
                parts = auth_header.split('error_description=')
                if len(parts) > 1:
                    desc_part = parts[1].split(',')[0].strip(' "')
                    if desc_part:
                        error_desc = desc_part

            logger.error(f"[Unified MCP] Authentication failed with configured credentials: {error_desc}")
            raise ValueError(
                f"Authorization failed: {error_desc}. "
                f"Please check the credentials in the toolkit settings."
            )
        resource_metadata_url = extract_resource_metadata_url(auth_header, self.url)

        # First, try authorization_uri from WWW-Authenticate header (preferred)
        authorization_uri = extract_authorization_uri(auth_header)

        metadata = None
        if authorization_uri:
            # Fetch OAuth metadata directly from authorization_uri
            auth_server_metadata = fetch_oauth_authorization_server_metadata(authorization_uri, timeout=30)
            if auth_server_metadata:
                # Extract base authorization server URL from the issuer or the well-known URL
                base_auth_server = auth_server_metadata.get('issuer')
                if not base_auth_server and '/.well-known/' in authorization_uri:
                    base_auth_server = authorization_uri.split('/.well-known/')[0]

                metadata = {
                    'authorization_servers': [base_auth_server] if base_auth_server else [authorization_uri],
                    'oauth_authorization_server': auth_server_metadata
                }

        # Fall back to resource_metadata if authorization_uri didn't work
        if not metadata:
            if resource_metadata_url:
                async with self._metadata_client(resource_metadata_url) as session:
                    metadata = await fetch_resource_metadata_async(
                        resource_metadata_url,
                        session=session,
                        timeout=30
                    )
                    # If we got resource_metadata, also fetch oauth_authorization_server
                    if metadata and metadata.get('authorization_servers'):
                        auth_server_metadata = fetch_oauth_authorization_server_metadata(
                            metadata['authorization_servers'][0], timeout=30
                        )
                        if auth_server_metadata:
                            metadata['oauth_authorization_server'] = auth_server_metadata

        # Infer authorization servers if not in metadata
        if not metadata or not metadata.get('authorization_servers'):
            inferred_servers = infer_authorization_servers_from_realm(auth_header, self.url)
            if inferred_servers:
                if not metadata:
                    metadata = {}
                metadata['authorization_servers'] = inferred_servers

                # Fetch OAuth metadata
                auth_server_metadata = fetch_oauth_authorization_server_metadata(inferred_servers[0], timeout=30)
                if auth_server_metadata:
                    metadata['oauth_authorization_server'] = auth_server_metadata

        raise McpAuthorizationRequired(
            message=f"MCP server {self.url} requires OAuth authorization",
            server_url=canonical_resource(self.url),
            resource_metadata_url=resource_metadata_url,
            www_authenticate=auth_header,
            resource_metadata=metadata,
            status=401,
            tool_name=self.tool_name or self.url,
            toolkit_type=self.toolkit_type or "mcp",
            toolkit_name=self.toolkit_name or self.tool_name or self.url,
        )

    @property
    def detected_transport(self) -> Optional[str]:
        """Get the detected transport type."""
        return self._detected_transport

    @property
    def server_session_id(self) -> Optional[str]:
        """
        Get the server-provided session ID.

        For compatibility with custom McpClient interface.
        langchain-mcp-adapters may provide session IDs differently,
        so we return the session_id we're using.
        """
        return self.session_id

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize MCP session.

        Returns server capabilities and info.

        For langchain-mcp-adapters, initialization happens during session creation,
        so this is mostly a no-op for compatibility.
        """
        if self._initialized:
            return {'status': 'already_initialized'}

        if not self._session:
            await self._connect()

        # Session is already initialized by _connect
        self._initialized = True

        logger.info("[Unified MCP] MCP session initialized")
        return {'status': 'initialized'}

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server.

        Returns:
            List of tool dictionaries with name, description, and inputSchema.
        """
        if not self._session:
            await self._connect()

        try:
            from langchain_mcp_adapters.tools import load_mcp_tools
        except ModuleNotFoundError as exc:
            if exc.name != 'langchain_mcp_adapters':
                raise
            raise ImportError(
                "langchain-mcp-adapters is required. "
                "Install with: pip install langchain-mcp-adapters"
            ) from exc

        # Load tools using langchain-mcp-adapters
        connection = self._client.connections.get(self._server_name)
        try:
            tools = await load_mcp_tools(
                self._session,
                connection=connection,
                server_name=self._server_name
            )
        except BaseException:
            self._raise_if_size_tripped()
            raise

        # Convert LangChain tools to our format
        tool_list = []
        for tool in tools:
            tool_dict = {
                'name': tool.name,
                'description': tool.description or '',
            }

            # Add inputSchema if available
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    # Handle both dict (already JSON schema) and Pydantic model
                    if isinstance(tool.args_schema, dict):
                        # langchain-mcp-adapters returns dict (already JSON schema)
                        tool_dict['inputSchema'] = tool.args_schema
                    else:
                        # Pydantic model - convert to JSON schema
                        tool_dict['inputSchema'] = tool.args_schema.model_json_schema()
                except Exception as e:
                    logger.warning(f"[Unified MCP] Failed to convert args_schema for {tool.name}: {e}")

            tool_list.append(tool_dict)

        logger.info(f"[Unified MCP] Listed {len(tool_list)} tools")
        return tool_list

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if not self._session:
            await self._connect()

        # Strip None values from arguments — MCP servers expect optional params
        # to be omitted rather than sent as null (e.g., Go servers reject nil for float64)
        clean_args = {k: v for k, v in (arguments or {}).items() if v is not None}

        logger.debug(f"[Unified MCP] Calling tool {tool_name} with args: {clean_args}")
        try:
            result = await self._session.call_tool(tool_name, clean_args)
        except BaseException:
            # A size abort surfaces here as a timeout or a task-group failure; the trip
            # record is what tells us the real cause.
            self._raise_if_size_tripped()
            raise

        if isinstance(result, dict):
            return result
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json", by_alias=True, exclude_none=True)
        raise TypeError(f"Unsupported MCP tool result type: {type(result).__name__}")

    async def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a raw JSON-RPC request.

        This is for compatibility with custom McpClient interface.
        Most code should use higher-level methods like list_tools() or call_tool().
        """
        if not self._session:
            await self._connect()

        # Map common methods to our interface
        if method == 'tools/list':
            tools = await self.list_tools()
            return {'result': {'tools': tools}}
        elif method == 'tools/call':
            tool_name = params.get('name') if params else None
            arguments = params.get('arguments', {}) if params else {}
            result = await self.call_tool(tool_name, arguments)
            return {'result': result}
        elif method == 'initialize':
            result = await self.initialize()
            return {'result': result}
        else:
            logger.warning(f"[Unified MCP] Unsupported method: {method}")
            raise NotImplementedError(f"Method '{method}' not supported by UnifiedMcpClient")

    async def close(self):
        """Close the MCP connection."""
        logger.info("[Unified MCP] Closing connection...")

        if self._session_context:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"[Unified MCP] Error closing session: {e}")
            except BaseException as e:
                # A size abort tears the transport task group down here; swallowing it keeps
                # the McpResponseTooLargeError from call_tool as the reported failure (#6141).
                if not self._size_trip.tripped:
                    raise
                logger.warning(f"[Unified MCP] Session closed after size abort: {e!r}")

        self._session = None
        self._session_context = None
        self._client = None
        self._initialized = False

        logger.info("[Unified MCP] Connection closed")


# Convenience function for creating clients
def create_mcp_client(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 300,
    transport: str = "auto",
    ssl_verify: bool = True
) -> UnifiedMcpClient:
    """
    Create a unified MCP client.

    This is a drop-in replacement for creating custom McpClient instances.

    Args:
        url: MCP server URL
        headers: HTTP headers (authentication, etc.)
        timeout: Request timeout in seconds
        transport: Transport type ("auto", "sse", "streamable_http")
        ssl_verify: Whether to verify SSL certificates (default: True)

    Returns:
        UnifiedMcpClient instance
    """
    return UnifiedMcpClient(
        url=url,
        headers=headers,
        timeout=timeout,
        transport=transport,
        ssl_verify=ssl_verify
    )
