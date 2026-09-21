import argparse
import asyncio
import contextlib
import json
import socket
import threading
import time

import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Mount, Route

MONDAY_CSP_HEADER_BYTES = 11108
OVERSIZED_CSP = ("default-src 'self'; img-src " + "https://cdn.example.com " * 1000)[:MONDAY_CSP_HEADER_BYTES]
LOGIN_PAGE = "<!doctype html><html><body><form action='/login'>Sign in</form></body></html>"
ECHO_TOOL = types.Tool(
    name="echo",
    description="Echoes the argument dict it received",
    inputSchema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
)


def build_mcp_server() -> Server:
    server = Server("elitea-6688-rig")

    @server.list_tools()
    async def list_tools():
        return [ECHO_TOOL]

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        return [types.TextContent(type="text", text=json.dumps({"tool": name, "received": arguments}))]

    return server


def origin_of(request: Request) -> str:
    return f"{request.url.scheme}://{request.url.netloc}"


def legacy_sse_endpoint(server: Server, transport: SseServerTransport):
    async def endpoint(request: Request):
        async with transport.connect_sse(request.scope, request.receive, request._send) as (read, write):
            await server.run(read, write, server.create_initialization_options())
        return Response()
    return endpoint


async def oversized_header_challenge(request: Request) -> Response:
    metadata_url = f"{origin_of(request)}/.well-known/oauth-protected-resource/big401/mcp"
    return PlainTextResponse(
        "unauthorized",
        status_code=401,
        headers={
            "Content-Security-Policy": OVERSIZED_CSP,
            "WWW-Authenticate": f'Bearer resource_metadata="{metadata_url}"',
        },
    )


async def oversized_header_resource_metadata(request: Request) -> Response:
    origin = origin_of(request)
    return JSONResponse(
        {"resource": f"{origin}/big401/mcp", "authorization_servers": [f"{origin}/as"]},
        headers={"Content-Security-Policy": OVERSIZED_CSP},
    )


async def authorization_server_metadata(request: Request) -> Response:
    issuer = f"{origin_of(request)}/as"
    return JSONResponse({
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/authorize",
        "token_endpoint": f"{issuer}/token",
        "registration_endpoint": f"{issuer}/register",
        "response_types_supported": ["code"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
    })


def cross_origin_login_redirect(login_port: int):
    async def endpoint(request: Request) -> Response:
        return RedirectResponse(f"{request.url.scheme}://{request.url.hostname}:{login_port}/login", status_code=302)
    return endpoint


async def moved_permanently(request: Request) -> Response:
    return RedirectResponse("/mcp/", status_code=301)


async def html_login_page(request: Request) -> Response:
    return HTMLResponse(LOGIN_PAGE)


async def retired_legacy_endpoint(request: Request) -> Response:
    if request.method == "POST":
        return PlainTextResponse("Method Not Allowed", status_code=405)
    return PlainTextResponse("Gone", status_code=410)


async def rejects_token(request: Request) -> Response:
    return PlainTextResponse("Bad Request", status_code=400)


def build_rig_app(login_port: int) -> Starlette:
    server = build_mcp_server()
    manager = StreamableHTTPSessionManager(app=server, stateless=False)
    events_transport = SseServerTransport("/events-messages/")
    sse_transport = SseServerTransport("/sse-messages/")

    async def streamable(scope, receive, send):
        await manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with manager.run():
            yield

    return Starlette(routes=[
        Mount("/mcp", app=streamable),
        Route("/events", endpoint=legacy_sse_endpoint(server, events_transport), methods=["GET"]),
        Mount("/events-messages/", app=events_transport.handle_post_message),
        Route("/sse", endpoint=legacy_sse_endpoint(server, sse_transport), methods=["GET"]),
        Mount("/sse-messages/", app=sse_transport.handle_post_message),
        Route("/big401/mcp", endpoint=oversized_header_challenge, methods=["GET", "POST"]),
        Route("/.well-known/oauth-protected-resource/big401/mcp", endpoint=oversized_header_resource_metadata),
        Route("/.well-known/oauth-authorization-server/as", endpoint=authorization_server_metadata),
        Route("/moved/mcp", endpoint=moved_permanently, methods=["GET", "POST"]),
        Route("/sso/mcp", endpoint=cross_origin_login_redirect(login_port), methods=["GET", "POST"]),
        Route("/html/mcp", endpoint=html_login_page, methods=["GET", "POST"]),
        Route("/retired/sse", endpoint=retired_legacy_endpoint, methods=["GET", "POST"]),
        Route("/github400/mcp", endpoint=rejects_token, methods=["GET", "POST"]),
    ], lifespan=lifespan)


def build_login_app() -> Starlette:
    return Starlette(routes=[Route("/login", endpoint=html_login_page, methods=["GET", "POST"])])


class RequestLog:
    def __init__(self, app, requests: list):
        self.app = app
        self.requests = requests

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            self.requests.append((scope["method"], scope["path"]))
        await self.app(scope, receive, send)


def bind_socket(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    return sock


def build_servers(host: str, port: int, login_port: int, log_level: str, requests: list):
    rig_socket = bind_socket(host, port)
    login_socket = bind_socket(host, login_port)
    bound_login_port = login_socket.getsockname()[1]
    rig_app = RequestLog(build_rig_app(bound_login_port), requests)
    rig = uvicorn.Server(uvicorn.Config(rig_app, log_level=log_level))
    login = uvicorn.Server(uvicorn.Config(build_login_app(), log_level=log_level))
    return (rig, rig_socket), (login, login_socket)


async def serve(servers) -> None:
    await asyncio.gather(*(server.serve(sockets=[sock]) for server, sock in servers))


class RunningRig:
    def __init__(self, host: str = "127.0.0.1"):
        self._host = host
        self.requests = []
        self._servers = build_servers(host, 0, 0, "warning", self.requests)
        self._thread = threading.Thread(target=asyncio.run, args=(serve(self._servers),), daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._servers[0][1].getsockname()[1]}"

    def __enter__(self) -> "RunningRig":
        self._thread.start()
        deadline = time.monotonic() + 10
        while not all(server.started for server, _ in self._servers):
            if time.monotonic() > deadline:
                raise RuntimeError("MCP rig did not start within 10s")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc_info) -> None:
        for server, _ in self._servers:
            server.should_exit = True
        self._thread.join(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MCP servers reproducing issue #6688")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--login-port", type=int, default=8766)
    args = parser.parse_args()
    asyncio.run(serve(build_servers(args.host, args.port, args.login_port, "info", [])))


if __name__ == "__main__":
    main()
