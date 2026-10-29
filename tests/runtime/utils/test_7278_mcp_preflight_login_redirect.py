"""The transport preflight does not re-POST into a login redirect, and names the exchange it failed on (#7278).

A same-origin redirect to a GET-only login route used to be followed with POST and reported as an opaque
"Method not allowed (405)". The preflight now stops at such a redirect, and an unsupported status names the
method, the URL, each hop's status and the edge request id, never a body or a credential.
"""

import asyncio

import httpx
import pytest

from elitea_sdk.runtime.utils.mcp_adapter import UnifiedMcpClient
from elitea_sdk.runtime.utils.mcp_oauth import McpEndpointError, extract_user_friendly_mcp_error
from elitea_sdk.runtime.utils.mcp_transport_negotiation import TransportDecision, is_login_redirect
from tests.runtime.utils.mcp_probe_transport import patch_probe_transport, respond

SERVER = "https://platform.example.test"
ENDPOINT = "/app/8/mcp/elitea_core/skills"
HEADERS = {"Authorization": "Bearer fixture-credential"}


def routes(table):
    async def handler(request):
        return await table.get((request.method, request.url.path), respond(599))(request)
    return handler


def negotiate(url=f"{SERVER}{ENDPOINT}", **kwargs):
    client = UnifiedMcpClient(url=url, timeout=10, headers=HEADERS, **kwargs)
    return asyncio.run(client._preflight_auth_check())


@pytest.mark.parametrize("target, expected", [
    (f"{SERVER}/forward-auth/login?next=/app/8/mcp", True),
    (f"{SERVER}/forward-auth/verify", True),
    (f"{SERVER}/login", True),
    (f"{SERVER}/auth/LOGIN/", True),
    (f"{SERVER}/oauth2/authorize?client_id=elitea&response_type=code&state=x", True),
    (f"{SERVER}/realms/r/protocol/openid-connect/auth?response_type=code&client_id=elitea", True),
    (f"{SERVER}{ENDPOINT}/", False),
    (f"{SERVER}/mcp/", False),
    (f"{SERVER}/login-tools/mcp", False),
    (f"{SERVER}/oauth2/authorize", False),
    (f"{SERVER}/mcp?client_id=elitea", False),
])
def test_login_redirect_rule(target, expected):
    assert is_login_redirect(target) is expected


def test_a_redirect_to_the_login_page_is_not_followed_and_is_reported_readably(monkeypatch):
    transport = patch_probe_transport(monkeypatch, routes({
        ("POST", ENDPOINT): respond(302, {"Location": "/forward-auth/login?target=%2Fapp%2F8", "cf-ray": "8f1c-AMS"},
                                    b"<html>secret page body</html>"),
        ("POST", "/forward-auth/login"): respond(405),   # the GET-only login route
    }))

    with pytest.raises(McpEndpointError) as raised:
        negotiate()

    message = str(raised.value)
    assert transport.calls == [("POST", f"{SERVER}{ENDPOINT}")]
    assert message == (
        f"The MCP server redirected the request to a login page ({SERVER}/forward-auth/login) instead of "
        f"answering POST {SERVER}{ENDPOINT} (statuses: 302; cf-ray: 8f1c-AMS). An MCP client cannot complete "
        "that login. Please verify the server URL and authentication in the toolkit settings."
    )
    assert extract_user_friendly_mcp_error(raised.value, HEADERS) == message
    assert "(405)" not in message and "fixture-credential" not in message and "secret page body" not in message


@pytest.mark.parametrize("header", ["cf-ray", "x-request-id"])
def test_an_unsupported_status_names_the_method_url_hops_and_request_id(monkeypatch, header):
    patch_probe_transport(monkeypatch, routes({
        ("POST", "/mcp"): respond(307, {"Location": f"{SERVER}/mcp/"}),
        ("POST", "/mcp/"): respond(405, {header: "req-42", "Content-Type": "text/plain"}, b"secret body"),
        ("GET", "/mcp/"): respond(200, {"Content-Type": "text/html"}),
    }))

    with pytest.raises(McpEndpointError) as raised:
        negotiate("https://user:pw@platform.example.test/mcp?token=abc")

    message = str(raised.value)
    assert message == (
        f"The MCP server returned an error (405) to POST {SERVER}/mcp/ (statuses: 307 -> 405; {header}: req-42). "
        "Please verify the server URL and try again."
    )
    assert extract_user_friendly_mcp_error(raised.value, HEADERS) == message
    for secret in ("fixture-credential", "secret body", "pw", "token=abc"):
        assert secret not in message


def test_an_ordinary_same_origin_redirect_is_still_followed(monkeypatch):
    transport = patch_probe_transport(monkeypatch, routes({
        ("POST", ENDPOINT): respond(308, {"Location": f"{ENDPOINT}/"}),
        ("POST", f"{ENDPOINT}/"): respond(200, {"Content-Type": "application/json"}),
    }))

    decision = negotiate()

    assert decision == TransportDecision("streamable_http", f"{SERVER}{ENDPOINT}/")
    assert transport.calls == [("POST", f"{SERVER}{ENDPOINT}"), ("POST", f"{SERVER}{ENDPOINT}/")]


def test_a_429_keeps_the_mappers_own_guidance(monkeypatch):
    patch_probe_transport(monkeypatch, respond(429, {"cf-ray": "req-7"}))

    with pytest.raises(ValueError) as raised:
        negotiate()

    assert not isinstance(raised.value, McpEndpointError)
    assert extract_user_friendly_mcp_error(raised.value, HEADERS) == "Rate limit exceeded. Please wait a moment and try again."
