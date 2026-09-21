import asyncio
import json

import pytest

from elitea_sdk.runtime.utils import mcp_oauth
from elitea_sdk.runtime.utils.mcp_adapter import UnifiedMcpClient
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired, McpEndpointError, html_login_page_message
from tests.runtime.utils.mcp_rig_server import RunningRig

PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@pytest.fixture(scope="module")
def rig():
    with RunningRig() as running:
        yield running


@pytest.fixture(autouse=True)
def direct_connections(monkeypatch, rig):
    for variable in PROXY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    rig.requests.clear()


def list_and_call(url):
    async def session():
        async with UnifiedMcpClient(url=url, timeout=20) as client:
            tools = await client.list_tools()
            result = await client.call_tool("echo", {"q": "hi"})
            return client, [tool["name"] for tool in tools], result
    return asyncio.run(session())


def connect(url):
    async def session():
        async with UnifiedMcpClient(url=url, timeout=20):
            pass
    asyncio.run(session())


def echoed_arguments(result):
    return json.loads(result["content"][0]["text"])["received"]


def test_an_sse_only_server_connects_over_legacy_sse_and_runs_a_tool(rig):
    client, tools, result = list_and_call(f"{rig.base_url}/events")

    assert client.detected_transport == "sse"
    assert tools == ["echo"]
    assert echoed_arguments(result) == {"q": "hi"}
    assert rig.requests[:3] == [("POST", "/events"), ("GET", "/events"), ("GET", "/events")]


def test_a_trailing_slash_redirect_is_followed(rig):
    client, tools, result = list_and_call(f"{rig.base_url}/mcp")

    assert client.detected_transport == "streamable_http"
    assert client._resolved_url == f"{rig.base_url}/mcp/"
    assert tools == ["echo"]
    assert echoed_arguments(result) == {"q": "hi"}
    assert rig.requests[:2] == [("POST", "/mcp"), ("POST", "/mcp/")]


def test_a_permanent_redirect_keeps_posting_to_the_resolved_url(rig):
    client, tools, _ = list_and_call(f"{rig.base_url}/moved/mcp")

    assert client._resolved_url == f"{rig.base_url}/mcp/"
    assert tools == ["echo"]
    assert ("GET", "/mcp/") not in rig.requests[:3]


def test_a_challenge_behind_an_oversized_header_starts_the_oauth_flow(rig, monkeypatch):
    monkeypatch.setattr(mcp_oauth, "fetch_oauth_authorization_server_metadata", lambda *a, **k: None)

    with pytest.raises(McpAuthorizationRequired) as raised:
        connect(f"{rig.base_url}/big401/mcp")

    assert raised.value.resource_metadata["authorization_servers"] == [f"{rig.base_url}/as"]
    assert rig.requests == [
        ("POST", "/big401/mcp"),
        ("GET", "/.well-known/oauth-protected-resource/big401/mcp"),
    ]


def test_a_cross_origin_login_redirect_fails_fast(rig):
    with pytest.raises(McpEndpointError, match="redirected the request \\(302\\)"):
        connect(f"{rig.base_url}/sso/mcp")

    assert rig.requests == [("POST", "/sso/mcp")]


def test_an_html_login_page_fails_fast(rig):
    with pytest.raises(McpEndpointError) as raised:
        connect(f"{rig.base_url}/html/mcp")

    assert str(raised.value) == html_login_page_message()


def test_a_retired_sse_endpoint_suggests_the_mcp_url(rig):
    with pytest.raises(McpEndpointError) as raised:
        connect(f"{rig.base_url}/retired/sse")

    assert str(raised.value) == (
        f"The MCP endpoint {rig.base_url}/retired/sse has been retired or is no longer available. "
        f"Try {rig.base_url}/retired/mcp instead."
    )
    assert rig.requests == [("POST", "/retired/sse"), ("GET", "/retired/sse")]
