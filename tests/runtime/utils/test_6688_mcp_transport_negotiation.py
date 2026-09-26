import asyncio
import time

import httpx
import pytest

from elitea_sdk.runtime.utils import mcp_oauth, mcp_transport_negotiation
from elitea_sdk.runtime.utils.mcp_adapter import UnifiedMcpClient, find_unauthorized_response
from elitea_sdk.runtime.utils.mcp_oauth import (
    McpAuthorizationRequired,
    McpEndpointError,
    extract_user_friendly_mcp_error,
    fetch_resource_metadata_async,
    html_login_page_message,
    retired_endpoint_message,
    sso_redirect_message,
)
from elitea_sdk.runtime.utils.mcp_transport_negotiation import (
    GITHUB_BAD_TOKEN_MESSAGE,
    TransportDecision,
    current_transport_url,
    is_same_origin_redirect,
)
from tests.runtime.utils.mcp_probe_transport import patch_probe_transport, respond

SERVER = "https://mcp.example.test"
EVENT_STREAM = {"Content-Type": "text/event-stream"}
JSON = {"Content-Type": "application/json"}


def run(coro):
    return asyncio.run(coro)


def routes(table):
    async def handler(request):
        key = (request.method, request.url.path)
        if key not in table:
            return httpx.Response(599, text=f"unexpected {key}")
        return await table[key](request)
    return handler


def endpoint_event(data):
    return respond(200, EVENT_STREAM, f"event: endpoint\ndata: {data}\n\n".encode())


def negotiate(url=f"{SERVER}/mcp", **kwargs):
    client = UnifiedMcpClient(url=url, timeout=10, **kwargs)
    return client, run(client._preflight_auth_check())


def no_authorization_server_lookup(monkeypatch):
    monkeypatch.setattr(mcp_oauth, "fetch_oauth_authorization_server_metadata", lambda *a, **k: None)


class TestStreamableHttp:
    def test_a_2xx_answer_selects_streamable_http_at_the_requested_url(self, monkeypatch):
        transport = patch_probe_transport(monkeypatch, routes({("POST", "/mcp"): respond(200, EVENT_STREAM)}))

        _, decision = negotiate()

        assert decision == TransportDecision("streamable_http", f"{SERVER}/mcp")
        assert transport.calls == [("POST", f"{SERVER}/mcp")]

    def test_a_2xx_answer_on_an_sse_suffixed_url_is_not_forced_onto_legacy_sse(self, monkeypatch):
        transport = patch_probe_transport(monkeypatch, routes({("POST", "/sse"): respond(200, EVENT_STREAM)}))

        _, decision = negotiate(f"{SERVER}/sse")

        assert decision == TransportDecision("streamable_http", f"{SERVER}/sse")
        assert transport.calls == [("POST", f"{SERVER}/sse")]

    def test_the_probe_does_not_share_the_transport_size_trip(self, monkeypatch):
        transport = patch_probe_transport(monkeypatch, routes({("POST", "/mcp"): respond(200, JSON)}))

        client, _ = negotiate()

        assert transport.size_trips and all(trip is not client._size_trip for trip in transport.size_trips)


class TestRedirects:
    def test_a_same_origin_trailing_slash_redirect_is_followed_and_the_final_url_is_used(self, monkeypatch):
        transport = patch_probe_transport(monkeypatch, routes({
            ("POST", "/mcp"): respond(307, {"Location": f"{SERVER}/mcp/"}),
            ("POST", "/mcp/"): respond(200, EVENT_STREAM),
        }))

        _, decision = negotiate()

        assert decision == TransportDecision("streamable_http", f"{SERVER}/mcp/")
        assert transport.calls == [("POST", f"{SERVER}/mcp"), ("POST", f"{SERVER}/mcp/")]

    def test_a_relative_same_origin_redirect_is_followed(self, monkeypatch):
        patch_probe_transport(monkeypatch, routes({
            ("POST", "/mcp"): respond(308, {"Location": "/mcp/"}),
            ("POST", "/mcp/"): respond(200, JSON),
        }))

        _, decision = negotiate()

        assert decision.url == f"{SERVER}/mcp/"

    def test_a_redirect_that_drops_https_on_the_same_host_is_followed_over_https(self, monkeypatch):
        transport = patch_probe_transport(monkeypatch, routes({
            ("POST", "/mcp/"): respond(307, {"Location": "http://mcp.example.test/mcp"}),
            ("POST", "/mcp"): respond(200, EVENT_STREAM),
        }))

        _, decision = negotiate(f"{SERVER}/mcp/")

        assert decision == TransportDecision("streamable_http", f"{SERVER}/mcp")
        assert transport.calls == [("POST", f"{SERVER}/mcp/"), ("POST", f"{SERVER}/mcp")]

    def test_a_redirect_to_plain_http_on_another_port_is_refused(self, monkeypatch):
        transport = patch_probe_transport(monkeypatch, respond(307, {"Location": "http://mcp.example.test:8080/mcp"}))

        with pytest.raises(McpEndpointError) as raised:
            negotiate(f"{SERVER}/mcp/")

        assert str(raised.value) == sso_redirect_message(307, "http://mcp.example.test:8080/mcp")
        assert transport.calls == [("POST", f"{SERVER}/mcp/")]

    @pytest.mark.parametrize("source, location", [
        ("https://mcp.example.test:8443/mcp", "https://mcp.example.test/mcp"),
        ("http://mcp.example.test:8765/mcp", "http://mcp.example.test/mcp"),
    ])
    def test_a_redirect_to_another_port_without_a_downgrade_is_refused(self, monkeypatch, source, location):
        transport = patch_probe_transport(monkeypatch, respond(307, {"Location": location}))

        with pytest.raises(McpEndpointError) as raised:
            negotiate(source)

        assert str(raised.value) == sso_redirect_message(307, location)
        assert transport.calls == [("POST", source)]

    def test_a_redirect_to_plain_http_on_another_host_is_refused(self, monkeypatch):
        transport = patch_probe_transport(monkeypatch, respond(307, {"Location": "http://login.example.test/sso"}))

        with pytest.raises(McpEndpointError) as raised:
            negotiate(f"{SERVER}/mcp/")

        assert str(raised.value) == sso_redirect_message(307, "http://login.example.test/sso")
        assert transport.calls == [("POST", f"{SERVER}/mcp/")]

    def test_a_cross_origin_redirect_fails_fast_with_the_sso_message(self, monkeypatch):
        transport = patch_probe_transport(monkeypatch, respond(302, {"Location": "https://login.example.test/sso"}))

        with pytest.raises(McpEndpointError) as raised:
            negotiate()

        assert str(raised.value) == sso_redirect_message(302, "https://login.example.test/sso")
        assert transport.calls == [("POST", f"{SERVER}/mcp")]

    def test_a_303_is_treated_as_a_login_redirect_even_on_the_same_origin(self, monkeypatch):
        transport = patch_probe_transport(monkeypatch, routes({
            ("POST", "/mcp"): respond(303, {"Location": f"{SERVER}/mcp/"}),
            ("POST", "/mcp/"): respond(200, JSON),
        }))

        with pytest.raises(McpEndpointError, match="redirected"):
            negotiate()

        assert transport.calls == [("POST", f"{SERVER}/mcp")]

    def test_a_redirect_loop_stops_after_the_hop_limit(self, monkeypatch):
        transport = patch_probe_transport(monkeypatch, respond(307, {"Location": f"{SERVER}/mcp"}))

        with pytest.raises(McpEndpointError, match="redirected"):
            negotiate()

        assert len(transport.calls) == mcp_transport_negotiation.MAX_SAME_ORIGIN_REDIRECTS + 1

    @pytest.mark.parametrize("source, target, expected", [
        ("http://host:8765/mcp", "http://host:8765/mcp/", True),
        ("http://host/mcp", "https://host/mcp", True),
        ("https://host/mcp", "http://host/mcp", False),
        ("http://host:8765/mcp", "http://host:8766/login", False),
        ("https://a.example/mcp", "https://b.example/mcp", False),
        ("https://HOST/mcp", "https://host:443/mcp/", True),
    ])
    def test_same_origin_rule(self, source, target, expected):
        assert is_same_origin_redirect(source, target) is expected


class TestLoginPages:
    def test_a_2xx_html_page_fails_fast_with_the_html_message(self, monkeypatch):
        patch_probe_transport(monkeypatch, respond(200, {"Content-Type": "text/html; charset=utf-8"}))

        with pytest.raises(McpEndpointError) as raised:
            negotiate()

        assert str(raised.value) == html_login_page_message()

    def test_a_same_origin_redirect_ending_on_an_html_page_fails_with_the_html_message(self, monkeypatch):
        patch_probe_transport(monkeypatch, routes({
            ("POST", "/mcp"): respond(302, {"Location": f"{SERVER}/login"}),
            ("POST", "/login"): respond(200, {"Content-Type": "text/html"}),
        }))

        with pytest.raises(McpEndpointError) as raised:
            negotiate()

        assert str(raised.value) == html_login_page_message()

    def test_an_html_404_is_reported_as_missing_not_as_an_sso_proxy(self, monkeypatch):
        patch_probe_transport(monkeypatch, respond(404, {"Content-Type": "text/html"}))

        with pytest.raises(McpEndpointError) as raised:
            negotiate()

        assert str(raised.value) == retired_endpoint_message(f"{SERVER}/mcp", None)


class TestLegacySseFallback:
    def test_a_405_followed_by_an_endpoint_event_selects_legacy_sse(self, monkeypatch):
        transport = patch_probe_transport(monkeypatch, routes({
            ("POST", "/events"): respond(405),
            ("GET", "/events"): endpoint_event("/messages/?session_id=1"),
        }))

        _, decision = negotiate(f"{SERVER}/events")

        assert decision == TransportDecision("sse", f"{SERVER}/events")
        assert transport.calls == [("POST", f"{SERVER}/events"), ("GET", f"{SERVER}/events")]
        assert transport.requests[1].headers["accept"] == "text/event-stream"

    def test_an_endpoint_event_on_another_origin_is_not_accepted(self, monkeypatch):
        patch_probe_transport(monkeypatch, routes({
            ("POST", "/events"): respond(405),
            ("GET", "/events"): endpoint_event("https://elsewhere.example/messages/"),
        }))

        with pytest.raises(ValueError) as raised:
            negotiate(f"{SERVER}/events")

        assert type(raised.value) is ValueError
        assert str(raised.value) == (
            "The MCP server returned an error (405). Please verify the server URL and try again."
        )

    def test_a_retired_sse_endpoint_names_itself_and_suggests_mcp(self, monkeypatch):
        patch_probe_transport(monkeypatch, routes({
            ("POST", "/sse"): respond(405),
            ("GET", "/sse"): respond(410),
        }))

        with pytest.raises(McpEndpointError) as raised:
            negotiate(f"{SERVER}/sse")

        assert str(raised.value) == (
            f"The MCP endpoint {SERVER}/sse has been retired or is no longer available. Try {SERVER}/mcp instead."
        )

    def test_a_missing_non_sse_endpoint_asks_to_verify_the_url(self, monkeypatch):
        patch_probe_transport(monkeypatch, respond(404))

        with pytest.raises(McpEndpointError) as raised:
            negotiate(f"{SERVER}/v2/api")

        assert str(raised.value) == (
            f"The MCP endpoint {SERVER}/v2/api has been retired or is no longer available. "
            "Please verify the server URL in the toolkit settings."
        )

    def test_a_legacy_get_that_never_sends_an_event_is_bounded(self, monkeypatch):
        monkeypatch.setattr(mcp_transport_negotiation, "LEGACY_SSE_PROBE_TIMEOUT_SECONDS", 0.3)

        class Silent(httpx.AsyncByteStream):
            async def __aiter__(self):
                await asyncio.sleep(30)
                yield b""

        async def silent_stream(request):
            return httpx.Response(200, headers=EVENT_STREAM, stream=Silent())

        patch_probe_transport(monkeypatch, routes({("POST", "/mcp"): respond(405), ("GET", "/mcp"): silent_stream}))
        started = time.monotonic()

        with pytest.raises(ValueError, match=r"returned an error \(405\)"):
            negotiate()

        assert time.monotonic() - started < 3

    def test_a_get_only_challenge_after_a_405_starts_the_oauth_flow(self, monkeypatch):
        no_authorization_server_lookup(monkeypatch)
        patch_probe_transport(monkeypatch, routes({
            ("POST", "/sse"): respond(405),
            ("GET", "/sse"): respond(401, {"WWW-Authenticate": 'Bearer realm="mcp"'}),
        }))

        with pytest.raises(McpAuthorizationRequired) as raised:
            negotiate(f"{SERVER}/sse")

        assert raised.value.www_authenticate == 'Bearer realm="mcp"'

    def test_an_explicit_sse_transport_probes_with_get_only(self, monkeypatch):
        transport = patch_probe_transport(monkeypatch, endpoint_event("/messages/"))

        _, decision = negotiate(f"{SERVER}/events", transport="sse")

        assert decision == TransportDecision("sse", f"{SERVER}/events")
        assert transport.calls == [("GET", f"{SERVER}/events")]

    def test_an_explicit_streamable_transport_never_falls_back(self, monkeypatch):
        transport = patch_probe_transport(monkeypatch, respond(405))

        with pytest.raises(ValueError, match=r"returned an error \(405\)"):
            negotiate(transport="streamable_http")

        assert transport.calls == [("POST", f"{SERVER}/mcp")]


class TestGithubBadToken:
    @pytest.mark.parametrize("get_status", [400, 401, 405])
    def test_a_400_with_a_static_token_keeps_the_invalid_token_message(self, monkeypatch, get_status):
        patch_probe_transport(monkeypatch, routes({
            ("POST", "/mcp"): respond(400),
            ("GET", "/mcp"): respond(get_status),
        }))

        with pytest.raises(ValueError) as raised:
            negotiate(headers={"Authorization": "Bearer bad"}, configured_auth=True)

        assert type(raised.value) is ValueError
        assert str(raised.value) == GITHUB_BAD_TOKEN_MESSAGE

    def test_a_400_without_a_static_token_does_not_claim_the_token_is_invalid(self, monkeypatch):
        patch_probe_transport(monkeypatch, respond(400))

        with pytest.raises(ValueError) as raised:
            negotiate()

        assert str(raised.value) == "The MCP server returned an error (400). Please verify the server URL and try again."


class TestUnauthorized:
    def test_a_401_resolves_the_authorization_server_from_resource_metadata(self, monkeypatch):
        no_authorization_server_lookup(monkeypatch)
        metadata_url = f"{SERVER}/.well-known/oauth-protected-resource/mcp"
        patch_probe_transport(monkeypatch, routes({
            ("POST", "/mcp"): respond(401, {"WWW-Authenticate": f'Bearer resource_metadata="{metadata_url}"'}),
            ("GET", "/.well-known/oauth-protected-resource/mcp"): respond(
                200, JSON, b'{"authorization_servers": ["https://auth.example.test/mcp"]}'),
        }))

        with pytest.raises(McpAuthorizationRequired) as raised:
            negotiate()

        assert raised.value.resource_metadata_url == metadata_url
        assert raised.value.resource_metadata["authorization_servers"] == ["https://auth.example.test/mcp"]

    @pytest.mark.parametrize("metadata_origin, expected_verify", [
        (SERVER, False),
        ("https://identity.example.test", True),
        ("http://mcp.example.test", True),
    ])
    def test_disabled_certificate_checks_stay_on_the_configured_host(self, monkeypatch, metadata_origin, expected_verify):
        no_authorization_server_lookup(monkeypatch)
        metadata_url = f"{metadata_origin}/.well-known/oauth-protected-resource"
        transport = patch_probe_transport(monkeypatch, routes({
            ("POST", "/mcp"): respond(401, {"WWW-Authenticate": f'Bearer resource_metadata="{metadata_url}"'}),
            ("GET", "/.well-known/oauth-protected-resource"): respond(200, JSON, b'{"authorization_servers": []}'),
        }))

        with pytest.raises(McpAuthorizationRequired):
            negotiate(ssl_verify=False)

        assert transport.ssl_verify_flags == [False, expected_verify]

    @pytest.mark.parametrize("ssl_verify, followed", [(False, False), (True, True)])
    def test_an_unverified_metadata_fetch_never_follows_a_redirect_off_the_configured_host(
            self, monkeypatch, ssl_verify, followed):
        no_authorization_server_lookup(monkeypatch)
        metadata_url = f"{SERVER}/.well-known/oauth-protected-resource"
        foreign_metadata = "https://attacker.example.test/prm"

        async def handler(request):
            if request.method == "POST":
                return httpx.Response(401, headers={"WWW-Authenticate": f'Bearer resource_metadata="{metadata_url}"'})
            if str(request.url) == metadata_url:
                return httpx.Response(302, headers={"Location": foreign_metadata})
            return httpx.Response(200, json={"authorization_servers": ["https://attacker.example.test"]})

        transport = patch_probe_transport(monkeypatch, handler)

        with pytest.raises(McpAuthorizationRequired) as raised:
            negotiate(ssl_verify=ssl_verify)

        assert (("GET", foreign_metadata) in transport.calls) is followed
        assert (raised.value.resource_metadata["authorization_servers"] == ["https://attacker.example.test"]) is followed

    def test_a_401_with_a_static_token_reports_invalid_credentials(self, monkeypatch):
        patch_probe_transport(monkeypatch, respond(401, {"WWW-Authenticate": 'Bearer error="invalid_token"'}))

        with pytest.raises(ValueError) as raised:
            negotiate(headers={"Authorization": "Bearer bad"}, configured_auth=True)

        assert str(raised.value) == (
            "Authorization failed: Token is invalid or has expired. Please check the credentials in the toolkit settings."
        )

    def test_a_401_from_the_real_transport_starts_the_oauth_flow(self, monkeypatch):
        no_authorization_server_lookup(monkeypatch)
        patch_probe_transport(monkeypatch, respond(401, {"WWW-Authenticate": 'Bearer realm="mcp"'}))
        client = UnifiedMcpClient(url=f"{SERVER}/mcp", timeout=10, transport="streamable_http")
        monkeypatch.setattr(client, "_preflight_auth_check", lambda: asyncio.sleep(0))

        async def connect():
            async with client:
                pass

        with pytest.raises(McpAuthorizationRequired) as raised:
            run(connect())

        assert raised.value.www_authenticate == 'Bearer realm="mcp"'

    def test_a_nested_unauthorized_status_error_is_found(self):
        request = httpx.Request("POST", f"{SERVER}/mcp")
        unauthorized = httpx.Response(401, request=request)
        error = httpx.HTTPStatusError("401", request=request, response=unauthorized)

        found = find_unauthorized_response(BaseExceptionGroup("outer", [BaseExceptionGroup("inner", [error])]))

        assert found is unauthorized


class TestProbeFailures:
    def test_a_network_error_falls_back_to_the_url_suffix(self, monkeypatch):
        async def refuse(request):
            raise httpx.ConnectError("connection refused", request=request)

        patch_probe_transport(monkeypatch, refuse)

        client, decision = negotiate(f"{SERVER}/sse")

        assert decision is None
        assert client._detect_transport() == "sse"


class TestUserFacingMessages:
    CURATED = [
        sso_redirect_message(302, "https://login.example.test/sso"),
        html_login_page_message(),
        retired_endpoint_message(f"{SERVER}/sse", f"{SERVER}/mcp"),
        retired_endpoint_message(f"{SERVER}/v2", None),
    ]

    @pytest.mark.parametrize("message", CURATED)
    def test_the_mapper_passes_a_curated_error_through_unchanged(self, message):
        assert extract_user_friendly_mcp_error(McpEndpointError(message)) == message

    @pytest.mark.parametrize("message", CURATED)
    def test_the_mapper_passes_a_curated_message_string_through_unchanged(self, message):
        assert extract_user_friendly_mcp_error(Exception(message), {"Authorization": "x"}) == message

    @pytest.mark.parametrize("message", CURATED)
    def test_a_curated_message_never_triggers_the_ui_oauth_substitution(self, message):
        lowered = message.lower()
        assert not any(word in lowered for word in ("401", "403", "unauthorized", "forbidden"))

    @pytest.mark.parametrize("url, expected", [
        ("https://mcp.deepwiki.com/sse", "https://mcp.deepwiki.com/mcp"),
        ("https://mcp.linear.app/sse/", "https://mcp.linear.app/mcp"),
        ("https://host/v1/sse?x=1", "https://host/v1/mcp"),
        ("https://host/events", None),
    ])
    def test_the_replacement_url_swaps_the_sse_suffix(self, url, expected):
        assert current_transport_url(url) == expected


def test_resource_metadata_is_fetched_with_the_given_httpx_client():
    async def serve(request):
        return httpx.Response(200, json={"authorization_servers": ["https://as.example"]})

    async def fetch():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            return await fetch_resource_metadata_async(f"{SERVER}/.well-known/oauth-protected-resource", session=client)

    assert run(fetch()) == {"authorization_servers": ["https://as.example"]}


def test_test_connection_probes_the_current_atlassian_endpoint(monkeypatch):
    from elitea_sdk.runtime.clients.client import EliteAClient
    from elitea_sdk.runtime.utils import mcp_adapter

    probed = []

    class RecordingClient:
        def __init__(self, url, **kwargs):
            probed.append(url)

        async def __aenter__(self):
            raise ValueError("stop after construction")

        async def __aexit__(self, *exc_info):
            return False

    monkeypatch.setattr(mcp_adapter, "UnifiedMcpClient", RecordingClient)

    result = EliteAClient.test_mcp_connection(
        object.__new__(EliteAClient), {"settings": {"url": "https://mcp.atlassian.com/v1/sse"}},
    )

    assert probed == ["https://mcp.atlassian.com/v1/mcp/authv2"]
    assert result["success"] is False
