"""C-1009 (#7214): a governed Elitea run's internal MCP authenticates with the run token the worker hung on the client.

Only an internal Elitea MCP endpoint (``.../app/<project>/mcp...``) on one of the token's origins gets it, replacing
whatever Authorization its settings carried; any other server, and a run without a token, is left exactly as before.
The discovery cache is keyed by who the run acts for, so a token that changes every run does not defeat it.
"""
from types import SimpleNamespace

import pytest

from elitea_sdk.runtime.toolkits.mcp import McpToolkit
from elitea_sdk.runtime.utils.internal_run_auth import internal_run_headers
from elitea_sdk.runtime.utils.mcp_discovery_cache import register_discovery_cache_backend

TOKEN = "eio1.claims.mac"
INTERNAL = "http://pylon_main:8080/app/5/mcp/elitea_core/applications?scope_project_id=2"


def client(token=TOKEN, origins=("http://pylon_main:8080",), identity="who-1"):
    return SimpleNamespace(internal_run_auth=SimpleNamespace(token=token, origins=tuple(origins), identity=identity))


@pytest.mark.parametrize("url", [
    INTERNAL, "http://PYLON_MAIN:8080/app/2/mcp", "http://pylon_main:8080/app/2/mcp/secrets",
    "http://pylon_main:8080/x/../app/2/mcp/elitea_core/applications", "http://pylon_main:8080/./app/2/mcp",
    "http://pylon_main:8080/app/2/%6Dcp/secrets", "http://pylon_main:8080//app//2/mcp/secrets",
])
def test_an_internal_endpoint_on_an_allowed_origin_gets_the_run_token(url):
    headers, identity = internal_run_headers(client(), url, {"authorization": "Bearer pat", "X-Other": "1"})
    assert headers == {"X-Other": "1", "Authorization": f"Bearer {TOKEN}"} and identity == "who-1"


@pytest.mark.parametrize("url", [
    "https://evil.example/app/5/mcp/elitea_core/applications",      # another origin, same path
    "http://pylon_main:8081/app/5/mcp/secrets",                      # another port
    "https://mcp.example.com/mcp",
    "https://evil.example/x/../app/5/mcp",
])
def test_a_server_on_another_origin_is_left_alone(url):
    original = {"Authorization": "Bearer external"}
    assert internal_run_headers(client(), url, original) == (original, None)


@pytest.mark.parametrize("url", [
    "http://pylon_main:8080/api/v2/elitea_core/applications/prompt_lib/2",   # not an MCP endpoint
    "http://pylon_main:8080/x/app/5/mcp/secrets",                    # the shape must start the path
    "http://pylon_main:8080/app/5/mcpx",
])
def test_another_route_on_the_platform_gets_neither_the_run_token_nor_its_own_credentials(url):
    headers, identity = internal_run_headers(
        client(), url, {"Authorization": "Bearer pat", "cookie": "s=1", "Proxy-Authorization": "x", "X-Trace": "1"})
    assert headers == {"X-Trace": "1"} and identity is None


@pytest.mark.parametrize("path,expected", [
    ("/x/../app/2/mcp", "/app/2/mcp"), ("/app/2/%6Dcp", "/app/2/mcp"), ("//app//2/mcp/", "/app/2/mcp/"),
    ("/app/2/%256Dcp", "/app/2/mcp"), ("", "/"),
])
def test_the_client_path_is_what_the_client_sends(path, expected):
    from elitea_sdk.runtime.utils.internal_run_auth import client_path
    assert client_path(path) == expected


def test_a_run_without_a_token_changes_nothing():
    original = {"Authorization": "Bearer pat"}
    for no_token in (None, SimpleNamespace(), SimpleNamespace(internal_run_auth=None), client(token="")):
        assert internal_run_headers(no_token, INTERNAL, original) == (original, None)


def _discover(calls):
    def stub(cls, **kwargs):  # pylint: disable=unused-argument
        calls.append(kwargs)
        return [{"name": "list_apps", "description": "d", "inputSchema": {"type": "object"}}], "session"
    return classmethod(stub)


@pytest.fixture
def backend():
    store = {}
    cache = SimpleNamespace(get=store.get, set=lambda key, value, ttl: store.__setitem__(key, value), store=store)
    register_discovery_cache_backend(cache)
    yield cache
    register_discovery_cache_backend(None)


def test_the_toolkit_sends_the_run_token_and_keys_its_cache_by_identity(monkeypatch, backend):
    calls = []
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", _discover(calls))
    run_1 = client(token="eio1.run-one.mac")
    toolkit = McpToolkit.get_toolkit(url=INTERNAL, toolkit_name="apps", cache_ttl=300, client=run_1,
                                     headers={"Authorization": "Bearer pat"})
    [tool] = toolkit.get_tools()
    assert tool.server_headers == {"Authorization": "Bearer eio1.run-one.mac"}
    # the next run of the same person and agent: new token, same cache entry, no new discovery
    toolkit = McpToolkit.get_toolkit(url=INTERNAL, toolkit_name="apps", cache_ttl=300,
                                     client=client(token="eio1.run-two.mac"))
    assert len(calls) == 1 and len(backend.store) == 1
    assert toolkit.get_tools()[0].server_headers == {"Authorization": "Bearer eio1.run-two.mac"}
    assert all("eio1" not in key for key in backend.store)


def test_the_cache_identity_is_never_serialized_with_the_connection():
    from elitea_sdk.runtime.models.mcp_models import McpConnectionConfig
    config = McpConnectionConfig(url=INTERNAL, headers={}, cache_identity="who-1")
    assert "cache_identity" not in config.model_dump() and "who-1" not in config.model_dump_json()


def test_another_identity_does_not_share_the_cache_entry(monkeypatch, backend):
    calls = []
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", _discover(calls))
    McpToolkit.get_toolkit(url=INTERNAL, toolkit_name="apps", cache_ttl=300, client=client(identity="who-1"))
    McpToolkit.get_toolkit(url=INTERNAL, toolkit_name="apps", cache_ttl=300, client=client(identity="who-2"))
    assert len(calls) == 2
