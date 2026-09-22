"""Regression coverage for #6691's Cache TTL fix on the remote-MCP discovery path.

Pins: a registered discovery-cache backend is actually consulted (read before
discover, write after discover, filtered on read not on write), TTL/enable_caching
are honoured, a bad/raising backend never breaks discovery, and the cache key is a
url+headers+ssl_verify digest that never leaks a credential or the server URL.
Re-breaks if any of those steps is skipped, reordered, or the key stops binding to
the connection identity.
"""

import json
import re
import uuid

import asyncio

import pytest

from elitea_sdk.runtime.models.mcp_models import McpConnectionConfig
from elitea_sdk.runtime.toolkits import tools as runtime_tools
from elitea_sdk.runtime.toolkits import mcp as mcp_toolkit_module
from elitea_sdk.runtime.toolkits.mcp import McpToolkit
from elitea_sdk.runtime.tools.mcp_remote_tool import McpRemoteTool
from elitea_sdk.runtime.utils.mcp_discovery_cache import (
    CACHE_TTL_MAX,
    GENERATION_TTL,
    build_discovery_cache_key,
    clamp_cache_ttl,
    invalidate_server_discovery,
    register_discovery_cache_backend,
)
from elitea_sdk.runtime.tools import mcp_remote_tool
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired, McpContext
from tests.runtime.utils.mcp_rig_server import RunningRig

URL = "https://mcp.example.test/mcp"
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def _tool_dict(name):
    return {"name": name, "description": f"{name} tool", "inputSchema": {"type": "object"}}


def _counting_discover_stub(batches, session_id="server-session"):
    """Build a classmethod-compatible `_discover_tools_sync` replacement.

    `batches[i]` is either a list of tool dicts (returned as-is with `session_id`)
    or an exception instance, raised instead. Beyond the last entry, the last
    batch repeats.
    """
    state = {"calls": 0}

    def _stub(cls, **kwargs):  # pylint: disable=unused-argument
        state["calls"] += 1
        batch = batches[min(state["calls"] - 1, len(batches) - 1)]
        if isinstance(batch, BaseException):
            raise batch
        return [dict(tool) for tool in batch], session_id

    return _stub, state


class _Clock:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now


class FakeCacheBackend:
    """In-memory backend implementing McpDiscoveryCacheBackend with a real TTL."""

    def __init__(self, clock):
        self.clock = clock
        self.store = {}

    def get(self, key):
        entry = self.store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if self.clock() >= expires_at:
            self.store.pop(key, None)
            return None
        return value

    def set(self, key, value, ttl):
        self.store[key] = (value, self.clock() + ttl)


class _RaisingBackend:
    def get(self, key):
        raise ConnectionError("redis down")

    def set(self, key, value, ttl):
        raise ConnectionError("redis down")


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
def cache_backend(clock):
    backend = FakeCacheBackend(clock)
    register_discovery_cache_backend(backend)
    yield backend
    register_discovery_cache_backend(None)


def test_second_toolkit_within_ttl_performs_no_discovery(monkeypatch, cache_backend):
    stub, calls = _counting_discover_stub([[_tool_dict("echo")]])
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))

    toolkits = [
        McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=300, client=None)
        for _ in range(2)
    ]

    assert calls["calls"] == 1
    for toolkit in toolkits:
        tools = toolkit.get_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], McpRemoteTool)
        assert tools[0].session_id


def test_run_after_ttl_expiry_discovers_again(monkeypatch, cache_backend, clock):
    stub, calls = _counting_discover_stub([[_tool_dict("echo")], [_tool_dict("echo")]])
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))

    McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=60, client=None)
    clock.now += 61
    McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=60, client=None)

    assert calls["calls"] == 2


def test_enable_caching_false_discovers_every_time_and_writes_nothing(monkeypatch, cache_backend):
    stub, calls = _counting_discover_stub([[_tool_dict("echo")], [_tool_dict("echo")]])
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))

    for _ in range(2):
        McpToolkit.get_toolkit(url=URL, toolkit_name="t1", enable_caching=False, cache_ttl=300, client=None)

    assert calls["calls"] == 2
    assert cache_backend.store == {}


def test_cache_ttl_zero_disables_caching_instead_of_clamping_to_the_minimum(monkeypatch, cache_backend):
    stub, calls = _counting_discover_stub([[_tool_dict("echo")], [_tool_dict("echo")]])
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))

    for _ in range(2):
        McpToolkit.get_toolkit(url=URL, toolkit_name="t1", enable_caching=True, cache_ttl=0, client=None)

    assert calls["calls"] == 2
    assert cache_backend.store == {}


def test_discovery_failure_writes_nothing_to_cache(monkeypatch, cache_backend):
    stub, _ = _counting_discover_stub([RuntimeError("boom")])
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))

    with pytest.raises(RuntimeError):
        McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=300, client=None)

    assert cache_backend.store == {}


def test_discovery_auth_required_writes_nothing_to_cache(monkeypatch, cache_backend):
    stub, _ = _counting_discover_stub([McpAuthorizationRequired(message="auth required", server_url=URL)])
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))

    with pytest.raises(McpAuthorizationRequired):
        McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=300, client=None)

    assert cache_backend.store == {}


def test_discovery_returning_no_tools_writes_nothing_and_next_call_discovers_again(monkeypatch, cache_backend):
    stub, calls = _counting_discover_stub([[], []])
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))

    first = McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=300, client=None)
    second = McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=300, client=None)

    assert first.get_tools() == []
    assert second.get_tools() == []
    assert calls["calls"] == 2
    assert cache_backend.store == {}


def test_different_authorization_header_values_use_different_cache_keys(monkeypatch, cache_backend):
    stub, calls = _counting_discover_stub([[_tool_dict("echo")], [_tool_dict("echo")]])
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))

    McpToolkit.get_toolkit(
        url=URL, toolkit_name="t1", cache_ttl=300, client=None, headers={"Authorization": "Bearer token-a"}
    )
    McpToolkit.get_toolkit(
        url=URL, toolkit_name="t1", cache_ttl=300, client=None, headers={"Authorization": "Bearer token-b"}
    )

    assert calls["calls"] == 2
    assert len(cache_backend.store) == 2


def test_selected_tools_filters_on_read_not_on_write(monkeypatch, cache_backend):
    stub, calls = _counting_discover_stub([[_tool_dict("echo"), _tool_dict("ping")]])
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))

    first = McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=300, client=None, selected_tools=["echo"])
    second = McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=300, client=None, selected_tools=[])
    third = McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=300, client=None, selected_tools=["ping"])

    assert calls["calls"] == 1
    assert {t.name for t in first.get_tools()} == {"echo"}
    assert {t.name for t in second.get_tools()} == {"echo", "ping"}
    assert {t.name for t in third.get_tools()} == {"ping"}


def test_cache_hit_uses_the_supplied_session_id_not_a_fresh_uuid(monkeypatch, cache_backend):
    stub, _ = _counting_discover_stub([[_tool_dict("echo")]], session_id="discovery-session")
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))

    McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=300, client=None, session_id="abc")
    cached = McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=300, client=None, session_id="abc")

    tools = cached.get_tools()
    assert len(tools) == 1
    assert tools[0].session_id == "abc"


def test_backend_get_and_set_failures_fall_back_to_live_discovery_without_raising(monkeypatch):
    register_discovery_cache_backend(_RaisingBackend())
    try:
        stub, calls = _counting_discover_stub([[_tool_dict("echo")]])
        monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))

        toolkit = McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=300, client=None)

        assert calls["calls"] == 1
        assert len(toolkit.get_tools()) == 1
    finally:
        register_discovery_cache_backend(None)


@pytest.mark.parametrize("raw_entry", ["nope", json.dumps({"v": 1, "tools": "x"})])
def test_malformed_cache_entry_falls_back_to_live_discovery(monkeypatch, cache_backend, clock, raw_entry):
    key = build_discovery_cache_key(URL, None, True)
    cache_backend.store[key] = (raw_entry, clock() + 300)
    stub, calls = _counting_discover_stub([[_tool_dict("echo")]])
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))

    toolkit = McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=300, client=None)

    assert calls["calls"] == 1
    assert len(toolkit.get_tools()) == 1


def test_get_toolkit_passes_clamped_ttl_and_caching_flag_through(monkeypatch):
    captured = {}

    def fake_create_tools_from_server(cls, **kwargs):  # pylint: disable=unused-argument
        captured.update(kwargs)
        return []

    monkeypatch.setattr(McpToolkit, "_create_tools_from_server", classmethod(fake_create_tools_from_server))

    McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl="600", enable_caching=False, client=None)

    assert captured["cache_ttl"] == 600
    assert captured["enable_caching"] is False


def test_cache_key_is_a_hex_digest_without_token_or_url():
    key = build_discovery_cache_key(URL, {"Authorization": "Bearer super-secret-token"}, True)

    assert re.fullmatch(r"[0-9a-f]{64}", key)
    assert "super-secret-token" not in key
    assert "mcp.example.test" not in key


def test_cache_key_ignores_header_case_and_order_and_url_case_and_trailing_slash():
    key_a = build_discovery_cache_key(
        "https://MCP.example.test/mcp/", {"authorization": "Bearer t", "X-Foo": "1"}, True
    )
    key_b = build_discovery_cache_key(
        "https://mcp.example.test/mcp", {"X-Foo": "1", "Authorization": "Bearer t"}, True
    )

    assert key_a == key_b


def test_cache_key_is_sensitive_to_ssl_verify():
    headers = {"Authorization": "Bearer t"}

    assert build_discovery_cache_key(URL, headers, True) != build_discovery_cache_key(URL, headers, False)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("300", 300),
        (10, 60),
        (99999, 3600),
        ("abc", 300),
        (None, 300),
        (0, 0),
        (-5, 300),
    ],
)
def test_clamp_cache_ttl_bounds(value, expected):
    assert clamp_cache_ttl(value) == expected


@pytest.fixture(scope="module")
def rig():
    with RunningRig() as running:
        yield running


def test_second_get_toolkit_against_the_rig_performs_no_additional_http_requests(rig, monkeypatch, cache_backend):
    for variable in PROXY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")

    url = f"{rig.base_url}/schemas/mcp/"
    McpToolkit.get_toolkit(url=url, toolkit_name="schemas", timeout=20, cache_ttl=300, session_id=str(uuid.uuid4()))
    requests_after_first_call = len(rig.requests)

    McpToolkit.get_toolkit(url=url, toolkit_name="schemas", timeout=20, cache_ttl=300, session_id=str(uuid.uuid4()))

    assert len(rig.requests) == requests_after_first_call


def _toolkit_with(monkeypatch, headers, **kwargs):
    stub, state = _counting_discover_stub([[_tool_dict("echo")]])
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))
    kwargs.setdefault("cache_ttl", 120)
    toolkit = McpToolkit.get_toolkit(url=URL, headers=headers, toolkit_name="t1", **kwargs)
    return toolkit, state


def test_load_tools_invalidation_retires_every_credentials_entry(monkeypatch, cache_backend):
    headers_a = {"Authorization": "Bearer user-a"}
    headers_b = {"Authorization": "Bearer user-b"}
    _toolkit_with(monkeypatch, headers_a)
    _, state = _toolkit_with(monkeypatch, headers_b)
    assert state["calls"] == 1

    invalidate_server_discovery(URL)

    _, state_a = _toolkit_with(monkeypatch, headers_a)
    _, state_b = _toolkit_with(monkeypatch, headers_b)
    assert (state_a["calls"], state_b["calls"]) == (1, 1)


def test_entries_written_after_invalidation_are_served_again(monkeypatch, cache_backend):
    invalidate_server_discovery(URL)
    _toolkit_with(monkeypatch, None)
    _, state = _toolkit_with(monkeypatch, None)

    assert state["calls"] == 0


def test_load_tools_retires_entries_written_under_either_tls_setting(monkeypatch, cache_backend):
    _toolkit_with(monkeypatch, None, ssl_verify=True)
    _toolkit_with(monkeypatch, None, ssl_verify=False)

    invalidate_server_discovery(URL)

    _, state_verified = _toolkit_with(monkeypatch, None, ssl_verify=True)
    _, state_unverified = _toolkit_with(monkeypatch, None, ssl_verify=False)
    assert (state_verified["calls"], state_unverified["calls"]) == (1, 1)


def test_load_tools_after_the_generation_expired_does_not_revive_a_retired_entry(monkeypatch, cache_backend, clock):
    invalidate_server_discovery(URL)
    clock.now = GENERATION_TTL - 600
    _toolkit_with(monkeypatch, None, cache_ttl=CACHE_TTL_MAX)
    clock.now = GENERATION_TTL + 1

    invalidate_server_discovery(URL)

    _, state = _toolkit_with(monkeypatch, None)
    assert state["calls"] == 1


def _discovery_straddling_load_tools(clock, finishes_at):
    def _stub(cls, **kwargs):  # pylint: disable=unused-argument
        clock.now += 10
        invalidate_server_discovery(URL)
        clock.now = finishes_at
        return [_tool_dict("stale")], "server-session"

    return _stub


def test_a_discovery_straddling_load_tools_is_not_served_after_it(monkeypatch, cache_backend, clock):
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(_discovery_straddling_load_tools(clock, 50)))
    McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=CACHE_TTL_MAX, client=None)

    _, state = _toolkit_with(monkeypatch, None)

    assert state["calls"] == 1


def test_a_discovery_straddling_load_tools_stays_retired_once_the_generation_would_have_expired(
    monkeypatch, cache_backend, clock
):
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(_discovery_straddling_load_tools(clock, 50)))
    McpToolkit.get_toolkit(url=URL, toolkit_name="t1", cache_ttl=CACHE_TTL_MAX, client=None)
    clock.now = CACHE_TTL_MAX + 40

    _, state = _toolkit_with(monkeypatch, None)

    assert state["calls"] == 1


def test_invalidation_failure_is_reported_not_raised(monkeypatch):
    register_discovery_cache_backend(_RaisingBackend())
    try:
        assert invalidate_server_discovery(URL) is False
    finally:
        register_discovery_cache_backend(None)


def test_invalidation_reports_success_with_a_working_backend_and_with_none(cache_backend):
    assert invalidate_server_discovery(URL) is True
    register_discovery_cache_backend(None)
    assert invalidate_server_discovery(URL) is True


@pytest.mark.parametrize(
    ("headers", "kwargs", "expected"),
    [
        ({"Authorization": "Bearer configured-pat"}, {}, True),
        ({"authorization": "Bearer configured-pat"}, {}, True),
        ({"Authorization": "Bearer oauth-token"}, {"_oauth_token_injected": True}, False),
        ({"Authorization": "Bearer {github_token}"}, {}, True),
        ({"Authorization": "Bearer "}, {}, True),
        (None, {}, False),
    ],
)
def test_tools_report_a_configured_credential_whenever_the_header_they_send_was_not_injected(monkeypatch, cache_backend, headers, kwargs, expected):
    _toolkit_with(monkeypatch, headers, **kwargs)
    cached_toolkit, state = _toolkit_with(monkeypatch, headers, **kwargs)
    assert state["calls"] == 0

    assert all(tool.configured_auth is expected for tool in cached_toolkit.get_tools())


def test_tool_calls_pass_the_configured_credential_flag_to_the_client(monkeypatch, cache_backend):
    toolkit, _ = _toolkit_with(monkeypatch, {"Authorization": "Bearer configured-pat"})
    seen = {}

    class _CapturingClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        async def __aenter__(self):
            raise RuntimeError("stop before any network use")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(mcp_remote_tool, "McpClient", _CapturingClient)

    with pytest.raises(RuntimeError, match="stop before"):
        asyncio.run(toolkit.get_tools()[0]._execute_remote_tool({}))

    assert seen["configured_auth"] is True


@pytest.mark.parametrize(
    ("headers", "oauth_token_injected", "expected"),
    [
        ({"Authorization": "Bearer configured-pat"}, False, True),
        ({"Authorization": "Bearer "}, False, True),
        ({"Authorization": "Bearer {github_token}"}, False, True),
        ({"Authorization": "Bearer oauth-token"}, True, False),
        (None, False, False),
    ],
)
def test_discovery_reports_a_configured_credential_whenever_the_header_it_sends_was_not_injected(
    monkeypatch, headers, oauth_token_injected, expected
):
    seen = {}

    class _CapturingClient:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        async def __aenter__(self):
            raise RuntimeError("stop before any network use")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(mcp_toolkit_module, "McpClient", _CapturingClient)
    connection_config = McpConnectionConfig(url=URL, headers=headers)

    with pytest.raises(RuntimeError, match="stop before"):
        asyncio.run(McpToolkit._discover_tools_async(
            toolkit_name="t1",
            toolkit_type="mcp",
            connection_config=connection_config,
            timeout=5,
            oauth_token_injected=oauth_token_injected,
        ))

    assert seen["configured_auth"] is expected


def test_an_internal_style_config_without_enable_caching_still_caches_for_its_ttl(monkeypatch, cache_backend):
    """Auto-injected internal toolkits are `type: mcp` with only `cache_ttl` set; the form default
    (caching on) applies to them, and they have no Load Tools to invalidate — a bounded, stated trade-off."""
    stub, state = _counting_discover_stub([[_tool_dict("echo")]])
    monkeypatch.setattr(McpToolkit, "_discover_tools_sync", classmethod(stub))
    config = {"type": "mcp", "toolkit_name": "Elitea platform MCP", "settings": {
        "url": URL, "headers": {"Authorization": "Bearer system-token"}, "timeout": 300, "cache_ttl": 300}}

    runtime_tools.get_tools([config], mcp_context=McpContext(tokens={}))
    runtime_tools.get_tools([config], mcp_context=McpContext(tokens={}))

    assert state["calls"] == 1


def test_the_cache_ttl_field_tells_the_user_that_zero_disables():
    field = McpToolkit.toolkit_config_schema().model_fields["cache_ttl"]

    assert "0 disables" in field.description


def _reject_with_401(monkeypatch, headers, configured_auth=False):
    from elitea_sdk.runtime.utils import mcp_oauth
    from elitea_sdk.runtime.utils.mcp_adapter import UnifiedMcpClient
    from tests.runtime.utils.mcp_probe_transport import patch_probe_transport, respond

    monkeypatch.setattr(mcp_oauth, "fetch_oauth_authorization_server_metadata", lambda *a, **k: None)
    patch_probe_transport(monkeypatch, respond(401))
    client = UnifiedMcpClient(url=URL, timeout=10, headers=headers, configured_auth=configured_auth)
    with pytest.raises((McpAuthorizationRequired, ValueError)):
        asyncio.run(client._preflight_auth_check())


@pytest.mark.parametrize("configured_auth", [False, True])
def test_a_401_retires_only_the_rejected_credentials_entry(monkeypatch, cache_backend, configured_auth):
    alice = {"Authorization": "Bearer alice-token"}
    bob = {"Authorization": "Bearer bob-token"}
    _toolkit_with(monkeypatch, alice)
    _toolkit_with(monkeypatch, bob)

    _reject_with_401(monkeypatch, alice, configured_auth)

    _, alice_state = _toolkit_with(monkeypatch, alice)
    _, bob_state = _toolkit_with(monkeypatch, bob)
    assert (alice_state["calls"], bob_state["calls"]) == (1, 0)


def test_a_first_login_challenge_from_a_user_without_a_token_leaves_every_entry_alone(monkeypatch, cache_backend):
    alice = {"Authorization": "Bearer alice-token"}
    _toolkit_with(monkeypatch, alice)

    _reject_with_401(monkeypatch, headers=None)

    _, state = _toolkit_with(monkeypatch, alice)
    assert state["calls"] == 0


def test_a_retired_entry_is_rewritten_by_the_next_live_discovery(monkeypatch, cache_backend):
    alice = {"Authorization": "Bearer alice-token"}
    _toolkit_with(monkeypatch, alice)
    _reject_with_401(monkeypatch, alice)

    _toolkit_with(monkeypatch, alice)
    _, state = _toolkit_with(monkeypatch, alice)

    assert state["calls"] == 0
