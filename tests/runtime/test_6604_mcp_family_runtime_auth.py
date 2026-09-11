"""Regression coverage for silent MCP auth-family reuse during runtime loading."""

from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool

from elitea_sdk.runtime.toolkits import tools as runtime_tools
from elitea_sdk.runtime.toolkits.mcp_config import McpConfigToolkit
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired, McpContext


AUTHORIZATION_SERVER = "https://login.example.test/oauth2/default"
SOURCE_URL = "https://mcp.example.test/mcp/staffing"
TARGET_URL = "https://mcp.example.test/mcp/search"
FAMILY_ID = f"https://mcp.example.test|{AUTHORIZATION_SERVER}"


@pytest.fixture(autouse=True)
def _disable_unrelated_remote_tool_inventory(monkeypatch):
    monkeypatch.setattr(runtime_tools, "_mcp_tools", lambda *_args, **_kwargs: [])


def _family_token(scopes=None):
    return {
        "access_token": "family-access-token",
        "auth_family_id": FAMILY_ID,
        "authorization_server": AUTHORIZATION_SERVER,
        "resource_server_url": SOURCE_URL,
        "resource_scopes": scopes or [],
    }


def _target_auth_error(scopes=None):
    return McpAuthorizationRequired(
        message="Authorization required",
        server_url=TARGET_URL,
        resource_metadata={
            "authorization_servers": [AUTHORIZATION_SERVER],
            "scopes_supported": scopes or [],
        },
    )


def _real_tool():
    return StructuredTool.from_function(
        func=lambda: "ok",
        name="search",
        description="Search the target MCP server",
    )


def test_static_mcp_prefers_canonical_url_token_over_name_alias(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    loaded = McpConfigToolkit._load_http_tools(
        server_name="Search",
        server_config={"type": "http", "url": TARGET_URL},
        user_config={},
        selected_tools=None,
        excluded_tools=None,
        toolkit_name="Search",
        toolkit_type="mcp_Search",
        mcp_tokens={
            "mcp_Search": {"access_token": "stale-alias-token"},
            TARGET_URL: {"access_token": "target-url-token"},
        },
    )

    assert calls[0]["headers"] == {"Authorization": "Bearer target-url-token"}
    assert {tool.name for tool in loaded} == {"search"}


def test_static_mcp_skips_empty_alias_and_uses_canonical_url_token(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    loaded = McpConfigToolkit._load_http_tools(
        server_name="Search",
        server_config={"type": "http", "url": TARGET_URL},
        user_config={},
        selected_tools=None,
        excluded_tools=None,
        toolkit_name="Search",
        toolkit_type="mcp_Search",
        mcp_tokens={
            "mcp_Search": {"access_token": None, "refresh_token": "refresh-token"},
            TARGET_URL: {"access_token": "target-url-token"},
        },
    )

    assert calls[0]["headers"] == {"Authorization": "Bearer target-url-token"}
    assert {tool.name for tool in loaded} == {"search"}


def test_static_mcp_keeps_historical_name_alias_when_url_token_is_absent(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    loaded = McpConfigToolkit._load_http_tools(
        server_name="Search",
        server_config={"type": "http", "url": TARGET_URL},
        user_config={},
        selected_tools=None,
        excluded_tools=None,
        toolkit_name="Search",
        toolkit_type="mcp_Search",
        mcp_tokens={"mcp_Search": {"access_token": "name-alias-token"}},
    )

    assert calls[0]["headers"] == {"Authorization": "Bearer name-alias-token"}
    assert {tool.name for tool in loaded} == {"search"}


def test_missing_token_keeps_the_first_login_guard(monkeypatch):
    monkeypatch.setattr(
        runtime_tools.McpToolkit,
        "get_toolkit",
        lambda **_kwargs: (_ for _ in ()).throw(_target_auth_error()),
    )

    loaded = runtime_tools.get_tools(
        [{"type": "mcp", "toolkit_name": "Search", "settings": {"url": TARGET_URL}}]
    )

    assert any(tool.name.startswith("mcp_authorize_") for tool in loaded)
    assert any(tool.name == "mcp_auth_control" for tool in loaded)


def test_direct_target_token_remains_available_across_consecutive_turns(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        assert (kwargs.get("headers") or {}).get("Authorization") == "Bearer target-access-token"
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)
    context = McpContext(tokens={TARGET_URL: {"access_token": "target-access-token"}})

    loaded_turns = [
        runtime_tools.get_tools(
            [{"type": "mcp", "toolkit_name": "Search", "settings": {"url": TARGET_URL}}],
            mcp_context=context,
        )
        for _ in range(2)
    ]

    assert len(calls) == 2
    assert all({tool.name for tool in loaded} == {"search"} for loaded in loaded_turns)


def test_remote_mcp_reuses_compatible_family_token_on_consecutive_turns_without_auth_guard(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        authorization = (kwargs.get("headers") or {}).get("Authorization")
        if authorization != "Bearer family-access-token":
            raise _target_auth_error()
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    loaded_turns = [
        runtime_tools.get_tools(
            [{"type": "mcp", "toolkit_name": "Search", "settings": {"url": TARGET_URL}}],
            mcp_context=McpContext(tokens={SOURCE_URL: _family_token()}),
        )
        for _ in range(2)
    ]

    assert len(calls) == 4
    assert calls[1]["headers"] == {"Authorization": "Bearer family-access-token"}
    assert calls[1]["_oauth_token_injected"] is True
    assert calls[3]["headers"] == {"Authorization": "Bearer family-access-token"}
    assert all({tool.name for tool in loaded} == {"search"} for loaded in loaded_turns)


def test_preconfigured_mcp_reuses_compatible_family_token_before_building_auth_guard(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        target_token = (kwargs.get("mcp_tokens") or {}).get(TARGET_URL)
        if not target_token:
            raise _target_auth_error()
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpConfigToolkit, "get_toolkit", get_toolkit)

    loaded = runtime_tools.get_tools(
        [
            {
                "type": "mcp_search",
                "toolkit_name": "Search",
                "settings": {
                    "server_name": "search",
                    "server_config": {"type": "http", "url": TARGET_URL},
                },
            }
        ],
        mcp_context=McpContext(tokens={SOURCE_URL: _family_token()}),
    )

    assert len(calls) == 2
    assert calls[1]["mcp_tokens"][TARGET_URL] == {"access_token": "family-access-token"}
    assert {tool.name for tool in loaded} == {"search"}


def test_family_reuse_rejects_different_resource_origin(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        raise _target_auth_error()

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)
    token = _family_token()
    token["resource_server_url"] = "https://other.example.test/mcp/staffing"
    token["auth_family_id"] = f"https://other.example.test|{AUTHORIZATION_SERVER}"

    loaded = runtime_tools.get_tools(
        [{"type": "mcp", "toolkit_name": "Search", "settings": {"url": TARGET_URL}}],
        mcp_context=McpContext(tokens={SOURCE_URL: token}),
    )

    assert len(calls) == 1
    assert any(tool.name.startswith("mcp_authorize_") for tool in loaded)
    assert any(tool.name == "mcp_auth_control" for tool in loaded)


def test_family_reuse_rejects_different_authorization_server(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        raise _target_auth_error()

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)
    token = _family_token()
    token["authorization_server"] = "https://other-login.example.test/oauth2/default"
    token["auth_family_id"] = (
        "https://mcp.example.test|https://other-login.example.test/oauth2/default"
    )

    loaded = runtime_tools.get_tools(
        [{"type": "mcp", "toolkit_name": "Search", "settings": {"url": TARGET_URL}}],
        mcp_context=McpContext(tokens={SOURCE_URL: token}),
    )

    assert len(calls) == 1
    assert any(tool.name.startswith("mcp_authorize_") for tool in loaded)


def test_family_reuse_rejects_insufficient_resource_scopes(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        raise _target_auth_error(["search.read", "search.write"])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    loaded = runtime_tools.get_tools(
        [{"type": "mcp", "toolkit_name": "Search", "settings": {"url": TARGET_URL}}],
        mcp_context=McpContext(tokens={SOURCE_URL: _family_token(["search.read"])}),
    )

    assert len(calls) == 1
    assert any(tool.name.startswith("mcp_authorize_") for tool in loaded)


def test_rejected_family_token_falls_back_to_normal_auth_guard(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        raise _target_auth_error()

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    loaded = runtime_tools.get_tools(
        [{"type": "mcp", "toolkit_name": "Search", "settings": {"url": TARGET_URL}}],
        mcp_context=McpContext(tokens={SOURCE_URL: _family_token()}),
    )

    assert len(calls) == 2
    assert any(tool.name.startswith("mcp_authorize_") for tool in loaded)
    assert any(tool.name == "mcp_auth_control" for tool in loaded)
