"""Regression coverage for the runtime MCP loading path honouring a configured PAT
over an OAuth token (#6691 Fault 2), including the family-retry copy of the merge.

Scaffolded like test_6604_mcp_family_runtime_auth.py: `McpToolkit.get_toolkit` is
replaced with a call-recording stub so the headers/flags reaching it can be
asserted directly. Re-breaks if `tools.py` goes back to `setdefault` (duplicate
lower-case header), starts overwriting a configured header unconditionally, drops
the `_oauth_token_injected` flag, or lets the family retry replace a configured PAT.
"""

from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool, ToolException

from elitea_sdk.runtime.toolkits import tools as runtime_tools
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired, McpContext


AUTHORIZATION_SERVER = "https://login.example.test/oauth2/default"
SOURCE_URL = "https://mcp.example.test/mcp/staffing"
TARGET_URL = "https://mcp.example.test/mcp/search"
FAMILY_ID = f"https://mcp.example.test|{AUTHORIZATION_SERVER}"


@pytest.fixture(autouse=True)
def _disable_unrelated_remote_tool_inventory(monkeypatch):
    monkeypatch.setattr(runtime_tools, "_mcp_tools", lambda *_args, **_kwargs: [])


def _family_token():
    return {
        "access_token": "family-access-token",
        "auth_family_id": FAMILY_ID,
        "authorization_server": AUTHORIZATION_SERVER,
        "resource_server_url": SOURCE_URL,
        "resource_scopes": [],
    }


def _target_auth_error():
    return McpAuthorizationRequired(
        message="Authorization required",
        server_url=TARGET_URL,
        resource_metadata={"authorization_servers": [AUTHORIZATION_SERVER], "scopes_supported": []},
    )


def _real_tool():
    return StructuredTool.from_function(func=lambda: "ok", name="search", description="Search the target MCP server")


def _tool_config(headers=None):
    settings = {"url": TARGET_URL}
    if headers is not None:
        settings["headers"] = headers
    return [{"type": "mcp", "toolkit_name": "Search", "settings": settings}]


def test_configured_pat_wins_over_an_oauth_token(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    loaded = runtime_tools.get_tools(
        _tool_config(headers={"Authorization": "Bearer configured-pat"}),
        mcp_context=McpContext(tokens={TARGET_URL: {"access_token": "oauth-token"}}),
    )

    assert len(calls) == 1
    assert calls[0]["headers"] == {"Authorization": "Bearer configured-pat"}
    assert "_oauth_token_injected" not in calls[0]
    assert {tool.name for tool in loaded} == {"search"}


def test_lower_case_configured_pat_is_sent_once_without_a_duplicate_key(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    runtime_tools.get_tools(
        _tool_config(headers={"authorization": "Bearer configured-pat"}),
        mcp_context=McpContext(tokens={TARGET_URL: {"access_token": "oauth-token"}}),
    )

    assert calls[0]["headers"] == {"authorization": "Bearer configured-pat"}
    assert list(calls[0]["headers"].keys()) == ["authorization"]


def test_no_configured_pat_injects_the_oauth_token_and_sets_the_flag(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    runtime_tools.get_tools(
        _tool_config(),
        mcp_context=McpContext(tokens={TARGET_URL: {"access_token": "oauth-token"}}),
    )

    assert calls[0]["headers"] == {"Authorization": "Bearer oauth-token"}
    assert calls[0]["_oauth_token_injected"] is True


def test_family_retry_does_not_replace_a_configured_pat(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        raise _target_auth_error()

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    loaded = runtime_tools.get_tools(
        _tool_config(headers={"Authorization": "Bearer configured-pat"}),
        mcp_context=McpContext(tokens={SOURCE_URL: _family_token()}),
    )

    assert len(calls) == 1
    assert calls[0]["headers"] == {"Authorization": "Bearer configured-pat"}
    assert any(tool.name.startswith("mcp_authorize_") for tool in loaded)
    assert any(tool.name == "mcp_auth_control" for tool in loaded)


def test_family_retry_without_a_configured_pat_uses_the_shared_merge(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        authorization = (kwargs.get("headers") or {}).get("Authorization")
        if authorization != "Bearer family-access-token":
            raise _target_auth_error()
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    loaded = runtime_tools.get_tools(
        _tool_config(),
        mcp_context=McpContext(tokens={SOURCE_URL: _family_token()}),
    )

    assert len(calls) == 2
    assert calls[1]["headers"] == {"Authorization": "Bearer family-access-token"}
    assert calls[1]["_oauth_token_injected"] is True
    assert {tool.name for tool in loaded} == {"search"}


def test_family_retry_replaces_an_unresolved_placeholder_credential(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        authorization = (kwargs.get("headers") or {}).get("Authorization")
        if authorization != "Bearer family-access-token":
            raise _target_auth_error()
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    loaded = runtime_tools.get_tools(
        _tool_config(headers={"Authorization": "Bearer {github_token}"}),
        mcp_context=McpContext(tokens={SOURCE_URL: _family_token()}),
    )

    assert len(calls) == 2
    assert calls[1]["headers"] == {"Authorization": "Bearer family-access-token"}
    assert calls[1]["_oauth_token_injected"] is True
    assert {tool.name for tool in loaded} == {"search"}


def test_an_unresolved_template_without_a_token_is_not_sent_by_the_run(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    runtime_tools.get_tools(
        _tool_config(headers={"Authorization": "Bearer {github_token}", "X-Trace": "1"}),
        mcp_context=McpContext(tokens={}),
    )

    assert calls[0]["headers"] == {"X-Trace": "1"}
    assert "_oauth_token_injected" not in calls[0]


def test_an_unresolved_secret_reference_stays_the_configured_credential_on_the_run(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    runtime_tools.get_tools(
        _tool_config(headers={"Authorization": "Bearer {{secret.github_pat}}"}),
        mcp_context=McpContext(tokens={TARGET_URL: {"access_token": "oauth-token"}}),
    )

    assert calls[0]["headers"] == {"Authorization": "Bearer {{secret.github_pat}}"}
    assert "_oauth_token_injected" not in calls[0]


def test_a_credential_containing_braces_is_sent_as_typed_by_the_run(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    runtime_tools.get_tools(
        _tool_config(headers={"Authorization": "Bearer x}{y"}),
        mcp_context=McpContext(tokens={TARGET_URL: {"access_token": "oauth-token"}}),
    )

    assert calls[0]["headers"] == {"Authorization": "Bearer x}{y"}


def test_headers_stored_as_a_json_string_reach_the_toolkit_instead_of_crashing_the_run(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    runtime_tools.get_tools(
        _tool_config(headers='{"Authorization": "Bearer configured-pat"}'),
        mcp_context=McpContext(tokens={TARGET_URL: {"access_token": "oauth-token"}}),
    )

    assert calls[0]["headers"] == {"Authorization": "Bearer configured-pat"}
    assert "_oauth_token_injected" not in calls[0]


def test_headers_stored_as_an_unparseable_string_still_reach_the_toolkits_own_error(monkeypatch):
    def get_toolkit(**kwargs):
        raise ValueError("Invalid headers JSON format: expecting value")

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    with pytest.raises(ToolException, match="Invalid headers JSON format"):
        runtime_tools.get_tools(
            _tool_config(headers="not json at all"),
            mcp_context=McpContext(tokens={}),
        )


def test_family_retry_injects_the_token_into_headers_stored_as_a_json_string(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        authorization = (kwargs.get("headers") or {}).get("Authorization")
        if authorization != "Bearer family-access-token":
            raise _target_auth_error()
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)

    loaded = runtime_tools.get_tools(
        _tool_config(headers='{"X-Trace": "1"}'),
        mcp_context=McpContext(tokens={SOURCE_URL: _family_token()}),
    )

    assert calls[1]["headers"] == {"X-Trace": "1", "Authorization": "Bearer family-access-token"}
    assert calls[1]["_oauth_token_injected"] is True
    assert {tool.name for tool in loaded} == {"search"}
