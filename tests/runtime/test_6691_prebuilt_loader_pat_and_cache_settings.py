"""Prebuilt (`mcp_*`) toolkits follow the same rules as Remote MCP ones (#6691).

`_load_http_tools` must keep a configured Authorization header over an OAuth token, tell
the toolkit when the token was injected, and forward the caching settings; toolkits whose
forms have no Cache TTL control only cache when their definition opts in.
"""

from types import SimpleNamespace

from langchain_core.tools import StructuredTool

from elitea_sdk.runtime.toolkits import tools as runtime_tools
from elitea_sdk.runtime.toolkits.mcp_config import McpConfigToolkit

TARGET_URL = "https://mcp.example.test/mcp/search"


def _real_tool():
    return StructuredTool.from_function(func=lambda: "ok", name="search", description="Search")


def _capture_get_toolkit(monkeypatch):
    calls = []

    def get_toolkit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(get_tools=lambda: [_real_tool()])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", get_toolkit)
    return calls


def _load(server_config, user_config=None, mcp_tokens=None):
    return McpConfigToolkit._load_http_tools(
        server_name="Search",
        server_config=server_config,
        user_config=user_config or {},
        selected_tools=None,
        excluded_tools=None,
        toolkit_name="Search",
        toolkit_type="mcp_Search",
        mcp_tokens=mcp_tokens or {},
    )


def test_a_configured_pat_wins_over_the_oauth_token(monkeypatch):
    calls = _capture_get_toolkit(monkeypatch)

    _load(
        {"type": "http", "url": TARGET_URL, "headers": {"authorization": "Bearer configured-pat"}},
        mcp_tokens={TARGET_URL: {"access_token": "oauth-token"}},
    )

    assert calls[0]["headers"] == {"authorization": "Bearer configured-pat"}
    assert calls[0]["_oauth_token_injected"] is False


def test_an_oauth_token_fills_an_absent_header_and_is_flagged(monkeypatch):
    calls = _capture_get_toolkit(monkeypatch)

    _load(
        {"type": "http", "url": TARGET_URL},
        mcp_tokens={TARGET_URL: {"access_token": "oauth-token", "token_type": "Bearer"}},
    )

    assert calls[0]["headers"] == {"Authorization": "Bearer oauth-token"}
    assert calls[0]["_oauth_token_injected"] is True


def test_a_blank_pat_field_yields_to_the_oauth_token(monkeypatch):
    calls = _capture_get_toolkit(monkeypatch)

    _load(
        {"type": "http", "url": TARGET_URL, "headers": {"Authorization": "Bearer {github_token}"}},
        user_config={"github_token": ""},
        mcp_tokens={TARGET_URL: {"access_token": "oauth-token"}},
    )

    assert calls[0]["headers"] == {"Authorization": "Bearer oauth-token"}
    assert calls[0]["_oauth_token_injected"] is True


def test_caching_settings_are_forwarded_from_the_user_config(monkeypatch):
    calls = _capture_get_toolkit(monkeypatch)

    _load({"type": "http", "url": TARGET_URL}, user_config={"enable_caching": False, "cache_ttl": 0})

    assert calls[0]["enable_caching"] is False
    assert calls[0]["cache_ttl"] == 0


def test_caching_is_opt_in_for_toolkits_without_a_cache_control(monkeypatch):
    calls = _capture_get_toolkit(monkeypatch)

    _load({"type": "http", "url": TARGET_URL, "cache_ttl": 900})
    _load({"type": "http", "url": TARGET_URL})
    _load({"type": "http", "url": TARGET_URL, "enable_caching": True, "cache_ttl": 900})

    assert (calls[0]["enable_caching"], calls[0]["cache_ttl"]) == (False, 900)
    assert (calls[1]["enable_caching"], calls[1]["cache_ttl"]) == (False, 300)
    assert (calls[2]["enable_caching"], calls[2]["cache_ttl"]) == (True, 900)


def test_a_credential_containing_braces_is_kept_on_the_prebuilt_path(monkeypatch):
    calls = _capture_get_toolkit(monkeypatch)

    _load({"type": "http", "url": TARGET_URL, "headers": {"Authorization": "Bearer x}{y"}})

    assert calls[0]["headers"] == {"Authorization": "Bearer x}{y"}


def test_an_unresolved_template_is_dropped_even_when_a_personal_token_exists(monkeypatch):
    calls = _capture_get_toolkit(monkeypatch)

    _load(
        {"type": "http", "url": TARGET_URL, "headers": {"Authorization": "Bearer {api_key}", "X-A": "1"}},
        user_config={"personal_token": "pat-123"},
    )

    assert calls[0]["headers"] == {"X-A": "1"}
