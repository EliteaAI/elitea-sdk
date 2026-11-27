"""#7643: a rejected browser token falls back to the team's saved MCP login.

Main adds `fallback_access_token` next to a browser token when the toolkit has a saved login.
If the server answers 401 to the browser token, the runtime retries once with the fallback;
without one, or when the fallback is rejected too, the usual deferred sign-in applies.
"""
from types import SimpleNamespace

from elitea_sdk.runtime.toolkits import tools as runtime_tools
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired, McpContext

URL = "https://mcp.example.test/mcp"


def _tool():
    return {"type": "mcp", "toolkit_name": "remote", "name": "remote",
            "settings": {"url": URL, "headers": {}, "selected_tools": []}}


def _patch(monkeypatch, accepted):
    seen = []

    class FakeToolkit:
        def __init__(self, tools):
            self._tools = tools

        def get_tools(self):
            return self._tools

    def get_toolkit(**kwargs):
        auth = (kwargs.get("headers") or {}).get("Authorization", "")
        seen.append(auth)
        if auth not in accepted:
            raise McpAuthorizationRequired(message="auth", server_url=URL)
        return FakeToolkit([SimpleNamespace(name="whoami", metadata={})])

    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", staticmethod(get_toolkit))
    monkeypatch.setattr(runtime_tools, "_build_deferred_mcp_auth_tools",
                        lambda *a, **k: [SimpleNamespace(name="deferred_auth", metadata={})])
    monkeypatch.setattr(runtime_tools, "_inject_display_metadata", lambda *a, **k: None)
    return seen


def _run(tokens):
    return [t.name for t in runtime_tools.get_tools([_tool()], mcp_context=McpContext(tokens=tokens))
            if t.name in ("whoami", "deferred_auth")]


def test_rejected_browser_token_retries_with_the_saved_login(monkeypatch):
    seen = _patch(monkeypatch, accepted={"Bearer saved"})

    assert _run({URL: {"access_token": "stale", "fallback_access_token": "saved"}}) == ["whoami"]
    assert seen == ["Bearer stale", "Bearer saved"]


def test_valid_browser_token_never_uses_the_fallback(monkeypatch):
    seen = _patch(monkeypatch, accepted={"Bearer fresh"})

    assert _run({URL: {"access_token": "fresh", "fallback_access_token": "saved"}}) == ["whoami"]
    assert seen == ["Bearer fresh"]


def test_rejected_fallback_ends_in_the_usual_sign_in(monkeypatch):
    seen = _patch(monkeypatch, accepted=set())

    assert _run({URL: {"access_token": "stale", "fallback_access_token": "saved"}}) == ["deferred_auth"]
    assert seen == ["Bearer stale", "Bearer saved"]


def test_no_fallback_keeps_todays_behaviour(monkeypatch):
    seen = _patch(monkeypatch, accepted=set())

    assert _run({URL: {"access_token": "stale"}}) == ["deferred_auth"]
    assert seen == ["Bearer stale"]
