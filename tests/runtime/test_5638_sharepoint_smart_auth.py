"""Smart-auth deferred-proxy coverage for built-in delegated-OAuth toolkits (issue #5638).

Background: the "smart MCP auth" flow shows an Authorize/Skip prompt when a toolkit needs
browser OAuth. For remote MCP toolkits, clicking Skip records the decline and the agent
stops re-prompting. For the built-in SharePoint toolkit (delegated OAuth) the loader raised
``McpAuthorizationRequired`` eagerly and unconditionally at tool-load time, from the batch
``elitea_tools`` path that sits OUTSIDE any auth handler in ``get_tools`` — so the exception
propagated out, the indexer re-emitted the auth event, and Skip looped forever.

Fix: ``get_tools`` loads unhandled built-in toolkits one at a time and, on
``McpAuthorizationRequired``, builds deferred proxies via ``_build_deferred_mcp_auth_tools`` with
``reraise_on_invoke=True``. When the LLM invokes such a proxy:

- if the server was skipped (``user_declined_mcp_servers``) it returns ``status="declined"`` so the
  loop terminates;
- otherwise it RE-RAISES the stored rich ``McpAuthorizationRequired`` in place — exactly as
  ``discover_mcp_tools`` does for real MCP servers — so the indexer ``on_tool_error`` callback emits
  the ``mcp_authorization_required`` event and the Authorize/Skip dialog appears. (A built-in
  SharePoint site is not an MCP server, so routing through ``mcp_auth_control`` ->
  ``discover_mcp_tools`` would fail silently and never surface the dialog.)

These tests lock in that behaviour without any network access.
"""

import json

import pytest
from langchain_core.tools import StructuredTool

import elitea_sdk.tools as elitea_tools_mod
from elitea_sdk.runtime.toolkits import tools as runtime_tools
from elitea_sdk.runtime.toolkits.tools import get_tools
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired, McpContext

SITE_URL = "https://tenant.sharepoint.com/sites/demo"
OAUTH_ENDPOINT = "https://login.microsoftonline.com/tenant-id"


def _sharepoint_tool(site_url: str = SITE_URL, *, oauth: bool = True) -> dict:
    """Minimal SharePoint tool config as get_tools receives it."""
    sp_conf = {"site_url": site_url}
    if oauth:
        sp_conf["oauth_discovery_endpoint"] = OAUTH_ENDPOINT
        sp_conf["configuration_uuid"] = "cfg-uuid-1"
    return {
        "type": "sharepoint",
        "toolkit_name": "MySharePoint",
        "id": 42,
        "settings": {
            "sharepoint_configuration": sp_conf,
            "selected_tools": [],
        },
    }


def _auth_error(site_url: str = SITE_URL) -> McpAuthorizationRequired:
    """A McpAuthorizationRequired shaped like the one SharePoint raises."""
    return McpAuthorizationRequired(
        message=f"SharePoint site {site_url} requires OAuth authorization.",
        server_url=site_url,
        resource_metadata_url=f"{OAUTH_ENDPOINT}/v2.0/.well-known/openid-configuration",
        resource_metadata={"resource_name": "SharePoint", "resource": site_url},
        tool_name=site_url,
    )


def _patch_sharepoint_loader(monkeypatch, loader):
    """Point AVAILABLE_TOOLS['sharepoint']['get_tools'] at ``loader``."""
    entry = dict(elitea_tools_mod.AVAILABLE_TOOLS.get("sharepoint") or {})
    entry["get_tools"] = loader
    monkeypatch.setitem(elitea_tools_mod.AVAILABLE_TOOLS, "sharepoint", entry)


def _find(tools, name):
    return next((t for t in tools if getattr(t, "name", None) == name), None)


def _proxy_tools(tools):
    """Deferred-auth gateway proxies are named mcp_authorize_*."""
    return [t for t in tools if getattr(t, "name", "").startswith("mcp_authorize_")]


def _invoke_proxy(proxy) -> dict:
    """Invoke a deferred proxy StructuredTool and parse its JSON decision payload."""
    result = proxy.func()
    return json.loads(result)


def test_declined_sharepoint_returns_declined_proxy_no_raise(monkeypatch):
    """Skip path: site already in user_declined_mcp_servers -> declined proxy, no raise,
    loader is never called (no re-triggering the eager auth error)."""
    calls = []

    def loader(tool):
        calls.append(tool)
        raise _auth_error()

    _patch_sharepoint_loader(monkeypatch, loader)

    tools = get_tools(
        [_sharepoint_tool()],
        mcp_context=McpContext(user_declined_servers=[{"server_url": SITE_URL}]),
    )

    proxies = _proxy_tools(tools)
    assert proxies, "expected a deferred auth proxy for the declined SharePoint toolkit"
    payload = _invoke_proxy(proxies[0])
    assert payload["status"] == "declined"
    assert _find(tools, "mcp_auth_control") is not None


def test_no_token_sharepoint_raises_mcp_auth_required_on_proxy_invoke(monkeypatch):
    """First run, no token, not declined -> invoking the proxy RAISES McpAuthorizationRequired
    (not an eager raise at load, and not an inert JSON payload).

    The re-raise is how the built-in toolkit surfaces the Authorize/Skip dialog: it propagates to
    the indexer on_tool_error callback which emits mcp_authorization_required. A JSON
    'authorization_required' payload would instead tell the LLM to call mcp_auth_control ->
    discover_mcp_tools, which cannot probe a non-MCP SharePoint site, so the dialog would never
    appear (issue #5638)."""
    def loader(tool):
        raise _auth_error()

    _patch_sharepoint_loader(monkeypatch, loader)

    tools = get_tools([_sharepoint_tool()])

    proxies = _proxy_tools(tools)
    assert proxies, "expected a deferred auth proxy when SharePoint has no token"
    # Loading must not have raised; the raise happens only when the LLM invokes the proxy.
    with pytest.raises(McpAuthorizationRequired) as exc_info:
        proxies[0].func()
    # The rich exception (with the metadata the UI keys on) is preserved through the re-raise.
    assert exc_info.value.server_url == SITE_URL
    assert (exc_info.value.resource_metadata or {}).get("resource_name") == "SharePoint"
    assert _find(tools, "mcp_auth_control") is not None


def test_app_only_sharepoint_loads_normally(monkeypatch):
    """No oauth_discovery_endpoint -> loader returns real tools, no proxy injected."""
    real_tool = StructuredTool.from_function(
        func=lambda: "ok", name="sp_read_document", description="read"
    )

    def loader(tool):
        return [real_tool]

    _patch_sharepoint_loader(monkeypatch, loader)

    tools = get_tools([_sharepoint_tool(oauth=False)])

    assert _find(tools, "sp_read_document") is not None
    assert not _proxy_tools(tools)
    assert _find(tools, "mcp_auth_control") is None


def test_other_declined_site_still_raises_mcp_auth_required(monkeypatch):
    """A DIFFERENT declined site must not decline this one -> the proxy still RAISES
    McpAuthorizationRequired on invoke (so the Authorize dialog surfaces for this site)."""
    def loader(tool):
        raise _auth_error()

    _patch_sharepoint_loader(monkeypatch, loader)

    tools = get_tools(
        [_sharepoint_tool()],
        mcp_context=McpContext(user_declined_servers=[
            {"server_url": "https://other-tenant.sharepoint.com/sites/other"}
        ]),
    )

    proxies = _proxy_tools(tools)
    assert proxies
    with pytest.raises(McpAuthorizationRequired):
        proxies[0].func()


def test_auth_failure_does_not_abort_sibling_toolkits(monkeypatch):
    """Batch isolation: one toolkit raising McpAuthorizationRequired must not prevent a
    sibling unhandled toolkit from loading."""
    sibling_tool = StructuredTool.from_function(
        func=lambda: "ok", name="sibling_action", description="sibling"
    )

    def loader(tool):
        sp_conf = tool["settings"]["sharepoint_configuration"]
        if sp_conf.get("oauth_discovery_endpoint"):
            raise _auth_error(sp_conf["site_url"])
        return [sibling_tool]

    _patch_sharepoint_loader(monkeypatch, loader)

    failing = _sharepoint_tool(site_url=SITE_URL)  # oauth -> raises
    sibling = _sharepoint_tool(site_url="https://tenant.sharepoint.com/sites/other", oauth=False)
    sibling["id"] = 43  # avoid dedup by id

    tools = get_tools([failing, sibling])

    # Sibling still loaded despite the first toolkit's auth failure.
    assert _find(tools, "sibling_action") is not None
    # And the failing one produced a deferred proxy rather than propagating.
    assert _proxy_tools(tools)
