"""#7728: local Pyodide sandboxes must get a hostname allowlist, never bare --allow-net.

Bare --allow-net let sandbox code reach internal services on the worker network
(an internal secrets service answered). The fix maps allow_net=True to the package
hosts plus operator-supplied SANDBOX_ALLOWED_HOSTS.
"""
from unittest.mock import patch

import pytest

from elitea_sdk.runtime.tools import sandbox as sandbox_mod
from elitea_sdk.runtime.tools.sandbox import (
    PyodideSandboxTool,
    StatefulPyodideSandboxTool,
    platform_host,
    resolve_allow_net,
    sandbox_allowed_hosts,
)

DEFAULTS = ["pypi.org", "files.pythonhosted.org", "cdn.jsdelivr.net"]


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("SANDBOX_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("SANDBOX_SERVICE_URL", raising=False)
    monkeypatch.setenv("PYODIDE_SANDBOX_PKG", "/fake/pyodide_sandbox/main.ts")


def test_defaults_are_package_hosts():
    assert sandbox_allowed_hosts() == DEFAULTS


def test_operator_hosts_appended_trimmed_deduped(monkeypatch):
    monkeypatch.setenv("SANDBOX_ALLOWED_HOSTS", " API.example.com, ,pypi.org,api.example.com,x.org:8443 ")
    assert sandbox_allowed_hosts() == [*DEFAULTS, "api.example.com", "x.org:8443"]


@pytest.fixture(autouse=True)
def _clear_cache():
    sandbox_mod._cached_allowed_hosts.cache_clear()


@pytest.mark.parametrize("bad", [
    "*.example.com", "*", "example.com/*/anything/*", "example.com/path", "https://example.com",
    "exa mple.com", "-bad.com", "example.com:99999x", "example.com:99999", "example.com:0", "user@example.com", "example.com:*",
])
def test_wildcards_paths_and_schemes_rejected(monkeypatch, bad):
    # Colleague review: wildcard subdomains and path patterns must never reach --allow-net.
    monkeypatch.setenv("SANDBOX_ALLOWED_HOSTS", f"{bad},ok.example.com")
    assert sandbox_allowed_hosts() == [*DEFAULTS, "ok.example.com"]
    assert resolve_allow_net([bad]) is False


def test_resolve_allow_net():
    assert resolve_allow_net(True) == DEFAULTS
    assert resolve_allow_net(False) is False
    assert resolve_allow_net(["only.example.com"]) == ["only.example.com"]


def _built_command(tool):
    # Build the real Deno command line without needing Deno installed.
    with patch.object(sandbox_mod, "_is_deno_available", return_value=True), \
            patch("elitea_sdk.runtime.langchain.pyodide_sandbox.shutil.which", return_value="/usr/bin/deno"):
        tool._initialize_sandbox()
    return tool._sandbox._build_command()


@pytest.mark.parametrize("tool_cls", [PyodideSandboxTool, StatefulPyodideSandboxTool])
def test_default_tool_never_gets_bare_allow_net(tool_cls):
    cmd = _built_command(tool_cls())
    assert "--allow-net" not in cmd
    assert "--allow-net=" + ",".join(DEFAULTS) in cmd


def test_operator_hosts_reach_deno_flag(monkeypatch):
    monkeypatch.setenv("SANDBOX_ALLOWED_HOSTS", "api.example.com")
    cmd = _built_command(PyodideSandboxTool(allow_net=True))
    assert "--allow-net=" + ",".join([*DEFAULTS, "api.example.com"]) in cmd


def test_allow_net_false_has_no_net_flag():
    cmd = _built_command(PyodideSandboxTool(allow_net=False))
    assert not any(c.startswith("--allow-net") for c in cmd)


def test_all_entry_points_resolve_to_allowlist():
    # The three production call sites (sandbox toolkit, internal 'pyodide' tool, agent
    # auto-attached sandbox) all pass allow_net=True into these constructors.
    tools = [
        *sandbox_mod.SandboxToolkit.get_toolkit(stateful=False, allow_net=True).get_tools(),
        sandbox_mod.create_sandbox_tool(stateful=False, allow_net=True),
    ]
    for tool in tools:
        cmd = _built_command(tool)
        assert "--allow-net" not in cmd
        assert "--allow-net=" + ",".join(DEFAULTS) in cmd


class _Client:
    base_url = "https://platform.example.org:8443/"
    project_id = 3


def test_platform_host_added_without_bridge(monkeypatch):
    # Default (bridge off): sandbox code uses SandboxClient over HTTP, so only the platform host is added.
    monkeypatch.delenv("SANDBOX_CLIENT_BRIDGE", raising=False)
    cmd = _built_command(PyodideSandboxTool(elitea_client=_Client()))
    assert "--allow-net=" + ",".join([*DEFAULTS, "platform.example.org:8443"]) in cmd


def test_platform_host_not_added_in_bridge_mode(monkeypatch):
    # Bridge on: platform calls go over stdio, so the platform host must stay unreachable.
    monkeypatch.setenv("SANDBOX_CLIENT_BRIDGE", "1")
    cmd = _built_command(PyodideSandboxTool(elitea_client=_Client()))
    assert "--allow-net=" + ",".join(DEFAULTS) in cmd


def test_platform_host_not_added_when_net_off():
    cmd = _built_command(PyodideSandboxTool(allow_net=False, elitea_client=_Client()))
    assert not any(c.startswith("--allow-net") for c in cmd)


def test_invalid_entry_warned_once(monkeypatch, caplog):
    monkeypatch.setenv("SANDBOX_ALLOWED_HOSTS", "*.example.com")
    sandbox_allowed_hosts(); sandbox_allowed_hosts()
    assert sum("Ignoring sandbox" in r.message for r in caplog.records) == 1


@pytest.mark.parametrize("url", [
    "https://example.com,internal:80/",
    "https://example.com,internal/",
    "https://user@example.com,internal:80/",
    "https://exa mple.com/",
    "https://example.com:99999/",
    "https://[::1]:8080/",
    "ftp://example.com/",
    "example.com",
    "",
    None,
])
def test_platform_host_rejects_anything_but_one_plain_host(url):
    # Reviewer finding: a crafted base_url must never serialize into extra --allow-net grants.
    assert platform_host(url) is None


@pytest.mark.parametrize("url,expected", [
    ("https://Platform.Example.org/", "platform.example.org"),
    ("http://platform.example.org:8443/api", "platform.example.org:8443"),
    ("https://user:pw@platform.example.org/", "platform.example.org"),
])
def test_platform_host_valid(url, expected):
    assert platform_host(url) == expected


def test_crafted_base_url_adds_no_grants(monkeypatch):
    monkeypatch.delenv("SANDBOX_CLIENT_BRIDGE", raising=False)
    client = type("C", (), {"base_url": "https://example.com,internal:80/", "project_id": 3})()
    cmd = _built_command(PyodideSandboxTool(elitea_client=client))
    assert "--allow-net=" + ",".join(DEFAULTS) in cmd
