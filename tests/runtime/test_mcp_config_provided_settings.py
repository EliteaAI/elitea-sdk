"""Regression coverage for OAuth credentials declared on config-defined MCP servers.

Without the server-definition fallback the browser receives no client_id and falls back to
dynamic client registration, authorizing against a throwaway client instead of the
pre-registered one.
"""

import pytest

from elitea_sdk.runtime.toolkits import mcp_config
from elitea_sdk.runtime.toolkits.tools import _build_provided_settings


CLIENT_ID = "82987136-1967-4d9e-8f68-0359c432219f"
SCOPE = "https://mcp.example.test/mcp/user_impersonation offline_access"


@pytest.fixture(autouse=True)
def _server_definitions():
    previous = mcp_config.get_all_mcp_server_configs()
    mcp_config.refresh_mcp_server_configs(
        {
            "Example CRM": {
                "type": "http",
                "url": "https://mcp.example.test/mcp",
                "client_id": CLIENT_ID,
                "client_secret": "super-secret-value",
                "scope": SCOPE,
            },
            "No Oauth Server": {"type": "http", "url": "https://other.example.test/mcp"},
        }
    )
    yield
    mcp_config.refresh_mcp_server_configs(previous)


def test_credentials_resolve_from_server_definition():
    provided = _build_provided_settings({"server_name": "Example CRM"})

    assert provided["mcp_client_id"] == CLIENT_ID
    assert provided["scopes"] == SCOPE
    assert "super-secret-value" not in provided["mcp_client_secret"]


def test_server_name_matches_case_insensitively():
    assert _build_provided_settings({"server_name": "example crm"})["mcp_client_id"] == CLIENT_ID


def test_server_name_inferred_from_prebuilt_toolkit_type():
    provided = _build_provided_settings({}, {"type": "mcp_Example CRM"})

    assert provided["mcp_client_id"] == CLIENT_ID


def test_toolkit_settings_take_precedence_over_server_definition():
    provided = _build_provided_settings({"server_name": "Example CRM", "client_id": "from-settings"})

    assert provided["mcp_client_id"] == "from-settings"
    assert "mcp_client_secret" not in provided


@pytest.mark.parametrize(
    "settings",
    [{}, {"server_name": "No Oauth Server"}, {"server_name": "unregistered"}],
)
def test_servers_without_credentials_provide_nothing(settings):
    assert _build_provided_settings(settings) is None
