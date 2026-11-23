"""#7643: the mcp_login credential model and its place in the remote MCP toolkit schema."""
import pytest

from elitea_sdk.configurations import get_configurations
from elitea_sdk.configurations.mcp_login import McpLoginConfiguration
from elitea_sdk.runtime.toolkits.mcp import McpToolkit


def test_mcp_login_is_a_registered_credential_type():
    assert "mcp_login" in get_configurations()


def test_login_fields_are_hidden_and_secrets_are_secret():
    props = McpLoginConfiguration.model_json_schema()["properties"]
    for field in ("refresh_token", "token_endpoint", "resource", "used_dcr", "account"):
        assert props[field].get("hidden") is True
    for field in ("refresh_token", "client_secret"):
        assert "password" in str(props[field])


@pytest.mark.parametrize("url, error", [
    ("https://api.example.com/mcp", None),
    ("http://10.0.0.1:8765/mcp", None),
    ("ftp://x", "MCP server URL must start with http:// or https://"),
    ("", "MCP server URL must start with http:// or https://"),
])
def test_check_connection_validates_only_the_url(url, error):
    assert McpLoginConfiguration.check_connection({"url": url}) == error


def test_remote_mcp_toolkit_offers_an_optional_saved_login():
    prop = McpToolkit.toolkit_config_schema().model_json_schema()["properties"]["mcp_login_configuration"]
    assert prop["configuration_types"] == ["mcp_login"]
    assert prop["default"] is None
