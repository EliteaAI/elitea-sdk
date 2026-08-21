"""Reactive delegated-auth coverage for stale SharePoint tokens (issue #6298)."""

from unittest.mock import MagicMock, patch

import pytest

from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired
from elitea_sdk.tools.sharepoint.api_wrapper import SharepointApiWrapper
from elitea_sdk.tools.sharepoint.graph_wrapper import SharepointGraphWrapper

SITE_URL = "https://tenant.sharepoint.com/sites/demo"
OAUTH_ENDPOINT = "https://login.microsoftonline.com/tenant-id"


def _response(status_code: int, body: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.ok = 200 <= status_code < 400
    response.url = "https://graph.microsoft.com/v1.0/sites/example"
    response.text = str(body)
    response.json.return_value = body
    if status_code >= 400:
        response.raise_for_status.side_effect = RuntimeError(f"HTTP {status_code}")
    return response


def _openid_metadata():
    return {
        "issuer": OAUTH_ENDPOINT,
        "authorization_endpoint": f"{OAUTH_ENDPOINT}/oauth2/authorize",
        "token_endpoint": f"{OAUTH_ENDPOINT}/oauth2/token",
    }


@patch(
    "elitea_sdk.runtime.utils.mcp_oauth.fetch_oauth_authorization_server_metadata",
    return_value=_openid_metadata(),
)
@patch("elitea_sdk.tools.sharepoint.graph_wrapper.requests.get")
def test_stale_token_raises_rich_auth_signal_from_real_tool(
    mock_get,
    _mock_discovery,
):
    """A Graph 401 reaches the tool runner as auth-required, not a generic tool error."""
    mock_get.return_value = _response(
        401,
        {"error": {"code": "InvalidAuthenticationToken", "message": "Access token expired"}},
    )
    graph_wrapper = SharepointGraphWrapper(
        site_url=SITE_URL,
        token="stale-token",
        scopes=["Files.Read.All"],
        oauth_discovery_endpoint=OAUTH_ENDPOINT,
        configuration_uuid="cfg-uuid-1",
        toolkit_name="sharepoint",
        toolkit_id=42,
        client_id="configured-client",
        client_secret="configured-secret",
    )
    api_wrapper = SharepointApiWrapper.model_construct(site_url=SITE_URL, elitea=None, llm=None)
    api_wrapper._backend = graph_wrapper

    with pytest.raises(McpAuthorizationRequired) as exc_info:
        api_wrapper.get_files_list()

    auth_error = exc_info.value
    assert auth_error.status == 401
    assert auth_error.server_url == SITE_URL
    assert auth_error.toolkit_name == "sharepoint"
    assert auth_error.toolkit_type == "sharepoint"
    assert auth_error.toolkit_id == 42
    assert auth_error.tool_name is None
    assert auth_error.resource_metadata["configuration_uuid"] == "cfg-uuid-1"
    assert auth_error.resource_metadata["toolkit_id"] == 42
    provided = auth_error.resource_metadata["provided_settings"]
    assert provided["mcp_client_id"] == "configured-client"
    assert provided["mcp_client_secret"] != "configured-secret"
    assert "Files.Read.All" in provided["scopes"]


@patch("elitea_sdk.tools.sharepoint.graph_wrapper.requests.get")
def test_successful_refresh_retries_original_graph_request(mock_get):
    """A refreshable 401 remains transparent and retries the interrupted operation once."""
    mock_get.side_effect = [
        _response(401, {"error": {"message": "expired"}}),
        _response(200, {"value": ["ok"]}),
    ]
    wrapper = SharepointGraphWrapper(
        site_url=SITE_URL,
        token="expired-token",
        scopes=["Files.Read.All"],
        oauth_discovery_endpoint=OAUTH_ENDPOINT,
    )
    wrapper._try_refresh_token = MagicMock(return_value=True)

    assert wrapper._get("https://graph.microsoft.com/v1.0/test") == {"value": ["ok"]}
    assert mock_get.call_count == 2
    wrapper._try_refresh_token.assert_called_once_with()


@patch(
    "elitea_sdk.runtime.utils.mcp_oauth.fetch_oauth_authorization_server_metadata",
    return_value=_openid_metadata(),
)
@patch("elitea_sdk.tools.sharepoint.graph_wrapper.requests.get")
def test_raw_download_401_uses_same_auth_signal(mock_get, _mock_discovery):
    """Binary/download helpers cannot demote a stale-token 401 to ToolException."""
    mock_get.return_value = _response(401, {"error": {"message": "revoked"}})
    wrapper = SharepointGraphWrapper(
        site_url=SITE_URL,
        token="revoked-token",
        scopes=["Files.Read.All"],
        oauth_discovery_endpoint=OAUTH_ENDPOINT,
        configuration_uuid="cfg-uuid-1",
    )

    with pytest.raises(McpAuthorizationRequired):
        wrapper._get_raw("https://graph.microsoft.com/v1.0/content", timeout=60)
