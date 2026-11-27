"""#7197: the SharePoint Site URL belongs to the toolkit, not the credential."""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.configurations.sharepoint import SharepointConfiguration
from elitea_sdk.tools.sharepoint import SharepointToolkit


def _wrapper_site_url(credential, **toolkit):
    with patch("elitea_sdk.tools.sharepoint.SharepointApiWrapper") as cls:
        cls.return_value = MagicMock(site_url="x", get_available_tools=MagicMock(return_value=[]))
        SharepointToolkit.get_toolkit(selected_tools=[], sharepoint_configuration=credential, **toolkit)
    return cls.call_args.kwargs["site_url"]


CRED = {"client_id": "c", "client_secret": "s"}


def test_credential_schema_no_longer_requires_site_url():
    schema = SharepointConfiguration.model_json_schema()
    assert "site_url" not in schema.get("required", [])
    assert SharepointConfiguration(**CRED).site_url is None


def test_toolkit_schema_requires_site_url():
    schema = SharepointToolkit.toolkit_config_schema().model_json_schema()
    assert "site_url" in schema["required"]


def test_toolkit_site_url_is_used_with_site_less_credential():
    assert _wrapper_site_url(CRED, site_url="https://t.sharepoint.com/sites/a") == "https://t.sharepoint.com/sites/a"


def test_one_credential_serves_two_sites():
    a = _wrapper_site_url(CRED, site_url="https://t.sharepoint.com/sites/a")
    b = _wrapper_site_url(CRED, site_url="https://t.sharepoint.com/sites/b")
    assert (a, b) == ("https://t.sharepoint.com/sites/a", "https://t.sharepoint.com/sites/b")


def test_toolkit_site_url_wins_over_a_legacy_credential_site_url():
    cred = {**CRED, "site_url": "https://t.sharepoint.com/sites/old"}
    assert _wrapper_site_url(cred, site_url="https://t.sharepoint.com/sites/new") == "https://t.sharepoint.com/sites/new"


def test_legacy_toolkit_falls_back_to_the_credential_site_url():
    cred = {**CRED, "site_url": "https://t.sharepoint.com/sites/old"}
    assert _wrapper_site_url(cred) == "https://t.sharepoint.com/sites/old"


def test_site_path_is_resolved_against_the_toolkit_site_url():
    got = _wrapper_site_url(CRED, site_url="https://t.sharepoint.com", site_path="sites/my-site")
    assert got == "https://t.sharepoint.com/sites/my-site"


@pytest.mark.parametrize("site_url", [None, "", "   "])
def test_no_site_anywhere_is_a_clear_error(site_url):
    with pytest.raises(ToolException, match="Site URL is required"):
        _wrapper_site_url(CRED, site_url=site_url)


def test_app_only_check_without_a_site_validates_the_credentials_only():
    assert SharepointConfiguration.check_connection(dict(CRED)) is None
    assert "Client ID" in SharepointConfiguration.check_connection({"client_secret": "s"})


def test_app_only_check_with_a_legacy_site_still_probes_it():
    with patch.object(SharepointConfiguration, "_validate_site_url", return_value=(None, "probed")) as v:
        assert SharepointConfiguration.check_connection({**CRED, "site_url": "https://t.sharepoint.com"}) == "probed"
    v.assert_called_once()


def test_delegated_check_without_a_site_asks_for_sign_in_then_passes():
    settings = {"oauth_discovery_endpoint": "https://login.microsoftonline.com/tid", "client_id": "c"}
    with patch("elitea_sdk.runtime.utils.mcp_oauth.access_token_from_stored_login", return_value=None), \
            patch.object(SharepointConfiguration, "_build_mcp_authorization_required",
                         side_effect=RuntimeError("auth-required")) as build:
        with pytest.raises(RuntimeError, match="auth-required"):
            SharepointConfiguration.check_connection(settings)
    assert build.call_args.kwargs["site_url"] == "https://graph.microsoft.com"
    with patch("elitea_sdk.runtime.utils.mcp_oauth.access_token_from_stored_login", return_value="tok"):
        assert SharepointConfiguration.check_connection(settings) is None
