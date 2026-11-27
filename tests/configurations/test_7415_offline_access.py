"""#7415: offline_access is asked for at sign-in only while token refresh is on."""
from unittest.mock import patch

import pytest

from elitea_sdk.configurations.oauth_app_mode import sign_in_scopes
from elitea_sdk.configurations.outlook import OutlookConfiguration
from elitea_sdk.configurations.sharepoint import SharepointConfiguration
from elitea_sdk.configurations.teams import TeamsConfiguration

ENDPOINT = "https://login.microsoftonline.com/contoso"
DISCOVERY = {"authorization_endpoint": f"{ENDPOINT}/oauth2/v2.0/authorize",
             "token_endpoint": f"{ENDPOINT}/oauth2/v2.0/token"}


@pytest.mark.parametrize("auto_refresh, expected", [
    (True, ["offline_access", "Mail.Read"]),
    (None, ["offline_access", "Mail.Read"]),  # older credentials never stored the switch
    (False, ["Mail.Read"]),
])
def test_sign_in_scopes(auto_refresh, expected):
    assert sign_in_scopes(["offline_access", "Mail.Read"], auto_refresh) == expected


def _prompt_scopes(model, auto_refresh, **extra):
    with patch("elitea_sdk.runtime.utils.mcp_oauth.fetch_oauth_authorization_server_metadata",
               return_value=DISCOVERY):
        error = model._build_mcp_authorization_required(
            message="sign in", oauth_discovery_endpoint=ENDPOINT, scopes=extra.pop("scopes"),
            auto_refresh_token=auto_refresh, **extra)
    return error.resource_metadata["scopes_supported"]


@pytest.mark.parametrize("model, scopes, extra", [
    (TeamsConfiguration, ["Chat.Read"], {}),
    (OutlookConfiguration, ["Mail.Read"], {}),
    (SharepointConfiguration, ["Sites.Read.All"], {"site_url": "https://contoso.sharepoint.com/sites/qa"}),
])
@pytest.mark.parametrize("auto_refresh, has_offline", [(True, True), (None, True), (False, False)])
def test_sign_in_prompt_follows_auto_refresh(model, scopes, extra, auto_refresh, has_offline):
    requested = _prompt_scopes(model, auto_refresh, scopes=scopes, **extra)
    assert ("offline_access" in requested) is has_offline
    assert scopes[0] in requested
