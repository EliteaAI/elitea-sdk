"""XrayApiWrapper must not build `//api/...` URLs from a base_url with a trailing slash.

Xray Cloud answers `https://eu.xray.cloud.getxray.app//api/v2/graphql` with 400 Bad Request.
"""

from unittest.mock import MagicMock, patch

import pytest

from elitea_sdk.tools.xray.api_wrapper import XrayApiWrapper


@pytest.mark.parametrize("configured_url", [
    "https://eu.xray.cloud.getxray.app/",
    "https://eu.xray.cloud.getxray.app",
])
def test_base_url_trailing_slash_is_stripped(configured_url):
    auth_response = MagicMock()
    auth_response.json.return_value = "token"
    values = {"base_url": configured_url, "client_id": "id", "client_secret": "secret"}

    with patch("elitea_sdk.tools.xray.api_wrapper.requests.post", return_value=auth_response) as post, \
            patch("elitea_sdk.tools.xray.api_wrapper.NonCodeIndexerToolkit.validate_toolkit",
                  side_effect=lambda v: v):
        result = XrayApiWrapper.validate_toolkit(values)

    assert post.call_args.args[0] == "https://eu.xray.cloud.getxray.app/api/v1/authenticate"
    assert result["base_url"] == "https://eu.xray.cloud.getxray.app"
    assert result["_client_endpoint"] == "https://eu.xray.cloud.getxray.app/api/v2/graphql"
