from unittest.mock import Mock, patch

import pytest
from pydantic import SecretStr

from elitea_sdk.configurations.aha import AhaConfiguration


def _settings(base_url: str) -> dict:
    return {
        "base_url": base_url,
        "api_key": SecretStr("valid-token"),
    }


@pytest.mark.parametrize(
    "base_url",
    [
        "https://company.aha.io/",
        "https://roadmaps.company.example/",
    ],
)
def test_check_connection_accepts_aha_user_profile(base_url):
    response = Mock(status_code=200)
    response.json.return_value = {
        "user": {
            "id": "123",
            "name": "Aha User",
        }
    }

    with patch("requests.get", return_value=response) as get:
        error = AhaConfiguration.check_connection(_settings(base_url))

    assert error is None
    get.assert_called_once_with(
        f"{base_url.rstrip('/')}/api/v1/me",
        headers={"Authorization": "Bearer valid-token"},
        timeout=10,
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"user": {}},
        {"status": "ok"},
    ],
)
def test_check_connection_rejects_non_aha_success_payload(payload):
    response = Mock(status_code=200)
    response.json.return_value = payload

    with patch("requests.get", return_value=response):
        error = AhaConfiguration.check_connection(
            _settings("https://company.aha.io")
        )

    assert error == "Aha! API returned an unexpected response: verify the base URL"


def test_check_connection_rejects_non_aha_domain_success_response():
    response = Mock(status_code=200)
    response.json.side_effect = ValueError("not JSON")

    with patch("requests.get", return_value=response) as get:
        error = AhaConfiguration.check_connection(_settings("https://gitlab.io"))

    assert error == "Aha! API returned an unexpected response: verify the base URL"
    get.assert_called_once_with(
        "https://gitlab.io/api/v1/me",
        headers={"Authorization": "Bearer valid-token"},
        timeout=10,
    )
