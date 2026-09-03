import pytest
from pydantic import SecretStr

from elitea_sdk.configurations.openapi import OpenApiConfiguration
from elitea_sdk.tools.openapi import EliteAOpenAPIToolkit, _build_headers_from_settings


OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Test API", "version": "1.0.0"},
    "servers": [{"url": "https://example.com"}],
    "paths": {
        "/items": {
            "get": {
                "operationId": "list_items",
                "responses": {"200": {"description": "ok"}},
            }
        }
    },
}


def test_openapi_configuration_marks_header_values_as_secrets():
    configuration = OpenApiConfiguration(headers={"x-api-key": "top-secret"})

    assert configuration.headers["x-api-key"].get_secret_value() == "top-secret"

    headers_schema = OpenApiConfiguration.model_json_schema()["properties"]["headers"]
    assert headers_schema["ui_component"] == "secret_headers"
    assert headers_schema["additionalProperties"]["format"] == "password"
    assert headers_schema["description"] == (
        "Static HTTP headers sent with every API request. Use a standard name such as "
        "X-API-Key or X-Tenant-ID. Header names cannot contain spaces, /, :, or line breaks. "
        "Values are stored securely."
    )


@pytest.mark.parametrize("name", ["", "bad header", "bad:header"])
def test_openapi_configuration_rejects_invalid_header_names(name):
    with pytest.raises(ValueError, match="Invalid additional header name"):
        OpenApiConfiguration(headers={name: "top-secret"})


def test_openapi_configuration_rejects_empty_header_values():
    with pytest.raises(ValueError, match="must have a value"):
        OpenApiConfiguration(headers={"x-api-key": ""})


def test_additional_headers_work_without_primary_authentication():
    headers = _build_headers_from_settings(
        {"headers": {"x-api-key": SecretStr("gateway-key")}}
    )

    assert headers == {"x-api-key": "gateway-key"}


@pytest.mark.parametrize(
    ("auth_type", "custom_header_name", "expected_primary_header"),
    [
        ("Bearer", None, ("Authorization", "Bearer primary-key")),
        ("Basic", None, ("Authorization", "Basic primary-key")),
        ("Custom", "x-primary-key", ("x-primary-key", "primary-key")),
    ],
)
def test_additional_headers_are_combined_with_api_key_authentication(
    auth_type,
    custom_header_name,
    expected_primary_header,
):
    headers = _build_headers_from_settings(
        {
            "headers": {
                "x-api-key": SecretStr("gateway-key"),
                "x-tenant": SecretStr("tenant-a"),
            },
            "api_key": SecretStr("primary-key"),
            "auth_type": auth_type,
            "custom_header_name": custom_header_name,
        }
    )

    expected = {
        "x-api-key": "gateway-key",
        "x-tenant": "tenant-a",
    }
    expected[expected_primary_header[0]] = expected_primary_header[1]

    assert headers == expected


def test_primary_authentication_header_cannot_be_overridden_by_additional_headers(monkeypatch):
    monkeypatch.setattr(
        "elitea_sdk.tools.openapi._get_oauth_access_token",
        lambda settings: ("oauth-token", None),
    )

    headers = _build_headers_from_settings(
        {"headers": {"authorization": SecretStr("untrusted-value")}}
    )

    assert headers == {"Authorization": "Bearer oauth-token"}


def test_additional_headers_are_combined_with_delegated_oauth_token():
    discovery_endpoint = "https://identity.example.com"

    toolkit = EliteAOpenAPIToolkit.get_toolkit(
        spec=OPENAPI_SPEC,
        openapi_configuration={
            "oauth_discovery_endpoint": discovery_endpoint,
            "headers": {"x-api-key": SecretStr("gateway-key")},
        },
        tokens={discovery_endpoint: {"access_token": "delegated-token"}},
    )

    assert toolkit.request_session.base_headers == {
        "x-api-key": "gateway-key",
        "Authorization": "Bearer delegated-token",
    }
