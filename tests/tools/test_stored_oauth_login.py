"""Stored-login support for the Microsoft delegated toolkits (SharePoint, Outlook, Teams).

A credential may carry a hidden ``refresh_token`` saved by the toolkit page's Login bar. With no
browser token, check_connection exchanges it once for an access token, so the credential can be
tested and scheduled runs (which never have browser tokens) can work. The toolkits themselves read
the access token the platform injects under ``<configuration_uuid>:<oauth_discovery_endpoint>``.
"""
import pytest

from elitea_sdk.configurations.outlook import OutlookConfiguration
from elitea_sdk.configurations.sharepoint import SharepointConfiguration
from elitea_sdk.configurations.teams import TeamsConfiguration
from elitea_sdk.runtime.utils import mcp_oauth
from elitea_sdk.runtime.utils.mcp_oauth import (
    McpAuthorizationRequired,
    access_token_from_stored_login,
    entra_token_endpoint,
)

AUTHORITY = "https://login.microsoftonline.com/tenant-1"
CONFIGS = [OutlookConfiguration, TeamsConfiguration, SharepointConfiguration]


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


@pytest.fixture
def token_endpoint(monkeypatch):
    """Fakes Entra's token endpoint; tests set .response and read .calls."""
    class State:
        calls = []
        response = FakeResponse(200, {"access_token": "fresh-at", "expires_in": 3600})

    def fake_post(url, data=None, headers=None, timeout=None):
        State.calls.append({"url": url, "data": data})
        return State.response

    monkeypatch.setattr(mcp_oauth.requests, "post", fake_post)
    return State


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_refresh_token_field_is_hidden_and_stored_as_secret(config_cls):
    # Hidden in the UI; password format makes the platform keep it as a vault secret.
    field = config_cls.model_json_schema()["properties"]["refresh_token"]
    assert field["hidden"] is True
    assert {"format": "password", "type": "string", "writeOnly": True} in field["anyOf"]
    assert "refresh_token" not in config_cls.model_json_schema().get("required", [])


def test_entra_token_endpoint_shape():
    # The v2 token endpoint is <authority>/oauth2/v2.0/token (SharePoint used a wrong path before).
    assert entra_token_endpoint(AUTHORITY + "/") == f"{AUTHORITY}/oauth2/v2.0/token"


def test_stored_login_refresh_request(token_endpoint):
    token = access_token_from_stored_login({
        "oauth_discovery_endpoint": AUTHORITY, "client_id": "client-1", "client_secret": "secret-1",
        "refresh_token": "refresh-1", "scopes": ["Mail.Read", "offline_access"],
    })
    assert token == "fresh-at"
    call = token_endpoint.calls[0]
    assert call["url"] == f"{AUTHORITY}/oauth2/v2.0/token"
    assert call["data"]["grant_type"] == "refresh_token"
    assert call["data"]["scope"] == "offline_access Mail.Read"
    assert call["data"]["client_secret"] == "secret-1"


def test_stored_login_defaults_to_graph_scope(token_endpoint):
    access_token_from_stored_login({"oauth_discovery_endpoint": AUTHORITY, "refresh_token": "r"})
    assert token_endpoint.calls[0]["data"]["scope"] == "offline_access https://graph.microsoft.com/.default"


def test_existing_access_token_wins_without_refresh(token_endpoint):
    assert access_token_from_stored_login({"access_token": "browser-at", "refresh_token": "r",
                                           "oauth_discovery_endpoint": AUTHORITY}) == "browser-at"
    assert token_endpoint.calls == []


def test_failed_refresh_returns_none_and_hides_secrets(token_endpoint, caplog):
    token_endpoint.response = FakeResponse(400, {"error": "invalid_grant", "error_description": "AADSTS700082"})
    assert access_token_from_stored_login({"oauth_discovery_endpoint": AUTHORITY, "refresh_token": "refresh-1",
                                           "client_secret": "secret-1"}) is None
    assert "refresh-1" not in caplog.text and "secret-1" not in caplog.text


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_check_connection_without_any_login_still_asks_to_log_in(config_cls, token_endpoint, monkeypatch):
    monkeypatch.setattr(mcp_oauth, "fetch_oauth_authorization_server_metadata", lambda *a, **k: None)
    settings = {"oauth_discovery_endpoint": AUTHORITY, "client_id": "c", "client_secret": "s",
                "site_url": "https://contoso.sharepoint.com/sites/x"}
    with pytest.raises(McpAuthorizationRequired):
        config_cls.check_connection(settings)
    assert token_endpoint.calls == []


@pytest.mark.parametrize("config_cls, graph_attr", [
    (OutlookConfiguration, "_call_graph_api"),
    (TeamsConfiguration, "_call_graph_api"),
])
def test_check_connection_uses_stored_login(config_cls, graph_attr, token_endpoint, monkeypatch):
    seen = {}

    def fake_graph(access_token, *args, **kwargs):
        seen["token"] = access_token
        return None

    monkeypatch.setattr(config_cls, graph_attr, staticmethod(fake_graph))
    result = config_cls.check_connection({"oauth_discovery_endpoint": AUTHORITY, "client_id": "c",
                                          "client_secret": "s", "refresh_token": "refresh-1"})
    assert result is None
    assert seen["token"] == "fresh-at"
    assert len(token_endpoint.calls) == 1


@pytest.mark.parametrize("module_path, settings_key", [
    ("elitea_sdk.tools.outlook", "outlook_configuration"),
    ("elitea_sdk.tools.teams", "teams_configuration"),
])
def test_injected_token_without_refresh_keeps_stored_login(module_path, settings_key, monkeypatch):
    # The platform injects only an access token; the credential's stored refresh token must survive
    # so the toolkit can still retry after a 401.
    import importlib
    module = importlib.import_module(module_path)
    captured = {}
    wrapper_cls = next(v for k, v in vars(module).items() if k.endswith("ApiWrapper"))

    def capture(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(module, wrapper_cls.__name__, capture)
    toolkit_cls = next(v for k, v in vars(module).items() if k.endswith("Toolkit") and hasattr(v, "get_toolkit"))
    config = {"oauth_discovery_endpoint": AUTHORITY, "configuration_uuid": "uuid-1", "client_id": "c",
              "client_secret": "s", "refresh_token": "stored-refresh"}
    with pytest.raises(RuntimeError, match="stop"):
        toolkit_cls.get_toolkit(**{settings_key: config, "tokens": {f"uuid-1:{AUTHORITY}": {"access_token": "at"}}})
    assert captured["token"] == "at"
    assert captured["refresh_token"] == "stored-refresh"


# --- Secrets never leave for a non-Entra host -------------------------------------------------
# Anyone who can edit a shared credential could otherwise point the authority at their own server
# and receive the vault client_secret plus the stored refresh token on the next refresh.

EVIL = ["https://evil.example.com/tenant-1", "http://login.microsoftonline.com/tenant-1",
        "https://login.microsoftonline.com.evil.com/t", "https://user@login.microsoftonline.com/t", ""]


@pytest.mark.parametrize("authority", EVIL)
def test_entra_token_endpoint_refuses_other_hosts(authority):
    with pytest.raises(ValueError):
        entra_token_endpoint(authority)


@pytest.mark.parametrize("authority", EVIL[:4])
def test_stored_login_never_posts_to_other_hosts(authority, token_endpoint):
    settings = {"oauth_discovery_endpoint": authority, "refresh_token": "r", "client_secret": "s", "client_id": "c"}
    assert access_token_from_stored_login(settings) is None
    assert token_endpoint.calls == []


@pytest.mark.parametrize("config_cls", CONFIGS)
def test_credential_with_non_entra_authority_cannot_be_saved(config_cls):
    base = {"client_id": "c", "client_secret": "s", "site_url": "https://contoso.sharepoint.com/sites/a"}
    with pytest.raises(ValueError, match="login.microsoftonline.com"):
        config_cls.model_validate({**base, "oauth_discovery_endpoint": "https://evil.example.com/t"})
    config_cls.model_validate({**base, "oauth_discovery_endpoint": AUTHORITY})


@pytest.mark.parametrize("module_path,cls_name", [
    ("elitea_sdk.tools.outlook.graph_wrapper", "OutlookGraphWrapper"),
    ("elitea_sdk.tools.teams.graph_wrapper", "TeamsGraphWrapper"),
    ("elitea_sdk.tools.sharepoint.graph_wrapper", "SharepointGraphWrapper"),
])
def test_401_retry_refuses_non_entra_token_endpoint(module_path, cls_name, monkeypatch):
    # The reactive 401 refresh posts the stored login; a tampered endpoint must short-circuit it.
    import importlib
    module = importlib.import_module(module_path)
    posted = []
    monkeypatch.setattr(module.requests, "post", lambda *a, **k: posted.append(a) or None)
    wrapper = object.__new__(getattr(module, cls_name))
    wrapper._refresh_token, wrapper._oauth_token_endpoint = "r", "https://evil.example.com/token"
    wrapper._client_id, wrapper._client_secret, wrapper._scopes = "c", "s", []
    assert wrapper._try_refresh_token() is False
    assert posted == []
