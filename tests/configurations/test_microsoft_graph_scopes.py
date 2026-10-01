"""Tests for the Outlook / Teams scope checkboxes, scope normalization and 403 hints (issue 6835)."""

from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.tools import ToolException

from elitea_sdk.configurations.microsoft_graph_scopes import normalize_scopes
from elitea_sdk.configurations.outlook import (
    OUTLOOK_DEFAULT_SCOPES,
    OUTLOOK_SCOPES,
    OutlookConfiguration,
)
from elitea_sdk.configurations.teams import TEAMS_DEFAULT_SCOPES, TEAMS_SCOPES, TeamsConfiguration
from elitea_sdk.tools.outlook.api_wrapper import OutlookApiWrapper
from elitea_sdk.tools.outlook.graph_wrapper import OutlookGraphWrapper
from elitea_sdk.tools.teams.api_wrapper import TeamsApiWrapper

FAKE_TOKEN = "stub-token"
ENDPOINT = "https://login.microsoftonline.com/stub-tenant"


def _resp(status, url, json_body=None):
    resp = MagicMock()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.json.return_value = json_body or {}
    resp.text = str(json_body)
    resp.url = url
    resp.reason = "Forbidden"
    resp.headers = {}
    return resp


class TestNormalizeScopes:
    @pytest.mark.parametrize("value", [None, [], "", " , "])
    def test_empty_becomes_default(self, value):
        assert normalize_scopes(value, ["A.Read", "A.Write"], ["A.Read"]) == ["A.Read"]

    def test_string_is_split_on_commas_and_spaces(self):
        assert normalize_scopes("A.Write, A.Read\nA.Write", ["A.Read", "A.Write"], ["A.Read"]) == [
            "A.Read", "A.Write"]

    def test_case_and_graph_prefix_are_ignored(self):
        value = ["https://graph.microsoft.com/a.write", "A.READ"]
        assert normalize_scopes(value, ["A.Read", "A.Write"], ["A.Read"]) == ["A.Read", "A.Write"]

    def test_unknown_values_are_dropped_and_logged(self, caplog):
        result = normalize_scopes(["offline_access", "User.Read", "A.Write"], ["A.Read", "A.Write"], ["A.Read"])
        assert result == ["A.Write"]
        assert "offline_access, User.Read" in caplog.text

    def test_only_unknown_values_become_default(self):
        assert normalize_scopes(["Calendars.Read"], ["A.Read"], ["A.Read"]) == ["A.Read"]

    def test_default_is_a_copy(self):
        default = ["A.Read"]
        normalize_scopes(None, ["A.Read"], default).append("X")
        assert default == ["A.Read"]


class TestCredentialSchema:
    @pytest.mark.parametrize("model, options, default", [
        (OutlookConfiguration, OUTLOOK_SCOPES, OUTLOOK_DEFAULT_SCOPES),
        (TeamsConfiguration, TEAMS_SCOPES, TEAMS_DEFAULT_SCOPES),
    ])
    def test_scopes_field_is_a_checkbox_list(self, model, options, default):
        field = model.model_json_schema()["properties"]["scopes"]
        assert field["ui_component"] == "checkbox_list"
        assert [o["value"] for o in field["checkbox_options"]] == options
        assert all(o["description"] for o in field["checkbox_options"])
        assert field["default"] == default
        assert "myapps.microsoft.com" in field["description"]

    def test_user_read_and_offline_access_are_not_options(self):
        assert "User.Read" not in TEAMS_SCOPES
        assert "offline_access" not in TEAMS_SCOPES + OUTLOOK_SCOPES

    def test_default_scopes_are_offered(self):
        assert set(OUTLOOK_DEFAULT_SCOPES) <= set(OUTLOOK_SCOPES)
        assert set(TEAMS_DEFAULT_SCOPES) <= set(TEAMS_SCOPES)

    def test_saved_values_are_normalized(self):
        config = OutlookConfiguration(client_id="stub-client", client_secret="stub-secret",
                                      scopes="Mail.Send offline_access Calendars.Read mail.read")
        assert config.model_dump()["scopes"] == ["Mail.Read", "Mail.Send"]

    def test_missing_scopes_use_default(self):
        config = TeamsConfiguration(client_id="stub-client", client_secret="stub-secret", scopes=None)
        assert config.scopes == TEAMS_DEFAULT_SCOPES


class TestSignInScopes:
    @pytest.mark.parametrize("model, scopes, expected", [
        (OutlookConfiguration, ["User.Read", "Mail.Send"], ["offline_access", "Mail.Send"]),
        (OutlookConfiguration, None, ["offline_access", "Mail.Read"]),
        (TeamsConfiguration, "offline_access Chat.Read", ["offline_access", "Chat.Read"]),
        (TeamsConfiguration, [], ["offline_access", *TEAMS_DEFAULT_SCOPES]),
    ])
    def test_sign_in_requests_only_offered_scopes(self, model, scopes, expected):
        with patch("elitea_sdk.runtime.utils.mcp_oauth.fetch_oauth_authorization_server_metadata",
                   return_value=None):
            error = model._build_mcp_authorization_required(
                message="stub", oauth_discovery_endpoint=ENDPOINT, scopes=scopes)
        assert error.resource_metadata["scopes_supported"] == expected
        assert error.resource_metadata["provided_settings"]["scopes"] == expected


class TestRefreshScopes:
    @pytest.mark.parametrize("wrapper_cls, scopes, expected", [
        (OutlookApiWrapper, ["Mail.Read", "User.Read", "offline_access"], "Mail.Read"),
        (TeamsApiWrapper, "Chat.Read,Calendars.Read", "Chat.Read"),
    ])
    def test_refresh_requests_only_offered_scopes(self, wrapper_cls, scopes, expected):
        wrapper = wrapper_cls(token=FAKE_TOKEN, refresh_token="stub-refresh", scopes=scopes,
                              oauth_discovery_endpoint=ENDPOINT, client_id="stub-client")
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"access_token": "stub-new"}
        with patch("requests.post", return_value=ok) as post:
            assert wrapper._backend._try_refresh_token()
        assert post.call_args.kwargs["data"]["scope"] == expected


class TestOutlookForbiddenHint:
    @pytest.mark.parametrize("method, url, mailbox, needed", [
        ("get", "https://graph.microsoft.com/v1.0/me/messages", None, "Mail.Read"),
        ("patch", "https://graph.microsoft.com/v1.0/me/messages/m1", None, "Mail.ReadWrite"),
        ("post", "https://graph.microsoft.com/v1.0/me/sendMail", None, "Mail.Send"),
        ("post", "https://graph.microsoft.com/v1.0/users/shared@example.com/messages/m1/move",
         "shared@example.com", "Mail.ReadWrite.Shared"),
    ])
    def test_403_names_needed_permission_and_stays_http_error(self, method, url, mailbox, needed):
        wrapper = OutlookGraphWrapper(token=FAKE_TOKEN, scopes=["Mail.Read"], mailbox=mailbox)
        call = {"get": wrapper._get, "patch": lambda u: wrapper._patch(u, {}),
                "post": lambda u: wrapper._post(u, {})}[method]
        resp = _resp(403, url, {"error": {"code": "ErrorAccessDenied", "message": "Access is denied"}})
        with patch(f"requests.{method}", return_value=resp):
            with pytest.raises(requests.HTTPError) as exc:
                call(url)
        assert f"needs the Microsoft Graph permission {needed}:" in str(exc.value)
        assert exc.value.response is resp


class TestTeamsForbiddenHint:
    @pytest.mark.parametrize("method, url, needed", [
        ("GET", "https://graph.microsoft.com/v1.0/me/joinedTeams", "Team.ReadBasic.All"),
        ("GET", "https://graph.microsoft.com/v1.0/teams/t1/channels", "Channel.ReadBasic.All"),
        ("POST", "https://graph.microsoft.com/v1.0/teams/t1/channels/c1/messages", "ChannelMessage.Send"),
        ("GET", "https://graph.microsoft.com/v1.0/me/chats", "Chat.Read"),
        ("POST", "https://graph.microsoft.com/v1.0/chats", "Chat.Create"),
        ("POST", "https://graph.microsoft.com/v1.0/chats/c1/messages", "ChatMessage.Send"),
        ("GET", "https://graph.microsoft.com/v1.0/users/a@example.com", "User.ReadBasic.All"),
    ])
    def test_403_names_needed_permission(self, method, url, needed):
        wrapper = TeamsApiWrapper(token=FAKE_TOKEN, scopes=["Chat.Read"])
        resp = _resp(403, url, {"error": {"message": "Missing role"}})
        with patch("requests.request", return_value=resp):
            with pytest.raises(ToolException) as exc:
                wrapper._backend._request(method, url)
        assert f"needs the Microsoft Graph permission {needed}:" in str(exc.value)
