"""Tests for errors caused by a mailbox value that is not a mailbox (e.g. a distribution list)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from elitea_sdk.configurations.outlook import OutlookConfiguration
from elitea_sdk.tools.outlook.graph_wrapper import OutlookGraphWrapper

DL = "support-dl@contoso.com"
INACTIVE = "The mailbox is either inactive, soft-deleted, or is hosted on-premise."


def _resp(status, code=None, message=None, url="https://graph.microsoft.com/v1.0/me/mailFolders/inbox/messages"):
    resp = MagicMock()
    resp.status_code = status
    resp.ok = 200 <= status < 300
    resp.url = url
    resp.reason = "Not Found"
    resp.json.return_value = {"error": {"code": code, "message": message}} if code or message else {}
    return resp


def _raised(wrapper, resp):
    with pytest.raises(requests.HTTPError) as exc_info:
        wrapper._raise_with_body(resp)
    return exc_info.value


def test_distribution_list_as_mailbox_explains_how_to_use_it():
    wrapper = OutlookGraphWrapper(token="stub", scopes=[], mailbox=DL)
    exc = _raised(wrapper, _resp(404, "MailboxNotEnabledForRESTAPI", INACTIVE,
                                 url=f"https://graph.microsoft.com/v1.0/users/{DL}/mailFolders/inbox/messages"))
    text = str(exc)
    assert INACTIVE.rstrip(".") in text
    assert "mail folder was not found" not in text
    assert "distribution list" in text and "clear the mailbox field" in text
    assert f"recipients=['{DL}']" in text and f"'to:{DL}'" in text
    assert exc.provider_error_category == "resource_not_found"


def test_inactive_message_without_code_is_recognised():
    wrapper = OutlookGraphWrapper(token="stub", scopes=[], mailbox=DL)
    assert "distribution list" in str(_raised(wrapper, _resp(404, None, INACTIVE)))


def test_own_mailbox_missing_does_not_mention_the_mailbox_field():
    wrapper = OutlookGraphWrapper(token="stub", scopes=[])
    text = str(_raised(wrapper, _resp(404, "MailboxNotEnabledForRESTAPI", INACTIVE)))
    assert "signed-in user has no Exchange Online mailbox" in text
    assert "mailbox field" not in text and "mail folder was not found" not in text


def test_unknown_folder_keeps_the_folder_hint():
    wrapper = OutlookGraphWrapper(token="stub", scopes=[])
    text = str(_raised(wrapper, _resp(404, "ErrorInvalidIdMalformed", "Id is malformed.",
                                      url="https://graph.microsoft.com/v1.0/me/mailFolders/Projects")))
    assert "mail folder was not found" in text and "distribution list" not in text


# --- connection check ----------------------------------------------------------------

def _check(mailbox, status):
    settings = {"oauth_discovery_endpoint": "https://login.microsoftonline.com/t", "access_token": "tok",
                "mailbox": mailbox}
    with patch("elitea_sdk.configurations.outlook.requests.get", return_value=_resp(status)) as get:
        result = OutlookConfiguration.check_connection(settings)
    return result, get.call_args.args[0]


def test_connection_check_targets_the_configured_mailbox():
    result, url = _check(DL, 404)
    assert url == f"https://graph.microsoft.com/v1.0/users/{DL}/mailFolders?$top=1"
    assert "distribution list" in result and "leave the mailbox field empty" in result


def test_connection_check_without_mailbox_uses_me():
    result, url = _check("  ", 200)
    assert result is None and url == "https://graph.microsoft.com/v1.0/me/mailFolders?$top=1"


def test_connection_check_shared_mailbox_without_access():
    result, _ = _check("shared@contoso.com", 403)
    assert "Full Access" in result and "Mail.Read.Shared" in result
