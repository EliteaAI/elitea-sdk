"""The toolkit "Test connection" path follows the same credential rules as Load Tools and
the agent run (#6691): a configured header wins over an OAuth token, an unresolved template
or a blank yields to the token or is not sent at all, and a header that is sent uninjected
makes a 401 report invalid credentials. Re-breaks if `test_mcp_connection` goes back to `setdefault` or
to an any-Authorization check.
"""

import pytest

from elitea_sdk.runtime.clients.client import EliteAClient
from elitea_sdk.runtime.utils import mcp_adapter

URL = "https://mcp.example.test/mcp"


class _RecordingClient:
    seen = {}

    def __init__(self, **kwargs):
        type(self).seen = kwargs

    async def __aenter__(self):
        raise RuntimeError("stop before any network use")

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def recorded(monkeypatch):
    _RecordingClient.seen = {}
    monkeypatch.setattr(mcp_adapter, "UnifiedMcpClient", _RecordingClient)
    return _RecordingClient


def _test_connection(headers, mcp_tokens=None):
    return EliteAClient.test_mcp_connection(
        object.__new__(EliteAClient), {"settings": {"url": URL, "headers": headers}}, mcp_tokens=mcp_tokens,
    )


def test_a_configured_pat_wins_over_the_oauth_token(recorded):
    _test_connection({"authorization": "Bearer configured-pat"}, {URL: {"access_token": "oauth-token"}})

    assert recorded.seen["headers"] == {"authorization": "Bearer configured-pat"}
    assert recorded.seen["configured_auth"] is True


def test_an_unresolved_template_yields_to_the_oauth_token(recorded):
    _test_connection({"Authorization": "Bearer {github_token}"}, {URL: {"access_token": "oauth-token"}})

    assert recorded.seen["headers"] == {"Authorization": "Bearer oauth-token"}
    assert recorded.seen["configured_auth"] is False


def test_an_unresolved_template_without_a_token_is_not_sent(recorded):
    _test_connection({"Authorization": "Bearer {github_token}", "X-A": "1"})

    assert recorded.seen["headers"] == {"X-A": "1"}
    assert recorded.seen["configured_auth"] is False


def test_a_blank_header_is_not_sent_so_the_401_can_open_the_login(recorded):
    _test_connection({"Authorization": "Bearer ", "X-A": "1"})

    assert recorded.seen["headers"] == {"X-A": "1"}
    assert recorded.seen["configured_auth"] is False


def test_headers_stored_as_a_json_string_are_honoured_instead_of_crashing_the_check(recorded):
    _test_connection('{"Authorization": "Bearer configured-pat"}', {URL: {"access_token": "oauth-token"}})

    assert recorded.seen["headers"] == {"Authorization": "Bearer configured-pat"}
    assert recorded.seen["configured_auth"] is True


def test_headers_stored_as_an_unparseable_string_return_a_failure_result_not_a_raise():
    result = _test_connection("not json at all")

    assert result["success"] is False
    assert result["error"]
