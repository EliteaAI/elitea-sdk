"""
elitea_issues#6486: when the user who started a run has no PAT, the runtime used to
authenticate as the project system user. It now carries the real user's platform
session reference and sends it as the session cookie, exactly like the browser does.

Covers: PAT path unchanged; session path sets the cookie and drops Authorization;
an invalid/expired session fails loudly instead of yielding a parsed login page.
"""
from unittest.mock import MagicMock, patch

import pytest

BASE = "https://platform.example.com"
PROJECT_ID = 42
COOKIE = "auth_session_id"
SESSION = "sess-ref-abc"


def make_client(**kwargs):
    from elitea_sdk.runtime.clients.client import EliteAClient
    return EliteAClient(base_url=BASE, project_id=PROJECT_ID, **kwargs)


class TestPatPathUnchanged:
    def test_authorization_header_is_set(self):
        client = make_client(auth_token="tok")
        assert client.headers["Authorization"] == "Bearer tok"

    def test_a_session_is_ignored_when_a_pat_exists(self):
        # A PAT holder must keep working even if their browser session has expired.
        client = make_client(auth_token="tok", auth_session=SESSION, session_cookie_name=COOKIE)
        assert client.auth_session is None
        assert client._session.cookies.get(COOKIE) is None


class TestSessionPath:
    def test_cookie_is_set_and_authorization_omitted(self):
        client = make_client(auth_token=None, auth_session=SESSION, session_cookie_name=COOKIE)
        assert "Authorization" not in client.headers
        assert client._session.cookies.get(COOKIE) == SESSION

    def test_llm_kwargs_carry_the_cookie_and_a_placeholder_key(self):
        client = make_client(auth_token=None, auth_session=SESSION, session_cookie_name=COOKIE)
        assert client._llm_cookie_headers == {"Cookie": f"{COOKIE}={SESSION}"}
        # The openai/anthropic clients raise if no key lands in the merged headers.
        assert client._llm_api_key == "session"

    def test_no_cookie_headers_on_the_pat_path(self):
        assert make_client(auth_token="tok")._llm_cookie_headers == {}


class TestExpiredSession:
    @pytest.mark.parametrize("status", [401, 403])
    def test_rejected_session_raises(self, status):
        from elitea_sdk.runtime.clients.client import AuthSessionExpiredError
        client = make_client(auth_token=None, auth_session=SESSION, session_cookie_name=COOKIE)
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(status_code=status, history=[])
            with pytest.raises(AuthSessionExpiredError):
                client.get_mcp_toolkits()

    def test_a_login_redirect_raises_instead_of_returning_html(self):
        from elitea_sdk.runtime.clients.client import AuthSessionExpiredError
        client = make_client(auth_token=None, auth_session=SESSION, session_cookie_name=COOKIE)
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=200, history=[MagicMock(status_code=302)]
            )
            with pytest.raises(AuthSessionExpiredError):
                client.get_mcp_toolkits()

    def test_pat_path_still_returns_the_response_on_403(self):
        client = make_client(auth_token="tok")
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=403, history=[], json=lambda: {"error": "nope"}
            )
            assert client.get_mcp_toolkits() == {"error": "nope"}
