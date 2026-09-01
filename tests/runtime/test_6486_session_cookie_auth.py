"""
elitea_issues#6486: when the user who started a run has no PAT, the runtime used to
authenticate as the project system user. It now carries the real user's platform
session reference and sends it as the session cookie, exactly like the browser does.

Covers: PAT path unchanged; session path sets the cookie (scoped to our own host) and
drops Authorization; an invalid/expired session fails loudly instead of yielding a parsed
login page; a bare 403 is treated as a real permission denial, not session expiry.
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

    def test_cookie_is_scoped_to_our_own_host_not_wildcard(self):
        # A cookiejar entry with no domain/secure attaches to ANY host the session is
        # later pointed at (e.g. a redirect target) — scope it or it leaks (#6486 review).
        client = make_client(auth_token=None, auth_session=SESSION, session_cookie_name=COOKIE)
        jar_cookie = next(iter(client._session.cookies))
        assert jar_cookie.domain == "platform.example.com"
        assert jar_cookie.secure is True

    def test_cookie_is_not_sent_to_a_different_host(self):
        client = make_client(auth_token=None, auth_session=SESSION, session_cookie_name=COOKIE)
        other_host_cookies = client._session.cookies.get_dict(domain="evil.example.com")
        assert other_host_cookies == {}


class TestExpiredSession:
    def test_a_rejected_session_raises_on_401(self):
        from elitea_sdk.runtime.clients.client import AuthSessionExpiredError
        client = make_client(auth_token=None, auth_session=SESSION, session_cookie_name=COOKIE)
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(status_code=401, history=[])
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

    def test_a_bare_403_is_a_permission_denial_not_expiry(self):
        # A valid session can still get 403 from a real authorization check (e.g. no
        # membership in the target project) — that must reach the caller as-is, not be
        # reported as an expired session.
        client = make_client(auth_token=None, auth_session=SESSION, session_cookie_name=COOKIE)
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=403, history=[], json=lambda: {"error": "access_denied"}
            )
            assert client.get_mcp_toolkits() == {"error": "access_denied"}

    def test_pat_path_still_returns_the_response_on_403(self):
        client = make_client(auth_token="tok")
        with patch("requests.Session.request") as mock_request:
            mock_request.return_value = MagicMock(
                status_code=403, history=[], json=lambda: {"error": "nope"}
            )
            assert client.get_mcp_toolkits() == {"error": "nope"}
