"""Issue #7065 - the SDK no longer sends X-SECRET, and a 403 on an agent fetch is surfaced.

PATCH version (the expanded agent fetch) now needs the unsecret permission. Inside the platform
the runtime gets expanded details with the predict (#7064), so it only calls PATCH for a sub-agent
the prefetch could not expand. As a viewer that call now gets 403.

Review round 1: a sub-agent that cannot be loaded must keep being skipped, as before - a
pipeline may never use it, so it must not fail a viewer's run inside the platform. Only the
top-level agent fetch raises AgentAccessDeniedError; in practice only the CLI and Streamlit
reach it, because platform runs always supply the top-level details.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from elitea_sdk.runtime.clients.client import ApiDetailsRequestError, EliteAClient
from elitea_sdk.runtime.clients.sandbox_client import SandboxClient
from elitea_sdk.runtime.exceptions import AgentAccessDeniedError


def _client(**kwargs):
    return EliteAClient("https://elitea.invalid", 3, "pat-token", **kwargs)


def test_client_sends_no_secret_header():
    client = _client()

    assert 'X-SECRET' not in client.headers
    assert client.headers == {'Authorization': 'Bearer pat-token'}


def test_the_old_xsecret_kwarg_is_ignored():
    assert 'X-SECRET' not in _client(XSECRET='secret').headers


def test_sandbox_client_sends_no_secret_header():
    client = SandboxClient("https://elitea.invalid", 3, "pat-token")

    assert 'X-SECRET' not in client.headers


def test_session_cookie_client_sends_no_identity_headers():
    client = EliteAClient("https://elitea.invalid", 3, None, auth_session='ref', session_cookie_name='sid')

    assert client.headers == {}


def _response(status, body=None):
    return MagicMock(ok=200 <= status < 300, status_code=status, history=[], text='denied',
                     json=lambda: body or {})


def test_a_403_on_the_agent_fetch_raises_access_denied():
    client = _client()

    with patch("requests.Session.request", return_value=_response(403)):
        with pytest.raises(AgentAccessDeniedError, match='requires editor access to project 3'):
            client.get_app_version_details(11, 22)


def test_other_fetch_errors_keep_the_old_error_type():
    client = _client()

    with patch("requests.Session.request", return_value=_response(404)):
        with pytest.raises(ApiDetailsRequestError):
            client.get_app_version_details(11, 22)


def test_application_does_not_wrap_access_denied_in_a_tool_exception():
    client = _client()

    with patch("requests.Session.request", return_value=_response(403)):
        with pytest.raises(AgentAccessDeniedError):
            client.application(11, 22)


def test_get_tools_skips_a_sub_agent_the_user_may_not_load():
    # A viewer in the platform hits this for a sub-agent the prefetch could not expand
    from elitea_sdk.runtime.toolkits import tools as tools_mod

    tool = {'type': 'application', 'name': 'child', 'settings': {
        'application_id': 11, 'application_version_id': 22}}

    def denied(*args, **kwargs):
        raise AgentAccessDeniedError("Loading agent 11/22 requires editor access to project 3 (403)")

    with patch.object(tools_mod.ApplicationToolkit, 'get_toolkit', side_effect=denied):
        assert tools_mod.get_tools([tool], elitea_client=SimpleNamespace(project_id=3), llm=None) == []


def test_the_error_message_does_not_assume_a_local_run():
    client = _client()

    with patch("requests.Session.request", return_value=_response(403)):
        with pytest.raises(AgentAccessDeniedError) as exc:
            client.get_app_version_details(11, 22)

    assert 'locally' not in str(exc.value)


def test_get_tools_still_skips_a_sub_agent_that_fails_otherwise():
    # A broken sub-agent must not stop the run (#7064); only the access error is surfaced
    from elitea_sdk.runtime.toolkits import tools as tools_mod

    tool = {'type': 'application', 'name': 'child', 'settings': {
        'application_id': 11, 'application_version_id': 22}}

    with patch.object(tools_mod.ApplicationToolkit, 'get_toolkit', side_effect=RuntimeError('deleted')):
        assert tools_mod.get_tools([tool], elitea_client=SimpleNamespace(project_id=3), llm=None) == []
