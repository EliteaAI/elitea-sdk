import pytest

from elitea_sdk.configurations.confluence import ConfluenceConfiguration
from elitea_sdk.configurations.jira import JiraConfiguration
from elitea_sdk.configurations.utils import _hosting_to_cloud, _resolve_jira_version_candidates


@pytest.mark.parametrize(
    ('hosting', 'base_url', 'expected_cloud'),
    [
        ('Cloud', 'https://jira.company.com', True),
        ('Server', 'https://company.atlassian.net', False),
        ('Auto', 'https://company.atlassian.net', True),
        ('Auto', 'https://jira.company.com', False),
    ],
)
def test_hosting_to_cloud_respects_credential_hosting_values(hosting, base_url, expected_cloud):
    assert _hosting_to_cloud(hosting, base_url) is expected_cloud


@pytest.mark.parametrize(
    ('settings', 'expected_message'),
    [
        (
            {
                'hosting': 'Cloud',
                'base_url': 'https://jira.company.com',
                'username': 'user@example.com',
                'api_key': 'token',
            },
            'Hosting is set to Cloud',
        ),
        (
            {
                'hosting': 'Server',
                'base_url': 'https://company.atlassian.net',
                'username': 'user@example.com',
                'api_key': 'token',
            },
            'Hosting is set to Server',
        ),
    ],
)
def test_jira_check_connection_rejects_hosting_url_mismatch(monkeypatch, settings, expected_message):
    def fail_get(*args, **kwargs):
        raise AssertionError('requests.get should not be called when hosting and URL mismatch')

    monkeypatch.setattr('requests.get', fail_get)

    error = JiraConfiguration.check_connection(settings)

    assert expected_message in error
    assert '.atlassian.net' in error


@pytest.mark.parametrize(
    ('settings', 'expected_message'),
    [
        (
            {
                'hosting': 'Cloud',
                'base_url': 'https://confluence.company.com',
                'username': 'user@example.com',
                'api_key': 'token',
            },
            'Hosting is set to Cloud',
        ),
        (
            {
                'hosting': 'Server',
                'base_url': 'https://company.atlassian.net/wiki',
                'username': 'user@example.com',
                'api_key': 'token',
            },
            'Hosting is set to Server',
        ),
    ],
)
def test_confluence_check_connection_rejects_hosting_url_mismatch(monkeypatch, settings, expected_message):
    def fail_get(*args, **kwargs):
        raise AssertionError('requests.get should not be called when hosting and URL mismatch')

    monkeypatch.setattr('requests.get', fail_get)

    error = ConfluenceConfiguration.check_connection(settings)

    assert expected_message in error
    assert '.atlassian.net' in error


@pytest.mark.parametrize(
    ('check_connection', 'settings', 'expected_message'),
    [
        (
            JiraConfiguration.check_connection,
            {
                'hosting': 'Cloud',
                'base_url': 'https://',
                'username': 'user@example.com',
                'api_key': 'token',
            },
            'Jira URL is invalid',
        ),
        (
            ConfluenceConfiguration.check_connection,
            {
                'hosting': 'Cloud',
                'base_url': 'https://',
                'username': 'user@example.com',
                'api_key': 'token',
            },
            'Confluence URL is invalid',
        ),
    ],
)
def test_check_connection_rejects_malformed_urls_before_hosting_validation(
    monkeypatch,
    check_connection,
    settings,
    expected_message,
):
    def fail_get(*args, **kwargs):
        raise AssertionError('requests.get should not be called when URL is malformed')

    monkeypatch.setattr('requests.get', fail_get)

    error = check_connection(settings)

    assert error == expected_message


def test_jira_check_connection_auto_keeps_existing_url_inference(monkeypatch):
    class Response:
        status_code = 200
        headers = {'Content-Type': 'application/json'}

        def json(self):
            return {}

    seen_urls = []

    def fake_get(url, **kwargs):
        seen_urls.append(url)
        return Response()

    monkeypatch.setattr('requests.get', fake_get)

    error = JiraConfiguration.check_connection(
        {
            'hosting': 'Auto',
            'base_url': 'https://jira.company.com',
            'username': 'user@example.com',
            'api_key': 'token',
        }
    )

    assert error is None
    # Auto on a non-Atlassian URL resolves to Server, and a 200 on the preferred
    # version must not trigger the fallback probe.
    assert seen_urls == ['https://jira.company.com/rest/api/2/myself']


def test_confluence_check_connection_auto_keeps_existing_url_inference(monkeypatch):
    class Response:
        status_code = 200

    seen_urls = []

    def fake_get(url, **kwargs):
        seen_urls.append(url)
        return Response()

    monkeypatch.setattr('requests.get', fake_get)

    error = ConfluenceConfiguration.check_connection(
        {
            'hosting': 'Auto',
            'base_url': 'https://confluence.company.com',
            'username': 'user@example.com',
            'api_key': 'token',
        }
    )

    assert error is None
    assert seen_urls == ['https://confluence.company.com/rest/api/user/current']

class _Resp:
    def __init__(self, status_code, content_type='application/json'):
        self.status_code = status_code
        self.headers = {'Content-Type': content_type}

    def __bool__(self):
        # Mirrors requests.Response, which is falsy for every error status.
        return 200 <= self.status_code < 300

    def json(self):
        return {}


def _collect(monkeypatch, responses):
    """Serve *responses* in order, recording the URLs they were requested from."""
    seen = []

    def fake_get(url, **kwargs):
        seen.append(url)
        return responses[min(len(seen) - 1, len(responses) - 1)]

    monkeypatch.setattr('requests.get', fake_get)
    return seen


def _jira_settings(**overrides):
    settings = {
        'hosting': 'Auto',
        'base_url': 'https://company.atlassian.net',
        'username': 'user@example.com',
        'api_key': 'token',
    }
    settings.update(overrides)
    return settings


@pytest.mark.parametrize(
    ('hosting', 'base_url', 'expected'),
    [
        ('Cloud', 'https://company.atlassian.net', ['3']),
        ('Server', 'https://jira.company.com', ['2']),
        ('Server', 'https://company.atlassian.net', ['2']),
        ('Auto', 'https://company.atlassian.net', ['3', '2']),
        ('Auto', 'https://jira.company.com', ['2', '3']),
        (None, 'https://company.atlassian.net', ['3', '2']),
        ('', 'https://jira.company.com', ['2', '3']),
        ('cloud', 'https://jira.company.com', ['3']),
        ('server', 'https://company.atlassian.net', ['2']),
    ],
)
def test_resolve_jira_version_candidates(hosting, base_url, expected):
    assert _resolve_jira_version_candidates(hosting, base_url) == expected


def test_explicit_cloud_uses_v3_in_a_single_request(monkeypatch):
    seen = _collect(monkeypatch, [_Resp(200)])

    assert JiraConfiguration.check_connection(_jira_settings(hosting='Cloud')) is None
    assert seen == ['https://company.atlassian.net/rest/api/3/myself']


def test_explicit_server_uses_v2_and_never_probes_v3(monkeypatch):
    seen = _collect(monkeypatch, [_Resp(200)])

    settings = _jira_settings(hosting='Server', base_url='https://jira.company.com')
    assert JiraConfiguration.check_connection(settings) is None
    assert seen == ['https://jira.company.com/rest/api/2/myself']


def test_auto_falls_back_to_v2_when_v3_is_forbidden(monkeypatch):
    """The fallback finds the working version, but only the check can use it."""
    seen = _collect(monkeypatch, [_Resp(403), _Resp(200)])

    error = JiraConfiguration.check_connection(_jira_settings())

    assert seen == [
        'https://company.atlassian.net/rest/api/3/myself',
        'https://company.atlassian.net/rest/api/2/myself',
    ]
    assert 'Connected using REST API v2' in error
    # Hosting would be a dead end here: the matching value is refused by the
    # hosting/URL consistency check. The wording also has to name the Base URL —
    # the UI routes an error to the field its text mentions, and the Hosting
    # select renders none.
    assert "Set the toolkit's API Version to 2." in error
    assert 'Set Hosting' not in error


def test_fallback_win_names_cloud_when_v3_is_the_working_version(monkeypatch):
    """Auto misdetects a custom-domain Cloud site as Server; v2 is what breaks."""
    seen = _collect(monkeypatch, [_Resp(403), _Resp(200)])

    settings = _jira_settings(base_url='https://jira.acme.com')
    error = JiraConfiguration.check_connection(settings)

    assert seen == [
        'https://jira.acme.com/rest/api/2/myself',
        'https://jira.acme.com/rest/api/3/myself',
    ]
    assert 'Connected using REST API v3' in error
    assert 'resolves to v2' in error
    assert "Set the toolkit's API Version to 3." in error
    assert 'Set Hosting' not in error


def test_preferred_version_success_reports_no_mismatch(monkeypatch):
    """A credential the toolkit will actually work with must stay silent."""
    seen = _collect(monkeypatch, [_Resp(200)])

    assert JiraConfiguration.check_connection(_jira_settings()) is None
    assert seen == ['https://company.atlassian.net/rest/api/3/myself']


def test_explicit_hosting_never_reports_a_mismatch(monkeypatch):
    """Explicit hosting yields one candidate, so a win is always the toolkit's."""
    _collect(monkeypatch, [_Resp(200)])

    settings = _jira_settings(hosting='Server', base_url='https://jira.company.com')
    assert JiraConfiguration.check_connection(settings) is None


def test_auto_reports_both_versions_when_all_are_forbidden(monkeypatch):
    seen = _collect(monkeypatch, [_Resp(403)])

    error = JiraConfiguration.check_connection(_jira_settings())

    assert len(seen) == 2
    assert 'tried API v3, v2' in error


def test_explicit_hosting_403_message_omits_the_version_list(monkeypatch):
    _collect(monkeypatch, [_Resp(403)])

    error = JiraConfiguration.check_connection(_jira_settings(hosting='Cloud'))

    assert 'tried API' not in error


def test_401_short_circuits_without_probing_the_other_version(monkeypatch):
    seen = _collect(monkeypatch, [_Resp(401)])

    error = JiraConfiguration.check_connection(_jira_settings())

    assert len(seen) == 1
    assert 'Authentication failed' in error


@pytest.mark.parametrize('status', [429, 500, 503])
def test_non_version_statuses_do_not_fall_back(monkeypatch, status):
    seen = _collect(monkeypatch, [_Resp(status)])

    error = JiraConfiguration.check_connection(_jira_settings())

    assert len(seen) == 1
    assert str(status) in error


def test_permissions_failure_wins_over_a_missing_version(monkeypatch):
    """A 404 only means the probed version is absent, so the 403 is the real answer."""
    seen = _collect(monkeypatch, [_Resp(404), _Resp(403)])

    error = JiraConfiguration.check_connection(_jira_settings())

    assert len(seen) == 2
    assert 'Access forbidden' in error
    assert 'tried API v3, v2' in error
    assert 'not found' not in error


def test_permissions_failure_wins_over_a_less_actionable_status(monkeypatch):
    """403 names a fixable cause; 400 does not, so probe order must not decide."""
    seen = _collect(monkeypatch, [_Resp(400), _Resp(403)])

    error = JiraConfiguration.check_connection(_jira_settings())

    assert len(seen) == 2
    assert 'Access forbidden' in error
    assert 'Connection failed with status 400' not in error


def test_all_versions_404_still_reports_a_missing_endpoint(monkeypatch):
    _collect(monkeypatch, [_Resp(404)])

    error = JiraConfiguration.check_connection(_jira_settings())

    assert '404' in error
    assert 'tried API v3, v2' in error


def test_later_server_error_is_reported_over_the_first_403(monkeypatch):
    _collect(monkeypatch, [_Resp(403), _Resp(503)])

    error = JiraConfiguration.check_connection(_jira_settings())

    assert '503' in error


def test_non_json_200_is_rejected_without_falling_back(monkeypatch):
    seen = _collect(monkeypatch, [_Resp(200, content_type='text/html')])

    error = JiraConfiguration.check_connection(_jira_settings())

    assert len(seen) == 1
    assert 'non-JSON response' in error


def test_discovery_runs_once_after_all_versions_404(monkeypatch):
    seen = _collect(monkeypatch, [_Resp(404)])

    settings = _jira_settings(hosting='Auto', base_url='https://company.com/jira')
    error = JiraConfiguration.check_connection(settings)

    assert seen[:2] == [
        'https://company.com/jira/rest/api/2/myself',
        'https://company.com/jira/rest/api/3/myself',
    ]
    assert 'https://company.com/rest/api/2/myself' in seen[2:]
    assert '404' in error


def test_confluence_check_is_unchanged_by_jira_version_selection(monkeypatch):
    seen = _collect(monkeypatch, [_Resp(200)])

    settings = {
        'hosting': 'Cloud',
        'base_url': 'https://company.atlassian.net',
        'username': 'user@example.com',
        'api_key': 'token',
    }
    assert ConfluenceConfiguration.check_connection(settings) is None
    assert seen == ['https://company.atlassian.net/wiki/rest/api/user/current']


def test_pinned_api_version_overrides_hosting_and_skips_the_fallback(monkeypatch):
    """An explicit toolkit api_version must be the version the check probes."""
    seen = _collect(monkeypatch, [_Resp(200)])

    settings = _jira_settings(hosting='Auto', api_version='2')
    assert JiraConfiguration.check_connection(settings) is None
    assert seen == ['https://company.atlassian.net/rest/api/2/myself']


def test_auto_api_version_leaves_hosting_in_charge(monkeypatch):
    seen = _collect(monkeypatch, [_Resp(200)])

    settings = _jira_settings(hosting='Auto', api_version='Auto')
    assert JiraConfiguration.check_connection(settings) is None
    assert seen == ['https://company.atlassian.net/rest/api/3/myself']


@pytest.mark.parametrize(
    ('base_url', 'working_version'),
    [
        ('https://company.atlassian.net', '2'),
        ('https://jira.acme.com', '3'),
    ],
)
def test_fallback_win_advice_is_actually_satisfiable(monkeypatch, base_url, working_version):
    """Following the remediation must produce a passing check, not a new rejection."""
    seen = _collect(monkeypatch, [_Resp(403), _Resp(200)])
    error = JiraConfiguration.check_connection(_jira_settings(base_url=base_url))
    assert f"Set the toolkit's API Version to {working_version}." in error

    seen.clear()
    _collect(monkeypatch, [_Resp(200)])
    settings = _jira_settings(base_url=base_url, api_version=working_version)

    assert JiraConfiguration.check_connection(settings) is None


def test_confluence_rejects_cloud_only_v2_on_a_server_deployment(monkeypatch):
    """Server serves the v1 identity endpoint, so only an explicit guard catches this."""
    def fail_get(*args, **kwargs):
        raise AssertionError('no request should be made for an impossible version')

    monkeypatch.setattr('requests.get', fail_get)

    error = ConfluenceConfiguration.check_connection({
        'hosting': 'Server',
        'base_url': 'https://confluence.company.com',
        'api_version': '2',
        'username': 'user@example.com',
        'api_key': 'token',
    })

    assert 'Cloud only' in error
    assert 'Set API Version to 1 or Auto.' in error


@pytest.mark.parametrize('api_version', ['1', 'Auto', None])
def test_confluence_server_accepts_every_valid_version(monkeypatch, api_version):
    """The remediation must be satisfiable, and Auto must never trip the guard."""
    seen = _collect(monkeypatch, [_Resp(200)])

    error = ConfluenceConfiguration.check_connection({
        'hosting': 'Server',
        'base_url': 'https://confluence.company.com',
        'api_version': api_version,
        'username': 'user@example.com',
        'api_key': 'token',
    })

    assert error is None
    assert seen == ['https://confluence.company.com/rest/api/user/current']


def test_confluence_cloud_allows_v2(monkeypatch):
    seen = _collect(monkeypatch, [_Resp(200)])

    error = ConfluenceConfiguration.check_connection({
        'hosting': 'Cloud',
        'base_url': 'https://company.atlassian.net',
        'api_version': '2',
        'username': 'user@example.com',
        'api_key': 'token',
    })

    assert error is None
    assert seen == ['https://company.atlassian.net/wiki/rest/api/user/current']