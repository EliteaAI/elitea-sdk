import pytest

from elitea_sdk.configurations.confluence import ConfluenceConfiguration
from elitea_sdk.configurations.jira import JiraConfiguration
from elitea_sdk.configurations.utils import _hosting_to_cloud


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
    assert seen_urls == ['https://jira.company.com/rest/api/latest/myself']


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