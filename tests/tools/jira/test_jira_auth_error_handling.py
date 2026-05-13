"""
Tests for Jira toolkit authentication error handling.

This module tests the fix for bug #3850 where Jira toolkit displayed ambiguous
error message "Issue does not exist or you do not have permission to see it"
for authentication failures instead of clear credential validation errors.

The tests validate the JiraClient.raise_for_status override which performs a
pre-flight check against /myself to distinguish auth failures from real 404s.
"""
import pytest
from unittest.mock import Mock
from requests.exceptions import HTTPError
from langchain_core.tools import ToolException


def create_http_error(status_code: int, message: str) -> HTTPError:
    """Create an HTTPError with a mock response."""
    response = Mock()
    response.status_code = status_code
    response.text = message
    error = HTTPError(message)
    error.response = response
    return error


def create_mock_response(status_code: int, text: str = "") -> Mock:
    """Create a mock response object."""
    response = Mock()
    response.status_code = status_code
    response.text = text
    response.reason = "Not Found" if status_code == 404 else "OK"
    return response


@pytest.fixture
def jira_client(monkeypatch):
    from elitea_sdk.tools.jira.api_wrapper import JiraClient
    from atlassian import Jira

    client = object.__new__(JiraClient)

    def mock_parent_raise_for_status(response):
        if response.status_code >= 400:
            raise create_http_error(response.status_code, response.text)

    monkeypatch.setattr(Jira, 'raise_for_status', lambda self, response: mock_parent_raise_for_status(response))

    mock_session = Mock()
    monkeypatch.setattr(Jira, 'session', mock_session, raising=False)

    client.url = "https://test.atlassian.net"

    return client


class TestJiraClientAuthErrorHandling:
    def test_ambiguous_404_with_invalid_auth_returns_auth_error(self, jira_client):
        """Bug #3850: ambiguous 404 + invalid credentials = clear auth error."""
        jira_client.session.get.return_value = create_mock_response(401, "Unauthorized")

        response = create_mock_response(
            404, "Issue does not exist or you do not have permission to see it."
        )

        with pytest.raises(ToolException, match="Authentication failed"):
            jira_client.raise_for_status(response)

        jira_client.session.get.assert_called_once()
        call_args = jira_client.session.get.call_args
        assert "/rest/api/2/myself" in call_args[0][0]
        assert call_args[1]['headers'] == {'Accept': 'application/json'}
        assert call_args[1]['timeout'] == 10

    def test_ambiguous_404_with_valid_auth_returns_original_error(self, jira_client):
        """Ambiguous 404 + valid credentials = pass through original error."""
        jira_client.session.get.return_value = create_mock_response(200, '{"name": "testuser"}')

        response = create_mock_response(
            404, "Issue does not exist or you do not have permission to see it."
        )

        with pytest.raises(HTTPError, match="Issue does not exist"):
            jira_client.raise_for_status(response)

        jira_client.session.get.assert_called_once()

    def test_non_404_errors_pass_through_unchanged(self, jira_client):
        """Non-404 errors pass through without auth check."""
        response = create_mock_response(500, "Internal Server Error")

        with pytest.raises(HTTPError, match="Internal Server Error"):
            jira_client.raise_for_status(response)

        jira_client.session.get.assert_not_called()

    def test_200_response_does_not_raise(self, jira_client):
        """Successful responses don't raise exceptions."""
        response = create_mock_response(200, "OK")

        jira_client.raise_for_status(response)

        jira_client.session.get.assert_not_called()

    def test_session_get_network_error_returns_auth_error(self, jira_client):
        """If session.get() raises network error, treat as auth failure."""
        jira_client.session.get.side_effect = ConnectionError("Network unreachable")

        response = create_mock_response(404, "Issue does not exist or you do not have permission")

        with pytest.raises(ToolException, match="Authentication failed"):
            jira_client.raise_for_status(response)

    def test_no_recursion_possible(self, jira_client):
        """Verify that auth check cannot recurse - uses session directly."""
        jira_client.session.get.return_value = create_mock_response(404, "/myself not found")

        response = create_mock_response(404, "Issue does not exist")

        with pytest.raises(ToolException, match="Authentication failed"):
            jira_client.raise_for_status(response)

        assert jira_client.session.get.call_count == 1


class TestJiraClientAdvancedMode:
    """Test that both standard and advanced_mode code paths are protected."""

    def test_advanced_mode_true_calls_raise_for_status_explicitly(self, jira_client):
        """
        When advanced_mode=True, request() returns response without raising.
        execute_generic_rq then calls raise_for_status() explicitly.
        This test verifies our override catches it.
        """
        jira_client.session.get.return_value = create_mock_response(401, "Unauthorized")

        response = create_mock_response(404, "Issue does not exist or you do not have permission")

        with pytest.raises(ToolException, match="Authentication failed"):
            jira_client.raise_for_status(response)

    def test_standard_mode_false_raise_for_status_called_internally(self, jira_client):
        """
        When advanced_mode=False, request() internally calls raise_for_status().
        This test verifies our override catches it in that path too.
        """
        jira_client.session.get.return_value = create_mock_response(200, '{"name": "testuser"}')

        response = create_mock_response(404, "Issue does not exist or you do not have permission")

        with pytest.raises(HTTPError):
            jira_client.raise_for_status(response)
