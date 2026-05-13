"""
Tests for Jira toolkit authentication error handling.

This module tests the fix for bugs #3850 and #3267 where Jira toolkit displayed
ambiguous error messages for authentication failures.

The fix implements fail-fast credential validation at client construction time
by calling /myself endpoint. This ensures invalid credentials are caught
immediately rather than producing misleading "no documents to index" messages.
"""
import pytest
from unittest.mock import Mock
from langchain_core.tools import ToolException


def create_mock_response(status_code: int, text: str = "") -> Mock:
    """Create a mock response object."""
    response = Mock()
    response.status_code = status_code
    response.text = text
    return response


@pytest.fixture
def jira_client(monkeypatch):
    """Create a JiraClient with mocked session for testing."""
    from elitea_sdk.tools.jira.api_wrapper import JiraClient
    from atlassian import Jira

    client = object.__new__(JiraClient)

    # Mock the session property
    mock_session = Mock()
    monkeypatch.setattr(Jira, 'session', mock_session, raising=False)

    client.url = "https://test.atlassian.net"

    return client


class TestJiraClientCredentialValidation:
    """Test fail-fast credential validation at client construction."""

    def test_invalid_credentials_401_raises_auth_error(self, jira_client):
        """Bug #3850/#3267: Invalid credentials should fail fast with clear error."""
        jira_client.session.get.return_value = create_mock_response(401, "Unauthorized")

        with pytest.raises(ToolException, match="Authentication failed: Invalid username or API key."):
            jira_client._validate_credentials()

        jira_client.session.get.assert_called_once()
        call_args = jira_client.session.get.call_args
        assert "/rest/api/2/myself" in call_args[0][0]
        assert call_args[1]['headers'] == {'Accept': 'application/json'}
        assert call_args[1]['timeout'] == 10

    def test_forbidden_403_raises_permission_error(self, jira_client):
        """403 Forbidden should raise clear permission error."""
        jira_client.session.get.return_value = create_mock_response(403, "Forbidden")

        with pytest.raises(ToolException, match="Authentication failed: Access forbidden."):
            jira_client._validate_credentials()

    def test_other_4xx_errors_raise_generic_auth_error(self, jira_client):
        """Other 4xx errors should raise generic auth error with status code."""
        jira_client.session.get.return_value = create_mock_response(404, "Not Found")

        with pytest.raises(ToolException, match=r"Authentication failed: Unable to connect to Jira \(HTTP 404\)"):
            jira_client._validate_credentials()

    def test_server_error_5xx_raises_auth_error(self, jira_client):
        """5xx errors should also raise auth error (can't validate)."""
        jira_client.session.get.return_value = create_mock_response(500, "Internal Server Error")

        with pytest.raises(ToolException, match=r"Authentication failed: Unable to connect to Jira \(HTTP 500\)"):
            jira_client._validate_credentials()

    def test_valid_credentials_200_passes_silently(self, jira_client):
        """Valid credentials (200 OK) should not raise any exception."""
        jira_client.session.get.return_value = create_mock_response(200, '{"name": "testuser"}')

        jira_client._validate_credentials()

        jira_client.session.get.assert_called_once()


class TestJiraClientInitialization:
    """Test that validation is called during client initialization."""

    def test_init_calls_validate_credentials(self, monkeypatch):
        """JiraClient.__init__ should call _validate_credentials."""
        from elitea_sdk.tools.jira.api_wrapper import JiraClient
        from atlassian import Jira

        validation_called = []

        def mock_validate(self):
            validation_called.append(True)

        monkeypatch.setattr(Jira, '__init__', lambda self, *args, **kwargs: None)
        monkeypatch.setattr(JiraClient, '_validate_credentials', mock_validate)

        _ = JiraClient(url="https://test.atlassian.net")

        assert len(validation_called) == 1, "_validate_credentials should be called once during __init__"

    def test_invalid_credentials_prevent_client_creation(self, monkeypatch):
        """Invalid credentials should prevent client from being usable."""
        from elitea_sdk.tools.jira.api_wrapper import JiraClient
        from atlassian import Jira

        mock_session = Mock()
        mock_session.get.return_value = create_mock_response(401, "Unauthorized")

        monkeypatch.setattr(Jira, '__init__', lambda self, *args, **kwargs: None)
        monkeypatch.setattr(Jira, 'session', mock_session, raising=False)

        with pytest.raises(ToolException, match="Authentication failed"):
            JiraClient(url="https://test.atlassian.net")
