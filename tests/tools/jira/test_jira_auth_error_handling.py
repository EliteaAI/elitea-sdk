"""
Tests for Jira toolkit authentication error handling.

Bug #3850: Jira returns 404 for both auth failures AND missing issues.
This module validates JiraClient which calls /myself to distinguish between them.
"""
import pytest
from unittest.mock import Mock, patch
from requests.exceptions import HTTPError
from langchain_core.tools import ToolException

from elitea_sdk.tools.jira.api_wrapper import JiraClient


def create_http_error(status_code: int, message: str) -> HTTPError:
    """Create HTTPError with mock response."""
    response = Mock()
    response.status_code = status_code
    error = HTTPError(message)
    error.response = response
    return error


@pytest.fixture
def client():
    """Create JiraClient with mocked parent methods."""
    instance = object.__new__(JiraClient)
    instance.url = "https://test.atlassian.net"
    return instance


class TestJiraAuthErrorHandling:
    """Test JiraClient's 404 auth error disambiguation at HTTP layer."""

    def test_bug_3850_replaces_ambiguous_404_with_clear_auth_error(self, client):
        """
        Bug #3850: Ambiguous 404 for auth failures becomes clear error.

        BEFORE FIX: "Issue does not exist or you do not have permission to see it"
        AFTER FIX:  "Authentication failed: ... verify your Jira toolkit credentials ..."

        This test validates that the ambiguous Jira error message is REPLACED
        with a clear authentication error when /myself confirms invalid credentials.
        """
        ambiguous_msg = "Issue does not exist or you do not have permission to see it."
        error_404 = create_http_error(404, ambiguous_msg)

        # Mock /myself to fail (invalid credentials)
        with patch.object(JiraClient.__bases__[0], 'myself',
                         side_effect=create_http_error(401, "Unauthorized")):
            with pytest.raises(ToolException) as exc:
                client._handle_404_auth_check(error_404)

        error_message = str(exc.value)

        # Critical: ambiguous message is REPLACED, not included
        assert ambiguous_msg not in error_message, \
            "Ambiguous Jira message should be replaced with clear auth error"

        # User gets clear, actionable error instead
        assert "Authentication failed" in error_message
        assert "credentials" in error_message
        assert "verify" in error_message

    def test_auth_failure_returns_clear_error(self, client):
        """404 + /myself fails with 401 = clear 'Authentication failed' message."""
        error_404 = create_http_error(404, "Issue not found")

        # Mock parent's myself() to fail with 401
        with patch.object(JiraClient.__bases__[0], 'myself', side_effect=create_http_error(401, "Unauthorized")):
            with pytest.raises(ToolException) as exc:
                client._handle_404_auth_check(error_404)

        assert "Authentication failed" in str(exc.value)
        assert "credentials" in str(exc.value)

    def test_valid_auth_reraises_original_404(self, client):
        """404 + /myself succeeds = re-raise original 404 (legitimate missing issue)."""
        original_msg = "Issue does not exist or you do not have permission to see it."
        error_404 = create_http_error(404, original_msg)

        # Mock parent's myself() to succeed
        with patch.object(JiraClient.__bases__[0], 'myself', return_value={"name": "user"}):
            with pytest.raises(HTTPError) as exc:
                client._handle_404_auth_check(error_404)

        assert original_msg in str(exc.value)
        assert "Authentication failed" not in str(exc.value)

    def test_any_404_triggers_auth_check(self, client):
        """Any 404 message triggers /myself check (no string matching)."""
        error_404 = create_http_error(404, "Not Found")

        # Mock parent's myself() to fail
        with patch.object(JiraClient.__bases__[0], 'myself', side_effect=create_http_error(401, "Unauthorized")):
            with pytest.raises(ToolException) as exc:
                client._handle_404_auth_check(error_404)

        assert "Authentication failed" in str(exc.value)

    def test_403_on_myself_treated_as_auth_failure(self, client):
        """404 + /myself returns 403 = treat as auth error."""
        error_404 = create_http_error(404, "Not found")

        # Mock parent's myself() to fail with 403
        with patch.object(JiraClient.__bases__[0], 'myself', side_effect=create_http_error(403, "Forbidden")):
            with pytest.raises(ToolException) as exc:
                client._handle_404_auth_check(error_404)

        assert "Authentication failed" in str(exc.value)

    def test_request_integration(self, client):
        """Test complete request() flow with 404 and auth check."""
        error_404 = create_http_error(404, "Not found")

        # Mock both parent methods
        with patch.object(JiraClient.__bases__[0], 'request', side_effect=error_404):
            with patch.object(JiraClient.__bases__[0], 'myself', side_effect=create_http_error(401, "Unauthorized")):
                with pytest.raises(ToolException) as exc:
                    client.request("GET", "/rest/api/2/issue/EL-14")

        assert "Authentication failed" in str(exc.value)

    def test_request_passes_non_404_errors(self, client):
        """Non-404 errors pass through without /myself check."""
        error_500 = create_http_error(500, "Internal Server Error")

        # Mock request to raise 500, myself should NOT be called
        with patch.object(JiraClient.__bases__[0], 'request', side_effect=error_500):
            with patch.object(JiraClient.__bases__[0], 'myself') as mock_myself:
                with pytest.raises(HTTPError) as exc:
                    client.request("GET", "/rest/api/2/issue/EL-14")

        assert "Internal Server Error" in str(exc.value)
        mock_myself.assert_not_called()
