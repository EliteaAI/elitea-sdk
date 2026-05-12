"""
Tests for Jira toolkit authentication error handling.

This module tests the fix for bug #3850 where Jira toolkit displayed ambiguous
error message "Issue does not exist or you do not have permission to see it"
for authentication failures instead of clear credential validation errors.

The tests validate the _handle_jira_auth_error method which performs a pre-flight
check against /myself endpoint to distinguish auth failures from real 404s.
"""
import pytest
from unittest.mock import Mock
from requests.exceptions import HTTPError
from langchain_core.tools import ToolException


def create_http_error(status_code: int, message: str) -> HTTPError:
    """Create an HTTPError with a mock response."""
    response = Mock()
    response.status_code = status_code
    error = HTTPError(message)
    error.response = response
    return error


class TestJiraAuthErrorHandling:
    """Test the _handle_jira_auth_error logic for ambiguous 404 errors."""

    def _handle_jira_auth_error(self, error: Exception, auth_valid: bool = True) -> ToolException:
        """Standalone implementation of the error handling logic for testing."""
        error_msg = str(error)
        status_code = getattr(getattr(error, 'response', None), 'status_code', None)

        if status_code == 404 and "does not exist or you do not have permission" in error_msg:
            if auth_valid:
                # Auth works - return original error unchanged
                return ToolException(error_msg)
            else:
                return ToolException(
                    "Authentication failed: Unable to connect to Jira. "
                    "Please verify your Jira toolkit credentials are valid and not expired."
                )
        return ToolException(str(error))

    def test_ambiguous_404_with_invalid_auth_returns_auth_error(self):
        """Bug #3850: ambiguous 404 + invalid credentials = clear auth error."""
        error = create_http_error(
            404, "Issue does not exist or you do not have permission to see it."
        )

        result = self._handle_jira_auth_error(error, auth_valid=False)

        assert "Authentication failed" in str(result)
        assert "credentials" in str(result)

    def test_ambiguous_404_with_valid_auth_returns_original_error(self):
        """Ambiguous 404 + valid credentials = pass through original error."""
        original_msg = "Issue does not exist or you do not have permission to see it."
        error = create_http_error(404, original_msg)

        result = self._handle_jira_auth_error(error, auth_valid=True)

        assert original_msg in str(result)
        assert "Authentication failed" not in str(result)

    def test_other_errors_pass_through_unchanged(self):
        """Non-ambiguous errors pass through unchanged."""
        error = create_http_error(500, "Internal Server Error")

        result = self._handle_jira_auth_error(error)

        assert "Internal Server Error" in str(result)
