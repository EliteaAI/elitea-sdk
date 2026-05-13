"""
Tests for Confluence add_file_to_page version conflict retry logic.

Validates the fix for bug #4093 where add_file_to_page raises
ConflictException (Hibernate StaleStateException) on page update
when position='prepend'. The attachment upload modifies the page's
HIBERNATEVERSION, causing a version conflict on the subsequent
update_page call.
"""
import pytest
from unittest.mock import Mock, MagicMock
from requests.exceptions import HTTPError
from langchain_core.tools import ToolException


def create_http_error(status_code: int, message: str = "") -> HTTPError:
    """Create an HTTPError with a mock response."""
    response = Mock()
    response.status_code = status_code
    response.text = message
    response.reason = "Conflict" if status_code == 409 else "Server Error"
    error = HTTPError(message)
    error.response = response
    return error


def make_page(version=1, body="<p>existing content</p>"):
    """Create a mock Confluence page response."""
    return {
        'title': 'Test Page',
        'body': {'storage': {'value': body}},
        'version': {'number': version},
        '_links': {'webui': '/spaces/TEST/pages/123/Test+Page'},
    }


@pytest.fixture
def wrapper():
    """Create a ConfluenceAPIWrapper with mocked dependencies."""
    from elitea_sdk.tools.confluence.api_wrapper import ConfluenceAPIWrapper

    w = ConfluenceAPIWrapper.model_construct(
        base_url="https://test.atlassian.net",
        cloud=True,
        client=MagicMock(),
        elitea=MagicMock(),
    )
    return w


class TestAddFileToPageRetry:
    """Tests for version conflict retry in add_file_to_page."""

    def test_prepend_succeeds_on_first_attempt(self, wrapper):
        """Normal case: no conflict, update succeeds immediately."""
        wrapper._upload_file_from_artifact = Mock(return_value={
            'filename': 'test.txt', 'mime_type': 'text/plain',
            'size': 100, 'download_url': ''
        })
        wrapper.client.get_page_by_id.return_value = make_page()
        wrapper.client.update_page.return_value = {}

        result = wrapper.add_file_to_page(
            page_id='123', filepath='/bucket/test.txt', position='prepend'
        )

        assert "test.txt" in result
        assert "uploaded and added" in result
        wrapper.client.update_page.assert_called_once()

    def test_prepend_retries_on_409_conflict(self, wrapper):
        """Bug #4093: 409 conflict triggers retry with fresh page fetch."""
        wrapper._upload_file_from_artifact = Mock(return_value={
            'filename': 'test.txt', 'mime_type': 'text/plain',
            'size': 100, 'download_url': ''
        })
        wrapper.client.get_page_by_id.return_value = make_page()
        # First call raises 409, second succeeds
        wrapper.client.update_page.side_effect = [
            create_http_error(409, "StaleStateException"),
            {},
        ]

        result = wrapper.add_file_to_page(
            page_id='123', filepath='/bucket/test.txt', position='prepend'
        )

        assert "uploaded and added" in result
        assert wrapper.client.update_page.call_count == 2
        # Page was re-fetched before retry
        assert wrapper.client.get_page_by_id.call_count == 2

    def test_append_retries_on_409_conflict(self, wrapper):
        """Retry logic also works for append position."""
        wrapper._upload_file_from_artifact = Mock(return_value={
            'filename': 'img.png', 'mime_type': 'image/png',
            'size': 200, 'download_url': ''
        })
        wrapper.client.get_page_by_id.return_value = make_page()
        wrapper.client.update_page.side_effect = [
            create_http_error(409, "StaleStateException"),
            {},
        ]

        result = wrapper.add_file_to_page(
            page_id='123', filepath='/bucket/img.png', position='append'
        )

        assert "uploaded and added" in result
        assert wrapper.client.update_page.call_count == 2

    def test_raises_after_max_retries_exhausted(self, wrapper):
        """All 3 retries fail with 409 → raises ToolException."""
        wrapper._upload_file_from_artifact = Mock(return_value={
            'filename': 'test.txt', 'mime_type': 'text/plain',
            'size': 100, 'download_url': ''
        })
        wrapper.client.get_page_by_id.return_value = make_page()
        wrapper.client.update_page.side_effect = create_http_error(
            409, "StaleStateException"
        )

        with pytest.raises(ToolException, match="Failed to add file to page"):
            wrapper.add_file_to_page(
                page_id='123', filepath='/bucket/test.txt', position='prepend'
            )

        assert wrapper.client.update_page.call_count == 3

    def test_non_409_error_not_retried(self, wrapper):
        """Non-409 HTTPError is raised immediately without retry."""
        wrapper._upload_file_from_artifact = Mock(return_value={
            'filename': 'test.txt', 'mime_type': 'text/plain',
            'size': 100, 'download_url': ''
        })
        wrapper.client.get_page_by_id.return_value = make_page()
        wrapper.client.update_page.side_effect = create_http_error(
            500, "Internal Server Error"
        )

        with pytest.raises(ToolException, match="Failed to add file to page"):
            wrapper.add_file_to_page(
                page_id='123', filepath='/bucket/test.txt', position='prepend'
            )

        wrapper.client.update_page.assert_called_once()

    def test_retry_uses_fresh_page_body(self, wrapper):
        """On retry, the page body is re-fetched to get the latest content."""
        wrapper._upload_file_from_artifact = Mock(return_value={
            'filename': 'test.txt', 'mime_type': 'text/plain',
            'size': 100, 'download_url': ''
        })
        # First fetch returns old body, second returns updated body
        wrapper.client.get_page_by_id.side_effect = [
            make_page(body="<p>old</p>"),
            make_page(body="<p>updated by attachment</p>"),
        ]
        wrapper.client.update_page.side_effect = [
            create_http_error(409, "StaleStateException"),
            {},
        ]

        wrapper.add_file_to_page(
            page_id='123', filepath='/bucket/test.txt', position='prepend'
        )

        # Second update_page call should use the fresh body
        second_call_body = wrapper.client.update_page.call_args_list[1]
        assert "updated by attachment" in second_call_body.kwargs.get('body', '')
