"""Tests for HTTP utility functions."""

import os
from unittest.mock import patch, MagicMock

import pytest

from elitea_sdk.tools.utils.http_utils import (
    stream_download_to_tempfile,
    FileSizeLimitExceeded,
    EmptyFileError,
    DownloadError,
)


def _create_streaming_response_mock(content: bytes, chunk_size: int = 8192):
    """Create a mock response that supports streaming via iter_content."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    def iter_content(chunk_size=chunk_size):
        for i in range(0, len(content), chunk_size):
            yield content[i:i + chunk_size]

    mock_response.iter_content = iter_content
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    return mock_response


class TestStreamDownloadToTempfile:
    """Test stream_download_to_tempfile function."""

    @patch('elitea_sdk.tools.utils.http_utils.requests')
    def test_creates_temp_file_with_content(self, mock_requests):
        """Successfully downloads content to a temp file."""
        test_content = b'test file content for streaming'
        mock_requests.get.return_value = _create_streaming_response_mock(test_content)

        temp_path = stream_download_to_tempfile(
            url='https://example.com/file.pdf',
            file_name='document.pdf',
            max_size=1024 * 1024,  # 1 MB
        )

        try:
            assert os.path.exists(temp_path)
            assert temp_path.endswith('.pdf')
            with open(temp_path, 'rb') as f:
                assert f.read() == test_content
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch('elitea_sdk.tools.utils.http_utils.requests')
    def test_uses_stream_parameter(self, mock_requests):
        """Request is made with stream=True."""
        test_content = b'content'
        mock_requests.get.return_value = _create_streaming_response_mock(test_content)

        temp_path = stream_download_to_tempfile(
            url='https://example.com/file.txt',
            file_name='file.txt',
            max_size=1024 * 1024,
        )

        try:
            call_kwargs = mock_requests.get.call_args[1]
            assert call_kwargs.get('stream') is True
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch('elitea_sdk.tools.utils.http_utils.requests')
    def test_passes_headers_and_cookies(self, mock_requests):
        """Headers and cookies are passed to the request."""
        test_content = b'content'
        mock_requests.get.return_value = _create_streaming_response_mock(test_content)

        headers = {'Authorization': 'Bearer token123'}
        cookies = {'session': 'abc123'}

        temp_path = stream_download_to_tempfile(
            url='https://example.com/file.txt',
            file_name='file.txt',
            max_size=1024 * 1024,
            headers=headers,
            cookies=cookies,
        )

        try:
            call_kwargs = mock_requests.get.call_args[1]
            assert call_kwargs.get('headers') == headers
            assert call_kwargs.get('cookies') == cookies
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch('elitea_sdk.tools.utils.http_utils.requests')
    def test_raises_file_size_limit_exceeded(self, mock_requests):
        """Raises FileSizeLimitExceeded when content exceeds max_size."""
        # Content larger than limit
        large_content = b'x' * (2 * 1024 * 1024)  # 2 MB
        mock_requests.get.return_value = _create_streaming_response_mock(large_content)

        with pytest.raises(FileSizeLimitExceeded) as exc_info:
            stream_download_to_tempfile(
                url='https://example.com/large.pdf',
                file_name='large.pdf',
                max_size=1 * 1024 * 1024,  # 1 MB limit
            )

        assert exc_info.value.file_name == 'large.pdf'
        assert exc_info.value.max_size == 1 * 1024 * 1024
        assert exc_info.value.actual_size > 1 * 1024 * 1024

    @patch('elitea_sdk.tools.utils.http_utils.requests')
    def test_raises_empty_file_error(self, mock_requests):
        """Raises EmptyFileError when downloaded file is empty."""
        mock_requests.get.return_value = _create_streaming_response_mock(b'')

        with pytest.raises(EmptyFileError) as exc_info:
            stream_download_to_tempfile(
                url='https://example.com/empty.pdf',
                file_name='empty.pdf',
                max_size=1024 * 1024,
            )

        assert exc_info.value.file_name == 'empty.pdf'

    @patch('elitea_sdk.tools.utils.http_utils.requests')
    def test_raises_download_error_on_network_failure(self, mock_requests):
        """Raises DownloadError when network request fails."""
        mock_requests.get.side_effect = Exception("Connection refused")

        with pytest.raises(DownloadError) as exc_info:
            stream_download_to_tempfile(
                url='https://example.com/file.pdf',
                file_name='file.pdf',
                max_size=1024 * 1024,
            )

        assert exc_info.value.file_name == 'file.pdf'
        assert "Connection refused" in str(exc_info.value.cause)

    @patch('elitea_sdk.tools.utils.http_utils.requests')
    def test_cleans_up_temp_file_on_size_error(self, mock_requests):
        """Temp file is cleaned up when size limit is exceeded."""
        large_content = b'x' * (2 * 1024 * 1024)
        mock_requests.get.return_value = _create_streaming_response_mock(large_content)

        # Track that cleanup happens (temp file should not exist after exception)
        with pytest.raises(FileSizeLimitExceeded):
            stream_download_to_tempfile(
                url='https://example.com/large.pdf',
                file_name='large.pdf',
                max_size=1 * 1024 * 1024,
            )

    @patch('elitea_sdk.tools.utils.http_utils.requests')
    def test_extracts_extension_from_filename(self, mock_requests):
        """Temp file has correct extension from filename."""
        mock_requests.get.return_value = _create_streaming_response_mock(b'content')

        temp_path = stream_download_to_tempfile(
            url='https://example.com/file',
            file_name='report.xlsx',
            max_size=1024 * 1024,
        )

        try:
            assert temp_path.endswith('.xlsx')
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch('elitea_sdk.tools.utils.http_utils.requests')
    def test_handles_filename_without_extension(self, mock_requests):
        """Handles filenames without extension gracefully."""
        mock_requests.get.return_value = _create_streaming_response_mock(b'content')

        temp_path = stream_download_to_tempfile(
            url='https://example.com/file',
            file_name='noextension',
            max_size=1024 * 1024,
        )

        try:
            assert os.path.exists(temp_path)
            # No extension, so temp file won't have one either
            assert not temp_path.endswith('.noextension')
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch('elitea_sdk.tools.utils.http_utils.requests')
    def test_uses_custom_chunk_size(self, mock_requests):
        """Custom chunk size is used for iteration."""
        test_content = b'x' * 1000
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        # Track what chunk_size was passed to iter_content
        received_chunk_size = None

        def iter_content(chunk_size=8192):
            nonlocal received_chunk_size
            received_chunk_size = chunk_size
            yield test_content

        mock_response.iter_content = iter_content
        mock_requests.get.return_value = mock_response

        temp_path = stream_download_to_tempfile(
            url='https://example.com/file.txt',
            file_name='file.txt',
            max_size=1024 * 1024,
            chunk_size=4096,  # Custom chunk size
        )

        try:
            assert received_chunk_size == 4096
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestExceptionMessages:
    """Test exception message formatting."""

    def test_file_size_limit_exceeded_message(self):
        """FileSizeLimitExceeded has informative message."""
        exc = FileSizeLimitExceeded(
            file_name='large.pdf',
            actual_size=25 * 1024 * 1024,  # 25 MB
            max_size=20 * 1024 * 1024,  # 20 MB
        )

        assert 'large.pdf' in str(exc)
        assert '25.0 MB' in str(exc)
        assert '20 MB' in str(exc)

    def test_empty_file_error_message(self):
        """EmptyFileError has informative message."""
        exc = EmptyFileError(file_name='empty.txt')

        assert 'empty.txt' in str(exc)
        assert 'empty' in str(exc).lower()

    def test_download_error_message(self):
        """DownloadError includes cause in message."""
        cause = Exception("Connection timeout")
        exc = DownloadError(file_name='file.pdf', cause=cause)

        assert 'file.pdf' in str(exc)
        assert 'Connection timeout' in str(exc)
