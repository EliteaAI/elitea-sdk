"""HTTP utilities for file downloads with streaming support."""

import os
import tempfile
from typing import Optional

import requests


# Custom exceptions for streaming downloads
class StreamingDownloadError(Exception):
    """Base exception for streaming download errors."""
    pass


class FileSizeLimitExceeded(StreamingDownloadError):
    """Raised when downloaded file exceeds the size limit."""

    def __init__(self, file_name: str, actual_size: int, max_size: int):
        self.file_name = file_name
        self.actual_size = actual_size
        self.max_size = max_size
        self.actual_size_mb = actual_size / (1024 * 1024)
        self.max_size_mb = max_size / (1024 * 1024)
        super().__init__(
            f"File '{file_name}' exceeds size limit: "
            f"{self.actual_size_mb:.1f} MB > {self.max_size_mb:.0f} MB"
        )


class EmptyFileError(StreamingDownloadError):
    """Raised when downloaded file is empty."""

    def __init__(self, file_name: str):
        self.file_name = file_name
        super().__init__(f"Downloaded file '{file_name}' is empty")


class DownloadError(StreamingDownloadError):
    """Raised when download fails due to network or server error."""

    def __init__(self, file_name: str, cause: Exception):
        self.file_name = file_name
        self.cause = cause
        super().__init__(f"Failed to download '{file_name}': {cause}")


# Default chunk size for streaming downloads (8 KB)
DEFAULT_CHUNK_SIZE = 8 * 1024


def stream_download_to_tempfile(
    url: str,
    file_name: str,
    max_size: int,
    timeout: int = 120,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    headers: Optional[dict] = None,
    cookies: Optional[dict] = None,
) -> str:
    """Stream download a file to a temporary file with size limit enforcement.

    Downloads in chunks to minimize memory usage. Aborts early if size limit
    is exceeded during download, cleaning up the partial temp file.

    Args:
        url: Download URL
        file_name: Original file name (used for temp file extension and error messages)
        max_size: Maximum allowed file size in bytes
        timeout: Request timeout in seconds (default 120)
        chunk_size: Size of chunks to download (default 8 KB)
        headers: Optional request headers
        cookies: Optional request cookies

    Returns:
        Path to temporary file containing downloaded content.
        Caller is responsible for cleaning up the temp file.

    Raises:
        FileSizeLimitExceeded: If download exceeds max_size (partial file is cleaned up)
        EmptyFileError: If downloaded file has zero bytes
        DownloadError: If download fails due to network/server error
    """
    # Extract extension for temp file (helps loaders detect file type)
    ext = ('.' + file_name.rsplit('.', 1)[-1].lower()) if '.' in file_name else ''

    temp_path = None
    try:
        with requests.get(url, stream=True, timeout=timeout, headers=headers, cookies=cookies) as resp:
            resp.raise_for_status()

            # Create temp file with proper extension
            with tempfile.NamedTemporaryFile(mode='wb', suffix=ext, delete=False) as tmp:
                temp_path = tmp.name
                total_size = 0

                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive chunks
                        total_size += len(chunk)

                        # Early abort if size limit exceeded
                        if total_size > max_size:
                            raise FileSizeLimitExceeded(
                                file_name=file_name,
                                actual_size=total_size,
                                max_size=max_size,
                            )

                        tmp.write(chunk)

        if total_size == 0:
            raise EmptyFileError(file_name=file_name)

        return temp_path

    except (FileSizeLimitExceeded, EmptyFileError):
        # Clean up temp file on validation errors
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    except Exception as e:
        # Clean up temp file on any error
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        # Re-raise our custom exceptions as-is
        if isinstance(e, StreamingDownloadError):
            raise

        # Wrap other exceptions
        raise DownloadError(file_name=file_name, cause=e) from e
