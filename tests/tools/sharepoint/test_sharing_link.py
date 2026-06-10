"""Tests for SharePoint sharing link support."""

import base64
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.tools import ToolException
from elitea_sdk.tools.sharepoint.graph_wrapper import SharepointGraphWrapper
from elitea_sdk.tools.sharepoint.api_wrapper import ReadFromSharingLink


def _encode_sharing_url(sharing_url: str) -> str:
    """Encode a sharing URL for the Graph /shares endpoint (test helper)."""
    encoded = base64.b64encode(sharing_url.encode('utf-8')).decode('utf-8')
    return 'u!' + encoded.rstrip('=').replace('/', '_').replace('+', '-')


class TestSharingUrlEncoding:
    """Test the URL encoding logic for Graph /shares endpoint."""

    def test_encode_simple_url(self):
        """Basic URL encoding produces valid base64url with u! prefix."""
        url = "https://example.sharepoint.com/:x:/s/site/file"
        encoded = _encode_sharing_url(url)

        assert encoded.startswith("u!")
        # Verify base64url chars (no +, /, =)
        payload = encoded[2:]
        assert '+' not in payload
        assert '/' not in payload
        assert '=' not in payload

    def test_encode_url_with_special_chars(self):
        """URL with special characters encodes and decodes correctly."""
        url = "https://company-my.sharepoint.com/:x:/p/user/ABC+DEF/GHI?e=token"
        encoded = _encode_sharing_url(url)

        assert encoded.startswith("u!")
        payload = encoded[2:]

        # Reverse the encoding to verify correctness
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += '=' * padding
        payload = payload.replace('_', '/').replace('-', '+')
        decoded = base64.b64decode(payload).decode('utf-8')
        assert decoded == url

    def test_encode_personal_onedrive_url(self):
        """Personal OneDrive URLs encode correctly."""
        url = "https://contoso-my.sharepoint.com/:x:/p/john_doe/EbC123ABC?e=xyz789"
        encoded = _encode_sharing_url(url)

        assert encoded.startswith("u!")
        # Should be valid base64url
        payload = encoded[2:]
        assert all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_' for c in payload)

    def test_encode_sharepoint_site_url(self):
        """SharePoint site sharing URLs encode correctly."""
        url = "https://contoso.sharepoint.com/:w:/s/TeamSite/Document.docx?e=abc123"
        encoded = _encode_sharing_url(url)

        assert encoded.startswith("u!")

    @pytest.mark.parametrize("file_type_indicator", [
        ":x:",  # Excel
        ":w:",  # Word
        ":p:",  # PowerPoint
        ":t:",  # Text
        ":b:",  # PDF
        ":f:",  # Folder
    ])
    def test_encode_various_file_types(self, file_type_indicator):
        """Different file type indicators all encode correctly."""
        url = f"https://company.sharepoint.com/{file_type_indicator}/s/site/file"
        encoded = _encode_sharing_url(url)

        assert encoded.startswith("u!")
        assert len(encoded) > 2


class TestReadFileFromSharingLinkValidation:
    """Test input validation for read_file_from_sharing_link."""

    def test_invalid_url_empty_string(self):
        """Empty URL raises ToolException."""
        
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        with pytest.raises(ToolException) as exc_info:
            wrapper.read_file_from_sharing_link("")
        assert "Invalid sharing URL" in str(exc_info.value)

    def test_invalid_url_not_https(self):
        """Non-HTTPS URL raises ToolException."""

        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        with pytest.raises(ToolException) as exc_info:
            wrapper.read_file_from_sharing_link("http://insecure.com/file")
        assert "Invalid sharing URL" in str(exc_info.value)

    def test_invalid_url_none(self):
        """None URL raises ToolException."""

        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        with pytest.raises(ToolException) as exc_info:
            wrapper.read_file_from_sharing_link(None)
        assert "Invalid sharing URL" in str(exc_info.value)


class TestReadFileFromSharingLinkBaseWrapper:
    """Test base wrapper behavior for sharing links."""

    def test_base_wrapper_raises_not_supported(self):
        """Base wrapper method raises ToolException with helpful message."""
        from elitea_sdk.tools.sharepoint.base_wrapper import BaseSharepointWrapper

        # Create a minimal mock that inherits from BaseSharepointWrapper
        class MockWrapper(BaseSharepointWrapper):
            def read_list(self, *args, **kwargs): pass
            def get_lists(self, *args, **kwargs): pass
            def get_list_columns(self, *args, **kwargs): pass
            def create_list_item(self, *args, **kwargs): pass
            def get_files_list(self, *args, **kwargs): pass
            def read_file(self, *args, **kwargs): pass
            def load_file_content_in_bytes(self, *args, **kwargs): pass
            def upload_file(self, *args, **kwargs): pass
            def add_attachment_to_list_item(self, *args, **kwargs): pass

        wrapper = MockWrapper()

        with pytest.raises(ToolException) as exc_info:
            wrapper.read_file_from_sharing_link("https://example.com/:x:/...")

        assert "Graph API" in str(exc_info.value)
        assert "delegated access" in str(exc_info.value)


class TestReadFileFromSharingLinkGraphWrapper:
    """Test Graph wrapper implementation for sharing links."""

    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.parse_file_content')
    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.requests')
    def test_calls_shares_endpoint_with_encoded_url(self, mock_requests, mock_parse):
        """Graph wrapper calls /shares endpoint with properly encoded URL."""

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'name': 'test.xlsx',
            '@microsoft.graph.downloadUrl': 'https://download.url/file'
        }

        mock_content_response = MagicMock()
        mock_content_response.content = b'file content'
        mock_content_response.raise_for_status = MagicMock()

        mock_requests.get.side_effect = [mock_response, mock_content_response]
        mock_parse.return_value = "parsed content"

        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        result = wrapper.read_file_from_sharing_link(
            "https://company-my.sharepoint.com/:x:/p/user/file"
        )

        # Verify /shares endpoint was called with encoded URL
        call_url = mock_requests.get.call_args_list[0][0][0]
        assert '/shares/u!' in call_url
        assert '/driveItem' in call_url

    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.parse_file_content')
    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.requests')
    def test_uses_download_url_when_available(self, mock_requests, mock_parse):
        """Uses @microsoft.graph.downloadUrl when present in response."""

        download_url = 'https://direct-download.sharepoint.com/file.xlsx'

        mock_metadata_response = MagicMock()
        mock_metadata_response.ok = True
        mock_metadata_response.status_code = 200
        mock_metadata_response.json.return_value = {
            'name': 'test.xlsx',
            '@microsoft.graph.downloadUrl': download_url
        }

        mock_content_response = MagicMock()
        mock_content_response.content = b'excel file bytes'
        mock_content_response.raise_for_status = MagicMock()

        mock_requests.get.side_effect = [mock_metadata_response, mock_content_response]
        mock_parse.return_value = "parsed content"

        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        wrapper.read_file_from_sharing_link(
            "https://company.sharepoint.com/:x:/s/site/file"
        )

        # Second call should be to the download URL
        second_call_url = mock_requests.get.call_args_list[1][0][0]
        assert second_call_url == download_url

    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.parse_file_content')
    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.requests')
    def test_passes_prefer_header(self, mock_requests, mock_parse):
        """Request includes Prefer header for link redemption."""

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'name': 'test.txt',
            '@microsoft.graph.downloadUrl': 'https://download.url/file'
        }

        mock_content_response = MagicMock()
        mock_content_response.content = b'content'
        mock_content_response.raise_for_status = MagicMock()

        mock_requests.get.side_effect = [mock_response, mock_content_response]
        mock_parse.return_value = "parsed"

        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        wrapper.read_file_from_sharing_link(
            "https://company.sharepoint.com/:t:/s/site/file.txt"
        )

        # Check headers of first call
        call_kwargs = mock_requests.get.call_args_list[0][1]
        assert 'Prefer' in call_kwargs['headers']
        assert 'redeemSharingLinkIfNecessary' in call_kwargs['headers']['Prefer']

    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.parse_file_content')
    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.requests')
    def test_extracts_filename_from_response(self, mock_requests, mock_parse):
        """Filename is extracted from driveItem response."""

        expected_filename = 'quarterly-report.xlsx'

        mock_metadata_response = MagicMock()
        mock_metadata_response.ok = True
        mock_metadata_response.status_code = 200
        mock_metadata_response.json.return_value = {
            'name': expected_filename,
            '@microsoft.graph.downloadUrl': 'https://download.url/file'
        }

        mock_content_response = MagicMock()
        mock_content_response.content = b'file bytes'
        mock_content_response.raise_for_status = MagicMock()

        mock_requests.get.side_effect = [mock_metadata_response, mock_content_response]
        mock_parse.return_value = "parsed"

        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        wrapper.read_file_from_sharing_link(
            "https://company.sharepoint.com/:x:/s/site/EncodedId"
        )

        # Verify parse_file_content was called with correct filename
        mock_parse.assert_called_once()
        call_kwargs = mock_parse.call_args[1]
        assert call_kwargs['file_name'] == expected_filename

    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.parse_file_content')
    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.requests')
    def test_403_triggers_public_link_fallback(self, mock_requests, mock_parse):
        """Graph 403 response triggers _download_public_link fallback."""

        mock_403_response = MagicMock()
        mock_403_response.status_code = 403
        mock_403_response.json.return_value = {
            'error': {'message': 'Access denied'}
        }

        mock_requests.get.return_value = mock_403_response
        mock_parse.return_value = "parsed fallback content"

        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Mock the fallback method on the instance
        with patch.object(wrapper, '_download_public_link') as mock_download:
            mock_download.return_value = ('fallback-file.pdf', b'fallback content')

            result = wrapper.read_file_from_sharing_link(
                "https://other-tenant.sharepoint.com/:b:/g/public/file"
            )

            # Verify fallback was called
            mock_download.assert_called_once_with(
                "https://other-tenant.sharepoint.com/:b:/g/public/file"
            )
            # Verify parse was called with fallback result
            mock_parse.assert_called_once()
            call_kwargs = mock_parse.call_args[1]
            assert call_kwargs['file_name'] == 'fallback-file.pdf'
            assert call_kwargs['file_content'] == b'fallback content'

    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.requests')
    def test_403_fallback_failure_returns_graph_error(self, mock_requests):
        """When 403 fallback fails, original Graph error is returned."""

        mock_403_response = MagicMock()
        mock_403_response.status_code = 403
        mock_403_response.json.return_value = {
            'error': {'message': 'The sharing link no longer exists'}
        }

        mock_requests.get.return_value = mock_403_response

        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Mock fallback to return None (failure)
        with patch.object(wrapper, '_download_public_link') as mock_download:
            mock_download.return_value = ('unknown', None)

            with pytest.raises(ToolException) as exc_info:
                wrapper.read_file_from_sharing_link(
                    "https://other-tenant.sharepoint.com/:x:/specific-people/file"
                )

            assert "The sharing link no longer exists" in str(exc_info.value)


class TestSharingLinkFileValidation:
    """Test file type and size validation for sharing links."""

    def test_rejects_unsupported_file_type_zip(self):
        """ZIP files are rejected with clear error message."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        with pytest.raises(ToolException) as exc_info:
            wrapper._validate_sharing_link_file("archive.zip", 1024)

        assert "not supported" in str(exc_info.value).lower()
        assert ".zip" in str(exc_info.value)

    def test_rejects_unsupported_file_type_mp4(self):
        """MP4 video files are rejected with clear error message."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        with pytest.raises(ToolException) as exc_info:
            wrapper._validate_sharing_link_file("video.mp4", 1024)

        assert "not supported" in str(exc_info.value).lower()
        assert ".mp4" in str(exc_info.value)

    def test_rejects_file_exceeding_size_limit(self):
        """Files exceeding 20 MB are rejected."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # 25 MB file
        file_size = 25 * 1024 * 1024

        with pytest.raises(ToolException) as exc_info:
            wrapper._validate_sharing_link_file("large_file.pdf", file_size)

        assert "too large" in str(exc_info.value).lower()
        assert "25.0 MB" in str(exc_info.value)
        assert "20 MB" in str(exc_info.value)

    def test_accepts_supported_file_type_pdf(self):
        """PDF files are accepted."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Should not raise
        wrapper._validate_sharing_link_file("document.pdf", 5 * 1024 * 1024)

    def test_accepts_supported_file_type_docx(self):
        """DOCX files are accepted."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Should not raise
        wrapper._validate_sharing_link_file("report.docx", 1024 * 1024)

    def test_accepts_supported_file_type_xlsx(self):
        """XLSX files are accepted."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Should not raise
        wrapper._validate_sharing_link_file("data.xlsx", 2 * 1024 * 1024)

    def test_accepts_image_files(self):
        """Image files (PNG, JPG) are accepted."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Should not raise
        wrapper._validate_sharing_link_file("screenshot.png", 500 * 1024)
        wrapper._validate_sharing_link_file("photo.jpg", 1024 * 1024)

    def test_accepts_file_at_size_limit(self):
        """File exactly at 20 MB limit is accepted."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Exactly 20 MB
        file_size = 20 * 1024 * 1024

        # Should not raise
        wrapper._validate_sharing_link_file("large_doc.pdf", file_size)

    def test_handles_none_file_size(self):
        """Validation works when file size is not available."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Size is None - should only check file type
        wrapper._validate_sharing_link_file("document.pdf", None)

    def test_rejects_rar_archive(self):
        """RAR archives are rejected."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        with pytest.raises(ToolException) as exc_info:
            wrapper._validate_sharing_link_file("archive.rar", 1024)

        assert "not supported" in str(exc_info.value).lower()

    def test_rejects_exe_file(self):
        """Executable files are rejected."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        with pytest.raises(ToolException) as exc_info:
            wrapper._validate_sharing_link_file("program.exe", 1024)

        assert "not supported" in str(exc_info.value).lower()

    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.parse_file_content')
    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.requests')
    def test_validates_before_download(self, mock_requests, mock_parse):
        """Validation happens BEFORE downloading file content."""
        # Create a wrapper
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Mock metadata response with unsupported file type (small size to ensure type check triggers)
        mock_metadata_response = MagicMock()
        mock_metadata_response.ok = True
        mock_metadata_response.status_code = 200
        mock_metadata_response.json.return_value = {
            'name': 'video.mp4',
            'size': 5 * 1024 * 1024,  # 5 MB - under size limit, but unsupported type
            '@microsoft.graph.downloadUrl': 'https://download.url/file'
        }

        mock_requests.get.return_value = mock_metadata_response

        # Should raise before trying to download
        with pytest.raises(ToolException) as exc_info:
            wrapper.read_file_from_sharing_link(
                "https://company.sharepoint.com/:v:/s/site/video"
            )

        # Verify error mentions file type
        assert "not supported" in str(exc_info.value).lower()

        # Verify only metadata call was made, not content download
        # First call is metadata, no second call for content
        assert mock_requests.get.call_count == 1

    def test_rejects_file_without_extension(self):
        """Files without extensions are rejected."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        with pytest.raises(ToolException) as exc_info:
            wrapper._validate_sharing_link_file("malware", 1024)

        assert "no extension" in str(exc_info.value).lower()
        assert "malware" in str(exc_info.value)

    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.requests')
    def test_public_link_validates_before_download(self, mock_requests):
        """Public link path validates file via HEAD request BEFORE downloading content."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Mock the initial redirect response (for FedAuth cookie)
        mock_redirect_response = MagicMock()
        mock_redirect_response.status_code = 302
        mock_redirect_response.cookies = [MagicMock(name='FedAuth', value='test-cookie')]
        mock_redirect_response.headers = {
            'Location': 'https://company.sharepoint.com/path?id=/personal/user/large_video.mp4',
            'Set-Cookie': 'FedAuth=test-cookie'
        }

        # Mock the HEAD response with large file size
        mock_head_response = MagicMock()
        mock_head_response.headers = {'Content-Length': str(50 * 1024 * 1024)}  # 50 MB

        # Configure mock to return different responses for different calls
        def get_side_effect(url, **kwargs):
            if kwargs.get('allow_redirects') is False:
                return mock_redirect_response
            return MagicMock()  # Should not reach here

        mock_requests.get.side_effect = get_side_effect
        mock_requests.head.return_value = mock_head_response

        # Should raise ToolException before downloading content
        with pytest.raises(ToolException) as exc_info:
            wrapper._download_public_link("https://company.sharepoint.com/:v:/p/user/video")

        assert "too large" in str(exc_info.value).lower()
        # Verify HEAD was called (validation) but no GET for content
        assert mock_requests.head.call_count == 1
        # Only the initial redirect GET should be called, not content download
        assert mock_requests.get.call_count == 1

    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.requests')
    def test_public_link_rejects_unsupported_type_before_download(self, mock_requests):
        """Public link path rejects unsupported file types via HEAD before download."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Mock the initial redirect response
        mock_redirect_response = MagicMock()
        mock_redirect_response.status_code = 302
        mock_redirect_response.cookies = [MagicMock(name='FedAuth', value='test-cookie')]
        mock_redirect_response.headers = {
            'Location': 'https://company.sharepoint.com/path?id=/personal/user/archive.zip',
            'Set-Cookie': 'FedAuth=test-cookie'
        }

        # Mock the HEAD response with small file size (type check should trigger)
        mock_head_response = MagicMock()
        mock_head_response.headers = {'Content-Length': str(1024)}  # 1 KB

        mock_requests.get.side_effect = lambda url, **kwargs: mock_redirect_response if kwargs.get('allow_redirects') is False else MagicMock()
        mock_requests.head.return_value = mock_head_response

        # Should raise ToolException for unsupported type
        with pytest.raises(ToolException) as exc_info:
            wrapper._download_public_link("https://company.sharepoint.com/:u:/p/user/archive")

        assert "not supported" in str(exc_info.value).lower()
        assert ".zip" in str(exc_info.value)

    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.requests')
    def test_public_link_rejects_oversized_download_despite_head_lie(self, mock_requests):
        """Public link rejects file if actual download exceeds limit (server lied in HEAD)."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Mock the initial redirect response
        mock_redirect_response = MagicMock()
        mock_redirect_response.status_code = 302
        mock_redirect_response.cookies = [MagicMock(name='FedAuth', value='test-cookie')]
        mock_redirect_response.headers = {
            'Location': 'https://company.sharepoint.com/path?id=/personal/user/document.pdf',
            'Set-Cookie': 'FedAuth=test-cookie'
        }

        # Mock HEAD response claiming small file (server lies)
        mock_head_response = MagicMock()
        mock_head_response.headers = {'Content-Length': str(1024)}  # Claims 1 KB

        # Mock GET response with actual large content (25 MB)
        mock_download_response = MagicMock()
        mock_download_response.content = b'x' * (25 * 1024 * 1024)  # Actually 25 MB
        mock_download_response.raise_for_status = MagicMock()

        def get_side_effect(url, **kwargs):
            if kwargs.get('allow_redirects') is False:
                return mock_redirect_response
            return mock_download_response

        mock_requests.get.side_effect = get_side_effect
        mock_requests.head.return_value = mock_head_response

        # Should raise ToolException after download when actual size is checked
        with pytest.raises(ToolException) as exc_info:
            wrapper._download_public_link("https://company.sharepoint.com/:b:/p/user/doc")

        assert "too large" in str(exc_info.value).lower()
        assert "25.0 MB" in str(exc_info.value)

    def test_rejects_double_extension_zip_pdf(self):
        """Files with dangerous intermediate extensions like .zip.pdf are rejected."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        with pytest.raises(ToolException) as exc_info:
            wrapper._validate_sharing_link_file("malware.zip.pdf", 1024)

        assert "double extension" in str(exc_info.value).lower()
        assert ".zip" in str(exc_info.value)

    def test_rejects_double_extension_exe_docx(self):
        """Files with .exe intermediate extension are rejected."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        with pytest.raises(ToolException) as exc_info:
            wrapper._validate_sharing_link_file("virus.exe.docx", 1024)

        assert "double extension" in str(exc_info.value).lower()
        assert ".exe" in str(exc_info.value)

    def test_rejects_double_extension_tar_gz_txt(self):
        """Files with .tar.gz intermediate extensions are rejected."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        with pytest.raises(ToolException) as exc_info:
            wrapper._validate_sharing_link_file("archive.tar.gz.txt", 1024)

        assert "double extension" in str(exc_info.value).lower()

    def test_allows_normal_dotted_filename(self):
        """Files with dots in name but safe extensions are allowed."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Should not raise - "report" and "2024" are not dangerous extensions
        wrapper._validate_sharing_link_file("quarterly.report.2024.pdf", 1024)

    def test_allows_version_numbered_files(self):
        """Files with version numbers like file.v2.pdf are allowed."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Should not raise
        wrapper._validate_sharing_link_file("document.v2.final.pdf", 1024)

    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.parse_file_content')
    @patch('elitea_sdk.tools.sharepoint.graph_wrapper.requests')
    def test_graph_api_rejects_oversized_download_despite_metadata(self, mock_requests, mock_parse):
        """Graph API path rejects file if actual download exceeds limit (metadata was wrong)."""
        wrapper = SharepointGraphWrapper(
            site_url="https://test.sharepoint.com/sites/test",
            token="test-token",
            scopes=["Files.Read"]
        )

        # Mock metadata response claiming small file
        mock_metadata_response = MagicMock()
        mock_metadata_response.ok = True
        mock_metadata_response.status_code = 200
        mock_metadata_response.json.return_value = {
            'name': 'document.pdf',
            'size': 1024,  # Claims 1 KB
            '@microsoft.graph.downloadUrl': 'https://download.url/file'
        }

        # Mock download response with large content
        mock_download_response = MagicMock()
        mock_download_response.content = b'x' * (25 * 1024 * 1024)  # Actually 25 MB
        mock_download_response.raise_for_status = MagicMock()

        mock_requests.get.side_effect = [mock_metadata_response, mock_download_response]

        with pytest.raises(ToolException) as exc_info:
            wrapper.read_file_from_sharing_link(
                "https://company.sharepoint.com/:b:/s/site/doc"
            )

        assert "too large" in str(exc_info.value).lower()
        assert "25.0 MB" in str(exc_info.value)
        # parse_file_content should NOT be called since size check happens first
        mock_parse.assert_not_called()


class TestReadFromSharingLinkSchema:
    """Test Pydantic schema for the tool."""

    def test_schema_has_required_fields(self):
        """Schema includes sharing_url as only required field."""

        schema = ReadFromSharingLink.model_json_schema()

        assert 'sharing_url' in schema['properties']
        assert len(schema['properties']) == 1  # Only sharing_url
        assert 'sharing_url' in schema.get('required', [])

    def test_schema_sharing_url_description(self):
        """sharing_url field has helpful description."""

        schema = ReadFromSharingLink.model_json_schema()
        description = schema['properties']['sharing_url'].get('description', '')

        # Verify description mentions SharePoint (checking schema docs, not URL validation)
        assert 'sharepoint' in description.lower() or 'onedrive' in description.lower()
