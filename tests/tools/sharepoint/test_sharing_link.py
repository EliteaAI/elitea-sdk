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

        assert 'sharepoint.com' in description.lower()
