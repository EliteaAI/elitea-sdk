"""
Pytest tests for EliteAEmailLoader.

Test cases are auto-discovered from:
  tests/runtime/langchain/document_loaders/test_data/EliteAEmailLoader/input/*.json

Each (input_name, config_index) pair becomes a standalone test.
Tags declared in the input JSON are applied as pytest marks for -m filtering.

Run:
  pytest tests/runtime/langchain/document_loaders/test_elitea_email_loader.py -v
  pytest tests/runtime/langchain/document_loaders/test_elitea_email_loader.py -v -k "email_simple"
  pytest -m "loader_email" -v
  pytest -m "loader_email and edge_encoding" -v
"""

from pathlib import Path
from typing import Any, Dict

import pytest
from loader_helpers import collect_loader_test_params, run_loader_assert

_LOADER_NAME = "EliteAEmailLoader"


@pytest.mark.parametrize(
    "input_name, config_index, config, file_path, baseline_path",
    collect_loader_test_params(_LOADER_NAME),
)
def test_loader(
    tmp_path: Path,
    input_name: str,
    config_index: int,
    config: Dict[str, Any],
    file_path: Path,
    baseline_path: Path,
) -> None:
    run_loader_assert(_LOADER_NAME, tmp_path, input_name, config_index, config, file_path, baseline_path)


# Unit tests for new features added in PR #123

class TestRecursionGuard:
    """Test recursion guard for nested attachments."""

    def test_depth_limit_default(self):
        """Test default depth limit is 2."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader

        loader = EliteAEmailLoader(file_path='/tmp/test.eml')
        assert loader.max_attachment_depth == 2

    def test_depth_limit_custom(self):
        """Test custom depth limit."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader

        loader = EliteAEmailLoader(file_path='/tmp/test.eml', max_attachment_depth=5)
        assert loader.max_attachment_depth == 5

    def test_size_limit_default(self):
        """Test default size limit is 10MB."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader

        loader = EliteAEmailLoader(file_path='/tmp/test.eml')
        assert loader.max_attachment_size_mb == 10

    def test_size_limit_custom(self):
        """Test custom size limit."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader

        loader = EliteAEmailLoader(file_path='/tmp/test.eml', max_attachment_size_mb=50)
        assert loader.max_attachment_size_mb == 50

    def test_depth_limit_enforced(self, tmp_path):
        """Test that depth limit prevents deep recursion."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader

        # Create a simple email
        email_file = tmp_path / "test.eml"
        email_file.write_text("""From: test@example.com
To: user@example.com
Subject: Test
Date: Mon, 1 Jan 2024 10:00:00 +0000

Test email
""")

        loader = EliteAEmailLoader(file_path=str(email_file), max_attachment_depth=1)

        # Test _parse_attachment_content with depth at limit
        result = loader._parse_attachment_content("test.txt", b"content", current_depth=1)
        assert result == ""  # Should return empty at depth limit

    def test_size_limit_enforced(self, tmp_path):
        """Test that size limit prevents large attachments."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader

        email_file = tmp_path / "test.eml"
        email_file.write_text("""From: test@example.com
To: user@example.com
Subject: Test

Test
""")

        loader = EliteAEmailLoader(file_path=str(email_file), max_attachment_size_mb=1)

        # Create 2MB of content
        large_content = b"x" * (2 * 1024 * 1024)
        result = loader._parse_attachment_content("large.txt", large_content)
        assert result == ""  # Should return empty for oversized attachment

    def test_dynamic_class_check(self, tmp_path):
        """Test that class checking works dynamically (not hardcoded)."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader
        from elitea_sdk.runtime.langchain.document_loaders.constants import loaders_map

        email_file = tmp_path / "test.eml"
        email_file.write_text("""From: test@example.com
To: user@example.com
Subject: Test

Test
""")

        # Test with base class
        loader = EliteAEmailLoader(file_path=str(email_file))
        email_loader_cls = loaders_map['.eml']['class']

        # Verify the check works dynamically
        is_email_loader = isinstance(email_loader_cls, type) and issubclass(email_loader_cls, loader.__class__)
        assert is_email_loader, "Dynamic class check should work with base class"

        # Test with subclass (simulated)
        class CustomEmailLoader(EliteAEmailLoader):
            pass

        custom_loader = CustomEmailLoader(file_path=str(email_file))
        # The base EliteAEmailLoader class should still be recognized
        is_still_email_loader = isinstance(email_loader_cls, type) and issubclass(email_loader_cls, custom_loader.__class__)
        # This will be False because EliteAEmailLoader is not a subclass of CustomEmailLoader
        # But CustomEmailLoader IS a subclass of EliteAEmailLoader
        is_custom_subclass = isinstance(custom_loader.__class__, type) and issubclass(custom_loader.__class__, EliteAEmailLoader)
        assert is_custom_subclass, "Subclass should be recognized as email loader"


class TestGetContent:
    """Test get_content() method for ADO integration."""

    def test_get_content_exists(self):
        """Test that get_content method exists."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader

        loader = EliteAEmailLoader(file_path='/tmp/test.eml')
        assert hasattr(loader, 'get_content')

    def test_get_content_returns_string(self, tmp_path):
        """Test that get_content returns a string."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader

        email_file = tmp_path / "test.eml"
        email_file.write_text("""From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Mon, 1 Jan 2024 10:00:00 +0000

This is a test.
""")

        loader = EliteAEmailLoader(file_path=str(email_file))
        content = loader.get_content()

        assert isinstance(content, str)
        assert len(content) > 0
        assert "Test Email" in content
        assert "sender@example.com" in content

    def test_get_content_with_file_content(self):
        """Test get_content with file_content parameter."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader

        email_bytes = b"""From: test@example.com
To: user@example.com
Subject: Content Test

Test body
"""

        loader = EliteAEmailLoader(file_content=email_bytes, file_name='test.eml')
        content = loader.get_content()

        assert isinstance(content, str)
        assert "Content Test" in content

    def test_get_content_matches_page_content(self, tmp_path):
        """Test that get_content matches document page_content (without chunking)."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader

        email_file = tmp_path / "test.eml"
        email_file.write_text("""From: sender@example.com
To: recipient@example.com
Subject: Match Test

Body content
""")

        # Disable chunking with max_tokens=-1
        loader = EliteAEmailLoader(file_path=str(email_file), max_tokens=-1)

        # Get content via get_content()
        get_content_result = loader.get_content()

        # Get content via _load_raw() (no chunking)
        docs = loader._load_raw()
        page_content_result = docs[0].page_content if docs else ""

        # They should be the same
        assert get_content_result == page_content_result


class TestNullByteSanitization:
    """Test null byte sanitization in filenames and metadata."""

    def test_sanitize_text_removes_null_bytes(self):
        """Test that _sanitize_text removes null bytes."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import _sanitize_text

        text_with_null = "test\x00data"
        result = _sanitize_text(text_with_null)
        assert "\x00" not in result
        assert result == "testdata"

    def test_extension_lookup_with_null_bytes(self, tmp_path):
        """Test that extension lookup works with null bytes in filename."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader

        email_file = tmp_path / "test.eml"
        email_file.write_text("""From: test@example.com
To: user@example.com
Subject: Test

Test
""")

        loader = EliteAEmailLoader(file_path=str(email_file))

        # Simulate filename with null byte (as comes from MSG files)
        filename_with_null = "document.json\x00"
        content = b'{"test": "data"}'

        # Should not crash and should find the .json loader
        result = loader._parse_attachment_content(filename_with_null, content)
        # Result might be empty or contain parsed JSON, but shouldn't crash
        assert isinstance(result, str)

    def test_temp_file_creation_with_null_bytes(self, tmp_path):
        """Test that temp files can be created with sanitized names."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader
        import os

        email_file = tmp_path / "test.eml"
        email_file.write_text("""From: test@example.com
To: user@example.com
Subject: Test

Test
""")

        loader = EliteAEmailLoader(file_path=str(email_file))

        # Filename with null byte should be sanitized before file creation
        filename_with_null = "file\x00name.txt"
        content = b"test content"

        # This should not raise "embedded null byte" error
        try:
            result = loader._parse_attachment_content(filename_with_null, content)
            assert isinstance(result, str)
            success = True
        except ValueError as e:
            if "embedded null byte" in str(e):
                success = False
            else:
                raise

        assert success, "Should handle null bytes in filename without crashing"


class TestMSGHeaderExtraction:
    """Test MSG-specific header extraction."""

    def test_extract_headers_from_msg_exists(self):
        """Test that _extract_headers_from_msg method exists."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader

        loader = EliteAEmailLoader(file_path='/tmp/test.msg')
        assert hasattr(loader, '_extract_headers_from_msg')

    def test_msg_and_eml_use_different_extractors(self):
        """Test that MSG and EML use different header extraction methods."""
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader

        # This is a behavioral test - we can't easily test with real files
        # but we can verify the methods exist and are different
        loader = EliteAEmailLoader(file_path='/tmp/test.msg')

        eml_method = loader._extract_headers_from_eml
        msg_method = loader._extract_headers_from_msg

        # They should be different methods
        assert eml_method != msg_method
        assert callable(eml_method)
        assert callable(msg_method)
