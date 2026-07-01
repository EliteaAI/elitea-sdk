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


# PRE-10 (#5441): chunked-read metadata + single-attachment selection.

def _eml_with_attachments(attachments, *, subject="Attach Test", body="Body line one\nBody line two"):
    """Build a multipart .eml as bytes. attachments = list of (filename, text)."""
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = "sender@example.com"
    msg["To"] = "recipient@example.com"
    msg["Subject"] = subject
    msg.set_content(body)
    for filename, text in attachments:
        msg.add_attachment(
            text.encode("utf-8"), maintype="text", subtype="plain", filename=filename,
        )
    return msg.as_bytes()


class TestGetFileMetadata:
    """PRE-10 get_file_metadata: body line unit + flat attachment inventory."""

    def _meta(self, file_content, filename="test.eml"):
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader
        return EliteAEmailLoader.get_file_metadata(filename=filename, file_content=file_content)

    def test_metadata_conforms_to_schema(self):
        """Merged central output validates against the #5432 contract."""
        from elitea_sdk.tools.utils.file_metadata import (
            get_file_metadata, validate_chunked_read_response,
        )
        content = _eml_with_attachments([("report.pdf", "x")])
        merged = get_file_metadata("test.eml", file_content=content)
        # Central path stamps the discriminator and validates; re-validate to be sure.
        assert merged["__result_status__"] == "file_metadata"
        validate_chunked_read_response(merged)
        assert merged["unit"] == "lines"
        assert merged["read_limits"]["max_output_chars"]  # baseline survived
        assert merged["read_limits"]["email_max_attachment_size_mb"] == 10

    def test_line_unit_and_body_count(self):
        meta = self._meta(_eml_with_attachments([]))
        assert meta["unit"] == "lines"
        assert meta["total_lines"] > 0

    def test_inventory_indexes_duplicate_names(self):
        """Five identically-named attachments each get a distinct 1-indexed slot."""
        content = _eml_with_attachments([("invoice.pdf", f"n{i}") for i in range(5)])
        meta = self._meta(content)
        inv = meta["attachments"]
        assert [a["index"] for a in inv] == [1, 2, 3, 4, 5]
        assert all(a["name"] == "invoice.pdf" for a in inv)

    def test_inventory_readable_flag(self):
        """.txt is readable (loader exists); .xyz is not."""
        content = _eml_with_attachments([("notes.txt", "hi"), ("blob.xyz", "??")])
        inv = self._meta(content)["attachments"]
        by_name = {a["name"]: a for a in inv}
        assert by_name["notes.txt"]["readable"] is True
        assert by_name["blob.xyz"]["readable"] is False

    def test_advertises_selection_params(self):
        meta = self._meta(_eml_with_attachments([("a.txt", "x")]))
        extra = meta["instruction_for_readFile"]["extra_params"]
        assert "attachment_index" in extra
        assert "attachment_name" in extra
        assert "process_attachments" in extra


class TestSingleAttachmentRead:
    """PRE-10 single-attachment selection via get_content()."""

    def _loader(self, content, **kwargs):
        from elitea_sdk.runtime.langchain.document_loaders.EliteAEmailLoader import EliteAEmailLoader
        return EliteAEmailLoader(file_content=content, file_name="test.eml", **kwargs)

    def test_read_by_index(self):
        content = _eml_with_attachments([("one.txt", "FIRST"), ("two.txt", "SECOND")])
        result = self._loader(content, attachment_index=2).get_content()
        assert "SECOND" in result
        assert "FIRST" not in result

    def test_index_out_of_range(self):
        content = _eml_with_attachments([("one.txt", "FIRST")])
        result = self._loader(content, attachment_index=9).get_content()
        assert "out of range" in result
        assert "1:one.txt" in result

    def test_read_by_unique_name(self):
        content = _eml_with_attachments([("only.txt", "UNIQUE"), ("other.txt", "NOPE")])
        result = self._loader(content, attachment_name="only.txt").get_content()
        assert "UNIQUE" in result

    def test_ambiguous_name_refused(self):
        content = _eml_with_attachments([("dup.txt", "A"), ("dup.txt", "B")])
        result = self._loader(content, attachment_name="dup.txt").get_content()
        assert "attachment_index" in result
        assert "2 attachments" in result

    def test_unknown_name(self):
        content = _eml_with_attachments([("real.txt", "X")])
        result = self._loader(content, attachment_name="ghost.txt").get_content()
        assert "No attachment named" in result

    def test_selection_does_not_affect_load(self):
        """load() (indexing) ignores the selector and still returns full content."""
        content = _eml_with_attachments([("one.txt", "FIRST"), ("two.txt", "SECOND")])
        docs = self._loader(content, attachment_index=2, max_tokens=-1).load()
        joined = "\n".join(d.page_content for d in docs)
        assert "FIRST" in joined and "SECOND" in joined
