"""Tests for get_file_metadata with HTML/XML loaders (PRE-9 #5440).

Covers:
  * EliteAHTMLLoader (.html/.htm) — total_lines + start_line/end_line
  * EliteAXMLLoader (.xml) — same contract
  * Metadata line counts are based on extracted text, not raw source markup
  * No-content degrades gracefully
  * Output conforms to the PRE-1 chunked-read schema (#5432)

Note: Unstructured partition_html/partition_xml is exercised only when the
library is installed. Tests that require actual extraction are skipped when
the import fails, so the test suite stays green in lean CI environments.
"""

import pytest

from elitea_sdk.runtime.langchain.document_loaders.EliteAHTMLLoader import (
    EliteAHTMLLoader, EliteAXMLLoader,
)
from elitea_sdk.tools.utils.file_metadata import get_file_metadata

try:
    from unstructured.partition.html import partition_html  # noqa: F401
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False

SIMPLE_HTML = b"""<!DOCTYPE html>
<html>
<body>
<h1>Title</h1>
<p>Paragraph one.</p>
<p>Paragraph two.</p>
</body>
</html>
"""

SIMPLE_XML = b"""<?xml version="1.0"?>
<root>
  <item>Alpha</item>
  <item>Beta</item>
  <item>Gamma</item>
</root>
"""


# ---------------------------------------------------------------------------
# EliteAHTMLLoader
# ---------------------------------------------------------------------------

class TestHTMLLoaderMetadata:

    def test_basic_contract(self):
        meta = get_file_metadata("page.html", file_content=SIMPLE_HTML,
                                 file_size=len(SIMPLE_HTML))
        assert meta["__result_status__"] == "file_metadata"
        assert meta["extension"] == ".html"
        assert meta["unit"] == "lines"
        assert meta["read_limits"]["max_output_chars"] == 200000
        instr = meta["instruction_for_readFile"]
        assert "start_line" in instr["first_class_params"]
        assert "end_line" in instr["first_class_params"]

    def test_htm_extension(self):
        meta = get_file_metadata("index.htm", file_content=SIMPLE_HTML,
                                 file_size=len(SIMPLE_HTML))
        assert meta["extension"] == ".htm"
        assert meta["unit"] == "lines"

    def test_no_content_degrades(self):
        meta = get_file_metadata("empty.html", file_content=None, file_size=0)
        assert meta["__result_status__"] == "file_metadata"
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 0

    def test_direct_classmethod(self):
        result = EliteAHTMLLoader.get_file_metadata(
            filename="page.html", file_content=SIMPLE_HTML
        )
        assert result["unit"] == "lines"
        assert isinstance(result["total_lines"], int)

    @pytest.mark.skipif(not HAS_UNSTRUCTURED, reason="unstructured not installed")
    def test_line_count_matches_extracted_text(self):
        """total_lines must match the extracted text, not raw source line count."""
        meta = EliteAHTMLLoader.get_file_metadata(
            filename="page.html", file_content=SIMPLE_HTML
        )
        # Raw source has many more lines than extracted plain text
        raw_lines = SIMPLE_HTML.count(b"\n") + 1
        assert meta["total_lines"] < raw_lines, (
            "total_lines should count extracted text lines, not raw HTML source lines"
        )
        assert meta["total_lines"] > 0

    @pytest.mark.skipif(not HAS_UNSTRUCTURED, reason="unstructured not installed")
    def test_range_hint_present(self):
        meta = get_file_metadata("page.html", file_content=SIMPLE_HTML,
                                 file_size=len(SIMPLE_HTML))
        n = meta["total_lines"]
        if n > 0:
            assert f"1..{n}" in meta["instruction_for_readFile"]["first_class_params"]["start_line"]


# ---------------------------------------------------------------------------
# EliteAXMLLoader
# ---------------------------------------------------------------------------

class TestXMLLoaderMetadata:

    def test_basic_contract(self):
        meta = get_file_metadata("data.xml", file_content=SIMPLE_XML,
                                 file_size=len(SIMPLE_XML))
        assert meta["__result_status__"] == "file_metadata"
        assert meta["extension"] == ".xml"
        assert meta["unit"] == "lines"
        assert meta["read_limits"]["max_output_chars"] == 200000
        instr = meta["instruction_for_readFile"]
        assert "start_line" in instr["first_class_params"]
        assert "end_line" in instr["first_class_params"]

    def test_no_content_degrades(self):
        meta = get_file_metadata("empty.xml", file_content=None, file_size=0)
        assert meta["__result_status__"] == "file_metadata"
        assert meta["unit"] == "lines"
        assert meta["total_lines"] == 0

    def test_direct_classmethod(self):
        result = EliteAXMLLoader.get_file_metadata(
            filename="data.xml", file_content=SIMPLE_XML
        )
        assert result["unit"] == "lines"
        assert isinstance(result["total_lines"], int)

    @pytest.mark.skipif(not HAS_UNSTRUCTURED, reason="unstructured not installed")
    def test_line_count_matches_extracted_text(self):
        meta = EliteAXMLLoader.get_file_metadata(
            filename="data.xml", file_content=SIMPLE_XML
        )
        raw_lines = SIMPLE_XML.count(b"\n") + 1
        assert meta["total_lines"] < raw_lines, (
            "total_lines should count extracted text lines, not raw XML source lines"
        )
        assert meta["total_lines"] > 0
