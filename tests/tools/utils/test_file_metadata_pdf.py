"""Tests for file_metadata registry with PDF files (PRE-2 #5433).

Covers:
  * PDF loader advertises total_pages + page_number in first_class_params
  * Output conforms to the PRE-1 chunked-read schema (#5432)
  * total_pages reflects the real page count
  * No-content / unreadable PDF degrades gracefully to total_pages == 0
  * A single-page read via the loader stays bounded to that page
"""

import pymupdf
import pytest

from elitea_sdk.runtime.langchain.document_loaders.EliteAPDFLoader import (
    EliteAPDFLoader,
)
from elitea_sdk.tools.utils.file_metadata import get_file_metadata


def _make_pdf_bytes(num_pages: int = 3) -> bytes:
    doc = pymupdf.open()
    try:
        for i in range(num_pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i + 1} body text")
        return doc.tobytes()
    finally:
        doc.close()


def test_pdf_metadata_reports_total_pages_and_instruction():
    data = _make_pdf_bytes(3)

    meta = get_file_metadata("report.pdf", file_content=data, file_size=len(data))

    # Schema-level fields (PRE-1 contract)
    assert meta["__result_status__"] == "file_metadata"
    assert meta["filename"] == "report.pdf"
    assert meta["extension"] == ".pdf"
    assert meta["filesize"] == len(data)
    assert meta["read_limits"]["max_output_chars"] == 200000

    # PDF-specific
    assert meta["unit"] == "pages"
    assert meta["total_pages"] == 3

    instr = meta["instruction_for_readFile"]
    assert "page_number" in instr["first_class_params"]
    assert "1-indexed" in instr["first_class_params"]["page_number"]
    assert "1..3" in instr["first_class_params"]["page_number"]
    assert "page_number" in instr["notes"]


def test_pdf_metadata_no_content():
    meta = get_file_metadata("doc.pdf", file_content=None, file_size=8192)

    assert meta["unit"] == "pages"
    assert meta["total_pages"] == 0
    assert meta["filesize"] == 8192
    # Instruction still advertised even without page count.
    assert "page_number" in meta["instruction_for_readFile"]["first_class_params"]


def test_pdf_metadata_unreadable_content_degrades():
    meta = get_file_metadata("broken.pdf", file_content=b"not a real pdf",
                             file_size=14)

    assert meta["total_pages"] == 0
    assert meta["unit"] == "pages"


def test_pdf_single_page_read_is_bounded():
    data = _make_pdf_bytes(3)

    loader = EliteAPDFLoader(file_content=data, page_number=2)
    content = loader.get_content()

    # Only the requested page is returned, not the whole document.
    assert "Page: 2" in content
    assert "Page: 1" not in content
    assert "Page: 3" not in content
