"""Tests for the file_metadata registry (EL-4389).

Covers:
  * Excel handler advertises sheets + instruction_for_readFile
  * Generic fallback for unknown types returns base structure
  * extension-based mime detection when no content provided
  * filesize is reported when supplied
"""

import io

import pytest
from openpyxl import Workbook

from elitea_sdk.tools.utils.file_metadata import get_file_metadata


def _make_xlsx_bytes(sheets):
    wb = Workbook()
    default = wb.active
    wb.remove(default)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for r in rows:
            ws.append(r)
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


def test_excel_metadata_returns_sheets_and_instruction():
    data = _make_xlsx_bytes({
        "Alpha": [["a", "b"], [1, 2], [3, 4]],
        "Beta": [["x"], [10], [20]],
    })

    meta = get_file_metadata("report.xlsx", file_content=data, file_size=len(data))

    assert meta["filename"] == "report.xlsx"
    assert meta["extension"] == ".xlsx"
    assert meta["filesize"] == len(data)
    assert "sheets" in meta
    names = [s["name"] for s in meta["sheets"]]
    assert names == ["Alpha", "Beta"]
    instr = meta["instruction_for_readFile"]
    assert set(instr["extra_params"].keys()) >= {
        "sheet_name", "start_row", "end_row", "include_headers", "header_row"
    }
    assert "JSON string" in instr["notes"]


def test_excel_metadata_handles_xls_extension_without_content():
    # No content - handler must not crash, sheets list may be empty.
    meta = get_file_metadata("legacy.xls", file_content=None, file_size=4096)

    assert meta["extension"] == ".xls"
    assert meta["filesize"] == 4096
    assert meta["sheets"] == []
    assert "start_row" in meta["instruction_for_readFile"]["extra_params"]


def test_generic_fallback_for_unknown_extension():
    meta = get_file_metadata("notes.txt", file_content=b"hello", file_size=5)

    assert meta["filename"] == "notes.txt"
    assert meta["extension"] == ".txt"
    assert meta["filesize"] == 5
    # Generic - no per-type extra_params hints.
    assert meta["instruction_for_readFile"]["extra_params"] == {}
    # mime falls back to mimetypes for text/plain.
    assert meta["type"].startswith("text/")


def test_metadata_uses_provided_filesize_over_content_length():
    # Pretend the file is 1 MB even though we only sent a small sample.
    meta = get_file_metadata("notes.txt", file_content=b"hi", file_size=10_000_000)
    assert meta["filesize"] == 10_000_000
