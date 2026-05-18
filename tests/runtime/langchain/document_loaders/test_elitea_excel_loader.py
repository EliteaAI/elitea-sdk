"""Tests for streaming Excel helpers added for EL-4389.

Covers:
  * list_excel_sheets returns sheet names + dimensions
  * read_excel_rows reads a row range
  * header row is prepended when include_headers is True
  * out-of-range and unknown sheet handling
  * round-trip via bytes
"""

import io

import pytest
from openpyxl import Workbook

from elitea_sdk.runtime.langchain.document_loaders.EliteAExcelLoader import (
    list_excel_sheets,
    read_excel_rows,
)


def _build_workbook(tmp_path, sheets):
    """Create an .xlsx file with the given {sheet_name: rows} mapping."""
    wb = Workbook()
    # Remove default sheet
    default = wb.active
    wb.remove(default)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for r in rows:
            ws.append(r)
    path = tmp_path / "wb.xlsx"
    wb.save(path)
    return str(path)


def test_list_excel_sheets_returns_all_sheets(tmp_path):
    path = _build_workbook(tmp_path, {
        "Alpha": [["h1", "h2"], [1, 2], [3, 4]],
        "Beta": [["x"], [10], [20], [30]],
    })

    sheets = list_excel_sheets(path)
    names = [s["name"] for s in sheets]
    assert names == ["Alpha", "Beta"]
    alpha = next(s for s in sheets if s["name"] == "Alpha")
    beta = next(s for s in sheets if s["name"] == "Beta")
    assert alpha["max_row"] == 3
    assert alpha["max_column"] == 2
    assert beta["max_row"] == 4
    assert beta["max_column"] == 1


def test_list_excel_sheets_accepts_bytes(tmp_path):
    path = _build_workbook(tmp_path, {"S": [["a", "b"], [1, 2]]})
    with open(path, "rb") as f:
        data = f.read()
    sheets = list_excel_sheets(data, file_name="anything.xlsx")
    assert sheets[0]["name"] == "S"
    assert sheets[0]["max_row"] == 2


def test_read_excel_rows_basic_range_with_header(tmp_path):
    path = _build_workbook(tmp_path, {
        "Data": [
            ["name", "value"],
            ["a", 1],
            ["b", 2],
            ["c", 3],
            ["d", 4],
        ],
    })

    result = read_excel_rows(path, sheet_name="Data", start_row=3, end_row=4)

    assert result["sheet_name"] == "Data"
    assert result["start_row"] == 3
    assert result["end_row"] == 4
    assert result["total_rows"] == 5
    lines = result["content"].splitlines()
    # First line is the header (row 1), then rows 3 and 4 from the body
    assert lines[0] == "name | value"
    assert lines[1] == "b | 2"
    assert lines[2] == "c | 3"


def test_read_excel_rows_without_header(tmp_path):
    path = _build_workbook(tmp_path, {
        "S": [["h1"], ["x"], ["y"], ["z"]],
    })
    result = read_excel_rows(path, sheet_name="S", start_row=2, end_row=3,
                             include_headers=False)
    lines = result["content"].splitlines()
    assert lines == ["x", "y"]
    assert result["end_row"] == 3


def test_read_excel_rows_default_sheet_is_first(tmp_path):
    path = _build_workbook(tmp_path, {
        "First": [["a"], [1]],
        "Second": [["b"], [2]],
    })
    result = read_excel_rows(path, start_row=1, end_row=2)
    assert result["sheet_name"] == "First"


def test_read_excel_rows_end_row_clipped_to_total(tmp_path):
    path = _build_workbook(tmp_path, {"S": [["h"], [1], [2]]})
    result = read_excel_rows(path, sheet_name="S", start_row=1, end_row=999)
    assert result["end_row"] == 3
    assert result["total_rows"] == 3


def test_read_excel_rows_unknown_sheet_raises(tmp_path):
    path = _build_workbook(tmp_path, {"S": [["h"], [1]]})
    with pytest.raises(ValueError) as exc:
        read_excel_rows(path, sheet_name="Nope", start_row=1, end_row=1)
    assert "Nope" in str(exc.value)


def test_read_excel_rows_invalid_range_raises(tmp_path):
    path = _build_workbook(tmp_path, {"S": [["h"], [1]]})
    with pytest.raises(ValueError):
        read_excel_rows(path, sheet_name="S", start_row=5, end_row=2)


def test_read_excel_rows_via_bytes(tmp_path):
    path = _build_workbook(tmp_path, {"S": [["h1", "h2"], [1, 2], [3, 4]]})
    with open(path, "rb") as f:
        data = f.read()
    result = read_excel_rows(data, sheet_name="S", start_row=2, end_row=3,
                             file_name="src.xlsx")
    lines = result["content"].splitlines()
    assert lines[0] == "h1 | h2"
    assert lines[1] == "1 | 2"
    assert lines[2] == "3 | 4"


def test_read_excel_rows_header_row_in_range_no_duplication(tmp_path):
    path = _build_workbook(tmp_path, {
        "S": [["h1", "h2"], ["a", "b"], ["c", "d"]],
    })
    # When the requested range starts at the header row, it should not be
    # duplicated at the top of the output.
    result = read_excel_rows(path, sheet_name="S", start_row=1, end_row=2)
    lines = result["content"].splitlines()
    assert lines == ["h1 | h2", "a | b"]
