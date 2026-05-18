"""Integration test for extra_params plumbing through parse_file_content (EL-4389).

Verifies that when extra_params={'start_row': N, 'end_row': M, 'sheet_name': X}
are passed to parse_file_content for an Excel file, the EliteAExcelLoader
returns the row-range dict produced by read_excel_rows (instead of the
heavy chunking pipeline output).
"""

import io

import pytest
from openpyxl import Workbook

from elitea_sdk.tools.utils.content_parser import parse_file_content


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


def test_parse_file_content_row_range_excel():
    data = _make_xlsx_bytes({
        "Data": [["col_a", "col_b"], [1, 2], [3, 4], [5, 6], [7, 8]],
    })

    result = parse_file_content(
        file_name="big.xlsx",
        file_content=data,
        extra_params={
            "sheet_name": "Data",
            "start_row": 2,
            "end_row": 3,
            "include_headers": True,
            "header_row": 1,
        },
    )

    # Loader returns a dict in row-range mode.
    assert isinstance(result, dict), f"Expected dict, got {type(result)}: {result!r}"
    assert result["sheet_name"] == "Data"
    assert result["start_row"] == 2
    assert result["end_row"] == 3
    # Header should be prepended.
    assert "col_a" in result["content"]
    # Only rows 2 and 3 should be in content (values 1|2 and 3|4).
    assert "1 | 2" in result["content"]
    assert "3 | 4" in result["content"]
    # Row 4 must NOT be in content.
    assert "5 | 6" not in result["content"]


def test_parse_file_content_without_extra_params_uses_full_chunking():
    data = _make_xlsx_bytes({
        "Data": [["a", "b"], [1, 2], [3, 4]],
    })
    result = parse_file_content(file_name="small.xlsx", file_content=data)
    # Full pipeline returns dict {sheet_name: list_of_chunks}
    assert isinstance(result, dict)
    assert "Data" in result
    # Without row-range mode, value should be a list, not a row-range dict.
    assert isinstance(result["Data"], list)
