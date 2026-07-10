"""Tests for streaming Excel helpers added for EL-4389.

Covers:
  * list_excel_sheets returns sheet names + dimensions
  * read_excel_rows reads a row range
  * header row is prepended when include_headers is True
  * out-of-range and unknown sheet handling
  * round-trip via bytes
"""

import io
import os
import zipfile

import pytest
from openpyxl import Workbook

from elitea_sdk.runtime.langchain.document_loaders.EliteAExcelLoader import (
    EXCEL_READ_LIMIT_ERROR,
    EXCEL_MAX_FULL_READ_FILE_SIZE,
    EXCEL_MAX_REQUEST_ROWS,
    EliteAExcelLoader,
    ExcelReadLimitExceeded,
    check_excel_read_limits,
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


def _pad_file_without_corrupting_zip(path, min_size_bytes):
    """Increase file size while keeping the zip central directory at EOF."""
    current_size = os.path.getsize(path)
    if current_size >= min_size_bytes:
        return

    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("xl/padding.bin", b"0" * (min_size_bytes - current_size))


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


def test_check_excel_read_limits_allows_workbook_row_count_when_scope_is_safe(tmp_path):
    path = _build_workbook(tmp_path, {
        "S": [["h1"]] + [[idx] for idx in range(100)],
    })

    estimate = check_excel_read_limits(path, raise_on_violation=True)

    assert estimate.requested_rows == 101
    assert estimate.estimated_output_chars < 200000


def test_check_excel_read_limits_rejects_large_text_budget(tmp_path):
    large_value = "x" * 32_000
    path = _build_workbook(tmp_path, {
        "S": [["h1"]] + [[large_value] for _ in range(8)],
    })

    with pytest.raises(ValueError) as exc:
        check_excel_read_limits(path, raise_on_violation=True)

    assert "output size=" in str(exc.value)
    # ExcelReadLimitExceeded (Phase 4, #5446) IS a ValueError, and carries the
    # already-computed estimate so a caller can build structured guidance
    # without re-sampling the workbook.
    assert isinstance(exc.value, ExcelReadLimitExceeded)
    assert exc.value.estimate.target_sheet == "S"
    assert exc.value.estimate.violations


def test_check_excel_read_limits_counts_embedded_images(tmp_path):
    path = _build_workbook(tmp_path, {"S": [["h1"], [1]]})
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("xl/media/image1.png", b"fake")
        archive.writestr("xl/media/image2.jpeg", b"fake")

    estimate = check_excel_read_limits(path)

    assert estimate.embedded_images == 2


def test_check_excel_read_limits_rejects_too_many_images(tmp_path):
    path = _build_workbook(tmp_path, {"S": [["h1"], [1]]})
    with zipfile.ZipFile(path, "a") as archive:
        for idx in range(33):
            archive.writestr(f"xl/media/image{idx}.png", b"fake")

    with pytest.raises(ValueError) as exc:
        check_excel_read_limits(path, raise_on_violation=True)

    assert "embedded images=" in str(exc.value)
    assert isinstance(exc.value, ExcelReadLimitExceeded)
    assert exc.value.estimate.embedded_images == 33


def test_check_excel_read_limits_partial_scope_uses_requested_range(tmp_path):
    path = _build_workbook(tmp_path, {
        "S": [["h1"]] + [["x" * 50_000] for _ in range(3)] + [["ok"] for _ in range(50)],
    })

    estimate = check_excel_read_limits(path, sheet_name="S", start_row=5, end_row=10)

    assert estimate.requested_rows == 6
    assert estimate.estimated_output_chars < 200000


def test_check_excel_read_limits_rejects_large_full_read_by_file_size(tmp_path):
    path = _build_workbook(tmp_path, {"S": [["h1"], [1]]})
    _pad_file_without_corrupting_zip(path, EXCEL_MAX_FULL_READ_FILE_SIZE + 1)

    with pytest.raises(ValueError) as exc:
        check_excel_read_limits(path, raise_on_violation=True)

    assert "file size=" in str(exc.value)
    assert str(EXCEL_MAX_FULL_READ_FILE_SIZE) in str(exc.value)
    assert isinstance(exc.value, ExcelReadLimitExceeded)
    assert exc.value.estimate.file_size_bytes > EXCEL_MAX_FULL_READ_FILE_SIZE


def test_check_excel_read_limits_allows_large_file_partial_read_when_scope_is_safe(tmp_path):
    path = _build_workbook(tmp_path, {"S": [["h1"], ["ok"], ["ok2"]]})
    _pad_file_without_corrupting_zip(path, EXCEL_MAX_FULL_READ_FILE_SIZE + 1)

    estimate = check_excel_read_limits(
        path,
        sheet_name="S",
        start_row=2,
        end_row=2,
        raise_on_violation=True,
    )

    assert estimate.requested_rows == 1
    assert estimate.file_size_bytes > EXCEL_MAX_FULL_READ_FILE_SIZE


# --- Multi-sheet workbook guard (#5713) ---------------------------------------
# A full read materializes every sheet. Each sheet may individually pass the
# per-sheet thresholds while the workbook total blows past the output cap.


def test_full_workbook_read_sums_text_budget_across_all_sheets(tmp_path):
    # 20 sheets x ~15k chars each = ~300k >> 200k cap, though no single sheet trips it.
    big_value = "x" * 3_000
    sheets = {
        f"Sheet{idx}": [["h1"]] + [[big_value] for _ in range(5)]
        for idx in range(20)
    }
    path = _build_workbook(tmp_path, sheets)

    with pytest.raises(ExcelReadLimitExceeded) as exc:
        check_excel_read_limits(path, raise_on_violation=True)

    assert "output size=" in str(exc.value)
    # The rejected estimate reflects the whole workbook, not just sheet 0.
    assert exc.value.estimate.estimated_output_chars > 200_000


def test_full_workbook_read_sums_rows_across_all_sheets(tmp_path):
    # No single sheet exceeds EXCEL_MAX_REQUEST_ROWS, but the sum does.
    per_sheet = (EXCEL_MAX_REQUEST_ROWS // 2) + 100
    sheets = {
        f"Sheet{idx}": [[idx] for _ in range(per_sheet)]
        for idx in range(3)
    }
    path = _build_workbook(tmp_path, sheets)

    with pytest.raises(ExcelReadLimitExceeded) as exc:
        check_excel_read_limits(path, raise_on_violation=True)

    assert "requested rows=" in str(exc.value)
    assert exc.value.estimate.requested_rows == per_sheet * 3


def test_full_workbook_read_allows_small_multi_sheet_workbook(tmp_path):
    path = _build_workbook(tmp_path, {
        "A": [["h1", "h2"], [1, 2], [3, 4]],
        "B": [["x"], [10], [20]],
    })

    estimate = check_excel_read_limits(path, raise_on_violation=True)

    assert estimate.is_unbounded_read
    assert estimate.requested_rows == 6
    assert estimate.estimated_output_chars < 200_000


def test_named_sheet_read_scopes_to_that_sheet_only(tmp_path):
    # A tiny target sheet must pass even when a sibling sheet is enormous.
    big_value = "x" * 5_000
    path = _build_workbook(tmp_path, {
        "Small": [["h1"], ["ok"]],
        "Huge": [["h1"]] + [[big_value] for _ in range(200)],
    })

    estimate = check_excel_read_limits(
        path, sheet_name="Small", raise_on_violation=True)

    assert estimate.target_sheet == "Small"
    assert estimate.estimated_output_chars < 200_000


def test_get_content_full_read_rejects_oversized_multi_sheet_workbook(tmp_path):
    big_value = "x" * 3_000
    sheets = {
        f"Sheet{idx}": [["h1"]] + [[big_value] for _ in range(5)]
        for idx in range(20)
    }
    path = _build_workbook(tmp_path, sheets)

    loader = EliteAExcelLoader(file_path=path, file_name=path)

    with pytest.raises(ExcelReadLimitExceeded):
        loader.get_content()
