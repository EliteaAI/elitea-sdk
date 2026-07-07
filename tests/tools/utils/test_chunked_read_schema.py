"""Tests for the unified chunked-read schema contract (PRE-1, #5432).

Covers:
  * Base/fallback output carries the discriminator + schema_version and validates.
  * Excel reference output conforms (unit + flat total_<unit> + required read_limits).
  * build_over_limit_response reuses metadata, flips the discriminator, adds
    context, appends the unconditional get_file_metadata directive to notes,
    and stays machine-detectable.
  * A loader returning a non-conformant read_limits fails validation
    (enforcement bites future loaders).
  * Raw file content (str / dict) has NO discriminator — content is
    distinguishable from guidance.
"""

import io

import pytest
from openpyxl import Workbook
from pydantic import ValidationError

from elitea_sdk.tools.utils.file_metadata import (
    DEFAULT_MAX_OUTPUT_CHARS,
    GET_FILE_METADATA_DIRECTIVE,
    RESULT_STATUS_KEY,
    SCHEMA_VERSION,
    ResultStatus,
    build_error_response,
    build_over_limit_response,
    get_file_metadata,
    validate_chunked_read_response,
)


def _make_xlsx_bytes(sheets):
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for r in rows:
            ws.append(r)
    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()


def test_base_output_carries_discriminator_and_validates():
    meta = get_file_metadata("notes.unknownext", file_content=b"hello", file_size=5)

    assert meta[RESULT_STATUS_KEY] == ResultStatus.FILE_METADATA.value
    assert meta["schema_version"] == SCHEMA_VERSION
    # Discriminator is the plain string, not an enum object.
    assert isinstance(meta[RESULT_STATUS_KEY], str)
    # Canonical instruction block is present even for the generic fallback.
    instr = meta["instruction_for_readFile"]
    assert set(instr.keys()) >= {"first_class_params", "extra_params", "notes"}
    # Universal read_limits baseline is ALWAYS present, even with no loader.
    assert meta["read_limits"]["max_output_chars"] == DEFAULT_MAX_OUTPUT_CHARS
    assert "full_read_allowed" in meta["read_limits"]
    # Conforms to the contract.
    validate_chunked_read_response(meta)


def test_excel_reference_conforms_to_schema():
    data = _make_xlsx_bytes({
        "Alpha": [["a", "b"], [1, 2], [3, 4]],
        "Beta": [["x"], [10], [20]],
    })

    meta = get_file_metadata("report.xlsx", file_content=data, file_size=len(data))

    assert meta[RESULT_STATUS_KEY] == ResultStatus.FILE_METADATA.value
    assert meta["unit"] == "rows"
    # Flat total_<unit> keys.
    assert meta["total_rows"] == 6  # 3 rows Alpha + 3 rows Beta
    assert meta["total_sheets"] == 2
    # Required universal read_limits keys.
    assert "max_output_chars" in meta["read_limits"]
    assert "full_read_allowed" in meta["read_limits"]
    # Excel-specific extras are type-prefixed (no other loader assumes them).
    assert "excel_full_read_max_bytes" in meta["read_limits"]
    assert "max_full_read_file_size" not in meta["read_limits"]  # old name gone
    validate_chunked_read_response(meta)


def test_over_limit_reuses_metadata_and_flips_only_discriminator():
    data = _make_xlsx_bytes({"Alpha": [["a"], [1], [2]]})
    meta = get_file_metadata("report.xlsx", file_content=data, file_size=len(data))

    over = build_over_limit_response(
        meta, actual_chars=870123, limit_chars=200000, requested="full read"
    )

    # Discriminator changed (+ added context + pinned full_read_allowed). The
    # instruction block is otherwise reused verbatim except that "notes" gains
    # the unconditional get_file_metadata directive (Phase 4, #5446).
    assert over[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert over["unit"] == meta["unit"]
    assert over["total_rows"] == meta["total_rows"]
    assert over["instruction_for_readFile"]["extra_params"] == meta["instruction_for_readFile"]["extra_params"]
    assert GET_FILE_METADATA_DIRECTIVE in over["instruction_for_readFile"]["notes"]
    assert GET_FILE_METADATA_DIRECTIVE not in meta["instruction_for_readFile"]["notes"]
    # max_output_chars survives; full_read_allowed is pinned False on over-limit.
    assert over["read_limits"]["max_output_chars"] == meta["read_limits"]["max_output_chars"]
    assert over["read_limits"]["full_read_allowed"] is False
    # context describes the over-limit condition.
    assert over["context"] == {
        "limit_chars": 200000,
        "actual_chars": 870123,
        "requested": "full read",
    }
    validate_chunked_read_response(over)


def test_over_limit_notes_carry_directive_even_with_empty_notes():
    # A metadata dict with no notes of its own still gets the directive —
    # the append is unconditional, not "only if there's already text".
    bare_meta = {
        RESULT_STATUS_KEY: ResultStatus.FILE_METADATA.value,
        "read_limits": {"max_output_chars": DEFAULT_MAX_OUTPUT_CHARS, "full_read_allowed": True},
        "instruction_for_readFile": {"first_class_params": {}, "extra_params": {}, "notes": ""},
    }
    over = build_over_limit_response(bare_meta, actual_chars=999, limit_chars=200000)
    assert over["instruction_for_readFile"]["notes"] == GET_FILE_METADATA_DIRECTIVE


def test_docx_metadata_carries_read_limits_baseline():
    # Docx loader supplies no read_limits of its own; the central baseline must
    # still make max_output_chars present (the documented-required field).
    meta = get_file_metadata("doc.docx", file_content=None, file_size=1234)
    assert meta[RESULT_STATUS_KEY] == ResultStatus.FILE_METADATA.value
    assert meta["read_limits"]["max_output_chars"] == DEFAULT_MAX_OUTPUT_CHARS
    validate_chunked_read_response(meta)


def test_guidance_without_read_limits_fails_validation():
    # A file_metadata/content_too_large object MUST carry read_limits.
    for status in (ResultStatus.FILE_METADATA, ResultStatus.CONTENT_TOO_LARGE):
        with pytest.raises(ValidationError):
            validate_chunked_read_response({
                RESULT_STATUS_KEY: status.value,
                "filename": "x.dat",
            })


def test_error_response_carries_schema_version_and_validates():
    # Major-2: error emissions route through the model (schema_version + valid),
    # and an error response may omit read_limits.
    err = build_error_response("boom", filename="x.dat", filepath="/b/x.dat")
    assert err[RESULT_STATUS_KEY] == ResultStatus.ERROR.value
    assert err["schema_version"] == SCHEMA_VERSION
    assert err["message"] == "boom"
    assert err["filepath"] == "/b/x.dat"  # extra context rides through
    assert "read_limits" not in err
    validate_chunked_read_response(err)


def test_non_conformant_read_limits_fails_validation():
    # Missing the required full_read_allowed key.
    bad = {
        RESULT_STATUS_KEY: ResultStatus.FILE_METADATA.value,
        "filename": "x.dat",
        "read_limits": {"max_output_chars": 200000},
    }
    with pytest.raises(ValidationError):
        validate_chunked_read_response(bad)


def test_raw_content_has_no_discriminator():
    # A normal text read returns a str; an Excel row-range read returns a dict.
    # Neither carries __result_status__, so a node can tell content from guidance.
    text_content = "line1\nline2\n"
    assert not (isinstance(text_content, dict))

    excel_slice = {
        "sheet_name": "Alpha",
        "start_row": 1,
        "end_row": 100,
        "total_rows": 4800,
        "content": "a | b\n1 | 2\n",
    }
    assert RESULT_STATUS_KEY not in excel_slice
