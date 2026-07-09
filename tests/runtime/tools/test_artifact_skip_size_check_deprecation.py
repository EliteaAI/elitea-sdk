"""
Unit tests for Phase 3 (#5445): skip_size_check deprecated-and-neutered on the
artifact read tools.

Coverage:
  SSC_GUARD    — size guard always applies to LLM/pipeline read_file calls
  SSC_DEPR     — skip_size_check=True is inert but logs a deprecation warning
  SSC_BYPASS   — internal _bypass_size_limit=True still returns full content
                 (grep/edit regression guard)
  SSC_MULTI    — read_multiple_files enforces per file, no bypass forwarding
  SSC_SCHEMA   — skip_size_check absent from both LLM args_schemas

All pure unit tests: no network, no S3. The artifact client is mocked and the
wrapper is built with model_construct to skip the elitea-client validator.
"""

import logging
from unittest.mock import MagicMock

import pytest

from elitea_sdk.runtime.tools.artifact import (
    ArtifactWrapper,
    SKIP_SIZE_CHECK_DEPRECATION_MSG,
)
from elitea_sdk.tools.utils.file_metadata import (
    GET_FILE_METADATA_DIRECTIVE,
    RESULT_STATUS_KEY,
    ResultStatus,
)

MAX = 100  # small cap so tests do not build huge strings


def make_wrapper(content) -> ArtifactWrapper:
    """Build an ArtifactWrapper whose artifact.get() returns `content`."""
    wrapper = ArtifactWrapper.model_construct(
        bucket="test-bucket",
        max_single_read_size=MAX,
        artifact=MagicMock(),
    )
    wrapper.artifact.get = MagicMock(return_value=content)
    wrapper.llm = None
    return wrapper


def assert_over_limit(result, *, actual_chars=None):
    """Shared assertions for a structured content_too_large response."""
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert GET_FILE_METADATA_DIRECTIVE in result["instruction_for_readFile"]["notes"]
    if actual_chars is not None:
        assert result["context"]["actual_chars"] == actual_chars


# ---------------------------------------------------------------------------
# SSC_GUARD — guard always applies for LLM/pipeline calls
# ---------------------------------------------------------------------------

def test_large_content_is_guarded_without_any_flag():
    wrapper = make_wrapper("x" * (MAX + 50))
    result = wrapper.read_file(filename="big.txt")
    assert_over_limit(result, actual_chars=MAX + 50)


def test_small_content_returns_full():
    body = "y" * (MAX - 10)
    wrapper = make_wrapper(body)
    assert wrapper.read_file(filename="small.txt") == body


# ---------------------------------------------------------------------------
# SSC_DEPR — skip_size_check is inert but logs when True
# ---------------------------------------------------------------------------

def test_skip_size_check_true_still_guards_and_warns(caplog):
    wrapper = make_wrapper("z" * (MAX + 50))
    with caplog.at_level(logging.WARNING):
        result = wrapper.read_file(filename="big.txt", skip_size_check=True)
    assert_over_limit(result)  # inert: guard still applied
    assert SKIP_SIZE_CHECK_DEPRECATION_MSG in caplog.text


def test_skip_size_check_false_guards_without_warning(caplog):
    wrapper = make_wrapper("z" * (MAX + 50))
    with caplog.at_level(logging.WARNING):
        result = wrapper.read_file(filename="big.txt", skip_size_check=False)
    assert_over_limit(result)
    assert SKIP_SIZE_CHECK_DEPRECATION_MSG not in caplog.text


def test_default_call_emits_no_deprecation(caplog):
    wrapper = make_wrapper("ok")
    with caplog.at_level(logging.WARNING):
        wrapper.read_file(filename="small.txt")
    assert SKIP_SIZE_CHECK_DEPRECATION_MSG not in caplog.text


# ---------------------------------------------------------------------------
# SSC_BYPASS — internal callers keep full content (grep/edit regression guard)
# ---------------------------------------------------------------------------

def test_bypass_returns_full_content_for_large_file():
    body = "a" * (MAX + 500)
    wrapper = make_wrapper(body)
    # read_file with the private bypass (used by _read_file → grep/edit)
    assert wrapper.read_file(filename="big.txt", _bypass_size_limit=True) == body


def test_internal_read_file_returns_full_content():
    body = "b" * (MAX + 500)
    wrapper = make_wrapper(body)
    assert wrapper._read_file("big.txt") == body


# ---------------------------------------------------------------------------
# SSC_MULTI — read_multiple_files enforces per file, no bypass forwarding
# ---------------------------------------------------------------------------

def test_read_multiple_files_guards_each_file():
    wrapper = make_wrapper("c" * (MAX + 50))
    results = wrapper.read_multiple_files(file_paths=["big.txt"])
    assert_over_limit(results["big.txt"])


def test_read_multiple_files_true_is_inert_and_warns(caplog):
    wrapper = make_wrapper("c" * (MAX + 50))
    with caplog.at_level(logging.WARNING):
        results = wrapper.read_multiple_files(
            file_paths=["big.txt"], skip_size_check=True
        )
    assert_over_limit(results["big.txt"])
    assert SKIP_SIZE_CHECK_DEPRECATION_MSG in caplog.text


def test_read_multiple_files_each_over_cap_file_gets_own_guidance():
    # Two files that each individually exceed the per-file cap get their own
    # structured guidance dicts. The batch-wide cumulative cap (#5780) is
    # measured against each guidance dict's own (small) serialized size, not
    # the oversized file it refused to return, so it isn't tripped here.
    wrapper = ArtifactWrapper.model_construct(
        bucket="test-bucket", max_single_read_size=MAX, artifact=MagicMock(),
    )
    wrapper.llm = None
    bodies = {"a.txt": "a" * (MAX + 20), "b.txt": "b" * (MAX + 999)}
    wrapper.artifact.get = MagicMock(side_effect=lambda artifact_name, **_: bodies[artifact_name])

    results = wrapper.read_multiple_files(file_paths=["a.txt", "b.txt"])

    assert_over_limit(results["a.txt"], actual_chars=MAX + 20)
    assert_over_limit(results["b.txt"], actual_chars=MAX + 999)


# ---------------------------------------------------------------------------
# SSC_SCHEMA — skip_size_check absent from both LLM args_schemas
# ---------------------------------------------------------------------------

def test_read_multiple_files_schema_has_no_skip_size_check():
    wrapper = make_wrapper("ok")
    schema = wrapper._get_file_operation_schemas()["read_multiple_files"]
    assert "skip_size_check" not in schema.model_fields


def test_read_file_schema_has_no_skip_size_check():
    wrapper = make_wrapper("ok")
    tools = wrapper.get_available_tools()
    read_file_tool = next(t for t in tools if t["name"] == "read_file")
    assert "skip_size_check" not in read_file_tool["args_schema"].model_fields


# ---------------------------------------------------------------------------
# Phase 4 (#5446) — unconditional, type-agnostic 200K cap
# ---------------------------------------------------------------------------

def test_dict_content_over_limit_is_caught():
    # The #5404 repro shape: a dict result (e.g. Excel) was never measured by
    # the old str-only check. Now measured via json.dumps().
    big_dict = {"Sheet1": ["row " + str(i) for i in range(50)]}
    wrapper = make_wrapper(big_dict)
    result = wrapper.read_file(filename="big.xlsx")
    assert_over_limit(result)


def test_small_dict_content_returns_full():
    small_dict = {"Sheet1": ["a", "b"]}
    wrapper = make_wrapper(small_dict)
    assert wrapper.read_file(filename="small.xlsx") == small_dict


def test_excel_pre_flight_exception_builds_guidance_from_estimate_no_redownload():
    from elitea_sdk.runtime.langchain.document_loaders.EliteAExcelLoader import (
        ExcelReadEstimate,
        ExcelReadLimitExceeded,
    )

    estimate = ExcelReadEstimate(
        sheets=[{"name": "Sheet1", "max_row": 500000, "max_column": 3}],
        total_rows_workbook=500000,
        target_sheet="Sheet1",
        target_sheet_total_rows=500000,
        requested_start_row=1,
        requested_end_row=500000,
        requested_rows=500000,
        sampled_rows=10,
        sampled_chars=200,
        estimated_output_chars=10_000_000,
        embedded_images=0,
        file_size_bytes=25_000_000,
        is_unbounded_read=True,
        violations=["estimated output size=10000000 exceeds limit 200000"],
    )
    wrapper = make_wrapper(None)
    wrapper.artifact.get = MagicMock(
        side_effect=ExcelReadLimitExceeded("too big", estimate=estimate)
    )

    result = wrapper.read_file(filename="huge.xlsx")

    assert_over_limit(result, actual_chars=10_000_000)
    assert result["unit"] == "rows"
    assert result["total_rows"] == 500000
    assert result["total_sheets"] == 1
    # No re-download/re-parse: artifact.get was called exactly once.
    assert wrapper.artifact.get.call_count == 1


def test_bypass_reraises_excel_exception_instead_of_swallowing():
    from elitea_sdk.runtime.langchain.document_loaders.EliteAExcelLoader import (
        ExcelReadEstimate,
        ExcelReadLimitExceeded,
    )

    estimate = ExcelReadEstimate(
        sheets=[], total_rows_workbook=0, target_sheet=None,
        target_sheet_total_rows=0, requested_start_row=1, requested_end_row=0,
        requested_rows=0, sampled_rows=0, sampled_chars=0,
        estimated_output_chars=0, embedded_images=0, file_size_bytes=None,
        is_unbounded_read=True, violations=["x"],
    )
    wrapper = make_wrapper(None)
    wrapper.artifact.get = MagicMock(
        side_effect=ExcelReadLimitExceeded("too big", estimate=estimate)
    )

    with pytest.raises(ExcelReadLimitExceeded):
        wrapper.read_file(filename="huge.xlsx", _bypass_size_limit=True)


def test_over_limit_does_not_redownload_via_get_metadata():
    # Regression guard: the generic (non-Excel-exception) over-limit path
    # must build guidance from get_file_metadata(file_content=None) — never
    # from self.artifact.get_metadata(download_for_detection=True), which
    # would re-download and, for some loaders, re-render the whole file.
    wrapper = make_wrapper("x" * (MAX + 50))
    wrapper.artifact.get_metadata = MagicMock(
        side_effect=AssertionError("must not call artifact.get_metadata")
    )
    result = wrapper.read_file(filename="big.txt")
    assert_over_limit(result)
    wrapper.artifact.get_metadata.assert_not_called()


def test_max_single_read_size_default_matches_unified_constant():
    from elitea_sdk.tools.utils.file_metadata import DEFAULT_MAX_OUTPUT_CHARS
    assert ArtifactWrapper.model_fields["max_single_read_size"].default == DEFAULT_MAX_OUTPUT_CHARS
