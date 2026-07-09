"""Tests for SharePoint read_document read cap + structured guidance (Phase 5, #5451).

Assertions mirror tests/tools/bitbucket/test_bitbucket_read_file_cap.py and the
GitLab equivalent:
  * Small files return plain strings unchanged.
  * Over-cap text files return the PRE-1 (#5432) content_too_large object.
  * start_line/end_line slicing is applied before the cap is measured.
  * total_lines reflects the true full-file count even when a slice was
    requested, and the note never references get_file_metadata (no such tool).
  * A parsed dict (e.g. Excel) is guarded as non-chunkable, not sliced.
  * A ToolException from the backend passes through unchanged.

SharePoint's facade read_file delegates to a backend (REST or Graph) whose
read_file already returns fully-parsed content (str or dict); the cap is
applied once at the facade, so the backend is faked directly here.
"""

from langchain_core.tools import ToolException

from elitea_sdk.runtime.langchain.document_loaders.EliteAExcelLoader import (
    ExcelReadEstimate,
    ExcelReadLimitExceeded,
)
from elitea_sdk.tools.sharepoint.api_wrapper import SharepointApiWrapper
from elitea_sdk.tools.utils.file_metadata import (
    DEFAULT_MAX_OUTPUT_CHARS,
    GET_FILE_METADATA_DIRECTIVE,
    RESULT_STATUS_KEY,
    ResultStatus,
)


class FakeBackend:
    def __init__(self, result_by_path: dict):
        self._result_by_path = result_by_path
        self.elitea = None
        self.llm = None

    def read_file(self, path, is_capture_image=False, page_number=None,
                  sheet_name=None, excel_by_sheets=False):
        return self._result_by_path[path]


class FakeExcelOverLimitBackend:
    """Raises ExcelReadLimitExceeded, mirroring EliteAExcelLoader.get_content()."""

    def __init__(self, estimate: ExcelReadEstimate):
        self._estimate = estimate
        self.elitea = None
        self.llm = None

    def read_file(self, path, is_capture_image=False, page_number=None,
                  sheet_name=None, excel_by_sheets=False):
        raise ExcelReadLimitExceeded("Excel read request exceeds safety limits.",
                                      estimate=self._estimate)


def _make_wrapper(result_by_path: dict) -> SharepointApiWrapper:
    wrapper = SharepointApiWrapper.model_construct(site_url="https://test.sharepoint.com/sites/Test")
    wrapper._backend = FakeBackend(result_by_path)
    return wrapper


def _make_excel_estimate(sheet_names):
    return ExcelReadEstimate(
        sheets=[{"name": name, "max_row": 50000, "max_column": 20} for name in sheet_names],
        total_rows_workbook=50000 * len(sheet_names),
        target_sheet=sheet_names[0],
        target_sheet_total_rows=50000,
        requested_start_row=1,
        requested_end_row=50000,
        requested_rows=50000,
        sampled_rows=10,
        sampled_chars=5000,
        estimated_output_chars=25000000,
        embedded_images=0,
        file_size_bytes=30_000_000,
        is_unbounded_read=True,
        violations=["estimated output size=25000000 exceeds limit 200000"],
    )


def test_sharepoint_small_file_returns_plain_string():
    wrapper = _make_wrapper({"file.txt": "hello world"})
    assert wrapper.read_file("file.txt") == "hello world"


def test_sharepoint_over_limit_file_returns_structured_guidance():
    lines = [f"line {i} " + ("z" * 30) for i in range(1, 6000)]
    wrapper = _make_wrapper({"big.txt": "\n".join(lines)})

    result = wrapper.read_file("big.txt")

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["context"]["limit_chars"] == DEFAULT_MAX_OUTPUT_CHARS
    assert GET_FILE_METADATA_DIRECTIVE not in result["instruction_for_readFile"]["notes"]
    assert "start_line" in result["instruction_for_readFile"]["first_class_params"]


def test_sharepoint_start_end_line_slices_before_cap():
    lines = [f"line {i}" for i in range(1, 6)]
    wrapper = _make_wrapper({"file.txt": "\n".join(lines)})

    result = wrapper.read_file("file.txt", start_line=2, end_line=4)

    assert result == "line 2\nline 3\nline 4\n"


def test_sharepoint_small_slice_of_huge_file_avoids_cap():
    huge_lines = [f"line {i}" for i in range(1, 100000)]
    wrapper = _make_wrapper({"huge.txt": "\n".join(huge_lines)})

    result = wrapper.read_file("huge.txt", start_line=1, end_line=3)

    assert result == "line 1\nline 2\nline 3\n"


def test_sharepoint_over_limit_slice_reports_true_full_file_total_lines():
    huge_lines = ["x" * 1000 for _ in range(500)] + [f"line {i}" for i in range(500, 50000)]
    wrapper = _make_wrapper({"huge.txt": "\n".join(huge_lines)})

    result = wrapper.read_file("huge.txt", start_line=1, end_line=500)

    assert isinstance(result, dict)
    assert result["total_lines"] == 50000
    assert "start_line=1, end_line=500" in result["context"]["requested"]


def test_sharepoint_over_limit_dict_result_is_refused_not_sliced():
    big_dict = {"sheet1": ["row " + ("z" * 40) for _ in range(10000)]}
    wrapper = _make_wrapper({"big.xlsx": big_dict})

    result = wrapper.read_file("big.xlsx")

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["read_limits"]["full_read_allowed"] is False
    assert "start_line" not in result["instruction_for_readFile"]["first_class_params"]


def test_sharepoint_small_dict_result_passes_through_unchanged():
    small_dict = {"sheet1": ["row1", "row2"]}
    wrapper = _make_wrapper({"small.xlsx": small_dict})

    assert wrapper.read_file("small.xlsx") is small_dict


def test_sharepoint_tool_exception_passes_through_unchanged():
    exc = ToolException("File not found. Please, check file name and path.")
    wrapper = _make_wrapper({"missing.txt": exc})

    assert wrapper.read_file("missing.txt") is exc


def test_sharepoint_excel_over_limit_no_sheet_name_suggests_sheet_name():
    estimate = _make_excel_estimate(["Sheet1", "Sheet2"])
    wrapper = SharepointApiWrapper.model_construct(site_url="https://test.sharepoint.com/sites/Test")
    wrapper._backend = FakeExcelOverLimitBackend(estimate)

    result = wrapper.read_file("big.xlsx")

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["read_limits"]["full_read_allowed"] is False
    params = result["instruction_for_readFile"]["first_class_params"]
    assert "sheet_name" in params
    assert "Sheet1" in params["sheet_name"] and "Sheet2" in params["sheet_name"]
    # Only sheet_name may be suggested — read_document has no row-range params.
    assert "start_row" not in params
    assert result["instruction_for_readFile"]["extra_params"] == {}
    notes = result["instruction_for_readFile"]["notes"]
    assert "start_row" not in notes and "header_row" not in notes and "evaluate_formulas" not in notes
    assert GET_FILE_METADATA_DIRECTIVE not in notes


def test_sharepoint_excel_over_limit_with_sheet_name_refuses_plainly():
    estimate = _make_excel_estimate(["Sheet1"])
    wrapper = SharepointApiWrapper.model_construct(site_url="https://test.sharepoint.com/sites/Test")
    wrapper._backend = FakeExcelOverLimitBackend(estimate)

    result = wrapper.read_file("big.xlsx", sheet_name="Sheet1")

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["read_limits"]["full_read_allowed"] is False
    assert result["instruction_for_readFile"]["first_class_params"] == {}
    assert "Sheet1" in result["instruction_for_readFile"]["notes"]
