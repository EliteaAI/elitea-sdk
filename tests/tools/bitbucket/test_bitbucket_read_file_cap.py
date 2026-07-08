"""Tests for Bitbucket read_file cap + structured guidance (Phase 5, #5449).

Assertions mirror tests/tools/github/test_github_read_file_cap.py and the GitLab
equivalent:
  * Small files return plain strings unchanged.
  * Over-cap files return the PRE-1 (#5432) content_too_large object.
  * start_line/end_line slicing is applied before the cap is measured.
  * total_lines reflects the true full-file count even when a slice was
    requested, and the note never references get_file_metadata (no such tool).
  * _read_file (feeding create_file/edit_file) stays uncapped.
  * read_multiple_files is capped both per-file and cumulatively.

Bitbucket's get_file returns a decoded str for both cloud and server backends.
"""

from elitea_sdk.tools.bitbucket.api_wrapper import BitbucketAPIWrapper
from elitea_sdk.tools.utils.file_metadata import (
    DEFAULT_MAX_OUTPUT_CHARS,
    GET_FILE_METADATA_DIRECTIVE,
    RESULT_STATUS_KEY,
    ResultStatus,
)


class FakeBitbucket:
    def __init__(self, contents_by_path: dict):
        self._contents_by_path = contents_by_path

    def get_file(self, file_path, branch):
        return self._contents_by_path[file_path]


def _make_bitbucket(contents_by_path: dict) -> BitbucketAPIWrapper:
    wrapper = BitbucketAPIWrapper.model_construct(branch="main")
    wrapper._bitbucket = FakeBitbucket(contents_by_path)
    wrapper._active_branch = "main"
    return wrapper


def test_bitbucket_small_file_returns_plain_string():
    wrapper = _make_bitbucket({"src/main.py": "hello world"})
    assert wrapper.read_file("src/main.py", "main") == "hello world"


def test_bitbucket_over_limit_file_returns_structured_guidance():
    lines = [f"line {i} " + ("z" * 30) for i in range(1, 6000)]
    wrapper = _make_bitbucket({"big.py": "\n".join(lines)})

    result = wrapper.read_file("big.py", "main")

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["context"]["limit_chars"] == DEFAULT_MAX_OUTPUT_CHARS
    assert GET_FILE_METADATA_DIRECTIVE not in result["instruction_for_readFile"]["notes"]
    assert "start_line" in result["instruction_for_readFile"]["first_class_params"]


def test_bitbucket_start_end_line_slices_before_cap():
    lines = [f"line {i}" for i in range(1, 6)]
    wrapper = _make_bitbucket({"src/main.py": "\n".join(lines)})

    result = wrapper.read_file("src/main.py", "main", start_line=2, end_line=4)

    assert result == "line 2\nline 3\nline 4\n"


def test_bitbucket_small_slice_of_huge_file_avoids_cap():
    huge_lines = [f"line {i}" for i in range(1, 100000)]
    wrapper = _make_bitbucket({"huge.py": "\n".join(huge_lines)})

    result = wrapper.read_file("huge.py", "main", start_line=1, end_line=3)

    assert result == "line 1\nline 2\nline 3\n"


def test_bitbucket_over_limit_slice_reports_true_full_file_total_lines():
    huge_lines = ["x" * 1000 for _ in range(500)] + [f"line {i}" for i in range(500, 50000)]
    wrapper = _make_bitbucket({"huge.py": "\n".join(huge_lines)})

    result = wrapper.read_file("huge.py", "main", start_line=1, end_line=500)

    assert isinstance(result, dict)
    assert result["total_lines"] == 50000
    assert "start_line=1, end_line=500" in result["context"]["requested"]


def test_bitbucket_internal_read_file_is_uncapped_for_edit():
    body = "\n".join(f"line {i} " + ("z" * 30) for i in range(1, 6000))
    wrapper = _make_bitbucket({"big.py": body})

    assert wrapper._read_file("big.py", "main") == body
    assert isinstance(wrapper.read_file("big.py", "main"), dict)


class TestBitbucketReadMultipleFilesCumulativeCap:
    def test_small_batch_all_returned_normally(self):
        wrapper = _make_bitbucket({"a.py": "hello a", "b.py": "hello b"})

        result = wrapper.read_multiple_files(["a.py", "b.py"], "main")

        assert result == {"a.py": "hello a", "b.py": "hello b"}

    def test_batch_exceeding_cumulative_cap_skips_later_files(self):
        big_body = "x" * DEFAULT_MAX_OUTPUT_CHARS
        wrapper = _make_bitbucket({"big.py": big_body, "small.py": "hello"})

        result = wrapper.read_multiple_files(["big.py", "small.py"], "main")

        assert result["big.py"] == big_body
        assert "Skipped" in result["small.py"]
        assert "cumulative" in result["small.py"]

    def test_per_file_over_cap_result_is_measured_by_returned_size(self):
        # A file over the per-file cap returns a small guidance dict; its
        # serialized size (not the refused file's size) is what counts toward
        # the cumulative budget, so later small files are still read.
        first = [f"line {i} " + ("z" * 30) for i in range(1, 6000)]
        wrapper = _make_bitbucket({"first.py": "\n".join(first), "second.py": "hello"})

        result = wrapper.read_multiple_files(["first.py", "second.py"], "main")

        assert isinstance(result["first.py"], dict)
        assert result["second.py"] == "hello"
