"""Tests for LocalGit read_file cap + structured guidance (Phase 5, #5450).

Assertions mirror tests/tools/bitbucket/test_bitbucket_read_file_cap.py and the
GitLab/GitHub equivalents:
  * Small files return plain strings unchanged.
  * Over-cap files return the PRE-1 (#5432) content_too_large object.
  * start_line/end_line slicing is applied before the cap is measured.
  * total_lines reflects the true full-file count even when a slice was
    requested, and the note never references get_file_metadata (no such tool).
  * _read_file (feeding update_file/edit_file) stays uncapped.
  * read_multiple_files is capped both per-file and cumulatively.
  * The not-found path is unchanged (plain "cannot be read" string).

LocalGit reads the checked-out working tree directly, so a temp dir plus a
minimal fake repo object exposing `working_dir` stands in for a real clone.
"""

import os
from types import SimpleNamespace

from elitea_sdk.tools.localgit.local_git import LocalGit
from elitea_sdk.tools.utils.file_metadata import (
    DEFAULT_MAX_OUTPUT_CHARS,
    GET_FILE_METADATA_DIRECTIVE,
    RESULT_STATUS_KEY,
    ResultStatus,
)


def _make_localgit(tmp_path, contents_by_path: dict) -> LocalGit:
    for rel, body in contents_by_path.items():
        full = tmp_path / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(body)
    wrapper = LocalGit.model_construct(repo_path="r", base_path=str(tmp_path))
    # self.repo is set externally in production; a namespace with working_dir suffices.
    object.__setattr__(wrapper, "repo", SimpleNamespace(working_dir=str(tmp_path)))
    return wrapper


def test_localgit_small_file_returns_plain_string(tmp_path):
    wrapper = _make_localgit(tmp_path, {"src/main.py": "hello world"})
    assert wrapper.read_file("src/main.py") == "hello world"


def test_localgit_missing_file_returns_plain_string(tmp_path):
    wrapper = _make_localgit(tmp_path, {})
    result = wrapper.read_file("nope.py")
    assert isinstance(result, str)
    assert "cannot be read" in result


def test_localgit_over_limit_file_returns_structured_guidance(tmp_path):
    lines = [f"line {i} " + ("z" * 30) for i in range(1, 6000)]
    wrapper = _make_localgit(tmp_path, {"big.py": "\n".join(lines)})

    result = wrapper.read_file("big.py")

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["context"]["limit_chars"] == DEFAULT_MAX_OUTPUT_CHARS
    assert GET_FILE_METADATA_DIRECTIVE not in result["instruction_for_readFile"]["notes"]
    assert "start_line" in result["instruction_for_readFile"]["first_class_params"]


def test_localgit_start_end_line_slices_before_cap(tmp_path):
    lines = [f"line {i}" for i in range(1, 6)]
    wrapper = _make_localgit(tmp_path, {"src/main.py": "\n".join(lines)})

    result = wrapper.read_file("src/main.py", start_line=2, end_line=4)

    assert result == "line 2\nline 3\nline 4\n"


def test_localgit_small_slice_of_huge_file_avoids_cap(tmp_path):
    huge_lines = [f"line {i}" for i in range(1, 100000)]
    wrapper = _make_localgit(tmp_path, {"huge.py": "\n".join(huge_lines)})

    result = wrapper.read_file("huge.py", start_line=1, end_line=3)

    assert result == "line 1\nline 2\nline 3\n"


def test_localgit_over_limit_slice_reports_true_full_file_total_lines(tmp_path):
    huge_lines = ["x" * 1000 for _ in range(500)] + [f"line {i}" for i in range(500, 50000)]
    wrapper = _make_localgit(tmp_path, {"huge.py": "\n".join(huge_lines)})

    result = wrapper.read_file("huge.py", start_line=1, end_line=500)

    assert isinstance(result, dict)
    assert result["total_lines"] == 50000
    assert "start_line=1, end_line=500" in result["context"]["requested"]


def test_localgit_internal_read_file_is_uncapped_for_edit(tmp_path):
    body = "\n".join(f"line {i} " + ("z" * 30) for i in range(1, 6000))
    wrapper = _make_localgit(tmp_path, {"big.py": body})

    assert wrapper._read_file("big.py") == body
    assert isinstance(wrapper.read_file("big.py"), dict)


def test_localgit_file_op_schemas_advertise_no_branch(tmp_path):
    # LocalGit has no per-call branch; its tool schemas must not advertise one,
    # or the LLM passes branch and read_multiple_files raises TypeError.
    wrapper = _make_localgit(tmp_path, {})
    tools = {t["name"]: t for t in wrapper.get_available_tools()}

    rmf_fields = tools["read_multiple_files"]["args_schema"].model_fields
    assert "branch" not in rmf_fields
    assert set(rmf_fields) == {"file_paths", "offset", "limit"}

    assert "branch" not in tools["grep_file"]["args_schema"].model_fields


class TestLocalGitReadMultipleFilesCumulativeCap:
    def test_small_batch_all_returned_normally(self, tmp_path):
        wrapper = _make_localgit(tmp_path, {"a.py": "hello a", "b.py": "hello b"})

        result = wrapper.read_multiple_files(["a.py", "b.py"])

        assert result == {"a.py": "hello a", "b.py": "hello b"}

    def test_batch_exceeding_cumulative_cap_skips_later_files(self, tmp_path):
        big_body = "x" * DEFAULT_MAX_OUTPUT_CHARS
        wrapper = _make_localgit(tmp_path, {"big.py": big_body, "small.py": "hello"})

        result = wrapper.read_multiple_files(["big.py", "small.py"])

        assert result["big.py"] == big_body
        assert "Skipped" in result["small.py"]
        assert "cumulative" in result["small.py"]

    def test_per_file_over_cap_result_is_measured_by_returned_size(self, tmp_path):
        first = [f"line {i} " + ("z" * 30) for i in range(1, 6000)]
        wrapper = _make_localgit(tmp_path, {"first.py": "\n".join(first), "second.py": "hello"})

        result = wrapper.read_multiple_files(["first.py", "second.py"])

        assert isinstance(result["first.py"], dict)
        assert result["second.py"] == "hello"
