"""Tests for GitLab read_file cap + structured guidance (Phase 5, #5448).

Covers both GitLab wrappers with the same #5447 pattern:
  * GitLabAPIWrapper (gitlab) — read_file via parse_file_content (str or dict),
    plus a cumulative-capped read_multiple_files.
  * GitLabWorkspaceAPIWrapper (gitlab_org) — raw utf-8 read_file with a
    repository arg.

Assertions mirror tests/tools/github/test_github_read_file_cap.py:
  * Small files return plain strings unchanged.
  * Over-cap files return the PRE-1 (#5432) content_too_large object.
  * start_line/end_line slicing is applied before the cap is measured.
  * total_lines reflects the true full-file count even when a slice was
    requested, and the note never references get_file_metadata (no such tool).
  * A non-text (parsed dict) over-cap result is refused as non-chunkable.
"""

from elitea_sdk.tools.gitlab.api_wrapper import GitLabAPIWrapper
from elitea_sdk.tools.gitlab_org.api_wrapper import GitLabWorkspaceAPIWrapper
from elitea_sdk.tools.utils.file_metadata import (
    DEFAULT_MAX_OUTPUT_CHARS,
    GET_FILE_METADATA_DIRECTIVE,
    RESULT_STATUS_KEY,
    ResultStatus,
)


# --------------------------------------------------------------------------
# gitlab (GitLabAPIWrapper)
# --------------------------------------------------------------------------

class FakeGitLabFile:
    def __init__(self, content: str):
        self._content = content

    def decode(self):
        return self._content.encode("utf-8")


class FakeGitLabFiles:
    def __init__(self, contents_by_path: dict):
        self._contents_by_path = contents_by_path

    def get(self, file_path, branch):
        return FakeGitLabFile(self._contents_by_path[file_path])


class FakeGitLabRepo:
    def __init__(self, contents_by_path: dict):
        self.files = FakeGitLabFiles(contents_by_path)
        self.default_branch = "main"


def _make_gitlab(contents_by_path: dict) -> GitLabAPIWrapper:
    wrapper = GitLabAPIWrapper.model_construct(branch="main", llm=None)
    wrapper._active_branch = "main"
    wrapper._repo_instance = FakeGitLabRepo(contents_by_path)
    return wrapper


def test_gitlab_small_file_returns_plain_string():
    wrapper = _make_gitlab({"src/main.py": "hello world"})
    assert wrapper.read_file("src/main.py", "main") == "hello world"


def test_gitlab_over_limit_file_returns_structured_guidance():
    lines = [f"line {i} " + ("z" * 30) for i in range(1, 6000)]
    wrapper = _make_gitlab({"big.py": "\n".join(lines)})

    result = wrapper.read_file("big.py", "main")

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["context"]["limit_chars"] == DEFAULT_MAX_OUTPUT_CHARS
    assert GET_FILE_METADATA_DIRECTIVE not in result["instruction_for_readFile"]["notes"]
    assert "start_line" in result["instruction_for_readFile"]["first_class_params"]


def test_gitlab_start_end_line_slices_before_cap():
    lines = [f"line {i}" for i in range(1, 6)]
    wrapper = _make_gitlab({"src/main.py": "\n".join(lines)})

    result = wrapper.read_file("src/main.py", "main", start_line=2, end_line=4)

    assert result == "line 2\nline 3\nline 4\n"


def test_gitlab_small_slice_of_huge_file_avoids_cap():
    huge_lines = [f"line {i}" for i in range(1, 100000)]
    wrapper = _make_gitlab({"huge.py": "\n".join(huge_lines)})

    result = wrapper.read_file("huge.py", "main", start_line=1, end_line=3)

    assert result == "line 1\nline 2\nline 3\n"


def test_gitlab_over_limit_slice_reports_true_full_file_total_lines():
    huge_lines = ["x" * 1000 for _ in range(500)] + [f"line {i}" for i in range(500, 50000)]
    wrapper = _make_gitlab({"huge.py": "\n".join(huge_lines)})

    result = wrapper.read_file("huge.py", "main", start_line=1, end_line=500)

    assert isinstance(result, dict)
    assert result["total_lines"] == 50000
    assert "start_line=1, end_line=500" in result["context"]["requested"]


def test_gitlab_non_text_over_limit_is_refused_as_non_chunkable(monkeypatch):
    # A parsed .xlsx returns a dict, not a str. An over-cap dict has no line
    # structure, so it must be refused as non-chunkable rather than passed through.
    big_dict = {"rows": ["x" * 1000 for _ in range(300)]}
    monkeypatch.setattr(
        "elitea_sdk.tools.gitlab.api_wrapper.parse_file_content",
        lambda **kwargs: big_dict,
    )
    wrapper = _make_gitlab({"data.xlsx": "irrelevant-bytes"})

    result = wrapper.read_file("data.xlsx", "main")

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["read_limits"]["full_read_allowed"] is False
    assert result.get("unit") is None
    # No line-chunk params advertised — line slicing does not apply.
    assert result["instruction_for_readFile"]["first_class_params"] == {}


def test_gitlab_non_text_within_limit_passes_through(monkeypatch):
    small_dict = {"rows": ["a", "b"]}
    monkeypatch.setattr(
        "elitea_sdk.tools.gitlab.api_wrapper.parse_file_content",
        lambda **kwargs: small_dict,
    )
    wrapper = _make_gitlab({"data.xlsx": "irrelevant-bytes"})

    assert wrapper.read_file("data.xlsx", "main") == small_dict


def test_gitlab_internal_read_file_is_uncapped_for_edit_and_append():
    # _read_file feeds edit_file/append_file, which must see the full file —
    # never the capped guidance dict the public read_file returns.
    body = "\n".join(f"line {i} " + ("z" * 30) for i in range(1, 6000))
    wrapper = _make_gitlab({"big.py": body})

    assert wrapper._read_file("big.py", "main") == body
    assert isinstance(wrapper.read_file("big.py", "main"), dict)


class TestGitLabReadMultipleFilesCumulativeCap:
    def test_small_batch_all_returned_normally(self):
        wrapper = _make_gitlab({"a.py": "hello a", "b.py": "hello b"})

        result = wrapper.read_multiple_files(["a.py", "b.py"], "main")

        assert result == {"a.py": "hello a", "b.py": "hello b"}

    def test_batch_exceeding_cumulative_cap_skips_later_files(self):
        big_body = "x" * DEFAULT_MAX_OUTPUT_CHARS
        wrapper = _make_gitlab({"big.py": big_body, "small.py": "hello"})

        result = wrapper.read_multiple_files(["big.py", "small.py"], "main")

        assert result["big.py"] == big_body
        assert "Skipped" in result["small.py"]
        assert "cumulative" in result["small.py"]

    def test_per_file_over_cap_result_is_measured_by_returned_size(self):
        # A file over the per-file cap returns a small guidance dict; its
        # serialized size (not the refused file's size) is what counts toward
        # the cumulative budget, so later small files are still read.
        first = [f"line {i} " + ("z" * 30) for i in range(1, 6000)]
        wrapper = _make_gitlab({"first.py": "\n".join(first), "second.py": "hello"})

        result = wrapper.read_multiple_files(["first.py", "second.py"], "main")

        assert isinstance(result["first.py"], dict)
        assert result["second.py"] == "hello"


# --------------------------------------------------------------------------
# gitlab_org (GitLabWorkspaceAPIWrapper)
# --------------------------------------------------------------------------

class FakeOrgFile:
    def __init__(self, content: str):
        self._content = content

    def decode(self):
        return self._content.encode("utf-8")


class FakeOrgFiles:
    def __init__(self, content: str):
        self._content = content

    def get(self, file_path, branch):
        return FakeOrgFile(self._content)


class FakeOrgRepo:
    def __init__(self, content: str):
        self.files = FakeOrgFiles(content)


def _make_gitlab_org(content: str) -> GitLabWorkspaceAPIWrapper:
    wrapper = GitLabWorkspaceAPIWrapper.model_construct(branch="main")
    wrapper._active_branch = "main"
    repo = FakeOrgRepo(content)
    wrapper._get_repo = lambda repository=None: repo
    return wrapper


def test_gitlab_org_small_file_returns_plain_string():
    wrapper = _make_gitlab_org("hello world")
    assert wrapper.read_file("src/main.py", "main") == "hello world"


def test_gitlab_org_over_limit_file_returns_structured_guidance():
    lines = [f"line {i} " + ("z" * 30) for i in range(1, 6000)]
    wrapper = _make_gitlab_org("\n".join(lines))

    result = wrapper.read_file("big.py", "main")

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert GET_FILE_METADATA_DIRECTIVE not in result["instruction_for_readFile"]["notes"]


def test_gitlab_org_start_end_line_slices_before_cap():
    lines = [f"line {i}" for i in range(1, 6)]
    wrapper = _make_gitlab_org("\n".join(lines))

    result = wrapper.read_file("src/main.py", "main", start_line=2, end_line=4)

    assert result == "line 2\nline 3\nline 4\n"


def test_gitlab_org_over_limit_slice_reports_true_full_file_total_lines():
    huge_lines = ["x" * 1000 for _ in range(500)] + [f"line {i}" for i in range(500, 50000)]
    wrapper = _make_gitlab_org("\n".join(huge_lines))

    result = wrapper.read_file("huge.py", "main", start_line=1, end_line=500)

    assert isinstance(result, dict)
    assert result["total_lines"] == 50000
    assert "start_line=1, end_line=500" in result["context"]["requested"]


def test_gitlab_org_internal_read_file_is_uncapped_for_edit_and_append():
    body = "\n".join(f"line {i} " + ("z" * 30) for i in range(1, 6000))
    wrapper = _make_gitlab_org(body)

    assert wrapper._read_file("big.py", "main") == body
    assert isinstance(wrapper.read_file("big.py", "main"), dict)
