"""Tests for GitHub read_file cap + structured guidance (Phase 5, #5447).

Covers:
  * Small files are returned as plain strings, unchanged.
  * Over-cap files return the PRE-1 (#5432) structured content_too_large object.
  * start_line/end_line slicing is applied before the cap is measured.
  * Slicing a small range out of a huge file avoids the cap entirely.
  * total_lines reflects the true full-file count even when a slice was
    requested (bug #1) and never references get_file_metadata, which isn't a
    registered tool for GitHub (bug #2).
  * read_multiple_files enforces a cumulative cap across the whole batch, not
    just a per-file cap (bug #3).
"""

from elitea_sdk.tools.github.github_client import GitHubClient
from elitea_sdk.tools.utils.file_metadata import (
    DEFAULT_MAX_OUTPUT_CHARS,
    GET_FILE_METADATA_DIRECTIVE,
    RESULT_STATUS_KEY,
    ResultStatus,
)


class FakeFile:
    def __init__(self, content: str):
        self.decoded_content = content.encode('utf-8')


class FakeRepo:
    def __init__(self, content: str):
        self._content = content

    def get_contents(self, path, ref=None):
        return FakeFile(self._content)


class FakeGitHubApi:
    def __init__(self, repo):
        self.repo = repo

    def get_repo(self, repo_name):
        return self.repo


def _make_client(content: str) -> GitHubClient:
    repo = FakeRepo(content)
    return GitHubClient.model_construct(
        github_repository='owner/repo',
        active_branch='main',
        github_base_branch='main',
        github_api=FakeGitHubApi(repo),
        elitea=None,
    )


def test_small_file_returns_plain_string():
    client = _make_client("hello world")
    result = client.read_file(file_path='src/main.py')
    assert result == "hello world"


def test_over_limit_file_returns_structured_guidance():
    # Single giant line, no newlines at all -> triggers the "no usable line
    # breaks" honesty refusal path, not a start_line/end_line range hint.
    body = "x" * (DEFAULT_MAX_OUTPUT_CHARS + 500)
    client = _make_client(body)

    result = client.read_file(file_path='big.py')

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["context"]["actual_chars"] == DEFAULT_MAX_OUTPUT_CHARS + 500
    assert result["context"]["limit_chars"] == DEFAULT_MAX_OUTPUT_CHARS
    # bug #2: no get_file_metadata tool exists for GitHub — must never be referenced.
    assert GET_FILE_METADATA_DIRECTIVE not in result["instruction_for_readFile"]["notes"]
    assert result["unit"] == "lines"


def test_over_limit_note_never_references_get_file_metadata():
    lines = [f"line {i} " + ("z" * 30) for i in range(1, 6000)]
    client = _make_client("\n".join(lines))

    result = client.read_file(file_path='big.py')

    notes = result["instruction_for_readFile"]["notes"]
    assert GET_FILE_METADATA_DIRECTIVE not in notes
    assert "start_line" in result["instruction_for_readFile"]["first_class_params"]


def test_start_end_line_slices_before_measuring_cap():
    lines = [f"line {i}" for i in range(1, 6)]
    client = _make_client("\n".join(lines))

    result = client.read_file(file_path='src/main.py', start_line=2, end_line=4)

    assert result == "line 2\nline 3\nline 4\n"


def test_small_slice_of_huge_file_avoids_cap():
    huge_lines = [f"line {i}" for i in range(1, 100000)]
    client = _make_client("\n".join(huge_lines))

    result = client.read_file(file_path='huge.py', start_line=1, end_line=3)

    assert result == "line 1\nline 2\nline 3\n"


def test_over_limit_slice_still_returns_guidance():
    huge_lines = ["x" * 1000 for _ in range(500)]
    client = _make_client("\n".join(huge_lines))

    result = client.read_file(file_path='huge.py', start_line=1, end_line=500)

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert "start_line=1, end_line=500" in result["context"]["requested"]


def test_over_limit_slice_reports_true_full_file_total_lines():
    # bug #1: total_lines must reflect the whole file, not the requested
    # slice — a 50,000-line file sliced to 500 lines still reports 50,000.
    huge_lines = ["x" * 1000 for _ in range(500)] + [f"line {i}" for i in range(500, 50000)]
    client = _make_client("\n".join(huge_lines))

    result = client.read_file(file_path='huge.py', start_line=1, end_line=500)

    assert isinstance(result, dict)
    assert result["total_lines"] == 50000


class TestReadMultipleFilesCumulativeCap:
    """bug #3: read_multiple_files must enforce a cumulative cap across the
    whole batch, not just a per-file cap — many small-but-full files can sum
    to the same freeze risk as one oversized file."""

    def _make_multi_client(self, contents_by_path: dict) -> GitHubClient:
        class MultiFakeRepo:
            def get_contents(self, path, ref=None):
                return FakeFile(contents_by_path[path])

        return GitHubClient.model_construct(
            github_repository='owner/repo',
            active_branch='main',
            github_base_branch='main',
            github_api=FakeGitHubApi(MultiFakeRepo()),
            elitea=None,
        )

    def test_small_batch_all_returned_normally(self):
        client = self._make_multi_client({"a.py": "hello a", "b.py": "hello b"})

        result = client.read_multiple_files(file_paths=["a.py", "b.py"])

        assert result == {"a.py": "hello a", "b.py": "hello b"}

    def test_batch_exceeding_cumulative_cap_skips_later_files(self):
        # First file alone consumes the whole cumulative budget (right at the
        # per-file cap, so it's still returned as plain content); the second
        # file must be skipped without being fetched at all.
        big_body = "x" * DEFAULT_MAX_OUTPUT_CHARS
        client = self._make_multi_client({"big.py": big_body, "small.py": "hello"})

        result = client.read_multiple_files(file_paths=["big.py", "small.py"])

        assert result["big.py"] == big_body
        assert "Skipped" in result["small.py"]
        assert "cumulative" in result["small.py"]

    def test_per_file_over_cap_result_is_measured_by_its_actual_returned_size(self):
        # A file that individually exceeds the per-file cap returns a small
        # guidance dict, not its (uncapped) content — measure_result_chars
        # must size that dict's actual serialized length, not the size of the
        # oversized file it refused to return, so later small files in the
        # same batch are still read normally rather than wrongly skipped.
        first_body = "x" * (DEFAULT_MAX_OUTPUT_CHARS + 500)
        client = self._make_multi_client({"first.py": first_body, "second.py": "hello"})

        result = client.read_multiple_files(file_paths=["first.py", "second.py"])

        assert isinstance(result["first.py"], dict)
        assert result["second.py"] == "hello"
