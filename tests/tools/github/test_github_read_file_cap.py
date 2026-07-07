"""Tests for GitHub read_file cap + structured guidance (Phase 5, #5447).

Covers:
  * Small files are returned as plain strings, unchanged.
  * Over-cap files return the PRE-1 (#5432) structured content_too_large object.
  * start_line/end_line slicing is applied before the cap is measured.
  * Slicing a small range out of a huge file avoids the cap entirely.
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
    body = "x" * (DEFAULT_MAX_OUTPUT_CHARS + 500)
    client = _make_client(body)

    result = client.read_file(file_path='big.py')

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["context"]["actual_chars"] == DEFAULT_MAX_OUTPUT_CHARS + 500
    assert result["context"]["limit_chars"] == DEFAULT_MAX_OUTPUT_CHARS
    assert GET_FILE_METADATA_DIRECTIVE in result["instruction_for_readFile"]["notes"]
    assert result["unit"] == "lines"


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
