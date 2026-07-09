"""Tests for Azure DevOps Repos read_file cap + structured guidance (Phase 5, #5449).

ADO's read_file keeps its pre-existing offset/limit args (no start_line/end_line
added — removing/renaming tool args is a breaking change for pipelines). The
shared over-limit guidance is relabeled from start_line/end_line to offset/limit
so a retry from the guidance uses arg names ADO's tool actually accepts.

Assertions otherwise mirror tests/tools/github/test_github_read_file_cap.py:
  * Small files return plain strings unchanged.
  * Over-cap files return the PRE-1 (#5432) content_too_large object.
  * offset/limit slicing is applied before the cap is measured.
  * total_lines reflects the true full-file count even when a slice was
    requested, and the note never references get_file_metadata (no such tool).
  * _read_file (feeding edit_file) stays uncapped.

ADO's get_item_text returns a generator of bytes chunks, decoded in _read_file.
"""

from elitea_sdk.tools.ado.repos.repos_wrapper import ReposApiWrapper
from elitea_sdk.tools.utils.file_metadata import (
    DEFAULT_MAX_OUTPUT_CHARS,
    GET_FILE_METADATA_DIRECTIVE,
    RESULT_STATUS_KEY,
    ResultStatus,
)


class FakeAdoClient:
    def __init__(self, contents_by_path: dict):
        self._contents_by_path = contents_by_path

    def get_item_text(self, repository_id, project, path, version_descriptor):
        # Mimic the real API: a generator of utf-8 byte chunks.
        return (chunk for chunk in [self._contents_by_path[path].encode("utf-8")])


def _make_ado(contents_by_path: dict) -> ReposApiWrapper:
    wrapper = ReposApiWrapper.model_construct(
        repository_id="repo",
        project="proj",
        base_branch="main",
        active_branch="main",
        ado_client_instance=FakeAdoClient(contents_by_path),
    )
    wrapper._file_content_cache = {}
    return wrapper


def test_ado_small_file_returns_plain_string():
    wrapper = _make_ado({"src/main.py": "hello world"})
    assert wrapper.read_file("src/main.py", "main") == "hello world"


def test_ado_over_limit_file_returns_structured_guidance():
    lines = [f"line {i} " + ("z" * 30) for i in range(1, 6000)]
    wrapper = _make_ado({"big.py": "\n".join(lines)})

    result = wrapper.read_file("big.py", "main")

    assert isinstance(result, dict)
    assert result[RESULT_STATUS_KEY] == ResultStatus.CONTENT_TOO_LARGE.value
    assert result["context"]["limit_chars"] == DEFAULT_MAX_OUTPUT_CHARS
    assert GET_FILE_METADATA_DIRECTIVE not in result["instruction_for_readFile"]["notes"]


def test_ado_over_limit_guidance_advertises_offset_limit_not_start_end_line():
    # The shared helper emits start_line/end_line; ADO's tool takes offset/limit,
    # so the guidance must be relabeled to offset/limit for a valid retry.
    lines = [f"line {i} " + ("z" * 30) for i in range(1, 6000)]
    wrapper = _make_ado({"big.py": "\n".join(lines)})

    result = wrapper.read_file("big.py", "main")

    params = result["instruction_for_readFile"]["first_class_params"]
    assert set(params) == {"offset", "limit"}
    assert "start_line" not in params
    assert "end_line" not in params


def test_ado_offset_limit_slices_before_cap():
    lines = [f"line {i}" for i in range(1, 6)]
    wrapper = _make_ado({"src/main.py": "\n".join(lines)})

    result = wrapper.read_file("src/main.py", "main", offset=2, limit=2)

    assert result == "line 2\nline 3\n"


def test_ado_limit_only_reads_head_not_whole_file():
    # apply_line_slice ignores limit when offset is None, so read_file must
    # default offset to 1 for a limit-only call to behave as a head read.
    lines = [f"line {i}" for i in range(1, 6)]
    wrapper = _make_ado({"src/main.py": "\n".join(lines)})

    result = wrapper.read_file("src/main.py", "main", limit=3)

    assert result == "line 1\nline 2\nline 3\n"


def test_ado_single_line_over_cap_guidance_is_not_relabeled():
    # A single-physical-line file over the cap gets the shared helper's
    # refuse-and-explain response (empty first_class_params). The relabel
    # helper must not re-inject offset/limit into that empty dict — doing so
    # would contradict the "bounded read is not possible" notes.
    huge_single_line = "x" * (200_000 + 1)
    wrapper = _make_ado({"bundle.min.js": huge_single_line})

    result = wrapper.read_file("bundle.min.js", "main")

    assert isinstance(result, dict)
    assert result["read_limits"]["full_read_allowed"] is False
    assert result["instruction_for_readFile"]["first_class_params"] == {}
    assert "refused" in result["instruction_for_readFile"]["notes"]


def test_ado_small_slice_of_huge_file_avoids_cap():
    huge_lines = [f"line {i}" for i in range(1, 100000)]
    wrapper = _make_ado({"huge.py": "\n".join(huge_lines)})

    result = wrapper.read_file("huge.py", "main", offset=1, limit=3)

    assert result == "line 1\nline 2\nline 3\n"


def test_ado_over_limit_slice_reports_true_full_file_total_lines():
    huge_lines = ["x" * 1000 for _ in range(500)] + [f"line {i}" for i in range(500, 50000)]
    wrapper = _make_ado({"huge.py": "\n".join(huge_lines)})

    result = wrapper.read_file("huge.py", "main", offset=1, limit=500)

    assert isinstance(result, dict)
    assert result["total_lines"] == 50000
    assert "offset=1, limit=500" in result["context"]["requested"]


def test_ado_internal_read_file_is_uncapped_for_edit():
    # _read_file feeds edit_file, which must see the full file — never the
    # capped guidance dict the public read_file returns.
    body = "\n".join(f"line {i} " + ("z" * 30) for i in range(1, 6000))
    wrapper = _make_ado({"big.py": body})

    assert wrapper._read_file("big.py", "main") == body
    assert isinstance(wrapper.read_file("big.py", "main"), dict)
