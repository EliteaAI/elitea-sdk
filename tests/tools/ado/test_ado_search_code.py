"""Tests for Azure DevOps Repos search_code.

Covers the bounded-response contract the tool promises to callers:
  * Defaults stay small and never return full file bodies.
  * Repository filters carry the repository *name* — the search index does not
    recognise the GUID that repository_id may hold.
  * total_count/truncated/next_skip describe results beyond the returned page.
  * Azure DevOps infoCode values surface as human-readable warnings.
  * Snippets are reconstructed from char offsets, because the service advertises
    includeSnippet but returns codeSnippet as null.
"""

import json
import threading

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.ado.repos.repos_wrapper import (
    MAX_FILES_FETCHED_FOR_SNIPPETS,
    PAGING,
    ReposApiWrapper,
)


class FakeHit:
    def __init__(self, char_offset=None, line=0, code_snippet=None):
        self.char_offset = char_offset
        self.line = line
        self.code_snippet = code_snippet


class FakeNamed:
    def __init__(self, name):
        self.name = name


class FakeVersion:
    def __init__(self, branch_name="main", change_id="abc123"):
        self.branch_name = branch_name
        self.change_id = change_id


class FakeResult:
    def __init__(self, path="/src/main.py", file_name="main.py", hits=None, change_id="abc123"):
        self.path = path
        self.file_name = file_name
        self.project = FakeNamed("proj")
        self.repository = FakeNamed("MyRepo")
        self.versions = [FakeVersion(change_id=change_id)]
        self.matches = {"content": hits or []}


class FakeResponse:
    def __init__(self, results=None, count=0, info_code=0):
        self.results = results or []
        self.count = count
        self.info_code = info_code


class FakeSearchClient:
    def __init__(self, response):
        self._response = response
        self.last_request = None
        self.last_project = None

    def fetch_code_search_results(self, request, project=None):
        self.last_request = request
        self.last_project = project
        return self._response


class FakeAdoClient:
    def __init__(self, contents_by_path=None):
        self._contents_by_path = contents_by_path or {}
        self._lock = threading.Lock()
        self.fetch_count = 0

    def get_item_text(self, repository_id, project, path, version_descriptor):
        with self._lock:
            self.fetch_count += 1
        return (chunk for chunk in [self._contents_by_path[path].encode("utf-8")])


def _make_ado(response, contents_by_path=None, repository_name="MyRepo"):
    wrapper = ReposApiWrapper.model_construct(
        repository_id="c1548045-29f6-4354-8114-55ef058be1a3",
        searchable_repository_name=repository_name,
        project="proj",
        base_branch="main",
        active_branch="main",
        ado_client_instance=FakeAdoClient(contents_by_path),
    )
    wrapper._file_content_cache = {}
    wrapper._search_client_instance = FakeSearchClient(response)
    return wrapper


def test_empty_query_returns_tool_exception():
    wrapper = _make_ado(FakeResponse())

    result = wrapper.search_code("   ")

    assert isinstance(result, ToolException)
    assert "cannot be empty" in str(result)


def test_defaults_scope_to_configured_project_and_repository_name():
    wrapper = _make_ado(FakeResponse(results=[FakeResult()], count=1))

    wrapper.search_code("QueueJobsNow", include_snippets=False)

    request = wrapper._search_client_instance.last_request
    assert request.filters == {"Project": ["proj"], "Repository": ["MyRepo"]}
    assert request.top == 5
    assert request.skip == 0
    assert wrapper._search_client_instance.last_project == "proj"


def test_repository_filter_falls_back_to_id_when_name_unknown():
    wrapper = _make_ado(FakeResponse(), repository_name=None)

    wrapper.search_code("anything")

    request = wrapper._search_client_instance.last_request
    assert request.filters["Repository"] == ["c1548045-29f6-4354-8114-55ef058be1a3"]


def test_branch_and_path_filters_are_opt_in():
    wrapper = _make_ado(FakeResponse())

    wrapper.search_code("q", branch="release/2.0", path="/src")

    request = wrapper._search_client_instance.last_request
    assert request.filters["Branch"] == ["release/2.0"]
    assert request.filters["Path"] == ["/src"]


@pytest.mark.parametrize(
    "top,skip,expected_top,expected_skip",
    [
        (10_000, 5_000, PAGING.max_top, PAGING.max_skip),
        (None, -5, PAGING.default_top, 0),
        (0, None, PAGING.default_top, 0),
    ],
)
def test_top_and_skip_are_clamped_to_service_limits(top, skip, expected_top, expected_skip):
    wrapper = _make_ado(FakeResponse())

    wrapper.search_code("q", top=top, skip=skip)

    request = wrapper._search_client_instance.last_request
    assert request.top == expected_top
    assert request.skip == expected_skip


def test_truncation_indicator_and_next_skip():
    results = [FakeResult(path=f"/f{i}.py") for i in range(5)]
    wrapper = _make_ado(FakeResponse(results=results, count=42))

    payload = json.loads(wrapper.search_code("q", include_snippets=False))

    assert payload["total_count"] == 42
    assert payload["returned"] == 5
    assert payload["truncated"] is True
    assert payload["next_skip"] == 5


def test_last_page_is_not_marked_truncated():
    results = [FakeResult(path=f"/f{i}.py") for i in range(2)]
    wrapper = _make_ado(FakeResponse(results=results, count=7))

    payload = json.loads(wrapper.search_code("q", skip=5, include_snippets=False))

    assert payload["truncated"] is False
    assert "next_skip" not in payload


def test_next_skip_is_withheld_past_the_paging_ceiling():
    results = [FakeResult(path=f"/f{i}.py") for i in range(5)]
    wrapper = _make_ado(FakeResponse(results=results, count=100_000))

    payload = json.loads(wrapper.search_code("q", skip=PAGING.max_skip, include_snippets=False))

    assert payload["truncated"] is True
    assert "next_skip" not in payload
    assert any("paging limit" in warning for warning in payload["warnings"])


def test_an_empty_window_ends_paging():
    wrapper = _make_ado(FakeResponse(results=[], count=500))

    payload = json.loads(wrapper.search_code("q", top=10, skip=100, include_snippets=False))

    assert payload["returned"] == 0
    assert payload["truncated"] is True
    assert "next_skip" not in payload


def test_a_partial_final_page_reports_no_further_window():
    results = [FakeResult(path=f"/f{i}.py") for i in range(3)]
    wrapper = _make_ado(FakeResponse(results=results, count=13))

    payload = json.loads(wrapper.search_code("q", top=5, skip=10, include_snippets=False))

    assert payload["returned"] == 3
    assert payload["truncated"] is False
    assert "next_skip" not in payload


def test_results_carry_metadata_and_never_full_bodies():
    wrapper = _make_ado(
        FakeResponse(results=[FakeResult(hits=[FakeHit(char_offset=1), FakeHit(char_offset=2)])], count=1)
    )

    payload = json.loads(wrapper.search_code("q", include_snippets=False))
    entry = payload["results"][0]

    assert entry == {
        "project": "proj",
        "repository": "MyRepo",
        "path": "/src/main.py",
        "file_name": "main.py",
        "branch": "main",
        "match_count": 2,
    }


def test_info_code_surfaces_as_warning():
    wrapper = _make_ado(FakeResponse(results=[FakeResult()], count=1, info_code=9))

    payload = json.loads(wrapper.search_code("q", include_snippets=False))

    assert any("Branches are still being indexed" in w for w in payload["warnings"])


def test_unindexed_filter_info_code_is_explained():
    wrapper = _make_ado(FakeResponse(results=[FakeResult()], count=1, info_code=15))

    payload = json.loads(
        wrapper.search_code("q", branch="feature/never-indexed", include_snippets=False)
    )

    assert any("Searchable branches" in w for w in payload["warnings"])
    assert not any("info code 15" in w for w in payload["warnings"])


def test_unknown_info_code_still_reported():
    wrapper = _make_ado(FakeResponse(results=[FakeResult()], count=1, info_code=77))

    payload = json.loads(wrapper.search_code("q", include_snippets=False))

    assert any("info code 77" in w for w in payload["warnings"])


def test_empty_results_explain_indexing_coverage():
    wrapper = _make_ado(FakeResponse(results=[], count=0))

    payload = json.loads(wrapper.search_code("nothing"))

    assert payload["results"] == []
    assert any("Searchable branches" in w for w in payload["warnings"])


def test_permission_trimmed_window_is_not_blamed_on_branch_indexing():
    wrapper = _make_ado(FakeResponse(results=[], count=500, info_code=11))

    payload = json.loads(wrapper.search_code("q", include_snippets=False))

    assert payload["returned"] == 0
    assert not any("Searchable branches" in w for w in payload["warnings"])
    assert any("read permission per repository" in w for w in payload["warnings"])


def test_snippets_reconstructed_from_char_offsets():
    content = "\n".join(f"line{i}" for i in range(1, 11))
    offset_of_line_5 = sum(len(f"line{i}\n") for i in range(1, 5))
    wrapper = _make_ado(
        FakeResponse(results=[FakeResult(hits=[FakeHit(char_offset=offset_of_line_5)])], count=1),
        contents_by_path={"/src/main.py": content},
    )

    payload = json.loads(wrapper.search_code("line5"))
    snippets = payload["results"][0]["snippets"]

    assert snippets == [{"line": 5, "snippet": "line3\nline4\nline5\nline6\nline7"}]


def test_inline_code_snippet_is_used_without_fetching_the_file():
    wrapper = _make_ado(
        FakeResponse(results=[FakeResult(hits=[FakeHit(line=12, code_snippet="  return total")])], count=1)
    )

    payload = json.loads(wrapper.search_code("total"))

    assert payload["results"][0]["snippets"] == [{"line": 12, "snippet": "  return total"}]
    assert wrapper.ado_client_instance.fetch_count == 0


def test_hits_on_the_same_line_are_collapsed():
    content = "alpha alpha alpha\nbeta\n"
    wrapper = _make_ado(
        FakeResponse(
            results=[FakeResult(hits=[FakeHit(char_offset=0), FakeHit(char_offset=6), FakeHit(char_offset=12)])],
            count=1,
        ),
        contents_by_path={"/src/main.py": content},
    )

    payload = json.loads(wrapper.search_code("alpha"))

    assert payload["results"][0]["snippets"] == [{"line": 1, "snippet": "alpha alpha alpha\nbeta"}]


def test_include_snippets_false_skips_file_fetches():
    wrapper = _make_ado(
        FakeResponse(results=[FakeResult(hits=[FakeHit(char_offset=0)])], count=1),
        contents_by_path={"/src/main.py": "anything"},
    )

    payload = json.loads(wrapper.search_code("q", include_snippets=False))

    assert "snippets" not in payload["results"][0]
    assert wrapper.ado_client_instance.fetch_count == 0


def test_snippet_file_fetches_are_capped():
    count = MAX_FILES_FETCHED_FOR_SNIPPETS + 3
    results = [FakeResult(path=f"/f{i}.py", hits=[FakeHit(char_offset=0)], change_id=f"sha{i}") for i in range(count)]
    contents = {f"/f{i}.py": "hello" for i in range(count)}
    wrapper = _make_ado(FakeResponse(results=results, count=count), contents_by_path=contents)

    payload = json.loads(wrapper.search_code("hello", top=count))

    assert wrapper.ado_client_instance.fetch_count == MAX_FILES_FETCHED_FOR_SNIPPETS
    assert "snippets" not in payload["results"][-1]
    assert any("3 further result(s) list metadata alone" in w for w in payload["warnings"])


def test_exactly_at_the_fetch_cap_does_not_warn_about_dropped_snippets():
    count = MAX_FILES_FETCHED_FOR_SNIPPETS
    results = [FakeResult(path=f"/f{i}.py", hits=[FakeHit(char_offset=0)], change_id=f"sha{i}") for i in range(count)]
    contents = {f"/f{i}.py": "hello" for i in range(count)}
    wrapper = _make_ado(FakeResponse(results=results, count=count), contents_by_path=contents)

    payload = json.loads(wrapper.search_code("hello", top=count))

    assert all("snippets" in entry for entry in payload["results"])
    assert "warnings" not in payload


def test_concurrent_fetches_keep_each_snippet_with_its_own_result():
    count = MAX_FILES_FETCHED_FOR_SNIPPETS
    results = [FakeResult(path=f"/f{i}.py", hits=[FakeHit(char_offset=0)], change_id=f"sha{i}") for i in range(count)]
    contents = {f"/f{i}.py": f"marker{i}\n" for i in range(count)}
    wrapper = _make_ado(FakeResponse(results=results, count=count), contents_by_path=contents)

    payload = json.loads(wrapper.search_code("marker", top=count))

    for i, entry in enumerate(payload["results"]):
        assert entry["path"] == f"/f{i}.py"
        assert entry["snippets"][0]["snippet"] == f"marker{i}"


def test_one_unreadable_file_does_not_lose_snippets_for_the_others():
    class OneBadFileClient(FakeAdoClient):
        def get_item_text(self, repository_id, project, path, version_descriptor):
            if path == "/f1.py":
                raise RuntimeError("TF401019: item not found")
            return super().get_item_text(repository_id, project, path, version_descriptor)

    results = [FakeResult(path=f"/f{i}.py", hits=[FakeHit(char_offset=0)], change_id=f"sha{i}") for i in range(3)]
    contents = {f"/f{i}.py": f"marker{i}\n" for i in range(3)}
    wrapper = _make_ado(FakeResponse(results=results, count=3), contents_by_path=contents)
    wrapper.ado_client_instance = OneBadFileClient(contents)

    entries = json.loads(wrapper.search_code("marker", top=3))["results"]

    assert entries[0]["snippets"][0]["snippet"] == "marker0"
    assert "snippets" not in entries[1]
    assert entries[2]["snippets"][0]["snippet"] == "marker2"


def test_body_failing_midstream_does_not_abort_the_whole_search():
    class MidStreamFailureClient(FakeAdoClient):
        def get_item_text(self, repository_id, project, path, version_descriptor):
            if path != "/f1.py":
                return super().get_item_text(repository_id, project, path, version_descriptor)

            def failing_body():
                yield b"marker1\n"
                raise ConnectionError("chunked read failed mid-body")

            return failing_body()

    results = [FakeResult(path=f"/f{i}.py", hits=[FakeHit(char_offset=0)], change_id=f"sha{i}") for i in range(3)]
    contents = {f"/f{i}.py": f"marker{i}\n" for i in range(3)}
    wrapper = _make_ado(FakeResponse(results=results, count=3), contents_by_path=contents)
    wrapper.ado_client_instance = MidStreamFailureClient(contents)

    entries = json.loads(wrapper.search_code("marker", top=3))["results"]

    assert entries[0]["snippets"][0]["snippet"] == "marker0"
    assert "snippets" not in entries[1]
    assert entries[2]["snippets"][0]["snippet"] == "marker2"


def test_early_exit_closes_the_response_stream():
    closed = []

    class ClosableStreamClient(FakeAdoClient):
        def get_item_text(self, repository_id, project, path, version_descriptor):
            self.fetch_count += 1

            class Stream:
                def __iter__(inner):
                    while True:
                        yield b"x" * 1000

                def close(inner):
                    closed.append(path)

            return Stream()

    wrapper = _make_ado(FakeResponse(results=[FakeResult(hits=[FakeHit(char_offset=0)])], count=1))
    wrapper.ado_client_instance = ClosableStreamClient()

    wrapper.search_code("q")

    assert closed == ["/src/main.py"]


def test_a_failure_closing_the_stream_does_not_discard_the_search():
    class UnclosableStreamClient(FakeAdoClient):
        def get_item_text(self, repository_id, project, path, version_descriptor):
            self.fetch_count += 1

            class Stream:
                def __iter__(inner):
                    yield b"alpha\nbeta\n"

                def close(inner):
                    raise RuntimeError("connection already released")

            return Stream()

    wrapper = _make_ado(FakeResponse(results=[FakeResult(hits=[FakeHit(char_offset=0)])], count=1))
    wrapper.ado_client_instance = UnclosableStreamClient()

    entry = json.loads(wrapper.search_code("alpha"))["results"][0]

    assert entry["snippets"][0]["snippet"] == "alpha\nbeta"


def test_char_offset_past_end_of_file_is_skipped():
    wrapper = _make_ado(
        FakeResponse(results=[FakeResult(hits=[FakeHit(char_offset=10_000)])], count=1),
        contents_by_path={"/src/main.py": "one\ntwo\n"},
    )

    entry = json.loads(wrapper.search_code("q"))["results"][0]

    assert "snippets" not in entry


def test_snippet_read_stops_before_consuming_the_whole_file():
    class ChunkedClient(FakeAdoClient):
        def __init__(self):
            super().__init__()
            self.chunks_consumed = 0

        def get_item_text(self, repository_id, project, path, version_descriptor):
            self.fetch_count += 1

            def chunks():
                for _ in range(1000):
                    self.chunks_consumed += 1
                    yield (b"x" * 1000)

            return chunks()

    wrapper = _make_ado(FakeResponse(results=[FakeResult(hits=[FakeHit(char_offset=0)])], count=1))
    wrapper.ado_client_instance = ChunkedClient()

    wrapper.search_code("q")

    assert wrapper.ado_client_instance.chunks_consumed < 1000


def test_snippet_reads_never_populate_the_shared_file_content_cache():
    wrapper = _make_ado(
        FakeResponse(results=[FakeResult(hits=[FakeHit(char_offset=0)])], count=1),
        contents_by_path={"/src/main.py": "alpha\nbeta\n"},
    )

    wrapper.search_code("alpha")

    assert wrapper._file_content_cache == {}


def test_file_name_only_match_explains_the_absent_snippet():
    result = FakeResult(path="/test.ts", file_name="test.ts")
    result.matches = {"fileName": [FakeHit(char_offset=0)]}
    wrapper = _make_ado(FakeResponse(results=[result], count=1))

    payload = json.loads(wrapper.search_code("test"))
    entry = payload["results"][0]

    assert entry["match_count"] == 0
    assert entry["matched_on"] == ["fileName"]
    assert "snippets" not in entry


def test_unreadable_file_yields_no_snippet_rather_than_an_error_payload():
    class ExplodingReadClient(FakeAdoClient):
        def get_item_text(self, repository_id, project, path, version_descriptor):
            self.fetch_count += 1
            raise RuntimeError("TF401019: item not found")

    wrapper = _make_ado(
        FakeResponse(results=[FakeResult(hits=[FakeHit(char_offset=0)])], count=1)
    )
    wrapper.ado_client_instance = ExplodingReadClient()

    entry = json.loads(wrapper.search_code("q"))["results"][0]

    assert "snippets" not in entry
    assert entry["match_count"] == 1


def test_empty_content_match_area_is_not_reported_as_matched():
    result = FakeResult(path="/test.ts", file_name="test.ts")
    result.matches = {"content": [], "fileName": [FakeHit(char_offset=0)]}
    wrapper = _make_ado(FakeResponse(results=[result], count=1))

    entry = json.loads(wrapper.search_code("test"))["results"][0]

    assert entry["matched_on"] == ["fileName"]


def test_search_failure_returns_actionable_tool_exception():
    class ExplodingSearchClient:
        def fetch_code_search_results(self, request, project=None):
            raise RuntimeError("VS403430: no access")

    wrapper = _make_ado(FakeResponse())
    wrapper._search_client_instance = ExplodingSearchClient()

    result = wrapper.search_code("q")

    assert isinstance(result, ToolException)
    assert "Code (read) scope" in str(result)


def test_search_code_is_registered_as_a_tool():
    names = [tool["name"] for tool in ReposApiWrapper.model_construct().get_available_tools()]

    assert "search_code" in names
