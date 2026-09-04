"""Tests for Azure DevOps Boards search_work_items_by_text.

Covers the bounded-response contract the tool promises to callers:
  * Defaults stay small and never return full work item bodies.
  * Field reads are casing-tolerant - the search service returns `system.id`
    style keys while WIQL returns `System.Id` style.
  * total_count/truncated/next_skip describe the result set beyond the window
    that was requested, and following next_skip always terminates.
  * Highlights are opt-in, and bounded per result, per response and per character.
  * Azure DevOps infoCode values surface as human-readable warnings.
"""

import inspect
import json

import pytest
from langchain_core.tools import ToolException
from pydantic import SecretStr

from elitea_sdk.tools.ado.work_item import ado_wrapper as wrapper_module
from elitea_sdk.tools.ado.work_item.ado_wrapper import (
    ADOWorkItemsTextSearch,
    AzureDevOpsApiWrapper,
    HIGHLIGHTS,
    PAGING,
)
from elitea_sdk.tools.ado import utils as ado_utils
from elitea_sdk.tools.ado.utils import SEARCH_INFO_CODES
from elitea_sdk.tools.utils import get_max_toolkit_length

MATCHES_HIDDEN_INFO_CODE = next(
    code.number for code in SEARCH_INFO_CODES.values() if code.matches_hidden_by_permissions
)

CANONICAL_FIELDS = {
    "system.id": "2",
    "system.title": "Rest Api User Story",
    "system.workitemtype": "User Story",
    "system.state": "Closed",
    "system.assignedto": "John Doe <jodoe@contoso.com>",
}


class FakeNamed:
    def __init__(self, name):
        self.name = name


class FakeHit:
    def __init__(self, field_reference_name, highlights):
        self.field_reference_name = field_reference_name
        self.highlights = highlights


class FakeResult:
    def __init__(self, fields=None, hits=None, project=None, url="https://dev.azure.com/org/_apis/wit/workItems/2"):
        self.fields = dict(CANONICAL_FIELDS) if fields is None else fields
        self.hits = hits or []
        self.project = FakeNamed("MyProject") if project is None else project
        self.url = url


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
        self.call_count = 0

    def fetch_work_item_search_results(self, request, project=None):
        self.call_count += 1
        self.last_request = request
        self.last_project = project
        return self._response


class PagingSearchClient:
    """Answers every window with the same count and the same number of rows."""

    def __init__(self, count, page_size):
        self._count = count
        self._page_size = page_size
        self.call_count = 0

    def fetch_work_item_search_results(self, request, project=None):
        self.call_count += 1
        results = [FakeResult() for _ in range(self._page_size)]
        return FakeResponse(
            results=results,
            count=self._count,
            info_code=0 if self._page_size else MATCHES_HIDDEN_INFO_CODE,
        )


class FailingSearchClient:
    def __init__(self, exception):
        self._exception = exception

    def fetch_work_item_search_results(self, request, project=None):
        raise self._exception


def build_wrapper(search_client, project="MyProject", limit=5):
    wrapper = AzureDevOpsApiWrapper.model_construct(
        organization_url="https://dev.azure.com/org",
        project=project,
        token=SecretStr("token"),
        limit=limit,
    )
    wrapper._search_client_instance = search_client
    return wrapper


def search(search_client, **kwargs):
    kwargs.setdefault("query", "login timeout")
    return json.loads(build_wrapper(search_client).search_work_items_by_text(**kwargs))


@pytest.mark.parametrize("blank_query", ["", "   "])
def test_empty_query_returns_tool_exception(blank_query):
    client = FakeSearchClient(FakeResponse())

    with pytest.raises(ToolException, match="cannot be empty"):
        build_wrapper(client).search_work_items_by_text(blank_query)

    assert client.call_count == 0


def test_search_is_scoped_to_the_configured_project():
    client = FakeSearchClient(FakeResponse())

    search(client)

    assert client.last_project == "MyProject"
    assert client.last_request.filters == {"System.TeamProject": ["MyProject"]}


def test_optional_filters_map_to_field_reference_names():
    client = FakeSearchClient(FakeResponse())

    search(
        client,
        work_item_type=["Bug"],
        state=["Active"],
        assigned_to=["a@b.c"],
        area_path=["MyProject\\Backend"],
    )

    assert client.last_request.filters == {
        "System.TeamProject": ["MyProject"],
        "System.WorkItemType": ["Bug"],
        "System.State": ["Active"],
        "System.AssignedTo": ["a@b.c"],
        "System.AreaPath": ["MyProject\\Backend"],
    }


def test_a_single_string_filter_value_is_accepted():
    client = FakeSearchClient(FakeResponse())

    search(client, work_item_type="Bug")

    assert client.last_request.filters == {
        "System.TeamProject": ["MyProject"],
        "System.WorkItemType": ["Bug"],
    }


def test_defaults_request_five_results():
    client = FakeSearchClient(FakeResponse())

    search(client)

    assert client.last_request.top == PAGING.default_top == 5
    assert client.last_request.skip == 0


@pytest.mark.parametrize(
    "top, skip, expected_top, expected_skip",
    [
        (None, None, PAGING.default_top, 0),
        (0, 0, PAGING.default_top, 0),
        (10_000, 10_000, PAGING.max_top, PAGING.max_skip),
        (-3, -3, 1, 0),
    ],
)
def test_top_and_skip_are_clamped_to_service_limits(top, skip, expected_top, expected_skip):
    client = FakeSearchClient(FakeResponse())

    search(client, top=top, skip=skip)

    assert client.last_request.top == expected_top
    assert client.last_request.skip == expected_skip


def test_top_is_capped_below_the_payload_ceiling():
    assert PAGING.max_top == 50

    top_field = ADOWorkItemsTextSearch.model_fields["top"]
    upper_bounds = [getattr(item, "le", None) for item in top_field.metadata]

    assert PAGING.max_top in upper_bounds


def test_the_signature_accepts_everything_the_schema_admits():
    """The schema decides what reaches the method, so a parameter annotated more narrowly
    than its field is a claim the tool does not honour."""
    signature = inspect.signature(AzureDevOpsApiWrapper.search_work_items_by_text)

    mismatched = {
        name: (signature.parameters[name].annotation, field.annotation)
        for name, field in ADOWorkItemsTextSearch.model_fields.items()
        if signature.parameters[name].annotation != field.annotation
    }

    assert mismatched == {}


@pytest.mark.parametrize("field", ["skip", "include_highlights", "top"])
def test_a_null_optional_matches_the_twin_tools_tolerance(field):
    """search_code accepts an explicit null on these; a model emitting one should not get a
    validation error from one ADO search tool and a result from the other."""
    assert ADOWorkItemsTextSearch(query="login", **{field: None}) is not None


def test_highlights_are_off_until_asked_for():
    hits = [FakeHit("system.title", ["match"])]
    response = FakeResponse(results=[FakeResult(hits=hits)], count=1)

    assert "highlights" not in search(FakeSearchClient(response))["results"][0]
    assert "highlights" not in search(FakeSearchClient(response), include_highlights=None)["results"][0]
    assert "highlights" in search(FakeSearchClient(response), include_highlights=True)["results"][0]


def test_every_paging_instruction_carries_the_empty_window_stop_rule():
    """A model reads the docstring and the skip description on every call and the runtime
    warning only after an empty window, so an unbounded instruction in either of the first
    two outlives the warning and walks the cursor to the ceiling."""
    tool = next(
        tool for tool in build_wrapper(None).get_available_tools()
        if tool["name"] == "search_work_items_by_text"
    )
    payload = search(FakeSearchClient(FakeResponse(count=0)))
    surfaces = [
        tool["description"],
        ADOWorkItemsTextSearch.model_fields["skip"].description,
        " ".join(payload["warnings"]),
    ]

    assert all("empty" in surface for surface in surfaces)


def test_the_no_results_hint_tells_a_cursorless_response_to_refine():
    payload = search(FakeSearchClient(FakeResponse(count=0)))

    hint = next(warning for warning in payload["warnings"] if "No matches in this window" in warning)
    assert "next_skip" not in payload
    assert "refine the query when it does not come back" in hint


@pytest.mark.parametrize(
    "rows, count, top, skip, info_code, truncated, next_skip",
    [
        (5, 42, 5, 0, 0, True, 5),
        (3, 42, 5, 0, 0, True, 5),
        (5, 500, 5, 10, 0, True, 15),
        (2, 7, 5, 5, 0, False, None),
        (3, 5, 5, 0, MATCHES_HIDDEN_INFO_CODE, False, None),
        (0, 42, 5, 100, 0, False, None),
        (0, 42, 5, 0, MATCHES_HIDDEN_INFO_CODE, True, PAGING.empty_window_stride),
        (0, 5000, 5, 0, 0, True, None),
    ],
)
def test_the_paging_contract(rows, count, top, skip, info_code, truncated, next_skip):
    """A cursor is offered only when the service counted matches beyond the requested window
    and that window is worth continuing from. It advances by top, or by the stride when
    permission trimming emptied the window, and is omitted rather than returned as null."""
    response = FakeResponse(
        results=[FakeResult() for _ in range(rows)], count=count, info_code=info_code
    )

    payload = search(FakeSearchClient(response), top=top, skip=skip)

    assert payload["total_count"] == count
    assert payload["returned"] == rows
    assert payload["skip"] == skip
    assert payload["truncated"] is truncated
    assert payload.get("next_skip") == next_skip
    assert ("next_skip" in payload) is (next_skip is not None)


def test_the_stride_clears_the_paging_ceiling_in_at_most_twenty_calls():
    assert PAGING.max_skip / PAGING.empty_window_stride <= 20


def test_a_null_skip_pages_from_the_start():
    client = FakeSearchClient(FakeResponse())

    search(client, skip=None)

    assert client.last_request.skip == 0


@pytest.mark.parametrize("configured_limit", [1, 5, 20, 200, -1, 0])
def test_the_search_default_is_independent_of_the_wiql_row_limit(configured_limit):
    client = FakeSearchClient(FakeResponse())

    build_wrapper(client, limit=configured_limit).search_work_items_by_text("login timeout")

    assert client.last_request.top == PAGING.default_top


def test_results_carry_the_summary_fields():
    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult()], count=1)))

    entry = payload["results"][0]
    assert entry["id"] == 2
    assert entry["title"] == "Rest Api User Story"
    assert entry["type"] == "User Story"
    assert entry["state"] == "Closed"
    assert entry["project"] == "MyProject"
    assert entry["assigned_to"] == "John Doe <jodoe@contoso.com>"
    assert not {"description", "body", "fields", "relations", "comments"} & set(entry)


def test_pascal_case_field_names_are_read_too():
    pascal_case_fields = {
        "System.Id": "2",
        "System.Title": "Rest Api User Story",
        "System.WorkItemType": "User Story",
        "System.State": "Closed",
        "System.AssignedTo": "John Doe <jodoe@contoso.com>",
    }

    lowercased = search(FakeSearchClient(FakeResponse(results=[FakeResult()], count=1)))
    pascal = search(FakeSearchClient(FakeResponse(results=[FakeResult(fields=pascal_case_fields)], count=1)))

    assert pascal["results"] == lowercased["results"]


@pytest.mark.parametrize("assignee", [None, ""])
def test_missing_assignee_is_omitted(assignee):
    fields = {key: value for key, value in CANONICAL_FIELDS.items() if key != "system.assignedto"}
    if assignee is not None:
        fields["system.assignedto"] = assignee

    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult(fields=fields)], count=1)))

    assert "assigned_to" not in payload["results"][0]


def test_non_numeric_id_is_passed_through():
    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult(fields={"system.id": "abc"})], count=1)))

    assert payload["results"][0]["id"] == "abc"


def test_url_points_at_the_board_item():
    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult()], count=1)))

    assert payload["results"][0]["url"] == "https://dev.azure.com/org/_workitems/edit/2"


def test_url_falls_back_to_the_rest_url_without_an_id():
    result = FakeResult(fields={"system.title": "No id"}, url="https://dev.azure.com/org/_apis/wit/workItems/9")

    payload = search(FakeSearchClient(FakeResponse(results=[result], count=1)))

    assert payload["results"][0]["url"] == "https://dev.azure.com/org/_apis/wit/workItems/9"


def test_project_falls_back_to_the_configured_project():
    result = FakeResult(project=False)

    payload = search(FakeSearchClient(FakeResponse(results=[result], count=1)))

    assert payload["results"][0]["project"] == "MyProject"


@pytest.mark.parametrize("page_size", [0, 5])
def test_the_paging_ceiling_withholds_a_cursor(page_size):
    response = FakeResponse(
        results=[FakeResult() for _ in range(page_size)],
        count=5000,
        info_code=0 if page_size else MATCHES_HIDDEN_INFO_CODE,
    )

    payload = search(FakeSearchClient(response), top=5, skip=PAGING.max_skip)

    assert payload["truncated"] is True
    assert "next_skip" not in payload
    assert any("paging limit" in warning for warning in payload["warnings"])


@pytest.mark.parametrize(
    "count, top, page_size",
    [(42, 5, 0), (42, 5, 3), (5000, 5, 0), (5000, 50, 50), (42, 1, 0)],
)
def test_following_the_cursor_always_terminates(count, top, page_size):
    client = PagingSearchClient(count, page_size)
    wrapper = build_wrapper(client)
    max_iterations = PAGING.max_skip // top + 2

    skip = 0
    iterations = 0
    while True:
        iterations += 1
        assert iterations <= max_iterations
        payload = json.loads(wrapper.search_work_items_by_text("login", top=top, skip=skip))
        next_skip = payload.get("next_skip")
        if next_skip is None:
            break
        assert isinstance(next_skip, int)
        assert next_skip > skip
        skip = next_skip


def test_the_no_results_hint_does_not_contradict_a_supplied_cursor():
    payload = search(FakeSearchClient(FakeResponse(count=42, info_code=MATCHES_HIDDEN_INFO_CODE)), top=5)

    hint = next(warning for warning in payload["warnings"] if "No matches in this window" in warning)
    assert payload["next_skip"] == PAGING.empty_window_stride
    assert "rather than paging further" not in hint
    assert "next_skip" in hint


def test_highlights_are_capped_per_result():
    hits = [FakeHit(f"system.field{index}", [f"match {index}"]) for index in range(5)]

    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult(hits=hits)], count=1)), include_highlights=True)

    highlights = payload["results"][0]["highlights"]
    assert len(highlights) == HIGHLIGHTS.max_per_result
    assert all(set(highlight) == {"field", "text"} for highlight in highlights)


def test_identical_field_and_text_hits_collapse_to_one():
    tags = "automation; faker; test -data; testing"
    hits = [FakeHit("system.tags", [tags]), FakeHit("system.tags", [tags])]

    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult(hits=hits)], count=1)), include_highlights=True)

    assert payload["results"][0]["highlights"] == [{"field": "system.tags", "text": tags}]


def test_a_duplicate_does_not_crowd_out_a_distinct_field():
    tags = "automation; faker; testing"
    hits = [
        FakeHit("system.tags", [tags]),
        FakeHit("system.tags", [tags]),
        FakeHit("system.title", ["login timeout"]),
        FakeHit("system.description", ["login timeout after 30s"]),
    ]

    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult(hits=hits)], count=1)), include_highlights=True)

    highlights = payload["results"][0]["highlights"]
    assert [highlight["field"] for highlight in highlights] == [
        "system.tags",
        "system.title",
        "system.description",
    ]


def test_the_same_field_with_different_text_is_kept():
    hits = [
        FakeHit("system.description", ["login timeout after 30s"]),
        FakeHit("system.description", ["the login page also times out"]),
    ]

    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult(hits=hits)], count=1)), include_highlights=True)

    highlights = payload["results"][0]["highlights"]
    assert [highlight["text"] for highlight in highlights] == [
        "login timeout after 30s",
        "the login page also times out",
    ]


def test_duplicates_are_detected_after_markup_is_stripped():
    hits = [
        FakeHit("system.title", ["<highlighthit>login</highlighthit> timeout"]),
        FakeHit("system.title", ["login <highlighthit>timeout</highlighthit>"]),
    ]

    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult(hits=hits)], count=1)), include_highlights=True)

    assert payload["results"][0]["highlights"] == [{"field": "system.title", "text": "login timeout"}]


def test_a_duplicate_still_spends_one_result_of_the_highlight_budget():
    duplicated = [FakeHit("system.tags", ["testing"]) for _ in range(3)]
    results = [FakeResult(hits=list(duplicated)) for _ in range(6)]

    payload = search(FakeSearchClient(FakeResponse(results=results, count=6)), include_highlights=True, top=6)

    highlighted = [entry for entry in payload["results"] if "highlights" in entry]
    assert len(highlighted) == HIGHLIGHTS.max_results
    assert any("1 further result(s)" in warning for warning in payload["warnings"])


def test_the_number_of_hits_parsed_is_bounded(monkeypatch):
    parses = []
    real_soup = wrapper_module.BeautifulSoup
    monkeypatch.setattr(
        wrapper_module,
        "BeautifulSoup",
        lambda markup, parser: parses.append(markup) or real_soup(markup, parser),
    )
    hits = [FakeHit("system.history", ["same excerpt"]) for _ in range(200)]

    search(FakeSearchClient(FakeResponse(results=[FakeResult(hits=hits)], count=1)), include_highlights=True)

    assert len(parses) == HIGHLIGHTS.max_hits_scanned


def test_a_zero_per_result_budget_yields_no_highlights(monkeypatch):
    monkeypatch.setattr(
        wrapper_module,
        "HIGHLIGHTS",
        wrapper_module.HighlightBudget(max_per_result=0),
    )
    hits = [FakeHit(f"system.field{index}", [f"match {index}"]) for index in range(5)]

    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult(hits=hits)], count=1)), include_highlights=True)

    assert payload["results"][0]["highlights"] == []


def test_populated_highlights_survive_leading_empty_hits():
    hits = [
        FakeHit("system.id", []),
        FakeHit("system.rev", []),
        FakeHit("system.createdby", []),
        FakeHit("system.title", ["<highlighthit>login</highlighthit> timeout"]),
        FakeHit("system.description", ["login timeout after 30s"]),
    ]

    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult(hits=hits)], count=1)), include_highlights=True)

    highlights = payload["results"][0]["highlights"]
    assert [highlight["field"] for highlight in highlights] == ["system.title", "system.description"]


def test_highlight_markup_is_stripped_and_truncated():
    hits = [
        FakeHit("system.title", ["<highlighthit>Rest</highlighthit> Api"]),
        FakeHit("system.description", ["x" * 500]),
    ]

    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult(hits=hits)], count=1)), include_highlights=True)

    highlights = payload["results"][0]["highlights"]
    assert highlights[0]["text"] == "Rest Api"
    assert len(highlights[1]["text"]) == HIGHLIGHTS.max_chars


def test_field_html_in_highlights_is_flattened_to_text():
    body = "<div><b>Payment</b> gateway</div><p>times out after <em>30s</em></p>"
    hits = [FakeHit("system.description", [f"<highlighthit>{body}</highlighthit>"])]

    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult(hits=hits)], count=1)), include_highlights=True)

    text = payload["results"][0]["highlights"][0]["text"]
    assert text == "Payment gateway times out after 30s"
    assert "<" not in text


def test_the_highlight_budget_warning_counts_only_withheld_highlights():
    with_highlights = [FakeResult(hits=[FakeHit("system.title", ["match"])]) for _ in range(7)]
    without_highlights = [FakeResult(hits=[FakeHit("system.id", [])]) for _ in range(3)]
    response = FakeResponse(results=with_highlights + without_highlights, count=10)

    payload = search(FakeSearchClient(response), include_highlights=True, top=10)

    assert any("2 further result(s)" in warning for warning in payload["warnings"])


def test_the_highlight_budget_is_spent_on_results_that_have_highlights():
    """Counting results seen rather than results given highlights lets a run of hit-less
    results exhaust the budget before the first highlight is attached."""
    without_highlights = [FakeResult(hits=[]) for _ in range(5)]
    with_highlights = [FakeResult(hits=[FakeHit("system.title", ["match"])]) for _ in range(5)]
    response = FakeResponse(results=without_highlights + with_highlights, count=10)

    payload = search(FakeSearchClient(response), include_highlights=True, top=10)

    highlighted = [entry for entry in payload["results"] if "highlights" in entry]
    assert len(highlighted) == HIGHLIGHTS.max_results
    assert all("further result(s)" not in warning for warning in payload.get("warnings", []))


def test_no_highlight_budget_warning_when_the_extra_results_never_matched_a_field():
    with_highlights = [FakeResult(hits=[FakeHit("system.title", ["match"])]) for _ in range(5)]
    without_highlights = [FakeResult(hits=[]) for _ in range(5)]
    response = FakeResponse(results=with_highlights + without_highlights, count=10)

    payload = search(FakeSearchClient(response), include_highlights=True, top=10)

    assert all("further result(s)" not in warning for warning in payload.get("warnings", []))


def test_hits_without_highlights_are_skipped():
    hits = [FakeHit("system.id", []), FakeHit("system.rev", [])]

    payload = search(FakeSearchClient(FakeResponse(results=[FakeResult(hits=hits)], count=1)), include_highlights=True)

    assert "highlights" not in payload["results"][0]


def test_include_highlights_false_omits_highlights():
    results = [FakeResult(hits=[FakeHit("system.title", ["match"])]) for _ in range(10)]

    payload = search(
        FakeSearchClient(FakeResponse(results=results, count=10)),
        top=10,
        include_highlights=False,
    )

    assert all("highlights" not in entry for entry in payload["results"])
    assert "warnings" not in payload or all("Highlights were attached" not in w for w in payload["warnings"])


def test_highlights_are_budgeted_across_results():
    results = [FakeResult(hits=[FakeHit("system.title", ["match"])]) for _ in range(10)]

    payload = search(FakeSearchClient(FakeResponse(results=results, count=10)), include_highlights=True, top=10)

    entries = payload["results"]
    assert all("highlights" in entry for entry in entries[:HIGHLIGHTS.max_results])
    assert all("highlights" not in entry for entry in entries[HIGHLIGHTS.max_results:])
    assert any(
        f"first {HIGHLIGHTS.max_results} result(s)" in warning and "5 further result(s)" in warning
        for warning in payload["warnings"]
    )


def test_worst_case_payload_stays_bounded():
    fields = {
        "system.id": "123456",
        "system.title": "T" * 255,
        "system.workitemtype": "User Story",
        "system.state": "Active",
        "system.assignedto": "A" * 60,
    }
    hits = [FakeHit(f"system.field{index}", ["x" * 500]) for index in range(3)]
    results = [FakeResult(fields=dict(fields), hits=hits) for _ in range(PAGING.max_top)]

    raw = build_wrapper(FakeSearchClient(FakeResponse(results=results, count=PAGING.max_top))).search_work_items_by_text(
        "login", top=PAGING.max_top, include_highlights=True
    )

    assert len(raw) < 40_000


@pytest.mark.parametrize(
    "info_code, expected_fragment",
    [
        (1, "being reindexed"),
        (MATCHES_HIDDEN_INFO_CODE, "not readable with the current token"),
        (15, "work item type, state, assignee or area path filtered on exists"),
        (8, "Azure DevOps chose the window size"),
        (77, "Azure DevOps returned info code 77."),
    ],
)
def test_info_codes_surface_as_warnings(info_code, expected_fragment):
    response = FakeResponse(results=[FakeResult()], count=1, info_code=info_code)

    payload = search(FakeSearchClient(response))

    assert any(expected_fragment in warning for warning in payload["warnings"])


def test_empty_results_explain_search_coverage():
    payload = search(FakeSearchClient(FakeResponse(count=0)))

    assert payload["returned"] == 0
    assert payload["truncated"] is False
    assert "next_skip" not in payload
    assert any("No matches in this window" in warning for warning in payload["warnings"])


def test_search_failure_returns_actionable_tool_exception():
    wrapper = build_wrapper(FailingSearchClient(Exception("boom")))

    with pytest.raises(ToolException) as exc_info:
        wrapper.search_work_items_by_text("login timeout")

    message = str(exc_info.value)
    assert "login timeout" in message
    assert "boom" in message
    assert "Work Items (read)" in message
    assert "Search extension" in message


def test_search_client_is_cached_per_instance(monkeypatch):
    constructions = []

    class FakeClientsV71:
        def get_search_client(self):
            client = object()
            constructions.append(client)
            return client

    class FakeConnection:
        def __init__(self, base_url=None, creds=None):
            self.clients_v7_1 = FakeClientsV71()

    monkeypatch.setattr(ado_utils, "Connection", FakeConnection)

    first = build_wrapper(None)
    second = build_wrapper(None)

    assert first._search_client is first._search_client
    assert first._search_client is not second._search_client
    assert len(constructions) == 2


def test_search_work_items_by_text_is_registered_as_a_tool():
    tools = build_wrapper(None).get_available_tools()
    by_name = {tool["name"]: tool for tool in tools}

    assert by_name["search_work_items_by_text"]["args_schema"] is ADOWorkItemsTextSearch
    assert by_name["search_work_items"]["args_schema"] is wrapper_module.ADOWorkItemsSearch


def test_tool_descriptions_keep_their_toolkit_suffix_under_the_longest_possible_suffixes():
    """An agent routes between several attached ADO toolkits on the toolkit suffix, and
    truncation drops it from the end first."""
    by_name = {tool["name"]: tool for tool in build_wrapper(None).get_available_tools()}
    toolkit_suffix = "\nToolkit: " + "T" * get_max_toolkit_length(None)
    instance_suffix = f"\nADO instance: https://dev.azure.com/{'o' * 50}/{'p' * 64}"

    for name in ("search_work_items", "search_work_items_by_text"):
        description = by_name[name]["description"] + instance_suffix + toolkit_suffix

        assert description[:1000].endswith(toolkit_suffix)


def test_available_tools_build_without_credentials():
    tools = AzureDevOpsApiWrapper.model_construct().get_available_tools()

    assert "search_work_items_by_text" in [tool["name"] for tool in tools]
