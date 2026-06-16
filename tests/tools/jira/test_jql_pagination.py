"""
Tests for `_jql_get_tickets` pagination.

Regression coverage for the bug where indexing capped at 100 issues regardless of
`max_total_issues`. Root causes:
  1. `maxResults` was set to the total limit; Jira caps it per-request (~100).
  2. The break check `len(issues) < limit` exited after the first page.
  3. The v3 `/search/jql` endpoint requires `nextPageToken` pagination, not `startAt`.
"""
import copy
from unittest.mock import Mock

import pytest

from elitea_sdk.tools.jira.api_wrapper import JiraApiWrapper


PAGE_SIZE = 100


class _FakeClient:
    """Records each call's params (deep-copied) so later mutations don't leak in."""

    def __init__(self, url: str):
        self._url = url
        self.calls: list[dict] = []
        self.resource_url = Mock(return_value=url)

    def get(self, url, params=None):
        # Snapshot params at call time — `_jql_get_tickets` mutates the same dict
        self.calls.append(copy.deepcopy(params or {}))
        return self._respond(self.calls[-1])

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def _respond(self, params: dict):  # overridden by subclasses
        raise NotImplementedError


class _FakeV2Client(_FakeClient):
    def __init__(self, total_issues: int):
        super().__init__("https://jira.example/rest/api/2/search")
        self._total = total_issues

    def _respond(self, params: dict):
        start = int(params.get("startAt", 0))
        page_size = int(params.get("maxResults", PAGE_SIZE))
        end = min(start + page_size, self._total)
        return {"issues": [{"key": f"PROJ-{i}"} for i in range(start, end)]}


class _FakeV3Client(_FakeClient):
    def __init__(self, total_issues: int):
        super().__init__("https://jira.example/rest/api/3/search/jql")
        self._total = total_issues

    def _respond(self, params: dict):
        page_size = int(params.get("maxResults", PAGE_SIZE))
        token = params.get("nextPageToken")
        start = int(token) if token else 0
        end = min(start + page_size, self._total)
        is_last = end >= self._total
        response = {
            "issues": [{"key": f"PROJ-{i}"} for i in range(start, end)],
            "isLast": is_last,
        }
        if not is_last:
            response["nextPageToken"] = str(end)
        return response


def _make_wrapper(api_version: str) -> JiraApiWrapper:
    """Build a JiraApiWrapper without running validators (no real connection)."""
    return JiraApiWrapper.model_construct(api_version=api_version)


def _make_v2_client(total_issues: int) -> _FakeV2Client:
    return _FakeV2Client(total_issues)


def _make_v3_client(total_issues: int) -> _FakeV3Client:
    return _FakeV3Client(total_issues)


class TestJqlPaginationV2:
    """Legacy v2 `/search` endpoint with startAt-based pagination."""

    def test_fetches_more_than_one_page_when_limit_above_page_size(self):
        wrapper = _make_wrapper(api_version="2")
        client = _make_v2_client(total_issues=350)

        batches = list(wrapper._jql_get_tickets(client, jql="project = X", limit=1000))
        issues = [issue for batch in batches for issue in batch]

        assert len(issues) == 350, "Should fetch all matching issues, not stop after one 100-issue page"
        assert client.call_count >= 4

    def test_honors_total_limit_across_pages(self):
        wrapper = _make_wrapper(api_version="2")
        client = _make_v2_client(total_issues=500)

        batches = list(wrapper._jql_get_tickets(client, jql="project = X", limit=250))
        issues = [issue for batch in batches for issue in batch]

        assert len(issues) == 250

    def test_per_request_max_results_capped_at_page_size(self):
        wrapper = _make_wrapper(api_version="2")
        client = _make_v2_client(total_issues=350)

        list(wrapper._jql_get_tickets(client, jql="project = X", limit=1000))

        for params in client.calls:
            assert int(params["maxResults"]) <= PAGE_SIZE, (
                f"maxResults={params['maxResults']} exceeds Jira's per-request cap"
            )

    def test_stops_when_short_page_returned(self):
        wrapper = _make_wrapper(api_version="2")
        client = _make_v2_client(total_issues=150)

        batches = list(wrapper._jql_get_tickets(client, jql="project = X", limit=1000))
        issues = [issue for batch in batches for issue in batch]

        assert len(issues) == 150
        assert client.call_count == 2

    def test_empty_result_set(self):
        wrapper = _make_wrapper(api_version="2")
        client = _make_v2_client(total_issues=0)

        batches = list(wrapper._jql_get_tickets(client, jql="project = EMPTY", limit=1000))

        assert batches == []
        assert client.call_count == 1

    def test_limit_smaller_than_page_size(self):
        wrapper = _make_wrapper(api_version="2")
        client = _make_v2_client(total_issues=500)

        batches = list(wrapper._jql_get_tickets(client, jql="project = X", limit=25))
        issues = [issue for batch in batches for issue in batch]

        assert len(issues) == 25
        # Page size must shrink to match the smaller limit
        assert int(client.calls[0]["maxResults"]) == 25

    def test_trims_final_page_to_match_limit_exactly(self):
        wrapper = _make_wrapper(api_version="2")
        client = _make_v2_client(total_issues=500)

        batches = list(wrapper._jql_get_tickets(client, jql="project = X", limit=150))
        issues = [issue for batch in batches for issue in batch]

        assert len(issues) == 150


class TestSearchUsingJql:
    """`search_using_jql` must paginate and honor the per-call `limit`."""

    def _make_wrapper_with_client(self, api_version: str, total_issues: int):
        """Build a wrapper wired to a fake client (skipping toolkit __init__)."""
        wrapper = _make_wrapper(api_version=api_version)
        wrapper.limit = 5  # toolkit default — must be overridden by per-call limit
        client = _make_v2_client(total_issues) if api_version == "2" else _make_v3_client(total_issues)
        # Inject a fake client.url for `_parse_issues`'s `issue_url` formatting
        client.url = "https://jira.example/"
        wrapper._client = client
        # Bypass `_get_client` (which would re-validate connection) by patching it
        wrapper._get_client = lambda: client
        return wrapper, client

    def test_returns_more_than_one_page_when_limit_exceeds_page_size(self):
        wrapper, client = self._make_wrapper_with_client(api_version="2", total_issues=350)

        result = wrapper.search_using_jql(jql="project = X", limit=300)

        assert "Found 300 Jira issues" in result
        # Multiple pages were fetched
        assert client.call_count >= 3

    def test_per_call_limit_overrides_toolkit_default(self):
        # toolkit-level self.limit=5 — without an override the old code returned only 5
        wrapper, client = self._make_wrapper_with_client(api_version="2", total_issues=200)

        result = wrapper.search_using_jql(jql="project = X", limit=150)

        assert "Found 150 Jira issues" in result

    def test_falls_back_to_toolkit_limit_when_unspecified(self):
        wrapper, client = self._make_wrapper_with_client(api_version="2", total_issues=200)

        result = wrapper.search_using_jql(jql="project = X")

        # toolkit default is 5 — backward-compatible behavior
        assert "Found 5 Jira issues" in result

    def test_empty_results(self):
        wrapper, client = self._make_wrapper_with_client(api_version="2", total_issues=0)

        result = wrapper.search_using_jql(jql="project = EMPTY", limit=1500)

        assert result == "No Jira issues found"

    def test_v3_endpoint_paginates_via_token(self):
        wrapper, client = self._make_wrapper_with_client(api_version="3", total_issues=250)

        result = wrapper.search_using_jql(jql="project = X", limit=250)

        assert "Found 250 Jira issues" in result
        # First call has no token; subsequent calls use nextPageToken
        assert "nextPageToken" not in client.calls[0]
        assert all("nextPageToken" in c for c in client.calls[1:])


class TestJqlPaginationV3:
    """v3 `/search/jql` endpoint with nextPageToken-based pagination."""

    def test_uses_next_page_token_not_start_at(self):
        wrapper = _make_wrapper(api_version="3")
        client = _make_v3_client(total_issues=250)

        batches = list(wrapper._jql_get_tickets(client, jql="project = X", limit=1000))
        issues = [issue for batch in batches for issue in batch]

        assert len(issues) == 250
        # First call has no token; subsequent calls must use nextPageToken (not startAt)
        assert len(client.calls) >= 3
        assert "nextPageToken" not in client.calls[0]
        assert "startAt" not in client.calls[0]
        for params in client.calls[1:]:
            assert "nextPageToken" in params
            assert "startAt" not in params, "v3 endpoint must not use startAt"

    def test_stops_on_is_last_flag(self):
        wrapper = _make_wrapper(api_version="3")
        client = _make_v3_client(total_issues=200)

        batches = list(wrapper._jql_get_tickets(client, jql="project = X", limit=1000))
        issues = [issue for batch in batches for issue in batch]

        assert len(issues) == 200
        assert client.call_count == 2

    def test_honors_total_limit(self):
        wrapper = _make_wrapper(api_version="3")
        client = _make_v3_client(total_issues=500)

        batches = list(wrapper._jql_get_tickets(client, jql="project = X", limit=150))
        issues = [issue for batch in batches for issue in batch]

        assert len(issues) == 150
