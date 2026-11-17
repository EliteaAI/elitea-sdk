"""GitHub toolkit fixes: trigger_workflow (#7518), generic call lists (#7519),
run_id validation (#7520), logs note (#7521), empty code search (#7551)."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from github import GithubException
from github.ContentFile import ContentFile
from github.PaginatedList import PaginatedList

import elitea_sdk.tools.github.github_client as github_client
from elitea_sdk.tools.github.github_client import GitHubClient


def _client(api=None, repo=None):
    api = api or SimpleNamespace(get_repo=lambda name: repo)
    return GitHubClient.model_construct(
        github_repository="o/r", active_branch="main", github_base_branch="main",
        github_api=api, elitea=None,
    )


# --- #7518 trigger_workflow -------------------------------------------------------------

class FakeWorkflow:
    id = 7
    name = "Test Workflow"
    html_url = "https://github.com/o/r/blob/main/.github/workflows/build.yml"
    url = "https://api.github.com/repos/o/r/actions/workflows/7"

    def __init__(self, dispatch_error=None, runs=()):
        self._requester = MagicMock()
        if dispatch_error:
            self._requester.requestJsonAndCheck.side_effect = dispatch_error
        self.runs = list(runs)
        self.get_runs_calls = []

    def get_runs(self, **kw):
        self.get_runs_calls.append(kw)
        return iter(self.runs)


class FakeRepo:
    full_name = "o/r"

    def __init__(self, workflow):
        self.workflow = workflow

    def get_workflow(self, ref):
        if ref == "build.yml":
            return self.workflow
        raise Exception("404")

    def get_workflows(self):
        return [self.workflow]

    def get_workflow_run(self, run_id):
        raise GithubException(404, {"message": "Not Found"})


@pytest.fixture(autouse=True)
def no_lookup_sleep(monkeypatch):
    monkeypatch.setattr(github_client, "DISPATCHED_RUN_LOOKUP_INTERVAL_SECONDS", 0)


def _run(run_id, created_at):
    return SimpleNamespace(id=run_id, created_at=created_at, html_url=f"https://github.com/o/r/actions/runs/{run_id}")


def test_rejected_dispatch_returns_githubs_reason():
    error = GithubException(422, {"message": "No ref found for: branch-does-not-exist-xyz"})
    out = _client(repo=FakeRepo(FakeWorkflow(dispatch_error=error))).trigger_workflow(
        "build.yml", "branch-does-not-exist-xyz")
    assert out["error"] is True
    assert "No ref found for: branch-does-not-exist-xyz" in out["message"]
    assert "success" not in out


def test_successful_dispatch_returns_the_started_run():
    later = datetime.now(timezone.utc) + timedelta(seconds=5)
    workflow = FakeWorkflow(runs=[_run(42, later)])
    out = _client(repo=FakeRepo(workflow)).trigger_workflow("build.yml", "main", {"x": "1"})
    assert out["success"] is True and out["run_id"] == 42
    assert out["run_url"].endswith("/runs/42")
    workflow._requester.requestJsonAndCheck.assert_called_once_with(
        "POST", f"{workflow.url}/dispatches", input={"ref": "main", "inputs": {"x": "1"}})
    assert workflow.get_runs_calls[0] == {"event": "workflow_dispatch", "branch": "main"}


def test_older_run_is_not_reported_as_the_dispatched_one():
    earlier = datetime.now(timezone.utc) - timedelta(hours=1)
    workflow = FakeWorkflow(runs=[_run(41, earlier)])
    out = _client(repo=FakeRepo(workflow)).trigger_workflow("build.yml", "main")
    assert out["success"] is True and out["run_id"] is None
    assert "get_workflow_runs" in out["note"]
    assert len(workflow.get_runs_calls) == github_client.DISPATCHED_RUN_LOOKUP_ATTEMPTS


def test_unknown_workflow_is_a_structured_error():
    out = _client(repo=FakeRepo(FakeWorkflow())).trigger_workflow("nope.yml", "main")
    assert out["error"] is True and "nope.yml" in out["message"]


# --- #7519 generic_github_api_call --------------------------------------------------------

class Item:
    """A lazily-listed PyGithub object: raw_data would refetch it."""

    def __init__(self, n):
        self._rawData = {"id": n}

    @property
    def raw_data(self):
        raise AssertionError("raw_data refetches each listed item")


def _paginated(n):
    requester = MagicMock(per_page=30)
    requester.requestJsonAndCheck.return_value = ({}, [{"id": i} for i in range(n)])
    return PaginatedList(ContentFile, requester, "/repos/o/r/branches", None)


class FakeRepository:
    full_name = "o/r"

    def get_branches(self):
        return _paginated(3)

    def get_workflow_runs(self):
        return [Item(i) for i in range(40)]

    def delete(self):
        return None


def _generic(method, monkeypatch):
    monkeypatch.setattr(github_client.Repository, "Repository", FakeRepository)
    api = SimpleNamespace(get_repo=lambda name: FakeRepository())
    return _client(api=api).generic_github_api_call(method)


def test_paginated_list_returns_item_data(monkeypatch):
    out = _generic("get_branches", monkeypatch)
    assert out == {"items": [{"id": 0}, {"id": 1}, {"id": 2}], "count": 3, "truncated": False}


def test_long_list_is_capped_without_refetching_items(monkeypatch):
    out = _generic("get_workflow_runs", monkeypatch)
    assert out["count"] == github_client.GENERIC_CALL_LIST_LIMIT and out["truncated"] is True
    assert out["items"][0] == {"id": 0}


def test_plain_return_value_is_passed_through(monkeypatch):
    assert _generic("delete", monkeypatch) is None


# --- #7520 / #7521 workflow run tools -----------------------------------------------------

@pytest.mark.parametrize("tool", ["get_workflow_status", "get_workflow_logs"])
@pytest.mark.parametrize("run_id", ["abc", "", None, "12a"])
def test_non_numeric_run_id_is_a_clear_validation_error(tool, run_id):
    out = getattr(_client(repo=FakeRepo(FakeWorkflow())), tool)(run_id)
    assert out == {"error": True, "message": github_client.INVALID_RUN_ID_MESSAGE}


def test_unknown_run_points_to_get_workflow_runs():
    client = _client(repo=FakeRepo(FakeWorkflow()))
    assert "get_workflow_runs" in client.get_workflow_status("189651695")["message"]
    assert "get_workflow_runs" in client.get_workflow_logs("189651695")


def test_default_logs_note_does_not_read_like_a_failure():
    step = SimpleNamespace(name="s", status="completed", conclusion="success", number=1,
                           started_at=None, completed_at=None)
    job = SimpleNamespace(id=1, name="j", status="completed", conclusion="success", steps=[step])
    run = SimpleNamespace(id=5, status="completed", conclusion="success", html_url="u",
                          head_branch="main", head_sha="abc", jobs=lambda: [job])
    repo = SimpleNamespace(get_workflow_run=lambda run_id: run)
    out = _client(repo=repo).get_workflow_logs(" 5 ")
    assert "include_logs: true" in out["note"]
    assert "couldn't" not in out["note"]


# --- #7551 search_code ----------------------------------------------------------------------

def _search_api(items):
    """PyGithub's real PaginatedList over a requester that, like GitHub, rejects a query without q."""
    requests = []

    def request(verb, url, parameters=None, headers=None):
        if url != "/search/code":  # PyGithub completing a listed item
            return {}, next(i for i in items if i["url"] == url)
        requests.append(dict(parameters or {}))
        if "q" not in (parameters or {}):
            raise GithubException(400, {"data": "Failed to deserialize query string: missing field `q`"})
        return {}, {"total_count": len(items), "incomplete_results": False, "items": items}

    requester = MagicMock(per_page=30)
    requester.requestJsonAndCheck.side_effect = request
    api = SimpleNamespace(search_code=lambda query, **kw: PaginatedList(
        ContentFile, requester, "/search/code", {"q": query}))
    return api, requests


def test_search_without_matches_says_no_code_found():
    api, requests = _search_api([])
    out = json.loads(_client(api=api).search_code("zzqq-not-there"))
    assert out["total_count"] == 0 and out["items"] == []
    assert "No code found" in out["message"]
    assert requests == [{"q": "zzqq-not-there repo:o/r"}]


def test_search_with_matches_reports_the_total():
    item = {"name": "a.py", "path": "src/a.py", "sha": "s", "html_url": "h",
            "url": "https://api.github.com/repos/o/r/contents/src/a.py",
            "repository": {"full_name": "o/r", "html_url": "r", "description": None, "private": False}}
    api, requests = _search_api([item])
    out = json.loads(_client(api=api).search_code("429"))
    assert out["total_count"] == 1 and out["items"][0]["path"] == "src/a.py"
    assert len(requests) == 1
