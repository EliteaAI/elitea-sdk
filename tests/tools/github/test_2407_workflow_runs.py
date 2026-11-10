"""Tests for the GitHub get_workflow_runs tool (#2407)."""
from datetime import datetime, timezone
from types import SimpleNamespace

from elitea_sdk.tools.github.github_client import GitHubClient


def _run(i, **kw):
    base = dict(
        id=i, name=f"run {i}", workflow_id=7, run_number=i, event="push", status="completed",
        conclusion="success", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), updated_at=None,
        head_branch="main", head_sha="abc", html_url=f"https://github.com/o/r/actions/runs/{i}",
    )
    base.update(kw)
    return SimpleNamespace(**base)


class FakeWorkflow:
    id = 7

    def __init__(self, calls):
        self.calls = calls

    def get_runs(self, **kw):
        self.calls.append(("workflow", kw))
        return [_run(i) for i in range(1, 6)]


class FakeRepo:
    full_name = "o/r"

    def __init__(self):
        self.calls = []

    def get_workflow_runs(self, **kw):
        self.calls.append(("repo", kw))
        return iter([_run(i) for i in range(1, 6)])  # iterator: must not be len()'d or sliced

    def get_workflow(self, ref):
        if ref in ("build.yml", "7"):
            return FakeWorkflow(self.calls)
        raise Exception("404")

    def get_workflows(self):
        return [FakeWorkflow(self.calls)]


def _client(repo):
    api = SimpleNamespace(get_repo=lambda name: repo)
    return GitHubClient.model_construct(
        github_repository="o/r", active_branch="main", github_base_branch="main",
        github_api=api, elitea=None,
    )


def test_lists_runs_of_all_workflows_with_filters_and_limit():
    repo = FakeRepo()
    out = _client(repo).get_workflow_runs(branch="main", status="failure", limit=2)
    assert out["count"] == 2 and [r["id"] for r in out["runs"]] == [1, 2]
    assert repo.calls == [("repo", {"branch": "main", "status": "failure"})]
    assert out["runs"][0]["created_at"] == "2026-01-01T00:00:00+00:00"


def test_workflow_by_file_name_or_numeric_id():
    for ref in ("build.yml", "7"):
        repo = FakeRepo()
        out = _client(repo).get_workflow_runs(workflow_id=ref)
        assert out["count"] == 5 and repo.calls[0][0] == "workflow"


def test_unknown_workflow_is_a_readable_error():
    out = _client(FakeRepo()).get_workflow_runs(workflow_id="nope.yml")
    assert out["error"] is True and "nope.yml" in out["message"]


def test_limit_is_clamped():
    assert _client(FakeRepo()).get_workflow_runs(limit=0)["count"] == 1


def test_tool_is_registered_and_read_only_group():
    tools = {t["name"]: t for t in _client(FakeRepo()).get_available_tools()}
    assert "get_workflow_runs" in tools
    assert tools["get_workflow_runs"]["args_schema"].model_json_schema()["properties"]["limit"]["maximum"] == 100
