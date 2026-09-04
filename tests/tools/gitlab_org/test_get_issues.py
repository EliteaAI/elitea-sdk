"""GitLab org get_issues returns native issue data, not a display string (#6532)."""
import json
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("gitlab", MagicMock())

from elitea_sdk.tools.gitlab_org.api_wrapper import GitLabWorkspaceAPIWrapper


class FakeIssue:
    def __init__(self, attributes):
        for key, value in attributes.items():
            setattr(self, key, value)


class FakeIssuesManager:
    def __init__(self, list_result):
        self._list_result = list_result
        self.list_calls = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return self._list_result


class FakeRepo:
    def __init__(self, issues):
        self.issues = issues


ISSUE_ATTRS = {
    "title": "Bug in login",
    "iid": 7,
    "state": "opened",
    "labels": ["bug"],
    "author": {"username": "alice"},
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-02T00:00:00Z",
}


def _make_wrapper(issues_manager):
    wrapper = GitLabWorkspaceAPIWrapper.model_construct(repo_instances={})
    wrapper._get_repo = lambda repository=None: FakeRepo(issues_manager)
    return wrapper


def test_get_issues_returns_json_serializable_list():
    manager = FakeIssuesManager([FakeIssue(ISSUE_ATTRS)])

    result = _make_wrapper(manager).get_issues()

    assert result == [
        {
            "title": "Bug in login",
            "number": 7,
            "state": "opened",
            "labels": ["bug"],
            "author": "alice",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-02T00:00:00Z",
        }
    ]
    assert json.loads(json.dumps(result)) == result


def test_get_issues_returns_empty_list_when_nothing_matches():
    assert _make_wrapper(FakeIssuesManager([])).get_issues() == []
