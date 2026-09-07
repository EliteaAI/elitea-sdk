"""Tests for gitlab_org date-bound forwarding (#6533).

get_commits used to call datetime.fromisoformat on these values, which raises on
Python 3.10 for both the `Z` suffix and any fraction that is not 3 or 6 digits --
the shapes GitLab itself emits -- and otherwise handed python-gitlab a datetime
where every sibling tool passes a string.
"""

import sys
from unittest.mock import MagicMock

gitlab_mock = MagicMock()
gitlab_mock.GitlabGetError = type("GitlabGetError", (Exception,), {})
sys.modules.setdefault("gitlab", gitlab_mock)

from elitea_sdk.tools.gitlab_org.api_wrapper import GitLabWorkspaceAPIWrapper


class FakeManager:
    def __init__(self):
        self.list_calls = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return []


class FakeRepo:
    def __init__(self, issues, commits):
        self.issues = issues
        self.commits = commits


def _make_wrapper(issues=None, commits=None) -> GitLabWorkspaceAPIWrapper:
    repo = FakeRepo(issues or FakeManager(), commits or FakeManager())
    return GitLabWorkspaceAPIWrapper.model_construct(repo_instances={"group/proj": repo})


class TestGetIssues:
    def test_upper_bounds_expanded_lower_bounds_forwarded_verbatim(self):
        issues = FakeManager()
        wrapper = _make_wrapper(issues=issues)

        wrapper.get_issues(
            repository="group/proj",
            created_after="2026-09-03T09:24:40.498Z",
            created_before="2026-09-03T09:24:40.498Z",
            updated_after="2026-09-03T10:02:18.692Z",
            updated_before="2026-09-03T10:02:18.692Z",
        )

        call = issues.list_calls[0]
        assert call["created_after"] == "2026-09-03T09:24:40.498Z"
        assert call["created_before"] == "2026-09-03T09:24:40.498999Z"
        assert call["updated_after"] == "2026-09-03T10:02:18.692Z"
        assert call["updated_before"] == "2026-09-03T10:02:18.692999Z"

    def test_date_only_upper_bound_covers_the_whole_day(self):
        issues = FakeManager()
        wrapper = _make_wrapper(issues=issues)

        wrapper.get_issues(repository="group/proj", created_before="2026-09-03")

        assert issues.list_calls[0]["created_before"] == "2026-09-03T23:59:59.999999"


class TestGetCommits:
    def test_until_expanded_and_since_forwarded_verbatim(self):
        commits = FakeManager()
        wrapper = _make_wrapper(commits=commits)

        wrapper.get_commits(repository="group/proj", since="2026-08-20", until="2026-08-20")

        call = commits.list_calls[0]
        assert call["since"] == "2026-08-20"
        assert call["until"] == "2026-08-20T23:59:59.999999"

    def test_gitlab_rendered_timestamp_survives_as_a_string(self):
        commits = FakeManager()
        wrapper = _make_wrapper(commits=commits)

        wrapper.get_commits(
            repository="group/proj",
            since="2026-08-20T09:24:40.498Z",
            until="2026-08-20T09:24:40.498Z",
        )

        call = commits.list_calls[0]
        assert isinstance(call["since"], str)
        assert isinstance(call["until"], str)
        assert call["until"] == "2026-08-20T09:24:40.498999Z"


def test_both_toolkits_forward_identical_commit_bounds():
    """The two get_commits implementations diverged on exactly these inputs."""
    from elitea_sdk.tools.gitlab.api_wrapper import GitLabAPIWrapper

    org_commits = FakeManager()
    _make_wrapper(commits=org_commits).get_commits(
        repository="group/proj", since="2026-08-20", until="2026-08-20T09:24:40.498Z"
    )

    repo_commits = FakeManager()
    wrapper = GitLabAPIWrapper.model_construct(branch="main", llm=None)
    wrapper._active_branch = "main"
    wrapper._repo_instance = FakeRepo(FakeManager(), repo_commits)
    wrapper.get_commits(since="2026-08-20", until="2026-08-20T09:24:40.498Z")

    for key in ("since", "until"):
        assert org_commits.list_calls[0][key] == repo_commits.list_calls[0][key]
