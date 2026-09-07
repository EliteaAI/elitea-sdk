"""Tests for GitLab get_issues state/pagination/filter support (#6213).

Covers happy path, filter/pagination parameter forwarding, the fetch_all
flag, and comment created_at inclusion in get_issue.
"""

from elitea_sdk.tools.gitlab.api_wrapper import GitLabAPIWrapper


class FakeIssue:
    def __init__(self, attributes):
        for key, value in attributes.items():
            setattr(self, key, value)


class FakeIssuesManager:
    def __init__(self, list_result=None, issue_by_iid=None):
        self._list_result = list_result or []
        self._issue_by_iid = issue_by_iid or {}
        self.list_calls = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return self._list_result

    def get(self, issue_number):
        return self._issue_by_iid[issue_number]


class FakeNotesManager:
    def __init__(self, notes):
        self._notes = notes

    def list(self, page=1):
        return self._notes if page == 1 else []

    def get(self, note_id):
        return next(n for n in self._notes if n.id == note_id)


class FakeRepo:
    def __init__(self, issues):
        self.issues = issues


def _make_wrapper(issues) -> GitLabAPIWrapper:
    wrapper = GitLabAPIWrapper.model_construct(branch="main", llm=None)
    wrapper._active_branch = "main"
    wrapper._repo_instance = FakeRepo(issues)
    return wrapper


FULL_ISSUE_ATTRS = {
    "title": "Bug in login",
    "iid": 7,
    "state": "opened",
    "labels": ["bug"],
    "author": {"username": "alice"},
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-02T00:00:00Z",
}


class TestGetIssues:
    def test_happy_path_list(self):
        issue = FakeIssue(FULL_ISSUE_ATTRS)
        mgr = FakeIssuesManager(list_result=[issue])
        wrapper = _make_wrapper(mgr)

        result = wrapper.get_issues()

        assert "Found 1 issues" in result
        assert "'number': 7" in result
        assert "'state': 'opened'" in result

    def test_no_issues_found(self):
        mgr = FakeIssuesManager(list_result=[])
        wrapper = _make_wrapper(mgr)

        result = wrapper.get_issues()

        assert result == "No issues found matching the given filters"

    def test_default_state_is_opened_and_paginated(self):
        mgr = FakeIssuesManager(list_result=[])
        wrapper = _make_wrapper(mgr)

        wrapper.get_issues()

        call = mgr.list_calls[0]
        assert call["state"] == "opened"
        assert call["page"] == 1
        assert call["per_page"] == 20

    def test_filter_and_pagination_params_forwarded(self):
        mgr = FakeIssuesManager(list_result=[])
        wrapper = _make_wrapper(mgr)

        wrapper.get_issues(
            state="closed",
            page=3,
            per_page=100,
            created_after="2026-01-01T00:00:00Z",
            created_before="2026-02-01T00:00:00Z",
            author_username="alice",
            labels="bug,urgent",
        )

        call = mgr.list_calls[0]
        assert call["state"] == "closed"
        assert call["page"] == 3
        assert call["per_page"] == 100
        assert call["created_after"] == "2026-01-01T00:00:00Z"
        assert call["created_before"] == "2026-02-01T00:00:00.999999Z"
        assert call["author_username"] == "alice"
        assert call["labels"] == "bug,urgent"

    def test_upper_bounds_expanded_lower_bounds_forwarded_verbatim(self):
        """#6533: the timestamp GitLab renders for an issue must reach that issue."""
        mgr = FakeIssuesManager(list_result=[])
        wrapper = _make_wrapper(mgr)

        wrapper.get_issues(
            created_after="2026-09-03T09:24:40.498Z",
            created_before="2026-09-03T09:24:40.498Z",
            updated_after="2026-09-03T10:02:18.692Z",
            updated_before="2026-09-03T10:02:18.692Z",
        )

        call = mgr.list_calls[0]
        assert call["created_after"] == "2026-09-03T09:24:40.498Z"
        assert call["created_before"] == "2026-09-03T09:24:40.498999Z"
        assert call["updated_after"] == "2026-09-03T10:02:18.692Z"
        assert call["updated_before"] == "2026-09-03T10:02:18.692999Z"

    def test_date_only_upper_bound_covers_the_whole_day(self):
        mgr = FakeIssuesManager(list_result=[])
        wrapper = _make_wrapper(mgr)

        wrapper.get_issues(created_after="2026-09-03", created_before="2026-09-03")

        call = mgr.list_calls[0]
        assert call["created_after"] == "2026-09-03"
        assert call["created_before"] == "2026-09-03T23:59:59.999999"

    def test_unparseable_upper_bound_forwarded_for_gitlab_to_reject(self):
        mgr = FakeIssuesManager(list_result=[])
        wrapper = _make_wrapper(mgr)

        wrapper.get_issues(created_before="last week")

        assert mgr.list_calls[0]["created_before"] == "last week"

    def test_state_all_omits_state_param(self):
        mgr = FakeIssuesManager(list_result=[])
        wrapper = _make_wrapper(mgr)

        wrapper.get_issues(state="all")

        call = mgr.list_calls[0]
        assert "state" not in call

    def test_fetch_all_flag_skips_pagination(self):
        mgr = FakeIssuesManager(list_result=[])
        wrapper = _make_wrapper(mgr)

        wrapper.get_issues(fetch_all=True)

        call = mgr.list_calls[0]
        assert call.get("all") is True
        assert "page" not in call
        assert "per_page" not in call


class TestGetIssueComments:
    def test_comments_include_created_at(self):
        note = FakeIssue({
            "id": 1,
            "body": "Looks good",
            "author": {"username": "bob"},
            "created_at": "2026-01-03T00:00:00Z",
        })
        issue = FakeIssue({
            "iid": 7,
            "title": "Bug in login",
            "description": "Steps to reproduce",
            "notes": FakeNotesManager([note]),
        })
        mgr = FakeIssuesManager(issue_by_iid={7: issue})
        wrapper = _make_wrapper(mgr)

        result = wrapper.get_issue(7)

        assert result["comments"] == [
            {"body": "Looks good", "user": "bob", "created_at": "2026-01-03T00:00:00Z"}
        ]
