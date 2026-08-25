"""Tests for GitLab MR metadata tools (#6212).

Covers list_merge_requests and get_merge_request: happy path list/get,
missing MR (404), partial/missing optional fields, and pagination/filter
parameter forwarding.
"""

import pytest
from gitlab import GitlabGetError
from langchain_core.tools import ToolException

from elitea_sdk.tools.gitlab.api_wrapper import GitLabAPIWrapper


class FakeMR:
    def __init__(self, attributes, description=None):
        self.attributes = attributes
        self.description = description
        self.id = attributes.get("id")
        self.iid = attributes.get("iid")


class FakeMergeRequestsManager:
    def __init__(self, mrs_by_iid=None, list_result=None):
        self._mrs_by_iid = mrs_by_iid or {}
        self._list_result = list_result or []
        self.list_calls = []

    def get(self, pr_number):
        if pr_number not in self._mrs_by_iid:
            raise GitlabGetError("Not found", response_code=404)
        return self._mrs_by_iid[pr_number]

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return self._list_result


class FakeRepo:
    def __init__(self, mergerequests):
        self.mergerequests = mergerequests


def _make_wrapper(mergerequests) -> GitLabAPIWrapper:
    wrapper = GitLabAPIWrapper.model_construct(branch="main", llm=None)
    wrapper._active_branch = "main"
    wrapper._repo_instance = FakeRepo(mergerequests)
    return wrapper


FULL_MR_ATTRS = {
    "id": 1,
    "iid": 42,
    "title": "Add feature",
    "web_url": "https://gitlab.example.com/group/proj/-/merge_requests/42",
    "state": "merged",
    "draft": False,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-02T00:00:00Z",
    "merged_at": "2026-01-02T00:00:00Z",
    "closed_at": None,
    "source_branch": "feature/x",
    "target_branch": "main",
    "author": {"id": 10, "username": "alice", "name": "Alice"},
    "merge_user": {"id": 11, "username": "bob", "name": "Bob"},
    "labels": ["backend"],
    "sha": "abc123",
    "merge_commit_sha": "def456",
    "assignees": [{"id": 12, "username": "carol", "name": "Carol"}],
    "reviewers": [{"id": 13, "username": "dave", "name": "Dave"}],
}


class TestListMergeRequests:
    def test_happy_path_list(self):
        mr = FakeMR(FULL_MR_ATTRS)
        mgr = FakeMergeRequestsManager(list_result=[mr])
        wrapper = _make_wrapper(mgr)

        result = wrapper.list_merge_requests(state="merged")

        assert len(result) == 1
        assert result[0]["iid"] == 42
        assert result[0]["title"] == "Add feature"
        assert result[0]["author"] == {"id": 10, "username": "alice", "name": "Alice"}
        # No diff/raw content leaks into compact list output
        assert "diff" not in result[0]

    def test_pagination_and_filter_params_forwarded(self):
        mgr = FakeMergeRequestsManager(list_result=[])
        wrapper = _make_wrapper(mgr)

        wrapper.list_merge_requests(
            state="opened",
            page=3,
            per_page=50,
            created_after="2026-01-01T00:00:00Z",
            target_branch="main",
            source_branch="feature/x",
            author_username="alice",
            labels="backend,urgent",
        )

        assert len(mgr.list_calls) == 1
        call = mgr.list_calls[0]
        assert call["state"] == "opened"
        assert call["page"] == 3
        assert call["per_page"] == 50
        assert call["created_after"] == "2026-01-01T00:00:00Z"
        assert call["target_branch"] == "main"
        assert call["source_branch"] == "feature/x"
        assert call["author_username"] == "alice"
        assert call["labels"] == "backend,urgent"

    def test_all_flag_forwards_all_true_and_skips_pagination(self):
        mgr = FakeMergeRequestsManager(list_result=[])
        wrapper = _make_wrapper(mgr)

        wrapper.list_merge_requests(all=True)

        call = mgr.list_calls[0]
        assert call.get("all") is True
        assert "page" not in call
        assert "per_page" not in call

    def test_partial_field_availability_does_not_fail(self):
        # Missing author/merge_user/labels etc. must not raise.
        partial_attrs = {"id": 2, "iid": 43, "title": "Partial MR"}
        mr = FakeMR(partial_attrs)
        mgr = FakeMergeRequestsManager(list_result=[mr])
        wrapper = _make_wrapper(mgr)

        result = wrapper.list_merge_requests()

        assert result[0]["iid"] == 43
        assert result[0]["author"] is None
        assert result[0]["labels"] is None


class TestGetMergeRequest:
    def test_happy_path_get(self):
        mr = FakeMR(FULL_MR_ATTRS, description="Some description")
        mgr = FakeMergeRequestsManager(mrs_by_iid={42: mr})
        wrapper = _make_wrapper(mgr)

        result = wrapper.get_merge_request(42)

        assert result["iid"] == 42
        assert result["description"] == "Some description"
        assert result["assignees"] == [{"id": 12, "username": "carol", "name": "Carol"}]
        # Heavy optional fields are not included unless requested
        assert "discussions" not in result
        assert "approvals" not in result
        assert "pipeline" not in result
        assert "commits" not in result

    def test_missing_mr_raises_tool_exception(self):
        mgr = FakeMergeRequestsManager(mrs_by_iid={})
        wrapper = _make_wrapper(mgr)

        with pytest.raises(ToolException):
            wrapper.get_merge_request(999)

    def test_partial_field_availability(self):
        mr = FakeMR({"id": 3, "iid": 44})
        mgr = FakeMergeRequestsManager(mrs_by_iid={44: mr})
        wrapper = _make_wrapper(mgr)

        result = wrapper.get_merge_request(44)

        assert result["iid"] == 44
        assert result["state"] is None
        assert result["author"] is None
