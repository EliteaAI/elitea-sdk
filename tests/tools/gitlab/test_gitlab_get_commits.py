"""Tests for get_commits date-bound forwarding (#6533).

`until=2026-08-20` returns nothing from 2026-08-20 because GitLab floors a
coarse upper bound to the start of the unit; the expanded bound covers the day.
"""

from elitea_sdk.tools.gitlab.api_wrapper import GitLabAPIWrapper


class FakeCommitsManager:
    def __init__(self):
        self.list_calls = []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return []


class FakeRepo:
    def __init__(self, commits):
        self.commits = commits


def _make_wrapper(commits) -> GitLabAPIWrapper:
    wrapper = GitLabAPIWrapper.model_construct(branch="main", llm=None)
    wrapper._active_branch = "main"
    wrapper._repo_instance = FakeRepo(commits)
    return wrapper


def test_until_expanded_and_since_forwarded_verbatim():
    mgr = FakeCommitsManager()

    _make_wrapper(mgr).get_commits(since="2026-08-20", until="2026-08-20")

    call = mgr.list_calls[0]
    assert call["since"] == "2026-08-20"
    assert call["until"] == "2026-08-20T23:59:59.999999"


def test_bounds_are_forwarded_as_strings():
    """A datetime would reach the wire as '2026-08-20 23:59:59.999999+00:00'."""
    mgr = FakeCommitsManager()

    _make_wrapper(mgr).get_commits(since="2026-08-20T09:24:40.498Z", until="2026-08-20T09:24:40.498Z")

    call = mgr.list_calls[0]
    assert isinstance(call["since"], str)
    assert isinstance(call["until"], str)
    assert call["until"] == "2026-08-20T09:24:40.498999Z"
