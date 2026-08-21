"""Regression tests for GitLabWorkspaceAPIWrapper._get_repo against the real code.

The existing test_gitlab_org_repository_access.py only exercises a standalone
reimplementation of this logic (`simulate_get_repo`), not the actual method —
it would keep passing even if `_get_repo` itself broke. These tests call the
real `_get_repo` to cover the ToolException raise/re-raise behavior converted
from `return ToolException(...)`.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

gitlab_mock = MagicMock()
gitlab_mock.GitlabGetError = type("GitlabGetError", (Exception,), {})
sys.modules.setdefault("gitlab", gitlab_mock)

from langchain_core.tools import ToolException

from elitea_sdk.tools.gitlab_org.api_wrapper import GitLabWorkspaceAPIWrapper


def _make_wrapper(repo_instances=None):
    return GitLabWorkspaceAPIWrapper.model_construct(repo_instances=repo_instances or {})


def test_get_repo_returns_configured_repo():
    project = MagicMock()
    wrapper = _make_wrapper({"group/allowed": project})

    assert wrapper._get_repo("group/allowed") is project


def test_get_repo_raises_tool_exception_for_unconfigured_repo():
    wrapper = _make_wrapper({"group/allowed": MagicMock()})

    with pytest.raises(ToolException, match="not in the configured repositories list"):
        wrapper._get_repo("group/other")


def test_get_repo_raises_tool_exception_when_none_configured_and_no_name():
    wrapper = _make_wrapper({})

    with pytest.raises(ToolException, match="haven't configured any repositories"):
        wrapper._get_repo(None)


def test_get_repo_with_none_returns_first_configured():
    project = MagicMock()
    wrapper = _make_wrapper({"group/allowed": project})

    assert wrapper._get_repo(None) is project


def test_get_repo_wraps_unexpected_error_from_get_repo_instance():
    wrapper = _make_wrapper({})

    with patch.object(
        GitLabWorkspaceAPIWrapper, "_get_repo_instance", side_effect=RuntimeError("connection reset")
    ):
        with pytest.raises(ToolException, match="connection reset"):
            wrapper._get_repo("group/new-repo")
