"""Regression test for GitLabWorkspaceAPIWrapper.update_file's ToolException handling.

update_file used to catch a ToolException raised by edit_file and return
str(e) — a plain string that looks like a successful commit message to any
caller, silently disguising a real failure as success. This proves the fix
(re-raise) actually propagates instead of swallowing the error.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("gitlab", MagicMock())

from langchain_core.tools import ToolException

from elitea_sdk.tools.gitlab_org.api_wrapper import GitLabWorkspaceAPIWrapper


def _make_wrapper():
    return GitLabWorkspaceAPIWrapper.model_construct(repo_instances={})


def test_update_file_reraises_tool_exception_from_edit_file():
    wrapper = _make_wrapper()
    with patch.object(
        GitLabWorkspaceAPIWrapper, "edit_file", side_effect=ToolException("file not found")
    ):
        with pytest.raises(ToolException, match="file not found"):
            wrapper.update_file(
                file_path="f.py", update_query="OLD <<<<\nold\n>>>> OLD\nNEW <<<<\nnew\n>>>> NEW", branch="main"
            )


def test_update_file_clears_tmp_repository_context_even_on_raise():
    wrapper = _make_wrapper()
    with patch.object(GitLabWorkspaceAPIWrapper, "edit_file", side_effect=ToolException("boom")):
        with pytest.raises(ToolException):
            wrapper.update_file(
                file_path="f.py", update_query="OLD <<<<\nold\n>>>> OLD\nNEW <<<<\nnew\n>>>> NEW",
                branch="main", repository="group/repo",
            )
    assert not hasattr(wrapper, "_tmp_repository_for_edit")


def test_update_file_returns_success_message_when_edit_file_succeeds():
    wrapper = _make_wrapper()
    with patch.object(GitLabWorkspaceAPIWrapper, "edit_file", return_value="Updated f.py"):
        assert wrapper.update_file(
            file_path="f.py", update_query="OLD <<<<\nold\n>>>> OLD\nNEW <<<<\nnew\n>>>> NEW", branch="main"
        ) == "Updated f.py"
