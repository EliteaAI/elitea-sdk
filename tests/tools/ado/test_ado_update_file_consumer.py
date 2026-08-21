"""Regression test for ReposApiWrapper.update_file's ToolException handling.

update_file used to catch a ToolException raised by edit_file and return
str(e) — a plain string that looks like a successful commit message to any
caller, silently disguising a real failure as success. This proves the fix
(re-raise) actually propagates instead of swallowing the error.
"""
from unittest.mock import patch

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.ado.repos.repos_wrapper import ReposApiWrapper


def _make_wrapper():
    return ReposApiWrapper.model_construct(base_branch="main", active_branch="main")


def test_update_file_reraises_tool_exception_from_edit_file():
    wrapper = _make_wrapper()
    with patch.object(ReposApiWrapper, "edit_file", side_effect=ToolException("file not found")):
        with pytest.raises(ToolException, match="file not found"):
            wrapper.update_file(
                branch_name="feature",
                file_path="f.py",
                update_query="OLD <<<<\nold\n>>>> OLD\nNEW <<<<\nnew\n>>>> NEW",
            )


def test_update_file_still_refuses_direct_commit_to_base_branch():
    wrapper = _make_wrapper()
    with patch.object(ReposApiWrapper, "edit_file", side_effect=ToolException("should not be called")):
        result = wrapper.update_file(
            branch_name="main", file_path="f.py",
            update_query="OLD <<<<\nold\n>>>> OLD\nNEW <<<<\nnew\n>>>> NEW",
        )
    assert "protected" in result


def test_update_file_returns_success_message_when_edit_file_succeeds():
    wrapper = _make_wrapper()
    with patch.object(ReposApiWrapper, "edit_file", return_value="Updated f.py"):
        assert wrapper.update_file(
            branch_name="feature", file_path="f.py",
            update_query="OLD <<<<\nold\n>>>> OLD\nNEW <<<<\nnew\n>>>> NEW",
        ) == "Updated f.py"
