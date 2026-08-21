"""Regression test for BaseCodeToolApiWrapper.edit_file's consumer of
self._read_file (elitea_base.py:794).

All concrete _read_file implementations (github, gitlab, gitlab_org,
bitbucket, ado/repos, localgit) now raise ToolException instead of
returning it, so the `isinstance(current_content, Exception)` branch at
elitea_base.py:794 is unreachable via that path. This test proves the
raise from _read_file still propagates through edit_file as a
ToolException, without relying on line 794 to re-raise it.
"""
import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.elitea_base import BaseCodeToolApiWrapper


class _StubCodeWrapper(BaseCodeToolApiWrapper):
    def _get_files(self):
        raise NotImplementedError

    def _read_file(self, file_path, branch=None, **kwargs):
        raise ToolException(f"File not found: {file_path}")

    def _write_file(self, file_path, content, branch=None, commit_message=None):
        raise NotImplementedError


def _valid_edit_query():
    return (
        "OLD <<<<\n"
        "old text\n"
        ">>>> OLD\n"
        "NEW <<<<\n"
        "new text\n"
        ">>>> NEW\n"
    )


def test_read_file_exception_propagates_as_tool_exception():
    wrapper = _StubCodeWrapper.model_construct()

    with pytest.raises(ToolException, match="Failed to read file missing.txt"):
        wrapper.edit_file("missing.txt", _valid_edit_query())


def test_read_file_exception_message_is_preserved():
    wrapper = _StubCodeWrapper.model_construct()

    with pytest.raises(ToolException) as excinfo:
        wrapper.edit_file("missing.txt", _valid_edit_query())

    assert "File not found: missing.txt" in str(excinfo.value)
