"""Tests for the Bitbucket `delete_file` tool (#3401).

Covers three layers:
  * BitbucketAPIWrapper.delete_file  - wrapper-level orchestration (file existence
    check, error wrapping, commit message default).
  * BitbucketCloudApi.delete_file    - Cloud backend sends the `files` form field
    (with no matching content field) to the `src` endpoint, which Bitbucket Cloud
    interprets as a delete.
  * BitbucketServerApi.delete_file   - Server backend has no native delete-file
    REST endpoint, so it must raise a clear ToolException instead of silently
    failing or corrupting a file.
"""
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.bitbucket.api_wrapper import BitbucketAPIWrapper
from elitea_sdk.tools.bitbucket.cloud_api_wrapper import BitbucketCloudApi, BitbucketServerApi


class FakeBitbucket:
    def __init__(self, contents_by_path: dict, delete_error: Exception = None):
        self._contents_by_path = contents_by_path
        self._delete_error = delete_error
        self.delete_calls = []

    def get_file(self, file_path, branch):
        if file_path not in self._contents_by_path:
            raise ToolException(f"File not found: {file_path}")
        return self._contents_by_path[file_path]

    def delete_file(self, file_path, branch, commit_message=None):
        if self._delete_error:
            raise self._delete_error
        self.delete_calls.append({"file_path": file_path, "branch": branch, "commit_message": commit_message})
        return "deleted"


def _make_wrapper(contents_by_path: dict, delete_error: Exception = None) -> BitbucketAPIWrapper:
    wrapper = BitbucketAPIWrapper.model_construct(branch="main")
    wrapper._bitbucket = FakeBitbucket(contents_by_path, delete_error=delete_error)
    wrapper._active_branch = "main"
    return wrapper


class TestBitbucketAPIWrapperDeleteFile:
    def test_deletes_existing_file(self):
        wrapper = _make_wrapper({"hello.md": "hi"})

        result = wrapper.delete_file("hello.md", "main")

        assert result == "File has been deleted: hello.md."
        assert wrapper._bitbucket.delete_calls == [
            {"file_path": "hello.md", "branch": "main", "commit_message": "Delete hello.md"}
        ]

    def test_uses_provided_commit_message(self):
        wrapper = _make_wrapper({"hello.md": "hi"})

        wrapper.delete_file("hello.md", "main", commit_message="Remove stale doc")

        assert wrapper._bitbucket.delete_calls[0]["commit_message"] == "Remove stale doc"

    def test_falls_back_to_active_branch_when_branch_empty(self):
        wrapper = _make_wrapper({"hello.md": "hi"})

        wrapper.delete_file("hello.md", branch=None)

        assert wrapper._bitbucket.delete_calls[0]["branch"] == "main"

    def test_raises_when_file_does_not_exist(self):
        wrapper = _make_wrapper({"other.md": "hi"})

        with pytest.raises(ToolException, match="not found"):
            wrapper.delete_file("missing.md", "main")

        assert wrapper._bitbucket.delete_calls == []

    def test_wraps_backend_error_from_delete(self):
        wrapper = _make_wrapper({"hello.md": "hi"}, delete_error=RuntimeError("boom"))

        with pytest.raises(ToolException, match="was not deleted"):
            wrapper.delete_file("hello.md", "main")

    def test_propagates_tool_exception_from_backend_unchanged(self):
        backend_error = ToolException("Deleting a file is not supported by the Bitbucket Server REST API.")
        wrapper = _make_wrapper({"hello.md": "hi"}, delete_error=backend_error)

        with pytest.raises(ToolException, match="not supported by the Bitbucket Server REST API"):
            wrapper.delete_file("hello.md", "main")


class TestBitbucketCloudApiDeleteFile:
    def _make_cloud_api(self) -> BitbucketCloudApi:
        api = BitbucketCloudApi.__new__(BitbucketCloudApi)
        api.repository = MagicMock()
        return api

    def test_posts_files_field_without_content_to_mark_deletion(self):
        api = self._make_cloud_api()

        api.delete_file("path/to/file.txt", "main")

        api.repository.post.assert_called_once()
        _, kwargs = api.repository.post.call_args
        assert kwargs["path"] == "src"
        assert kwargs["data"] == {"branch": "main", "files": "path/to/file.txt"}

    def test_includes_commit_message_when_provided(self):
        api = self._make_cloud_api()

        api.delete_file("path/to/file.txt", "main", commit_message="Remove obsolete file")

        _, kwargs = api.repository.post.call_args
        assert kwargs["data"]["message"] == "Remove obsolete file"


class TestBitbucketServerApiDeleteFile:
    def _make_server_api(self) -> BitbucketServerApi:
        return BitbucketServerApi(url="http://bitbucket.example.com", project="PRJ",
                                   repository="repo", username="user", password="pass")

    def test_raises_tool_exception_explaining_platform_limitation(self):
        api = self._make_server_api()

        with pytest.raises(ToolException, match="not supported by the Bitbucket Server REST API"):
            api.delete_file("hello.md", "main")
