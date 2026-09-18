# Copyright (c) 2026 EPAM Systems
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Every git provider hands the loader a per-file identity for free (#6645).

The listing already carries a git blob SHA; the plain path listing must keep
working unchanged so the tools built on it are untouched.
"""

from unittest.mock import MagicMock

import pytest

from elitea_sdk.tools.bitbucket.api_wrapper import BitbucketAPIWrapper
from elitea_sdk.tools.code_indexer_toolkit import CodeIndexerToolkit
from elitea_sdk.tools.github.github_client import GitHubClient
from elitea_sdk.tools.gitlab.api_wrapper import GitLabAPIWrapper


def _tree_entry(path, sha, entry_type="blob"):
    entry = MagicMock()
    entry.type = entry_type
    entry.path = path
    entry.sha = sha
    return entry


def _github_client(entries, truncated=False):
    client = object.__new__(GitHubClient)
    repo = MagicMock()
    repo.full_name = "owner/repo"
    repo.get_branch.return_value.commit.commit.tree.sha = "tree-sha"
    tree = MagicMock()
    tree.tree = list(entries)
    tree.truncated = truncated
    repo.get_git_tree.return_value = tree
    object.__setattr__(client, "_github_repo_instance", repo)
    object.__setattr__(client, "_github_api", MagicMock())
    return client, repo


class TestGitHub:

    def test_the_tree_is_walked_once_for_both_shapes(self):
        client, repo = _github_client([_tree_entry("a.py", "sha-a")])

        client._get_files_with_identity("", "main")

        assert repo.get_git_tree.call_count == 1

    def test_every_blob_keeps_its_sha(self):
        client, _ = _github_client([_tree_entry("a.py", "sha-a"), _tree_entry("b.py", "sha-b")])

        assert client._get_files_with_identity("", "main") == {"a.py": "sha-a", "b.py": "sha-b"}

    def test_directories_are_left_out(self):
        client, _ = _github_client([_tree_entry("src", "sha-dir", entry_type="tree"),
                                    _tree_entry("a.py", "sha-a")])

        assert client._get_files_with_identity("", "main") == {"a.py": "sha-a"}

    def test_a_directory_filter_still_applies(self):
        client, _ = _github_client([_tree_entry("src/a.py", "sha-a"), _tree_entry("b.py", "sha-b")])

        assert client._get_files_with_identity("src", "main") == {"src/a.py": "sha-a"}

    def test_the_plain_listing_still_returns_paths(self):
        client, _ = _github_client([_tree_entry("a.py", "sha-a"), _tree_entry("b.py", "sha-b")])

        assert client._get_files("", "main") == ["a.py", "b.py"]

    def test_an_unresolvable_ref_still_returns_its_error_string(self):
        from github import GithubException

        client, repo = _github_client([])
        repo.get_branch.side_effect = GithubException(404, {"message": "no branch"}, None)
        repo.get_commit.side_effect = GithubException(404, {"message": "no commit"}, None)

        assert isinstance(client._get_files("", "nope"), str)


class TestGitLab:

    def test_the_whole_tree_is_listed_not_just_the_top_level(self):
        wrapper = object.__new__(GitLabAPIWrapper)
        wrapper._get_all_files = MagicMock(return_value=[])

        wrapper._get_files_with_identity(None, "main")

        assert wrapper._get_all_files.call_args.args[1] is True

    def test_blob_ids_become_the_identity(self):
        wrapper = object.__new__(GitLabAPIWrapper)
        wrapper._get_all_files = MagicMock(return_value=[
            {"path": "a.py", "id": "sha-a", "type": "blob"},
            {"path": "src", "id": "sha-dir", "type": "tree"},
            {"path": "b.py", "id": "sha-b", "type": "blob"},
        ])

        assert wrapper._get_files_with_identity(None, "main") == {"a.py": "sha-a", "b.py": "sha-b"}


class TestBitbucketKeepsTodaysBehaviour:

    def test_no_identity_is_claimed(self):
        wrapper = object.__new__(BitbucketAPIWrapper)

        assert wrapper._get_files_with_identity("", "main") is None


class TestTheBaseToolkitClaimsNothing:

    def test_a_toolkit_without_an_override_has_no_identity(self):
        toolkit = object.__new__(CodeIndexerToolkit)

        assert toolkit._get_files_with_identity("", "main") is None
