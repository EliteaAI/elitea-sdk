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

"""ADO Repos content identity and repository scoping (#6645).

The blob object id is per file; the commit id in the same response is per listing,
so using it would make every file look unchanged until any commit lands, and then
all of them changed at once.
"""

from unittest.mock import MagicMock

import pytest

from elitea_sdk.tools.ado.repos.repos_wrapper import ReposApiWrapper

LISTING_COMMIT_ID = "cccccccccccccccccccccccccccccccccccccccc"


def _blob(path, object_id):
    item = MagicMock()
    item.git_object_type = "blob"
    item.path = path
    item.object_id = object_id
    item.commit_id = LISTING_COMMIT_ID
    return item


def _wrapper(repositories=(), repository_id="alpha-id", searchable_repository_name="alpha",
             items=None):
    instance = ReposApiWrapper.model_construct()
    object.__setattr__(instance, "repositories", list(repositories))
    object.__setattr__(instance, "repository_id", repository_id)
    object.__setattr__(instance, "searchable_repository_name", searchable_repository_name)
    object.__setattr__(instance, "project", "Digital")
    object.__setattr__(instance, "base_branch", "main")
    object.__setattr__(instance, "active_branch", "main")
    client = MagicMock()
    client.get_items.return_value = list(items or [])
    object.__setattr__(instance, "ado_client_instance", client)
    return instance


class TestTheListingIdentity:

    def test_each_file_gets_its_own_blob_object_id(self):
        wrapper = _wrapper(items=[_blob("/a.py", "sha-a"), _blob("/b.py", "sha-b")])

        identities = wrapper._get_files_with_identity()

        assert identities == {"/a.py": "sha-a", "/b.py": "sha-b"}

    def test_the_listing_commit_id_is_not_used_as_the_identity(self):
        wrapper = _wrapper(items=[_blob("/a.py", "sha-a"), _blob("/b.py", "sha-b")])

        identities = wrapper._get_files_with_identity()

        assert LISTING_COMMIT_ID not in identities.values()

    def test_the_plain_listing_still_returns_paths(self):
        wrapper = _wrapper(items=[_blob("/a.py", "sha-a"), _blob("/b.py", "sha-b")])

        assert wrapper._get_files() == ["/a.py", "/b.py"]

    def test_the_whole_tree_is_listed_not_just_the_top_level(self):
        wrapper = _wrapper(items=[])

        wrapper._get_files_with_identity()

        assert wrapper._client.get_items.call_args.kwargs["recursion_level"] == "Full"

    def test_folders_are_left_out(self):
        folder = MagicMock()
        folder.git_object_type = "tree"
        folder.path = "/src"
        wrapper = _wrapper(items=[folder, _blob("/a.py", "sha-a")])

        assert wrapper._get_files_with_identity() == {"/a.py": "sha-a"}
