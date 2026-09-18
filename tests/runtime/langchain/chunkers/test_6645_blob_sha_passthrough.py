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

"""The provider-supplied content identity must survive chunking (#6645).

The code path rebuilds chunk metadata from a whitelist, so a key that is not
forwarded there is dropped in silence and the unchanged skip stops working with
no failing behaviour anywhere - only a larger bill.
"""

import pytest
from langchain_core.documents import Document

from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit
from elitea_sdk.tools.chunkers.universal_chunker import universal_chunker

BLOB_SHA = "9d2a1f6c0b7e4a538c1d6e2f0a9b8c7d6e5f4a3b"

FILES_BY_ROUTE = {
    "code-parsed": ("module.py", "def alpha():\n    return 1\n"),
    "code-unknown-language": ("deploy.sh", "echo hi\n"),
    "code-unknown-language-sql": ("schema.sql", "SELECT 1;\n"),
    "code-unknown-language-header": ("api.h", "int alpha(void);\n"),
    "markdown": ("notes.md", "# Title\n\nSome prose that is long enough to chunk.\n"),
    "json": ("data.json", '{"alpha": 1, "beta": 2}'),
    "text": ("notes.txt", "plain body text\n"),
}


def document_for(file_path, content):
    return Document(
        page_content=content,
        metadata={
            "file_path": file_path,
            "filename": file_path,
            "source": file_path,
            "commit_hash": "content-hash",
            "blob_sha": BLOB_SHA,
        },
    )


@pytest.mark.parametrize("route", sorted(FILES_BY_ROUTE))
def test_every_chunker_route_keeps_the_blob_sha(route):
    file_path, content = FILES_BY_ROUTE[route]
    chunks = list(universal_chunker(iter([document_for(file_path, content)])))

    assert chunks, f"the {route} route produced no chunks to assert on"
    assert all(chunk.metadata.get("blob_sha") == BLOB_SHA for chunk in chunks)


def test_the_code_routes_cover_both_language_branches():
    languages = set()
    for route, (file_path, content) in FILES_BY_ROUTE.items():
        if not route.startswith("code"):
            continue
        for chunk in universal_chunker(iter([document_for(file_path, content)])):
            languages.add(chunk.metadata.get("language"))

    assert "unknown" in languages and languages - {"unknown"}


@pytest.mark.parametrize("route", sorted(FILES_BY_ROUTE))
def test_a_document_without_an_identity_gains_no_empty_one(route):
    file_path, content = FILES_BY_ROUTE[route]
    document = document_for(file_path, content)
    del document.metadata["blob_sha"]

    chunks = list(universal_chunker(iter([document])))

    assert chunks
    assert all("blob_sha" not in chunk.metadata for chunk in chunks)


def test_the_identity_is_not_stripped_before_it_reaches_the_store():
    assert "blob_sha" not in BaseIndexerToolkit._remove_metadata_keys(BaseIndexerToolkit)
