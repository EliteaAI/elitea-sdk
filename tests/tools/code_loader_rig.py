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

"""Shared rig for driving CodeIndexerToolkit through a full index_data run.

Lives outside any test_*.py so that several suites can use it without importing one
another: the loader-level guards for #6645 and #5261 both need a toolkit whose file
listing, file contents, stored corpus and staging adapter are all controllable.
"""

import hashlib
import json

from langchain_core.documents import Document

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit
from elitea_sdk.tools.code_indexer_toolkit import CodeIndexerToolkit


def git_blob_sha(content):
    body = content.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest()


PY_BODY = "def alpha():\n    return 1\n"
PY_BODY_HASH = hashlib.sha256(PY_BODY.encode("utf-8")).hexdigest()
SHA_PY_BODY = git_blob_sha(PY_BODY)
CHANGED_BODY = "def alpha():\n    return 2\n"


class IdentityToolkit(CodeIndexerToolkit):
    def _get_files(self, path="", branch=None):
        return list(self.tree)

    def _get_files_with_identity(self, path="", branch=None):
        return dict(self.tree)

    def _read_file(self, file_path, branch, **kwargs):
        self.reads.append(file_path)
        return self.contents.get(file_path, PY_BODY)


class AttestingIdentityToolkit(IdentityToolkit):
    loader_attests_completion = True


class FakeStagingAdapter:
    supports_run_staging = True

    def __init__(self):
        self.calls = []
        self.promote_args = []
        self.stamped = {}
        self.stamp_calls = []

    def ensure_index_runs_table(self, wrapper):
        pass

    def register_index_run(self, wrapper, index_name, run_id, task_id=None, meta_lock_id=None):
        return (True, None)

    def sweep_stale_index_runs(self, wrapper, index_name, stale_before, except_run_id=None):
        return []

    def heartbeat_index_run(self, wrapper, index_name, run_id, meta_id, chunks_written=None):
        pass

    def update_index_meta_keys(self, wrapper, meta_id, run_id, patch):
        stored = wrapper._stored_meta
        if stored is None:
            return 0
        merged = {**stored.get("metadata", {}), **patch}
        object.__setattr__(wrapper, "_stored_meta", {**stored, "metadata": merged})
        return 1

    def stamp_code_identity(self, wrapper, identity_by_row_id):
        self.stamp_calls.append(dict(identity_by_row_id))
        self.stamped.update(identity_by_row_id)
        return len(identity_by_row_id)

    def promote_run(self, wrapper, index_name, run_id, superseded_ids, orphan_ids, damaged_ids):
        self.calls.append("promote")
        self.promote_args.append({"orphan_ids": list(orphan_ids)})
        return "promoted"

    def discard_run(self, wrapper, index_name, run_id):
        self.calls.append("discard")
        return "discarded"

    def get_pending_run_ids(self, wrapper, index_name, include_cancelled=True):
        return []

    def get_index_meta(self, wrapper, index_name):
        return [wrapper._stored_meta] if wrapper._stored_meta else []


def entry_for(filename, rows):
    """rows: list of (row_id, commit_hash, blob_sha), stored as the adapter stores them."""
    return {
        "metadata": {"collection": "x", "filename": filename, "collection_name": "x"},
        "commit_hashes": [commit_hash for _, commit_hash, _ in rows],
        "blob_shas": [blob_sha for _, _, blob_sha in rows],
        "ids": [row_id for row_id, _, _ in rows],
    }


def indexed_entry(filename, blob_sha, commit_hash="stored-hash"):
    return entry_for(filename, [(f"row-{filename}", commit_hash, blob_sha)])


def build_toolkit(monkeypatch, tree, indexed, toolkit_cls=IdentityToolkit, contents=None):
    instance = toolkit_cls.model_construct()
    object.__setattr__(instance, "_stored_meta", None)
    object.__setattr__(instance, "toolkit_id", None)
    object.__setattr__(instance, "max_docs_per_add", 100)
    object.__setattr__(instance, "llm", None)
    object.__setattr__(instance, "tree", dict(tree))
    object.__setattr__(instance, "contents", dict(contents or {}))
    object.__setattr__(instance, "reads", [])
    object.__setattr__(instance, "indexed", dict(indexed))
    object.__setattr__(instance, "indexed_fetches", [])
    object.__setattr__(instance, "vector_adapter", FakeStagingAdapter())
    flushed = []
    object.__setattr__(instance, "flushed", flushed)

    def fake_add_documents(vectorstore=None, documents=None, ids=None):
        flushed.append([d.metadata.get("filename") for d in documents])
        object.__setattr__(
            instance, "_stored_meta",
            {"id": "meta-1", "content": "index_meta_x", "metadata": dict(documents[0].metadata)},
        )
        return [f"row-{n}" for n in range(len(documents))]

    def fake_get_indexed_data(self, index_name):
        self.indexed_fetches.append(index_name)
        return dict(self.indexed)

    monkeypatch.setattr(
        "elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents", fake_add_documents
    )
    monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_index_meta", lambda self, name: self._stored_meta)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_indexed_count", lambda self, name: 7)
    monkeypatch.setattr(BaseIndexerToolkit, "_is_scheduled_run", lambda self: False)
    monkeypatch.setattr(BaseIndexerToolkit, "_emit_index_event",
                        lambda self, name, error=None, state=None: None)
    monkeypatch.setattr(BaseIndexerToolkit, "_clean_index", lambda self, name: None)
    monkeypatch.setattr(BaseIndexerToolkit, "_log_tool_event", lambda self, *a, **kw: None)
    monkeypatch.setattr(CodeIndexerToolkit, "_get_indexed_data", fake_get_indexed_data)
    return instance


def seed_completed_run(toolkit):
    metadata = {
        "collection": "x",
        "state": IndexerKeywords.INDEX_META_COMPLETED.value,
        "indexed": 3,
        "total": 3,
        "report": json.dumps({"status": "ok", "totals": {"indexed": 3}}),
        "skipped": None,
        "error": None,
        "history": json.dumps([{"state": "created"}, {"state": "completed"}]),
    }
    object.__setattr__(toolkit, "_stored_meta", {"id": "meta-1", "content": "c", "metadata": metadata})


def run_index(toolkit, **kwargs):
    return toolkit.index_data(index_name="x", **kwargs)


def totals_of(result):
    return result["report"]["totals"]
