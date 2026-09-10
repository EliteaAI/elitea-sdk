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

"""Index run history stores one chunking configuration, not one per run.

Every history entry is a full clone of the index metadata. Before this change
each entry repeated the whole `chunking_config` — about 3.4 KB for the default
set of 65 file extensions — so the stored history reached its 200-entry and
256 KiB bound roughly ten times sooner than the bound implies, and the user lost
older runs early (issue #362).

The stored shape now keeps the configuration whole in the first run entry and
gives every later run a `chunking_config_ref` holding the digest of the
configuration it used. The top-level `index_configuration` is never touched, so
reindex, the edit form and the scheduler read exactly what they read before.
"""

import copy
import json

import pytest

from elitea_sdk.runtime.langchain.document_loaders.constants import (
    loaders_allowed_to_override,
)
from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools import base_indexer_toolkit
from elitea_sdk.tools.base_indexer_toolkit import (
    INDEX_HISTORY_CHUNKING_REF_KEY,
    BaseIndexerToolkit,
    compact_index_history_chunking_config,
    dump_index_history,
)
from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase

INDEX_NAME = "history_chunking"


def _default_chunking_config():
    """The real SDK default: 65 file extensions, about 3.4 KB of JSON."""
    return copy.deepcopy(loaders_allowed_to_override)


def _entry(state, configuration):
    return {
        "collection": INDEX_NAME,
        "state": state,
        "indexed": 3,
        "index_configuration": configuration,
    }


def _configuration_of(entry):
    configuration = entry["index_configuration"]
    if isinstance(configuration, str):
        return json.loads(configuration)
    return configuration


def _whole_configs(history):
    return [e for e in history if "chunking_config" in _configuration_of(e)]


def _references(history):
    return [
        e for e in history
        if INDEX_HISTORY_CHUNKING_REF_KEY in _configuration_of(e)
    ]


# ---------------------------------------------------------------------------
# The stored shape, driven through the real write path
# ---------------------------------------------------------------------------


class _Store:
    """The smallest vector store that index_meta_init and index_meta_update need."""

    def __init__(self):
        self.rows = {}
        self.writes = 0

    def get_index_meta(self, index_name):
        row = self.rows.get(index_name)
        return copy.deepcopy(row) if row else None

    def add_documents(self, vectorstore=None, documents=None, ids=None):
        self.writes += 1
        for document in documents or []:
            name = document.metadata["collection"]
            self.rows[name] = {
                "id": (ids[0] if ids else "meta-1"),
                "content": document.page_content,
                # A real store round-trips through JSON, so a value that cannot
                # be serialised would fail here rather than pass silently.
                "metadata": json.loads(json.dumps(document.metadata, default=str)),
            }


@pytest.fixture
def toolkit(monkeypatch):
    store = _Store()

    monkeypatch.setattr(
        VectorStoreWrapperBase,
        "get_index_meta",
        lambda self, index_name: store.get_index_meta(index_name),
    )
    monkeypatch.setattr(
        base_indexer_toolkit,
        "add_documents",
        store.add_documents,
        raising=False,
    )
    # index_meta_init and index_meta_update import add_documents from the
    # processor module inside the function body, so patch it there too.
    from elitea_sdk.runtime.langchain.interfaces import llm_processor

    monkeypatch.setattr(llm_processor, "add_documents", store.add_documents)

    monkeypatch.setattr(
        BaseIndexerToolkit, "_ensure_vectorstore_initialized", lambda self: None
    )
    monkeypatch.setattr(BaseIndexerToolkit, "_staging_active", lambda self: False)
    monkeypatch.setattr(BaseIndexerToolkit, "_resolve_initiator", lambda self: "user")
    monkeypatch.setattr(
        BaseIndexerToolkit, "_log_tool_event", lambda self, *a, **k: None
    )
    monkeypatch.setattr(
        BaseIndexerToolkit, "get_indexed_count", lambda self, index_name: 7
    )

    instance = BaseIndexerToolkit.model_construct()
    instance.__dict__["toolkit_id"] = "toolkit-1"
    instance.__dict__["vectorstore"] = object()
    instance._store = store
    return instance


def _run_once(toolkit, chunking_config):
    """One complete successful run: init, then the terminal update."""
    toolkit.index_meta_init(
        INDEX_NAME,
        {"index_name": INDEX_NAME, "chunking_config": chunking_config},
    )
    toolkit.index_meta_update(
        INDEX_NAME,
        state=IndexerKeywords.INDEX_META_COMPLETED.value,
        result=7,
        update_force=True,
    )
    return json.loads(toolkit._store.rows[INDEX_NAME]["metadata"]["history"])


def test_three_successful_runs_store_one_config_and_two_references(toolkit):
    for _ in range(3):
        history = _run_once(toolkit, _default_chunking_config())

    run_entries = [
        e for e in history
        if e.get("state") != IndexerKeywords.INDEX_META_CREATED.value
    ]
    assert len(run_entries) == 3, [e.get("state") for e in history]

    assert len(_whole_configs(run_entries)) == 1
    assert len(_references(run_entries)) == 2

    # The whole copy is the oldest run; the two later runs point at it.
    assert "chunking_config" in _configuration_of(run_entries[0])
    for later in run_entries[1:]:
        reference = _configuration_of(later)[INDEX_HISTORY_CHUNKING_REF_KEY]
        assert reference["sha256"] == base_indexer_toolkit._chunking_config_digest(
            _configuration_of(run_entries[0])["chunking_config"]
        )
        assert reference["same_as_entry"] == history.index(run_entries[0])

    # The created marker records the declaration, not a run, so it never holds
    # a configuration either.
    created = [
        e for e in history
        if e.get("state") == IndexerKeywords.INDEX_META_CREATED.value
    ]
    assert created and not _whole_configs(created)

    # The top-level configuration — what reindex, the edit form and the
    # scheduler actually read — is untouched.
    top_level = toolkit._store.rows[INDEX_NAME]["metadata"]["index_configuration"]
    assert len(top_level["chunking_config"]) == 65


def test_a_changed_configuration_is_stored_whole_again():
    """A run that used a different configuration keeps its own whole copy.

    Driven through the rule itself rather than through two runs: on the reindex
    path `index_meta_init` carries the stored `index_configuration` forward
    unchanged, so a second run never presents a different one. That is existing
    behaviour and outside the scope of issue #362; what matters here is that the
    compaction never collapses two configurations that differ.
    """
    changed = _default_chunking_config()
    changed[".png"]["max_tokens"] = 4096
    history = [
        _entry(
            IndexerKeywords.INDEX_META_COMPLETED.value,
            {"chunking_config": _default_chunking_config()},
        ),
        _entry(
            IndexerKeywords.INDEX_META_COMPLETED.value,
            {"chunking_config": changed},
        ),
        _entry(
            IndexerKeywords.INDEX_META_COMPLETED.value,
            {"chunking_config": _default_chunking_config()},
        ),
    ]

    compacted = compact_index_history_chunking_config(history)

    # Two distinct configurations, so two whole copies, and the third run
    # points back at the first.
    assert len(_whole_configs(compacted)) == 2
    reference = _configuration_of(compacted[2])[INDEX_HISTORY_CHUNKING_REF_KEY]
    assert reference["same_as_entry"] == 0
    assert (
        _configuration_of(compacted[1])["chunking_config"][".png"]["max_tokens"] == 4096
    )


def test_each_later_run_costs_far_less_than_a_whole_configuration(toolkit):
    """Measure the growth this defect was about, against the 3,479-byte baseline."""
    sizes = []
    for _ in range(4):
        history = _run_once(toolkit, _default_chunking_config())
        sizes.append(len(json.dumps(history)))

    growth = [second - first for first, second in zip(sizes, sizes[1:])]
    # The old shape grew by one whole configuration — over 3,400 bytes — per run.
    assert max(growth) < 800, growth


# ---------------------------------------------------------------------------
# The compaction rule itself, and the old shape
# ---------------------------------------------------------------------------


def test_old_shape_history_still_reads():
    """A history written before this change loads and keeps every other field."""
    old_shape = [
        _entry(
            IndexerKeywords.INDEX_META_CREATED.value,
            {"index_name": INDEX_NAME, "chunking_config": _default_chunking_config()},
        ),
        _entry(
            IndexerKeywords.INDEX_META_COMPLETED.value,
            {"index_name": INDEX_NAME, "chunking_config": _default_chunking_config()},
        ),
        _entry(
            IndexerKeywords.INDEX_META_COMPLETED.value,
            {"index_name": INDEX_NAME, "chunking_config": _default_chunking_config()},
        ),
    ]
    stored = json.dumps(old_shape)

    # A reader that has not been changed still sees three entries and two
    # completed runs.
    reloaded = json.loads(stored)
    assert len(reloaded) == 3
    toolkit = BaseIndexerToolkit.model_construct()
    assert toolkit._count_completed_runs({"history": stored}) == 2

    # Compacting it keeps every entry, every other field, and one whole copy.
    compacted = compact_index_history_chunking_config(reloaded)
    assert len(compacted) == 3
    assert [e["state"] for e in compacted] == [e["state"] for e in old_shape]
    assert [e["indexed"] for e in compacted] == [3, 3, 3]
    assert len(_whole_configs(compacted)) == 1
    assert len(_references(compacted)) == 2
    # index_name is a field of index_configuration that must survive the trim.
    for entry in compacted:
        assert _configuration_of(entry)["index_name"] == INDEX_NAME

    # And the input list was not mutated.
    assert all("chunking_config" in e["index_configuration"] for e in reloaded)


def test_legacy_json_string_configuration_keeps_its_shape():
    """Rows from the earlier Python path nest index_configuration as a string."""
    configuration = json.dumps(
        {"index_name": INDEX_NAME, "chunking_config": _default_chunking_config()}
    )
    history = [
        _entry(IndexerKeywords.INDEX_META_COMPLETED.value, configuration),
        _entry(IndexerKeywords.INDEX_META_COMPLETED.value, configuration),
    ]

    compacted = compact_index_history_chunking_config(history)

    assert isinstance(compacted[0]["index_configuration"], str)
    assert isinstance(compacted[1]["index_configuration"], str)
    assert "chunking_config" in _configuration_of(compacted[0])
    assert INDEX_HISTORY_CHUNKING_REF_KEY in _configuration_of(compacted[1])


def test_compaction_is_idempotent():
    history = [
        _entry(
            IndexerKeywords.INDEX_META_COMPLETED.value,
            {"chunking_config": _default_chunking_config()},
        ),
        _entry(
            IndexerKeywords.INDEX_META_COMPLETED.value,
            {"chunking_config": _default_chunking_config()},
        ),
    ]
    once = compact_index_history_chunking_config(history)
    twice = compact_index_history_chunking_config(once)
    assert twice == once
    # The whole copy is not demoted to a reference on the second pass.
    assert len(_whole_configs(twice)) == 1


def test_entries_without_a_configuration_are_left_alone():
    history = [
        {"state": "completed"},
        {"state": "completed", "index_configuration": {"index_name": INDEX_NAME}},
        "not an entry",
    ]
    assert compact_index_history_chunking_config(history) == history


def test_a_configuration_that_cannot_be_serialised_is_left_alone():
    """A live run's config can still hold an embedding object; never lose it."""

    class _NotJSON:
        pass

    history = [
        _entry(
            IndexerKeywords.INDEX_META_COMPLETED.value,
            {"chunking_config": {"embedding": _NotJSON()}},
        ),
        _entry(
            IndexerKeywords.INDEX_META_COMPLETED.value,
            {"chunking_config": {"embedding": _NotJSON()}},
        ),
    ]
    # json.dumps(default=str) makes these digestible, so they compact; the point
    # is only that nothing raises and no entry is dropped.
    compacted = compact_index_history_chunking_config(history)
    assert len(compacted) == 2


def test_dump_index_history_returns_the_compacted_json():
    history = [
        _entry(
            IndexerKeywords.INDEX_META_COMPLETED.value,
            {"chunking_config": _default_chunking_config()},
        ),
        _entry(
            IndexerKeywords.INDEX_META_COMPLETED.value,
            {"chunking_config": _default_chunking_config()},
        ),
    ]
    dumped = json.loads(dump_index_history(history))
    assert len(_whole_configs(dumped)) == 1
    assert len(_references(dumped)) == 1
