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

"""What a run records for the next one: the scope of its listing and the items it skipped.

Covers the `index_data` wiring of both facts, the cap on the skipped items, and the way the
scope survives the platform's reset of the meta row at every dispatch.
Lives outside tests/tools/index/ because CI skips that directory.
"""

import json
from types import SimpleNamespace
from typing import ClassVar

import pytest

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.tools import base_indexer_toolkit
from elitea_sdk.tools import index_change_check as check
from elitea_sdk.tools.base_indexer_toolkit import (
    BaseIndexerToolkit,
    IndexingStatus,
    _IndexRunState,
)
from tests.tools.test_6585_document_failure_isolation import (
    StagingToolkit,
    build_staged_toolkit,
    make_documents,
)

# The keys the platform keeps when it rebuilds the meta row for a new run
# (INDEX_META_PRESERVED_KEYS of elitea_core utils/application_tools.py).
PLATFORM_PRESERVED_KEYS = ("indexed", "updated", "total", "report", "indexed_chunks")


class ListingToolkit(StagingToolkit):
    """A staged toolkit with a source listing and a scope."""

    def _list_source_items(self, index_configuration):
        return check.SourceListing(items=[])

    def _run_scope_keys(self, index_configuration):
        return {"path": index_configuration.get("path")}


def spy_on_meta_update(monkeypatch, toolkit_cls=ListingToolkit):
    calls = []
    original = BaseIndexerToolkit.index_meta_update

    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(toolkit_cls, "index_meta_update", spy)
    return calls


def save_with(unindexed=(), failed_keys=(), **counters):
    def save(self, base_documents, base_total, chunking_tool, chunking_config, result, index_name=None):
        list(base_documents)
        self._index_run.pipeline_failed_keys.update(failed_keys)
        for key in unindexed:
            self._index_run.unindexed_items[key] = "1"
        result.update(counters)
    return save


class TestIndexDataRecordsRunFacts:

    def run(self, monkeypatch, save, **kwargs):
        toolkit = build_staged_toolkit(monkeypatch, ListingToolkit)
        calls = spy_on_meta_update(monkeypatch)
        monkeypatch.setattr(StagingToolkit, "_base_loader", lambda self, **kw: iter(make_documents("d1", "d2")))
        monkeypatch.setattr(StagingToolkit, "_save_index_generator", save)
        outcome = toolkit.index_data(index_name="x", path="/lib", **kwargs)
        return toolkit, outcome, calls[-1]

    def test_a_completed_run_hands_its_scope_and_skipped_items_to_the_meta_update(self, monkeypatch):
        _, outcome, final = self.run(monkeypatch, save_with(unindexed=("d2",), count=2, docs_count=1, failed_docs=0))

        assert outcome["status"] == IndexingStatus.OK.value
        assert final["last_run_scope"] == {"path": "/lib"}
        assert final["unindexed_items"] == {"d2": "1"}

    def test_a_run_without_staging_records_the_same_facts(self, monkeypatch):
        toolkit = build_staged_toolkit(monkeypatch, ListingToolkit)
        calls = spy_on_meta_update(monkeypatch)
        toolkit.vector_adapter.supports_run_staging = False
        monkeypatch.setattr(ListingToolkit, "_staging_active", lambda self: False)
        monkeypatch.setattr(StagingToolkit, "_base_loader", lambda self, **kw: iter(make_documents("d1", "d2")))
        monkeypatch.setattr(StagingToolkit, "_save_index_generator",
                            save_with(unindexed=("d2",), count=2, docs_count=1, failed_docs=0))

        outcome = toolkit.index_data(index_name="x", path="/lib")

        assert outcome["status"] == IndexingStatus.OK.value
        assert calls[-1]["last_run_scope"] == {"path": "/lib"}
        assert calls[-1]["unindexed_items"] == {"d2": "1"}
        stored = toolkit._stored_meta["metadata"]
        assert json.loads(stored["last_run_scope"]) == {"path": "/lib"}
        assert json.loads(stored["unindexed_items"]) == {"d2": "1"}

    def test_a_failed_run_hands_over_neither_so_the_last_good_values_stay(self, monkeypatch):
        failing_save = save_with(failed_keys=("d1", "d2"), unindexed=("d2",), count=0, docs_count=0, failed_docs=2)

        _, outcome, final = self.run(monkeypatch, failing_save)

        assert outcome["status"] == IndexingStatus.ERROR.value
        assert final["last_run_scope"] is None
        assert final["unindexed_items"] is None

    def test_a_toolkit_without_a_listing_records_no_skipped_items(self, monkeypatch):
        toolkit = build_staged_toolkit(monkeypatch, StagingToolkit)
        calls = spy_on_meta_update(monkeypatch, StagingToolkit)
        monkeypatch.setattr(StagingToolkit, "_base_loader", lambda self, **kw: iter(make_documents("d1")))
        monkeypatch.setattr(StagingToolkit, "_save_index_generator", save_with(count=1, docs_count=1))

        toolkit.index_data(index_name="x")

        assert calls[-1]["unindexed_items"] is None
        assert calls[-1]["last_run_scope"] is None


def test_a_toolkit_with_its_own_index_scope_helper_still_gets_the_base_scope_hook():
    from elitea_sdk.tools.confluence.api_wrapper import ConfluenceAPIWrapper

    assert ConfluenceAPIWrapper.model_construct()._run_scope_keys({"space": "A"}) is None


class TestSkippedItemsCap:

    def toolkit(self):
        instance = ListingToolkit.model_construct()
        object.__setattr__(instance, "_index_run", _IndexRunState(run_id="r1"))
        return instance

    def test_items_beyond_the_cap_are_cut_from_the_record(self, monkeypatch):
        monkeypatch.setattr(base_indexer_toolkit, "MAX_UNINDEXED_ITEMS", 3)
        toolkit = self.toolkit()
        toolkit._index_run.unindexed_items.update({f"k{n}": "1" for n in range(5)})

        assert toolkit._unindexed_items_to_record() == {"k0": "1", "k1": "1", "k2": "1"}

    def test_items_within_the_cap_are_all_recorded(self, monkeypatch):
        monkeypatch.setattr(base_indexer_toolkit, "MAX_UNINDEXED_ITEMS", 3)
        toolkit = self.toolkit()
        toolkit._index_run.unindexed_items.update({"k0": "1", "k1": "1"})

        assert toolkit._unindexed_items_to_record() == {"k0": "1", "k1": "1"}


def dispatch_reset(stored, index_configuration):
    """The row the platform leaves at dispatch: its own keys, the preserved measurements and the history."""
    metadata = stored["metadata"]
    default = {"state": "in_progress", "index_configuration": index_configuration, "created_on": 5.0}
    preserved = {key: metadata[key] for key in PLATFORM_PRESERVED_KEYS if key in metadata}
    history = json.loads(metadata["history"]) + [dict(default)]
    return {**stored, "metadata": {**default, **preserved, "history": json.dumps(history)}}


class TestScopeSurvivesTheDispatchReset:
    """The platform rebuilds the meta row before every run; the scope must still reach the next run."""

    CONFIG: ClassVar[dict[str, str]] = {"path": "/lib"}

    @pytest.fixture
    def toolkit(self, monkeypatch):
        instance = ListingToolkit.model_construct()
        store = {}

        def add_documents(vectorstore=None, documents=None, ids=None):
            store["meta"] = {"id": "meta-1", "content": "index_meta_x", "metadata": dict(documents[0].metadata)}

        monkeypatch.setattr("elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents", add_documents)
        monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
        monkeypatch.setattr(VectorStoreWrapperBase, "get_index_meta", lambda self, name: store.get("meta"))
        monkeypatch.setattr(VectorStoreWrapperBase, "get_indexed_count", lambda self, name: 3)
        monkeypatch.setattr(ListingToolkit, "_staging_active", lambda self: False)
        monkeypatch.setattr(ListingToolkit, "_resolve_initiator", lambda self: None)
        monkeypatch.setattr(ListingToolkit, "_is_scheduled_run", lambda self: False)
        monkeypatch.setattr(ListingToolkit, "_log_tool_event", lambda self, *a, **k: None)
        object.__setattr__(instance, "vectorstore", object())
        object.__setattr__(instance, "toolkit_id", None)
        object.__setattr__(instance, "vector_adapter", SimpleNamespace(supports_run_staging=False))
        object.__setattr__(instance, "store", store)
        return instance

    def start_run(self, toolkit, run_id):
        run = _IndexRunState(run_id=run_id)
        object.__setattr__(toolkit, "_index_run", run)
        toolkit.index_meta_init("x", self.CONFIG)
        return run

    def finish_run(self, toolkit, state, scope=True):
        toolkit.index_meta_update(
            "x", state, 3, last_run_scope=toolkit._index_run.run_scope if scope else None,
        )

    def reset(self, toolkit):
        toolkit.store["meta"] = dispatch_reset(toolkit.store["meta"], self.CONFIG)

    def test_the_next_run_reads_the_scope_the_last_good_run_recorded(self, toolkit):
        self.start_run(toolkit, "r1")
        self.finish_run(toolkit, "completed")
        self.reset(toolkit)
        assert "last_run_scope" not in toolkit.store["meta"]["metadata"]

        second = self.start_run(toolkit, "r2")

        assert second.last_run_scope == {"path": "/lib"}
        assert second.run_scope == second.last_run_scope
        assert second.meta_existed is True

    def test_a_failed_run_in_between_does_not_lose_the_scope(self, toolkit):
        self.start_run(toolkit, "r1")
        self.finish_run(toolkit, "completed")
        self.reset(toolkit)
        self.start_run(toolkit, "r2")
        self.finish_run(toolkit, "failed", scope=False)
        self.reset(toolkit)

        third = self.start_run(toolkit, "r3")

        assert third.last_run_scope == {"path": "/lib"}

    def test_only_the_newest_history_entry_keeps_a_scope(self, toolkit):
        self.start_run(toolkit, "r1")
        self.finish_run(toolkit, "completed")
        self.reset(toolkit)
        self.start_run(toolkit, "r2")
        self.finish_run(toolkit, "completed")

        history = json.loads(toolkit.store["meta"]["metadata"]["history"])

        assert [bool(entry.get("last_run_scope")) for entry in history].count(True) == 1
        assert history[-1]["last_run_scope"] == {"path": "/lib"}

    def test_a_row_the_platform_created_for_a_first_run_counts_as_a_legacy_row(self, toolkit):
        created = {"state": "in_progress", "index_configuration": self.CONFIG, "created_on": 5.0}
        toolkit.store["meta"] = {"id": "meta-1", "content": "index_meta_x",
                                 "metadata": {**created, "history": json.dumps([created])}}

        first = self.start_run(toolkit, "r1")

        assert first.meta_existed is True
        assert first.last_run_scope is None

    @pytest.mark.parametrize("history", ["", "not json", "{}", "[1, null]", None])
    def test_an_unreadable_history_gives_no_scope(self, history):
        assert BaseIndexerToolkit._stored_last_run_scope({"history": history}) is None

    def test_an_entry_of_a_failed_run_is_not_a_source_of_the_scope(self):
        history = json.dumps([{"state": "failed", "last_run_scope": {"path": "/x"}}])

        assert BaseIndexerToolkit._stored_last_run_scope({"history": history}) is None
