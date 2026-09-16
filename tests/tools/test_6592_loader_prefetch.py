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

"""Guards the loader/writer overlap in ``index_data`` (#6592).

The source fetch and the embed used to run strictly back to back, because the loader was
materialised with ``list()`` before the first chunk was flushed. Re-broken by anything
that drains the loader eagerly again, lets a zero-document loader reach dedup, reads
``_index_workers`` before the loader has published it, drops the loader's progress
events, or leaks the producer thread. Lives outside tests/tools/index/ because CI skips
that directory.
"""

import contextvars
import json
import threading
import time

import pytest
from langchain_core.documents import Document

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.code_indexer_toolkit import CodeIndexerToolkit
from elitea_sdk.tools.base_indexer_toolkit import (
    BaseIndexerToolkit,
    IndexingStats,
    _LoaderPrefetch,
    _resolve_base_total,
)

BARRIER_TIMEOUT = 5
LOADER_PROLOGUE_SECONDS = 0.25


class PrefetchToolkit(BaseIndexerToolkit):
    def key_fn(self, document: Document):
        return document.metadata.get("id")


class AttestingToolkit(PrefetchToolkit):
    loader_attests_completion = True


class FakeStagingAdapter:
    supports_run_staging = True

    def __init__(self):
        self.calls = []
        self.promote_args = []

    def ensure_index_runs_table(self, wrapper):
        pass

    def register_index_run(self, wrapper, index_name, run_id, task_id=None, meta_lock_id=None):
        return (True, None)

    def sweep_stale_index_runs(self, wrapper, index_name, stale_before):
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


def build_toolkit(monkeypatch, toolkit_cls=PrefetchToolkit, staging=False, max_docs_per_add=100):
    instance = toolkit_cls.model_construct()
    object.__setattr__(instance, "_stored_meta", None)
    object.__setattr__(instance, "toolkit_id", None)
    object.__setattr__(instance, "max_docs_per_add", max_docs_per_add)
    object.__setattr__(instance, "llm", None)
    flushed = []
    object.__setattr__(instance, "flushed", flushed)
    if staging:
        object.__setattr__(instance, "vector_adapter", FakeStagingAdapter())

    def fake_add_documents(vectorstore=None, documents=None, ids=None):
        flushed.append([d.metadata.get("id") for d in documents])
        metadata = dict(documents[0].metadata)
        object.__setattr__(
            instance, "_stored_meta",
            {"id": "meta-1", "content": "index_meta_x", "metadata": metadata},
        )
        return ["row-id"]

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
    if not staging:
        monkeypatch.setattr(BaseIndexerToolkit, "_log_tool_event", lambda self, *a, **kw: None)
    return instance


def seed_completed_run(toolkit, **overrides):
    metadata = {
        "collection": "x",
        "state": IndexerKeywords.INDEX_META_COMPLETED.value,
        "indexed": 191,
        "total": 205,
        "report": json.dumps({"status": "ok", "totals": {"indexed": 179}}),
        "skipped": None,
        "error": None,
        "history": json.dumps([{"state": "created"}, {"state": "completed"}]),
    }
    metadata.update(overrides)
    object.__setattr__(toolkit, "_stored_meta", {"id": "meta-1", "content": "c", "metadata": metadata})


def docs(*ids):
    return [Document(page_content=f"body-{i}", metadata={"id": i, "updated_on": "1"}) for i in ids]


class TestTheLoaderRunsWhileTheWriterFlushes:
    """The issue itself: fetch and embed must overlap, not run back to back."""

    def test_a_flush_happens_before_the_loader_is_exhausted(self, monkeypatch):
        toolkit = build_toolkit(monkeypatch, max_docs_per_add=1)
        barrier = threading.Barrier(2, timeout=BARRIER_TIMEOUT)
        seen = []

        def loader(self, **kwargs):
            yield from docs("d1")
            barrier.wait()
            seen.append("loader-resumed")
            yield from docs("d2")

        def flushing_add(vectorstore=None, documents=None, ids=None):
            batch = [d.metadata.get("id") for d in documents]
            toolkit.flushed.append(batch)
            if "d1" in batch:
                barrier.wait()
                seen.append("writer-released")
            object.__setattr__(
                toolkit, "_stored_meta",
                {"id": "meta-1", "content": "c", "metadata": dict(documents[0].metadata)},
            )
            return ["row-id"]

        monkeypatch.setattr(
            "elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents", flushing_add
        )
        monkeypatch.setattr(PrefetchToolkit, "_base_loader", loader)
        monkeypatch.setattr(PrefetchToolkit, "_reduce_duplicates", lambda self, d, n: d)

        result = toolkit.index_data(index_name="x")

        assert result["status"] == "ok"
        assert "loader-resumed" in seen and "writer-released" in seen

    def test_documents_in_flight_stay_within_the_configured_bound(self, monkeypatch):
        toolkit = build_toolkit(monkeypatch, max_docs_per_add=1)
        monkeypatch.setattr(PrefetchToolkit, "loader_prefetch_depth", 2)
        produced, consumed, gaps = [], [], []

        def loader(self, **kwargs):
            for doc in docs(*[f"d{i}" for i in range(30)]):
                produced.append(doc.metadata["id"])
                yield doc

        def watching_reduce(self, documents, index_name):
            for doc in documents:
                consumed.append(doc.metadata["id"])
                gaps.append(len(produced) - len(consumed))
                yield doc

        monkeypatch.setattr(PrefetchToolkit, "_base_loader", loader)
        monkeypatch.setattr(PrefetchToolkit, "_reduce_duplicates", watching_reduce)
        toolkit.index_data(index_name="x")

        queued_depth, in_the_consumers_hand = 2, 1
        assert max(gaps) <= queued_depth + in_the_consumers_hand, gaps


class TestAZeroDocumentLoaderNeverReachesDedup:
    """An empty stream in _reduce_duplicates leaves seen_keys empty, which nominates
    every previously indexed document as an orphan."""

    def test_dedup_is_not_entered(self, monkeypatch):
        toolkit = build_toolkit(monkeypatch)
        seed_completed_run(toolkit)
        entered = []

        def spying_reduce(self, documents, index_name):
            entered.append(index_name)
            yield from documents

        monkeypatch.setattr(PrefetchToolkit, "_base_loader", lambda self, **kw: iter(()))
        monkeypatch.setattr(PrefetchToolkit, "_reduce_duplicates", spying_reduce)

        result = toolkit.index_data(index_name="x")

        assert entered == []
        assert result["status"] == "error"
        assert "no content" in result["message"]

    def test_no_orphan_is_nominated_for_deletion(self, monkeypatch):
        toolkit = build_toolkit(monkeypatch, toolkit_cls=AttestingToolkit, staging=True)
        monkeypatch.setattr(BaseIndexerToolkit, "_log_tool_event", lambda self, *a, **kw: None)
        seed_completed_run(toolkit)
        deleted = []
        object.__setattr__(toolkit, "vectorstore", type("VS", (), {
            "delete": lambda _self, ids=None: deleted.append(list(ids or []))
        })())
        monkeypatch.setattr(AttestingToolkit, "_base_loader", lambda self, **kw: iter(()))
        monkeypatch.setattr(
            AttestingToolkit, "_get_indexed_data",
            lambda self, name: {"gone": {"metadata": {"collection": "x"}, "ids": ["old-1"]}},
        )

        result = toolkit.index_data(index_name="x")

        assert result["status"] == "error"
        assert toolkit._index_run.orphan_candidate_ids == []
        assert deleted == []
        assert "promote" not in toolkit.vector_adapter.calls
        assert "discard" in toolkit.vector_adapter.calls


class TestLoaderPublishedStateIsVisibleToTheWriter:
    """ADO loaders assign self._index_workers inside the generator body, and the writer
    reads it before pulling any document."""

    def test_index_workers_set_before_the_first_yield_is_read_by_the_writer(self, monkeypatch):
        toolkit = build_toolkit(monkeypatch)
        observed = {}

        def loader(self, **kwargs):
            time.sleep(LOADER_PROLOGUE_SECONDS)
            object.__setattr__(self, "_index_workers", 4)
            yield from docs("d1", "d2")

        original = BaseIndexerToolkit._save_index_generator

        def recording_save(self, base_documents, base_total, *args, **kwargs):
            observed["workers"] = getattr(self, "_index_workers", 1)
            return original(self, base_documents, base_total, *args, **kwargs)

        monkeypatch.setattr(PrefetchToolkit, "_base_loader", loader)
        monkeypatch.setattr(PrefetchToolkit, "_reduce_duplicates", lambda self, d, n: d)
        monkeypatch.setattr(PrefetchToolkit, "_save_index_generator", recording_save)

        toolkit.index_data(index_name="x")

        assert observed["workers"] == 4


class TestLoaderFailuresSurfaceOnTheCaller:
    def test_a_loader_raising_before_the_first_document_reraises(self, monkeypatch):
        toolkit = build_toolkit(monkeypatch)

        def loader(self, **kwargs):
            raise RuntimeError("loader prologue exploded")
            yield  # pragma: no cover

        monkeypatch.setattr(PrefetchToolkit, "_base_loader", loader)

        with pytest.raises(RuntimeError, match="loader prologue exploded"):
            toolkit.index_data(index_name="x")

        assert toolkit._stored_meta["metadata"]["state"] == IndexerKeywords.INDEX_META_FAILED.value

    def test_a_midstream_failure_reraises_after_the_delivered_documents(self, monkeypatch):
        toolkit = build_toolkit(monkeypatch)
        delivered = []

        def loader(self, **kwargs):
            yield from docs("d1", "d2")
            raise RuntimeError("loader died at page 2")

        def watching_reduce(self, documents, index_name):
            for doc in documents:
                delivered.append(doc.metadata["id"])
                yield doc

        monkeypatch.setattr(PrefetchToolkit, "_base_loader", loader)
        monkeypatch.setattr(PrefetchToolkit, "_reduce_duplicates", watching_reduce)

        with pytest.raises(RuntimeError, match="loader died at page 2"):
            toolkit.index_data(index_name="x")

        assert delivered == ["d1", "d2"]


class TestTheProducerThreadDoesNotLeak:
    def _loader_threads(self):
        return [t for t in threading.enumerate() if t.name.startswith("index-loader-")]

    def test_two_runs_on_one_instance_leave_no_thread_behind(self, monkeypatch):
        toolkit = build_toolkit(monkeypatch)
        monkeypatch.setattr(PrefetchToolkit, "_base_loader", lambda self, **kw: iter(docs("d1")))
        monkeypatch.setattr(PrefetchToolkit, "_reduce_duplicates", lambda self, d, n: d)

        assert toolkit.index_data(index_name="x")["status"] == "ok"
        assert toolkit.index_data(index_name="x")["status"] == "ok"

        assert self._loader_threads() == []

    def test_a_consumer_failure_releases_a_producer_blocked_on_a_full_queue(self, monkeypatch):
        toolkit = build_toolkit(monkeypatch)
        monkeypatch.setattr(PrefetchToolkit, "loader_prefetch_depth", 2)

        def loader(self, **kwargs):
            yield from docs(*[f"d{i}" for i in range(50)])

        def exploding_reduce(self, documents, index_name):
            next(iter(documents))
            raise RuntimeError("consumer gave up")
            yield  # pragma: no cover

        monkeypatch.setattr(PrefetchToolkit, "_base_loader", loader)
        monkeypatch.setattr(PrefetchToolkit, "_reduce_duplicates", exploding_reduce)

        with pytest.raises(RuntimeError, match="consumer gave up"):
            toolkit.index_data(index_name="x")

        assert self._loader_threads() == []


class TestLoaderStatsAreStampedAfterTheDrain:
    def test_a_skip_recorded_in_the_loader_epilogue_is_counted(self, monkeypatch):
        toolkit = build_toolkit(monkeypatch)

        def loader(self, **kwargs):
            yield from docs("d1", "d2", "d3")
            self._indexing_stats.files_skipped_empty.add("dropped-in-the-epilogue.py")

        monkeypatch.setattr(PrefetchToolkit, "_base_loader", loader)
        monkeypatch.setattr(PrefetchToolkit, "_reduce_duplicates", lambda self, d, n: d)

        toolkit.index_data(index_name="x")
        stats = toolkit.get_indexing_stats()

        assert stats.items_processed == 3
        assert stats.total_fetched == 4


class TestTheCodeLoaderCounterSurvivesConcurrentDecrements:
    """code_indexer_toolkit assigned items_processed absolutely. Once the loader and the
    writer run at the same time, that assignment erases the writer's decrements."""

    def _code_toolkit(self, files):
        toolkit = CodeIndexerToolkit.model_construct()
        object.__setattr__(toolkit, "llm", None)
        object.__setattr__(toolkit, "active_branch", "main")
        object.__setattr__(toolkit, "_get_files", lambda path, branch: list(files))
        object.__setattr__(toolkit, "_read_file", lambda f, branch: f"content of {f}")
        object.__setattr__(toolkit, "_log_tool_event", lambda *a, **kw: None)
        return toolkit

    def test_a_decrement_taken_mid_load_is_not_overwritten(self):
        files = [f"f{i}.py" for i in range(40)]
        toolkit = self._code_toolkit(files)
        stream = toolkit.loader(branch="main", chunked=False)

        first = next(stream)
        assert first.metadata["filename"] == "f0.py"
        toolkit._track_document_failed("f0.py")
        after_failure = toolkit._indexing_stats.items_processed
        remaining = sum(1 for _ in stream)

        stats = toolkit._indexing_stats
        assert after_failure == 0
        assert remaining == len(files) - 1
        assert stats.items_processed == len(files) - 1, (
            "an absolute assignment in the loader overwrote the writer's decrement"
        )
        assert stats.total_fetched == len(files)


class TestLoaderProgressEventsStillReachTheRun:
    """_log_tool_event dispatches through a ContextVar that a bare thread does not
    inherit, so the loader's progress would vanish from the UI."""

    def test_the_run_context_is_carried_onto_the_producer_thread(self, monkeypatch):
        probe = contextvars.ContextVar("elitea_test_run_probe", default=None)
        probe.set("main-thread-run")
        seen = {}

        def loader(self, **kwargs):
            seen["ctx"] = probe.get()
            yield from docs("d1")

        toolkit = build_toolkit(monkeypatch)
        monkeypatch.setattr(PrefetchToolkit, "_base_loader", loader)
        monkeypatch.setattr(PrefetchToolkit, "_reduce_duplicates", lambda self, d, n: d)

        toolkit.index_data(index_name="x")

        assert seen["ctx"] == "main-thread-run"


class TestBaseTotalRendering:
    def test_an_int_is_passed_through(self):
        assert _resolve_base_total(7) == 7

    def test_a_callable_is_resolved(self):
        assert _resolve_base_total(lambda: 3) == 3

    def test_an_unknown_total_is_none(self):
        assert _resolve_base_total(lambda: None) is None

    def test_save_index_generator_still_accepts_a_literal_int(self, monkeypatch):
        toolkit = build_toolkit(monkeypatch)
        result = {"count": 0, "failed_count": 0, "docs_count": 0, "failed_docs": 0}
        toolkit._save_index_generator(iter(docs("d1", "d2")), 2, None, None, result, index_name="x")
        assert result["docs_count"] == 2


class TestLoaderPrefetchDirectly:
    def test_it_reports_the_yield_count_only_once_exhausted(self):
        with _LoaderPrefetch(iter(docs("d1", "d2")), 8, "unit") as stream:
            assert stream.total() is None
            assert list(stream.documents())
            assert stream.total() == 2
            assert stream.produced_count == 2

    def test_a_non_iterable_fails_on_the_caller(self):
        with pytest.raises(TypeError):
            _LoaderPrefetch(object(), 8, "unit")

    def test_has_documents_is_false_for_an_empty_loader(self):
        with _LoaderPrefetch(iter(()), 8, "unit") as stream:
            assert stream.has_documents() is False
            assert list(stream.documents()) == []
