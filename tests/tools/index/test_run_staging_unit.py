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

"""Unit tests for the pending-run staging mechanism (issue #6232).

Covers the read filter's legacy-row-visibility shape, staging-key
normalization, promote-set assembly, flush accounting, the registration
refusal, the count-honesty rules in index_meta_update, and the promote
outcome handling in index_data — all without a database.
"""

import copy
import json
import time

import pytest
from langchain_core.documents import Document

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.base_indexer_toolkit import (
    BaseIndexerToolkit,
    IDLESS_STAGING_KEY,
    IndexRunRefusedError,
    IndexingStatus,
    TASK_DISCONNECTED_TIMEOUT_DEFAULT,
    _IndexRunState,
    render_report_text,
    snapshot_loader_metadata,
)

RUN_ID_KEY = IndexerKeywords.RUN_ID.value

# The autouse fixture below replaces this method class-wide; the no-cache tests
# need the real implementation.
_REAL_GET_PENDING_RUN_IDS = VectorStoreWrapperBase.get_pending_run_ids


class StagingToolkit(BaseIndexerToolkit):
    """Minimal concrete toolkit for exercising the staging mechanism."""

    def key_fn(self, document: Document):
        return document.metadata.get('id')


def make_toolkit(pending_run_ids=None):
    instance = StagingToolkit.model_construct()
    object.__setattr__(instance, "_pending_stub", list(pending_run_ids or []))
    return instance


@pytest.fixture(autouse=True)
def stub_pending_fetch(monkeypatch):
    monkeypatch.setattr(
        VectorStoreWrapperBase, "get_pending_run_ids",
        lambda self, index_name: list(getattr(self, "_pending_stub", [])),
    )


class TestCollectionFilterPendingExclusion:
    def test_steady_state_filter_is_byte_identical_to_today(self):
        toolkit = make_toolkit()

        built = toolkit._build_collection_filter({}, "idx")

        assert built == {
            "$and": [
                {"collection": {"$eq": "idx"}},
                {"$or": [
                    {"type": {"$exists": False}},
                    {"type": {"$ne": "index_meta"}},
                ]},
            ]
        }

    def test_pending_clause_lands_in_the_populated_filter_branch(self):
        toolkit = make_toolkit(pending_run_ids=["r1", "r2"])

        built = toolkit._build_collection_filter({}, "idx")

        assert built["$and"][2] == {"$or": [
            {RUN_ID_KEY: {"$exists": False}},
            {RUN_ID_KEY: {"$nin": ["r1", "r2"]}},
        ]}

    def test_pending_clause_lands_in_the_empty_filter_branch_too(self):
        # A bare $nin is NULL-false for legacy rows: without the $exists:False
        # disjunct in BOTH branches every pre-mechanism row would vanish.
        toolkit = make_toolkit(pending_run_ids=["r1"])

        built = toolkit._build_collection_filter({}, "")

        assert built == {"$and": [
            {"$or": [
                {"type": {"$exists": False}},
                {"type": {"$ne": "index_meta"}},
            ]},
            {"$or": [
                {RUN_ID_KEY: {"$exists": False}},
                {RUN_ID_KEY: {"$nin": ["r1"]}},
            ]},
        ]}

    def test_empty_filter_without_pending_keeps_the_bare_type_exclusion(self):
        toolkit = make_toolkit()

        built = toolkit._build_collection_filter({}, "")

        assert built == {"$or": [
            {"type": {"$exists": False}},
            {"type": {"$ne": "index_meta"}},
        ]}


class TestStagingKey:
    def test_parent_wins_over_own_id(self):
        assert BaseIndexerToolkit._staging_key(
            {IndexerKeywords.PARENT.value: "p1", "id": "dep-9"}
        ) == "p1"

    def test_own_id_when_no_parent(self):
        assert BaseIndexerToolkit._staging_key({"id": 42}) == "42"

    def test_filename_fallback_for_code_chunks(self):
        assert BaseIndexerToolkit._staging_key({"filename": "src/a.py"}) == "src/a.py"

    def test_idless_documents_share_the_excluded_key(self):
        assert BaseIndexerToolkit._staging_key({}) == IDLESS_STAGING_KEY


class TestPromoteSetAssembly:
    def make_run(self):
        return _IndexRunState(run_id="abc123abc123")

    def toolkit_with(self, run):
        toolkit = make_toolkit()
        object.__setattr__(toolkit, "_index_run", run)
        return toolkit

    def test_fully_flushed_doc_commits_its_staged_ids(self):
        run = self.make_run()
        run.staged_removal_ids["doc1"] = {"old-1", "old-2"}
        run.pending_chunk_counts["doc1"] = 0

        superseded, damaged, orphans = self.toolkit_with(run)._assemble_promote_sets()

        assert sorted(superseded) == ["old-1", "old-2"]
        assert damaged == []
        assert orphans == []

    def test_damaged_doc_keeps_old_rows_and_loses_its_partial_new_rows(self):
        run = self.make_run()
        run.staged_removal_ids["doc1"] = {"old-1"}
        run.pending_chunk_counts["doc1"] = 3
        run.damaged_keys.add("doc1")
        run.recorded_row_pks["doc1"] = ["new-1", "new-2"]

        superseded, damaged, orphans = self.toolkit_with(run)._assemble_promote_sets()

        assert superseded == []
        assert sorted(damaged) == ["new-1", "new-2"]

    def test_pipeline_failed_doc_never_commits(self):
        run = self.make_run()
        run.staged_removal_ids["doc1"] = {"old-1"}
        run.pipeline_failed_keys.add("doc1")

        superseded, damaged, orphans = self.toolkit_with(run)._assemble_promote_sets()

        assert superseded == []

    def test_zero_chunk_doc_commits_as_a_legitimate_supersede(self):
        run = self.make_run()
        run.staged_removal_ids["blanked-page"] = {"old-1"}

        superseded, damaged, orphans = self.toolkit_with(run)._assemble_promote_sets()

        assert superseded == ["old-1"]

    def test_unflushed_chunks_block_the_commit(self):
        run = self.make_run()
        run.staged_removal_ids["doc1"] = {"old-1"}
        run.pending_chunk_counts["doc1"] = 2

        superseded, damaged, orphans = self.toolkit_with(run)._assemble_promote_sets()

        assert superseded == []

    def test_orphans_stay_inert_without_class_level_attestation(self):
        run = self.make_run()
        run.orphan_candidate_ids = ["gone-1"]
        run.loader_attested = True

        superseded, damaged, orphans = self.toolkit_with(run)._assemble_promote_sets()

        # loader_attests_completion defaults False on BaseIndexerToolkit and no
        # toolkit opts in: orphan propagation must be scaffolded but inert.
        assert orphans == []

    def test_orphans_need_the_per_run_attestation_too(self, monkeypatch):
        run = self.make_run()
        run.orphan_candidate_ids = ["gone-1"]
        toolkit = self.toolkit_with(run)
        monkeypatch.setattr(StagingToolkit, "loader_attests_completion", True)

        superseded, damaged, orphans = toolkit._assemble_promote_sets()

        assert orphans == []
        run.loader_attested = True
        assert toolkit._assemble_promote_sets()[2] == ["gone-1"]


class TestFlushAccounting:
    def test_successful_flush_records_pks_and_settles_counts(self):
        run = _IndexRunState(run_id="r")
        run.pending_chunk_counts = {"doc1": 2}
        toolkit = make_toolkit()

        toolkit._record_flushed_chunk(run, ["doc1", "doc1"], ["pk-1", "pk-2"])

        assert run.pending_chunk_counts["doc1"] == 0
        assert run.recorded_row_pks["doc1"] == ["pk-1", "pk-2"]

    def test_id_count_mismatch_damages_the_whole_batch(self):
        run = _IndexRunState(run_id="r")
        toolkit = make_toolkit()

        toolkit._record_flushed_chunk(run, ["doc1", "doc2"], ["pk-1"])

        assert run.damaged_keys == {"doc1", "doc2"}
        assert run.recorded_row_pks == {}

    def test_idless_chunks_are_never_damage_attributed(self):
        run = _IndexRunState(run_id="r")

        BaseIndexerToolkit._mark_batch_damaged(run, ["doc1", IDLESS_STAGING_KEY])

        assert run.damaged_keys == {"doc1"}


class FakeStagingAdapter:
    supports_run_staging = True

    def __init__(self):
        self.register_result = (True, None)
        self.promote_outcome = "promoted"
        self.calls = []
        self.pending = []
        self.sweeps = []

    def ensure_index_runs_table(self, wrapper):
        self.calls.append("ensure")

    def register_index_run(self, wrapper, index_name, run_id, task_id=None, meta_lock_id=None):
        self.calls.append("register")
        return self.register_result

    def sweep_stale_index_runs(self, wrapper, index_name, stale_before):
        self.calls.append("sweep")
        self.sweeps.append((index_name, stale_before))
        return []

    def heartbeat_index_run(self, wrapper, index_name, run_id, meta_id):
        self.calls.append("heartbeat")

    def promote_run(self, wrapper, index_name, run_id, superseded_ids, orphan_ids, damaged_ids):
        self.calls.append("promote")
        return self.promote_outcome

    def discard_run(self, wrapper, index_name, run_id):
        self.calls.append("discard")
        return "discarded"

    def get_pending_run_ids(self, wrapper, index_name, include_cancelled=True):
        return list(self.pending)

    def get_index_meta(self, wrapper, index_name):
        return [wrapper._stored_meta] if wrapper._stored_meta else []


@pytest.fixture
def staged_toolkit(monkeypatch):
    instance = StagingToolkit.model_construct()
    adapter = FakeStagingAdapter()
    object.__setattr__(instance, "vector_adapter", adapter)
    object.__setattr__(instance, "_stored_meta", None)
    object.__setattr__(instance, "toolkit_id", None)
    written = []
    object.__setattr__(instance, "written", written)
    emitted = []
    object.__setattr__(instance, "emitted", emitted)

    def fake_add_documents(vectorstore=None, documents=None, ids=None):
        metadata = dict(documents[0].metadata)
        written.append(metadata)
        object.__setattr__(
            instance, "_stored_meta",
            {"id": "meta-1", "content": "index_meta_x", "metadata": metadata},
        )
        return ["meta-row-id"]

    monkeypatch.setattr(
        "elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents", fake_add_documents
    )
    monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
    monkeypatch.setattr(
        VectorStoreWrapperBase, "get_index_meta",
        lambda self, name: self._stored_meta,
    )
    monkeypatch.setattr(VectorStoreWrapperBase, "get_indexed_count", lambda self, name: 965)
    monkeypatch.setattr(StagingToolkit, "_is_scheduled_run", lambda self: False)
    monkeypatch.setattr(StagingToolkit, "_log_tool_event", lambda self, *a, **kw: None)
    monkeypatch.setattr(
        StagingToolkit, "_emit_index_event",
        lambda self, name, error=None, state=None: self.emitted.append(
            {"error": error, "state": state}
        ),
    )
    monkeypatch.setattr(StagingToolkit, "_reduce_duplicates", lambda self, docs, name: docs)
    return instance


def seed_completed_meta(toolkit):
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
    object.__setattr__(
        toolkit, "_stored_meta", {"id": "meta-1", "content": "c", "metadata": metadata}
    )


def run_index_data(toolkit, monkeypatch, documents=None, save=None):
    documents = documents if documents is not None else [
        Document(page_content="body", metadata={"id": "d1", "updated_on": "1"})
    ]
    monkeypatch.setattr(StagingToolkit, "_base_loader", lambda self, **kwargs: iter(documents))

    def default_save(self, base_documents, base_total, chunking_tool, chunking_config, result, index_name=None):
        result["count"] = len(list(base_documents))
        result["docs_count"] = result["count"]

    monkeypatch.setattr(StagingToolkit, "_save_index_generator", save or default_save)
    return toolkit.index_data(index_name="x")


class TestIndexDataPromoteOutcomes:
    def test_successful_run_promotes_before_the_terminal_write(self, staged_toolkit, monkeypatch):
        seed_completed_meta(staged_toolkit)

        outcome = run_index_data(staged_toolkit, monkeypatch)

        adapter = staged_toolkit.vector_adapter
        assert outcome["status"] == IndexingStatus.OK.value
        assert adapter.calls.index("promote") < len(adapter.calls)
        assert staged_toolkit.written[-1]["state"] == IndexerKeywords.INDEX_META_COMPLETED.value
        assert staged_toolkit.emitted[-1]["state"] == IndexerKeywords.INDEX_META_COMPLETED.value
        assert "discard" not in adapter.calls

    def test_aborted_cancelled_skips_terminal_write_and_success_emit(self, staged_toolkit, monkeypatch):
        seed_completed_meta(staged_toolkit)
        staged_toolkit.vector_adapter.promote_outcome = "aborted-cancelled"
        writes_before_run = len(staged_toolkit.written)

        outcome = run_index_data(staged_toolkit, monkeypatch)

        assert outcome["status"] == IndexingStatus.ERROR.value
        assert "cancelled" in outcome["message"]
        assert "previously indexed data remains available" in outcome["message"]
        # Only the reindex-branch reset write may land — never the terminal write
        # that would overwrite cancel's snapshot with the discarded run's counts.
        assert len(staged_toolkit.written) == writes_before_run + 1
        # Only the start emit fires; the terminal emit is skipped on every aborted branch.
        assert len(staged_toolkit.emitted) == 1
        assert staged_toolkit.emitted[0]["state"] is None

    def test_failed_run_discards_and_never_promotes(self, staged_toolkit, monkeypatch):
        seed_completed_meta(staged_toolkit)

        def failing_save(self, base_documents, base_total, chunking_tool, chunking_config, result, index_name=None):
            list(base_documents)
            result["count"] = 5
            result["failed_count"] = 5
            result.setdefault("errors", []).append("pgvector down")

        outcome = run_index_data(staged_toolkit, monkeypatch, save=failing_save)

        adapter = staged_toolkit.vector_adapter
        assert outcome["status"] == IndexingStatus.ERROR.value
        assert "promote" not in adapter.calls
        assert "discard" in adapter.calls
        stored = staged_toolkit.written[-1]
        assert stored["state"] == IndexerKeywords.INDEX_META_FAILED.value
        # The error report keeps the retained counts untouched.
        assert stored["indexed"] == 191
        assert stored["total"] == 205
        assert staged_toolkit.emitted[-1]["state"] == IndexerKeywords.INDEX_META_FAILED.value
        assert staged_toolkit.emitted[-1]["error"]

    def test_damaged_doc_downgrades_a_clean_ladder_to_partly_indexed(self, staged_toolkit, monkeypatch):
        seed_completed_meta(staged_toolkit)

        def save_with_damage(self, base_documents, base_total, chunking_tool, chunking_config, result, index_name=None):
            list(base_documents)
            result["count"] = 5
            result["docs_count"] = 2
            self._index_run.pipeline_failed_keys.add("doc-broken")

        outcome = run_index_data(staged_toolkit, monkeypatch, save=save_with_damage)

        assert outcome["status"] == IndexingStatus.PARTLY_INDEXED.value
        assert staged_toolkit.written[-1]["state"] == IndexerKeywords.INDEX_META_PARTLY_OK.value
        assert "promote" in staged_toolkit.vector_adapter.calls

    def test_empty_loader_over_previous_index_fails_without_promoting(self, staged_toolkit, monkeypatch):
        seed_completed_meta(staged_toolkit)

        outcome = run_index_data(staged_toolkit, monkeypatch, documents=[])

        adapter = staged_toolkit.vector_adapter
        assert outcome["status"] == IndexingStatus.ERROR.value
        assert "promote" not in adapter.calls
        assert "discard" in adapter.calls
        stored = staged_toolkit.written[-1]
        assert stored["state"] == IndexerKeywords.INDEX_META_FAILED.value
        assert stored["indexed"] == 191
        assert stored["total"] == 205


class TestRegistrationRefusal:
    def test_refusal_skips_meta_write_and_failed_emit(self, staged_toolkit, monkeypatch):
        seed_completed_meta(staged_toolkit)
        staged_toolkit.vector_adapter.register_result = (
            False, {"run_id": "other", "heartbeat": __import__("time").time(), "started_on": 0.0},
        )
        writes_before_run = len(staged_toolkit.written)

        with pytest.raises(IndexRunRefusedError, match="already in progress"):
            run_index_data(staged_toolkit, monkeypatch)

        assert len(staged_toolkit.written) == writes_before_run
        assert staged_toolkit.emitted == []

    def test_refused_run_never_rewrites_the_live_runs_meta_row(self, staged_toolkit, monkeypatch):
        seed_completed_meta(staged_toolkit)
        original_created_on = staged_toolkit._stored_meta["metadata"].get("created_on")
        staged_toolkit.vector_adapter.register_result = (False, None)

        with pytest.raises(IndexRunRefusedError):
            run_index_data(staged_toolkit, monkeypatch)

        assert staged_toolkit._stored_meta["metadata"].get("created_on") == original_created_on

    def test_stale_blocker_produces_the_distinct_message(self, staged_toolkit, monkeypatch):
        seed_completed_meta(staged_toolkit)
        staged_toolkit.vector_adapter.register_result = (
            False, {"run_id": "dead", "heartbeat": 1.0, "started_on": 1.0},
        )

        with pytest.raises(IndexRunRefusedError, match="stale run marker"):
            run_index_data(staged_toolkit, monkeypatch)


class TestFreshIndexReclaim:
    """A pending row can outlive its meta row (a crash before the row is
    written, or an index removal), and the partial unique index has no
    staleness carve-out — without a sweep the name would refuse every run."""

    def test_fresh_branch_sweeps_before_registering(self, staged_toolkit, monkeypatch):
        run_index_data(staged_toolkit, monkeypatch)

        calls = staged_toolkit.vector_adapter.calls
        assert calls.index("sweep") < calls.index("register")

    def test_fresh_sweep_uses_the_default_staleness_timeout(self, staged_toolkit, monkeypatch):
        run_index_data(staged_toolkit, monkeypatch)

        index_name, stale_before = staged_toolkit.vector_adapter.sweeps[0]
        assert index_name == "x"
        assert time.time() - stale_before == pytest.approx(
            TASK_DISCONNECTED_TIMEOUT_DEFAULT, abs=10
        )

    def test_fresh_run_starts_once_the_sweep_reclaims_the_stale_row(self, staged_toolkit, monkeypatch):
        adapter = staged_toolkit.vector_adapter
        adapter.register_result = (False, {"run_id": "dead", "heartbeat": 1.0, "started_on": 1.0})
        original_sweep = adapter.sweep_stale_index_runs

        def reclaiming_sweep(wrapper, index_name, stale_before):
            original_sweep(wrapper, index_name, stale_before)
            adapter.register_result = (True, None)
            return ["dead"]

        adapter.sweep_stale_index_runs = reclaiming_sweep

        run_index_data(staged_toolkit, monkeypatch)

        assert "promote" in adapter.calls


class TestPipelineRetryIsolation:
    """The pipeline consumes the document's own metadata; without a snapshot the
    retry starts from attempt 1's leftovers — duplicated dependent ids, extra
    round-trips and a silent zero-chunk re-parse once the content bytes are gone."""

    def run_two_attempts(self, monkeypatch, failures=1):
        toolkit = StagingToolkit.model_construct()
        object.__setattr__(toolkit, "max_docs_per_add", 100)
        seen = []
        attempts = {"count": 0}

        def fake_extend_data(self, documents):
            base_doc = next(iter(documents))
            seen.append(copy.deepcopy(base_doc.metadata))
            base_doc.metadata.pop("content_in_bytes", None)
            base_doc.metadata.setdefault("dependent_docs", []).append("dep-1")
            attempts["count"] += 1
            if attempts["count"] <= failures:
                raise RuntimeError("transient parse failure")
            return iter([base_doc])

        monkeypatch.setattr(StagingToolkit, "_extend_data", fake_extend_data)
        monkeypatch.setattr(StagingToolkit, "_collect_dependencies", lambda self, docs: docs)
        monkeypatch.setattr(
            StagingToolkit, "_apply_loaders_chunkers",
            lambda self, docs, chunking_tool=None, chunking_config=None: docs,
        )
        monkeypatch.setattr(StagingToolkit, "_clean_metadata", lambda self, docs: docs)
        monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
        monkeypatch.setattr(StagingToolkit, "_log_tool_event", lambda self, *a, **kw: None)
        monkeypatch.setattr(StagingToolkit, "index_meta_update", lambda self, *a, **kw: None)
        monkeypatch.setattr(
            "elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents",
            lambda vectorstore=None, documents=None, ids=None: ["row-1"],
        )

        base_doc = Document(
            page_content="body",
            metadata={"id": "d1", "updated_on": "1", "content_in_bytes": b"payload"},
        )
        toolkit._save_index_generator(
            iter([base_doc]), 1, None, None,
            {"count": 0, "docs_count": 0, "errors": []}, index_name="x"
        )
        return seen

    def test_both_attempts_start_from_the_loaders_metadata(self, monkeypatch):
        seen = self.run_two_attempts(monkeypatch)

        assert len(seen) == 2
        assert seen[1] == seen[0]

    def test_dependent_ids_are_not_accumulated_across_attempts(self, monkeypatch):
        seen = self.run_two_attempts(monkeypatch)

        assert "dependent_docs" not in seen[1]

    def test_content_bytes_are_restored_for_the_retry(self, monkeypatch):
        seen = self.run_two_attempts(monkeypatch)

        assert seen[1]["content_in_bytes"] == b"payload"

    def test_an_uncopyable_metadata_value_does_not_fail_the_document(self, monkeypatch):
        class Uncopyable:
            def __deepcopy__(self, memo):
                raise TypeError("cannot copy")

        document = Document(page_content="body", metadata={"handle": Uncopyable()})

        assert snapshot_loader_metadata(document) is None


class TestForeignLiveRunSkip:
    def test_generic_failure_with_foreign_live_run_skips_the_meta_flip(self, staged_toolkit, monkeypatch):
        seed_completed_meta(staged_toolkit)
        writes_before_run = len(staged_toolkit.written)

        def exploding_save(self, base_documents, base_total, chunking_tool, chunking_config, result, index_name=None):
            self.vector_adapter.pending = ["someone-else"]
            raise RuntimeError("boom mid-run")

        with pytest.raises(RuntimeError, match="boom mid-run"):
            run_index_data(staged_toolkit, monkeypatch, save=exploding_save)

        adapter = staged_toolkit.vector_adapter
        assert "discard" in adapter.calls
        # The reset write from init is the only one; the FAILED flip is skipped.
        assert len(staged_toolkit.written) == writes_before_run + 1
        # The failure still surfaces through the event channel.
        assert staged_toolkit.emitted[-1]["state"] == IndexerKeywords.INDEX_META_FAILED.value
        assert "boom mid-run" in staged_toolkit.emitted[-1]["error"]


class TestCountHonesty:
    def test_throttled_update_writes_run_chunks_not_updated(self, staged_toolkit):
        seed_completed_meta(staged_toolkit)
        staged_toolkit._stored_meta["metadata"]["updated"] = 777

        staged_toolkit.index_meta_update(
            "x", IndexerKeywords.INDEX_META_IN_PROGRESS.value, 12, update_force=False
        )

        stored = staged_toolkit.written[-1]
        assert stored["run_chunks"] == 12
        assert stored["updated"] == 777
        assert stored["indexed"] == 191

    def test_reportless_terminal_success_still_equates_indexed_with_chunks(self, staged_toolkit):
        seed_completed_meta(staged_toolkit)

        staged_toolkit.index_meta_update(
            "x", IndexerKeywords.INDEX_META_COMPLETED.value, 12, update_force=True
        )

        assert staged_toolkit.written[-1]["indexed"] == 965

    def test_reportless_terminal_failure_leaves_indexed_alone(self, staged_toolkit):
        seed_completed_meta(staged_toolkit)

        staged_toolkit.index_meta_update(
            "x", IndexerKeywords.INDEX_META_FAILED.value, 0, update_force=True, error="boom"
        )

        assert staged_toolkit.written[-1]["indexed"] == 191


class TestReduceDuplicatesStaging:
    def build_toolkit(self, monkeypatch, clean_index=False, indexed_data=None):
        toolkit = make_toolkit()
        adapter = FakeStagingAdapter()
        object.__setattr__(toolkit, "vector_adapter", adapter)
        object.__setattr__(
            toolkit, "_index_run", _IndexRunState(run_id="r1", clean_index=clean_index)
        )
        monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
        monkeypatch.setattr(StagingToolkit, "_log_tool_event", lambda self, *a, **kw: None)
        monkeypatch.setattr(StagingToolkit, "_get_indexed_data", lambda self, name: indexed_data or {})
        monkeypatch.setattr(StagingToolkit, "compare_fn", lambda self, doc, idx: True)
        monkeypatch.setattr(
            StagingToolkit, "remove_ids_fn", lambda self, idx_data, key: idx_data[key]["all_chunks"]
        )
        return toolkit

    def indexed_entry(self, chunks):
        return {"metadata": {"collection": "idx"}, "all_chunks": chunks}

    def test_changed_doc_is_staged_not_deleted(self, monkeypatch):
        toolkit = self.build_toolkit(
            monkeypatch,
            indexed_data={"d1": self.indexed_entry(["old-1"])},
        )
        monkeypatch.setattr(StagingToolkit, "compare_fn", lambda self, doc, idx: False)
        deleted = []
        object.__setattr__(toolkit, "vectorstore", type(
            "VS", (), {"delete": lambda self, ids: deleted.append(ids)}
        )())

        documents = [Document(page_content="b", metadata={"id": "d1"})]
        yielded = list(toolkit._reduce_duplicates(iter(documents), "idx"))

        assert len(yielded) == 1
        assert toolkit._index_run.staged_removal_ids == {"d1": {"old-1"}}
        assert deleted == []

    def test_clean_index_disables_the_unchanged_skip_but_still_nominates(self, monkeypatch):
        toolkit = self.build_toolkit(
            monkeypatch, clean_index=True,
            indexed_data={"d1": self.indexed_entry(["old-1"])},
        )

        documents = [Document(page_content="b", metadata={"id": "d1"})]
        yielded = list(toolkit._reduce_duplicates(iter(documents), "idx"))

        # compare_fn says "unchanged", yet the doc is re-yielded AND nominated.
        assert len(yielded) == 1
        assert toolkit._index_run.staged_removal_ids == {"d1": {"old-1"}}

    def test_seen_keys_are_recorded_for_the_orphan_math(self, monkeypatch):
        toolkit = self.build_toolkit(
            monkeypatch,
            indexed_data={"d1": self.indexed_entry(["old-1"])},
        )

        documents = [Document(page_content="b", metadata={"id": "d1"})]
        list(toolkit._reduce_duplicates(iter(documents), "idx"))

        assert toolkit._index_run.seen_keys == {"d1"}


class TestUnreadableIndexNeverPublishes:
    def test_a_failed_indexed_data_read_fails_the_run_instead_of_republishing(
        self, staged_toolkit, monkeypatch
    ):
        seed_completed_meta(staged_toolkit)
        monkeypatch.setattr(
            StagingToolkit, "_reduce_duplicates", BaseIndexerToolkit._reduce_duplicates
        )
        monkeypatch.setattr(
            StagingToolkit, "_get_indexed_data",
            lambda self, index_name: (_ for _ in ()).throw(RuntimeError("pgvector read failed")),
        )

        with pytest.raises(RuntimeError):
            run_index_data(staged_toolkit, monkeypatch)

        adapter = staged_toolkit.vector_adapter
        assert "promote" not in adapter.calls
        assert "discard" in adapter.calls
        assert staged_toolkit.written[-1]["state"] == IndexerKeywords.INDEX_META_FAILED.value

    def test_a_readable_empty_index_still_indexes_everything(self, staged_toolkit, monkeypatch):
        monkeypatch.setattr(
            StagingToolkit, "_reduce_duplicates", BaseIndexerToolkit._reduce_duplicates
        )
        monkeypatch.setattr(StagingToolkit, "_get_indexed_data", lambda self, index_name: {})

        outcome = run_index_data(staged_toolkit, monkeypatch)

        assert outcome["status"] == IndexingStatus.OK.value
        assert "promote" in staged_toolkit.vector_adapter.calls


class TestOrphanCandidates:
    def test_pk_fallback_rows_are_excluded(self):
        toolkit = make_toolkit()
        indexed_data = {
            "row-pk-1": {"metadata": {}, "all_chunks": ["row-pk-1"], "pk_fallback": True,
                         IndexerKeywords.PARENT.value: -1,
                         IndexerKeywords.DEPENDENT_DOCS.value: []},
            "gone-doc": {"metadata": {"id": "gone-doc"}, "all_chunks": ["c1", "c2"],
                         "pk_fallback": False,
                         IndexerKeywords.PARENT.value: -1,
                         IndexerKeywords.DEPENDENT_DOCS.value: []},
        }

        orphan_ids, orphan_doc_count = toolkit._collect_orphan_candidates(indexed_data, seen_keys=set())

        assert sorted(orphan_ids) == ["c1", "c2"]
        assert orphan_doc_count == 1

    def test_dependents_fall_with_their_parents_not_as_orphans(self):
        toolkit = make_toolkit()
        indexed_data = {
            "dep-1": {"metadata": {"id": "dep-1"}, "all_chunks": ["d1"], "pk_fallback": False,
                      IndexerKeywords.PARENT.value: None,
                      IndexerKeywords.DEPENDENT_DOCS.value: []},
        }

        # PARENT=None means "dependent of an id-less parent" — the adapter's
        # no-parent sentinel is -1, so None must NOT classify as top-level.
        assert toolkit._collect_orphan_candidates(indexed_data, seen_keys=set()) == ([], 0)


class TestPendingRunIdsNeverCached:
    class _RecordingAdapter:
        def __init__(self, results):
            self.results = list(results)
            self.calls = 0

        def get_pending_run_ids(self, wrapper, index_name, include_cancelled=True):
            self.calls += 1
            return self.results.pop(0)

    def make_wrapper(self, adapter):
        wrapper = VectorStoreWrapperBase.model_construct()
        object.__setattr__(wrapper, "vectorstore", object())
        object.__setattr__(wrapper, "vector_adapter", adapter)
        return wrapper

    def test_an_empty_fetch_is_not_cached_across_calls(self):
        # A run registering right after an empty observation must be excluded
        # by the very next search — a cached empty set would serve the run's
        # staged rows unfiltered until it expired.
        adapter = self._RecordingAdapter([[], ["r1"]])
        wrapper = self.make_wrapper(adapter)

        assert _REAL_GET_PENDING_RUN_IDS(wrapper, "idx") == []
        assert _REAL_GET_PENDING_RUN_IDS(wrapper, "idx") == ["r1"]
        assert adapter.calls == 2

    def test_a_non_empty_fetch_is_re_read_per_search(self):
        adapter = self._RecordingAdapter([["r1"], []])
        wrapper = self.make_wrapper(adapter)

        assert _REAL_GET_PENDING_RUN_IDS(wrapper, "idx") == ["r1"]
        assert _REAL_GET_PENDING_RUN_IDS(wrapper, "idx") == []
        assert adapter.calls == 2


class TestRetainedOrphanWarning:
    def toolkit_with(self, run):
        toolkit = make_toolkit()
        object.__setattr__(toolkit, "_index_run", run)
        return toolkit

    def test_gated_candidates_warn_in_the_report(self):
        run = _IndexRunState(run_id="abc123abc123")
        run.orphan_candidate_ids = ["c1", "c2", "c3"]
        run.orphan_candidate_doc_count = 2
        report = {}

        self.toolkit_with(run)._append_retained_orphan_warning(report)

        assert len(report["warnings"]) == 1
        assert "Kept 2 previously indexed documents" in report["warnings"][0]
        assert "stays searchable" in report["warnings"][0]

    def test_no_warning_without_candidates(self):
        run = _IndexRunState(run_id="abc123abc123")
        report = {}

        self.toolkit_with(run)._append_retained_orphan_warning(report)

        assert "warnings" not in report

    def test_no_warning_when_the_loader_attested(self, monkeypatch):
        monkeypatch.setattr(StagingToolkit, "loader_attests_completion", True)
        run = _IndexRunState(run_id="abc123abc123", loader_attested=True)
        run.orphan_candidate_ids = ["c1"]
        run.orphan_candidate_doc_count = 1
        report = {}

        self.toolkit_with(run)._append_retained_orphan_warning(report)

        assert "warnings" not in report

    def test_render_places_warning_lines_before_errors(self):
        report = {
            "totals": {"indexed": 1},
            "item_labels": {"singular": "document", "plural": "documents"},
            "categories": [],
            "warnings": ["Kept 2 previously indexed documents that were not returned."],
            "errors": ["boom"],
        }

        text = render_report_text(report)

        assert "⚠ Kept 2 previously indexed documents" in text
        assert text.index("⚠ Kept") < text.index("Errors:")


class TestDiscardFailureStillRecordsTheFailure:
    """A discard that cannot delete its chunks leaves the run row pending, so the failure
    reaches the platform with the index still looking busy. Both the meta write and the
    emit must carry the failure, including the cleanup error appended to it."""

    def _failing_save(self, base_documents, base_total, chunking_tool, chunking_config,
                      result, index_name=None):
        list(base_documents)
        result["count"] = 5
        result["failed_count"] = 5
        result.setdefault("errors", []).append("pgvector down")

    def test_the_failed_state_is_written_and_emitted(self, staged_toolkit, monkeypatch):
        seed_completed_meta(staged_toolkit)

        def failing_discard(wrapper, index_name, run_id):
            raise RuntimeError("chunk delete timed out")

        monkeypatch.setattr(staged_toolkit.vector_adapter, "discard_run", failing_discard)

        with pytest.raises(RuntimeError):
            run_index_data(staged_toolkit, monkeypatch, save=self._failing_save)

        assert staged_toolkit.written[-1]["state"] == IndexerKeywords.INDEX_META_FAILED.value
        assert staged_toolkit.emitted[-1]["state"] == IndexerKeywords.INDEX_META_FAILED.value
        assert "failed to discard staged rows" in staged_toolkit.emitted[-1]["error"]

    def test_the_retained_counts_survive_the_failed_write(self, staged_toolkit, monkeypatch):
        seed_completed_meta(staged_toolkit)

        def failing_discard(wrapper, index_name, run_id):
            raise RuntimeError("chunk delete timed out")

        monkeypatch.setattr(staged_toolkit.vector_adapter, "discard_run", failing_discard)

        with pytest.raises(RuntimeError):
            run_index_data(staged_toolkit, monkeypatch, save=self._failing_save)

        assert staged_toolkit.written[-1]["indexed"] == 191
        assert staged_toolkit.written[-1]["total"] == 205
