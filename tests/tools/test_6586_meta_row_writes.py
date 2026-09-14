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

"""Guards the embed-free, keyed index_meta write (issue #6586).

Re-broken by anything that puts an embedding call back on the meta-row write
path, widens the write back to the whole cmetadata column, lets a committed
cancel be reverted to in_progress, or reports a promoted corpus as failed.

Lives outside tests/tools/index/ because CI skips that directory.
"""

from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from langchain_core.documents import Document
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import declarative_base

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.base_indexer_toolkit import (
    BaseIndexerToolkit,
    SDK_OWNED_META_KEYS,
    _IndexRunState,
)
from elitea_sdk.tools.vector_adapters import VectorStoreAdapter as vsa_module
from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import PGVectorAdapter

_Base = declarative_base()


class _EmbeddingStore(_Base):
    __tablename__ = "langchain_pg_embedding"

    id = sa.Column(sa.String, primary_key=True)
    document = sa.Column(sa.String)
    embedding = sa.Column(sa.String)
    cmetadata = sa.Column(postgresql.JSONB)


def compile_sql(clause) -> str:
    return str(clause.compile(dialect=postgresql.dialect()))


class RecordingSession:
    """Session double: records the statements built against it, executes none."""

    def __init__(self, bind=None, rowcount=1):
        self.bind = bind
        self.statements = []
        self.commits = 0
        self._rowcount = rowcount

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, statement):
        self.statements.append(statement)
        return SimpleNamespace(rowcount=self._rowcount)

    def commit(self):
        self.commits += 1


@pytest.fixture
def sessions(monkeypatch):
    opened = []

    def session_factory(bind=None):
        session = RecordingSession(bind=bind)
        opened.append(session)
        return session

    # The adapter binds Session at module import, so the module attribute is the
    # one that has to be replaced.
    monkeypatch.setattr(vsa_module, "Session", session_factory)
    return opened


def make_wrapper():
    return SimpleNamespace(
        vectorstore=SimpleNamespace(
            EmbeddingStore=_EmbeddingStore,
            session_maker=SimpleNamespace(bind=None),
            collection_name="p_1_toolkit",
        ),
        _log_tool_event=lambda *a, **kw: None,
    )


def meta_statement(opened):
    """The statement that targets the meta row, whichever session carried it."""
    for session in opened:
        for statement in session.statements:
            if "langchain_pg_embedding" in compile_sql(statement):
                return statement
    raise AssertionError("no statement was issued against the meta row")


class TestTheWriteIsKeyedAndEmbedFree:

    def test_the_statement_is_a_keyed_update_of_cmetadata_only(self, sessions):
        PGVectorAdapter().update_index_meta_keys(
            make_wrapper(), "meta-1", None, {"state": "completed", "indexed": 500},
        )
        sql = compile_sql(meta_statement(sessions))

        assert sql.startswith("UPDATE langchain_pg_embedding SET cmetadata=")
        assert "langchain_pg_embedding.id = " in sql
        # The two columns whose rewrite required an embedding call.
        assert "embedding=" not in sql
        assert "document=" not in sql

    def test_the_merge_leaves_unpatched_keys_alone(self, sessions):
        PGVectorAdapter().update_index_meta_keys(
            make_wrapper(), "meta-1", None, {"state": "completed"},
        )
        sql = compile_sql(meta_statement(sessions))

        # `cmetadata || patch`, not `cmetadata = patch`: a whole-column write is
        # what reverted the platform's task_id and a committed cancel.
        assert "||" in sql
        assert "coalesce" in sql.lower()

    def test_a_cancelled_run_cannot_be_reverted(self, sessions):
        PGVectorAdapter().update_index_meta_keys(
            make_wrapper(), "meta-1", "run-1", {"state": "in_progress"},
        )
        sql = compile_sql(meta_statement(sessions))

        assert "NOT (EXISTS" in sql
        assert "elitea_index_runs" in sql
        assert "status" in sql

    def test_the_guard_admits_a_promoted_run(self, sessions):
        """The heartbeat's EXISTS(pending) guard would drop every terminal write:
        promote_run has already moved the run row off `pending` by then."""
        PGVectorAdapter().update_index_meta_keys(
            make_wrapper(), "meta-1", "run-1", {"state": "completed"},
        )
        sql = compile_sql(meta_statement(sessions))

        assert "promoted" not in sql
        assert "pending" not in sql

    def test_without_a_run_row_the_write_is_unguarded(self, sessions):
        PGVectorAdapter().update_index_meta_keys(
            make_wrapper(), "meta-1", None, {"state": "completed"},
        )
        sql = compile_sql(meta_statement(sessions))

        assert "elitea_index_runs" not in sql

    def test_the_caller_learns_when_no_row_matched(self, monkeypatch):
        opened = []

        def session_factory(bind=None):
            session = RecordingSession(bind=bind, rowcount=0)
            opened.append(session)
            return session

        monkeypatch.setattr(vsa_module, "Session", session_factory)
        matched = PGVectorAdapter().update_index_meta_keys(
            make_wrapper(), "meta-1", None, {"state": "completed"},
        )
        assert matched == 0


class TestTheHeartbeatCarriesProgress:

    def test_the_tick_patches_updated_on_and_run_chunks(self, sessions):
        PGVectorAdapter().heartbeat_index_run(
            make_wrapper(), "idx", "run-1", "meta-1", chunks_written=42,
        )
        sql = compile_sql(meta_statement(sessions))
        params = meta_statement(sessions).compile(dialect=postgresql.dialect()).params

        assert "||" in sql
        payload = "".join(str(v) for v in params.values())
        assert "run_chunks" in payload
        assert "42" in payload
        assert "updated_on" in payload

    def test_a_tick_without_a_count_still_refreshes_liveness(self, sessions):
        PGVectorAdapter().heartbeat_index_run(make_wrapper(), "idx", "run-1", "meta-1")
        params = meta_statement(sessions).compile(dialect=postgresql.dialect()).params
        payload = "".join(str(v) for v in params.values())

        assert "updated_on" in payload
        assert "run_chunks" not in payload


# --------------------------------------------------------------------------
# Toolkit-level behaviour
# --------------------------------------------------------------------------

class StagingToolkit(BaseIndexerToolkit):

    def key_fn(self, document: Document):
        return document.metadata.get("id")


class FakeStagingAdapter:
    supports_run_staging = True

    def __init__(self):
        self.promote_outcome = "promoted"
        self.calls = []
        self.pending = []
        self.patches = []
        self.attempts = []
        self.heartbeat_chunks = []
        self.meta_write_fails = False

    def ensure_index_runs_table(self, wrapper):
        self.calls.append("ensure")

    def register_index_run(self, wrapper, index_name, run_id, task_id=None, meta_lock_id=None):
        self.calls.append("register")
        return (True, None)

    def sweep_stale_index_runs(self, wrapper, index_name, stale_before):
        return []

    def heartbeat_index_run(self, wrapper, index_name, run_id, meta_id, chunks_written=None):
        self.calls.append("heartbeat")
        self.heartbeat_chunks.append(chunks_written)

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

    def update_index_meta_keys(self, wrapper, meta_id, run_id, patch):
        # Recorded BEFORE the failure: "never attempted" and "attempted, then
        # raised" are the two things this suite has to tell apart.
        self.attempts.append(patch)
        if self.meta_write_fails:
            raise RuntimeError("meta row write failed")
        self.patches.append(patch)
        stored = wrapper._stored_meta
        merged = {**stored.get("metadata", {}), **patch}
        wrapper.written.append(merged)
        object.__setattr__(wrapper, "_stored_meta", {**stored, "metadata": merged})
        return 1


@pytest.fixture
def toolkit(monkeypatch):
    instance = StagingToolkit.model_construct()
    object.__setattr__(instance, "vector_adapter", FakeStagingAdapter())
    object.__setattr__(instance, "_stored_meta", None)
    object.__setattr__(instance, "toolkit_id", None)
    object.__setattr__(instance, "max_docs_per_add", 100)
    object.__setattr__(instance, "written", [])
    object.__setattr__(instance, "embedded", [])
    object.__setattr__(instance, "emitted", [])

    def fake_add_documents(vectorstore=None, documents=None, ids=None):
        metadata = dict(documents[0].metadata)
        instance.embedded.append(metadata)
        object.__setattr__(
            instance, "_stored_meta",
            {"id": "meta-1", "content": "index_meta_x", "metadata": metadata},
        )
        return ["meta-row-id"]

    monkeypatch.setattr(
        "elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents", fake_add_documents
    )
    monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_index_meta", lambda self, name: self._stored_meta)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_indexed_count", lambda self, name: 965)
    monkeypatch.setattr(StagingToolkit, "_is_scheduled_run", lambda self: False)
    monkeypatch.setattr(StagingToolkit, "_log_tool_event", lambda self, *a, **kw: None)
    monkeypatch.setattr(
        StagingToolkit, "_emit_index_event",
        lambda self, name, error=None, state=None: instance.emitted.append(
            {"state": state, "error": error}
        ),
    )
    monkeypatch.setattr(StagingToolkit, "_reduce_duplicates", lambda self, docs, name: docs)
    return instance


def seed_meta(toolkit, state="completed"):
    metadata = {
        "collection": "x", "type": "index_meta", "state": state,
        "indexed": 191, "updated": 7, "total": 200, "created_on": 100.0,
        "task_id": "platform-task", "conversation_id": "conv-1",
        "task_disconnected_timeout_sec": 60,
        "index_configuration": {"a": 1},
        "history": '[{"state": "in_progress", "created_on": 100.0}]',
    }
    object.__setattr__(
        toolkit, "_stored_meta", {"id": "meta-1", "content": "index_meta_x", "metadata": metadata}
    )
    object.__setattr__(toolkit, "_index_run", _IndexRunState(run_id="run-1", meta_id="meta-1"))


class TestNoEmbeddingOnTheMetaWrite:

    def test_the_terminal_write_embeds_nothing(self, toolkit):
        seed_meta(toolkit)
        embeds_before = len(toolkit.embedded)

        toolkit.index_meta_update("x", IndexerKeywords.INDEX_META_COMPLETED.value, 12)

        # The whole point of the issue: completion no longer depends on the
        # embedding provider being reachable.
        assert len(toolkit.embedded) == embeds_before
        assert toolkit.vector_adapter.patches, "the write did not go through the keyed patch"

    def test_the_patch_carries_only_sdk_owned_keys(self, toolkit):
        seed_meta(toolkit)

        toolkit.index_meta_update("x", IndexerKeywords.INDEX_META_COMPLETED.value, 12)

        patch = toolkit.vector_adapter.patches[-1]
        assert set(patch) <= set(SDK_OWNED_META_KEYS)
        for platform_key in ("task_id", "conversation_id", "index_configuration",
                             "task_disconnected_timeout_sec", "created_on"):
            assert platform_key not in patch

    def test_history_report_and_skipped_stay_json_strings(self, toolkit):
        seed_meta(toolkit)

        toolkit.index_meta_update(
            "x", IndexerKeywords.INDEX_META_COMPLETED.value, 12,
            skipped={"items_processed": 1, "total_skipped": 0, "total_fetched": 1},
            report={"status": "ok", "totals": {"total": 1, "indexed": 1, "unchanged": 0}},
        )

        patch = toolkit.vector_adapter.patches[-1]
        for key in ("history", "report", "skipped"):
            assert isinstance(patch[key], str), (
                f"{key} must stay a JSON string: core reads it back with an "
                f"unguarded .strip() and kills the dispatched task when it raises"
            )


def make_documents(*ids):
    return [Document(page_content="body", metadata={"id": i, "name": i, "updated_on": "1"})
            for i in ids]


def run_index_data(toolkit, monkeypatch):
    def save(self, base_documents, base_total, chunking_tool, chunking_config, result, index_name=None):
        for _ in base_documents:
            result["count"] += 2
            result["docs_count"] += 1

    monkeypatch.setattr(StagingToolkit, "_base_loader", lambda self, **kw: iter(make_documents("a")))
    monkeypatch.setattr(StagingToolkit, "_save_index_generator", save)
    return toolkit.index_data(index_name="x")


def break_the_meta_write_after_promote(monkeypatch):
    """Let index_meta_init land, then fail every write from the terminal one on."""
    original = FakeStagingAdapter.update_index_meta_keys

    def failing(self, wrapper, meta_id, run_id, patch):
        if "promote" in self.calls:
            self.attempts.append(patch)
            raise RuntimeError("meta row write failed")
        return original(self, wrapper, meta_id, run_id, patch)

    monkeypatch.setattr(FakeStagingAdapter, "update_index_meta_keys", failing)


class TestAFailedRunIsAlwaysRecordedBeforeItIsAnnounced:
    """core's `update_toolkit_index_meta_failed_state` is a BACKSTOP, and its guard
    is justified by this invariant: on a genuine failure the SDK's own FAILED write
    lands before the event. Skipping the write for some class of run silently
    promotes that backstop to primary writer."""

    def test_a_failure_writes_failed_then_emits_it(self, toolkit, monkeypatch):
        def exploding_save(self, base_documents, base_total, chunking_tool,
                           chunking_config, result, index_name=None):
            raise RuntimeError("boom mid-run")

        monkeypatch.setattr(StagingToolkit, "_base_loader",
                            lambda self, **kw: iter(make_documents("a")))
        monkeypatch.setattr(StagingToolkit, "_save_index_generator", exploding_save)

        with pytest.raises(RuntimeError, match="boom mid-run"):
            toolkit.index_data(index_name="x")

        failed = IndexerKeywords.INDEX_META_FAILED.value
        assert toolkit.vector_adapter.attempts[-1]["state"] == failed
        assert toolkit.emitted[-1]["state"] == failed

    def test_a_promoted_run_is_no_exception_to_that(self, toolkit, monkeypatch):
        # A terminal write that fails AFTER promote still reports failure. Suppressing
        # it here changes nothing a user sees — the worker synthesizes the event
        # (indexer_worker/methods/indexer_test_toolkit.py) and core persists it — while
        # breaking the invariant above. Closing that tail needs core-side fences, not
        # SDK silence.
        break_the_meta_write_after_promote(monkeypatch)

        with pytest.raises(RuntimeError, match="meta row write failed"):
            run_index_data(toolkit, monkeypatch)

        assert "promote" in toolkit.vector_adapter.calls
        failed = IndexerKeywords.INDEX_META_FAILED.value
        assert toolkit.emitted[-1]["state"] == failed
