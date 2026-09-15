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

import logging
import time

import pytest
import sqlalchemy as sa
from langchain_core.documents import Document
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import declarative_base

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.base_indexer_toolkit import (
    BaseIndexerToolkit,
    INDEX_RUN_HEARTBEAT_INTERVAL,
    IndexingStatus,
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


def compiled_params(clause) -> dict:
    return dict(clause.compile(dialect=postgresql.dialect()).params)


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

    next_run_row = None

    def execute(self, statement):
        self.statements.append(statement)
        return SimpleNamespace(rowcount=self._rowcount)

    def get(self, model, primary_key, with_for_update=False):
        return RecordingSession.next_run_row

    def rollback(self):
        pass

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
        statement = meta_statement(sessions)
        sql = compile_sql(statement)

        assert "NOT (EXISTS" in sql
        assert "elitea_index_runs" in sql
        # The status is a BIND PARAMETER, so asserting on SQL text says nothing about
        # WHICH status is guarded — swapping it for `pending` leaves the text identical
        # and silently drops every terminal write.
        assert "cancelled" in compiled_params(statement).values()

    def test_the_guard_is_not_the_heartbeats_pending_rule(self, sessions):
        """The heartbeat's EXISTS(pending) guard would drop every terminal write:
        promote_run has already moved the run row off `pending` by then."""
        PGVectorAdapter().update_index_meta_keys(
            make_wrapper(), "meta-1", "run-1", {"state": "completed"},
        )
        values = compiled_params(meta_statement(sessions)).values()

        assert "pending" not in values
        assert "promoted" not in values

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


class TestTheHeartbeatTwinContract:
    """Core mirrors this interval and multiplies it by five for the display horizon
    (INDEX_RUN_HEARTBEAT_INTERVAL_SEC / HEARTBEAT_STALE_INTERVALS in
    utils/application_tools.py). Core pins its own copy; without this the SDK side
    could move to 120 and a healthy run would read stale after 2.5 ticks, with both
    suites green."""

    def test_the_interval_matches_the_value_core_mirrors(self):
        assert INDEX_RUN_HEARTBEAT_INTERVAL == 60.0


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
        # Load-bearing absence, not an oversight. The EXISTS(pending) guard is an
        # uncorrelated subquery, so a tick already queued behind promote's lock still
        # applies after promote commits. That is only safe while this patch carries
        # liveness keys and nothing else — adding `state` here resurrects in_progress
        # over a committed cancel, which is Fault 4 in its original shape.
        assert "state" not in payload

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


@pytest.fixture
def orm_sessions(monkeypatch):
    """promote_run imports Session inside the method, so the module attribute the
    `sessions` fixture replaces is not the one it binds."""
    opened = []

    def session_factory(bind=None):
        session = RecordingSession(bind=bind)
        opened.append(session)
        return session

    monkeypatch.setattr("sqlalchemy.orm.Session", session_factory)
    yield opened
    RecordingSession.next_run_row = None


def sessions_with_row(run_row):
    """Make the next Session(...) hand back this run row from session.get()."""
    RecordingSession.next_run_row = run_row


def _raise_lock_failure(session, store, index_name):
    raise RuntimeError("meta lock unavailable")


def _promote_adapter():
    adapter = PGVectorAdapter()
    adapter._lock_index_meta_rows = lambda session, store, index_name: ["meta-1"]
    adapter._delete_rows_by_pk = lambda session, store, ids: None
    adapter._delete_run_chunks = lambda session, store, run_id: 0
    adapter._discard_stranded_run_chunks = lambda store, run_id: None
    return adapter


def _bare_toolkit(monkeypatch, max_docs_per_add):
    """A toolkit wired just enough to drive _save_index_generator directly."""
    instance = StagingToolkit.model_construct()
    object.__setattr__(instance, "vector_adapter", FakeStagingAdapter())
    object.__setattr__(instance, "max_docs_per_add", max_docs_per_add)
    object.__setattr__(instance, "_index_run", _IndexRunState(run_id="run-1"))
    monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
    monkeypatch.setattr(StagingToolkit, "_log_tool_event", lambda self, *a, **kw: None)
    monkeypatch.setattr(StagingToolkit, "_staging_active", lambda self: False)
    monkeypatch.setattr(StagingToolkit, "_extend_data", lambda self, docs: docs)
    monkeypatch.setattr(StagingToolkit, "_collect_dependencies", lambda self, docs: docs)
    monkeypatch.setattr(
        StagingToolkit, "_apply_loaders_chunkers",
        lambda self, docs, chunking_tool=None, chunking_config=None: docs,
    )
    monkeypatch.setattr(StagingToolkit, "_clean_metadata", lambda self, docs: docs)
    return instance


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


class TestARunCompletesWithTheEmbeddingBackendDown:
    """#6586's headline acceptance criterion, end to end.

    The real pipeline runs here rather than a counter stub, so chunk batches
    actually reach the patched add_documents. That matters twice: the shim's
    "let chunks through, fail index_meta" split is exercised instead of being
    unreachable, and the run reaches COMPLETED the way a real one does.

    The shim fails index_meta embeds after the first — the fresh-row INSERT has to
    land or there is no row to patch. It must never touch chunk flushes: a run that
    loses those ends FAILED on its own and the assertions below would pass without
    exercising anything.
    """

    def _run_with_embeddings_down(self, toolkit, monkeypatch, documents=("a", "b")):
        seen = SimpleNamespace(meta=[], chunks=[])

        def failing_add_documents(vectorstore=None, documents=None, ids=None):
            metadata = dict(documents[0].metadata)
            if metadata.get("type") != IndexerKeywords.INDEX_META_TYPE.value:
                seen.chunks.append(len(documents))
                return [f"chunk-{n}" for n in range(len(documents))]
            seen.meta.append(metadata)
            if len(seen.meta) > 1:
                raise RuntimeError("EL6586_EMBED_DOWN: embedding backend unavailable")
            object.__setattr__(
                toolkit, "_stored_meta",
                {"id": "meta-1", "content": "index_meta_x", "metadata": metadata},
            )
            return ["meta-row-id"]

        monkeypatch.setattr(
            "elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents",
            failing_add_documents,
        )
        # The real _save_index_generator, with only the fetch/chunk stages stubbed.
        monkeypatch.setattr(StagingToolkit, "_extend_data", lambda self, docs: docs)
        monkeypatch.setattr(StagingToolkit, "_collect_dependencies", lambda self, docs: docs)
        monkeypatch.setattr(
            StagingToolkit, "_apply_loaders_chunkers",
            lambda self, docs, chunking_tool=None, chunking_config=None: docs,
        )
        monkeypatch.setattr(StagingToolkit, "_clean_metadata", lambda self, docs: docs)
        monkeypatch.setattr(StagingToolkit, "_base_loader",
                            lambda self, **kw: iter(make_documents(*documents)))

        result = toolkit.index_data(index_name="x")
        return result, seen

    def test_the_run_reports_completed_with_its_real_counts(self, toolkit, monkeypatch):
        result, seen = self._run_with_embeddings_down(toolkit, monkeypatch)

        assert result["status"] == IndexingStatus.OK.value
        # The shim was armed and the chunk half really ran: had it fired on chunks the
        # run would have ended FAILED and this assertion would pass for the wrong reason.
        assert sum(seen.chunks) == 2
        # Only index_meta_init embedded. The terminal write never asked, which is why a
        # dead embedding backend can no longer report a promoted corpus as failed.
        assert len(seen.meta) == 1
        assert "promote" in toolkit.vector_adapter.calls

    def test_a_second_meta_embed_would_have_raised(self, toolkit, monkeypatch):
        """Positive control: the shim is live, not inert."""
        _, seen = self._run_with_embeddings_down(toolkit, monkeypatch)
        from elitea_sdk.runtime.langchain.interfaces import llm_processor

        assert seen.meta, "the shim never saw the init embed"
        with pytest.raises(RuntimeError, match="EL6586_EMBED_DOWN"):
            llm_processor.add_documents(
                documents=[Document(page_content="index_meta_x",
                                    metadata={"type": IndexerKeywords.INDEX_META_TYPE.value})]
            )

    def test_the_terminal_state_and_counts_reach_the_row(self, toolkit, monkeypatch):
        self._run_with_embeddings_down(toolkit, monkeypatch)

        patch = toolkit.vector_adapter.patches[-1]
        failed = IndexerKeywords.INDEX_META_FAILED.value
        assert patch["state"] == IndexerKeywords.INDEX_META_COMPLETED.value
        assert patch["updated"] == 2
        # Folded in from a test that could never observe these being false: on the
        # green path both are empty by construction, and under the terminal-write
        # mutant the run raises before either assertion is reached.
        assert [p for p in toolkit.vector_adapter.attempts if p.get("state") == failed] == []
        assert [e for e in toolkit.emitted if e["state"] == failed] == []
        # Pins the CALLER's ternary, not the writer's default: index_data passes the
        # rendered run summary as `error` on every non-OK status, so dropping the
        # condition parks a "2 documents indexed" message in the field core and the UI
        # read as the failure reason. The seeded test below cannot see this — it calls
        # index_meta_update directly, where `error` simply defaults to None.
        assert patch["error"] is None
        # Same ternary, second channel. The worker forwards this event and core
        # persists it, so announcing the success summary as `error` reports a healthy
        # index as failed just as effectively as writing it to the row.
        assert toolkit.emitted[-1] == {"state": IndexerKeywords.INDEX_META_COMPLETED.value,
                                       "error": None}

    def test_a_stale_error_from_the_previous_run_is_cleared(self, toolkit, monkeypatch):
        """Seeded, because a fresh row already carries error: None from init — the
        assertion passes without the success branch clearing anything."""
        self._run_with_embeddings_down(toolkit, monkeypatch)
        object.__setattr__(
            toolkit, "_stored_meta",
            {**toolkit._stored_meta,
             "metadata": {**toolkit._stored_meta["metadata"], "error": "previous failure"}},
        )

        toolkit.index_meta_update("x", IndexerKeywords.INDEX_META_COMPLETED.value, 5)

        assert toolkit.vector_adapter.patches[-1]["error"] is None


class TestAFailedRunCarriesItsReasonOnBothChannels:
    """The sibling of the completed-run class above. Both branches of
    `error=message if status is not IndexingStatus.OK else None` are live code on
    every run; testing only the success half leaves an expression whose other half
    nothing contradicts, on the row write AND on the event.

    The shim fails every CHUNK flush and no index_meta embed, which is the mirror of
    the class above. `_flush_chunk` swallows the exception into `failed_count`, so the
    run reaches its terminal write with a non-OK status instead of raising.
    """

    def _run_with_chunk_writes_down(self, toolkit, monkeypatch, documents=("a", "b")):
        def failing_add_documents(vectorstore=None, documents=None, ids=None):
            metadata = dict(documents[0].metadata)
            if metadata.get("type") == IndexerKeywords.INDEX_META_TYPE.value:
                object.__setattr__(
                    toolkit, "_stored_meta",
                    {"id": "meta-1", "content": "index_meta_x", "metadata": metadata},
                )
                return ["meta-row-id"]
            raise RuntimeError("EL6586_CHUNKS_DOWN: vectorstore write rejected")

        monkeypatch.setattr(
            "elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents",
            failing_add_documents,
        )
        monkeypatch.setattr(StagingToolkit, "_extend_data", lambda self, docs: docs)
        monkeypatch.setattr(StagingToolkit, "_collect_dependencies", lambda self, docs: docs)
        monkeypatch.setattr(
            StagingToolkit, "_apply_loaders_chunkers",
            lambda self, docs, chunking_tool=None, chunking_config=None: docs,
        )
        monkeypatch.setattr(StagingToolkit, "_clean_metadata", lambda self, docs: docs)
        monkeypatch.setattr(StagingToolkit, "_base_loader",
                            lambda self, **kw: iter(make_documents(*documents)))

        return toolkit.index_data(index_name="x")

    def test_the_run_really_failed(self, toolkit, monkeypatch):
        """Positive control: without this the two assertions below would hold on a
        run that quietly succeeded and reported no reason because there was none."""
        result = self._run_with_chunk_writes_down(toolkit, monkeypatch)

        assert result["status"] != IndexingStatus.OK.value
        assert result["message"]

    def test_the_row_write_carries_the_reason(self, toolkit, monkeypatch):
        result = self._run_with_chunk_writes_down(toolkit, monkeypatch)

        assert toolkit.vector_adapter.patches[-1]["error"] == result["message"]

    def test_the_event_carries_the_reason(self, toolkit, monkeypatch):
        result = self._run_with_chunk_writes_down(toolkit, monkeypatch)

        assert toolkit.emitted[-1]["error"] == result["message"]


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


class TestProgressIsSeededAndCounted:

    def test_the_heartbeat_ticks_before_the_first_interval(self, toolkit):
        # run_chunks does not survive the platform's run-start reset, so until the
        # first tick lands the list view has only the previous run's counts.
        seed_meta(toolkit)
        toolkit._start_run_heartbeat("x")
        toolkit._stop_run_heartbeat()

        assert toolkit.vector_adapter.calls.count("heartbeat") >= 1
        assert toolkit.vector_adapter.heartbeat_chunks[0] == 0

    def test_the_tick_reports_the_runs_current_chunk_count(self, toolkit):
        seed_meta(toolkit)
        toolkit._index_run.chunks_written = 17

        toolkit._start_run_heartbeat("x")
        toolkit._stop_run_heartbeat()

        assert toolkit.vector_adapter.heartbeat_chunks[-1] == 17

    def test_only_chunks_that_landed_are_counted(self, monkeypatch):
        flushed = []
        instance = _bare_toolkit(monkeypatch, max_docs_per_add=2)

        def fake_add_documents(vectorstore=None, documents=None, ids=None):
            flushed.append(len(documents))
            return [f"id-{n}" for n in range(len(documents))]

        monkeypatch.setattr(
            "elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents",
            fake_add_documents,
        )
        documents = make_documents("a", "b", "c")
        result = {"count": 0, "failed_count": 0, "docs_count": 0, "failed_docs": 0}
        instance._save_index_generator(iter(documents), len(documents), None, None, result, "x")

        assert sum(flushed) == 3
        assert instance._index_run.chunks_written == 3

    def test_a_failed_flush_is_not_counted_as_written(self, monkeypatch):
        instance = _bare_toolkit(monkeypatch, max_docs_per_add=1)

        def exploding_add_documents(vectorstore=None, documents=None, ids=None):
            raise RuntimeError("flush failed")

        monkeypatch.setattr(
            "elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents",
            exploding_add_documents,
        )
        result = {"count": 0, "failed_count": 0, "docs_count": 0, "failed_docs": 0}
        instance._save_index_generator(iter(make_documents("a")), 1, None, None, result, "x")

        assert instance._index_run.chunks_written == 0


class TestPromoteRefreshesTheHeartbeatItBlocks:
    """promote_run holds the run row FOR UPDATE across every batched delete, so the
    worker's own tick blocks on it and then matches zero rows — its predicate is
    status IN (pending, cancelled) and the status has moved by the time the lock
    lifts. Without a stamp, readers see a heartbeat frozen at promote start for the
    whole promote plus one interval."""

    def test_the_stamp_is_committed_before_promote_opens_its_transaction(self, orm_sessions):
        """The mechanism is transactional, so the assertion has to be too.

        Asserting `run_row.heartbeat > started` on a double would pass for an
        in-transaction assignment, which no reader can ever see: it is invisible
        until commit, and that same commit moves the status off `pending`, which
        every reader filters on. What has to hold is that the stamp lands in its
        OWN committed transaction, opened and closed before promote's."""
        run_row = SimpleNamespace(run_id="run-1", status="pending",
                                  heartbeat=time.time() - 600, promoted_on=None)
        sessions_with_row(run_row)

        _promote_adapter().promote_run(make_wrapper(), "idx", "run-1", [], [], [])

        assert len(orm_sessions) >= 2, "the stamp must not share promote's session"
        stamp_session, promote_session = orm_sessions[0], orm_sessions[1]

        sql = compile_sql(stamp_session.statements[0])
        assert sql.startswith("UPDATE elitea_index_runs SET heartbeat=")
        # Committed on its own, before the session that takes the meta lock.
        assert stamp_session.commits == 1
        assert stamp_session.statements, "the first session must carry the stamp"
        assert promote_session is not stamp_session

    def test_the_stamp_only_touches_a_run_that_is_still_live(self, orm_sessions):
        run_row = SimpleNamespace(run_id="run-1", status="pending",
                                  heartbeat=time.time(), promoted_on=None)
        sessions_with_row(run_row)

        _promote_adapter().promote_run(make_wrapper(), "idx", "run-1", [], [], [])

        # The IN clause binds as one list parameter, so flatten before asserting.
        bound = []
        for value in compiled_params(orm_sessions[0].statements[0]).values():
            bound.extend(value if isinstance(value, (list, tuple)) else [value])

        assert "pending" in bound
        assert "cancelled" in bound
        assert "promoted" not in bound

    def test_a_failing_stamp_never_fails_the_run(self, orm_sessions, monkeypatch):
        """The stamp is decorative; the run is not.

        It executes before promote sets run.finalized, so a raise here reaches the
        generic handler with the latch still open, _discard_index_run throws away the
        rows this run already flushed and paid to embed, and the run reports FAILED —
        over a write that only affects how fresh the card looks. The tick that issues
        this same statement two call sites away is already non-fatal."""
        calls = []
        real_execute = RecordingSession.execute

        def exploding_first_execute(self, statement):
            calls.append(statement)
            if len(calls) == 1:
                raise RuntimeError("EL6586: prepared statement already exists")
            return real_execute(self, statement)

        monkeypatch.setattr(RecordingSession, "execute", exploding_first_execute)
        sessions_with_row(SimpleNamespace(run_id="run-1", status="pending",
                                          heartbeat=time.time(), promoted_on=None))

        outcome = _promote_adapter().promote_run(make_wrapper(), "idx", "run-1", [], [], [])

        assert outcome == "promoted", "a decorative write must not abort the promote"

    @pytest.mark.parametrize("error,swallowed", [
        (RuntimeError("EL6586: prepared statement already exists"), True),
        # Not an Exception: an interrupt must never be absorbed by a write whose only
        # job is to make a card look fresh.
        (KeyboardInterrupt(), False),
    ])
    def test_only_ordinary_failures_of_the_stamp_are_swallowed(
        self, orm_sessions, monkeypatch, caplog, error, swallowed
    ):
        calls = []
        real_execute = RecordingSession.execute

        def exploding_first_execute(self, statement):
            calls.append(statement)
            if len(calls) == 1:
                raise error
            return real_execute(self, statement)

        monkeypatch.setattr(RecordingSession, "execute", exploding_first_execute)
        sessions_with_row(SimpleNamespace(run_id="run-1", status="pending",
                                          heartbeat=time.time(), promoted_on=None))
        adapter = _promote_adapter()

        if swallowed:
            with caplog.at_level(logging.WARNING):
                assert adapter.promote_run(make_wrapper(), "idx", "run-1", [], [], []) == "promoted"
            # Swallowed, not silent — the operator still has to be able to see it.
            assert "heartbeat" in caplog.text
        else:
            with pytest.raises(KeyboardInterrupt):
                adapter.promote_run(make_wrapper(), "idx", "run-1", [], [], [])

    def test_a_real_promote_failure_still_propagates(self, orm_sessions, monkeypatch):
        """The sibling half: only the DECORATIVE write is swallowed.

        Widening the guard around promote's own transaction would turn a failed
        publish into a reported success, which is the opposite of this issue."""
        adapter = _promote_adapter()
        adapter._lock_index_meta_rows = _raise_lock_failure
        sessions_with_row(SimpleNamespace(run_id="run-1", status="pending",
                                          heartbeat=time.time(), promoted_on=None))

        with pytest.raises(RuntimeError, match="meta lock unavailable"):
            adapter.promote_run(make_wrapper(), "idx", "run-1", [], [], [])

    def test_a_run_row_that_is_gone_does_not_break_promote(self, orm_sessions):
        # The stamp must not turn a missing run row into an AttributeError; promote
        # still has to reach its own abort outcome.
        sessions_with_row(None)
        adapter = _promote_adapter()

        assert adapter.promote_run(make_wrapper(), "idx", "run-1", [], [], []) == "aborted-not-pending"


class TestTheHeartbeatOutlivesTheDocumentLoop:
    """promote_run holds the meta row for seconds to minutes on a large corpus, and
    it runs AFTER the document loop. A heartbeat that stops at the end of the loop
    leaves a healthy run with a frozen heartbeat and a still-pending run row, which
    every liveness reader is then entitled to call dead."""

    def test_the_heartbeat_is_still_running_when_promote_is_called(self, toolkit, monkeypatch):
        ticks_at_promote = {}

        def save(self, base_documents, base_total, chunking_tool, chunking_config,
                 result, index_name=None):
            for _ in base_documents:
                result["count"] += 2
                result["docs_count"] += 1

        original_promote = FakeStagingAdapter.promote_run

        def recording_promote(self, wrapper, index_name, run_id, superseded, orphan, damaged):
            run = wrapper._index_run
            ticks_at_promote["stopped"] = (
                run.heartbeat_stop is not None and run.heartbeat_stop.is_set()
            )
            return original_promote(self, wrapper, index_name, run_id, superseded, orphan, damaged)

        monkeypatch.setattr(FakeStagingAdapter, "promote_run", recording_promote)
        monkeypatch.setattr(StagingToolkit, "_base_loader",
                            lambda self, **kw: iter(make_documents("a")))
        monkeypatch.setattr(StagingToolkit, "_save_index_generator", save)

        toolkit.index_data(index_name="x")

        assert ticks_at_promote["stopped"] is False, (
            "the heartbeat must still be live while promote holds the meta row"
        )

    def test_the_run_still_stops_its_heartbeat_when_it_finishes(self, toolkit, monkeypatch):
        def save(self, base_documents, base_total, chunking_tool, chunking_config,
                 result, index_name=None):
            for _ in base_documents:
                result["count"] += 2
                result["docs_count"] += 1

        monkeypatch.setattr(StagingToolkit, "_base_loader",
                            lambda self, **kw: iter(make_documents("a")))
        monkeypatch.setattr(StagingToolkit, "_save_index_generator", save)

        toolkit.index_data(index_name="x")

        assert toolkit._index_run.heartbeat_stop.is_set()
