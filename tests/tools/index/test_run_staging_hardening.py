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

"""Unit tests for the staging mechanism's hardening pass (issue #6232).

Database-free: every SQL-shape assertion compiles the built statement against
the PostgreSQL dialect, and every transaction is driven through a recording
session double.

Covers:
  - promote/discard abort branches deleting the run's own staged chunks, and
    never touching a promoted run's rows
  - index removal clearing the collection's terminal run rows while leaving a
    live run's row (and therefore its exclusion marker) intact
  - one fail-open polarity across all three pending-run read paths
  - the sargable (containment) shape of the index_meta predicate and the
    dropped, unread document column in get_indexed_data
  - the reclaim sweep re-verifying staleness under the run-row lock, and its
    reclamation pass for chunks stranded under an already-discarded run
"""

import logging
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import declarative_base

from elitea_sdk.runtime.tools.index_runs_model import (
    RUN_STATUS_CANCELLED,
    RUN_STATUS_DISCARDED,
    RUN_STATUS_PENDING,
    RUN_STATUS_PROMOTED,
    is_degradable_run_lookup_error,
)
from elitea_sdk.runtime.tools.vectorstore import VectorStoreWrapper
from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.tools.elitea_base import BaseVectorStoreToolApiWrapper
from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import (
    STRANDED_RECLAIM_RUN_LIMIT,
    PGVectorAdapter,
)

_Base = declarative_base()


class _EmbeddingStore(_Base):
    __tablename__ = "langchain_pg_embedding"

    id = sa.Column(sa.String, primary_key=True)
    document = sa.Column(sa.String)
    cmetadata = sa.Column(postgresql.JSONB)


def compile_sql(clause) -> str:
    return str(clause.compile(dialect=postgresql.dialect()))


def compiled_params(clause) -> dict:
    return dict(clause.compile(dialect=postgresql.dialect()).params)


class RecordingQuery:
    def __init__(self, recorder, entities):
        self.recorder = recorder
        self.entities = entities
        self.filters = []

    def filter(self, *criteria):
        self.filters.extend(criteria)
        self.recorder.filters.extend(criteria)
        return self

    def with_for_update(self):
        return self

    def all(self):
        return []

    def delete(self, synchronize_session=False):
        self.recorder.deletes.append(self)
        return 0


class RecordingSession:
    """Session double: records the statements built against it, executes none."""

    def __init__(self, bind=None, run_row=None):
        self.bind = bind
        self.run_row = run_row
        self.queries = []
        self.filters = []
        self.deletes = []
        self.commits = 0
        self.rollbacks = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def query(self, *entities):
        query = RecordingQuery(self, entities)
        self.queries.append(query)
        return query

    def get(self, model, primary_key, with_for_update=False):
        return self.run_row

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


@pytest.fixture
def sessions(monkeypatch):
    """Every ``Session(...)`` opened by the adapter, in order."""
    opened = []
    run_rows = []

    def session_factory(bind=None):
        session = RecordingSession(bind=bind, run_row=run_rows[0] if run_rows else None)
        opened.append(session)
        return session

    monkeypatch.setattr("sqlalchemy.orm.Session", session_factory)
    return SimpleNamespace(opened=opened, run_rows=run_rows)


def make_store():
    return SimpleNamespace(
        EmbeddingStore=_EmbeddingStore,
        session_maker=SimpleNamespace(bind=None),
        collection_name="p_1_toolkit",
    )


def make_wrapper():
    return SimpleNamespace(
        vectorstore=make_store(),
        _log_tool_event=lambda *args, **kwargs: None,
    )


def make_run_row(status):
    return SimpleNamespace(run_id="run-1", status=status, promoted_on=None)


class TestPromoteAbortsDeleteOwnChunks:
    """A terminal run row puts its staged rows out of reach of both the sweep
    (pending/cancelled only) and the read filter, so every abort has to take
    them with it."""

    def setup_adapter(self, monkeypatch, meta_ids):
        adapter = PGVectorAdapter()
        chunk_deletes = []
        stranded_deletes = []
        monkeypatch.setattr(
            PGVectorAdapter, "_lock_index_meta_rows",
            lambda self, session, store, index_name: list(meta_ids),
        )
        monkeypatch.setattr(
            PGVectorAdapter, "_delete_run_chunks",
            lambda self, session, store, run_id: chunk_deletes.append(run_id) or 0,
        )
        monkeypatch.setattr(
            PGVectorAdapter, "_discard_stranded_run_chunks",
            lambda self, store, run_id: stranded_deletes.append(run_id),
        )
        return adapter, chunk_deletes, stranded_deletes

    def test_row_deleted_abort_takes_its_staged_chunks_along(self, monkeypatch, sessions):
        adapter, chunk_deletes, _ = self.setup_adapter(monkeypatch, meta_ids=[])
        run_row = make_run_row(RUN_STATUS_PENDING)
        sessions.run_rows.append(run_row)

        outcome = adapter.promote_run(make_wrapper(), "idx", "run-1", [], [], [])

        assert outcome == "aborted-row-deleted"
        assert chunk_deletes == ["run-1"]
        assert run_row.status == RUN_STATUS_DISCARDED
        assert sessions.opened[0].commits == 1

    def test_row_deleted_abort_cleans_up_even_without_a_run_row(self, monkeypatch, sessions):
        adapter, chunk_deletes, _ = self.setup_adapter(monkeypatch, meta_ids=[])

        outcome = adapter.promote_run(make_wrapper(), "idx", "run-1", [], [], [])

        assert outcome == "aborted-row-deleted"
        assert chunk_deletes == ["run-1"]

    def test_row_deleted_abort_reclaims_the_chunks_of_an_already_swept_run(self, monkeypatch, sessions):
        adapter, chunk_deletes, _ = self.setup_adapter(monkeypatch, meta_ids=[])
        run_row = make_run_row(RUN_STATUS_DISCARDED)
        sessions.run_rows.append(run_row)

        outcome = adapter.promote_run(make_wrapper(), "idx", "run-1", [], [], [])

        assert outcome == "aborted-row-deleted"
        assert chunk_deletes == ["run-1"]
        assert run_row.status == RUN_STATUS_DISCARDED

    def test_row_deleted_abort_reclaims_a_cancelled_runs_chunks(self, monkeypatch, sessions):
        adapter, chunk_deletes, _ = self.setup_adapter(monkeypatch, meta_ids=[])
        run_row = make_run_row(RUN_STATUS_CANCELLED)
        sessions.run_rows.append(run_row)

        outcome = adapter.promote_run(make_wrapper(), "idx", "run-1", [], [], [])

        assert outcome == "aborted-row-deleted"
        assert chunk_deletes == ["run-1"]
        assert run_row.status == RUN_STATUS_DISCARDED

    def test_row_deleted_abort_never_touches_a_promoted_runs_rows(self, monkeypatch, sessions):
        adapter, chunk_deletes, stranded = self.setup_adapter(monkeypatch, meta_ids=[])
        run_row = make_run_row(RUN_STATUS_PROMOTED)
        sessions.run_rows.append(run_row)

        outcome = adapter.promote_run(make_wrapper(), "idx", "run-1", [], [], [])

        assert outcome == "aborted-row-deleted"
        assert chunk_deletes == []
        assert stranded == []
        assert run_row.status == RUN_STATUS_PROMOTED

    def test_not_pending_abort_cleans_up_after_the_rollback(self, monkeypatch, sessions):
        adapter, chunk_deletes, stranded = self.setup_adapter(monkeypatch, meta_ids=["meta-1"])
        sessions.run_rows.append(make_run_row(RUN_STATUS_DISCARDED))

        outcome = adapter.promote_run(make_wrapper(), "idx", "run-1", [], [], [])

        assert outcome == "aborted-not-pending"
        assert sessions.opened[0].rollbacks == 1
        assert chunk_deletes == []
        assert stranded == ["run-1"]

    def test_not_pending_abort_spares_a_promoted_run(self, monkeypatch, sessions):
        adapter, _, stranded = self.setup_adapter(monkeypatch, meta_ids=["meta-1"])
        sessions.run_rows.append(make_run_row(RUN_STATUS_PROMOTED))

        outcome = adapter.promote_run(make_wrapper(), "idx", "run-1", [], [], [])

        assert outcome == "aborted-not-pending"
        assert stranded == []

    def test_cancelled_abort_still_deletes_in_the_promote_transaction(self, monkeypatch, sessions):
        adapter, chunk_deletes, stranded = self.setup_adapter(monkeypatch, meta_ids=["meta-1"])
        run_row = make_run_row(RUN_STATUS_CANCELLED)
        sessions.run_rows.append(run_row)

        outcome = adapter.promote_run(make_wrapper(), "idx", "run-1", [], [], [])

        assert outcome == "aborted-cancelled"
        assert chunk_deletes == ["run-1"]
        assert stranded == []
        assert run_row.status == RUN_STATUS_DISCARDED

    def test_discard_noop_cleans_up_after_the_rollback(self, monkeypatch, sessions):
        adapter, chunk_deletes, stranded = self.setup_adapter(monkeypatch, meta_ids=["meta-1"])
        sessions.run_rows.append(make_run_row(RUN_STATUS_DISCARDED))

        outcome = adapter.discard_run(make_wrapper(), "idx", "run-1")

        assert outcome == "noop"
        assert sessions.opened[0].rollbacks == 1
        assert chunk_deletes == []
        assert stranded == ["run-1"]

    def test_discard_noop_spares_a_promoted_run(self, monkeypatch, sessions):
        adapter, _, stranded = self.setup_adapter(monkeypatch, meta_ids=["meta-1"])
        sessions.run_rows.append(make_run_row(RUN_STATUS_PROMOTED))

        assert adapter.discard_run(make_wrapper(), "idx", "run-1") == "noop"
        assert stranded == []

    def test_stranded_cleanup_runs_in_its_own_committed_transaction(self, monkeypatch, sessions):
        adapter = PGVectorAdapter()
        deleted = []
        monkeypatch.setattr(
            PGVectorAdapter, "_delete_run_chunks",
            lambda self, session, store, run_id: deleted.append((session, run_id)) or 0,
        )

        adapter._discard_stranded_run_chunks(make_store(), "run-1")

        assert [run_id for _, run_id in deleted] == ["run-1"]
        assert len(sessions.opened) == 1
        assert sessions.opened[0].commits == 1

    def test_a_failed_stranded_cleanup_stays_silent(self, monkeypatch, caplog):
        adapter = PGVectorAdapter()

        def failing_session(bind=None):
            raise statement_timeout_error()

        monkeypatch.setattr("sqlalchemy.orm.Session", failing_session)

        with caplog.at_level(logging.WARNING):
            adapter._discard_stranded_run_chunks(make_store(), "run-1")

        assert "run-1" in caplog.text

    def test_abort_outcome_survives_a_failed_stranded_cleanup(self, monkeypatch, sessions):
        adapter, _, _ = self.setup_adapter(monkeypatch, meta_ids=["meta-1"])
        sessions.run_rows.append(make_run_row(RUN_STATUS_DISCARDED))
        monkeypatch.setattr(
            PGVectorAdapter, "_discard_stranded_run_chunks",
            PGVectorAdapter.__dict__["_discard_stranded_run_chunks"],
        )
        monkeypatch.setattr(
            PGVectorAdapter, "_delete_run_chunks",
            lambda self, session, store, run_id: (_ for _ in ()).throw(statement_timeout_error()),
        )

        assert adapter.promote_run(make_wrapper(), "idx", "run-1", [], [], []) == "aborted-not-pending"


class TestIndexRemovalClearsRunRows:
    """Index deletion garbage-collects the collection's finished run rows, and
    must not touch a live one: that row both holds the name against a second
    registration and keeps the run's staged chunks out of every read."""

    def test_full_removal_clears_the_collections_run_rows(self, monkeypatch, sessions):
        adapter = PGVectorAdapter()
        cleared = []
        monkeypatch.setattr(
            PGVectorAdapter, "_delete_index_runs",
            lambda self, wrapper, index_name: cleared.append(index_name),
        )

        adapter.clean_collection(make_wrapper(), index_name="idx", including_index_meta=True)

        assert cleared == ["idx"]

    def test_pre_load_clean_keeps_the_live_runs_row(self, monkeypatch, sessions):
        adapter = PGVectorAdapter()
        cleared = []
        monkeypatch.setattr(
            PGVectorAdapter, "_delete_index_runs",
            lambda self, wrapper, index_name: cleared.append(index_name),
        )

        adapter.clean_collection(make_wrapper(), index_name="idx", including_index_meta=False)

        assert cleared == []

    def test_run_row_delete_is_scoped_to_the_collection(self, monkeypatch, sessions):
        adapter = PGVectorAdapter()

        adapter._delete_index_runs(make_wrapper(), "idx")

        session = sessions.opened[0]
        assert len(session.deletes) == 1
        assert "elitea_index_runs.collection =" in compile_sql(session.filters[0])
        assert list(compiled_params(session.filters[0]).values()) == ["idx"]
        assert session.commits == 1

    def test_run_row_delete_spares_pending_and_cancelled_rows(self, monkeypatch, sessions):
        adapter = PGVectorAdapter()

        adapter._delete_index_runs(make_wrapper(), "idx")

        status_clause = sessions.opened[0].filters[1]
        assert "elitea_index_runs.status NOT IN" in compile_sql(status_clause)
        assert sorted(*compiled_params(status_clause).values()) == [
            RUN_STATUS_CANCELLED, RUN_STATUS_PENDING
        ]

    def test_a_schema_that_never_staged_a_run_is_not_an_error(self, monkeypatch):
        adapter = PGVectorAdapter()
        undefined_table = ProgrammingError(
            "DELETE", {}, SimpleNamespace(sqlstate="42P01")
        )

        def failing_session(bind=None):
            raise undefined_table

        monkeypatch.setattr("sqlalchemy.orm.Session", failing_session)

        adapter._delete_index_runs(make_wrapper(), "idx")

    def test_other_programming_errors_still_surface(self, monkeypatch):
        adapter = PGVectorAdapter()
        denied = ProgrammingError("DELETE", {}, SimpleNamespace(sqlstate="42501"))

        def failing_session(bind=None):
            raise denied

        monkeypatch.setattr("sqlalchemy.orm.Session", failing_session)

        with pytest.raises(ProgrammingError):
            adapter._delete_index_runs(make_wrapper(), "idx")

    def test_a_failed_run_row_cleanup_does_not_fail_the_removal(self, monkeypatch, sessions, caplog):
        adapter = PGVectorAdapter()
        monkeypatch.setattr(
            PGVectorAdapter, "_delete_index_runs",
            lambda self, wrapper, index_name: (_ for _ in ()).throw(permission_denied_error()),
        )

        with caplog.at_level(logging.WARNING):
            deleted_count = adapter.clean_collection(
                make_wrapper(), index_name="idx", including_index_meta=True
            )

        assert deleted_count == 0
        assert sessions.opened[0].commits == 1
        assert "idx" in caplog.text


class FailingAdapter:
    def __init__(self, error):
        self.error = error

    def get_pending_run_ids(self, wrapper, index_name, include_cancelled=True):
        raise self.error


def undefined_table_error():
    return ProgrammingError("SELECT", {}, SimpleNamespace(sqlstate="42P01"))


def statement_timeout_error():
    return OperationalError("SELECT", {}, SimpleNamespace(sqlstate="57014"))


def permission_denied_error():
    return ProgrammingError("SELECT", {}, SimpleNamespace(sqlstate="42501"))


class TestPendingRunLookupPolarity:
    """All three read paths degrade identically: fail-open for a missing runs
    table alone, surface anything else. A transient DB error answered with an
    unfiltered read would publish an in-flight run's staged rows."""

    def toolkit_path(self, monkeypatch, error):
        toolkit = BaseVectorStoreToolApiWrapper.model_construct()
        wrapper = SimpleNamespace(vector_adapter=FailingAdapter(error))
        monkeypatch.setattr(
            BaseVectorStoreToolApiWrapper, "_init_vector_store", lambda self: wrapper
        )
        return lambda: toolkit._get_pending_run_ids_safe("idx")

    def standalone_wrapper_path(self, monkeypatch, error):
        wrapper = VectorStoreWrapper.model_construct()
        object.__setattr__(wrapper, "vector_adapter", FailingAdapter(error))
        return lambda: wrapper._get_pending_run_ids_safe()

    def base_wrapper_path(self, monkeypatch, error):
        wrapper = VectorStoreWrapperBase.model_construct()
        object.__setattr__(wrapper, "vector_adapter", FailingAdapter(error))
        monkeypatch.setattr(
            VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None
        )
        return lambda: wrapper.get_pending_run_ids("idx")

    @pytest.fixture(params=["toolkit_path", "standalone_wrapper_path", "base_wrapper_path"])
    def read_path(self, request):
        return getattr(self, request.param)

    def test_a_missing_runs_table_degrades_to_an_unfiltered_read(
        self, monkeypatch, read_path
    ):
        assert read_path(monkeypatch, undefined_table_error())() == []

    @pytest.mark.parametrize(
        "error_factory", [statement_timeout_error, permission_denied_error]
    )
    def test_transient_and_permission_errors_surface(
        self, monkeypatch, read_path, error_factory
    ):
        with pytest.raises((OperationalError, ProgrammingError)):
            read_path(monkeypatch, error_factory())()

    def test_unexpected_errors_surface(self, monkeypatch, read_path):
        with pytest.raises(RuntimeError):
            read_path(monkeypatch, RuntimeError("boom"))()

    def test_degradable_classification_keys_on_sqlstate(self):
        assert is_degradable_run_lookup_error(undefined_table_error())
        assert not is_degradable_run_lookup_error(statement_timeout_error())
        assert not is_degradable_run_lookup_error(permission_denied_error())
        assert not is_degradable_run_lookup_error(RuntimeError("boom"))
        assert not is_degradable_run_lookup_error(ValueError("boom"))


class TestIndexMetaPredicateIsSargable:
    """The only jsonb index on the embedding table is GIN jsonb_path_ops, which
    serves @> alone; a ->> predicate sequential-scans inside promote's
    transaction with every core meta writer queued behind it."""

    def test_meta_clause_uses_containment(self):
        sql = compile_sql(PGVectorAdapter._index_meta_clause(make_store(), "idx"))

        assert "@>" in sql
        assert "->>" not in sql
        assert "jsonb_extract_path_text" not in sql
        assert list(compiled_params(PGVectorAdapter._index_meta_clause(
            make_store(), "idx")).values()) == [{"type": "index_meta", "collection": "idx"}]

    def test_meta_lock_reuses_the_containment_clause(self, sessions):
        adapter = PGVectorAdapter()
        session = RecordingSession()

        adapter._lock_index_meta_rows(session, make_store(), "idx")

        assert "@>" in compile_sql(session.filters[0])
        assert "->>" not in compile_sql(session.filters[0])

    def test_get_index_meta_uses_the_containment_clause(self, sessions):
        adapter = PGVectorAdapter()

        assert adapter.get_index_meta(make_wrapper(), "idx") == []

        sql = compile_sql(sessions.opened[0].filters[0])
        assert "@>" in sql
        assert "jsonb_extract_path_text" not in sql


class TestUnreadableIndexIsNotAnEmptyIndex:
    """An empty read result is what tells the caller there is nothing to
    supersede. A swallowed read error would publish this run's generation on
    top of the surviving one, duplicating every document silently."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.setattr(
            PGVectorAdapter, "get_pending_run_ids",
            lambda self, wrapper, index_name, include_cancelled=True: [],
        )
        return PGVectorAdapter()

    def break_session(self, monkeypatch, error):
        def failing_session(bind=None):
            raise error

        monkeypatch.setattr("sqlalchemy.orm.Session", failing_session)

    @pytest.mark.parametrize("read", ["get_indexed_data", "get_code_indexed_data"])
    @pytest.mark.parametrize(
        "error_factory", [statement_timeout_error, permission_denied_error]
    )
    def test_a_failed_read_fails_the_run(self, adapter, monkeypatch, read, error_factory):
        self.break_session(monkeypatch, error_factory())

        with pytest.raises((OperationalError, ProgrammingError)):
            getattr(adapter, read)(make_wrapper(), "idx")

    @pytest.mark.parametrize("read", ["get_indexed_data", "get_code_indexed_data"])
    def test_a_failed_pending_run_lookup_fails_the_run_too(self, monkeypatch, read):
        adapter = PGVectorAdapter()
        monkeypatch.setattr(
            PGVectorAdapter, "get_pending_run_ids",
            lambda self, wrapper, index_name, include_cancelled=True:
                (_ for _ in ()).throw(statement_timeout_error()),
        )

        with pytest.raises(OperationalError):
            getattr(adapter, read)(make_wrapper(), "idx")

    @pytest.mark.parametrize("read", ["get_indexed_data", "get_code_indexed_data"])
    def test_a_missing_embedding_table_stays_an_honest_empty(self, adapter, monkeypatch, read):
        self.break_session(monkeypatch, undefined_table_error())

        assert getattr(adapter, read)(make_wrapper(), "idx") == {}

    @pytest.mark.parametrize("read", ["get_indexed_data", "get_code_indexed_data"])
    def test_a_readable_empty_index_still_reads_as_empty(self, adapter, sessions, read):
        assert getattr(adapter, read)(make_wrapper(), "idx") == {}


class TestIndexedDataSelectsOnlyWhatItReads:
    def test_get_indexed_data_does_not_fetch_the_document_column(self, monkeypatch, sessions):
        adapter = PGVectorAdapter()
        monkeypatch.setattr(
            PGVectorAdapter, "get_pending_run_ids",
            lambda self, wrapper, index_name, include_cancelled=True: [],
        )

        assert adapter.get_indexed_data(make_wrapper(), "idx") == {}

        selected = sessions.opened[0].queries[0].entities
        assert [column.key for column in selected] == ["id", "cmetadata"]

    def test_code_indexed_data_keeps_the_same_two_columns(self, monkeypatch, sessions):
        adapter = PGVectorAdapter()
        monkeypatch.setattr(
            PGVectorAdapter, "get_pending_run_ids",
            lambda self, wrapper, index_name, include_cancelled=True: [],
        )

        assert adapter.get_code_indexed_data(make_wrapper(), "idx") == {}

        selected = sessions.opened[0].queries[0].entities
        assert [column.key for column in selected] == ["id", "cmetadata"]


class TestDiscardNeverOutrunsItsCleanup:
    """A discard whose chunk DELETE fails leaves the run row pending.

    Flipping the row terminal anyway would publish the staged rows: 'discarded' is
    outside the read filter's status set AND outside the sweep's, so nothing would
    ever hide or reclaim them. Pending keeps them invisible until the next run's
    sweep; the run's own failure is still recorded by the SDK's meta write.
    """

    @pytest.fixture
    def failing_chunk_delete(self, monkeypatch):
        def delete(self, synchronize_session=False):
            raise statement_timeout_error()

        monkeypatch.setattr(RecordingQuery, "delete", delete)

    @pytest.mark.parametrize("status", [RUN_STATUS_PENDING, RUN_STATUS_CANCELLED])
    def test_a_failed_chunk_delete_leaves_the_status_alone(
        self, sessions, failing_chunk_delete, status
    ):
        run_row = make_run_row(status)
        sessions.run_rows.append(run_row)

        with pytest.raises(OperationalError):
            PGVectorAdapter().discard_run(make_wrapper(), "idx", "run-1")

        assert run_row.status == status
        assert sessions.opened[0].commits == 0

    def test_a_successful_cleanup_still_flips_the_row(self, sessions):
        run_row = make_run_row(RUN_STATUS_PENDING)
        sessions.run_rows.append(run_row)

        assert PGVectorAdapter().discard_run(make_wrapper(), "idx", "run-1") == "discarded"
        assert run_row.status == RUN_STATUS_DISCARDED
        assert sessions.opened[0].commits == 1


STALE_BEFORE = 1_000.0


def make_sweep_row(run_id, status, heartbeat, has_chunks=True):
    return SimpleNamespace(
        run_id=run_id, status=status, heartbeat=heartbeat, has_chunks=has_chunks
    )


class SweepQuery:
    def __init__(self, session):
        self.session = session
        self.limit_value = None

    def filter(self, *criteria):
        self.session.filters.extend(criteria)
        return self

    def order_by(self, *criteria):
        self.session.order_bys.extend(criteria)
        return self

    def limit(self, value):
        self.limit_value = value
        return self

    def all(self):
        # The capped read is the discarded-chunk reclamation pass; the uncapped
        # one is the staleness candidate read. The surviving-chunk restriction is
        # applied only when the statement carries the EXISTS probe, so dropping it
        # from the production query shows up as a behavioural failure here.
        table = self.session.table
        if self.limit_value is None:
            return [(row.run_id,) for row in table.stale_candidates()]
        probes_chunks = any(
            isinstance(criterion, sa.sql.selectable.Exists)
            for criterion in self.session.filters
        )
        rows = table.discarded_rows(self.limit_value, probes_chunks)
        return [(row.run_id,) for row in rows]


class SweepSession:
    """Drives the sweep against an in-memory runs table.

    ``get`` applies the test's interleaving hook before it hands the row back:
    ``FOR UPDATE`` blocks until the concurrent writer commits and then re-reads
    the row it committed, which is exactly the window this suite covers.
    """

    def __init__(self, table):
        self.table = table
        self.filters = []
        self.order_bys = []
        self.commits = 0
        self.rollbacks = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def query(self, *entities):
        return SweepQuery(self)

    def get(self, model, run_id, with_for_update=False):
        self.table.locked.append(run_id)
        hook = self.table.on_lock.pop(run_id, None)
        if hook is not None:
            hook()
        return self.table.rows.get(run_id)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class SweepTable:
    def __init__(self, rows):
        self.rows = {row.run_id: row for row in rows}
        self.locked = []
        self.on_lock = {}
        self.sessions = []
        self.chunk_deletes = []

    def stale_candidates(self):
        return [
            row for row in self.rows.values()
            if row.status in (RUN_STATUS_PENDING, RUN_STATUS_CANCELLED)
            and row.heartbeat < STALE_BEFORE
        ]

    def discarded_rows(self, limit, probes_chunks):
        return [
            row for row in self.rows.values()
            if row.status == RUN_STATUS_DISCARDED
            and (row.has_chunks or not probes_chunks)
        ][:limit]


@pytest.fixture
def sweep(monkeypatch):
    def build(*rows, delete_error=None):
        table = SweepTable(rows)

        def session_factory(bind=None):
            session = SweepSession(table)
            table.sessions.append(session)
            return session

        def delete_chunks(self, session, store, run_ids):
            if delete_error is not None and delete_error(run_ids):
                raise statement_timeout_error()
            table.chunk_deletes.append((session, list(run_ids)))
            for run_id in run_ids:
                run_row = table.rows.get(run_id)
                if run_row is not None:
                    run_row.has_chunks = False
            return 0

        monkeypatch.setattr("sqlalchemy.orm.Session", session_factory)
        monkeypatch.setattr(PGVectorAdapter, "_delete_runs_chunks", delete_chunks)
        return table

    return build


def deleted_run_ids(table):
    return [run_ids for _, run_ids in table.chunk_deletes]


class TestSweepReVerifiesUnderTheRunRowLock:
    """The candidate read is lock-free and staleness is only a heartbeat guess:
    heartbeat failures are swallowed, so a worker whose heartbeat thread died
    keeps indexing and can promote inside the sweep's window. Deleting by run id
    without re-checking the row would erase the corpus that run just published —
    silently, with the run reporting success."""

    def sweep_once(self, table):
        return PGVectorAdapter().sweep_stale_index_runs(make_wrapper(), "idx", STALE_BEFORE)

    def test_a_run_that_promotes_inside_the_window_keeps_its_corpus(self, sweep):
        row = make_sweep_row("run-1", RUN_STATUS_PENDING, heartbeat=1.0)
        table = sweep(row)
        table.on_lock["run-1"] = lambda: setattr(row, "status", RUN_STATUS_PROMOTED)

        assert self.sweep_once(table) == []
        assert deleted_run_ids(table) == []
        assert row.status == RUN_STATUS_PROMOTED

    def test_a_run_that_heartbeats_inside_the_window_keeps_its_rows(self, sweep):
        row = make_sweep_row("run-1", RUN_STATUS_PENDING, heartbeat=1.0)
        table = sweep(row)
        table.on_lock["run-1"] = lambda: setattr(row, "heartbeat", STALE_BEFORE + 1)

        assert self.sweep_once(table) == []
        assert deleted_run_ids(table) == []
        assert row.status == RUN_STATUS_PENDING

    def test_a_row_that_vanished_inside_the_window_deletes_nothing(self, sweep):
        row = make_sweep_row("run-1", RUN_STATUS_PENDING, heartbeat=1.0)
        table = sweep(row)
        table.on_lock["run-1"] = lambda: table.rows.pop("run-1")

        assert self.sweep_once(table) == []
        assert deleted_run_ids(table) == []

    @pytest.mark.parametrize("status", [RUN_STATUS_PENDING, RUN_STATUS_CANCELLED])
    def test_a_genuinely_stale_run_is_still_reclaimed(self, sweep, status):
        row = make_sweep_row("run-dead", status, heartbeat=1.0)
        table = sweep(row)

        assert self.sweep_once(table) == ["run-dead"]
        assert deleted_run_ids(table) == [["run-dead"]]
        assert row.status == RUN_STATUS_DISCARDED

    def test_the_chunk_delete_and_the_flip_commit_together(self, sweep):
        row = make_sweep_row("run-dead", RUN_STATUS_PENDING, heartbeat=1.0)
        table = sweep(row)

        self.sweep_once(table)

        deleting_session, _ = table.chunk_deletes[0]
        assert deleting_session is table.sessions[-1]
        assert deleting_session.commits == 1
        # A crash before that commit leaves the row reclaimable; a split into a
        # committed delete plus a later flip would strand chunks under a terminal
        # row instead.
        assert [session.commits for session in table.sessions] == [0, 0, 1]

    def test_the_row_is_locked_before_its_chunks_are_deleted(self, sweep, monkeypatch):
        row = make_sweep_row("run-dead", RUN_STATUS_PENDING, heartbeat=1.0)
        table = sweep(row)
        order = []
        table.on_lock["run-dead"] = lambda: order.append("lock")
        recording_delete = PGVectorAdapter._delete_runs_chunks

        def tracking_delete(self, session, store, run_ids):
            order.append("delete")
            return recording_delete(self, session, store, run_ids)

        monkeypatch.setattr(PGVectorAdapter, "_delete_runs_chunks", tracking_delete)

        self.sweep_once(table)

        assert order == ["lock", "delete"]

    def test_a_fresh_run_is_never_a_candidate(self, sweep):
        row = make_sweep_row("run-live", RUN_STATUS_PENDING, heartbeat=STALE_BEFORE + 1)
        table = sweep(row)

        assert self.sweep_once(table) == []
        assert table.locked == []


class TestSweepReclaimsStrandedDiscardedChunks:
    """The abort branches' cleanup is best effort by design — it must not turn a
    silent abort into a reported failure. When it fails the run row is already
    terminal and its chunks survive, which is the one state the invariant
    forbids, so the sweep has to be the retry."""

    def sweep_once(self, table):
        return PGVectorAdapter().sweep_stale_index_runs(make_wrapper(), "idx", STALE_BEFORE)

    def test_a_discarded_runs_surviving_chunks_are_reclaimed(self, sweep):
        table = sweep(make_sweep_row("run-old", RUN_STATUS_DISCARDED, heartbeat=1.0))

        assert self.sweep_once(table) == []
        assert deleted_run_ids(table) == [["run-old"]]

    def test_a_promoted_runs_chunks_are_never_touched(self, sweep):
        table = sweep(
            make_sweep_row("run-live", RUN_STATUS_PROMOTED, heartbeat=2.0),
            make_sweep_row("run-old", RUN_STATUS_DISCARDED, heartbeat=1.0),
        )

        self.sweep_once(table)

        assert deleted_run_ids(table) == [["run-old"]]

    def test_a_collection_without_discarded_runs_deletes_nothing(self, sweep):
        table = sweep(make_sweep_row("run-live", RUN_STATUS_PROMOTED, heartbeat=1.0))

        assert self.sweep_once(table) == []
        assert deleted_run_ids(table) == []

    def test_a_discarded_run_without_surviving_chunks_is_never_a_candidate(self, sweep):
        table = sweep(
            make_sweep_row("run-clean", RUN_STATUS_DISCARDED, heartbeat=2.0, has_chunks=False),
            make_sweep_row("run-strand", RUN_STATUS_DISCARDED, heartbeat=1.0),
        )

        self.sweep_once(table)

        assert deleted_run_ids(table) == [["run-strand"]]

    def test_the_pass_is_bounded_per_sweep(self, sweep):
        rows = [
            make_sweep_row(f"run-{index}", RUN_STATUS_DISCARDED, heartbeat=float(index))
            for index in range(STRANDED_RECLAIM_RUN_LIMIT + 10)
        ]
        table = sweep(*rows)

        self.sweep_once(table)

        assert len(deleted_run_ids(table)[0]) == STRANDED_RECLAIM_RUN_LIMIT

    def test_the_strands_the_cap_left_behind_are_reclaimed_by_the_next_sweep(self, sweep):
        rows = [
            make_sweep_row(f"run-{index}", RUN_STATUS_DISCARDED, heartbeat=float(index))
            for index in range(STRANDED_RECLAIM_RUN_LIMIT + 10)
        ]
        table = sweep(*rows)

        self.sweep_once(table)
        self.sweep_once(table)

        assert sorted(run_id for batch in deleted_run_ids(table) for run_id in batch) == sorted(
            row.run_id for row in rows
        )

    def test_the_oldest_strand_survives_a_windows_worth_of_newer_discarded_rows(self, sweep):
        # A run stranded by the sweep stops heartbeating when it goes terminal, so
        # its heartbeat is the oldest of any discarded row: a newest-first window
        # would exclude it forever, and this pass is its only retry.
        stranded = make_sweep_row("run-stranded", RUN_STATUS_DISCARDED, heartbeat=1.0)
        table = sweep(
            *[
                make_sweep_row(
                    f"run-{index}",
                    RUN_STATUS_DISCARDED,
                    heartbeat=float(index + 2),
                    has_chunks=False,
                )
                for index in range(STRANDED_RECLAIM_RUN_LIMIT + 10)
            ],
            stranded,
        )

        self.sweep_once(table)

        assert deleted_run_ids(table) == [["run-stranded"]]

    def test_a_failed_reclaim_never_fails_the_run_that_sweeps(self, sweep, caplog):
        table = sweep(
            make_sweep_row("run-old", RUN_STATUS_DISCARDED, heartbeat=2.0),
            make_sweep_row("run-dead", RUN_STATUS_PENDING, heartbeat=1.0),
            delete_error=lambda run_ids: run_ids == ["run-old"],
        )

        with caplog.at_level(logging.WARNING):
            reclaimed = self.sweep_once(table)

        assert reclaimed == ["run-dead"]
        assert deleted_run_ids(table) == [["run-dead"]]
        assert "idx" in caplog.text

    def test_the_reclaim_read_is_scoped_to_this_collections_discarded_rows(self, sweep):
        table = sweep(make_sweep_row("run-old", RUN_STATUS_DISCARDED, heartbeat=1.0))

        self.sweep_once(table)

        collection_clause, status_clause, chunks_clause = table.sessions[0].filters
        assert "elitea_index_runs.collection =" in compile_sql(collection_clause)
        assert list(compiled_params(collection_clause).values()) == ["idx"]
        assert "elitea_index_runs.status =" in compile_sql(status_clause)
        assert list(compiled_params(status_clause).values()) == [RUN_STATUS_DISCARDED]
        chunks_sql = compile_sql(chunks_clause)
        # Correlated containment: the candidate set is exactly the deletable set,
        # and every probe is served by the jsonb_path_ops GIN.
        assert "EXISTS (SELECT langchain_pg_embedding.id" in chunks_sql
        assert (
            "langchain_pg_embedding.cmetadata @> jsonb_build_object("
            in chunks_sql
        )
        assert "elitea_index_runs.run_id)" in chunks_sql
        assert "_elitea_run_id" in compiled_params(chunks_clause).values()
        # No recency ordering: a swept run's heartbeat freezes when it goes
        # terminal, so any ordered window starves the strands it excludes.
        assert table.sessions[0].order_bys == []


class TestRunChunkDeleteStaysSargable:
    """Every chunk delete is anchored on containment so the jsonb_path_ops GIN
    serves it; an unknown run id can only be reached through an id the caller
    already resolved, never through a full-table anti-join."""

    def delete_clauses(self, run_ids):
        session = RecordingSession()
        PGVectorAdapter()._delete_runs_chunks(session, make_store(), run_ids)
        return session

    def test_each_run_id_gets_its_own_containment_probe(self):
        session = self.delete_clauses(["run-1", "run-2"])

        sql = compile_sql(session.filters[0])
        assert sql.count("@>") == 2
        assert "->>" not in sql
        assert [
            params for params in compiled_params(session.filters[0]).values()
        ] == [{"_elitea_run_id": "run-1"}, {"_elitea_run_id": "run-2"}]

    def test_the_meta_row_is_excluded_and_the_collection_is_not_a_conjunct(self):
        session = self.delete_clauses(["run-1"])

        assert len(session.deletes) == 1
        assert "index_meta" in str(compiled_params(session.filters[1]).values())
        assert "collection" not in compile_sql(session.filters[0])

    def test_the_single_run_delete_reuses_the_same_shape(self):
        session = RecordingSession()

        PGVectorAdapter()._delete_run_chunks(session, make_store(), "run-1")

        assert compile_sql(session.filters[0]).count("@>") == 1
