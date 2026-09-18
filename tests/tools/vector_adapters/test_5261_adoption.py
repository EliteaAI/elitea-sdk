"""#5261 — a later run adopts an interrupted run's staged rows.

The invariant every test here defends: a staged row is never reachable by a reader.
It starts hidden under the interrupted run ('pending' or 'cancelled'), stays hidden while
it is parked, and becomes hidden under the live run the moment it is re-stamped. The one
status that hides nothing is 'discarded', so the parked row may only take it once no row
carries its id any more.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Column, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

from elitea_sdk.runtime.tools.index_runs_model import (
    RUN_STATUS_CANCELLED,
    RUN_STATUS_DISCARDED,
    RUN_STATUS_PENDING,
    RUN_STATUS_PROMOTED,
)
from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import PGVectorAdapter


def _embedding_store():
    base = declarative_base()

    class EmbeddingStore(base):
        __tablename__ = "langchain_pg_embedding"
        id = Column(String, primary_key=True)
        cmetadata = Column(JSONB)

    return EmbeddingStore


def compiled(clause):
    """SQL text plus bound values — JSONB literals have no literal renderer."""
    rendered = clause.compile(dialect=postgresql.dialect())
    return f"{rendered} {rendered.params!r}"


class FakeQuery:
    def __init__(self, session, rows):
        self.session = session
        self.rows = rows

    def filter(self, *criteria):
        self.session.filters.extend(criteria)
        return self

    def order_by(self, *_):
        return self

    def with_for_update(self):
        return self

    def all(self):
        return list(self.rows)

    def limit(self, *_):
        return self

    def first(self):
        return self.rows[0] if self.rows else None

    def scalar(self):
        return self.session.count_result

    def update(self, values, synchronize_session=False):
        self.session.updates.append(values)
        return len(self.rows)

    def delete(self, synchronize_session=False):
        if self.session.delete_error is not None:
            raise self.session.delete_error
        self.session.deletes.append(self.rows)
        return len(self.rows)

    def subquery(self):
        return self

    def select_from(self, *_):
        return self


class FakeSession:
    def __init__(self, rows=(), run_row=None):
        self.filters = []
        self.updates = []
        self.rows = list(rows)
        self.run_row = run_row
        self.commits = 0
        self.rollbacks = 0
        self.count_result = 0
        self.deletes = []
        self.delete_error = None
        self.locked_for_update = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, *_entities):
        return FakeQuery(self, self.rows)

    def get(self, _model, key, with_for_update=False):
        if with_for_update:
            self.locked_for_update.append(key)
        return self.run_row

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def run_row(status, heartbeat=0.0):
    return SimpleNamespace(run_id="old-run", status=status, heartbeat=heartbeat,
                           started_on=0.0, promoted_on=None)


@pytest.fixture
def adapter_and_wrapper(monkeypatch):
    wrapper = MagicMock()
    wrapper.vectorstore.EmbeddingStore = _embedding_store()

    def install(session):
        monkeypatch.setattr(
            "elitea_sdk.tools.vector_adapters.VectorStoreAdapter.Session",
            lambda *a, **kw: session)
        return session

    return PGVectorAdapter(), wrapper, install


class TestClaimAdoptableRun:

    def test_a_stale_pending_run_is_parked_as_cancelled_and_named(self, adapter_and_wrapper):
        """Mutation: park it as RUN_STATUS_DISCARDED."""
        adapter, wrapper, install = adapter_and_wrapper
        row = run_row(RUN_STATUS_PENDING, heartbeat=1.0)
        install(FakeSession(rows=[("old-run", RUN_STATUS_PENDING)], run_row=row))

        claimed = adapter.claim_adoptable_run(wrapper, "docs", stale_before=100.0)

        assert claimed == "old-run"
        assert row.status == RUN_STATUS_CANCELLED

    def test_a_live_pending_run_is_never_claimed(self, adapter_and_wrapper):
        """A heartbeating pending row is a concurrent run; parking it would hide a
        running corpus from its own owner. Mutation: drop the heartbeat check."""
        adapter, wrapper, install = adapter_and_wrapper
        row = run_row(RUN_STATUS_PENDING, heartbeat=500.0)
        session = install(FakeSession(rows=[("old-run", RUN_STATUS_PENDING)], run_row=row))

        assert adapter.claim_adoptable_run(wrapper, "docs", stale_before=100.0) is None
        assert row.status == RUN_STATUS_PENDING
        assert session.rollbacks == 1

    def test_an_already_cancelled_run_is_claimed_whatever_its_heartbeat(self, adapter_and_wrapper):
        adapter, wrapper, install = adapter_and_wrapper
        row = run_row(RUN_STATUS_CANCELLED, heartbeat=500.0)
        install(FakeSession(rows=[("old-run", RUN_STATUS_CANCELLED)], run_row=row))

        assert adapter.claim_adoptable_run(wrapper, "docs", stale_before=100.0) == "old-run"
        assert row.status == RUN_STATUS_CANCELLED

    def test_nothing_to_claim_returns_none(self, adapter_and_wrapper):
        adapter, wrapper, install = adapter_and_wrapper
        install(FakeSession(rows=[], run_row=None))

        assert adapter.claim_adoptable_run(wrapper, "docs", stale_before=100.0) is None

    def test_a_run_larger_than_the_reuse_cap_is_never_claimed(self, adapter_and_wrapper):
        """Claiming it would exclude it from this run's sweep while adoption declines it
        anyway, pinning the largest runs out of reach of the reclaim that should retire
        them. Mutation: drop the max_chunks guard from the claim."""
        adapter, wrapper, install = adapter_and_wrapper
        row = run_row(RUN_STATUS_PENDING, heartbeat=1.0)
        session = install(FakeSession(rows=[("old-run", RUN_STATUS_PENDING)], run_row=row))
        session.count_result = 3

        claimed = adapter.claim_adoptable_run(
            wrapper, "docs", stale_before=100.0, max_chunks=2)

        assert claimed is None
        assert row.status == RUN_STATUS_PENDING
        assert session.rollbacks == 1

    def test_a_run_inside_the_cap_is_claimed(self, adapter_and_wrapper):
        adapter, wrapper, install = adapter_and_wrapper
        row = run_row(RUN_STATUS_PENDING, heartbeat=1.0)
        session = install(FakeSession(rows=[("old-run", RUN_STATUS_PENDING)], run_row=row))
        session.count_result = 1

        assert adapter.claim_adoptable_run(
            wrapper, "docs", stale_before=100.0, max_chunks=10) == "old-run"
        assert row.status == RUN_STATUS_CANCELLED

    def test_the_candidate_row_is_locked_before_it_is_parked(self, adapter_and_wrapper):
        adapter, wrapper, install = adapter_and_wrapper
        session = install(FakeSession(rows=[("old-run", RUN_STATUS_PENDING)],
                                      run_row=run_row(RUN_STATUS_PENDING, heartbeat=1.0)))

        adapter.claim_adoptable_run(wrapper, "docs", stale_before=100.0)

        assert session.locked_for_update == ["old-run"]


class TestAdoptRunChunks:

    def test_the_parked_row_stays_cancelled_so_a_surviving_worker_stays_hidden(
            self, adapter_and_wrapper):
        """A Stop is best-effort: core skips stop_task when it cannot corroborate the id,
        so the worker may still be writing under this run id. 'discarded' is hidden from
        no read path, so retiring the row here would publish those writes.
        Mutation: set RUN_STATUS_DISCARDED after the re-stamp."""
        adapter, wrapper, install = adapter_and_wrapper
        row = run_row(RUN_STATUS_CANCELLED)
        session = install(FakeSession(rows=["r1", "r2"], run_row=row))

        adopted = adapter.adopt_run_chunks(wrapper, "docs", "old-run", "new-run")

        assert adopted == 2
        assert row.status == RUN_STATUS_CANCELLED
        assert row.status != RUN_STATUS_DISCARDED
        assert len(session.updates) == 1
        assert session.commits == 1

    def test_drop_run_chunks_propagates_a_failed_delete(self, adapter_and_wrapper):
        """The rows are live under the current run's id, so a delete that failed quietly
        would let promote publish them beside the copies the run embeds. This is the real
        adapter path, not a stub. Mutation: delegate to _discard_stranded_run_chunks."""
        adapter, wrapper, install = adapter_and_wrapper
        session = install(FakeSession(rows=["r1"]))
        session.delete_error = RuntimeError("delete blew up")

        with pytest.raises(RuntimeError):
            adapter.drop_run_chunks(wrapper, "old-run")

    def test_a_run_that_is_not_parked_is_never_adopted(self, adapter_and_wrapper):
        """Only a 'cancelled' row is known to have no live writer. Mutation: accept
        RUN_STATUS_PENDING too."""
        adapter, wrapper, install = adapter_and_wrapper
        row = run_row(RUN_STATUS_PROMOTED)
        session = install(FakeSession(rows=["r1"], run_row=row))

        assert adapter.adopt_run_chunks(wrapper, "docs", "old-run", "new-run") == 0
        assert row.status == RUN_STATUS_PROMOTED
        assert session.updates == []
        assert session.rollbacks == 1

    def test_a_vanished_run_row_is_not_adopted(self, adapter_and_wrapper):
        adapter, wrapper, install = adapter_and_wrapper
        session = install(FakeSession(rows=["r1"], run_row=None))

        assert adapter.adopt_run_chunks(wrapper, "docs", "old-run", "new-run") == 0
        assert session.updates == []


class TestRestampPredicateMatchesTheDeletePredicate:

    def test_the_restamp_is_scoped_by_containment_on_the_source_run_id(self, adapter_and_wrapper):
        adapter, wrapper, install = adapter_and_wrapper
        session = install(FakeSession(rows=["r1"]))

        adapter._restamp_run_chunks(session, wrapper.vectorstore, "old-run", "new-run")

        rendered = " ".join(compiled(clause) for clause in session.filters)
        assert "@>" in rendered
        assert "old-run" in rendered

    def test_the_restamp_has_no_collection_conjunct(self, adapter_and_wrapper):
        """Multi-index rows carry an appended "a;b" collection, so an equality conjunct
        would skip them and leave them stamped with a retired run id."""
        adapter, wrapper, install = adapter_and_wrapper
        session = install(FakeSession(rows=["r1"]))

        adapter._restamp_run_chunks(session, wrapper.vectorstore, "old-run", "new-run")

        rendered = " ".join(compiled(clause) for clause in session.filters)
        assert "collection" not in rendered

    def test_the_restamp_spares_the_index_meta_row(self, adapter_and_wrapper):
        adapter, wrapper, install = adapter_and_wrapper
        session = install(FakeSession(rows=["r1"]))

        adapter._restamp_run_chunks(session, wrapper.vectorstore, "old-run", "new-run")

        rendered = " ".join(compiled(clause) for clause in session.filters)
        assert "index_meta" in rendered

    def test_the_restamp_writes_the_target_run_id(self, adapter_and_wrapper):
        adapter, wrapper, install = adapter_and_wrapper
        session = install(FakeSession(rows=["r1"]))

        adapter._restamp_run_chunks(session, wrapper.vectorstore, "old-run", "new-run")

        written = " ".join(compiled(value) for value in session.updates[0].values())
        assert "jsonb_set" in written
        assert "new-run" in written
        assert "_elitea_run_id" in written


class TestNonStagingAdaptersStayInert:

    def test_the_base_adapter_claims_nothing_and_adopts_nothing(self):
        from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import ChromaAdapter

        adapter = ChromaAdapter()
        assert adapter.claim_adoptable_run(MagicMock(), "docs", 0.0) is None
        assert adapter.adopt_run_chunks(MagicMock(), "docs", "a", "b") == 0
