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

"""Both adapters must read the stored content identity back (#6645).

Without blob_shas on the way out, the loader has nothing to compare the listing
against and every file is downloaded again.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Column, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
from tenacity import wait_none

from elitea_sdk.tools.base_indexer_toolkit import indexed_rows_of
from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import (
    IDENTITY_STAMP_BATCH_SIZE,
    PGVectorAdapter,
)



class _RecordingSession:
    def __init__(self):
        self.statements = []
        self.commits = 0
        self.closes = 0
        self.cleanup_error = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        self.closes += 1
        if self.cleanup_error is not None:
            error, self.cleanup_error = self.cleanup_error, None
            raise error

    def execute(self, statement):
        self.statements.append(statement)
        return SimpleNamespace(rowcount=2)

    def commit(self):
        self.commits += 1


def values_payload(statement):
    """The (row_id, identity) pairs the statement's VALUES clause actually carries."""
    sql = str(statement.compile(dialect=postgresql.dialect(),
                                compile_kwargs={"literal_binds": True}))
    opening = "FROM (VALUES "
    clause = sql[sql.index(opening) + len(opening):sql.index(") AS stamps")]
    return [tuple(field.strip().strip("'") for field in pair.split(", "))
            for pair in clause.strip("()").split("), (")]


def _embedding_store():
    base = declarative_base()

    class EmbeddingStore(base):
        __tablename__ = "langchain_pg_embedding"
        id = Column(String, primary_key=True)
        cmetadata = Column(JSONB)

    return EmbeddingStore


@pytest.fixture
def pg_adapter(monkeypatch):
    session = _RecordingSession()
    session.binds = []

    def build_session(*args, **kwargs):
        session.binds.append(args[0] if args else None)
        return session

    monkeypatch.setattr(
        "elitea_sdk.tools.vector_adapters.VectorStoreAdapter.Session", build_session)
    wrapper = MagicMock()
    wrapper.vectorstore.EmbeddingStore = _embedding_store()
    return PGVectorAdapter(), wrapper, session


class TestPGVectorIdentityStamp:

    @pytest.fixture
    def pg(self, monkeypatch):
        session = _RecordingSession()
        monkeypatch.setattr(
            "elitea_sdk.tools.vector_adapters.VectorStoreAdapter.Session",
            lambda *a, **kw: session)
        wrapper = MagicMock()
        wrapper.vectorstore.EmbeddingStore = _embedding_store()
        return PGVectorAdapter(), wrapper, session

    def test_nothing_to_stamp_issues_no_statement(self, pg):
        adapter, wrapper, session = pg
        assert adapter.stamp_code_identity(wrapper, {}) == 0
        assert session.statements == []

    def test_one_statement_covers_many_distinct_identities(self, pg):
        adapter, wrapper, session = pg
        adapter.stamp_code_identity(wrapper, {"r1": "sha-a", "r2": "sha-a", "r3": "sha-b"})
        assert len(session.statements) == 1

    def test_the_batch_size_caps_the_transaction(self, pg):
        adapter, wrapper, session = pg
        rows = {f"r{position}": f"sha-{position}"
                for position in range(IDENTITY_STAMP_BATCH_SIZE + 1)}
        adapter.stamp_code_identity(wrapper, rows)
        assert len(session.statements) == 2
        assert session.commits == 2

    def _compiled(self, session):
        return str(session.statements[0].compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        ))

    def test_the_statement_compiles_as_postgres_sql(self, pg):
        adapter, wrapper, session = pg
        adapter.stamp_code_identity(wrapper, {"r1": "sha-a", "r2": "sha-b"})

        sql = self._compiled(session)

        assert "UPDATE langchain_pg_embedding" in sql
        assert "sha-a" in sql and "sha-b" in sql

    def test_each_row_is_matched_by_its_id_not_its_identity(self, pg):
        adapter, wrapper, session = pg
        adapter.stamp_code_identity(wrapper, {"r1": "sha-a"})

        assert ("WHERE langchain_pg_embedding.id = stamps.row_id"
                in self._compiled(session))

    def test_the_whole_set_clause_is_exactly_this(self, pg):
        """One structural assertion, because a bare substring per operand cannot see an
        operand in the wrong position - and jsonb_set's arguments are all interchangeable
        as text. A SQLAlchemy upgrade is the expected cause of a false red here: verify
        the new rendering by hand against a real server, then copy it in."""
        adapter, wrapper, session = pg
        adapter.stamp_code_identity(wrapper, {"r1": "sha-a"})

        assert (
            "SET cmetadata=jsonb_set("
            "coalesce(langchain_pg_embedding.cmetadata, '{}'::jsonb), "
            "'{blob_sha}'::text[], "
            "to_jsonb(CAST(stamps.identity AS TEXT)), "
            "true)"
        ) in self._compiled(session)

    def test_the_values_alias_names_both_columns(self, pg):
        adapter, wrapper, session = pg
        adapter.stamp_code_identity(wrapper, {"r1": "sha-a"})

        assert "AS stamps (row_id, identity)" in self._compiled(session)

    def test_the_row_metadata_is_updated_not_replaced(self, pg):
        adapter, wrapper, session = pg
        adapter.stamp_code_identity(wrapper, {"r1": "sha-a"})

        assert ("coalesce(langchain_pg_embedding.cmetadata, '{}'::jsonb)"
                in self._compiled(session))

    def test_the_path_written_is_the_blob_sha_key(self, pg):
        adapter, wrapper, session = pg
        adapter.stamp_code_identity(wrapper, {"r1": "sha-a"})

        assert "'{blob_sha}'::text[]" in self._compiled(session)

    def test_the_stamped_row_count_is_returned(self, pg):
        adapter, wrapper, _ = pg
        assert adapter.stamp_code_identity(wrapper, {"r1": "sha-a"}) == 2


class TestABadBatchDoesNotStrandTheRest:

    @pytest.fixture
    def failing_second_batch(self, monkeypatch):
        class Session(_RecordingSession):
            def execute(self, statement):
                self.statements.append(statement)
                if len(self.statements) == 2:
                    raise RuntimeError("connection reset")
                return SimpleNamespace(rowcount=IDENTITY_STAMP_BATCH_SIZE)

        session = Session()
        monkeypatch.setattr(
            "elitea_sdk.tools.vector_adapters.VectorStoreAdapter.Session",
            lambda *a, **kw: session)
        wrapper = MagicMock()
        wrapper.vectorstore.EmbeddingStore = _embedding_store()
        return PGVectorAdapter(), wrapper, session

    def test_later_batches_still_run(self, failing_second_batch):
        adapter, wrapper, session = failing_second_batch
        rows = {f"r{n}": f"sha-{n}" for n in range(IDENTITY_STAMP_BATCH_SIZE * 3)}

        adapter.stamp_code_identity(wrapper, rows)

        assert len(session.statements) == 3

    def test_the_count_reflects_only_what_landed(self, failing_second_batch):
        adapter, wrapper, _ = failing_second_batch
        rows = {f"r{n}": f"sha-{n}" for n in range(IDENTITY_STAMP_BATCH_SIZE * 3)}

        assert adapter.stamp_code_identity(wrapper, rows) == IDENTITY_STAMP_BATCH_SIZE * 2

    def test_every_batch_is_closed_out(self, failing_second_batch):
        adapter, wrapper, session = failing_second_batch
        rows = {f"r{n}": f"sha-{n}" for n in range(IDENTITY_STAMP_BATCH_SIZE * 3)}

        adapter.stamp_code_identity(wrapper, rows)

        assert session.closes == 3


class TestTheJsonbOperandsAreTypedInTheStatement:
    """A bound '{}' makes Postgres reject COALESCE(jsonb, varchar) at execution time,
    which literal_binds hides because a rendered literal is inferred as jsonb."""

    def test_no_jsonb_operand_is_left_as_a_bind_parameter(self, pg_adapter):
        adapter, wrapper, session = pg_adapter
        adapter.stamp_code_identity(wrapper, {"r1": "sha-a"})

        sql = str(session.statements[0].compile(dialect=postgresql.dialect()))

        assert "'{}'::jsonb" in sql
        assert "'{blob_sha}'::text[]" in sql


class TestTheSessionIsBoundToTheStoresEngine:
    """A Session built without its bind raises UnboundExecutionError on execute, which the
    per-batch handler swallows - another silent disarm."""

    def test_the_session_is_built_on_the_stores_bind(self, pg_adapter):
        adapter, wrapper, session = pg_adapter

        adapter.stamp_code_identity(wrapper, {"r1": "sha-a"})

        assert session.binds == [wrapper.vectorstore.session_maker.bind]


class TestTheUpdateDoesNotWaitForARowFetch:
    """Without the option SQLAlchemy falls back to 'auto', which degrades to 'fetch' -
    an extra round trip per batch and a rowcount the caller then reports as stamped."""

    def test_the_statement_says_not_to_synchronise_the_session(self, pg_adapter):
        adapter, wrapper, session = pg_adapter
        adapter.stamp_code_identity(wrapper, {"r1": "sha-a"})

        options = session.statements[0].get_execution_options()

        assert options["synchronize_session"] is False


class TestABatchThatFailsAtCommitIsNotCounted:

    def test_only_committed_batches_are_counted(self, monkeypatch):
        class Session(_RecordingSession):
            def commit(self):
                self.commits += 1
                if self.commits == 1:
                    raise RuntimeError("deadlock detected")

        session = Session()
        monkeypatch.setattr(
            "elitea_sdk.tools.vector_adapters.VectorStoreAdapter.Session",
            lambda *a, **kw: session)
        wrapper = MagicMock()
        wrapper.vectorstore.EmbeddingStore = _embedding_store()
        rows = {f"r{n}": f"sha-{n}" for n in range(IDENTITY_STAMP_BATCH_SIZE * 2)}

        assert PGVectorAdapter().stamp_code_identity(wrapper, rows) == 2


class TestATransientDatabaseErrorIsRetried:
    """A prepared-plan collision behind PgBouncer is the likeliest failure here, and it
    recovers on retry - without it one collision drops a whole batch of stamps."""

    def _pgbouncer_collision(self):
        from psycopg.errors import DuplicatePreparedStatement
        from sqlalchemy.exc import ProgrammingError

        original = DuplicatePreparedStatement("prepared statement already exists")
        return ProgrammingError("SELECT 1", {}, original)

    def _adapter_failing(self, monkeypatch, failures, error):
        attempts = {"count": 0}

        class Session(_RecordingSession):
            def execute(self, statement):
                self.statements.append(statement)
                attempts["count"] += 1
                if attempts["count"] <= failures:
                    raise error
                return SimpleNamespace(rowcount=1)

        constructed = []

        def build_session(*args, **kwargs):
            session = Session()
            constructed.append(session)
            return session

        monkeypatch.setattr(
            "elitea_sdk.tools.vector_adapters.VectorStoreAdapter.Session", build_session)
        monkeypatch.setattr(
            PGVectorAdapter._stamp_one_batch.retry, "wait", wait_none())
        wrapper = MagicMock()
        wrapper.vectorstore.EmbeddingStore = _embedding_store()
        attempts["sessions"] = constructed
        return PGVectorAdapter(), wrapper, attempts

    def test_a_collision_is_retried_and_lands(self, monkeypatch):
        adapter, wrapper, attempts = self._adapter_failing(
            monkeypatch, failures=1, error=self._pgbouncer_collision())

        assert adapter.stamp_code_identity(wrapper, {"r1": "sha-a"}) == 1
        assert attempts["count"] == 2

    def test_the_retry_attempt_gets_a_session_of_its_own(self, monkeypatch):
        adapter, wrapper, attempts = self._adapter_failing(
            monkeypatch, failures=1, error=self._pgbouncer_collision())

        adapter.stamp_code_identity(wrapper, {"r1": "sha-a"})

        assert len(attempts["sessions"]) == 2
        assert all(session.closes == 1 for session in attempts["sessions"])

    def test_two_consecutive_collisions_still_land(self, monkeypatch):
        adapter, wrapper, attempts = self._adapter_failing(
            monkeypatch, failures=2, error=self._pgbouncer_collision())

        assert adapter.stamp_code_identity(wrapper, {"r1": "sha-a"}) == 1
        assert attempts["count"] == 3

    def test_a_batch_is_given_up_after_three_attempts(self, monkeypatch):
        adapter, wrapper, attempts = self._adapter_failing(
            monkeypatch, failures=99, error=self._pgbouncer_collision())

        assert adapter.stamp_code_identity(wrapper, {"r1": "sha-a"}) == 0
        assert attempts["count"] == 3

    def test_a_deterministic_error_is_not_retried(self, monkeypatch):
        from sqlalchemy.exc import IntegrityError

        adapter, wrapper, attempts = self._adapter_failing(
            monkeypatch, failures=1, error=IntegrityError("x", {}, Exception("dup key")))

        assert adapter.stamp_code_identity(wrapper, {"r1": "sha-a"}) == 0
        assert attempts["count"] == 1


class TestAFailingCleanupDoesNotCostTheRetry:
    """Cleanup that raises must not replace the exception being handled: the replacement
    classifies as non-transient, so the retry the original error earned would be lost."""

    def test_a_transient_error_is_still_retried_when_cleanup_raises(self, monkeypatch):
        from sqlalchemy.exc import InvalidRequestError, OperationalError

        attempts = {"count": 0}

        class Session(_RecordingSession):
            def execute(self, statement):
                self.statements.append(statement)
                attempts["count"] += 1
                if attempts["count"] == 1:
                    self.cleanup_error = InvalidRequestError("close failed")
                    raise OperationalError("SELECT 1", {}, Exception("server closed"))
                return SimpleNamespace(rowcount=1)

        constructed = []

        def build_session(*args, **kwargs):
            session = Session()
            constructed.append(session)
            return session

        monkeypatch.setattr(
            "elitea_sdk.tools.vector_adapters.VectorStoreAdapter.Session", build_session)
        monkeypatch.setattr(PGVectorAdapter._stamp_one_batch.retry, "wait", wait_none())
        wrapper = MagicMock()
        wrapper.vectorstore.EmbeddingStore = _embedding_store()

        assert PGVectorAdapter().stamp_code_identity(wrapper, {"r1": "sha-a"}) == 1
        assert attempts["count"] == 2
        assert len(constructed) == 2

    def test_each_batch_gets_a_session_of_its_own(self, monkeypatch):
        constructed = []

        def build_session(*args, **kwargs):
            session = _RecordingSession()
            constructed.append(session)
            return session

        monkeypatch.setattr(
            "elitea_sdk.tools.vector_adapters.VectorStoreAdapter.Session", build_session)
        wrapper = MagicMock()
        wrapper.vectorstore.EmbeddingStore = _embedding_store()
        rows = {f"r{n}": f"sha-{n}" for n in range(IDENTITY_STAMP_BATCH_SIZE * 2)}

        PGVectorAdapter().stamp_code_identity(wrapper, rows)

        assert len(constructed) == 2
        assert all(session.closes == 1 for session in constructed)

    def test_a_poisoned_attempt_does_not_reach_the_next_batch(self, monkeypatch):
        from sqlalchemy.exc import InvalidRequestError, OperationalError

        constructed = []

        class Session(_RecordingSession):
            def execute(self, statement):
                self.statements.append(statement)
                if len(constructed) == 1:
                    self.cleanup_error = InvalidRequestError("close failed")
                    raise OperationalError("UPDATE", {}, Exception("server closed"))
                return SimpleNamespace(rowcount=1)

        def build_session(*args, **kwargs):
            session = Session()
            constructed.append(session)
            return session

        monkeypatch.setattr(
            "elitea_sdk.tools.vector_adapters.VectorStoreAdapter.Session", build_session)
        monkeypatch.setattr(PGVectorAdapter._stamp_one_batch.retry, "wait", wait_none())
        wrapper = MagicMock()
        wrapper.vectorstore.EmbeddingStore = _embedding_store()
        rows = {f"r{n}": f"sha-{n}" for n in range(IDENTITY_STAMP_BATCH_SIZE * 2)}

        stamped = PGVectorAdapter().stamp_code_identity(wrapper, rows)

        assert constructed[0].closes == 1
        assert len(constructed) == 3
        assert stamped == 2


class TestEveryPairReachesTheDatabaseExactlyOnce:
    """The compiled-SQL assertions elsewhere read one statement and ignore its payload,
    so the batch arithmetic itself needs pinning: a slice that never advances re-stamps
    the first batch forever while rowcount still climbs and the run reports success."""

    @pytest.fixture
    def written_batches(self, monkeypatch):
        session = _RecordingSession()
        monkeypatch.setattr(
            "elitea_sdk.tools.vector_adapters.VectorStoreAdapter.Session",
            lambda *a, **kw: session)
        wrapper = MagicMock()
        wrapper.vectorstore.EmbeddingStore = _embedding_store()
        pairs = [(f"r{n}", f"sha-{n}") for n in range(IDENTITY_STAMP_BATCH_SIZE * 2 + 3)]

        PGVectorAdapter().stamp_code_identity(wrapper, dict(pairs))

        return pairs, [values_payload(statement) for statement in session.statements]

    def test_the_batches_together_carry_every_pair_once(self, written_batches):
        pairs, batches = written_batches

        assert [pair for batch in batches for pair in batch] == pairs

    def test_each_batch_carries_its_own_slice(self, written_batches):
        pairs, batches = written_batches
        size = IDENTITY_STAMP_BATCH_SIZE

        assert batches == [pairs[0:size], pairs[size:size * 2], pairs[size * 2:]]

    def test_the_tail_is_not_padded_out_to_a_full_batch(self, written_batches):
        _, batches = written_batches

        assert [len(batch) for batch in batches] == [IDENTITY_STAMP_BATCH_SIZE,
                                                     IDENTITY_STAMP_BATCH_SIZE, 3]


class TestPGVectorReadsTheIdentityBackOut:
    """The half of the feature the loader depends on: without blob_sha coming back out of
    the database the skip can never arm, and the back-fill re-derives the same work every
    run while the report claims success."""

    def _read_back(self, monkeypatch, rows):
        session = _RecordingSession()
        session.query = lambda *columns: SimpleNamespace(
            filter=lambda *criteria: SimpleNamespace(all=lambda: rows))
        monkeypatch.setattr("sqlalchemy.orm.Session", lambda *a, **kw: session)
        adapter = PGVectorAdapter()
        monkeypatch.setattr(PGVectorAdapter, "get_pending_run_ids",
                            lambda self, wrapper, name: [])
        wrapper = MagicMock()
        wrapper.vectorstore.EmbeddingStore = _embedding_store()
        return adapter.get_code_indexed_data(wrapper, "x")

    def test_the_identity_of_every_row_comes_back(self, monkeypatch):
        result = self._read_back(monkeypatch, [
            ("id-0", {"filename": "a.py", "commit_hash": "h1", "blob_sha": "sha-a"}),
            ("id-1", {"filename": "a.py", "commit_hash": "h1", "blob_sha": "sha-a"}),
        ])

        assert result["a.py"]["blob_shas"] == ["sha-a", "sha-a"]

    def test_a_row_that_never_gained_one_comes_back_as_none(self, monkeypatch):
        result = self._read_back(monkeypatch, [
            ("id-0", {"filename": "a.py", "commit_hash": "h1"}),
        ])

        assert result["a.py"]["blob_shas"] == [None]

    def test_the_three_lists_stay_aligned_when_a_hash_is_missing(self, monkeypatch):
        result = self._read_back(monkeypatch, [
            ("id-0", {"filename": "a.py", "blob_sha": "sha-a"}),
            ("id-1", {"filename": "a.py", "commit_hash": "h1", "blob_sha": "sha-b"}),
        ])
        entry = result["a.py"]

        assert len(entry["ids"]) == len(entry["commit_hashes"]) == len(entry["blob_shas"])

    def test_each_row_keeps_its_own_hash_and_identity_together(self, monkeypatch):
        result = self._read_back(monkeypatch, [
            ("id-0", {"filename": "a.py", "blob_sha": "sha-a"}),
            ("id-1", {"filename": "a.py", "commit_hash": "h1", "blob_sha": "sha-b"}),
        ])

        assert list(indexed_rows_of(result["a.py"])) == [
            ("id-0", None, "sha-a"),
            ("id-1", "h1", "sha-b"),
        ]

    def test_a_row_without_a_filename_is_left_out(self, monkeypatch):
        result = self._read_back(monkeypatch, [
            ("id-0", {"commit_hash": "h1", "blob_sha": "sha-a"}),
        ])

        assert result == {}
