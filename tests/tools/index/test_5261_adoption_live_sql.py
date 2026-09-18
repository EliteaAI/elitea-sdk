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

"""Runs #5261's adoption against a real Postgres.

Every other test for adoption stops at a compiled statement or a fake session, so the
jsonb re-stamp, the server-side content digest and the yield_per read have never touched
a database. The two properties that can only be proven here are that the re-stamp moves
exactly the rows the delete predicate would have deleted, and that the digest the server
computes matches the digest this process predicts for the same chunk — the whole reuse
mechanism is that equality.

Requires a reachable Postgres; skipped otherwise, and CI skips this directory.
"""

import functools
import json
import os
import time
import uuid

import pytest
import sqlalchemy as sa
from langchain_core.documents import Document

from elitea_sdk.runtime.tools.index_runs_model import ensure_index_runs_table
from elitea_sdk.tools.base_indexer_toolkit import candidate_chunk_digest, chunk_digest
from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import PGVectorAdapter

CONNECTION_STRING = os.getenv(
    "INDEX_RUNS_TEST_CONNECTION_STRING",
    "postgresql+psycopg://centry:changeme@localhost:5432/db",
)


@functools.lru_cache(maxsize=1)
def _base_engine():
    try:
        engine = sa.create_engine(
            CONNECTION_STRING, pool_pre_ping=True, connect_args={"connect_timeout": 2})
        with engine.connect():
            return engine
    except Exception:
        return None


@pytest.fixture
def index_schema():
    base = _base_engine()
    if base is None:
        pytest.skip("live Postgres at localhost:5432 is not reachable")
    schema = f"adopt5261_{uuid.uuid4().hex[:10]}"
    with base.begin() as connection:
        connection.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
        connection.execute(sa.text(
            f'CREATE TABLE "{schema}".langchain_pg_embedding '
            f'(id text primary key, document text, cmetadata jsonb)'
        ))
    engine = sa.create_engine(
        CONNECTION_STRING,
        connect_args={"options": f"-csearch_path={schema}"},
    )
    ensure_index_runs_table(engine, schema)
    try:
        yield schema, engine
    finally:
        engine.dispose()
        with base.begin() as connection:
            connection.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


def wrapper_for(schema, engine):
    from sqlalchemy import Column, String, Text
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.orm import declarative_base

    base = declarative_base()

    class EmbeddingStore(base):
        __tablename__ = "langchain_pg_embedding"
        __table_args__ = {"schema": schema}
        id = Column(String, primary_key=True)
        document = Column(Text)
        cmetadata = Column(JSONB)

    class SessionMaker:
        bind = engine

    class Store:
        session_maker = SessionMaker()
        EmbeddingStore = None

    Store.EmbeddingStore = EmbeddingStore

    class Wrapper:
        vectorstore = Store()

    return Wrapper()


def seed_run(engine, schema, run_id, status, heartbeat):
    with engine.begin() as connection:
        connection.execute(sa.text(
            f'INSERT INTO "{schema}".elitea_index_runs '
            f'(run_id, collection, status, started_on, heartbeat) '
            f'VALUES (:r, :c, :s, :o, :h)'
        ), {"r": run_id, "c": "docs", "s": status, "o": heartbeat, "h": heartbeat})


def seed_chunk(engine, schema, row_id, document, metadata):
    with engine.begin() as connection:
        connection.execute(sa.text(
            f'INSERT INTO "{schema}".langchain_pg_embedding (id, document, cmetadata) '
            f'VALUES (:i, :d, CAST(:m AS jsonb))'
        ), {"i": row_id, "d": document, "m": json.dumps(metadata)})


def run_ids_of(engine, schema):
    with engine.connect() as connection:
        return dict(connection.execute(sa.text(
            f"SELECT id, cmetadata->>'_elitea_run_id' FROM \"{schema}\".langchain_pg_embedding"
        )).all())


def run_status(engine, schema, run_id):
    with engine.connect() as connection:
        return connection.execute(sa.text(
            f'SELECT status FROM "{schema}".elitea_index_runs WHERE run_id = :r'
        ), {"r": run_id}).scalar()


class TestTheFullAdoptionCycle:

    def test_a_crashed_runs_rows_move_onto_the_next_run(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_run(engine, schema, "old-run", "pending", heartbeat=1.0)
        seed_chunk(engine, schema, "row-1", "alpha",
                   {"id": "1", "collection": "docs", "_elitea_run_id": "old-run"})
        seed_chunk(engine, schema, "row-2", "beta",
                   {"id": "2", "collection": "docs", "_elitea_run_id": "old-run"})

        claimed = adapter.claim_adoptable_run(wrapper, "docs", stale_before=time.time())
        assert claimed == "old-run"
        assert run_status(engine, schema, "old-run") == "cancelled"

        moved = adapter.adopt_run_chunks(wrapper, "docs", "old-run", "new-run")

        assert moved == 2
        assert run_ids_of(engine, schema) == {"row-1": "new-run", "row-2": "new-run"}
        # Still 'cancelled': a Stop is best-effort, so a surviving worker may write more
        # rows under this id and only a hidden status keeps them out of search.
        assert run_status(engine, schema, "old-run") == "cancelled"

    def test_a_live_run_is_left_alone(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_run(engine, schema, "live-run", "pending", heartbeat=time.time() + 600)

        assert adapter.claim_adoptable_run(wrapper, "docs", stale_before=time.time()) is None
        assert run_status(engine, schema, "live-run") == "pending"

    def test_a_stopped_run_is_adoptable_however_old_its_heartbeat(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_run(engine, schema, "stopped-run", "cancelled", heartbeat=1.0)
        seed_chunk(engine, schema, "row-1", "alpha",
                   {"id": "1", "collection": "docs", "_elitea_run_id": "stopped-run"})

        assert adapter.claim_adoptable_run(
            wrapper, "docs", stale_before=time.time()) == "stopped-run"

    def test_a_drained_run_is_not_claimed(self, index_schema):
        """Claiming it would exclude it from this run's sweep forever without adopting
        anything. Mutation: drop _run_chunks_exist_clause from the candidate query."""
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_run(engine, schema, "drained-run", "cancelled", heartbeat=1.0)

        assert adapter.claim_adoptable_run(wrapper, "docs", stale_before=time.time()) is None

    def test_a_run_over_the_reuse_cap_is_never_claimed_and_stays_sweepable(self, index_schema):
        """Claiming it would hand its id back as except_run_id, excluding it from the
        sweep, while adoption declines it anyway — so the largest runs would never be
        reclaimed. Mutation: drop the max_chunks guard from claim_adoptable_run."""
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_run(engine, schema, "big-run", "cancelled", heartbeat=1.0)
        for index in range(3):
            seed_chunk(engine, schema, f"row-{index}", f"body {index}",
                       {"id": str(index), "collection": "docs", "_elitea_run_id": "big-run"})

        assert adapter.claim_adoptable_run(
            wrapper, "docs", stale_before=time.time(), max_chunks=2) is None
        assert run_status(engine, schema, "big-run") == "cancelled"

        # Unclaimed, so the sweep is free to retire it.
        reclaimed = adapter.sweep_stale_index_runs(wrapper, "docs", time.time(), None)

        assert reclaimed == ["big-run"]
        assert run_ids_of(engine, schema) == {}

    def test_a_run_inside_the_cap_is_claimed(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_run(engine, schema, "small-run", "cancelled", heartbeat=1.0)
        seed_chunk(engine, schema, "row-1", "alpha",
                   {"id": "1", "collection": "docs", "_elitea_run_id": "small-run"})

        assert adapter.claim_adoptable_run(
            wrapper, "docs", stale_before=time.time(), max_chunks=10) == "small-run"

    def test_drop_run_chunks_raises_when_the_delete_fails(self, index_schema):
        """The rows are live under the current run's id, so a swallowed failure would let
        promote publish them beside the copies the run embeds. Mutation: delegate to
        _discard_stranded_run_chunks, which is best-effort by contract."""
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_chunk(engine, schema, "row-1", "alpha",
                   {"id": "1", "collection": "docs", "_elitea_run_id": "new-run"})
        with engine.begin() as connection:
            connection.execute(sa.text(f'DROP TABLE "{schema}".langchain_pg_embedding'))

        with pytest.raises(Exception):
            adapter.drop_run_chunks(wrapper, "new-run")

    def test_drop_run_chunks_removes_only_that_runs_rows(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_chunk(engine, schema, "mine", "alpha",
                   {"id": "1", "collection": "docs", "_elitea_run_id": "new-run"})
        seed_chunk(engine, schema, "theirs", "alpha",
                   {"id": "2", "collection": "docs", "_elitea_run_id": "other-run"})

        adapter.drop_run_chunks(wrapper, "new-run")

        assert run_ids_of(engine, schema) == {"theirs": "other-run"}


class TestTheRestampMovesExactlyTheDeletePredicatesRows:

    def test_a_multi_index_row_is_moved(self, index_schema):
        """Its collection is the appended "docs;other", which an equality conjunct would
        miss — leaving the row stamped with a retired run id and so permanently visible."""
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_run(engine, schema, "old-run", "cancelled", heartbeat=1.0)
        seed_chunk(engine, schema, "row-multi", "alpha",
                   {"id": "1", "collection": "docs;other", "_elitea_run_id": "old-run"})

        adapter.adopt_run_chunks(wrapper, "docs", "old-run", "new-run")

        assert run_ids_of(engine, schema)["row-multi"] == "new-run"

    def test_the_index_meta_row_is_never_moved(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_run(engine, schema, "old-run", "cancelled", heartbeat=1.0)
        seed_chunk(engine, schema, "meta-row", "index_meta_docs",
                   {"type": "index_meta", "collection": "docs", "_elitea_run_id": "old-run"})

        adapter.adopt_run_chunks(wrapper, "docs", "old-run", "new-run")

        assert run_ids_of(engine, schema)["meta-row"] == "old-run"

    def test_another_runs_rows_are_untouched(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_run(engine, schema, "old-run", "cancelled", heartbeat=1.0)
        seed_chunk(engine, schema, "mine", "alpha",
                   {"id": "1", "collection": "docs", "_elitea_run_id": "old-run"})
        seed_chunk(engine, schema, "theirs", "alpha",
                   {"id": "2", "collection": "docs", "_elitea_run_id": "other-run"})

        adapter.adopt_run_chunks(wrapper, "docs", "old-run", "new-run")

        stamped = run_ids_of(engine, schema)
        assert stamped == {"mine": "new-run", "theirs": "other-run"}

    def test_the_rest_of_the_metadata_survives_the_restamp(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_run(engine, schema, "old-run", "cancelled", heartbeat=1.0)
        seed_chunk(engine, schema, "row-1", "alpha",
                   {"id": "1", "collection": "docs", "commit_hash": "h1",
                    "_elitea_run_id": "old-run"})

        adapter.adopt_run_chunks(wrapper, "docs", "old-run", "new-run")

        with engine.connect() as connection:
            assert connection.execute(sa.text(
                f"SELECT cmetadata->>'commit_hash' FROM \"{schema}\".langchain_pg_embedding"
            )).scalar() == "h1"


class TestTheServerDigestMatchesThePredictedDigest:
    """The reuse mechanism is exactly this equality. If the server's sha256 of the stored
    text and this process's sha256 of the candidate text ever disagree, every chunk misses
    and the feature is silently dead."""

    def read_digests(self, adapter, wrapper, run_id):
        digests, row_pks, truncated = adapter.read_run_staged_digests(
            wrapper, run_id, chunk_digest, cap=1000)
        assert truncated is False
        return digests, row_pks

    def test_a_stored_row_matches_the_chunk_that_would_have_written_it(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        metadata = {"id": "1", "collection": "docs", "commit_hash": "h1"}
        seed_chunk(engine, schema, "row-1", "hello world",
                   {**metadata, "_elitea_run_id": "new-run"})

        digests, row_pks = self.read_digests(adapter, wrapper, "new-run")

        candidate = Document(page_content="hello world",
                             metadata={**metadata, "_elitea_run_id": "new-run"})
        assert row_pks == {"row-1"}
        assert digests[candidate_chunk_digest(candidate)] == ["row-1"]

    def test_a_different_body_does_not_match(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_chunk(engine, schema, "row-1", "hello world",
                   {"id": "1", "_elitea_run_id": "new-run"})

        digests, _ = self.read_digests(adapter, wrapper, "new-run")

        other = Document(page_content="goodbye world", metadata={"id": "1"})
        assert candidate_chunk_digest(other) not in digests

    def test_a_changed_commit_hash_does_not_match(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_chunk(engine, schema, "row-1", "hello world",
                   {"id": "1", "commit_hash": "old", "_elitea_run_id": "new-run"})

        digests, _ = self.read_digests(adapter, wrapper, "new-run")

        edited = Document(page_content="hello world",
                          metadata={"id": "1", "commit_hash": "new"})
        assert candidate_chunk_digest(edited) not in digests

    def test_identical_chunks_yield_two_claimable_rows(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        for row_id in ("row-1", "row-2"):
            seed_chunk(engine, schema, row_id, "same body",
                       {"id": "1", "_elitea_run_id": "new-run"})

        digests, row_pks = self.read_digests(adapter, wrapper, "new-run")

        candidate = Document(page_content="same body", metadata={"id": "1"})
        assert sorted(digests[candidate_chunk_digest(candidate)]) == ["row-1", "row-2"]
        assert row_pks == {"row-1", "row-2"}

    def test_the_meta_row_is_not_offered_for_reuse(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_chunk(engine, schema, "meta-row", "index_meta_docs",
                   {"type": "index_meta", "_elitea_run_id": "new-run"})

        _, row_pks = self.read_digests(adapter, wrapper, "new-run")

        assert row_pks == set()

    def test_the_cap_reports_truncation_rather_than_a_partial_map(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        for index in range(3):
            seed_chunk(engine, schema, f"row-{index}", f"body {index}",
                       {"id": str(index), "_elitea_run_id": "new-run"})

        _, _, truncated = adapter.read_run_staged_digests(
            wrapper, "new-run", chunk_digest, cap=2)

        assert truncated is True


class TestAdoptedRowsStayHiddenThroughout:

    def test_the_parked_and_adopted_runs_are_both_inside_the_hidden_set(self, index_schema):
        schema, engine = index_schema
        adapter, wrapper = PGVectorAdapter(), wrapper_for(schema, engine)
        seed_run(engine, schema, "old-run", "pending", heartbeat=1.0)
        seed_chunk(engine, schema, "row-1", "alpha",
                   {"id": "1", "collection": "docs", "_elitea_run_id": "old-run"})

        assert adapter.claim_adoptable_run(wrapper, "docs", stale_before=time.time()) == "old-run"
        assert "old-run" in adapter.get_pending_run_ids(wrapper, "docs")

        seed_run(engine, schema, "new-run", "pending", heartbeat=time.time())
        adapter.adopt_run_chunks(wrapper, "docs", "old-run", "new-run")

        # Both stay hidden: the live run because it is pending, the parked one because a
        # surviving worker could still be writing under it.
        hidden = adapter.get_pending_run_ids(wrapper, "docs")
        assert "new-run" in hidden
        assert "old-run" in hidden
