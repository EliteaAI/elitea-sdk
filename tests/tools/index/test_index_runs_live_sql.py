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

"""Live-SQL tests for the elitea_index_runs coordination substrate (issue #6232).

Requires a reachable Postgres (default: the local platform database). Every
test provisions its own r14_scratch_* schema and drops it afterwards; no
platform schema is ever touched. Skipped entirely when the database is
unreachable.

Covers:
  - registration arbitration through the partial unique index (sequential and
    concurrent-uncommitted), including the ON CONFLICT predicate-mismatch hazard
  - cancelled rows not blocking registration; stale rows blocking until swept
  - concurrent ensure_index_runs_table provisioning (duplicate-object recovery)
  - the three-class lock order: the single-transaction heartbeat DEADLOCKS
    against a meta-first promote while the split heartbeat does not
  - execution-options propagation: a run row registered through a PGVector
    store's engine lands in the toolkit schema
  - legacy-row visibility through the $exists/$nin read filter on a real store
  - promote/discard end-to-end against real rows
  - the sweep's reclamation of chunks stranded under an already-discarded run,
    whose candidate read is a correlated containment probe
"""

import os
import threading
import time
import uuid
from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from elitea_sdk.runtime.tools.index_runs_model import (
    RUN_STATUS_CANCELLED,
    RUN_STATUS_DISCARDED,
    RUN_STATUS_PENDING,
    RUN_STATUS_PROMOTED,
    IndexRun,
    ensure_index_runs_table,
)
from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import PGVectorAdapter

CONNECTION_STRING = os.getenv(
    "INDEX_RUNS_TEST_CONNECTION_STRING",
    "postgresql+psycopg://centry:changeme@localhost:5432/db",
)
RUN_ID_KEY = IndexerKeywords.RUN_ID.value


def _database_reachable() -> bool:
    try:
        engine = sa.create_engine(CONNECTION_STRING, pool_pre_ping=True)
        with engine.connect():
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _database_reachable(), reason="live Postgres at localhost:5432 is not reachable"
)


@pytest.fixture
def scratch_schema():
    schema = f"r14_scratch_{uuid.uuid4().hex[:10]}"
    admin_engine = sa.create_engine(CONNECTION_STRING)
    with admin_engine.begin() as connection:
        connection.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
    try:
        yield schema
    finally:
        with admin_engine.begin() as connection:
            connection.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        admin_engine.dispose()


@pytest.fixture
def scratch_engine(scratch_schema):
    engine = sa.create_engine(CONNECTION_STRING).execution_options(
        schema_translate_map={None: scratch_schema}
    )
    try:
        yield engine
    finally:
        engine.dispose()


def make_wrapper(engine, schema):
    store = SimpleNamespace(
        session_maker=SimpleNamespace(bind=engine),
        collection_name=schema,
    )
    return SimpleNamespace(vectorstore=store)


class TestRegistrationArbitration:
    def test_sequential_loser_gets_zero_rows_and_the_blocker(self, scratch_engine, scratch_schema):
        adapter = PGVectorAdapter()
        wrapper = make_wrapper(scratch_engine, scratch_schema)
        ensure_index_runs_table(scratch_engine, scratch_schema)

        registered, blocker = adapter.register_index_run(wrapper, "idx", "run-a")
        assert registered and blocker is None

        # This exercises a REAL conflict against the provisioned partial unique
        # index: an index_where predicate that did not match the DDL would raise
        # "no unique or exclusion constraint matching" here instead.
        registered, blocker = adapter.register_index_run(wrapper, "idx", "run-b")
        assert not registered
        assert blocker["run_id"] == "run-a"

        pending = adapter.get_pending_run_ids(wrapper, "idx", include_cancelled=False)
        assert pending == ["run-a"]

    def test_concurrent_uncommitted_winner_arbitrates(self, scratch_engine, scratch_schema):
        ensure_index_runs_table(scratch_engine, scratch_schema)
        table = f'"{scratch_schema}".elitea_index_runs'
        insert_sql = (
            f"INSERT INTO {table} (run_id, collection, status, started_on, heartbeat) "
            f"VALUES (:run_id, 'idx', 'pending', 1.0, 1.0) "
            f"ON CONFLICT (collection) WHERE status = 'pending' DO NOTHING"
        )
        raw_engine = sa.create_engine(CONNECTION_STRING)
        loser_rowcount = {}

        connection_a = raw_engine.connect()
        transaction_a = connection_a.begin()
        assert connection_a.execute(sa.text(insert_sql), {"run_id": "run-a"}).rowcount == 1

        def race_loser():
            with raw_engine.connect() as connection_b:
                with connection_b.begin():
                    loser_rowcount["value"] = connection_b.execute(
                        sa.text(insert_sql), {"run_id": "run-b"}
                    ).rowcount

        loser = threading.Thread(target=race_loser)
        loser.start()
        # The loser must be blocked on the winner's speculative insertion.
        time.sleep(1.0)
        assert loser.is_alive()
        transaction_a.commit()
        loser.join(timeout=15)
        connection_a.close()

        assert loser_rowcount["value"] == 0
        with raw_engine.connect() as connection:
            rows = connection.execute(
                sa.text(f"SELECT run_id FROM {table} WHERE status = 'pending'")
            ).fetchall()
        raw_engine.dispose()
        assert [row[0] for row in rows] == ["run-a"]

    def test_cancelled_row_does_not_block_registration(self, scratch_engine, scratch_schema):
        adapter = PGVectorAdapter()
        wrapper = make_wrapper(scratch_engine, scratch_schema)
        ensure_index_runs_table(scratch_engine, scratch_schema)

        adapter.register_index_run(wrapper, "idx", "run-a")
        with sa.orm.Session(scratch_engine) as session:
            session.execute(sa.text(
                f'UPDATE "{scratch_schema}".elitea_index_runs '
                "SET status = 'cancelled' WHERE run_id = 'run-a'"
            ))
            session.commit()

        registered, blocker = adapter.register_index_run(wrapper, "idx", "run-b")
        assert registered

    def test_stale_pending_row_blocks_until_the_sweep_reclaims_it(self, pgvector_wrapper, scratch_schema):
        adapter = PGVectorAdapter()
        wrapper = pgvector_wrapper
        adapter.ensure_index_runs_table(wrapper)
        scratch_engine = wrapper.vectorstore.session_maker.bind

        adapter.register_index_run(wrapper, "idx", "run-dead")
        eleven_hours_ago = time.time() - 11 * 3600
        with sa.orm.Session(scratch_engine) as session:
            session.execute(
                sa.text(f'UPDATE "{scratch_schema}".elitea_index_runs '
                        "SET heartbeat = :hb WHERE run_id = 'run-dead'"),
                {"hb": eleven_hours_ago},
            )
            session.commit()

        # The partial unique index has no staleness carve-out: the INSERT still
        # refuses against a dead pending row.
        registered, blocker = adapter.register_index_run(wrapper, "idx", "run-new")
        assert not registered
        assert blocker["run_id"] == "run-dead"

        reclaimed = adapter.sweep_stale_index_runs(wrapper, "idx", time.time() - 7200)
        assert reclaimed == ["run-dead"]
        with sa.orm.Session(scratch_engine) as session:
            status = session.execute(sa.text(
                f'SELECT status FROM "{scratch_schema}".elitea_index_runs '
                "WHERE run_id = 'run-dead'"
            )).scalar()
        assert status == RUN_STATUS_DISCARDED

        registered, blocker = adapter.register_index_run(wrapper, "idx", "run-new")
        assert registered


class TestConcurrentEnsure:
    def test_two_engines_provision_one_fresh_schema(self, scratch_schema):
        engines = [sa.create_engine(CONNECTION_STRING) for _ in range(2)]
        barrier = threading.Barrier(2)
        errors = []

        def provision(engine):
            try:
                barrier.wait(timeout=10)
                ensure_index_runs_table(engine, scratch_schema)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=provision, args=(engine,)) for engine in engines]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        for engine in engines:
            engine.dispose()

        assert errors == []
        verify_engine = sa.create_engine(CONNECTION_STRING)
        assert sa.inspect(verify_engine).has_table("elitea_index_runs", schema=scratch_schema)
        verify_engine.dispose()


class TestLockOrderDeadlocks:
    """Prove the single-transaction heartbeat deadlocks against a meta-first
    promote while the split (two-transaction) heartbeat does not."""

    @pytest.fixture
    def lock_fixture(self, scratch_engine, scratch_schema):
        ensure_index_runs_table(scratch_engine, scratch_schema)
        raw_engine = sa.create_engine(CONNECTION_STRING)
        with raw_engine.begin() as connection:
            connection.execute(sa.text(
                f'CREATE TABLE "{scratch_schema}".meta_stub (id int PRIMARY KEY)'
            ))
            connection.execute(sa.text(f'INSERT INTO "{scratch_schema}".meta_stub VALUES (1)'))
            connection.execute(sa.text(
                f'INSERT INTO "{scratch_schema}".elitea_index_runs '
                f"(run_id, collection, status, started_on, heartbeat) "
                f"VALUES ('r1', 'idx', 'pending', 1.0, 1.0)"
            ))
        yield raw_engine, scratch_schema
        raw_engine.dispose()

    def test_single_transaction_heartbeat_deadlocks(self, lock_fixture):
        raw_engine, schema = lock_fixture
        meta_locked = threading.Event()
        run_locked = threading.Event()
        outcomes = {}

        def promote_side():
            with raw_engine.connect() as connection:
                transaction = connection.begin()
                try:
                    connection.execute(sa.text(
                        f'SELECT id FROM "{schema}".meta_stub WHERE id = 1 FOR UPDATE'
                    ))
                    meta_locked.set()
                    assert run_locked.wait(timeout=10)
                    connection.execute(sa.text(
                        f'SELECT run_id FROM "{schema}".elitea_index_runs '
                        f"WHERE run_id = 'r1' FOR UPDATE"
                    ))
                    transaction.commit()
                    outcomes["promote"] = "ok"
                except Exception as exc:
                    transaction.rollback()
                    outcomes["promote"] = exc

        def single_txn_heartbeat_side():
            with raw_engine.connect() as connection:
                transaction = connection.begin()
                try:
                    connection.execute(sa.text(
                        f'UPDATE "{schema}".elitea_index_runs SET heartbeat = 2.0 '
                        f"WHERE run_id = 'r1'"
                    ))
                    run_locked.set()
                    assert meta_locked.wait(timeout=10)
                    connection.execute(sa.text(
                        f'UPDATE "{schema}".meta_stub SET id = id WHERE id = 1'
                    ))
                    transaction.commit()
                    outcomes["heartbeat"] = "ok"
                except Exception as exc:
                    transaction.rollback()
                    outcomes["heartbeat"] = exc

        threads = [
            threading.Thread(target=promote_side),
            threading.Thread(target=single_txn_heartbeat_side),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        failures = [value for value in outcomes.values() if value != "ok"]
        assert len(failures) == 1, f"expected exactly one deadlock victim, got: {outcomes}"
        assert "deadlock" in str(failures[0]).lower()

    def test_split_heartbeat_does_not_deadlock(self, lock_fixture):
        raw_engine, schema = lock_fixture
        promote_holds_both = threading.Event()
        outcomes = {}

        def promote_side():
            with raw_engine.connect() as connection:
                transaction = connection.begin()
                connection.execute(sa.text(
                    f'SELECT id FROM "{schema}".meta_stub WHERE id = 1 FOR UPDATE'
                ))
                connection.execute(sa.text(
                    f'SELECT run_id FROM "{schema}".elitea_index_runs '
                    f"WHERE run_id = 'r1' FOR UPDATE"
                ))
                promote_holds_both.set()
                time.sleep(2.0)
                transaction.commit()
                outcomes["promote"] = "ok"

        def split_heartbeat_side():
            assert promote_holds_both.wait(timeout=10)
            with raw_engine.connect() as connection:
                try:
                    # txn 1: the run-row tick — queues behind promote's run-row
                    # lock but holds NOTHING promote wants.
                    with connection.begin():
                        connection.execute(sa.text(
                            f'UPDATE "{schema}".elitea_index_runs SET heartbeat = 3.0 '
                            f"WHERE run_id = 'r1'"
                        ))
                    # txn 2: the meta bump, in its own transaction.
                    with connection.begin():
                        connection.execute(sa.text(
                            f'UPDATE "{schema}".meta_stub SET id = id WHERE id = 1'
                        ))
                    outcomes["heartbeat"] = "ok"
                except Exception as exc:
                    outcomes["heartbeat"] = exc

        threads = [
            threading.Thread(target=promote_side),
            threading.Thread(target=split_heartbeat_side),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert outcomes == {"promote": "ok", "heartbeat": "ok"}

    def test_meta_first_parties_serialize_without_deadlock(self, lock_fixture):
        raw_engine, schema = lock_fixture
        first_holds_meta = threading.Event()
        outcomes = {}

        def party(name):
            if name == "cancel":
                assert first_holds_meta.wait(timeout=10)
            with raw_engine.connect() as connection:
                try:
                    with connection.begin():
                        connection.execute(sa.text(
                            f'SELECT id FROM "{schema}".meta_stub WHERE id = 1 FOR UPDATE'
                        ))
                        if name == "promote":
                            first_holds_meta.set()
                        connection.execute(sa.text(
                            f'SELECT run_id FROM "{schema}".elitea_index_runs '
                            f"WHERE run_id = 'r1' FOR UPDATE"
                        ))
                        if name == "promote":
                            time.sleep(1.0)
                    outcomes[name] = "ok"
                except Exception as exc:
                    outcomes[name] = exc

        threads = [
            threading.Thread(target=party, args=("promote",)),
            threading.Thread(target=party, args=("cancel",)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert outcomes == {"promote": "ok", "cancel": "ok"}


@pytest.fixture
def pgvector_wrapper(scratch_schema):
    langchain_postgres = pytest.importorskip("langchain_postgres")
    from langchain_core.embeddings import DeterministicFakeEmbedding

    store = langchain_postgres.PGVector(
        embeddings=DeterministicFakeEmbedding(size=8),
        collection_name=scratch_schema,
        connection=CONNECTION_STRING,
        use_jsonb=True,
        create_extension=False,
        engine_args={
            "execution_options": {"schema_translate_map": {None: scratch_schema}},
        },
    )
    wrapper = SimpleNamespace(
        vectorstore=store,
        _log_tool_event=lambda *args, **kwargs: None,
    )
    yield wrapper


class TestExecutionOptionsPropagation:
    def test_run_row_lands_in_the_toolkit_schema(self, pgvector_wrapper, scratch_schema):
        adapter = PGVectorAdapter()
        adapter.ensure_index_runs_table(pgvector_wrapper)
        registered, blocker = adapter.register_index_run(pgvector_wrapper, "idx", "run-a")
        assert registered

        verify_engine = sa.create_engine(CONNECTION_STRING)
        with verify_engine.connect() as connection:
            rows = connection.execute(sa.text(
                f'SELECT run_id, status FROM "{scratch_schema}".elitea_index_runs'
            )).fetchall()
        verify_engine.dispose()
        assert rows == [("run-a", RUN_STATUS_PENDING)]


def add_store_documents(store, texts_with_metadata):
    texts = [text for text, _ in texts_with_metadata]
    metadatas = [metadata for _, metadata in texts_with_metadata]
    return store.add_texts(texts, metadatas=metadatas)


class TestReadFilterAndPromoteLive:
    def test_legacy_rows_stay_visible_while_pending_rows_hide(self, pgvector_wrapper, scratch_schema):
        adapter = PGVectorAdapter()
        adapter.ensure_index_runs_table(pgvector_wrapper)
        store = pgvector_wrapper.vectorstore
        adapter.register_index_run(pgvector_wrapper, "idx", "run-p")
        add_store_documents(store, [
            ("legacy document body", {"id": "legacy-1", "collection": "idx"}),
            ("pending document body", {"id": "pending-1", "collection": "idx", RUN_ID_KEY: "run-p"}),
        ])

        pending_run_ids = adapter.get_pending_run_ids(pgvector_wrapper, "idx")
        assert pending_run_ids == ["run-p"]
        search_filter = {
            "$and": [
                {"collection": {"$eq": "idx"}},
                {"$or": [
                    {"type": {"$exists": False}},
                    {"type": {"$ne": "index_meta"}},
                ]},
                {"$or": [
                    {RUN_ID_KEY: {"$exists": False}},
                    {RUN_ID_KEY: {"$nin": pending_run_ids}},
                ]},
            ]
        }

        found = store.similarity_search("document body", k=10, filter=search_filter)
        found_ids = {doc.metadata.get("id") for doc in found}
        assert "legacy-1" in found_ids
        assert "pending-1" not in found_ids

    def test_promote_deletes_superseded_and_publishes_the_run(self, pgvector_wrapper, scratch_schema):
        adapter = PGVectorAdapter()
        adapter.ensure_index_runs_table(pgvector_wrapper)
        store = pgvector_wrapper.vectorstore
        adapter.register_index_run(pgvector_wrapper, "idx", "run-p")
        add_store_documents(store, [
            ("index meta stub", {"id": "meta", "collection": "idx", "type": "index_meta"}),
            ("old generation", {"id": "doc-1", "collection": "idx"}),
        ])
        old_ids = [
            row_id for row_id in adapter.get_indexed_ids(pgvector_wrapper, "idx")
        ]
        add_store_documents(store, [
            ("new generation", {"id": "doc-1", "collection": "idx", RUN_ID_KEY: "run-p"}),
        ])

        outcome = adapter.promote_run(pgvector_wrapper, "idx", "run-p", old_ids, [], [])
        assert outcome == "promoted"

        remaining = store.similarity_search(
            "generation", k=10,
            filter={"$and": [
                {"collection": {"$eq": "idx"}},
                {"$or": [
                    {"type": {"$exists": False}},
                    {"type": {"$ne": "index_meta"}},
                ]},
            ]},
        )
        assert [doc.page_content for doc in remaining] == ["new generation"]
        assert adapter.get_pending_run_ids(pgvector_wrapper, "idx") == []
        with sa.orm.Session(store.session_maker.bind) as session:
            status = session.execute(sa.text(
                f'SELECT status FROM "{scratch_schema}".elitea_index_runs ' "WHERE run_id = 'run-p'"
            )).scalar()
        assert status == RUN_STATUS_PROMOTED

    def test_promote_aborts_to_discard_on_a_cancelled_run(self, pgvector_wrapper, scratch_schema):
        adapter = PGVectorAdapter()
        adapter.ensure_index_runs_table(pgvector_wrapper)
        store = pgvector_wrapper.vectorstore
        adapter.register_index_run(pgvector_wrapper, "idx", "run-p")
        add_store_documents(store, [
            ("index meta stub", {"id": "meta", "collection": "idx", "type": "index_meta"}),
            ("retained generation", {"id": "doc-1", "collection": "idx"}),
        ])
        retained_ids = adapter.get_indexed_ids(pgvector_wrapper, "idx")
        add_store_documents(store, [
            ("cancelled staged row", {"id": "doc-1", "collection": "idx", RUN_ID_KEY: "run-p"}),
        ])
        with sa.orm.Session(store.session_maker.bind) as session:
            session.execute(sa.text(
                f'UPDATE "{scratch_schema}".elitea_index_runs ' "SET status = 'cancelled' WHERE run_id = 'run-p'"
            ))
            session.commit()

        outcome = adapter.promote_run(pgvector_wrapper, "idx", "run-p", retained_ids, [], [])
        assert outcome == "aborted-cancelled"

        surviving_ids = adapter.get_indexed_ids(pgvector_wrapper, "idx")
        assert sorted(surviving_ids) == sorted(retained_ids)
        with sa.orm.Session(store.session_maker.bind) as session:
            status = session.execute(sa.text(
                f'SELECT status FROM "{scratch_schema}".elitea_index_runs ' "WHERE run_id = 'run-p'"
            )).scalar()
        assert status == RUN_STATUS_DISCARDED

    def test_discard_deletes_only_the_runs_rows(self, pgvector_wrapper, scratch_schema):
        adapter = PGVectorAdapter()
        adapter.ensure_index_runs_table(pgvector_wrapper)
        store = pgvector_wrapper.vectorstore
        adapter.register_index_run(pgvector_wrapper, "idx", "run-p")
        add_store_documents(store, [
            ("retained generation", {"id": "doc-1", "collection": "idx"}),
            ("staged row", {"id": "doc-2", "collection": "idx", RUN_ID_KEY: "run-p"}),
            ("multi index staged row", {"id": "doc-3", "collection": "idx;other", RUN_ID_KEY: "run-p"}),
        ])

        outcome = adapter.discard_run(pgvector_wrapper, "idx", "run-p")
        assert outcome == "discarded"

        with sa.orm.Session(store.session_maker.bind) as session:
            remaining = session.execute(sa.text(
                f'SELECT cmetadata->>\'id\' FROM "{scratch_schema}".langchain_pg_embedding ORDER BY 1'
            )).fetchall()
        # The run-scoped delete has no collection conjunct: the ";"-multi-index
        # staged row must fall with the run instead of becoming visible garbage.
        assert [row[0] for row in remaining] == ["doc-1"]

    def test_heartbeat_after_promote_is_a_zero_row_noop(self, pgvector_wrapper, scratch_schema):
        adapter = PGVectorAdapter()
        adapter.ensure_index_runs_table(pgvector_wrapper)
        store = pgvector_wrapper.vectorstore
        adapter.register_index_run(pgvector_wrapper, "idx", "run-p")
        add_store_documents(store, [
            ("index meta stub", {"id": "meta", "collection": "idx", "type": "index_meta",
                                 "updated_on": 111.0}),
        ])
        outcome = adapter.promote_run(pgvector_wrapper, "idx", "run-p", [], [], [])
        assert outcome == "promoted"
        with sa.orm.Session(store.session_maker.bind) as session:
            meta_id = session.execute(sa.text(
                f'SELECT id FROM "{scratch_schema}".langchain_pg_embedding ' "WHERE cmetadata->>'type' = 'index_meta'"
            )).scalar()
            heartbeat_before = session.execute(sa.text(
                f'SELECT heartbeat FROM "{scratch_schema}".elitea_index_runs ' "WHERE run_id = 'run-p'"
            )).scalar()

        adapter.heartbeat_index_run(pgvector_wrapper, "idx", "run-p", meta_id)

        with sa.orm.Session(store.session_maker.bind) as session:
            updated_on = session.execute(sa.text(
                f'SELECT cmetadata->>\'updated_on\' FROM "{scratch_schema}".langchain_pg_embedding '
                "WHERE cmetadata->>'type' = 'index_meta'"
            )).scalar()
            heartbeat_after = session.execute(sa.text(
                f'SELECT heartbeat FROM "{scratch_schema}".elitea_index_runs ' "WHERE run_id = 'run-p'"
            )).scalar()
        # Neither the meta bump (guarded by EXISTS pending) nor the run-row tick
        # (guarded by status) may touch anything after the promote flip.
        assert updated_on == "111.0"
        assert heartbeat_after == heartbeat_before

    def test_heartbeat_bumps_updated_on_while_pending(self, pgvector_wrapper, scratch_schema):
        adapter = PGVectorAdapter()
        adapter.ensure_index_runs_table(pgvector_wrapper)
        store = pgvector_wrapper.vectorstore
        adapter.register_index_run(pgvector_wrapper, "idx", "run-p")
        add_store_documents(store, [
            ("index meta stub", {"id": "meta", "collection": "idx", "type": "index_meta",
                                 "updated_on": 111.0}),
        ])
        with sa.orm.Session(store.session_maker.bind) as session:
            meta_id = session.execute(sa.text(
                f'SELECT id FROM "{scratch_schema}".langchain_pg_embedding ' "WHERE cmetadata->>'type' = 'index_meta'"
            )).scalar()

        adapter.heartbeat_index_run(pgvector_wrapper, "idx", "run-p", meta_id)

        with sa.orm.Session(store.session_maker.bind) as session:
            updated_on = session.execute(sa.text(
                f'SELECT (cmetadata->>\'updated_on\')::float8 FROM "{scratch_schema}".langchain_pg_embedding '
                "WHERE cmetadata->>'type' = 'index_meta'"
            )).scalar()
        assert updated_on > time.time() - 60


def stage_a_stranded_run(adapter, wrapper, scratch_schema):
    """The state a failed best-effort cleanup leaves: a terminal run row over
    surviving chunks, alongside a discarded run whose cleanup did succeed."""
    adapter.ensure_index_runs_table(wrapper)
    store = wrapper.vectorstore
    adapter.register_index_run(wrapper, "idx", "run-strand")
    add_store_documents(store, [
        ("index meta stub", {"id": "meta", "collection": "idx", "type": "index_meta"}),
        ("live generation", {"id": "doc-1", "collection": "idx"}),
        ("stranded row", {"id": "doc-2", "collection": "idx", RUN_ID_KEY: "run-strand"}),
        ("stranded multi index row", {"id": "doc-3", "collection": "idx;other",
                                      RUN_ID_KEY: "run-strand"}),
    ])
    with sa.orm.Session(store.session_maker.bind) as session:
        # The strand's heartbeat froze when the sweep flipped it, so it is OLDER
        # than that of every run discarded afterwards.
        session.execute(
            sa.text(f'UPDATE "{scratch_schema}".elitea_index_runs '
                    "SET status = :status, heartbeat = :heartbeat WHERE run_id = 'run-strand'"),
            {"status": RUN_STATUS_DISCARDED, "heartbeat": time.time() - 11 * 3600},
        )
        session.execute(
            sa.text(f'INSERT INTO "{scratch_schema}".elitea_index_runs '
                    "(run_id, collection, status, started_on, heartbeat) "
                    "VALUES ('run-clean', 'idx', :status, :now, :now)"),
            {"status": RUN_STATUS_DISCARDED, "now": time.time()},
        )
        session.commit()
    return store


class TestStrandedChunkReclaimLive:
    """The reclamation pass picks its candidates with a correlated containment
    probe built by jsonb_build_object. Only a real server proves it binds, that it
    discriminates chunk-bearing rows from clean ones, and that it never nominates
    the index_meta row."""

    def test_the_candidate_read_selects_only_runs_whose_chunks_survived(
        self, pgvector_wrapper, scratch_schema
    ):
        adapter = PGVectorAdapter()
        store = stage_a_stranded_run(adapter, pgvector_wrapper, scratch_schema)

        with sa.orm.Session(store.session_maker.bind) as session:
            candidates = session.query(IndexRun.run_id).filter(
                IndexRun.collection == "idx",
                IndexRun.status == RUN_STATUS_DISCARDED,
                adapter._run_chunks_exist_clause(store, IndexRun.run_id),
            ).all()

        # 'run-clean' is the newer row: a recency-ordered window would reclaim it
        # and starve the strand, which is the only row with anything to reclaim.
        assert [row[0] for row in candidates] == ["run-strand"]

    def test_the_sweep_deletes_the_strand_and_leaves_the_live_corpus(
        self, pgvector_wrapper, scratch_schema
    ):
        adapter = PGVectorAdapter()
        store = stage_a_stranded_run(adapter, pgvector_wrapper, scratch_schema)

        assert adapter.sweep_stale_index_runs(pgvector_wrapper, "idx", time.time() - 7200) == []

        with sa.orm.Session(store.session_maker.bind) as session:
            remaining = session.execute(sa.text(
                f'SELECT cmetadata->>\'id\' FROM "{scratch_schema}".langchain_pg_embedding ORDER BY 1'
            )).fetchall()
            statuses = session.execute(sa.text(
                f'SELECT run_id, status FROM "{scratch_schema}".elitea_index_runs ORDER BY 1'
            )).fetchall()
        # The ";"-multi-index staged row falls with the run; the meta row and the
        # previous generation do not. Run rows are never deleted, only flipped.
        assert [row[0] for row in remaining] == ["doc-1", "meta"]
        assert statuses == [
            ("run-clean", RUN_STATUS_DISCARDED),
            ("run-strand", RUN_STATUS_DISCARDED),
        ]

    def test_a_reclaimed_run_is_not_a_candidate_for_the_next_sweep(
        self, pgvector_wrapper, scratch_schema
    ):
        adapter = PGVectorAdapter()
        store = stage_a_stranded_run(adapter, pgvector_wrapper, scratch_schema)

        adapter.sweep_stale_index_runs(pgvector_wrapper, "idx", time.time() - 7200)

        with sa.orm.Session(store.session_maker.bind) as session:
            candidates = session.query(IndexRun.run_id).filter(
                IndexRun.collection == "idx",
                IndexRun.status == RUN_STATUS_DISCARDED,
                adapter._run_chunks_exist_clause(store, IndexRun.run_id),
            ).all()

        # The candidate predicate is the delete's own, so every sweep makes
        # progress and the cap can never hold a reclaimed run in the window.
        assert candidates == []
