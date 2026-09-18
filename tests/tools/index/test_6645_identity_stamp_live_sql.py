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

"""Executes the identity back-fill against a real Postgres (#6645).

Every other test for stamp_code_identity stops at a compiled statement. The
back-fill fails softly by design, so a statement that never runs would leave the
unchanged-file skip permanently disarmed while every run still reports success.
Requires a reachable Postgres; skipped otherwise, and CI skips this directory.
"""

import functools
import os
import uuid

import pytest
import sqlalchemy as sa

from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import (
    IDENTITY_STAMP_BATCH_SIZE,
    PGVectorAdapter,
)

CONNECTION_STRING = os.getenv(
    "INDEX_RUNS_TEST_CONNECTION_STRING",
    "postgresql+psycopg://centry:changeme@localhost:5432/db",
)


@functools.lru_cache(maxsize=1)
def _reachable_engine():
    try:
        engine = sa.create_engine(
            CONNECTION_STRING, pool_pre_ping=True, connect_args={"connect_timeout": 2})
        with engine.connect():
            return engine
    except Exception:
        return None


@pytest.fixture
def embedding_table():
    engine = _reachable_engine()
    if engine is None:
        pytest.skip("live Postgres at localhost:5432 is not reachable")
    schema = f"stamp6645_{uuid.uuid4().hex[:10]}"
    with engine.begin() as connection:
        connection.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
        connection.execute(sa.text(
            f'CREATE TABLE "{schema}".langchain_pg_embedding '
            f'(id text primary key, cmetadata jsonb)'
        ))
    try:
        yield schema, engine
    finally:
        with engine.begin() as connection:
            connection.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


def _wrapper_for(schema, engine):
    from sqlalchemy import Column, String
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.orm import declarative_base

    base = declarative_base()

    class EmbeddingStore(base):
        __tablename__ = "langchain_pg_embedding"
        __table_args__ = {"schema": schema}
        id = Column(String, primary_key=True)
        cmetadata = Column(JSONB)

    class SessionMaker:
        bind = engine

    class Store:
        session_maker = SessionMaker()

    Store.EmbeddingStore = EmbeddingStore

    class Wrapper:
        vectorstore = Store()

    return Wrapper()


def _rows(engine, schema):
    with engine.connect() as connection:
        return dict(connection.execute(sa.text(
            f"SELECT id, cmetadata->>'blob_sha' FROM \"{schema}\".langchain_pg_embedding"
        )).all())


def _seed(engine, schema, rows):
    with engine.begin() as connection:
        for row_id, metadata in rows:
            connection.execute(
                sa.text(f'INSERT INTO "{schema}".langchain_pg_embedding VALUES (:id, CAST(:m AS jsonb))'),
                {"id": row_id, "m": metadata},
            )


class TestTheStampReachesTheRows:

    def test_a_row_gains_the_identity_and_keeps_its_other_metadata(self, embedding_table):
        schema, engine = embedding_table
        _seed(engine, schema, [("row-1", '{"filename": "a.py", "commit_hash": "h1"}')])

        stamped = PGVectorAdapter().stamp_code_identity(
            _wrapper_for(schema, engine), {"row-1": "sha-aaa"})

        assert stamped == 1
        assert _rows(engine, schema) == {"row-1": "sha-aaa"}
        with engine.connect() as connection:
            assert connection.execute(sa.text(
                f"SELECT cmetadata->>'commit_hash' FROM \"{schema}\".langchain_pg_embedding"
            )).scalar() == "h1"

    def test_rows_take_their_own_identity(self, embedding_table):
        schema, engine = embedding_table
        _seed(engine, schema, [("row-1", '{"filename": "a.py"}'),
                               ("row-2", '{"filename": "b.py"}')])

        PGVectorAdapter().stamp_code_identity(
            _wrapper_for(schema, engine), {"row-1": "sha-aaa", "row-2": "sha-bbb"})

        assert _rows(engine, schema) == {"row-1": "sha-aaa", "row-2": "sha-bbb"}

    def test_a_row_with_no_metadata_at_all_is_stamped(self, embedding_table):
        schema, engine = embedding_table
        _seed(engine, schema, [("row-1", None)])

        PGVectorAdapter().stamp_code_identity(
            _wrapper_for(schema, engine), {"row-1": "sha-aaa"})

        assert _rows(engine, schema) == {"row-1": "sha-aaa"}

    def test_an_unlisted_row_is_left_alone(self, embedding_table):
        schema, engine = embedding_table
        _seed(engine, schema, [("row-1", '{"filename": "a.py"}'),
                               ("row-2", '{"filename": "b.py"}')])

        PGVectorAdapter().stamp_code_identity(
            _wrapper_for(schema, engine), {"row-1": "sha-aaa"})

        assert _rows(engine, schema) == {"row-1": "sha-aaa", "row-2": None}

    def test_more_rows_than_one_batch_all_land(self, embedding_table):
        schema, engine = embedding_table
        row_count = IDENTITY_STAMP_BATCH_SIZE + 3
        _seed(engine, schema, [(f"row-{n}", '{"filename": "a.py"}') for n in range(row_count)])

        stamped = PGVectorAdapter().stamp_code_identity(
            _wrapper_for(schema, engine), {f"row-{n}": f"sha-{n}" for n in range(row_count)})

        assert stamped == row_count
        assert _rows(engine, schema) == {f"row-{n}": f"sha-{n}" for n in range(row_count)}
