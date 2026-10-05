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

import json
import uuid

import pytest
import sqlalchemy as sa

from elitea_sdk.tools.base_indexer_toolkit import vectors_agree
from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import PGVectorAdapter
from tests.tools.index.test_5261_adoption_live_sql import CONNECTION_STRING, _base_engine


@pytest.fixture
def vector_schema():
    base = _base_engine()
    if base is None:
        pytest.skip("live Postgres at localhost:5432 is not reachable")
    schema = f"probe6863_{uuid.uuid4().hex[:10]}"
    with base.begin() as connection:
        if connection.execute(sa.text(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar() is None:
            pytest.skip("the vector extension is not installed")
        connection.execute(sa.text(f'CREATE SCHEMA "{schema}"'))
        connection.execute(sa.text(
            f'CREATE TABLE "{schema}".langchain_pg_embedding '
            f'(id text primary key, document text, embedding vector, cmetadata jsonb)'
        ))
    engine = sa.create_engine(CONNECTION_STRING)
    try:
        yield schema, engine
    finally:
        engine.dispose()
        with base.begin() as connection:
            connection.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


def wrapper_for(schema, engine):
    from pgvector.sqlalchemy import Vector
    from sqlalchemy import Column, String, Text
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.orm import declarative_base

    base = declarative_base()

    class EmbeddingStore(base):
        __tablename__ = "langchain_pg_embedding"
        __table_args__ = {"schema": schema}
        id = Column(String, primary_key=True)
        document = Column(Text)
        embedding = Column(Vector())
        cmetadata = Column(JSONB)

    class SessionMaker:
        bind = engine

    class Store:
        session_maker = SessionMaker()

    Store.EmbeddingStore = EmbeddingStore

    class Wrapper:
        vectorstore = Store()

    return Wrapper()


def seed(engine, schema, row_id, document, vector, metadata):
    with engine.begin() as connection:
        connection.execute(sa.text(
            f'INSERT INTO "{schema}".langchain_pg_embedding (id, document, embedding, cmetadata) '
            f'VALUES (:i, :d, CAST(:v AS vector), CAST(:m AS jsonb))'
        ), {"i": row_id, "d": document, "v": json.dumps(vector), "m": json.dumps(metadata)})


class TestTheProbeReadsTheStoredVectors:

    def test_a_stored_vector_round_trips_close_enough_to_agree(self, vector_schema):
        schema, engine = vector_schema
        written = [0.123456789, -0.987654321, 0.5]
        seed(engine, schema, "a", "chunk", written, {"_elitea_run_id": "run-1"})

        samples = PGVectorAdapter().read_run_embedding_samples(
            wrapper_for(schema, engine), "run-1", 3)

        assert [text for text, _ in samples] == ["chunk"]
        assert vectors_agree(samples[0][1], written, 1e-6)

    def test_only_this_runs_chunk_rows_are_sampled(self, vector_schema):
        schema, engine = vector_schema
        seed(engine, schema, "mine", "mine", [1.0, 0.0], {"_elitea_run_id": "run-1"})
        seed(engine, schema, "other", "other", [0.0, 1.0], {"_elitea_run_id": "run-2"})
        seed(engine, schema, "meta", "index_meta_docs", [0.0, 1.0],
             {"_elitea_run_id": "run-1", "type": "index_meta"})

        samples = PGVectorAdapter().read_run_embedding_samples(
            wrapper_for(schema, engine), "run-1", 3)

        assert [text for text, _ in samples] == ["mine"]

    def test_the_sample_is_bounded(self, vector_schema):
        schema, engine = vector_schema
        for index in range(5):
            seed(engine, schema, f"r{index}", f"chunk {index}", [1.0, 0.0],
                 {"_elitea_run_id": "run-1"})

        samples = PGVectorAdapter().read_run_embedding_samples(
            wrapper_for(schema, engine), "run-1", 3)

        assert len(samples) == 3
