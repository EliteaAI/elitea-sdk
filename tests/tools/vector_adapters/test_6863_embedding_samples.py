from unittest.mock import MagicMock

import numpy as np
from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

from tests.tools.vector_adapters.test_5261_adoption import (  # noqa: F401  (fixtures)
    FakeQuery,
    FakeSession,
    adapter_and_wrapper,
    compiled,
)


def _embedding_store():
    base = declarative_base()

    class EmbeddingStore(base):
        __tablename__ = "langchain_pg_embedding"
        id = Column(String, primary_key=True)
        document = Column(Text)
        embedding = Column(String)
        cmetadata = Column(JSONB)

    return EmbeddingStore


class LimitRecordingSession(FakeSession):
    def __init__(self):
        super().__init__()
        self.limits = []

    def query(self, *_entities):
        session = self

        class Recording(FakeQuery):
            def limit(self, count):
                session.limits.append(count)
                return self

        return Recording(self, self.rows)


class TestReadRunEmbeddingSamples:

    def test_pgvector_arrays_come_back_as_plain_floats(self, adapter_and_wrapper):
        adapter, wrapper, install = adapter_and_wrapper
        wrapper.vectorstore.EmbeddingStore = _embedding_store()
        install(FakeSession(rows=[("text", np.array([0.5, 0.25], dtype=np.float32))]))

        samples = adapter.read_run_embedding_samples(wrapper, "run-1", 3)

        assert samples == [("text", [0.5, 0.25])]
        assert all(type(component) is float for component in samples[0][1])

    def test_rows_without_text_or_vector_are_not_samples(self, adapter_and_wrapper):
        adapter, wrapper, install = adapter_and_wrapper
        wrapper.vectorstore.EmbeddingStore = _embedding_store()
        install(FakeSession(rows=[("", [1.0]), ("text", None), ("kept", [1.0])]))

        assert adapter.read_run_embedding_samples(wrapper, "run-1", 3) == [("kept", [1.0])]

    def test_only_this_runs_chunk_rows_are_read(self, adapter_and_wrapper):
        adapter, wrapper, install = adapter_and_wrapper
        wrapper.vectorstore.EmbeddingStore = _embedding_store()
        session = install(FakeSession())

        adapter.read_run_embedding_samples(wrapper, "run-1", 3)

        rendered = " ".join(compiled(criterion) for criterion in session.filters)
        assert "'_elitea_run_id': 'run-1'" in rendered
        assert "index_meta" in rendered

    def test_the_read_is_bounded_by_the_caller(self, adapter_and_wrapper):
        adapter, wrapper, install = adapter_and_wrapper
        wrapper.vectorstore.EmbeddingStore = _embedding_store()
        session = install(LimitRecordingSession())

        adapter.read_run_embedding_samples(wrapper, "run-1", 3)

        assert session.limits == [3]


def test_an_adapter_without_staging_yields_no_samples():
    from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import ChromaAdapter

    assert ChromaAdapter().read_run_embedding_samples(MagicMock(), "run-1", 3) == []
