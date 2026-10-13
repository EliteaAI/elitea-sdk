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

"""A failed index listing must not read as an empty one (#7018)."""

from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from langchain_core.tools import ToolException
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import declarative_base

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit
from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import PGVectorAdapter

_Base = declarative_base()

QUERY_FAILURE = 'relation "42.langchain_pg_embedding" does not exist'


class _EmbeddingStore(_Base):
    __tablename__ = "langchain_pg_embedding"

    id = sa.Column(sa.String, primary_key=True)
    cmetadata = sa.Column(postgresql.JSONB)


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *criteria):
        return self

    def all(self):
        if isinstance(self._rows, Exception):
            raise self._rows
        return self._rows


def _session_returning(rows):
    class _Session:
        def __init__(self, bind=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def query(self, *columns):
            return _Query(rows)

    return _Session


class _Toolkit(BaseIndexerToolkit):
    pass


@pytest.fixture
def toolkit_with_rows(monkeypatch):
    def build(rows):
        monkeypatch.setattr(sa.orm, "Session", _session_returning(rows))
        instance = _Toolkit.model_construct()
        object.__setattr__(instance, "vector_adapter", PGVectorAdapter())
        object.__setattr__(instance, "vectorstore", SimpleNamespace(
            session_maker=SimpleNamespace(bind=None),
            EmbeddingStore=_EmbeddingStore,
        ))
        return instance

    return build


def test_list_indexes_raises_when_the_index_query_fails(toolkit_with_rows):
    driver_error = Exception(f"{QUERY_FAILURE}\nLINE 1: SELECT distinct(jsonb_extract_path_text(...")
    toolkit = toolkit_with_rows(sa.exc.ProgrammingError("SELECT secret_column FROM t", {}, driver_error))

    with pytest.raises(ToolException) as raised:
        toolkit.list_indexes()

    assert str(raised.value) == f"Failed to list indexes: {QUERY_FAILURE}"


def test_list_indexes_names_the_error_type_when_the_driver_gives_no_message(toolkit_with_rows):
    toolkit = toolkit_with_rows(sa.exc.OperationalError("SELECT", {}, TimeoutError()))

    with pytest.raises(ToolException) as raised:
        toolkit.list_indexes()

    assert str(raised.value) == "Failed to list indexes: TimeoutError"


def test_list_indexes_reports_empty_only_when_the_query_finds_no_rows(toolkit_with_rows):
    assert toolkit_with_rows([]).list_indexes() == "No indexed collections"


def test_list_indexes_returns_the_stored_index_names(toolkit_with_rows):
    toolkit = toolkit_with_rows([("docs",), (None,), ("wiki",)])

    assert toolkit.list_indexes() == ["docs", "wiki"]


def test_remove_index_reports_success_when_listing_fails_after_the_committed_delete(
        toolkit_with_rows, monkeypatch):
    toolkit = toolkit_with_rows(sa.exc.OperationalError("SELECT", {}, Exception(QUERY_FAILURE)))
    removed_events = []
    monkeypatch.setattr(VectorStoreWrapperBase, "_clean_collection",
                        lambda self, index_name="", including_index_meta=False: 3)
    monkeypatch.setattr(_Toolkit, "_emit_index_data_removed_event",
                        lambda self, index_name: removed_events.append(index_name))

    result = toolkit.remove_index("docs")

    assert result.startswith("Index 'docs' has been removed from the vector store.")
    assert "Could not list the available indexes: Failed to list indexes" in result
    assert removed_events == ["docs"]


def test_remove_index_still_reports_not_found_when_listing_fails(toolkit_with_rows, monkeypatch):
    toolkit = toolkit_with_rows(sa.exc.OperationalError("SELECT", {}, Exception(QUERY_FAILURE)))
    monkeypatch.setattr(VectorStoreWrapperBase, "_clean_collection",
                        lambda self, index_name="", including_index_meta=False: 0)

    with pytest.raises(ToolException) as raised:
        toolkit.remove_index("typo")

    assert str(raised.value).startswith("Index 'typo' not found.")
    assert "Could not list the available indexes: Failed to list indexes" in str(raised.value)


def test_lazy_tools_still_treat_a_failed_listing_as_no_collections(toolkit_with_rows):
    toolkit = toolkit_with_rows(sa.exc.OperationalError("SELECT", {}, Exception(QUERY_FAILURE)))

    assert toolkit._has_collections() is False
