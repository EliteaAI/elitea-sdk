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

"""Index, change the source, check, then Clean index: no chunk of a removed or edited item is left.

Everything runs through the real `index_data`, `check_for_changes` and `search_index` of a toolkit.
Only the vector store is faked (an in-memory table that keeps the rows and answers filtered searches),
so what the run writes and deletes is what the check and the search later read.
Two adapters run the same toolkit: one without run staging (Clean index empties the collection up front)
and one with run staging, where a removed item's chunks go only when the promote step deletes them.
The staged adapter is a fake of the pgvector contract: its `promote_run` deletes the ids it is handed,
so the test proves which ids the toolkit hands over, not the SQL of the real promote.
Lives outside tests/tools/index/ because CI skips that directory.
"""

import pytest
from langchain_core.documents import Document

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools import index_change_check as check
from elitea_sdk.tools.chunkers import __all__ as chunkers
from elitea_sdk.tools.non_code_indexer_toolkit import NonCodeIndexerToolkit
from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import VectorStoreAdapter

INDEX_NAME = "docs"
META_TYPE = IndexerKeywords.INDEX_META_TYPE.value


def matches(condition, metadata):
    """The part of the pgvector filter language the search path of the toolkit sends."""
    for operator, operand in condition.items():
        if operator == "$and":
            if not all(matches(part, metadata) for part in operand):
                return False
        elif operator == "$or":
            if not any(matches(part, metadata) for part in operand):
                return False
        else:
            present = operator in metadata
            for comparison, value in operand.items():
                if comparison == "$eq" and metadata.get(operator) != value:
                    return False
                if comparison == "$ne" and metadata.get(operator) == value:
                    return False
                if comparison == "$exists" and present != value:
                    return False
                if comparison == "$nin" and metadata.get(operator) in value:
                    return False
    return True


class FakeVectorStore:
    """An in-memory embedding table: one row per chunk plus one index_meta row per index."""

    def __init__(self):
        self.rows = {}
        self.counter = 0

    def add_texts(self, texts, metadatas=None, ids=None):
        row_ids = []
        for position, text in enumerate(texts):
            row_id = ids[position] if ids and ids[position] else f"row-{self.counter}"
            self.counter += 1
            self.rows[row_id] = {"content": text, "metadata": dict(metadatas[position])}
            row_ids.append(row_id)
        return row_ids

    def similarity_search_with_score(self, query, filter=None, k=10):
        found = [
            (Document(page_content=row["content"], metadata=dict(row["metadata"])), 0.1)
            for row in self.rows.values()
            if query.lower() in row["content"].lower() and matches(filter or {}, row["metadata"])
        ]
        return found[:k]

    def chunks_of(self, source_id):
        return [row for row in self.rows.values() if row["metadata"].get("id") == source_id]

    def data_rows(self):
        return [row for row in self.rows.values() if row["metadata"].get("type") != META_TYPE]


class FakeAdapter(VectorStoreAdapter):
    """The adapter calls the toolkit makes; no run staging, so Clean index empties the collection up front."""

    supports_run_staging = False

    def get_vectorstore_params(self, collection_name, connection_string=None):
        return {}

    def list_collections(self, wrapper):
        return sorted({row["metadata"]["collection"] for row in wrapper.vectorstore.rows.values()})

    def remove_collection(self, wrapper, collection_name):
        self.clean_collection(wrapper, collection_name, True)

    def get_indexed_ids(self, wrapper, index_name=""):
        return list(wrapper.vectorstore.rows)

    def clean_collection(self, wrapper, index_name="", including_index_meta=False):
        rows = wrapper.vectorstore.rows
        doomed = [row_id for row_id, row in rows.items()
                  if row["metadata"].get("collection") == index_name
                  and (including_index_meta or row["metadata"].get("type") != META_TYPE)]
        for row_id in doomed:
            del rows[row_id]
        return len(doomed)

    def get_indexed_data(self, wrapper, index_name):
        result = {}
        for row_id, row in wrapper.vectorstore.rows.items():
            meta = row["metadata"]
            if meta.get("collection") != index_name or meta.get("type") == META_TYPE:
                continue
            doc_id = str(meta.get("id", row_id))
            if doc_id in result and meta.get("chunk_id"):
                result[doc_id]["all_chunks"].append(row_id)
            else:
                result[doc_id] = {"metadata": meta, "id": row_id, "all_chunks": [row_id],
                                  IndexerKeywords.DEPENDENT_DOCS.value: [],
                                  IndexerKeywords.PARENT.value: -1,
                                  "pk_fallback": "id" not in meta}
        return result

    def get_code_indexed_data(self, wrapper, index_name):
        return {}

    def add_to_collection(self, wrapper, entry_id, new_collection_value):
        return None

    def get_index_meta(self, wrapper, index_name):
        return [{"id": row_id, "content": row["content"], "metadata": row["metadata"]}
                for row_id, row in wrapper.vectorstore.rows.items()
                if row["metadata"].get("collection") == index_name and row["metadata"].get("type") == META_TYPE]

    def promote_run(self, *args, **kwargs):
        raise AssertionError("a run without staging never promotes")

    def discard_run(self, *args, **kwargs):
        raise AssertionError("a run without staging never discards")


class StagedAdapter(FakeAdapter):
    """The run-staging contract of the pgvector adapter: promote deletes what the toolkit hands over."""

    supports_run_staging = True

    def __init__(self):
        self.promotes = []

    def ensure_index_runs_table(self, wrapper):
        return None

    def register_index_run(self, wrapper, index_name, run_id, task_id=None, meta_lock_id=None):
        return (True, None)

    def sweep_stale_index_runs(self, wrapper, index_name, stale_before, except_run_id=None):
        return []

    def heartbeat_index_run(self, wrapper, index_name, run_id, meta_id, chunks_written=None):
        return None

    def update_index_meta_keys(self, wrapper, meta_id, run_id, patch):
        row = wrapper.vectorstore.rows.get(meta_id)
        if row is None:
            return 0
        row["metadata"].update(patch)
        return 1

    def promote_run(self, wrapper, index_name, run_id, superseded_ids, orphan_ids, damaged_ids):
        rows = wrapper.vectorstore.rows
        for row_id in dict.fromkeys([*superseded_ids, *orphan_ids, *damaged_ids]):
            rows.pop(row_id, None)
        self.promotes.append({"superseded": list(superseded_ids), "orphans": list(orphan_ids)})
        return "promoted"

    def discard_run(self, *args, **kwargs):
        raise AssertionError("a run that completes never discards")


class Source:
    """The mutable source the toolkit lists and loads."""

    def __init__(self):
        self.items = {}

    def put(self, key, text, modified):
        self.items[key] = {"text": text, "modified": modified}

    def remove(self, key):
        del self.items[key]


class SourceToolkit(NonCodeIndexerToolkit):
    """A listing toolkit over `Source`: the listing and the loader read the same items."""

    def _list_source_items(self, index_configuration):
        return check.SourceListing(items=[
            check.ListedItem(key=key, name=f"{key}.txt", path=f"/lib/{key}.txt", marker=item["modified"])
            for key, item in self.source.items.items()
        ])

    def _base_loader(self, **kwargs):
        for key, item in self.source.items.items():
            yield Document(page_content=item["text"],
                           metadata={"id": key, "name": f"{key}.txt", "path": f"/lib/{key}.txt",
                                     "updated_on": item["modified"]})


class AttestingSourceToolkit(SourceToolkit):
    """A listing toolkit whose loader confirms it enumerated the whole source, as SharePoint does."""

    loader_attests_completion = True

    def _base_loader(self, **kwargs):
        yield from super()._base_loader(**kwargs)
        self._attest_loader_completion()


def split_by_line(file_content_generator, config, *args, **kwargs):
    """A chunker with no tokenizer download: one chunk per line, so every item holds several chunks."""
    for document in file_content_generator:
        for number, line in enumerate(document.page_content.splitlines(), 1):
            metadata = dict(document.metadata)
            metadata.update({"chunk_id": number, "chunk_type": "document"})
            yield Document(page_content=line, metadata=metadata)


def build_world(monkeypatch, toolkit_class, adapter):
    source = Source()
    toolkit = toolkit_class.model_construct()
    store = FakeVectorStore()
    for name, value in (("vectorstore", store), ("vector_adapter", adapter), ("source", source),
                        ("toolkit_id", None), ("max_docs_per_add", 100), ("embeddings", None), ("llm", None)):
        object.__setattr__(toolkit, name, value)
    monkeypatch.setitem(chunkers, "by_line", split_by_line)
    monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_indexed_count",
                        lambda self, name: len(self.vectorstore.data_rows()))
    monkeypatch.setattr(SourceToolkit, "_is_scheduled_run", lambda self: False)
    monkeypatch.setattr(SourceToolkit, "_log_tool_event", lambda self, *a, **kw: None)
    monkeypatch.setattr(SourceToolkit, "_emit_index_event", lambda self, name, error=None, state=None: None)
    source.put("handbook", "handbook alpha rule\nhandbook beta rule\nhandbook gamma rule", "2026-01-01")
    source.put("budget", "budget plan one\nbudget plan two\nbudget plan three", "2026-01-01")
    source.put("roadmap", "roadmap item one\nroadmap item two", "2026-01-01")
    source.put("faq", "faq answer one\nfaq answer two", "2026-01-01")
    return toolkit, source, store


@pytest.fixture
def world(monkeypatch):
    return build_world(monkeypatch, SourceToolkit, FakeAdapter())


@pytest.fixture
def staged_world(monkeypatch):
    """Run staging on, and a loader that cannot confirm source deletions (the default)."""
    return build_world(monkeypatch, SourceToolkit, StagedAdapter())


@pytest.fixture
def attested_staged_world(monkeypatch):
    return build_world(monkeypatch, AttestingSourceToolkit, StagedAdapter())


def index(toolkit, clean_index=False):
    return toolkit.index_data(index_name=INDEX_NAME, chunking_tool="by_line", clean_index=clean_index)


def found(toolkit, query):
    """The ids of the source items a search of the index returns."""
    outcome = toolkit.search_index(query, index_name=INDEX_NAME)
    if isinstance(outcome, str):
        return set()
    return {hit["metadata"]["id"] for hit in outcome}


class TestCleanIndexAfterTheCheck:

    def test_an_index_of_the_unchanged_source_is_up_to_date(self, world):
        toolkit, _, store = world

        index(toolkit)
        report = toolkit.check_for_changes(INDEX_NAME)

        assert len(store.chunks_of("budget")) == 3
        assert report["status"] == check.STATUS_UP_TO_DATE
        assert report["counts"]["unchanged"] == 4

    def test_the_check_reports_a_removed_and_an_edited_item(self, world):
        toolkit, source, _ = world
        index(toolkit)

        source.remove("faq")
        source.put("budget", "budget plan revised", "2026-03-03")
        report = toolkit.check_for_changes(INDEX_NAME)

        assert report["status"] == check.STATUS_NEEDS_CLEAN_REINDEX
        assert report["counts"]["removed"] == 1
        assert report["counts"]["changed"] == 1
        assert report["items"]["removed"] == ["/lib/faq.txt"]
        assert report["items"]["changed"] == ["/lib/budget.txt"]

    def test_chunks_the_check_reports_as_removed_or_changed_are_searchable_until_the_clean_run(self, world):
        toolkit, source, _ = world
        index(toolkit)

        source.remove("faq")
        source.put("budget", "budget plan revised", "2026-03-03")
        report = toolkit.check_for_changes(INDEX_NAME)

        assert report["items"]["removed"] == ["/lib/faq.txt"]
        assert report["items"]["changed"] == ["/lib/budget.txt"]
        assert found(toolkit, "faq") == {"faq"}
        assert found(toolkit, "plan one") == {"budget"}

    def test_clean_index_leaves_no_chunk_of_a_removed_or_edited_item(self, world):
        toolkit, source, store = world
        index(toolkit)
        source.remove("faq")
        source.put("budget", "budget plan revised", "2026-03-03")
        assert toolkit.check_for_changes(INDEX_NAME)["status"] == check.STATUS_NEEDS_CLEAN_REINDEX

        index(toolkit, clean_index=True)

        assert store.chunks_of("faq") == []
        assert found(toolkit, "faq") == set()
        stale_budget = [row["content"] for row in store.chunks_of("budget")
                        if row["content"] in ("budget plan one", "budget plan two", "budget plan three")]
        assert stale_budget == []
        assert found(toolkit, "plan one") == set()
        assert found(toolkit, "plan three") == set()

    def test_clean_index_keeps_the_current_content_of_every_remaining_item(self, world):
        toolkit, source, store = world
        index(toolkit)
        source.remove("faq")
        source.put("budget", "budget plan revised", "2026-03-03")

        index(toolkit, clean_index=True)

        assert found(toolkit, "plan revised") == {"budget"}
        assert found(toolkit, "alpha") == {"handbook"}
        assert found(toolkit, "roadmap") == {"roadmap"}
        assert {row["metadata"]["id"] for row in store.data_rows()} == {"handbook", "budget", "roadmap"}
        assert len(store.chunks_of("handbook")) == 3
        assert len(store.chunks_of("budget")) == 1
        assert toolkit.check_for_changes(INDEX_NAME)["counts"]["unchanged"] == 3

    def test_the_check_after_the_clean_run_is_up_to_date(self, world):
        toolkit, source, _ = world
        index(toolkit)
        source.remove("faq")
        source.put("budget", "budget plan revised", "2026-03-03")
        index(toolkit, clean_index=True)

        report = toolkit.check_for_changes(INDEX_NAME)

        assert report["status"] == check.STATUS_UP_TO_DATE
        assert report["counts"] == {"added": 0, "changed": 0, "removed": 0, "unchanged": 3, "not_indexed": 0}

    def test_the_index_has_one_meta_row_after_the_clean_run(self, world):
        toolkit, source, store = world
        index(toolkit)
        source.remove("faq")

        index(toolkit, clean_index=True)

        meta_rows = [row for row in store.rows.values() if row["metadata"].get("type") == META_TYPE]
        assert len(meta_rows) == 1


def change_the_source(source):
    source.remove("faq")
    source.put("budget", "budget plan revised", "2026-03-03")


class TestStagedCleanIndexAfterTheCheck:
    """With run staging nothing is emptied up front: the promote step deletes the old chunks."""

    def test_a_loader_that_attests_lets_the_promote_delete_the_chunks_of_a_removed_item(
            self, attested_staged_world):
        toolkit, source, store = attested_staged_world
        index(toolkit)
        assert len(store.chunks_of("faq")) == 2
        change_the_source(source)
        assert toolkit.check_for_changes(INDEX_NAME)["status"] == check.STATUS_NEEDS_CLEAN_REINDEX

        index(toolkit, clean_index=True)

        assert store.chunks_of("faq") == []
        assert found(toolkit, "faq") == set()
        assert len(toolkit.vector_adapter.promotes[-1]["orphans"]) == 2
        assert {row["metadata"]["id"] for row in store.data_rows()} == {"handbook", "budget", "roadmap"}

    def test_a_loader_that_attests_leaves_no_chunk_of_an_edited_item_and_the_check_agrees(
            self, attested_staged_world):
        toolkit, source, store = attested_staged_world
        index(toolkit)
        change_the_source(source)

        index(toolkit, clean_index=True)

        assert [row["content"] for row in store.chunks_of("budget")] == ["budget plan revised"]
        assert found(toolkit, "plan one") == set()
        assert len(store.chunks_of("handbook")) == 3
        report = toolkit.check_for_changes(INDEX_NAME)
        assert report["status"] == check.STATUS_UP_TO_DATE
        assert report["counts"]["unchanged"] == 3

    def test_a_loader_that_cannot_attest_keeps_the_chunks_of_a_removed_item(self, staged_world):
        toolkit, source, store = staged_world
        index(toolkit)
        change_the_source(source)
        assert toolkit.check_for_changes(INDEX_NAME)["counts"]["removed"] == 1

        index(toolkit, clean_index=True)

        # The documented risk: Clean index cannot remove what the loader cannot confirm is gone.
        assert len(store.chunks_of("faq")) == 2
        assert found(toolkit, "faq") == {"faq"}
        assert toolkit.vector_adapter.promotes[-1]["orphans"] == []
        assert [row["content"] for row in store.chunks_of("budget")] == ["budget plan revised"]
