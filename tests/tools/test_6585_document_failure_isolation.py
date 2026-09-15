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

"""Guards the per-document failure boundary in ``_save_index_generator``.

Re-broken by anything that lets one document's failure end the run, discard the rows it
already flushed, leave a lost document out of the report, or collapse several failures
into one entry. Lives outside tests/tools/index/ because CI skips that directory.
"""

import pytest
from langchain_core.documents import Document

from elitea_sdk.runtime.tools.vectorstore_base import VectorStoreWrapperBase
from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.base_indexer_toolkit import (
    BaseIndexerToolkit,
    DEPENDENT_DOC_META_KEY,
    IndexingStats,
    IndexingStatus,
    _IndexRunState,
    render_grouped_errors,
)
from elitea_sdk.tools.code_indexer_toolkit import CodeIndexerToolkit
from elitea_sdk.tools.non_code_indexer_toolkit import NonCodeIndexerToolkit


class StagingToolkit(BaseIndexerToolkit):

    def key_fn(self, document: Document):
        return document.metadata.get('id')


class FakeStagingAdapter:
    supports_run_staging = True

    def __init__(self):
        self.promote_outcome = "promoted"
        self.calls = []
        self.heartbeat_chunks = []
        self.pending = []

    def ensure_index_runs_table(self, wrapper):
        self.calls.append("ensure")

    def register_index_run(self, wrapper, index_name, run_id, task_id=None, meta_lock_id=None):
        self.calls.append("register")
        return (True, None)

    def sweep_stale_index_runs(self, wrapper, index_name, stale_before):
        self.calls.append("sweep")
        return []

    def heartbeat_index_run(self, wrapper, index_name, run_id, meta_id, chunks_written=None):
        self.calls.append("heartbeat")
        self.heartbeat_chunks.append(chunks_written)

    def update_index_meta_keys(self, wrapper, meta_id, run_id, patch):
        # Mirrors the real merge: only the patched keys change, everything else on
        # the row survives. Replacing the dict here would hide exactly the bug the
        # keyed write exists to fix.
        stored = wrapper._stored_meta
        if stored is None:
            return 0
        merged = {**stored.get("metadata", {}), **patch}
        wrapper.written.append(merged)
        object.__setattr__(wrapper, "_stored_meta", {**stored, "metadata": merged})
        return 1

    def promote_run(self, wrapper, index_name, run_id, superseded_ids, orphan_ids, damaged_ids):
        self.calls.append("promote")
        return self.promote_outcome

    def discard_run(self, wrapper, index_name, run_id):
        self.calls.append("discard")
        return "discarded"

    def get_pending_run_ids(self, wrapper, index_name, include_cancelled=True):
        return list(self.pending)

    def get_index_meta(self, wrapper, index_name):
        return [wrapper._stored_meta] if wrapper._stored_meta else []


class ChunkYieldingToolkit(StagingToolkit):
    loader_yields_chunks = True

    def key_fn(self, document: Document):
        return document.metadata.get("filename")


def build_staged_toolkit(monkeypatch, toolkit_cls=None):
    instance = (toolkit_cls or StagingToolkit).model_construct()
    object.__setattr__(instance, "vector_adapter", FakeStagingAdapter())
    object.__setattr__(instance, "_stored_meta", None)
    object.__setattr__(instance, "toolkit_id", None)
    object.__setattr__(instance, "max_docs_per_add", 100)
    written = []
    object.__setattr__(instance, "written", written)

    def fake_add_documents(vectorstore=None, documents=None, ids=None):
        metadata = dict(documents[0].metadata)
        written.append(metadata)
        object.__setattr__(
            instance, "_stored_meta",
            {"id": "meta-1", "content": "index_meta_x", "metadata": metadata},
        )
        return ["meta-row-id"]

    monkeypatch.setattr(
        "elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents", fake_add_documents
    )
    monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_index_meta", lambda self, name: self._stored_meta)
    monkeypatch.setattr(VectorStoreWrapperBase, "get_indexed_count", lambda self, name: 965)
    monkeypatch.setattr(StagingToolkit, "_is_scheduled_run", lambda self: False)
    monkeypatch.setattr(StagingToolkit, "_log_tool_event", lambda self, *a, **kw: None)
    monkeypatch.setattr(StagingToolkit, "_emit_index_event", lambda self, name, error=None, state=None: None)
    monkeypatch.setattr(StagingToolkit, "_reduce_duplicates", lambda self, docs, name: docs)
    return instance


@pytest.fixture
def staged_toolkit(monkeypatch):
    return build_staged_toolkit(monkeypatch)


@pytest.fixture
def chunk_yielding_toolkit(monkeypatch):
    return build_staged_toolkit(monkeypatch, ChunkYieldingToolkit)


def run_index_data(toolkit, monkeypatch, documents, save):
    monkeypatch.setattr(StagingToolkit, "_base_loader", lambda self, **kwargs: iter(documents))
    monkeypatch.setattr(StagingToolkit, "_save_index_generator", save)
    return toolkit.index_data(index_name="x")


def make_documents(*ids):
    return [Document(page_content="body", metadata={"id": doc_id, "name": doc_id, "updated_on": "1"})
            for doc_id in ids]


def drive_pipeline(monkeypatch, documents, failing_names, workers=1, staging=False,
                   flush_raises=False, failure_messages=None, failing_ids=frozenset(),
                   items_processed=None):
    toolkit = StagingToolkit.model_construct()
    object.__setattr__(toolkit, "max_docs_per_add", 1)
    object.__setattr__(toolkit, "_index_workers", workers)
    object.__setattr__(toolkit, "_index_run", _IndexRunState(run_id="r1"))
    toolkit._init_indexing_stats()
    toolkit._indexing_stats.items_processed = (
        len(documents) if items_processed is None else items_processed
    )
    toolkit._indexing_stats.total_fetched = toolkit._indexing_stats.items_processed
    flushed = []

    def fake_extend_data(self, docs):
        base_doc = next(iter(docs))
        name = base_doc.metadata.get("name")
        if name in failing_names or base_doc.metadata.get("id") in failing_ids:
            raise RuntimeError((failure_messages or {}).get(name, "boom"))
        return iter([base_doc])

    def fake_add_documents(vectorstore=None, documents=None, ids=None):
        if flush_raises:
            raise RuntimeError("pgvector down")
        flushed.extend(documents)
        return [f"row-{index}" for index in range(len(documents))]

    monkeypatch.setattr(StagingToolkit, "_extend_data", fake_extend_data)
    monkeypatch.setattr(StagingToolkit, "_collect_dependencies", lambda self, docs: docs)
    monkeypatch.setattr(
        StagingToolkit, "_apply_loaders_chunkers",
        lambda self, docs, chunking_tool=None, chunking_config=None: docs,
    )
    monkeypatch.setattr(StagingToolkit, "_clean_metadata", lambda self, docs: docs)
    monkeypatch.setattr(StagingToolkit, "_staging_active", lambda self: staging)
    monkeypatch.setattr(VectorStoreWrapperBase, "_ensure_vectorstore_initialized", lambda self: None)
    monkeypatch.setattr(StagingToolkit, "_log_tool_event", lambda self, *a, **kw: None)
    monkeypatch.setattr(StagingToolkit, "index_meta_update", lambda self, *a, **kw: None)
    monkeypatch.setattr(
        "elitea_sdk.runtime.langchain.interfaces.llm_processor.add_documents", fake_add_documents
    )

    result = {"count": 0, "failed_count": 0, "docs_count": 0, "failed_docs": 0,
              "errors": [], "error_groups": {}}
    toolkit._save_index_generator(iter(documents), len(documents), None, None, result, index_name="x")
    return toolkit, result, flushed


class TrackingToolkit(NonCodeIndexerToolkit):
    pass


class TestSerialRunSurvivesADocumentFailure:
    def test_the_surviving_documents_still_reach_the_store(self, monkeypatch):
        documents = make_documents("d1", "d2", "d3")

        _, result, flushed = drive_pipeline(monkeypatch, documents, failing_names={"d2"})

        assert [doc.metadata["name"] for doc in flushed] == ["d1", "d3"]
        assert result["docs_count"] == 2
        assert result["failed_docs"] == 1

    def test_the_buffered_tail_is_flushed_when_the_last_document_fails(self, monkeypatch):
        documents = make_documents("d1", "d2")

        _, _, flushed = drive_pipeline(monkeypatch, documents, failing_names={"d2"})

        assert [doc.metadata["name"] for doc in flushed] == ["d1"]

    def test_the_parallel_branch_behaves_the_same_way(self, monkeypatch):
        documents = make_documents("d1", "d2", "d3")

        _, result, flushed = drive_pipeline(monkeypatch, documents, failing_names={"d2"}, workers=2)

        assert sorted(doc.metadata["name"] for doc in flushed) == ["d1", "d3"]
        assert result["failed_docs"] == 1

    def test_a_failed_document_is_marked_so_promote_cannot_supersede_it(self, monkeypatch):
        documents = make_documents("d1", "d2")

        toolkit, _, _ = drive_pipeline(monkeypatch, documents, failing_names={"d2"}, staging=True)

        assert toolkit._index_run.pipeline_failed_keys == {"d2"}
        assert "d2" not in toolkit._index_run.counted_doc_keys


class TestFailuresNameTheDocument:
    def test_a_pipeline_failure_names_the_document(self, monkeypatch):
        documents = make_documents("page-1", "page-2")

        _, result, _ = drive_pipeline(monkeypatch, documents, failing_names={"page-2"})

        assert render_grouped_errors(result["error_groups"]) == ["page-2: boom"]

    def test_a_flush_failure_names_the_documents_it_took_down(self, monkeypatch):
        documents = make_documents("page-1")

        _, result, _ = drive_pipeline(monkeypatch, documents, failing_names=set(), flush_raises=True)

        assert render_grouped_errors(result["error_groups"]) == ["page-1: pgvector down"]

    def test_many_documents_failing_on_one_root_cause_stay_one_error(self, monkeypatch):
        documents = make_documents(*[f"page-{index}" for index in range(20)])

        _, result, _ = drive_pipeline(
            monkeypatch, documents, failing_names={f"page-{index}" for index in range(20)}
        )

        rendered = render_grouped_errors(result["error_groups"])
        assert rendered == ["page-0, page-1, page-2 and 17 more: boom"]

    def test_a_distinct_error_keeps_its_own_line(self, monkeypatch):
        documents = make_documents("page-1", "page-2")

        _, result, _ = drive_pipeline(
            monkeypatch, documents, failing_names={"page-1", "page-2"},
            failure_messages={"page-2": "OpenAI rate limit exceeded"},
        )

        assert sorted(render_grouped_errors(result["error_groups"])) == [
            "page-1: boom", "page-2: OpenAI rate limit exceeded",
        ]

    def test_the_driver_tail_is_stripped_before_grouping(self, monkeypatch):
        documents = make_documents("page-1", "page-2")

        _, result, _ = drive_pipeline(
            monkeypatch, documents, failing_names={"page-1", "page-2"},
            failure_messages={
                "page-1": "duplicate key [SQL: INSERT INTO x] [parameters: (1,)]",
                "page-2": "duplicate key [SQL: INSERT INTO x] [parameters: (2,)]",
            },
        )

        assert render_grouped_errors(result["error_groups"]) == ["page-1, page-2: duplicate key"]

    def test_a_flush_failure_names_a_nameless_document_by_its_id(self, monkeypatch):
        documents = [Document(page_content="body",
                              metadata={"id": "1-abc!123", "updated_on": "1"})]

        _, result, _ = drive_pipeline(monkeypatch, documents, failing_names=set(),
                                      flush_raises=True)

        assert render_grouped_errors(result["error_groups"]) == ["1-abc!123: pgvector down"]

    def test_the_parse_error_label_prefers_the_document_over_the_content_type(self):
        label = BaseIndexerToolkit._document_label({"name": "design.pdf", "id": "42"}, ".pdf")

        assert label == "design.pdf"

    def test_an_id_beats_the_content_type_for_a_nameless_document(self):
        assert BaseIndexerToolkit._document_label({"id": "1-abc!123"}, ".html") == "1-abc!123"

    def test_the_content_type_is_the_last_resort(self):
        assert BaseIndexerToolkit._document_label({}, ".pdf") == ".pdf"

    def test_a_present_but_empty_title_falls_through_to_the_id(self):
        assert BaseIndexerToolkit._document_label(
            {"title": "", "id": "1-abc!123"}, "1-abc!123.html") == "1-abc!123"

    def test_a_base_document_parse_error_is_recorded_under_the_document(self, monkeypatch):
        toolkit = TrackingToolkit.model_construct()
        object.__setattr__(toolkit, "embeddings", None)
        object.__setattr__(toolkit, "llm", None)
        toolkit._init_indexing_stats()
        monkeypatch.setattr(
            "elitea_sdk.tools.base_indexer_toolkit.process_document_by_type",
            lambda **kwargs: iter([
                Document(page_content="Error during content parsing for file", metadata={})
            ]),
        )
        document = Document(page_content="", metadata={
            "id": "att-1",
            "name": "design.pdf",
            IndexerKeywords.CONTENT_FILE_NAME.value: ".pdf",
            IndexerKeywords.CONTENT_IN_BYTES.value: b"payload",
        })

        list(toolkit._apply_loaders_chunkers(iter([document])))

        assert toolkit.get_indexing_stats().runtime_skipped_error == {"design.pdf"}

    def test_a_failed_document_lands_in_the_report_failed_group(self, monkeypatch):
        documents = make_documents("page-1", "page-2")

        toolkit, _, _ = drive_pipeline(monkeypatch, documents, failing_names={"page-2"})

        report = toolkit.get_indexing_stats().build_report(
            status=IndexingStatus.PARTLY_INDEXED, indexed_count=1,
            item_labels=("page", "pages"), dependent_labels=("attachment", "attachments"),
        )
        failed = next(category for category in report["categories"] if category["kind"] == "failed")
        assert failed["groups"][0]["items"] == ["page-2"]
        assert report["totals"]["failed"] == 1


    def test_a_flush_damaged_document_is_named_in_the_report(self, staged_toolkit, monkeypatch):
        def save_with_damage(self, base_documents, base_total, chunking_tool, chunking_config,
                             result, index_name=None):
            list(base_documents)
            result["count"] = 4
            result["docs_count"] = 2
            self._index_run.damaged_keys.add("d3")
            self._index_run.counted_doc_keys.add("d3")
            self._index_run.doc_names["d3"] = "design.pdf"

        outcome = run_index_data(staged_toolkit, monkeypatch, make_documents("d1", "d2", "d3"),
                                 save_with_damage)

        failed = next(category for category in outcome["report"]["categories"]
                      if category["kind"] == "failed")
        assert failed["groups"][0]["items"] == ["design.pdf"]
        assert outcome["status"] == IndexingStatus.PARTLY_INDEXED.value


class TestTerminalState:
    @staticmethod
    def _save(failed_keys=(), unchanged=0, **counters):
        def save(self, base_documents, base_total, chunking_tool, chunking_config, result, index_name=None):
            list(base_documents)
            for key in failed_keys:
                self._index_run.pipeline_failed_keys.add(key)
            for index in range(unchanged):
                self._track_document_unchanged(f"unchanged-{index}")
            result.update(counters)
        return save

    def test_a_run_that_lost_every_document_fails(self, staged_toolkit, monkeypatch):
        outcome = run_index_data(
            staged_toolkit, monkeypatch, make_documents("d1", "d2", "d3"),
            self._save(failed_keys=("d1", "d2", "d3"), count=0, docs_count=0, failed_docs=3),
        )

        assert outcome["status"] == IndexingStatus.ERROR.value
        assert "discard" in staged_toolkit.vector_adapter.calls
        assert "promote" not in staged_toolkit.vector_adapter.calls

    def test_a_run_that_kept_some_documents_is_partly_indexed(self, staged_toolkit, monkeypatch):
        outcome = run_index_data(
            staged_toolkit, monkeypatch, make_documents("d1", "d2", "d3"),
            self._save(failed_keys=("d3",), count=4, docs_count=2, failed_docs=1),
        )

        assert outcome["status"] == IndexingStatus.PARTLY_INDEXED.value
        assert "promote" in staged_toolkit.vector_adapter.calls

    def test_an_all_fail_run_without_staging_marks_still_fails(self, staged_toolkit, monkeypatch):
        outcome = run_index_data(
            staged_toolkit, monkeypatch, make_documents("d1"),
            self._save(count=0, docs_count=0, failed_docs=1),
        )

        assert outcome["status"] == IndexingStatus.ERROR.value

    def test_an_incremental_run_is_not_condemned_by_its_one_changed_document(
            self, staged_toolkit, monkeypatch):
        outcome = run_index_data(
            staged_toolkit, monkeypatch, make_documents("changed"),
            self._save(failed_keys=("changed",), unchanged=499, count=0, docs_count=0,
                       failed_docs=1),
        )

        assert outcome["status"] == IndexingStatus.PARTLY_INDEXED.value
        assert "promote" in staged_toolkit.vector_adapter.calls
        assert "discard" not in staged_toolkit.vector_adapter.calls

    def test_a_clean_run_still_completes(self, staged_toolkit, monkeypatch):
        outcome = run_index_data(
            staged_toolkit, monkeypatch, make_documents("d1"),
            self._save(count=3, docs_count=1),
        )

        assert outcome["status"] == IndexingStatus.OK.value
        assert "promote" in staged_toolkit.vector_adapter.calls


class CodeToolkit(CodeIndexerToolkit):
    pass


class TestSkipSetReconciliation:
    def test_the_code_family_now_records_a_failed_document(self):
        toolkit = CodeToolkit.model_construct()
        toolkit._init_indexing_stats()
        toolkit._indexing_stats.items_processed = 3
        toolkit._indexing_stats.total_fetched = 3

        toolkit._track_document_failed("src/app.py", Document(page_content="", metadata={"file_path": "src/app.py"}))

        stats = toolkit.get_indexing_stats()
        assert stats.documents_skipped_error == {"src/app.py"}
        assert stats.items_processed == 2
        assert stats.total_fetched == stats.items_processed + stats.to_dict()["total_skipped"]

    def test_a_file_the_loader_already_dropped_is_not_counted_twice(self):
        toolkit = CodeToolkit.model_construct()
        toolkit._init_indexing_stats()
        stats = toolkit.get_indexing_stats()
        stats.files_skipped_empty.add("src/empty.py")
        stats.items_processed = 2
        stats.total_fetched = 3

        toolkit._track_document_failed("src/empty.py", Document(page_content="", metadata={"file_path": "src/empty.py"}))

        assert stats.documents_skipped_error == set()
        assert stats.items_processed == 2
        assert stats.total_fetched == stats.items_processed + stats.to_dict()["total_skipped"]

    def test_the_same_document_failing_twice_rolls_back_once(self):
        toolkit = CodeToolkit.model_construct()
        toolkit._init_indexing_stats()
        toolkit._indexing_stats.items_processed = 5

        toolkit._track_document_failed("src/app.py")
        toolkit._track_document_failed("src/app.py")

        assert toolkit.get_indexing_stats().items_processed == 4

    def test_a_damaged_document_keeps_its_processed_count(self):
        toolkit = CodeToolkit.model_construct()
        toolkit._init_indexing_stats()
        toolkit._indexing_stats.items_processed = 5

        toolkit._track_document_damaged("src/app.py")

        stats = toolkit.get_indexing_stats()
        assert stats.documents_skipped_error == {"src/app.py"}
        assert stats.items_processed == 5

    def test_the_tracker_builds_stats_on_demand(self):
        toolkit = CodeToolkit.model_construct()

        toolkit._track_document_failed("src/app.py")

        assert isinstance(toolkit.get_indexing_stats(), IndexingStats)


class TestDocumentNaming:

    @pytest.mark.parametrize("metadata,expected", [
        ({"title": "Release Notes", "id": "123", "source": "http://x"}, "Release Notes"),
        ({"id": "1", "issue_key": "EL-6585", "source": "http://x"}, "EL-6585"),
        ({"id": "42", "type": "Bug", "title": "Crash on save"}, "Crash on save"),
        ({"file_path": "src/app.py", "filename": "src/app.py"}, "src/app.py"),
        ({"name": "design.pdf", "title": "ignored"}, "design.pdf"),
        ({"id": "42"}, "42"),
        ({}, "unknown"),
    ], ids=["confluence-page", "jira-issue", "ado-work-item", "code-file",
            "name-wins-over-title", "id-is-the-last-resort", "nothing-at-all"])
    def test_the_name_chain_covers_the_real_loaders(self, metadata, expected):
        assert BaseIndexerToolkit._extract_doc_name(metadata) == expected

    def test_an_id_never_displaces_a_readable_name(self):
        assert BaseIndexerToolkit._extract_doc_name(
            {"id": "90210", "title": "Crash on save"}) == "Crash on save"
        assert BaseIndexerToolkit._extract_doc_name({"id": "90210"}) == "90210"

    def test_a_non_string_name_is_coerced(self):
        stats = IndexingStats()
        stats.documents_skipped_error.add(BaseIndexerToolkit._extract_doc_name({"key": 7}))
        stats.documents_skipped_error.add("a.md")

        assert stats.to_dict()["documents_skipped"]["error"] == ["7", "a.md"]

    @pytest.mark.parametrize("titles", [
        ["Overview"] * 10,
        [f"Page {index}" for index in range(10)],
    ], ids=["colliding-titles", "distinct-titles"])
    def test_the_rollback_stays_in_step_with_the_name_set(self, monkeypatch, titles):
        documents = [
            Document(page_content="b", metadata={"id": f"p{index}", "title": title,
                                                 "updated_on": "1"})
            for index, title in enumerate(titles)
        ]
        toolkit, _, _ = drive_pipeline(
            monkeypatch, documents, failing_names=set(),
            failing_ids={f"p{index}" for index in range(10)}, items_processed=10,
        )

        stats = toolkit.get_indexing_stats()
        assert stats.total_fetched == stats.items_processed + stats.to_dict()["total_skipped"]
        assert stats.documents_skipped_error == set(titles)


class TestChunkYieldingLoaderIsNotDoubleCounted:

    def run(self, monkeypatch, toolkit):
        def save(self, base_documents, base_total, chunking_tool, chunking_config,
                 result, index_name=None):
            list(base_documents)
            self._init_indexing_stats()
            self._indexing_stats.items_processed = 2
            self._indexing_stats.total_fetched = 2
            self._index_run.counted_doc_keys.update({"f1.py", "f2.py"})
            self._index_run.pipeline_failed_keys.add("f1.py")
            self._track_document_failed("f1.py")
            result.update(count=2, docs_count=2, failed_docs=1)

        return run_index_data(toolkit, monkeypatch, make_documents("d1"), save)

    def test_the_clean_file_survives(self, chunk_yielding_toolkit, monkeypatch):
        outcome = self.run(monkeypatch, chunk_yielding_toolkit)

        assert outcome["status"] == IndexingStatus.PARTLY_INDEXED.value
        assert outcome["report"]["totals"]["indexed"] == 1
        assert "promote" in chunk_yielding_toolkit.vector_adapter.calls
        assert "discard" not in chunk_yielding_toolkit.vector_adapter.calls


class TestAPipelineFailureIsNotAlsoFiledAsFlushDamage:

    def test_a_failed_document_the_loader_already_skipped_is_listed_once(
            self, staged_toolkit, monkeypatch):
        def save(self, base_documents, base_total, chunking_tool, chunking_config,
                 result, index_name=None):
            list(base_documents)
            self._init_indexing_stats()
            self._indexing_stats.items_processed = 1
            self._indexing_stats.total_fetched = 2
            self._indexing_stats.files_skipped_empty.add("f1.py")
            self._index_run.doc_names["f1.py"] = "f1.py"
            self._index_run.pipeline_failed_keys.add("f1.py")
            self._index_run.damaged_keys.add("f1.py")
            self._index_run.counted_doc_keys.add("f2.py")
            result.update(count=1, docs_count=1, failed_docs=1)

        outcome = run_index_data(staged_toolkit, monkeypatch, make_documents("d1"), save)

        stats = staged_toolkit.get_indexing_stats()
        assert stats.documents_skipped_error == set()
        assert outcome["report"]["totals"]["failed"] == 0
        assert outcome["report"]["totals"]["skipped"] == 1
        assert stats.total_fetched == stats.items_processed + stats.to_dict()["total_skipped"]


class TestDependentsAreReportedBesideTheirParent:

    def parse(self, monkeypatch, metadata, sentinel="Error during content parsing for file"):
        toolkit = TrackingToolkit.model_construct()
        object.__setattr__(toolkit, "embeddings", None)
        object.__setattr__(toolkit, "llm", None)
        toolkit._init_indexing_stats()
        toolkit._indexing_stats.items_processed = 1
        toolkit._indexing_stats.total_fetched = 1
        monkeypatch.setattr(
            "elitea_sdk.tools.base_indexer_toolkit.process_document_by_type",
            lambda **kwargs: iter([Document(page_content=sentinel, metadata={})]),
        )
        document = Document(page_content="", metadata={
            **metadata,
            IndexerKeywords.CONTENT_IN_BYTES.value: b"payload",
        })
        list(toolkit._apply_loaders_chunkers(iter([document])))
        return toolkit.get_indexing_stats()

    def test_a_failed_attachment_never_enters_the_document_totals(self, monkeypatch):
        stats = self.parse(monkeypatch, {
            "id": "att-1", "name": "design.pdf",
            IndexerKeywords.PARENT.value: "page-1",
            DEPENDENT_DOC_META_KEY: True,
            IndexerKeywords.CONTENT_FILE_NAME.value: "design.pdf",
        })

        assert stats.dependent_items_skipped == {"design.pdf"}
        assert stats.runtime_skipped_error == set()
        assert stats.to_dict()["total_skipped"] == 0

    def test_a_qtest_base_document_is_not_mistaken_for_an_attachment(self, monkeypatch):
        stats = self.parse(monkeypatch, {
            "id": "tc-1", "name": "Login works", "parent_id": "module-9",
            IndexerKeywords.CONTENT_FILE_NAME.value: "test_case.md",
        })

        assert stats.runtime_skipped_error == {"Login works"}
        assert stats.dependent_items_skipped == set()
        assert stats.to_dict()["total_skipped"] == 1

    def test_a_failed_base_document_comes_out_of_items_processed(self, monkeypatch):
        stats = self.parse(monkeypatch, {
            "id": "1", "issue_key": "PRJ-1",
            IndexerKeywords.CONTENT_FILE_NAME.value: "base_doc.md",
        })

        assert stats.runtime_skipped_error == {"PRJ-1"}
        assert stats.total_fetched == stats.items_processed + stats.to_dict()["total_skipped"]

    def test_an_attachment_keeps_its_parents_processed_count(self, monkeypatch):
        stats = self.parse(monkeypatch, {
            "id": "att-1", "name": "design.pdf",
            DEPENDENT_DOC_META_KEY: True,
            IndexerKeywords.CONTENT_FILE_NAME.value: "design.pdf",
        })

        assert stats.items_processed == 1

    def test_the_marker_never_reaches_the_store(self):
        assert DEPENDENT_DOC_META_KEY in BaseIndexerToolkit._remove_metadata_keys(
            BaseIndexerToolkit.model_construct())


class TestTheDependentMarkerIsStamped:

    def test_emitting_a_dependent_stamps_it(self):
        class DepToolkit(StagingToolkit):
            def _process_document(self, document):
                yield Document(page_content="att", metadata={"id": "att-1", "name": "design.pdf"})

        toolkit = DepToolkit.model_construct()
        object.__setattr__(toolkit, "_index_workers", 1)
        parent = Document(page_content="body", metadata={"id": "page-1", "updated_on": "1"})

        emitted = list(toolkit._collect_dependencies(iter([parent])))

        dependent = next(d for d in emitted if d.metadata.get("id") == "att-1")
        assert dependent.metadata[DEPENDENT_DOC_META_KEY] is True
        assert dependent.metadata[IndexerKeywords.PARENT.value] == "page-1"
        base = next(d for d in emitted if d.metadata.get("id") == "page-1")
        assert DEPENDENT_DOC_META_KEY not in base.metadata


class TestADependentIsNamedByItsOwnFile:

    def test_a_comment_is_not_named_after_its_issue(self, monkeypatch):
        toolkit = TrackingToolkit.model_construct()
        object.__setattr__(toolkit, "embeddings", None)
        object.__setattr__(toolkit, "llm", None)
        toolkit._init_indexing_stats()
        monkeypatch.setattr(
            "elitea_sdk.tools.base_indexer_toolkit.process_document_by_type",
            lambda **kwargs: iter([Document(
                page_content="Error during content parsing for file", metadata={})]),
        )
        comment = Document(page_content="", metadata={
            "id": "c-1", "issue_key": "PRJ-1",
            IndexerKeywords.PARENT.value: "PRJ-1",
            DEPENDENT_DOC_META_KEY: True,
            IndexerKeywords.CONTENT_FILE_NAME.value: "comment.md",
            IndexerKeywords.CONTENT_IN_BYTES.value: b"payload",
        })

        list(toolkit._apply_loaders_chunkers(iter([comment])))

        stats = toolkit.get_indexing_stats()
        assert stats.dependent_items_skipped == {"comment.md"}
        assert "PRJ-1" not in stats.dependent_items_skipped
        assert stats.runtime_skipped_error == set()


class TestLoadersNameTheAttachmentTheyFetch:

    def test_confluence_passes_the_attachment_filename(self, monkeypatch):
        from elitea_sdk.tools.confluence.api_wrapper import ConfluenceAPIWrapper

        wrapper = ConfluenceAPIWrapper.model_construct()
        object.__setattr__(wrapper, "_index_include_attachments", True)
        monkeypatch.setattr(ConfluenceAPIWrapper, "_attachment_passes_extension_filters",
                            lambda self, name: True)
        monkeypatch.setattr(ConfluenceAPIWrapper, "_build_page_url", lambda self, links: "http://x")

        class FakeClient:
            url = "http://confluence"
            def history(self, attachment_id):
                return {}
            def request(self, method=None, path=None, advanced_mode=False):
                return type("R", (), {"status_code": 200, "content": b"%PDF-1.4"})()

        object.__setattr__(wrapper, "client", FakeClient())
        parent = Document(page_content="body", metadata={"id": "page-1", "_attachments_data": [{
            "id": "att-1", "title": "design.pdf", "extensions": {"fileSize": 10},
            "metadata": {"mediaType": "application/pdf", "labels": {"results": []}},
            "_links": {"download": "/download/design.pdf"},
        }]})

        emitted = list(wrapper._process_document(parent))

        assert emitted, "the attachment should have been emitted"
        assert emitted[0].metadata[IndexerKeywords.CONTENT_FILE_NAME.value] == "design.pdf"

    def test_ado_wiki_passes_the_attachment_filename(self, monkeypatch):
        from elitea_sdk.tools.ado.wiki import ado_wrapper as ado
        Wrapper = ado.AzureDevOpsApiWrapper

        wrapper = Wrapper.model_construct()
        object.__setattr__(wrapper, "_index_include_attachments", True)
        object.__setattr__(wrapper, "_index_wiki_identifier", "wiki")
        object.__setattr__(wrapper, "_index_workers", 1)
        monkeypatch.setattr(Wrapper, "_apply_image_processing_to_parent", lambda self, doc: None)
        monkeypatch.setattr(Wrapper, "_get_repos_wrapper", lambda self, ident: object())
        monkeypatch.setattr(Wrapper, "_matches_extension_filter", lambda self, name: True)
        monkeypatch.setattr(Wrapper, "_download_attachment_with_retry",
                            lambda self, repos, path: b"\x89PNG payload")

        parent = Document(page_content="body", metadata={
            "id": "page-1", "path": "/Home",
            "_ado_wiki_attachments": ["/.attachments/diagram.png"],
        })

        emitted = list(wrapper._process_document(parent))

        assert emitted, "the attachment should have been emitted"
        assert emitted[0].metadata[IndexerKeywords.CONTENT_FILE_NAME.value] == "diagram.png"


class TestNamelessDocumentsDoNotCollapse:

    def test_two_untitled_pages_failing_are_two_failures(self, monkeypatch):
        documents = [
            Document(page_content="b", metadata={"id": f"1-abc!{index}", "title": "",
                                                 "updated_on": "1"})
            for index in (1, 2)
        ]

        toolkit, result, _ = drive_pipeline(
            monkeypatch, documents, failing_names=set(),
            failing_ids={"1-abc!1", "1-abc!2"}, items_processed=2,
        )

        stats = toolkit.get_indexing_stats()
        assert result["failed_docs"] == 2
        assert sorted(stats.documents_skipped_error) == ["1-abc!1", "1-abc!2"]
        assert render_grouped_errors(result["error_groups"]) == ["1-abc!1, 1-abc!2: boom"]
        assert stats.items_processed == 0
        assert stats.total_fetched == stats.items_processed + stats.to_dict()["total_skipped"]

    def test_the_pipeline_path_and_the_flush_path_agree(self, monkeypatch):
        document = [Document(page_content="b", metadata={"id": "1-abc!7", "updated_on": "1"})]

        _, pipeline_result, _ = drive_pipeline(monkeypatch, document, failing_names=set(),
                                               failing_ids={"1-abc!7"})
        _, flush_result, _ = drive_pipeline(monkeypatch, document, failing_names=set(),
                                            flush_raises=True)

        pipeline_name = render_grouped_errors(pipeline_result["error_groups"])[0].split(":")[0]
        flush_name = render_grouped_errors(flush_result["error_groups"])[0].split(":")[0]
        assert pipeline_name == flush_name == "1-abc!7"


class TestTheSecondParseBranch:

    def parse(self, monkeypatch, metadata, toolkit_cls=None):
        toolkit = (toolkit_cls or TrackingToolkit).model_construct()
        object.__setattr__(toolkit, "embeddings", None)
        object.__setattr__(toolkit, "llm", None)
        toolkit._init_indexing_stats()
        toolkit._indexing_stats.items_processed = 2
        toolkit._indexing_stats.total_fetched = 2
        monkeypatch.setattr(
            "elitea_sdk.tools.base_indexer_toolkit.process_document_by_type",
            lambda **kwargs: iter([Document(
                page_content="Error during content parsing for file", metadata={})]),
        )
        document = Document(page_content="", metadata={
            **metadata, IndexerKeywords.CONTENT_IN_BYTES.value: b"payload"})
        list(toolkit._apply_loaders_chunkers(iter([document]), chunking_tool="markdown"))
        return toolkit.get_indexing_stats()

    def test_an_integer_id_cannot_crash_the_report(self, monkeypatch):
        stats = self.parse(monkeypatch, {"id": 12345, "updated_on": "1"})
        stats.documents_skipped_error.add("PRJ-1")

        report = stats.build_report(
            status=IndexingStatus.PARTLY_INDEXED, indexed_count=1,
            item_labels=("case", "cases"), dependent_labels=("attachment", "attachments"),
        )

        assert report["totals"]["failed"] == 2

    def test_it_reports_the_readable_name_not_the_id(self, monkeypatch):
        stats = self.parse(monkeypatch, {"id": 7, "path": "/Home/Setup", "updated_on": "1"})

        assert stats.runtime_skipped_error == {"/Home/Setup"}

    def test_a_dependent_on_this_branch_is_still_reported_beside_its_parent(self, monkeypatch):
        stats = self.parse(monkeypatch, {
            "id": "att-1", "name": "notes.md", "updated_on": "1",
            DEPENDENT_DOC_META_KEY: True,
        })

        assert stats.dependent_items_skipped
        assert stats.runtime_skipped_error == set()
        assert stats.items_processed == 2

    def test_the_tracker_itself_refuses_to_put_a_non_string_in_a_skip_set(self):
        toolkit = TrackingToolkit.model_construct()
        toolkit._init_indexing_stats()
        toolkit._indexing_stats.items_processed = 1

        toolkit._track_base_parse_failure(12345, 'runtime_skipped_error')
        toolkit._indexing_stats.documents_skipped_error.add("PRJ-1")

        assert toolkit._indexing_stats.runtime_skipped_error == {"12345"}
        toolkit._indexing_stats.build_report(
            status=IndexingStatus.ERROR, indexed_count=0,
            item_labels=("case", "cases"), dependent_labels=("a", "a"),
        )

    def test_a_failed_document_here_comes_out_of_items_processed(self, monkeypatch):
        stats = self.parse(monkeypatch, {"id": 7, "path": "/Home/Setup", "updated_on": "1"})

        assert stats.total_fetched == stats.items_processed + stats.to_dict()["total_skipped"]


class TestSharePointImagesAreStamped:

    def test_a_onenote_image_carries_the_dependent_marker(self, monkeypatch):
        from elitea_sdk.tools.sharepoint.api_wrapper import SharepointApiWrapper

        class FakeBackend:
            def _onenote_parse_page_items(self, page_id, capture_images,
                                          include_attachments, read_attachment_content):
                return [{"type": "image", "raw_bytes": b"\x89PNG",
                         "description": "", "filename": "diagram.jpg"}]
            def onenote_get_page_content(self, page_id):
                return "<html>page</html>"

        wrapper = SharepointApiWrapper.model_construct()
        object.__setattr__(wrapper, "_backend", FakeBackend())
        object.__setattr__(wrapper, "_onenote_cfg", {"capture_images": True})
        monkeypatch.setattr(SharepointApiWrapper, "_sync_backend_context", lambda self: None)
        page = Document(page_content="", metadata={
            "source_type": "onenote", "id": "1-abc!123", "title": "Notes",
            "updated_on": "1", "webUrl": "http://x",
        })

        emitted = list(wrapper._extend_data(iter([page])))

        images = [d for d in emitted if d.metadata.get("source_type") == "onenote_image"]
        assert images, "the OneNote image should have been emitted"
        assert images[0].metadata[DEPENDENT_DOC_META_KEY] is True
        assert images[0].metadata[IndexerKeywords.PARENT.value] == "1-abc!123"


class TestReportBudget:

    @staticmethod
    def sharepoint_paths(count, depth="Specifications/Archives"):
        return [f"/drives/b!{'x' * 44}/root:/Shared Documents/Engineering/{depth}/2026/"
                f"q{index}/design-review-notes-final-approved-v{index}.docx"
                for index in range(count)]

    def test_the_names_survive_a_long_message(self):
        from elitea_sdk.tools.base_indexer_toolkit import (
            REPORT_ERROR_MAX_LENGTH, normalize_report_errors, render_grouped_errors,
        )
        groups = {"x" * (REPORT_ERROR_MAX_LENGTH * 2): {"docs/a.md": None, "docs/b.md": None}}

        sampled, _ = normalize_report_errors(render_grouped_errors(groups))

        assert sampled[0].startswith("docs/a.md, docs/b.md: ")
        assert len(sampled[0]) <= REPORT_ERROR_MAX_LENGTH

    def test_two_causes_over_very_long_paths_stay_two_errors(self):
        from elitea_sdk.tools.base_indexer_toolkit import (
            normalize_report_errors, render_grouped_errors,
        )
        names = {path: None for path in self.sharepoint_paths(3)}
        groups = {
            "duplicate key value violates unique constraint pg_vector_pkey": dict(names),
            "connection to server was lost mid-transaction": dict(names),
        }

        sampled, total = normalize_report_errors(render_grouped_errors(groups))

        assert total == 2
        assert any("duplicate key" in line for line in sampled)
        assert any("connection to server was lost" in line for line in sampled)

    def test_a_path_too_long_to_list_is_folded_into_the_count(self):
        from elitea_sdk.tools.base_indexer_toolkit import (
            ERROR_DOC_NAMES_MAX_LENGTH, render_grouped_errors,
        )
        names = {path: None for path in self.sharepoint_paths(3)}

        rendered = render_grouped_errors({"pgvector down": names})

        head = rendered[0].rsplit(": ", 1)[0]
        assert len(head) <= ERROR_DOC_NAMES_MAX_LENGTH + len(" and 2 more")
        assert "and 2 more" in rendered[0]
        assert rendered[0].endswith(": pgvector down")

    def test_a_single_unlistable_name_still_leaves_the_cause_intact(self):
        from elitea_sdk.tools.base_indexer_toolkit import (
            ERROR_DOC_NAMES_MAX_LENGTH, render_grouped_errors,
        )
        monster = "/" + "d" * (ERROR_DOC_NAMES_MAX_LENGTH * 2)

        rendered = render_grouped_errors({"pgvector down": {monster: None}})

        assert rendered == ["pgvector down"]


class TestEveryOverrideStripsTheMarker:
    @pytest.mark.parametrize("module_path,class_name", [
        ("elitea_sdk.tools.ado.wiki.ado_wrapper", "AzureDevOpsApiWrapper"),
        ("elitea_sdk.tools.figma.api_wrapper", "FigmaApiWrapper"),
    ])
    def test_the_marker_never_reaches_the_store(self, module_path, class_name):
        import importlib
        toolkit_cls = getattr(importlib.import_module(module_path), class_name)

        keys = toolkit_cls._remove_metadata_keys(toolkit_cls.model_construct())

        assert DEPENDENT_DOC_META_KEY in keys
