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

from elitea_sdk.runtime.langchain.interfaces import llm_processor
from elitea_sdk.runtime.tools.artifact import ArtifactWrapper
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
from elitea_sdk.tools.sharepoint.api_wrapper import SharepointApiWrapper
from elitea_sdk.tools.utils import content_parser


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

    def sweep_stale_index_runs(self, wrapper, index_name, stale_before, except_run_id=None):
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

    def discard_run(self, wrapper, index_name, run_id, retain_chunks=False):
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


def run_index_data(toolkit, monkeypatch, documents, save=None, loader=None, chunking_tool=None):
    monkeypatch.setattr(StagingToolkit, "_base_loader",
                        loader or (lambda self, **kwargs: iter(documents)))
    if save is not None:
        monkeypatch.setattr(StagingToolkit, "_save_index_generator", save)
    return toolkit.index_data(index_name="x", chunking_tool=chunking_tool)


def failed_items(outcome, reason=None):
    return [item
            for category in outcome["report"]["categories"] if category["kind"] == "failed"
            for group in category["groups"] if reason in (None, group["reason"])
            for item in group["items"]]


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


class ParsingStagingToolkit(StagingToolkit, NonCodeIndexerToolkit):
    pass


class CodeStagingToolkit(StagingToolkit, CodeIndexerToolkit):
    pass


class ArtifactStagingToolkit(StagingToolkit, ArtifactWrapper):
    pass


class SharepointStagingToolkit(StagingToolkit, SharepointApiWrapper):
    pass


PARSE_ERROR = "Error during content parsing for file"
UNSUPPORTED = "Unsupported extension for file"


def make_files(*names, dependent=()):
    return [Document(page_content="", metadata={
        "id": name, "name": name, "updated_on": "1",
        IndexerKeywords.CONTENT_FILE_NAME.value: name,
        IndexerKeywords.CONTENT_IN_BYTES.value: b"payload",
        **({DEPENDENT_DOC_META_KEY: True} if name in dependent else {}),
    }) for name in names]


def make_page(doc_id, name, body="real body"):
    return Document(page_content=body, metadata={"id": doc_id, "name": name, "updated_on": "1"})


def parsed_by(key_of, parsed_as):
    def parse(metadata):
        contents = parsed_as.get(key_of(metadata), "parsed body")
        return (contents,) if isinstance(contents, str) else contents

    return parse


def stub_parser(monkeypatch, parse):
    def process_document_by_type(document=None, **kwargs):
        return iter([Document(page_content=content, metadata=dict(document.metadata))
                     for content in parse(document.metadata)])

    monkeypatch.setattr(
        "elitea_sdk.tools.base_indexer_toolkit.process_document_by_type", process_document_by_type
    )


def index_through_real_pipeline(monkeypatch, documents, parsed_as=None, parse=None, dependents=None,
                                toolkit_cls=ParsingStagingToolkit, configure=lambda toolkit: None,
                                before_loading=lambda toolkit: None, chunking_tool=None,
                                workers=1, staging=True, source=None):
    toolkit = build_staged_toolkit(monkeypatch, toolkit_cls)
    toolkit.vector_adapter.supports_run_staging = staging
    object.__setattr__(toolkit, "embeddings", None)
    object.__setattr__(toolkit, "llm", None)
    object.__setattr__(toolkit, "max_docs_per_add", 1)
    object.__setattr__(toolkit, "_index_workers", workers)
    stub_parser(monkeypatch, parse or parsed_by(BaseIndexerToolkit._extract_doc_name, parsed_as or {}))
    configure(toolkit)
    if dependents is not None:
        monkeypatch.setattr(
            toolkit_cls, "_process_document",
            lambda self, document: iter(dependents.get(self._extract_doc_name(document.metadata), ())),
        )
    write_index_meta = llm_processor.add_documents

    def add_documents(vectorstore=None, documents=None, ids=None):
        if documents[0].page_content.startswith(IndexerKeywords.INDEX_META_TYPE.value):
            return write_index_meta(vectorstore=vectorstore, documents=documents, ids=ids)
        return [f"row-{BaseIndexerToolkit._staging_key(doc.metadata)}" for doc in documents]

    def loader(self, **kwargs):
        before_loading(self)
        yield from (source(self, **kwargs) if source else documents)

    monkeypatch.setattr(llm_processor, "add_documents", add_documents)
    return toolkit, run_index_data(toolkit, monkeypatch, documents, loader=loader,
                                   chunking_tool=chunking_tool)


def assert_terminal_state(toolkit, outcome, status):
    written_state = {
        IndexingStatus.OK: IndexerKeywords.INDEX_META_COMPLETED.value,
        IndexingStatus.PARTLY_INDEXED: IndexerKeywords.INDEX_META_PARTLY_OK.value,
        IndexingStatus.ERROR: IndexerKeywords.INDEX_META_FAILED.value,
    }[status]
    assert outcome["status"] == status.value
    assert outcome["report"]["status"] == status.value
    assert toolkit.written[-1]["state"] == written_state


def assert_totals(outcome, **expected):
    totals = outcome["report"]["totals"]
    assert {key: totals[key] for key in expected} == expected
    assert totals["total"] == (totals["indexed"] + totals["skipped"] + totals["not_indexed"]
                               + totals["failed"] + totals["unchanged"])


PIPELINE_MODES = [
    pytest.param({"workers": workers, "staging": staging},
                 id=f"workers={workers}-{'staging' if staging else 'direct'}")
    for workers in (1, 2) for staging in (True, False)
]


def assert_run_outcome_kept(toolkit, mode, kept):
    if not mode["staging"]:
        assert toolkit.vector_adapter.calls == []
        return
    assert ("promote" in toolkit.vector_adapter.calls) is kept
    assert ("discard" in toolkit.vector_adapter.calls) is not kept


def parsed_by_id(parsed_as):
    return parsed_by(lambda metadata: metadata["id"], parsed_as)


@pytest.mark.parametrize("mode", PIPELINE_MODES)
class TestAParseFailureDecidesTheTerminalState:

    def test_a_corrupt_file_beside_good_ones_is_partly_indexed(self, monkeypatch, mode):
        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, make_files("a.docx", "b.docx", "corrupted.docx"),
            parsed_as={"corrupted.docx": PARSE_ERROR}, **mode)

        assert toolkit.get_indexing_stats().runtime_skipped_error == {"corrupted.docx"}
        assert_terminal_state(toolkit, outcome, IndexingStatus.PARTLY_INDEXED)
        assert failed_items(outcome, "processing_error") == ["corrupted.docx"]
        assert_totals(outcome, indexed=2, failed=1, total=3)
        assert_run_outcome_kept(toolkit, mode, kept=True)

    def test_a_run_whose_every_file_fails_to_parse_fails(self, monkeypatch, mode):
        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, make_files("pw_protected.pdf", "corrupted.docx"),
            parsed_as={"pw_protected.pdf": PARSE_ERROR, "corrupted.docx": PARSE_ERROR}, **mode)

        assert_terminal_state(toolkit, outcome, IndexingStatus.ERROR)
        assert failed_items(outcome, "processing_error") == ["corrupted.docx", "pw_protected.pdf"]
        assert_totals(outcome, indexed=0, failed=2)
        assert_run_outcome_kept(toolkit, mode, kept=False)

    @pytest.mark.parametrize("failing_id", ["first", "second"])
    def test_a_good_file_sharing_a_name_with_a_corrupt_one_stays_indexed(
            self, monkeypatch, mode, failing_id):
        documents = make_files("report.docx", "report.docx")
        documents[0].metadata["id"], documents[1].metadata["id"] = "first", "second"

        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, documents,
            parse=parsed_by_id({failing_id: PARSE_ERROR}), **mode)

        assert_terminal_state(toolkit, outcome, IndexingStatus.PARTLY_INDEXED)
        assert failed_items(outcome, "processing_error") == ["report.docx"]
        assert outcome["report"]["totals"]["indexed"] == 1
        assert_run_outcome_kept(toolkit, mode, kept=True)

    def test_a_file_that_fails_mid_parse_is_counted_only_as_failed(self, monkeypatch, mode):
        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, make_files("good.pdf", "half.pdf"),
            parsed_as={"half.pdf": ("first page", PARSE_ERROR)}, **mode)

        assert toolkit.get_indexing_stats().runtime_skipped_error == {"half.pdf"}
        assert_terminal_state(toolkit, outcome, IndexingStatus.PARTLY_INDEXED)
        assert failed_items(outcome) == failed_items(outcome, "processing_error") == ["half.pdf"]
        assert_totals(outcome, indexed=1, failed=1, total=2)
        assert toolkit.written[-1]["indexed"] == 1

    @pytest.mark.parametrize("failing_id", ["first", "second"])
    def test_a_good_file_sharing_a_name_with_one_that_fails_mid_parse_stays_indexed(
            self, monkeypatch, mode, failing_id):
        documents = make_files("report.pdf", "report.pdf")
        documents[0].metadata["id"], documents[1].metadata["id"] = "first", "second"

        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, documents, parse=parsed_by_id({failing_id: ("first page", PARSE_ERROR)}), **mode)

        assert_terminal_state(toolkit, outcome, IndexingStatus.PARTLY_INDEXED)
        assert failed_items(outcome, "processing_error") == ["report.pdf"]
        assert_totals(outcome, indexed=1, failed=1, total=2)
        assert toolkit.written[-1]["indexed"] == 1

    def test_a_code_file_that_fails_to_parse_lands_with_the_other_parse_failures(self, monkeypatch, mode):
        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, make_files("app.py", "broken.py"),
            toolkit_cls=CodeStagingToolkit, parsed_as={"broken.py": PARSE_ERROR}, **mode)

        stats = toolkit.get_indexing_stats()
        assert stats.runtime_skipped_error == {"broken.py"}
        assert stats.documents_skipped_error == set()
        assert_terminal_state(toolkit, outcome, IndexingStatus.PARTLY_INDEXED)
        assert failed_items(outcome, "processing_error") == ["broken.py"]
        assert_totals(outcome, indexed=1, failed=1, total=2)

    def test_unchanged_documents_keep_an_incremental_run_partly_indexed(self, monkeypatch, mode):
        def find_unchanged(toolkit):
            for name in ("a.docx", "b.docx"):
                toolkit._track_document_unchanged(name)

        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, make_files("corrupted.docx"), parsed_as={"corrupted.docx": PARSE_ERROR},
            before_loading=find_unchanged, **mode)

        assert_terminal_state(toolkit, outcome, IndexingStatus.PARTLY_INDEXED)
        assert_totals(outcome, indexed=0, failed=1, unchanged=2)
        assert_run_outcome_kept(toolkit, mode, kept=True)


class RepositoryStagingToolkit(StagingToolkit, CodeIndexerToolkit):
    def key_fn(self, document: Document):
        return document.metadata.get("filename")

    def _get_files(self, path, branch):
        return ["app.py", "locked.py"]

    def _read_file(self, file_path, branch):
        if file_path == "locked.py":
            raise ConnectionError("403 Forbidden")
        return "def main():\n    return 1\n"


class TestARepositoryFileThatCannotBeReadDecidesTheTerminalState:

    @pytest.mark.parametrize("mode", PIPELINE_MODES)
    def test_an_unreadable_file_beside_a_readable_one_is_partly_indexed(self, monkeypatch, mode):
        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, [], toolkit_cls=RepositoryStagingToolkit,
            source=CodeIndexerToolkit._base_loader, **mode)

        assert toolkit.get_indexing_stats().files_skipped_read_error == {"locked.py"}
        assert_terminal_state(toolkit, outcome, IndexingStatus.PARTLY_INDEXED)
        assert failed_items(outcome, "read_error") == ["locked.py"]
        assert_totals(outcome, indexed=1, failed=1, total=2)
        assert_run_outcome_kept(toolkit, mode, kept=True)


class TestOnlyExplicitFailuresChangeTheTerminalState:

    @staticmethod
    def zero_byte_file(name):
        return Document(page_content="", metadata={
            "id": name, "name": name, "updated_on": "1",
            IndexerKeywords.CONTENT_FILE_NAME.value: name,
            IndexerKeywords.CONTENT_IN_BYTES.value: b"",
        })

    def test_a_page_that_produced_nothing_is_reported_but_keeps_the_run_completed(self, monkeypatch):
        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, [make_page("1", "Page A"), make_page("2", "Blank page", body="")])

        assert toolkit.get_indexing_stats().documents_skipped_error == {"Blank page"}
        assert failed_items(outcome, "processing_error") == ["Blank page"]
        assert_terminal_state(toolkit, outcome, IndexingStatus.OK)
        assert_totals(outcome, indexed=1, failed=1, total=2)

    def test_a_zero_byte_file_that_parsed_to_nothing_is_reported_but_keeps_the_run_completed(
            self, monkeypatch):
        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, [make_page("1", "Page A"), self.zero_byte_file("blank.txt")],
            configure=lambda toolkit: monkeypatch.setattr(
                "elitea_sdk.tools.base_indexer_toolkit.process_document_by_type",
                content_parser.process_document_by_type))

        assert toolkit.get_indexing_stats().documents_skipped_error == {"blank.txt"}
        assert failed_items(outcome, "processing_error") == ["blank.txt"]
        assert_terminal_state(toolkit, outcome, IndexingStatus.OK)
        assert_totals(outcome, indexed=1, failed=1, total=2)

    def test_an_empty_file_keeps_the_run_completed(self, monkeypatch):
        empty = self.zero_byte_file("blank.md")
        del empty.metadata[IndexerKeywords.CONTENT_FILE_NAME.value]

        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, [make_page("1", "Page A"), empty], chunking_tool="markdown")

        assert toolkit.get_indexing_stats().files_skipped_empty == {"blank.md"}
        assert_terminal_state(toolkit, outcome, IndexingStatus.OK)
        assert_totals(outcome, indexed=1, skipped=1, failed=0, total=2)

    def test_an_unsupported_file_keeps_the_run_completed(self, monkeypatch):
        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, make_files("a.docx", "image.bmp"), parsed_as={"image.bmp": UNSUPPORTED})

        assert toolkit.get_indexing_stats().files_unsupported_extension == {"image.bmp"}
        assert_terminal_state(toolkit, outcome, IndexingStatus.OK)
        assert_totals(outcome, indexed=1, not_indexed=1, failed=0)

    def test_a_failed_attachment_keeps_the_run_completed(self, monkeypatch):
        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, make_files("page.html"), parsed_as={"design.pdf": PARSE_ERROR},
            dependents={"page.html": make_files("design.pdf", dependent=("design.pdf",))})

        assert toolkit.get_indexing_stats().dependent_items_skipped == {"design.pdf"}
        assert_terminal_state(toolkit, outcome, IndexingStatus.OK)
        assert_totals(outcome, indexed=1, failed=0, total=1)


class FakeArtifactClient:

    def __init__(self, served):
        self.served = served

    def get_content_bytes(self, artifact_name):
        response = self.served.get(artifact_name, b"payload")
        if isinstance(response, Exception):
            raise response
        return response


class FakeSharepointBackend:

    def __init__(self, unreachable=(), unreachable_pages=()):
        self.unreachable = unreachable
        self.unreachable_pages = unreachable_pages

    def _onenote_parse_page_items(self, page_id, **kwargs):
        return [{"type": "image", "raw_bytes": b"png", "description": f"diagram on {page_id}",
                 "filename": "diagram.png"}]

    def load_file_content_in_bytes(self, path):
        if path in self.unreachable:
            raise ConnectionError("401 Unauthorized")
        return b"payload"

    def onenote_get_page_content(self, page_id):
        if page_id in self.unreachable_pages:
            raise ConnectionError("Graph 503")
        return "<html>page</html>"


@pytest.mark.parametrize("mode", PIPELINE_MODES)
class TestADownloadFailureDecidesTheTerminalState:

    @staticmethod
    def artifacts(*names):
        return [Document(page_content="", metadata={"name": name, "id": f"sha-{name}", "updated_on": "1"})
                for name in names]

    @staticmethod
    def index_artifacts(monkeypatch, mode, documents, served):
        def configure(toolkit):
            object.__setattr__(toolkit, "bucket", "sweep")
            object.__setattr__(toolkit, "artifact", FakeArtifactClient(served))

        return index_through_real_pipeline(
            monkeypatch, documents, toolkit_cls=ArtifactStagingToolkit, configure=configure, **mode)

    @staticmethod
    def index_sharepoint(monkeypatch, mode, documents, backend, capture_images=False):
        def configure(toolkit):
            object.__setattr__(toolkit, "_backend", backend)
            object.__setattr__(toolkit, "_onenote_cfg", {"capture_images": capture_images})

        monkeypatch.setattr(SharepointStagingToolkit, "_sync_backend_context", lambda self: None)
        return index_through_real_pipeline(
            monkeypatch, documents, toolkit_cls=SharepointStagingToolkit, configure=configure, **mode)

    @staticmethod
    def assert_partly_indexed_with_unreadable(toolkit, mode, outcome, unreadable):
        stats = toolkit.get_indexing_stats()
        assert stats.files_skipped_read_error == {unreadable}
        assert stats.documents_skipped_error == set()
        assert_terminal_state(toolkit, outcome, IndexingStatus.PARTLY_INDEXED)
        assert failed_items(outcome) == failed_items(outcome, "read_error") == [unreadable]
        assert_totals(outcome, indexed=1, failed=1, total=2)
        assert toolkit.written[-1]["indexed"] == 1
        assert_run_outcome_kept(toolkit, mode, kept=True)

    def test_an_artifact_the_store_refuses_to_serve_is_partly_indexed(self, monkeypatch, mode):
        toolkit, outcome = self.index_artifacts(
            monkeypatch, mode, self.artifacts("good.txt", "report.pdf"),
            served={"report.pdf": {"error": "NoSuchKey"}})

        self.assert_partly_indexed_with_unreadable(toolkit, mode, outcome, "report.pdf")

    def test_an_artifact_whose_download_raises_is_partly_indexed(self, monkeypatch, mode):
        toolkit, outcome = self.index_artifacts(
            monkeypatch, mode, self.artifacts("good.txt", "report.pdf"),
            served={"report.pdf": ConnectionError("minio unreachable")})

        self.assert_partly_indexed_with_unreadable(toolkit, mode, outcome, "report.pdf")

    @staticmethod
    def sharepoint_files(*paths):
        return [Document(page_content="", metadata={
            "Name": path.rsplit("/", 1)[-1], "Path": path, "id": f"sp-{path}", "updated_on": "1"})
            for path in paths]

    def test_a_sharepoint_file_whose_download_raises_is_partly_indexed(self, monkeypatch, mode):
        toolkit, outcome = self.index_sharepoint(
            monkeypatch, mode, self.sharepoint_files("/sites/docs/good.docx", "/sites/docs/locked.docx"),
            FakeSharepointBackend(unreachable={"/sites/docs/locked.docx"}))

        self.assert_partly_indexed_with_unreadable(toolkit, mode, outcome, "locked.docx")

    @pytest.mark.parametrize("locked_folder", ["a", "b"])
    def test_a_sharepoint_file_sharing_a_name_with_a_locked_one_stays_indexed(
            self, monkeypatch, mode, locked_folder):
        toolkit, outcome = self.index_sharepoint(
            monkeypatch, mode, self.sharepoint_files("/sites/docs/a/Notes.docx", "/sites/docs/b/Notes.docx"),
            FakeSharepointBackend(unreachable={f"/sites/docs/{locked_folder}/Notes.docx"}))

        assert_terminal_state(toolkit, outcome, IndexingStatus.PARTLY_INDEXED)
        assert failed_items(outcome, "read_error") == ["Notes.docx"]
        assert outcome["report"]["totals"]["indexed"] == 1
        assert toolkit.written[-1]["indexed"] == 1
        assert_run_outcome_kept(toolkit, mode, kept=True)

    @staticmethod
    def onenote_pages(*pages):
        return [Document(page_content="", metadata={
            "source_type": "onenote", "id": page_id, "title": title, "updated_on": "1"})
            for page_id, title in pages]

    def test_a_onenote_page_whose_body_cannot_be_fetched_is_partly_indexed(self, monkeypatch, mode):
        toolkit, outcome = self.index_sharepoint(
            monkeypatch, mode, self.onenote_pages(("1-abc!1", "Meeting notes"), ("1-abc!2", "Roadmap")),
            FakeSharepointBackend(unreachable_pages={"1-abc!2"}))

        self.assert_partly_indexed_with_unreadable(toolkit, mode, outcome, "Roadmap")

    def test_a_onenote_page_whose_images_were_indexed_is_counted_only_as_failed(self, monkeypatch, mode):
        toolkit, outcome = self.index_sharepoint(
            monkeypatch, mode, self.onenote_pages(("1-abc!1", "Meeting notes"), ("1-abc!2", "Roadmap")),
            FakeSharepointBackend(unreachable_pages={"1-abc!2"}),
            capture_images=True)

        self.assert_partly_indexed_with_unreadable(toolkit, mode, outcome, "Roadmap")

    @pytest.mark.parametrize("capture_images", [False, True])
    def test_a_onenote_page_sharing_a_title_with_an_unreachable_one_stays_indexed(
            self, monkeypatch, mode, capture_images):
        pages = self.onenote_pages(("1-abc!1", "Untitled Page"), ("1-abc!2", "Untitled Page"))

        toolkit, outcome = self.index_sharepoint(
            monkeypatch, mode, pages, FakeSharepointBackend(unreachable_pages={"1-abc!1"}),
            capture_images=capture_images)

        assert_terminal_state(toolkit, outcome, IndexingStatus.PARTLY_INDEXED)
        assert failed_items(outcome, "read_error") == ["Untitled Page"]
        assert outcome["report"]["totals"]["indexed"] == 1
        assert toolkit.written[-1]["indexed"] == 1
        assert_run_outcome_kept(toolkit, mode, kept=True)

    def test_a_run_whose_every_download_fails_fails(self, monkeypatch, mode):
        toolkit, outcome = self.index_artifacts(
            monkeypatch, mode, self.artifacts("a.pdf", "b.pdf"),
            served={"a.pdf": {"error": "NoSuchKey"}, "b.pdf": ConnectionError("minio unreachable")})

        assert toolkit.get_indexing_stats().files_skipped_read_error == {"a.pdf", "b.pdf"}
        assert_terminal_state(toolkit, outcome, IndexingStatus.ERROR)
        assert failed_items(outcome, "read_error") == ["a.pdf", "b.pdf"]
        assert_totals(outcome, indexed=0, failed=2, total=2)
        assert_run_outcome_kept(toolkit, mode, kept=False)
