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

"""A large file is indexed in full while only a flush batch of its chunks is held in memory (#7047)."""

import threading

import pytest
from langchain_core.documents import Document

from elitea_sdk.runtime.langchain.document_loaders import utils as loader_utils
from elitea_sdk.tools import chunkers
from elitea_sdk.tools.base_indexer_toolkit import DEPENDENT_DOC_META_KEY, PARSE_SOURCE_META_KEY, IndexingStatus
from tests.tools.test_6585_document_failure_isolation import (
    PIPELINE_MODES,
    ParsingStagingToolkit,
    assert_terminal_state,
    failed_items,
    index_through_real_pipeline,
    make_files,
    make_page,
)

LARGE = 300


class StreamProbe:
    def __init__(self, chunks_per_file, fail_after=None, fail_on_calls=(1,)):
        self.chunks_per_file = chunks_per_file
        self.fail_after = fail_after or {}
        self.fail_on_calls = fail_on_calls
        self.calls = {}
        self.produced = {}
        self.consumed = {}
        self.most_held = {}

    def parse(self, document=None, **kwargs):
        name = document.metadata["name"]
        self.calls[name] = self.calls.get(name, 0) + 1
        return self._stream(name, self.calls[name], dict(document.metadata))

    def _stream(self, name, call, metadata):
        for index in range(self.chunks_per_file.get(name, 1)):
            if self.fail_after.get(name) == index and call in self.fail_on_calls:
                raise RuntimeError(f"{name} broke after {index} chunks")
            self.produced[name] = self.produced.get(name, 0) + 1
            held = self.produced[name] - self.consumed.get(name, 0)
            self.most_held[name] = max(self.most_held.get(name, 0), held)
            yield Document(page_content=f"chunk {index}", metadata=metadata)

    def consume(self, document):
        name = document.metadata["name"]
        self.consumed[name] = self.consumed.get(name, 0) + 1
        return None


def index_streamed(monkeypatch, mode, probe, names=("small.docx", "huge.xlsx")):
    monkeypatch.setattr(ParsingStagingToolkit, "_claim_adopted_row", probe.consume)

    def install_probe(toolkit):
        monkeypatch.setattr("elitea_sdk.tools.base_indexer_toolkit.process_document_by_type", probe.parse)

    return index_through_real_pipeline(monkeypatch, make_files(*names), configure=install_probe, **mode)


STAGED_SERIAL = {"workers": 1, "staging": True}
STAGED_MATERIALISED = {"workers": 2, "staging": True}


def rows_kept(toolkit):
    return toolkit._index_run.chunks_written - len(set(rows_deleted_at_promote(toolkit)))


def report_summary(toolkit, outcome):
    report = outcome["report"]
    return report["status"], report["totals"], failed_items(outcome), toolkit.written[-1]["updated"]


@pytest.mark.parametrize("mode", PIPELINE_MODES)
def test_a_large_file_is_indexed_in_full(monkeypatch, mode):
    probe = StreamProbe({"huge.xlsx": LARGE})

    toolkit, outcome = index_streamed(monkeypatch, mode, probe)

    assert_terminal_state(toolkit, outcome, IndexingStatus.OK)
    assert outcome["report"]["totals"]["indexed"] == 2
    assert probe.produced["huge.xlsx"] == LARGE


def test_a_staged_serial_run_consumes_each_chunk_as_it_is_parsed(monkeypatch):
    probe = StreamProbe({"huge.xlsx": LARGE})

    index_streamed(monkeypatch, STAGED_SERIAL, probe)

    assert probe.consumed["huge.xlsx"] == LARGE
    assert probe.most_held["huge.xlsx"] == 1


@pytest.mark.parametrize("fail_on_calls", [(1,), (1, 2)], ids=["recovers-on-retry", "fails-twice"])
def test_a_file_raising_after_its_chunks_were_handed_over_ends_like_a_materialised_run(monkeypatch, fail_on_calls):
    def run(mode):
        probe = StreamProbe({"huge.xlsx": LARGE}, fail_after={"huge.xlsx": 5}, fail_on_calls=fail_on_calls)
        toolkit, outcome = index_streamed(monkeypatch, mode, probe)
        return probe.calls["huge.xlsx"], report_summary(toolkit, outcome), rows_kept(toolkit)

    assert run(STAGED_SERIAL) == run(STAGED_MATERIALISED)


def test_a_file_failing_before_its_first_chunk_is_retried_once(monkeypatch):
    probe = StreamProbe({"huge.xlsx": LARGE}, fail_after={"huge.xlsx": 0}, fail_on_calls=(1,))

    toolkit, outcome = index_streamed(monkeypatch, STAGED_SERIAL, probe)

    assert probe.calls["huge.xlsx"] == 2
    assert_terminal_state(toolkit, outcome, IndexingStatus.OK)
    assert probe.consumed["huge.xlsx"] == LARGE



def test_the_inner_chunking_pool_parses_on_its_worker_threads(monkeypatch):
    parsed_on = []

    def parse(document=None, **kwargs):
        def stream():
            parsed_on.append(threading.current_thread().name)
            yield Document(page_content="chunk", metadata=dict(document.metadata))
        return stream()

    toolkit = ParsingStagingToolkit.model_construct()
    object.__setattr__(toolkit, "embeddings", None)
    object.__setattr__(toolkit, "llm", None)
    object.__setattr__(toolkit, "_index_workers", 2)
    monkeypatch.setattr("elitea_sdk.tools.base_indexer_toolkit.process_document_by_type", parse)

    chunks = list(toolkit._apply_loaders_chunkers(iter(make_files("a.docx", "b.docx", "c.docx"))))

    assert len(chunks) == 3
    assert parsed_on and all(name.startswith("indexer-chunk") for name in parsed_on)


def test_an_image_lost_while_streaming_is_still_reported(monkeypatch):
    def parse(document=None, **kwargs):
        def stream():
            yield Document(page_content="text before the picture", metadata=dict(document.metadata))
            loader_utils._image_description_failures.get().append("model rejected the image")
            yield Document(page_content="text after the picture", metadata=dict(document.metadata))
        return stream()

    def install_parser(toolkit):
        monkeypatch.setattr("elitea_sdk.tools.base_indexer_toolkit.process_document_by_type", parse)

    toolkit, _ = index_through_real_pipeline(
        monkeypatch, make_files("deck.pptx"), configure=install_parser, **STAGED_SERIAL)

    stats = toolkit.get_indexing_stats()
    assert stats.images_not_described == {"deck.pptx: lost image 1"}
    assert stats.image_description_errors == {"model rejected the image": {"deck.pptx": None}}


def index_with_parse_error(monkeypatch, chunks_before_error, failing="huge.xlsx", documents=None, dependents=None,
                           mode=STAGED_SERIAL):
    def parse(document=None, **kwargs):
        name = document.metadata["name"]

        def stream():
            healthy = name != failing
            for index in range(1 if healthy else chunks_before_error):
                yield Document(page_content=f"{name} chunk {index}", metadata=dict(document.metadata))
            if not healthy:
                yield Document(page_content=f"Error during content parsing for file {name}:\nMemoryError",
                               metadata=dict(document.metadata))
        return stream()

    def install_parser(toolkit):
        monkeypatch.setattr("elitea_sdk.tools.base_indexer_toolkit.process_document_by_type", parse)

    return index_through_real_pipeline(
        monkeypatch, documents or make_files(failing), configure=install_parser,
        dependents=dependents, **mode)


def rows_deleted_at_promote(toolkit):
    _, damaged_row_ids, _ = toolkit._assemble_promote_sets()
    return damaged_row_ids


def test_a_parse_that_fails_after_handing_over_chunks_is_rolled_back(monkeypatch):
    toolkit, outcome = index_with_parse_error(monkeypatch, chunks_before_error=5)

    assert len(rows_deleted_at_promote(toolkit)) == 5
    assert rows_kept(toolkit) == 0
    assert failed_items(outcome) == ["huge.xlsx"]
    assert outcome["report"]["totals"]["indexed"] == 0


def test_a_parse_that_fails_before_any_chunk_keeps_its_previous_failure_handling(monkeypatch):
    toolkit, outcome = index_with_parse_error(monkeypatch, chunks_before_error=0)

    assert rows_deleted_at_promote(toolkit) == []
    assert "huge.xlsx" in toolkit._index_run.explicitly_failed_keys
    assert failed_items(outcome) == ["huge.xlsx"]


def test_an_attachment_that_fails_after_handing_over_chunks_is_dropped_and_its_page_stays_indexed(monkeypatch):
    def run(mode):
        return index_with_parse_error(
            monkeypatch, chunks_before_error=3, failing="att.xlsx", documents=make_files("page.docx"),
            dependents={"page.docx": make_files("att.xlsx", dependent=("att.xlsx",))}, mode=mode)

    streamed, streamed_outcome = run(STAGED_SERIAL)
    materialised, materialised_outcome = run(STAGED_MATERIALISED)

    assert len(rows_deleted_at_promote(streamed)) == 3
    assert streamed._index_run.pipeline_failed_keys == set()
    assert rows_kept(streamed) == rows_kept(materialised) == 1
    assert report_summary(streamed, streamed_outcome) == report_summary(materialised, materialised_outcome)
    assert streamed_outcome["report"]["totals"]["indexed"] == 1
    assert streamed.written[-1]["updated"] == 1


@pytest.mark.parametrize("workers", [1, 2])
def test_an_unstaged_run_writes_nothing_from_an_attachment_that_failed_mid_parse(monkeypatch, workers):
    toolkit, outcome = index_with_parse_error(
        monkeypatch, chunks_before_error=3, failing="att.xlsx", documents=make_files("page.docx"),
        dependents={"page.docx": make_files("att.xlsx", dependent=("att.xlsx",))},
        mode={"workers": workers, "staging": False})

    assert toolkit._index_run.chunks_written == 1
    assert outcome["report"]["totals"]["indexed"] == 1


def test_an_attachment_that_fails_before_any_chunk_leaves_its_parent_indexed(monkeypatch):
    toolkit, outcome = index_with_parse_error(
        monkeypatch, chunks_before_error=0, failing="att.xlsx", documents=make_files("page.docx"),
        dependents={"page.docx": make_files("att.xlsx", dependent=("att.xlsx",))})

    assert rows_deleted_at_promote(toolkit) == []


def test_the_flush_buffer_is_written_once_its_text_reaches_the_character_budget(monkeypatch):
    flushed_batches = []
    record_flushed_chunk = ParsingStagingToolkit._record_flushed_chunk

    def record_batch(self, run, chunk_keys, inserted_ids):
        flushed_batches.append(len(chunk_keys))
        return record_flushed_chunk(self, run, chunk_keys, inserted_ids)

    def parse(document=None, **kwargs):
        return iter([Document(page_content="x" * 400, metadata=dict(document.metadata)) for _ in range(6)])

    def configure(toolkit):
        object.__setattr__(toolkit, "max_docs_per_add", 20)
        object.__setattr__(toolkit, "max_chars_per_add", 1_000)
        monkeypatch.setattr(ParsingStagingToolkit, "_record_flushed_chunk", record_batch)
        monkeypatch.setattr("elitea_sdk.tools.base_indexer_toolkit.process_document_by_type", parse)

    index_through_real_pipeline(monkeypatch, make_files("wide.xlsx"), configure=configure, **STAGED_SERIAL)

    assert flushed_batches == [3, 3]


def test_an_image_lost_in_a_file_that_then_failed_is_not_reported(monkeypatch):
    def parse(document=None, **kwargs):
        def stream():
            yield Document(page_content="text before the picture", metadata=dict(document.metadata))
            loader_utils._image_description_failures.get().append("model rejected the image")
            yield Document(page_content="Error during content parsing for file deck.pptx:\nbroken slide",
                           metadata=dict(document.metadata))
        return stream()

    def install_parser(toolkit):
        monkeypatch.setattr("elitea_sdk.tools.base_indexer_toolkit.process_document_by_type", parse)

    toolkit, _ = index_through_real_pipeline(
        monkeypatch, make_files("deck.pptx"), configure=install_parser, **STAGED_SERIAL)

    assert toolkit.get_indexing_stats().images_not_described == set()


@pytest.mark.parametrize("mode", PIPELINE_MODES)
def test_stored_chunks_carry_no_parse_bookkeeping(monkeypatch, mode):
    produced = []

    def parse(document=None, **kwargs):
        def stream():
            for index in range(3):
                chunk = Document(page_content=f"chunk {index}", metadata=dict(document.metadata))
                produced.append(chunk)
                yield chunk
        return stream()

    def install_parser(toolkit):
        monkeypatch.setattr("elitea_sdk.tools.base_indexer_toolkit.process_document_by_type", parse)

    index_through_real_pipeline(monkeypatch, make_files("a.xlsx", "b.xlsx"), configure=install_parser, **mode)

    assert len(produced) == 6
    assert [chunk for chunk in produced if PARSE_SOURCE_META_KEY in chunk.metadata] == []


def test_chunker_output_handed_over_before_a_retry_ends_like_a_materialised_run(monkeypatch):
    def run(mode):
        page = make_page("page-1", "page.md")
        note = Document(page_content="note body", metadata={
            "id": "note-1", "name": "note.md", "updated_on": "1", DEPENDENT_DOC_META_KEY: True})
        failed_once = []

        def chunker_failing_once_on_the_page(file_content_generator=None, config=None):
            for document in file_content_generator:
                if not document.metadata.get(DEPENDENT_DOC_META_KEY) and not failed_once:
                    failed_once.append(True)
                    raise RuntimeError("transient embeddings failure")
                yield Document(page_content=f"{document.page_content} part 1", metadata=document.metadata)
                yield Document(page_content=f"{document.page_content} part 2", metadata=document.metadata)

        def configure(toolkit):
            monkeypatch.setitem(chunkers.__all__, "flaky", chunker_failing_once_on_the_page)
            monkeypatch.setattr(ParsingStagingToolkit, "_process_document", lambda self, document: iter([note]))

        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, [page], configure=configure, chunking_tool="flaky", **mode)
        return bool(failed_once), report_summary(toolkit, outcome), rows_kept(toolkit), note.metadata.get("collection")

    streamed = run(STAGED_SERIAL)

    assert streamed[0] is True
    assert streamed == run(STAGED_MATERIALISED)


def test_a_retried_attempt_spanning_several_keys_discards_every_row_it_handed_over(monkeypatch):
    def run(mode):
        page = Document(page_content="page body", metadata={"name": "page.md", "updated_on": "1"})
        notes = [Document(page_content=f"{name} body", metadata={"id": name, "name": name, "updated_on": "1"})
                 for name in ("note-1", "note-2")]
        failed_once = []

        def chunker_failing_once_on_the_page(file_content_generator=None, config=None):
            for document in file_content_generator:
                if not document.metadata.get(DEPENDENT_DOC_META_KEY) and not failed_once:
                    failed_once.append(True)
                    raise RuntimeError("transient embeddings failure")
                yield Document(page_content=document.page_content, metadata=document.metadata)

        def rejects_every_note_2_batch(document):
            return document.metadata.get("id") == "note-2"

        def configure(toolkit):
            monkeypatch.setitem(chunkers.__all__, "flaky", chunker_failing_once_on_the_page)
            monkeypatch.setattr(ParsingStagingToolkit, "_process_document", lambda self, document: iter(notes))

        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, [page], configure=configure, chunking_tool="flaky",
            rejects=rejects_every_note_2_batch, **mode)
        assert toolkit._index_run.damaged_keys == {"note-2"}
        return report_summary(toolkit, outcome), rows_kept(toolkit)

    streamed = run(STAGED_SERIAL)

    assert streamed == run(STAGED_MATERIALISED)


def index_with_a_flaky_page_chunker(monkeypatch, mode, page, note_chunks, page_failures):
    note = Document(page_content="note body", metadata={
        "id": "note-1", "name": "note.md", "updated_on": "1", DEPENDENT_DOC_META_KEY: True})
    failures = []

    def chunker(file_content_generator=None, config=None):
        for document in file_content_generator:
            if not document.metadata.get(DEPENDENT_DOC_META_KEY):
                if len(failures) < page_failures:
                    failures.append(True)
                    raise RuntimeError("transient embeddings failure")
                yield Document(page_content=document.page_content, metadata=document.metadata)
                continue
            for content in note_chunks:
                yield Document(page_content=content, metadata=document.metadata)

    def configure(toolkit):
        monkeypatch.setitem(chunkers.__all__, "flaky", chunker)
        monkeypatch.setattr(ParsingStagingToolkit, "_process_document", lambda self, document: iter([note]))

    toolkit, outcome = index_through_real_pipeline(
        monkeypatch, [page], configure=configure, chunking_tool="flaky", **mode)
    return report_summary(toolkit, outcome), rows_kept(toolkit)


def test_an_empty_chunk_of_a_discarded_attempt_does_not_lower_the_chunk_count(monkeypatch):
    def run(mode):
        return index_with_a_flaky_page_chunker(
            monkeypatch, mode, make_page("page-1", "page.md"), note_chunks=["", "note body"], page_failures=1)

    assert run(STAGED_SERIAL) == run(STAGED_MATERIALISED)


def test_a_retry_that_also_fails_leaves_no_rows_for_an_idless_document(monkeypatch):
    def run(mode):
        page = Document(page_content="page body", metadata={"name": "page.md", "updated_on": "1"})
        return index_with_a_flaky_page_chunker(monkeypatch, mode, page, note_chunks=["note body"], page_failures=2)

    streamed = run(STAGED_SERIAL)

    assert streamed[1] == 0
    assert streamed == run(STAGED_MATERIALISED)


def test_buffered_chunks_of_a_discarded_source_are_never_embedded(monkeypatch):
    stored = []

    def parse(document=None, **kwargs):
        name = document.metadata["name"]

        def stream():
            if name == "att.xlsx":
                for index in range(3):
                    yield Document(page_content=f"att.xlsx chunk {index}", metadata=dict(document.metadata))
                yield Document(page_content="Error during content parsing for file att.xlsx:\nchartsheet",
                               metadata=dict(document.metadata))
                return
            yield Document(page_content=f"{name} body", metadata=dict(document.metadata))
        return stream()

    def configure(toolkit):
        object.__setattr__(toolkit, "max_docs_per_add", 20)
        monkeypatch.setattr("elitea_sdk.tools.base_indexer_toolkit.process_document_by_type", parse)

    def store_sees(document):
        stored.append(document.page_content)
        return False

    toolkit, outcome = index_through_real_pipeline(
        monkeypatch, make_files("page.docx"), configure=configure, rejects=store_sees,
        dependents={"page.docx": make_files("att.xlsx", dependent=("att.xlsx",))}, **STAGED_SERIAL)

    assert stored == ["page.docx body"]
    assert toolkit._index_run.pending_chunk_counts == {"page.docx": 0}
    assert toolkit._index_run.chunks_written == 1
    assert outcome["report"]["totals"]["indexed"] == 1


def test_chunks_of_a_discarded_source_never_claim_a_reusable_row(monkeypatch):
    offered_for_reuse = []

    def parse(document=None, **kwargs):
        name = document.metadata["name"]

        def stream():
            yield Document(page_content=f"{name} chunk", metadata=dict(document.metadata))
            if name == "att.xlsx":
                yield Document(page_content="Error during content parsing for file att.xlsx:\nchartsheet",
                               metadata=dict(document.metadata))
        return stream()

    def configure(toolkit):
        monkeypatch.setattr("elitea_sdk.tools.base_indexer_toolkit.process_document_by_type", parse)
        monkeypatch.setattr(ParsingStagingToolkit, "_claim_adopted_row",
                            lambda self, document: offered_for_reuse.append(document.page_content))

    index_through_real_pipeline(
        monkeypatch, make_files("page.docx"), configure=configure,
        dependents={"page.docx": make_files("att.xlsx", dependent=("att.xlsx",))}, **STAGED_MATERIALISED)

    assert offered_for_reuse == ["page.docx chunk"]


def test_a_rejected_write_of_a_source_discarded_later_leaves_its_page_indexed(monkeypatch):
    def run(mode):
        def parse(document=None, **kwargs):
            name = document.metadata["name"]

            def stream():
                if name == "att.xlsx":
                    for index in range(3):
                        yield Document(page_content=f"att.xlsx chunk {index}", metadata=dict(document.metadata))
                    yield Document(page_content="Error during content parsing for file att.xlsx:\nchartsheet",
                                   metadata=dict(document.metadata))
                    return
                yield Document(page_content=f"{name} body", metadata=dict(document.metadata))
            return stream()

        def configure(toolkit):
            monkeypatch.setattr("elitea_sdk.tools.base_indexer_toolkit.process_document_by_type", parse)

        toolkit, outcome = index_through_real_pipeline(
            monkeypatch, make_files("page.docx"), configure=configure,
            rejects=lambda document: document.page_content == "att.xlsx chunk 0",
            dependents={"page.docx": make_files("att.xlsx", dependent=("att.xlsx",))}, **mode)
        return toolkit, outcome

    streamed, streamed_outcome = run(STAGED_SERIAL)
    materialised, materialised_outcome = run(STAGED_MATERIALISED)

    assert_terminal_state(streamed, streamed_outcome, IndexingStatus.OK)
    assert streamed._index_run.damaged_keys == set()
    assert streamed._index_run.pending_chunk_counts == {"page.docx": 0}
    assert streamed_outcome["report"]["totals"]["indexed"] == 1
    assert report_summary(streamed, streamed_outcome) == report_summary(materialised, materialised_outcome)
    assert rows_kept(streamed) == rows_kept(materialised) == 1


def test_a_rejected_write_mixing_sources_counts_only_the_chunks_that_were_kept(monkeypatch):
    def parse(document=None, **kwargs):
        name = document.metadata["name"]

        def stream():
            if name == "att.xlsx":
                for index in range(3):
                    yield Document(page_content=f"att.xlsx chunk {index}", metadata=dict(document.metadata))
                yield Document(page_content="Error during content parsing for file att.xlsx:\nchartsheet",
                               metadata=dict(document.metadata))
                return
            yield Document(page_content=f"{name} body", metadata=dict(document.metadata))
        return stream()

    def configure(toolkit):
        object.__setattr__(toolkit, "max_docs_per_add", 2)
        monkeypatch.setattr("elitea_sdk.tools.base_indexer_toolkit.process_document_by_type", parse)

    page = make_files("page.docx")[0]
    page.metadata.pop("id")
    toolkit, outcome = index_through_real_pipeline(
        monkeypatch, [page], configure=configure,
        rejects=lambda document: document.page_content == "att.xlsx chunk 0",
        dependents={"page.docx": make_files("ok.docx", "att.xlsx", dependent=("ok.docx", "att.xlsx"))},
        **STAGED_SERIAL)

    assert toolkit._index_run.damaged_keys == {"ok.docx"}
    assert outcome["report"]["errors"] == ["ok.docx: the store rejected the batch"]
    assert toolkit.written[-1]["updated"] == 1
