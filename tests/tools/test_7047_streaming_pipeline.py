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

from elitea_sdk.runtime.langchain.document_loaders.utils import _image_description_failures
from elitea_sdk.tools.base_indexer_toolkit import IndexingStatus
from tests.tools.test_6585_document_failure_isolation import (
    PIPELINE_MODES,
    ParsingStagingToolkit,
    assert_terminal_state,
    failed_items,
    index_through_real_pipeline,
    make_files,
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


def test_a_file_failing_after_its_chunks_were_handed_over_is_failed_not_retried(monkeypatch):
    probe = StreamProbe({"huge.xlsx": LARGE}, fail_after={"huge.xlsx": 5}, fail_on_calls=(1, 2))

    toolkit, outcome = index_streamed(monkeypatch, STAGED_SERIAL, probe)

    assert probe.calls["huge.xlsx"] == 1
    assert_terminal_state(toolkit, outcome, IndexingStatus.PARTLY_INDEXED)
    assert failed_items(outcome) == ["huge.xlsx"]
    assert "huge.xlsx" in toolkit._index_run.pipeline_failed_keys


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
            _image_description_failures.get().append("model rejected the image")
            yield Document(page_content="text after the picture", metadata=dict(document.metadata))
        return stream()

    def install_parser(toolkit):
        monkeypatch.setattr("elitea_sdk.tools.base_indexer_toolkit.process_document_by_type", parse)

    toolkit, _ = index_through_real_pipeline(
        monkeypatch, make_files("deck.pptx"), configure=install_parser, **STAGED_SERIAL)

    stats = toolkit.get_indexing_stats()
    assert stats.images_not_described == {"deck.pptx: lost image 1"}
    assert stats.image_description_errors == {"model rejected the image": {"deck.pptx": None}}


def index_with_parse_error(monkeypatch, chunks_before_error, failing="huge.xlsx", documents=None, dependents=None):
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
        dependents=dependents, **STAGED_SERIAL)


def rows_deleted_at_promote(toolkit):
    _, damaged_row_ids, _ = toolkit._assemble_promote_sets()
    return damaged_row_ids


def test_a_parse_that_fails_after_handing_over_chunks_is_rolled_back(monkeypatch):
    toolkit, outcome = index_with_parse_error(monkeypatch, chunks_before_error=5)

    assert len(rows_deleted_at_promote(toolkit)) == 5
    assert failed_items(outcome) == ["huge.xlsx"]


def test_a_parse_that_fails_before_any_chunk_keeps_its_previous_failure_handling(monkeypatch):
    toolkit, outcome = index_with_parse_error(monkeypatch, chunks_before_error=0)

    assert rows_deleted_at_promote(toolkit) == []
    assert "huge.xlsx" in toolkit._index_run.explicitly_failed_keys
    assert failed_items(outcome) == ["huge.xlsx"]


def test_an_attachment_that_fails_after_handing_over_chunks_rolls_back_with_its_parent(monkeypatch):
    toolkit, outcome = index_with_parse_error(
        monkeypatch, chunks_before_error=3, failing="att.xlsx", documents=make_files("page.docx"),
        dependents={"page.docx": make_files("att.xlsx", dependent=("att.xlsx",))})

    assert len(rows_deleted_at_promote(toolkit)) == 4
    assert_terminal_state(toolkit, outcome, IndexingStatus.PARTLY_INDEXED)


def test_an_attachment_that_fails_before_any_chunk_leaves_its_parent_indexed(monkeypatch):
    toolkit, outcome = index_with_parse_error(
        monkeypatch, chunks_before_error=0, failing="att.xlsx", documents=make_files("page.docx"),
        dependents={"page.docx": make_files("att.xlsx", dependent=("att.xlsx",))})

    assert rows_deleted_at_promote(toolkit) == []
