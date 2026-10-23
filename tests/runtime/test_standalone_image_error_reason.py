"""A standalone image whose description fails must report the model's reason, not a bare failure.

Re-breaks if the indexer again records image-description causes only for documents that kept
other content: a standalone image keeps nothing, so its cause was dropped and the run report
listed it as "Could not be processed" with no errors. Also re-breaks if a document that lost an
embedded image and then failed for another reason is reported as failing because of the image,
or if a chunker that maps to no file type crashes the run instead of failing its one document.
"""

import io

from langchain_core.documents import Document
from PIL import Image

from elitea_sdk.runtime.langchain.document_loaders.utils import perform_llm_prediction_for_image_bytes
from elitea_sdk.runtime.utils.utils import IndexerKeywords
from tests.tools.test_indexing_report import (  # noqa: F401
    RecordingToolkit,
    indexing_toolkit,
    run_index_data,
    toolkit,
)

NOT_A_VISION_MODEL = "model 'text-only' does not support image input"
PNG_CONFIG = {".png": {"use_llm": True, "use_default_prompt": True, "prompt": "", "max_tokens": 512}}


class _TextOnlyLlm:
    model_name = "text-only"

    def invoke(self, messages):
        raise RuntimeError(NOT_A_VISION_MODEL)


def _png_document(name):
    buffer = io.BytesIO()
    Image.new("RGB", (340, 90), "white").save(buffer, format="PNG")
    return Document(page_content="", metadata={
        "name": name,
        "id": name,
        IndexerKeywords.CONTENT_IN_BYTES.value: buffer.getvalue(),
        IndexerKeywords.CONTENT_FILE_NAME.value: name,
    })


def _index_standalone_images(indexing_toolkit, monkeypatch, names):
    object.__setattr__(indexing_toolkit, "llm", _TextOnlyLlm())
    object.__setattr__(indexing_toolkit, "embeddings", None)
    monkeypatch.setattr(RecordingToolkit, "key_fn", lambda self, document: document.metadata["id"])

    def parse_images(toolkit_instance):
        documents = iter([_png_document(name) for name in names])
        list(toolkit_instance._apply_loaders_chunkers(documents, chunking_config=dict(PNG_CONFIG)))

    return run_index_data(indexing_toolkit, monkeypatch, [object()], tracker=parse_images)


class TestStandaloneImageFailure:
    def test_the_report_names_the_model_and_the_reason(self, indexing_toolkit, monkeypatch):
        result = _index_standalone_images(indexing_toolkit, monkeypatch, ["shot.png"])

        assert result["report"]["errors"] == [
            f"shot.png: Model 'text-only' could not describe images: {NOT_A_VISION_MODEL}",
        ]

    def test_images_failing_for_one_reason_share_one_error(self, indexing_toolkit, monkeypatch):
        result = _index_standalone_images(indexing_toolkit, monkeypatch, ["a.png", "b.png"])

        assert result["report"]["errors"] == [
            f"a.png, b.png: Model 'text-only' could not describe images: {NOT_A_VISION_MODEL}",
        ]

    def test_the_image_counts_as_a_failed_file_not_a_lost_attachment(self, indexing_toolkit, monkeypatch):
        _index_standalone_images(indexing_toolkit, monkeypatch, ["shot.png"])

        stats = indexing_toolkit.get_indexing_stats()
        assert stats.runtime_skipped_error == {"shot.png"}
        assert stats.images_not_described == set()


class TestDocumentFailingForAnotherReason:
    def test_a_swallowed_image_error_is_not_blamed_for_the_crash(self, indexing_toolkit, monkeypatch):
        def deck_that_loses_an_image_then_crashes(content, extension_source, document=None, **kwargs):
            try:
                perform_llm_prediction_for_image_bytes(b"image", _TextOnlyLlm(), "describe")
            except RuntimeError:
                pass
            yield Document(page_content=f"Error during content parsing for file {extension_source}:\ncorrupt slide",
                           metadata={**document.metadata, "chunk_id": 1})

        monkeypatch.setattr("elitea_sdk.tools.base_indexer_toolkit.process_document_by_type",
                            deck_that_loses_an_image_then_crashes)

        result = _index_standalone_images(indexing_toolkit, monkeypatch, ["deck.pptx"])

        assert result["report"]["errors"] == []
        assert indexing_toolkit.get_indexing_stats().runtime_skipped_error == {"deck.pptx"}


class TestChunkerWithoutAFileType:
    def test_a_chunker_with_no_file_type_fails_only_its_document(self, indexing_toolkit, monkeypatch):
        object.__setattr__(indexing_toolkit, "llm", _TextOnlyLlm())
        object.__setattr__(indexing_toolkit, "embeddings", None)
        monkeypatch.setattr(RecordingToolkit, "key_fn", lambda self, document: document.metadata["id"])
        document = Document(page_content="", metadata={
            "name": "stats", "id": "stats", IndexerKeywords.CONTENT_IN_BYTES.value: b"1,2,3"})

        def parse_with_statistical_chunker(toolkit_instance):
            list(toolkit_instance._apply_loaders_chunkers(iter([document]), chunking_tool="statistical"))

        result = run_index_data(indexing_toolkit, monkeypatch, [object()], tracker=parse_with_statistical_chunker)

        assert result["report"]["errors"] == []
        assert indexing_toolkit.get_indexing_stats().runtime_skipped_error == {"stats"}
