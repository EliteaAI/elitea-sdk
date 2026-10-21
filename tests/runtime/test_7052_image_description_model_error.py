"""When images cannot be described, the run report must say which model failed and why.

Re-breaks if the indexer stops keeping the provider's reason per document, or if index_data stops
adding those reasons, with the model's name, to the report errors.
"""

import io
from types import SimpleNamespace

import fitz
import pytest
from PIL import Image

from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit, IndexingStats, render_report_text
from elitea_sdk.tools.utils.content_parser import process_content_by_type
from tests.tools.test_indexing_report import (  # noqa: F401
    RecordingToolkit,
    indexing_toolkit,
    run_index_data,
    toolkit,
)

NOT_A_VISION_MODEL = "model 'text-only' does not support image input"
PDF_IMAGE_CONFIG = {".pdf": {"extract_images": True, "use_llm": True, "use_default_prompt": True, "prompt": ""}}


class _TextOnlyLlm:
    def __init__(self, **name):
        self.__dict__.update(name)

    def invoke(self, messages):
        raise RuntimeError(NOT_A_VISION_MODEL)


def _pdf_with_image():
    image = Image.new("RGB", (400, 120), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Plain body text")
    page.insert_image(fitz.Rect(72, 120, 472, 240), stream=buffer.getvalue())
    return pdf.tobytes()


def _parse_pdf(llm):
    return lambda: list(process_content_by_type(_pdf_with_image(), "spec.pdf", llm=llm,
                                                chunking_config=PDF_IMAGE_CONFIG))


def _indexer_with(llm):
    stats = IndexingStats()
    return SimpleNamespace(llm=llm, get_indexing_stats=lambda: stats, _init_indexing_stats=lambda: stats)


class TestTheReasonIsKeptPerDocument:
    def test_the_provider_reason_is_recorded_against_the_document(self):
        indexer = _indexer_with(_TextOnlyLlm(model_name="text-only"))

        BaseIndexerToolkit._parse_tracking_lost_images(indexer, "spec.pdf", _parse_pdf(indexer.llm))

        assert indexer.get_indexing_stats().image_description_errors == {NOT_A_VISION_MODEL: {"spec.pdf": None}}

    def test_documents_failing_for_the_same_reason_share_one_error(self):
        indexer = _indexer_with(_TextOnlyLlm(model_name="text-only"))

        for label in ("a.pdf", "b.pdf"):
            BaseIndexerToolkit._parse_tracking_lost_images(indexer, label, _parse_pdf(indexer.llm))

        assert BaseIndexerToolkit._render_image_description_errors(indexer) == [
            f"a.pdf, b.pdf: Model 'text-only' could not describe images: {NOT_A_VISION_MODEL}",
        ]


class TestTheErrorNamesTheModel:
    @pytest.mark.parametrize("name", [{"model_name": "text-only"}, {"model": "text-only"}],
                             ids=["openai-style", "anthropic-style"])
    def test_the_model_is_named_whichever_attribute_carries_it(self, name):
        indexer = _indexer_with(_TextOnlyLlm(**name))
        BaseIndexerToolkit._parse_tracking_lost_images(indexer, "spec.pdf", _parse_pdf(indexer.llm))

        assert BaseIndexerToolkit._render_image_description_errors(indexer) == [
            f"spec.pdf: Model 'text-only' could not describe images: {NOT_A_VISION_MODEL}",
        ]

    def test_an_unnamed_model_still_reports_the_reason(self):
        indexer = _indexer_with(_TextOnlyLlm())
        BaseIndexerToolkit._parse_tracking_lost_images(indexer, "spec.pdf", _parse_pdf(indexer.llm))

        assert BaseIndexerToolkit._render_image_description_errors(indexer) == [
            f"spec.pdf: The model could not describe images: {NOT_A_VISION_MODEL}",
        ]

    def test_a_run_without_lost_images_adds_no_error(self):
        assert BaseIndexerToolkit._render_image_description_errors(_indexer_with(_TextOnlyLlm())) == []


class TestIndexDataReportsTheModelError:
    def test_the_run_report_and_message_carry_the_model_error(self, indexing_toolkit, monkeypatch):
        object.__setattr__(indexing_toolkit, "llm", _TextOnlyLlm(model_name="text-only"))

        def lose_an_image(toolkit_instance):
            toolkit_instance._parse_tracking_lost_images("spec.pdf", _parse_pdf(toolkit_instance.llm))

        result = run_index_data(indexing_toolkit, monkeypatch, [object()], tracker=lose_an_image)

        expected = f"spec.pdf: Model 'text-only' could not describe images: {NOT_A_VISION_MODEL}"
        assert result["report"]["errors"] == [expected]
        assert expected in render_report_text(result["report"])
