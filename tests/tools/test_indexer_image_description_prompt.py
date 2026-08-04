"""Unit coverage for the image-description prompt reaching dependent documents.

Attachments are emitted by a toolkit as dependent documents carrying raw bytes,
but they are parsed later by ``_apply_loaders_chunkers``. A prompt the toolkit
collected from the user therefore has to be re-supplied at that hop; when it is
dropped, image attachments silently fall back to the built-in prompt while
inline images honour the custom one.
"""

import io
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from PIL import Image

from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit
from elitea_sdk.tools.utils.content_parser import image_processing_prompt

CUSTOM_PROMPT = "Describe this image like a pirate would. Start with 'Ahoy matey!'"


class RecordingLLM:
    """Captures the prompt each vision call was handed."""

    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content[0]["text"])
        return SimpleNamespace(content="a description")


@pytest.fixture
def jpeg_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 10, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _toolkit(llm, image_prompt=None):
    stub = SimpleNamespace(embeddings=None, llm=llm, _image_cache=None, _index_workers=1)
    if image_prompt is not None:
        stub._index_image_description_prompt = image_prompt
    return stub


def _attachment(jpeg_bytes):
    return Document(page_content="", metadata={
        "id": "466::.attachments/screenshot.jpeg",
        IndexerKeywords.CONTENT_FILE_NAME.value: ".jpeg",
        IndexerKeywords.CONTENT_IN_BYTES.value: jpeg_bytes,
    })


def _chunk(toolkit, document):
    return list(BaseIndexerToolkit._apply_loaders_chunkers(
        toolkit, iter([document]), chunking_tool="markdown", chunking_config=None,
    ))


def test_attachment_uses_the_prompt_collected_for_the_run(jpeg_bytes):
    llm = RecordingLLM()
    _chunk(_toolkit(llm, CUSTOM_PROMPT), _attachment(jpeg_bytes))
    assert llm.prompts == [CUSTOM_PROMPT]


def test_attachment_falls_back_to_the_built_in_prompt(jpeg_bytes):
    llm = RecordingLLM()
    _chunk(_toolkit(llm), _attachment(jpeg_bytes))
    assert llm.prompts == [image_processing_prompt]


def test_run_prompt_outranks_a_per_extension_chunking_config_prompt(jpeg_bytes):
    llm = RecordingLLM()
    list(BaseIndexerToolkit._apply_loaders_chunkers(
        _toolkit(llm, CUSTOM_PROMPT), iter([_attachment(jpeg_bytes)]),
        chunking_tool="markdown",
        chunking_config={".jpeg": {"prompt": "per-extension", "use_default_prompt": False}},
    ))
    assert llm.prompts == [CUSTOM_PROMPT]


def test_text_content_is_unaffected_by_the_prompt():
    llm = RecordingLLM()
    document = Document(page_content="", metadata={
        "id": "466",
        IndexerKeywords.CONTENT_IN_BYTES.value: b"# All files\n\nplain markdown\n",
    })
    chunks = _chunk(_toolkit(llm, CUSTOM_PROMPT), document)
    assert llm.prompts == []
    assert "plain markdown" in chunks[0].page_content
