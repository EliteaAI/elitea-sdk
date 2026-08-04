"""Unit coverage for the image-description prompt reaching dependent documents.

Attachments are emitted by a toolkit as dependent documents carrying raw bytes,
but they are parsed later by ``_apply_loaders_chunkers``. A prompt the toolkit
collected from the user therefore has to be re-supplied at that hop; when it is
dropped, image attachments silently fall back to the built-in prompt while
inline images on the same page honour the custom one.
"""

import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from PIL import Image

from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.ado.wiki.ado_wrapper import AzureDevOpsApiWrapper as WikiApiWrapper
from elitea_sdk.tools.ado.work_item.ado_wrapper import AzureDevOpsApiWrapper as WorkItemApiWrapper
from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit
from elitea_sdk.tools.utils.content_parser import image_processing_prompt

CUSTOM_PROMPT = "Describe this image like a pirate would. Start with 'Ahoy matey!'"

# Toolkits that never collect a prompt leave the attribute unset, so the reader
# has to tolerate that shape as well as the None an ADO indexing run leaves.
ATTRIBUTE_UNSET = object()


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


def _toolkit(llm, image_prompt=ATTRIBUTE_UNSET):
    stub = SimpleNamespace(embeddings=None, llm=llm, _image_cache=None, _index_workers=1)
    if image_prompt is not ATTRIBUTE_UNSET:
        stub._index_image_description_prompt = image_prompt
    return stub


def _attachment(jpeg_bytes):
    return Document(page_content="", metadata={
        "id": "466::.attachments/screenshot.jpeg",
        IndexerKeywords.CONTENT_FILE_NAME.value: ".jpeg",
        IndexerKeywords.CONTENT_IN_BYTES.value: jpeg_bytes,
    })


def _chunk(toolkit, document, chunking_tool=None, chunking_config=None):
    return list(BaseIndexerToolkit._apply_loaders_chunkers(
        toolkit, iter([document]), chunking_tool=chunking_tool, chunking_config=chunking_config,
    ))


def test_attachment_uses_the_prompt_collected_for_the_run(jpeg_bytes):
    llm = RecordingLLM()
    _chunk(_toolkit(llm, CUSTOM_PROMPT), _attachment(jpeg_bytes))
    assert llm.prompts == [CUSTOM_PROMPT]


@pytest.mark.parametrize("stashed", [None, "", ATTRIBUTE_UNSET],
                         ids=["prompt_not_supplied", "prompt_blank", "toolkit_collects_no_prompt"])
def test_attachment_falls_back_to_the_built_in_prompt(jpeg_bytes, stashed):
    llm = RecordingLLM()
    _chunk(_toolkit(llm, stashed), _attachment(jpeg_bytes))
    assert llm.prompts == [image_processing_prompt]


def test_run_prompt_outranks_a_per_extension_chunking_config_prompt(jpeg_bytes):
    llm = RecordingLLM()
    _chunk(
        _toolkit(llm, CUSTOM_PROMPT), _attachment(jpeg_bytes),
        chunking_config={".jpeg": {"prompt": "per-extension", "use_default_prompt": False}},
    )
    assert llm.prompts == [CUSTOM_PROMPT]


def test_text_content_is_unaffected_by_the_prompt():
    llm = RecordingLLM()
    document = Document(page_content="", metadata={
        "id": "466",
        IndexerKeywords.CONTENT_IN_BYTES.value: b"# All files\n\nplain markdown\n",
    })
    chunks = _chunk(_toolkit(llm, CUSTOM_PROMPT), document, chunking_tool="markdown")
    assert llm.prompts == []
    assert "plain markdown" in chunks[0].page_content


class TestToolkitsStashThePromptTheIndexerReads:
    """The toolkit writes the prompt and the base indexer reads it back by name,
    across files. Renaming one side has to fail here rather than in production.
    """

    def test_ado_wiki(self):
        wrapper = WikiApiWrapper.model_construct()
        with patch.object(WikiApiWrapper, "_iter_wiki_pages", return_value=[]):
            list(wrapper._base_loader(
                wiki_identifier="Samvel_Simonyan.wiki",
                include_attachments=True,
                image_description_prompt=CUSTOM_PROMPT,
            ))
        assert getattr(wrapper, "_index_image_description_prompt", None) == CUSTOM_PROMPT

    def test_ado_boards(self):
        wrapper = WorkItemApiWrapper.model_construct()
        client = SimpleNamespace(
            query_by_wiql=lambda wiql: SimpleNamespace(work_items=[], work_item_relations=[])
        )
        with patch.object(WorkItemApiWrapper, "_client", client, create=True):
            list(wrapper._base_loader(
                wiql="SELECT [System.Id] FROM workitems",
                image_description_prompt=CUSTOM_PROMPT,
            ))
        assert getattr(wrapper, "_index_image_description_prompt", None) == CUSTOM_PROMPT
