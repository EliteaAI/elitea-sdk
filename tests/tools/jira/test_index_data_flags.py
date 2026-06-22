"""
Tests for the `process_images` / `include_comments` gates on the Jira indexer.

Regression coverage for the perf bug where `_extend_data` ran the
AttachmentResolver (3+ Jira REST calls) and vision-LLM image substitution for
every issue, even when no caller had opted in. With the gate in place, the hot
path must be a plain text passthrough unless:

  process_images=True AND self.llm is not None AND content has !ref! markup

Also pins the public surface of `_index_tool_params` so the new
`include_comments` and `process_images` flags stay exposed with their defaults.
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.jira.api_wrapper import JiraApiWrapper


def _make_wrapper(*, process_images: bool, llm, include_comments: bool = True) -> JiraApiWrapper:
    """Build a wrapper bypassing validators; `_extend_data` only needs these attrs."""
    wrapper = JiraApiWrapper.model_construct()
    wrapper._process_images = process_images
    wrapper._include_comments = include_comments
    wrapper._chunking_tool = "markdown"
    wrapper._client = MagicMock(name="jira_client")
    # `llm` is a real model field — assign through the attribute, not model_construct
    object.__setattr__(wrapper, "llm", llm)
    return wrapper


def _doc(content: str, issue_key: str = "PROJ-1") -> Document:
    return Document(page_content=content, metadata={"issue_key": issue_key})


class TestExtendDataImageGate:
    """`_extend_data` must skip resolver/LLM work unless explicitly opted in."""

    def test_skips_resolver_when_process_images_false(self):
        wrapper = _make_wrapper(process_images=False, llm=MagicMock(name="llm"))
        doc = _doc("desc with !image.png! marker")

        with patch("elitea_sdk.tools.jira.api_wrapper.AttachmentResolver") as resolver_cls, \
             patch.object(JiraApiWrapper, "process_image_match") as match_fn:
            list(wrapper._extend_data(iter([doc])))

        resolver_cls.assert_not_called()
        match_fn.assert_not_called()

    def test_skips_resolver_when_llm_missing_even_if_flag_true(self):
        # process_images=True but no LLM configured — must not attempt vision calls.
        wrapper = _make_wrapper(process_images=True, llm=None)
        doc = _doc("desc with !image.png! marker")

        with patch("elitea_sdk.tools.jira.api_wrapper.AttachmentResolver") as resolver_cls, \
             patch.object(JiraApiWrapper, "process_image_match") as match_fn:
            list(wrapper._extend_data(iter([doc])))

        resolver_cls.assert_not_called()
        match_fn.assert_not_called()

    def test_skips_resolver_when_no_image_markup(self):
        # Even with the gate fully open, plain text must not trigger Jira REST calls.
        wrapper = _make_wrapper(process_images=True, llm=MagicMock(name="llm"))
        doc = _doc("plain description with no images at all")

        with patch("elitea_sdk.tools.jira.api_wrapper.AttachmentResolver") as resolver_cls:
            list(wrapper._extend_data(iter([doc])))

        resolver_cls.assert_not_called()

    def test_runs_resolver_when_flag_llm_and_markup_all_present(self):
        wrapper = _make_wrapper(process_images=True, llm=MagicMock(name="llm"))
        doc = _doc("see !screenshot.png! for details", issue_key="PROJ-42")

        with patch("elitea_sdk.tools.jira.api_wrapper.AttachmentResolver") as resolver_cls, \
             patch.object(JiraApiWrapper, "process_image_match", return_value="[Image desc]") as match_fn:
            list(wrapper._extend_data(iter([doc])))

        resolver_cls.assert_called_once_with(wrapper._client, "PROJ-42")
        assert match_fn.called

    def test_passthrough_preserves_content_and_sets_content_in_bytes(self):
        wrapper = _make_wrapper(process_images=False, llm=None)
        original = "untouched description with !x.png! marker"
        doc = _doc(original)

        result = list(wrapper._extend_data(iter([doc])))

        assert len(result) == 1
        meta = result[0].metadata
        # Markup must stay as-is when gate is closed; chunker still gets the bytes.
        assert meta[IndexerKeywords.CONTENT_IN_BYTES.value].decode("utf-8") == original
        assert meta[IndexerKeywords.CONTENT_FILE_NAME.value].startswith("base_doc")

    def test_default_process_images_is_off_when_attribute_missing(self):
        # Belt-and-suspenders: even if `_process_images` was never set on the
        # instance (older code paths), the gate must default closed.
        wrapper = JiraApiWrapper.model_construct()
        wrapper._chunking_tool = "markdown"
        wrapper._client = MagicMock()
        object.__setattr__(wrapper, "llm", MagicMock(name="llm"))
        doc = _doc("desc with !image.png!")

        with patch("elitea_sdk.tools.jira.api_wrapper.AttachmentResolver") as resolver_cls:
            list(wrapper._extend_data(iter([doc])))

        resolver_cls.assert_not_called()


class TestProcessDocumentCommentsGate:
    """Comments path must thread `process_images` through, gated by LLM presence."""

    def _wire_for_comments(self, wrapper):
        # `_process_document` calls `self._get_client()` and then `client.issue` for
        # attachments; we disable attachments and stub out the client.
        wrapper._include_attachments = False
        wrapper._skipped_attachment_extensions = []
        wrapper.base_url = "https://jira.example"
        wrapper._get_client = lambda: wrapper._client

    def test_comments_skipped_entirely_when_include_comments_false(self):
        wrapper = _make_wrapper(process_images=True, llm=MagicMock(name="llm"), include_comments=False)
        self._wire_for_comments(wrapper)
        base = Document(page_content="x", metadata={"issue_key": "PROJ-1", "id": "1"})

        with patch.object(JiraApiWrapper, "get_processed_comments_list_with_image_description") as get_comments:
            list(wrapper._process_document(base))

        get_comments.assert_not_called()

    def test_comments_passes_process_images_false_when_flag_off(self):
        wrapper = _make_wrapper(process_images=False, llm=MagicMock(name="llm"))
        self._wire_for_comments(wrapper)
        base = Document(page_content="x", metadata={"issue_key": "PROJ-1", "id": "1"})

        with patch.object(JiraApiWrapper, "get_processed_comments_list_with_image_description",
                          return_value=[]) as get_comments:
            list(wrapper._process_document(base))

        get_comments.assert_called_once()
        _, kwargs = get_comments.call_args
        assert kwargs.get("process_images") is False

    def test_comments_passes_process_images_false_when_llm_missing(self):
        # Flag on but LLM None — still must be False so vision path is skipped.
        wrapper = _make_wrapper(process_images=True, llm=None)
        self._wire_for_comments(wrapper)
        base = Document(page_content="x", metadata={"issue_key": "PROJ-1", "id": "1"})

        with patch.object(JiraApiWrapper, "get_processed_comments_list_with_image_description",
                          return_value=[]) as get_comments:
            list(wrapper._process_document(base))

        _, kwargs = get_comments.call_args
        assert kwargs.get("process_images") is False

    def test_comments_passes_process_images_true_when_flag_and_llm_set(self):
        wrapper = _make_wrapper(process_images=True, llm=MagicMock(name="llm"))
        self._wire_for_comments(wrapper)
        base = Document(page_content="x", metadata={"issue_key": "PROJ-1", "id": "1"})

        with patch.object(JiraApiWrapper, "get_processed_comments_list_with_image_description",
                          return_value=[]) as get_comments:
            list(wrapper._process_document(base))

        _, kwargs = get_comments.call_args
        assert kwargs.get("process_images") is True


class TestIndexToolParams:
    """`include_comments` and `process_images` must be exposed on the public schema."""

    def test_include_comments_present_with_default_false(self):
        # Default flipped to False to avoid the per-issue comments fetch + image work
        # for large datasets. Callers must opt in when comment text matters.
        params = JiraApiWrapper.model_construct()._index_tool_params()

        assert "include_comments" in params
        _, field = params["include_comments"]
        assert field.default is False

    def test_process_images_present_with_default_false(self):
        params = JiraApiWrapper.model_construct()._index_tool_params()

        assert "process_images" in params
        _, field = params["process_images"]
        assert field.default is False

    def test_include_attachments_default_remains_false(self):
        # Backward-compat: existing flags untouched by the new exposure.
        params = JiraApiWrapper.model_construct()._index_tool_params()

        _, field = params["include_attachments"]
        assert field.default is False
