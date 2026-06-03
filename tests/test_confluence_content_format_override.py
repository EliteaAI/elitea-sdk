"""Tests for per-call content_format override on read_page_by_id.

Covers enhancement #3886: pages using structured Confluence macros render
empty under the default body.view format. Callers can now override the
content_format per call (e.g. 'storage') without reconfiguring the toolkit.
"""

import pytest
from unittest.mock import MagicMock

from langchain_community.document_loaders.confluence import ContentFormat

from elitea_sdk.tools.confluence.api_wrapper import (
    ConfluenceAPIWrapper,
    pageId,
    _AtlasDocFormat,
    _StyledView,
)


@pytest.fixture
def wrapper():
    """Wrapper with a mocked atlassian client and VIEW as the toolkit default."""
    w = ConfluenceAPIWrapper.model_construct(
        base_url="https://example.atlassian.net",
        space="AT",
        cloud=True,
        limit=25,
        max_pages=50,
        api_version="2",
        content_format=ContentFormat.VIEW,
        keep_markdown_format=False,
        keep_newlines=False,
        include_comments=False,
        include_attachments=False,
        number_of_retries=1,
        min_retry_seconds=0,
        max_retry_seconds=0,
    )
    w.client = MagicMock()
    w._errors = None
    return w


def _fake_page(page_id: str, **bodies):
    """Build a stub Confluence page with one entry per requested body format.

    Pass body HTML via kwargs: view='...', storage='...', export_view='...', etc.
    Defaults to populating view + storage when no kwargs are provided.
    """
    if not bodies:
        bodies = {"view": "<p>view body</p>", "storage": "<p>storage body</p>"}
    return {
        "id": page_id,
        "title": "Some Page",
        "status": "current",
        "_links": {"base": "https://example.atlassian.net", "webui": "/x"},
        "body": {key: {"value": html} for key, html in bodies.items()},
        "version": {"number": 1},
    }


class TestResolveContentFormat:
    def test_none_returns_none(self, wrapper):
        assert wrapper._resolve_content_format(None) is None

    def test_empty_string_returns_none(self, wrapper):
        assert wrapper._resolve_content_format("") is None

    def test_view(self, wrapper):
        assert wrapper._resolve_content_format("view") == ContentFormat.VIEW

    def test_storage(self, wrapper):
        assert wrapper._resolve_content_format("storage") == ContentFormat.STORAGE

    def test_export_view(self, wrapper):
        assert wrapper._resolve_content_format("export_view") == ContentFormat.EXPORT_VIEW

    def test_editor(self, wrapper):
        assert wrapper._resolve_content_format("editor") == ContentFormat.EDITOR

    def test_anonymous(self, wrapper):
        assert wrapper._resolve_content_format("anonymous") == ContentFormat.ANONYMOUS_EXPORT_VIEW

    def test_styled_view(self, wrapper):
        assert wrapper._resolve_content_format("styled_view") is _StyledView

    def test_atlas_doc_format(self, wrapper):
        assert wrapper._resolve_content_format("atlas_doc_format") is _AtlasDocFormat

    def test_case_insensitive(self, wrapper):
        assert wrapper._resolve_content_format("STORAGE") == ContentFormat.STORAGE
        assert wrapper._resolve_content_format("Atlas_Doc_Format") is _AtlasDocFormat

    def test_unknown_value_returns_none(self, wrapper):
        """Unknown formats fall back to None so the caller defers to the toolkit default."""
        assert wrapper._resolve_content_format("nonexistent_format") is None


class TestReadPageByIdExpand:
    """Verify the `expand` query the wrapper sends to Confluence."""

    def test_default_uses_toolkit_view(self, wrapper):
        wrapper.client.get_page_by_id.return_value = _fake_page("123")
        wrapper.read_page_by_id("123")
        kwargs = wrapper.client.get_page_by_id.call_args.kwargs
        assert kwargs["expand"] == "body.view,version"

    def test_override_storage_changes_expand(self, wrapper):
        wrapper.client.get_page_by_id.return_value = _fake_page("123")
        wrapper.read_page_by_id("123", content_format="storage")
        kwargs = wrapper.client.get_page_by_id.call_args.kwargs
        assert kwargs["expand"] == "body.storage,version"

    def test_override_export_view_changes_expand(self, wrapper):
        wrapper.client.get_page_by_id.return_value = _fake_page("123", export_view="<p>x</p>")
        wrapper.read_page_by_id("123", content_format="export_view")
        kwargs = wrapper.client.get_page_by_id.call_args.kwargs
        assert kwargs["expand"] == "body.export_view,version"

    def test_unknown_override_falls_back_to_toolkit_default(self, wrapper):
        wrapper.client.get_page_by_id.return_value = _fake_page("123")
        # When _resolve_content_format returns None (unknown key), get_pages_by_id falls back to toolkit default.
        # Bypass the schema layer by calling read_page_by_id directly with an unknown value.
        wrapper.read_page_by_id("123", content_format="nonexistent_format")
        kwargs = wrapper.client.get_page_by_id.call_args.kwargs
        assert kwargs["expand"] == "body.view,version"

    def test_override_does_not_mutate_toolkit_state(self, wrapper):
        """Per-call override must not change self.content_format for future calls."""
        wrapper.client.get_page_by_id.return_value = _fake_page("123")
        wrapper.read_page_by_id("123", content_format="storage")
        assert wrapper.content_format == ContentFormat.VIEW

        wrapper.client.get_page_by_id.return_value = _fake_page("456")
        wrapper.read_page_by_id("456")
        kwargs = wrapper.client.get_page_by_id.call_args.kwargs
        assert kwargs["expand"] == "body.view,version"


class TestProcessPageRendering:
    """The override must also drive get_content so fetch and render stay aligned."""

    def test_override_renders_storage_body(self, wrapper):
        page = _fake_page("123",
                          view="<p>view body</p>",
                          storage="<p>storage body</p>")
        wrapper.client.get_page_by_id.return_value = page

        out = wrapper.read_page_by_id("123", content_format="storage")

        assert "storage body" in out
        assert "view body" not in out

    def test_default_renders_view_body(self, wrapper):
        page = _fake_page("123",
                          view="<p>view body</p>",
                          storage="<p>storage body</p>")
        wrapper.client.get_page_by_id.return_value = page

        out = wrapper.read_page_by_id("123")

        assert "view body" in out
        assert "storage body" not in out


class TestStyledView:
    """styled_view returns HTML and goes through the same pipeline as VIEW."""

    def test_expand_is_styled_view(self, wrapper):
        wrapper.client.get_page_by_id.return_value = _fake_page("123", styled_view="<p>STYLED-MARKER</p>")
        wrapper.read_page_by_id("123", content_format="styled_view")
        kwargs = wrapper.client.get_page_by_id.call_args.kwargs
        assert kwargs["expand"] == "body.styled_view,version"

    def test_renders_styled_view_body(self, wrapper):
        wrapper.client.get_page_by_id.return_value = _fake_page(
            "123",
            view="<p>VIEW-MARKER</p>",
            styled_view="<p>STYLED-MARKER</p>",
        )
        out = wrapper.read_page_by_id("123", content_format="styled_view")
        assert "STYLED-MARKER" in out
        assert "VIEW-MARKER" not in out


class TestAtlasDocFormat:
    """atlas_doc_format returns ADF JSON; the HTML pipeline must be bypassed."""

    ADF_JSON = (
        '{"version":1,"type":"doc","content":['
        '{"type":"paragraph","content":[{"type":"text","text":"ADF-MARKER hello"}]}'
        ']}'
    )

    def test_expand_is_atlas_doc_format(self, wrapper):
        wrapper.client.get_page_by_id.return_value = _fake_page("123", atlas_doc_format=self.ADF_JSON)
        wrapper.read_page_by_id("123", content_format="atlas_doc_format")
        kwargs = wrapper.client.get_page_by_id.call_args.kwargs
        assert kwargs["expand"] == "body.atlas_doc_format,version"

    def test_returns_adf_json_verbatim(self, wrapper):
        """The ADF JSON must be passed through without HTML stripping."""
        wrapper.client.get_page_by_id.return_value = _fake_page("123", atlas_doc_format=self.ADF_JSON)
        out = wrapper.read_page_by_id("123", content_format="atlas_doc_format")
        # JSON structure preserved (not stripped by BeautifulSoup)
        assert '"type":"doc"' in out
        assert '"text":"ADF-MARKER hello"' in out

    def test_does_not_run_through_markdownify(self, wrapper):
        """Even with keep_markdown_format=True, ADF must not go through markdownify."""
        wrapper.keep_markdown_format = True
        wrapper.client.get_page_by_id.return_value = _fake_page("123", atlas_doc_format=self.ADF_JSON)
        out = wrapper.read_page_by_id("123", content_format="atlas_doc_format")
        # If markdownify ran, the curly braces and JSON keys would be mangled.
        assert out.startswith('{"version":1')


class TestPageIdSchema:
    def test_content_format_is_optional(self):
        m = pageId(page_id="123")
        assert m.content_format is None

    def test_content_format_accepts_storage(self):
        m = pageId(page_id="123", content_format="storage")
        assert m.content_format == "storage"

    def test_content_format_accepts_styled_view(self):
        m = pageId(page_id="123", content_format="styled_view")
        assert m.content_format == "styled_view"

    def test_content_format_accepts_atlas_doc_format(self):
        m = pageId(page_id="123", content_format="atlas_doc_format")
        assert m.content_format == "atlas_doc_format"

    def test_content_format_rejects_unknown(self):
        with pytest.raises(Exception):
            pageId(page_id="123", content_format="nonexistent_format")
