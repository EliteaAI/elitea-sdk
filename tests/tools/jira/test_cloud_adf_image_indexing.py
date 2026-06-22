"""
Tests for the Jira Cloud ADF (Atlassian Document Format) walker.

Background
----------
On Cloud, `description` and `comment.body` come back as ADF — a JSON dict —
not the wiki-markup string Server/DC returns. The pre-fix indexer just
`str()`-ified the dict into `page_content`, so:
  * the indexer regex `!([^!|]+)…!` matched nothing → `AttachmentResolver`
    and the vision-LLM substitution path never ran for Cloud,
  * embedded text was wrapped in literal `{'type': 'paragraph', ...}` noise.

The walker added in `_extract_image_data` recursively converts every ADF node
type used in real-world Jira issues into readable text, while emitting
wiki-style `!ref!` markup for two distinct image-reference sources:
  1. `media`/`mediaSingle`/`mediaGroup`/`mediaInline` nodes (alt → id fallback)
  2. text nodes carrying a `link` mark whose href points to an attachment URL
     (`/rest/api/3/attachment/content/{id}` Cloud,
      `/secure/attachment/{id}/...` Server/DC) — the visible link text is kept
     AND `!{id}!` is appended.

This module pins both the text-extraction quality and the image-ref detection.
"""
import re
from unittest.mock import MagicMock

import pytest

from elitea_sdk.tools.jira.api_wrapper import JiraApiWrapper


# `_extend_data`'s image-pattern regex — the walker's output must satisfy it
# whenever there's an image to substitute.
_IMAGE_PATTERN = re.compile(r'!([^!|]+)(?:\|[^!]*)?!')


def _wrapper() -> JiraApiWrapper:
    w = JiraApiWrapper.model_construct()
    w._client = MagicMock(name="jira_client")
    w.base_url = "https://example.atlassian.net"
    return w


def _adf_doc(*nodes):
    return {"type": "doc", "version": 1, "content": list(nodes)}


def _media_single(*, attachment_id: str, alt: str = ""):
    return {
        "type": "mediaSingle",
        "content": [{
            "type": "media",
            "attrs": {"type": "file", "id": attachment_id, "alt": alt},
        }],
    }


def _paragraph(*children):
    """Paragraph helper. Each child is either a plain string (wrapped as text node)
    or a pre-built node dict (e.g. a marked text node)."""
    content = []
    for c in children:
        if isinstance(c, str):
            content.append({"type": "text", "text": c})
        else:
            content.append(c)
    return {"type": "paragraph", "content": content}


def _heading(level: int, text: str):
    return {
        "type": "heading",
        "attrs": {"level": level},
        "content": [{"type": "text", "text": text}],
    }


def _list_item(*children):
    return {"type": "listItem", "content": list(children)}


def _bullet_list(*items):
    return {"type": "bulletList", "content": list(items)}


def _ordered_list(*items):
    return {"type": "orderedList", "content": list(items)}


def _link_text(text: str, href: str):
    return {
        "type": "text",
        "text": text,
        "marks": [{"type": "link", "attrs": {"href": href}}],
    }


def _strong_text(text: str):
    return {"type": "text", "text": text, "marks": [{"type": "strong"}]}


# ---------------------------------------------------------------------------
# Image-reference emission
# ---------------------------------------------------------------------------


class TestMediaSingleRefs:
    """`mediaSingle`/`media` is the canonical inline-image case."""

    def test_emits_wiki_ref_with_alt(self):
        adf = _adf_doc(_media_single(attachment_id="10042", alt="screenshot.png"))
        out = _wrapper()._extract_image_data(adf)
        assert '!screenshot.png|alt="screenshot.png"!' in out
        assert _IMAGE_PATTERN.search(out)

    def test_falls_back_to_id_when_alt_missing(self):
        # Pre-fix `!|alt=""!` token couldn't satisfy `[^!|]+` and was dropped.
        adf = _adf_doc(_media_single(attachment_id="10042", alt=""))
        out = _wrapper()._extract_image_data(adf)
        assert '!10042|alt=""!' in out
        assert _IMAGE_PATTERN.search(out)

    def test_skips_media_with_no_alt_and_no_id(self):
        # Defensive: malformed media must not produce a stray `!|alt=""!`.
        adf = _adf_doc({
            "type": "mediaSingle",
            "content": [{"type": "media", "attrs": {"type": "file"}}],
        })
        out = _wrapper()._extract_image_data(adf)
        assert "!" not in out

    def test_paragraph_text_preserved_alongside_media(self):
        adf = _adf_doc(
            _paragraph("see the diagram below"),
            _media_single(attachment_id="55", alt=""),
        )
        out = _wrapper()._extract_image_data(adf)
        assert "see the diagram below" in out
        assert '!55|alt=""!' in out


class TestLinkMarkRefs:
    """Text nodes with a `link` mark to an attachment URL → `!{id}!` token."""

    def test_cloud_attachment_link_emits_id_ref(self):
        # The exact shape from the user's umishra-1504 issue.
        adf = _adf_doc(_paragraph(
            "See attached file: ",
            _link_text("image.png",
                       "https://umishra-1504.atlassian.net/rest/api/3/attachment/content/10020"),
        ))
        out = _wrapper()._extract_image_data(adf)
        assert "image.png" in out          # link text preserved
        assert "!10020!" in out            # ref appended
        assert _IMAGE_PATTERN.search(out)

    def test_cloud_v2_attachment_path_also_recognised(self):
        adf = _adf_doc(_paragraph(
            _link_text("file.pdf",
                       "https://x.atlassian.net/rest/api/2/attachment/content/77"),
        ))
        out = _wrapper()._extract_image_data(adf)
        assert "!77!" in out

    def test_server_secure_attachment_path(self):
        adf = _adf_doc(_paragraph(
            _link_text("diagram.png",
                       "https://jira.acme.com/secure/attachment/12345/diagram.png"),
        ))
        out = _wrapper()._extract_image_data(adf)
        assert "!12345!" in out

    def test_server_thumbnail_path(self):
        adf = _adf_doc(_paragraph(
            _link_text("thumb",
                       "https://jira.acme.com/secure/thumbnail/9001/thumb.png"),
        ))
        out = _wrapper()._extract_image_data(adf)
        assert "!9001!" in out

    def test_external_url_link_is_not_image_ref(self):
        # No `!` token must be emitted for ordinary outbound hyperlinks.
        adf = _adf_doc(_paragraph(
            "Click ",
            _link_text("here", "https://example.com/page"),
        ))
        out = _wrapper()._extract_image_data(adf)
        assert "Click here" in out
        assert "!" not in out

    def test_link_to_browse_or_issue_is_not_image_ref(self):
        adf = _adf_doc(_paragraph(
            _link_text("PROJ-1",
                       "https://jira.acme.com/browse/PROJ-1"),
        ))
        out = _wrapper()._extract_image_data(adf)
        assert "PROJ-1" in out
        assert "!" not in out


# ---------------------------------------------------------------------------
# Text extraction quality (regression against the pre-fix walker)
# ---------------------------------------------------------------------------


class TestParagraphTextExtraction:
    def test_concatenates_all_text_nodes(self):
        # Pre-fix bug: only `content[0]['text']` was kept — second text dropped.
        adf = _adf_doc(_paragraph("See attached file: ", "image.png"))
        out = _wrapper()._extract_image_data(adf)
        assert "See attached file: image.png" in out

    def test_preserves_marked_text(self):
        # Pre-fix bug: paragraphs whose first child had marks were truncated
        # to just the first child via `content[0].get('text', '')`.
        adf = _adf_doc(_paragraph(_strong_text("Scenario 1 (confirmed):")))
        out = _wrapper()._extract_image_data(adf)
        assert "Scenario 1 (confirmed):" in out

    def test_mixed_marks_and_links_in_one_paragraph(self):
        adf = _adf_doc(_paragraph(
            _strong_text("Note: "),
            "see ",
            _link_text("docs", "https://example.com/docs"),
            " for more.",
        ))
        out = _wrapper()._extract_image_data(adf)
        assert "Note: see docs for more." in out


class TestHeadingsAndLists:
    def test_heading_prefix_by_level(self):
        adf = _adf_doc(_heading(2, "Description"))
        out = _wrapper()._extract_image_data(adf)
        assert out.lstrip().startswith("## Description")

    def test_heading_invalid_level_clamps_to_one(self):
        adf = {"type": "doc", "content": [{
            "type": "heading", "attrs": {"level": "bogus"},
            "content": [{"type": "text", "text": "Bad"}],
        }]}
        out = _wrapper()._extract_image_data(adf)
        assert "# Bad" in out

    def test_bullet_list_items_kept_with_dash_prefix(self):
        adf = _adf_doc(_bullet_list(
            _list_item(_paragraph("alpha")),
            _list_item(_paragraph("beta")),
            _list_item(_paragraph("gamma")),
        ))
        out = _wrapper()._extract_image_data(adf)
        assert "- alpha" in out
        assert "- beta" in out
        assert "- gamma" in out

    def test_ordered_list_numbered(self):
        adf = _adf_doc(_ordered_list(
            _list_item(_paragraph("first")),
            _list_item(_paragraph("second")),
        ))
        out = _wrapper()._extract_image_data(adf)
        assert "1. first" in out
        assert "2. second" in out

    def test_nested_bullet_list(self):
        adf = _adf_doc(_bullet_list(
            _list_item(
                _paragraph("outer"),
                _bullet_list(_list_item(_paragraph("inner"))),
            ),
        ))
        out = _wrapper()._extract_image_data(adf)
        assert "outer" in out
        assert "inner" in out


class TestOtherBlockTypes:
    def test_blockquote(self):
        adf = _adf_doc({
            "type": "blockquote",
            "content": [_paragraph("quoted line")],
        })
        out = _wrapper()._extract_image_data(adf)
        assert "> quoted line" in out

    def test_code_block(self):
        adf = _adf_doc({
            "type": "codeBlock",
            "content": [{"type": "text", "text": "print('hi')"}],
        })
        out = _wrapper()._extract_image_data(adf)
        assert "```" in out
        assert "print('hi')" in out

    def test_rule(self):
        adf = _adf_doc({"type": "rule"})
        out = _wrapper()._extract_image_data(adf)
        assert "---" in out

    def test_table_extracted_as_pipe_rows(self):
        adf = _adf_doc({
            "type": "table",
            "content": [
                {"type": "tableRow", "content": [
                    {"type": "tableHeader", "content": [_paragraph("A")]},
                    {"type": "tableHeader", "content": [_paragraph("B")]},
                ]},
                {"type": "tableRow", "content": [
                    {"type": "tableCell", "content": [_paragraph("1")]},
                    {"type": "tableCell", "content": [_paragraph("2")]},
                ]},
            ],
        })
        out = _wrapper()._extract_image_data(adf)
        assert "| A | B |" in out
        assert "| 1 | 2 |" in out

    def test_mention_and_emoji(self):
        adf = _adf_doc(_paragraph(
            {"type": "mention", "attrs": {"id": "abc", "text": "@john"}},
            " ",
            {"type": "emoji", "attrs": {"shortName": ":tada:"}},
        ))
        out = _wrapper()._extract_image_data(adf)
        assert "@john" in out
        assert ":tada:" in out

    def test_unknown_node_type_walks_children(self):
        # Future-proof: an ADF node we don't model explicitly but that has
        # children must still surrender its text.
        adf = _adf_doc({
            "type": "panel",
            "content": [_paragraph("inside a panel")],
        })
        out = _wrapper()._extract_image_data(adf)
        assert "inside a panel" in out


# ---------------------------------------------------------------------------
# Top-level shapes & passthroughs
# ---------------------------------------------------------------------------


class TestInputShapes:
    def test_string_passthrough_for_server_wiki_markup(self):
        s = "see !screenshot.png! for the bug repro"
        assert _wrapper()._extract_image_data(s) == s

    def test_empty_string(self):
        assert _wrapper()._extract_image_data("") == ""

    def test_list_of_nodes(self):
        out = _wrapper()._extract_image_data([_paragraph("one"), _paragraph("two")])
        assert "one" in out and "two" in out

    def test_legacy_filename_content_shape(self):
        # Backward-compat: raw attachment payload {filename, content: <bytes>}.
        out = _wrapper()._extract_image_data(
            {"filename": "a.png", "content": b"\x00\x01"}
        )
        assert "!a.png|alt=a.png!" in out

    def test_none_returns_empty(self):
        assert _wrapper()._extract_image_data(None) == ""


# ---------------------------------------------------------------------------
# Indexer integration: _process_issue_for_indexing
# ---------------------------------------------------------------------------


class TestProcessIssueForIndexingADF:
    """ADF descriptions must be converted before assembling page_content."""

    def _issue(self, description):
        return {
            "id": "1001",
            "key": "PROJ-1",
            "fields": {
                "summary": "S",
                "description": description,
                "reporter": None,
                "status": None,
                "updated": None,
                "created": None,
                "project": None,
                "issuetype": None,
            },
        }

    def test_adf_description_with_alt_yields_wiki_markup(self):
        adf = _adf_doc(
            _paragraph("intro"),
            _media_single(attachment_id="10042", alt="diagram.png"),
        )
        doc = _wrapper()._process_issue_for_indexing(self._issue(adf))

        assert '!diagram.png|alt="diagram.png"!' in doc.page_content
        # Python dict repr must NOT leak through (the str(dict) regression).
        assert "'type': 'doc'" not in doc.page_content
        assert "'mediaSingle'" not in doc.page_content

    def test_adf_description_without_alt_uses_attachment_id(self):
        adf = _adf_doc(_media_single(attachment_id="99001", alt=""))
        doc = _wrapper()._process_issue_for_indexing(self._issue(adf))
        assert '!99001|alt=""!' in doc.page_content

    def test_adf_description_with_attachment_link_mark(self):
        # The umishra-1504 case — image referenced via a link, not mediaSingle.
        adf = _adf_doc(_paragraph(
            "See attached file: ",
            _link_text("image.png",
                       "https://umishra-1504.atlassian.net/rest/api/3/attachment/content/10020"),
        ))
        doc = _wrapper()._process_issue_for_indexing(self._issue(adf))
        assert "!10020!" in doc.page_content
        assert "image.png" in doc.page_content
        assert _IMAGE_PATTERN.search(doc.page_content)

    def test_server_string_description_passthrough(self):
        original = "see !screenshot.png! for the bug repro"
        doc = _wrapper()._process_issue_for_indexing(self._issue(original))
        assert original in doc.page_content

    def test_empty_description_handled(self):
        doc = _wrapper()._process_issue_for_indexing(self._issue(""))
        assert "# Description" not in doc.page_content
        assert "# Summary" in doc.page_content


# ---------------------------------------------------------------------------
# Real-world fixture: the ADF the user pasted from umishra-1504
# ---------------------------------------------------------------------------


_REAL_USER_ADF = {
    "type": "doc", "version": 1, "content": [
        {"type": "paragraph", "content": [
            {"type": "text", "text": "Original GitHub Issue: #4348",
             "marks": [{"type": "strong"}]},
        ]},
        {"type": "paragraph", "content": [
            {"type": "text", "text": "Original author: @umishra1504 — 2026-03-24T06:43:35Z"},
        ]},
        {"type": "heading", "attrs": {"level": 2}, "content": [
            {"type": "text", "text": "Description"},
        ]},
        {"type": "paragraph", "content": [
            {"type": "text", "text": "In a pipeline containing a Toolkit node followed by an LLM node, when the Toolkit node calls a sensitive tool and the user approves..."},
        ]},
        {"type": "heading", "attrs": {"level": 2}, "content": [
            {"type": "text", "text": "Steps to Reproduce"},
        ]},
        {"type": "orderedList", "content": [
            {"type": "listItem", "content": [{"type": "paragraph", "content": [
                {"type": "text", "text": "step one"}]}]},
            {"type": "listItem", "content": [{"type": "paragraph", "content": [
                {"type": "text", "text": "step two"}]}]},
        ]},
        {"type": "heading", "attrs": {"level": 2}, "content": [
            {"type": "text", "text": "Test Scenarios"},
        ]},
        {"type": "paragraph", "content": [
            {"type": "text", "text": "Scenario 1 (confirmed):",
             "marks": [{"type": "strong"}]},
        ]},
        {"type": "bulletList", "content": [
            {"type": "listItem", "content": [{"type": "paragraph", "content": [
                {"type": "text", "text": "Parent agent: SDE (Agent ID: 256)"}]}]},
            {"type": "listItem", "content": [{"type": "paragraph", "content": [
                {"type": "text", "text": "Pipeline: Toolkit 1 → LLM 2"}]}]},
        ]},
        {"type": "paragraph", "content": [
            {"type": "text", "text": "See attached file: "},
            {"type": "text", "text": "image.png", "marks": [
                {"type": "link", "attrs": {
                    "href": "https://umishra-1504.atlassian.net/rest/api/3/attachment/content/10020"
                }},
            ]},
        ]},
    ],
}


class TestRealWorldUserADF:
    """End-to-end check on the exact ADF shape the user pasted."""

    def test_text_content_survives(self):
        out = _wrapper()._extract_image_data(_REAL_USER_ADF)
        assert "Original GitHub Issue: #4348" in out
        assert "## Description" in out
        assert "## Steps to Reproduce" in out
        assert "1. step one" in out
        assert "2. step two" in out
        assert "Scenario 1 (confirmed):" in out
        assert "- Parent agent: SDE (Agent ID: 256)" in out
        assert "See attached file: image.png" in out

    def test_attachment_link_becomes_indexable_ref(self):
        out = _wrapper()._extract_image_data(_REAL_USER_ADF)
        assert "!10020!" in out
        # The token must be matchable by `_extend_data`'s regex so the
        # AttachmentResolver / vision-LLM substitution chain can fire.
        assert _IMAGE_PATTERN.search(out)

    def test_no_python_dict_literal_leaks(self):
        out = _wrapper()._extract_image_data(_REAL_USER_ADF)
        # The pre-fix str(dict) noise must not appear anywhere.
        assert "'type': 'paragraph'" not in out
        assert "{'content':" not in out
