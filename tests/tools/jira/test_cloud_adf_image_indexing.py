"""
Tests for the Jira Cloud ADF (Atlassian Document Format) bridge in the indexer.

On Cloud, `description` and `comment.body` come back as a JSON dict, not the
wiki-markup string Server/DC returns. The previous indexer just `str()`-ified
the dict into `page_content`, so the `!ref!` regex in `_extend_data` matched
nothing and `AttachmentResolver` / vision-LLM were never invoked for Cloud.

This module pins:
  1. `_process_issue_for_indexing` converts ADF dict descriptions to wiki markup
     before assembling content (so `_extend_data` can find image refs).
  2. `_extract_image_data` falls back to the media node's `id` when `alt` is
     empty — a common Cloud case (paste-without-alt) that would otherwise produce
     `!|alt=""!` and never match the `[^!|]+` regex.
  3. Server/DC string descriptions still pass through unchanged.
"""
from unittest.mock import MagicMock

import pytest

from elitea_sdk.tools.jira.api_wrapper import JiraApiWrapper


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


def _paragraph(text: str):
    return {"type": "paragraph", "content": [{"type": "text", "text": text}]}


class TestExtractImageDataAltFallback:
    """`_extract_image_data` must produce a regex-matchable ref even without alt."""

    def test_uses_alt_when_present(self):
        adf = _adf_doc(_media_single(attachment_id="10042", alt="screenshot.png"))
        out = _wrapper()._extract_image_data(adf)
        assert '!screenshot.png|alt="screenshot.png"!' in out

    def test_falls_back_to_id_when_alt_missing(self):
        # The pre-fix output was `!|alt=""!` — the regex `[^!|]+` would never
        # match the empty token, so the resolver never ran on Cloud.
        adf = _adf_doc(_media_single(attachment_id="10042", alt=""))
        out = _wrapper()._extract_image_data(adf)
        assert '!10042|alt=""!' in out

        import re
        # Sanity-check: this token must satisfy the indexer's image regex.
        assert re.search(r'!([^!|]+)(?:\|[^!]*)?!', out)

    def test_skips_media_with_no_alt_and_no_id(self):
        # Defensive: a malformed media node with neither alt nor id should not
        # produce a stray `!|alt=""!` token in the output.
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


class TestProcessIssueForIndexingADF:
    """Cloud-shaped ADF descriptions must be converted before assembling content."""

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

        # Wiki markup must appear in the content the indexer will hand to `_extend_data`.
        assert '!diagram.png|alt="diagram.png"!' in doc.page_content
        # The Python dict repr must NOT leak through (the pre-fix bug).
        assert "'type': 'doc'" not in doc.page_content
        assert "'mediaSingle'" not in doc.page_content

    def test_adf_description_without_alt_uses_attachment_id(self):
        adf = _adf_doc(_media_single(attachment_id="99001", alt=""))
        doc = _wrapper()._process_issue_for_indexing(self._issue(adf))

        # Falls back to the media id so AttachmentResolver.by_id can resolve it.
        assert '!99001|alt=""!' in doc.page_content

    def test_server_string_description_passthrough(self):
        # Server/DC: description is already wiki markup. Must not be touched.
        original = "see !screenshot.png! for the bug repro"
        doc = _wrapper()._process_issue_for_indexing(self._issue(original))

        assert original in doc.page_content

    def test_empty_description_handled(self):
        doc = _wrapper()._process_issue_for_indexing(self._issue(""))
        # No description section, but doc must still be created with summary.
        assert "# Description" not in doc.page_content
        assert "# Summary" in doc.page_content
