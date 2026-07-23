"""Unit tests for ADO Wiki `_process_images` attachment classification.

Regression tests for issue #5869: ADO Wiki inserts every attachment (images
and non-image documents alike) using image-markdown syntax
`![name](/.attachments/<guid>.<ext>)`. `_process_images` must only treat URLs
with real image extensions as images; everything else must be left untouched
and handled by the `include_attachments` dependent-doc path.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.ado.wiki.ado_wrapper import (
    AzureDevOpsApiWrapper,
    _IMAGE_EXTENSIONS,
    _url_has_image_extension,
)


class TestUrlHasImageExtension:
    """Direct tests for the extension allow-list helper."""

    @pytest.mark.parametrize("url", [
        "/.attachments/abc.png",
        "/.attachments/abc.PNG",
        "/.attachments/abc.jpg",
        "/.attachments/abc.jpeg",
        "/.attachments/abc.gif",
        "/.attachments/abc.bmp",
        "/.attachments/abc.webp",
        "/.attachments/abc.svg",
        "/.attachments/abc.tiff",
        "/.attachments/abc.tif",
        "/.attachments/abc.ico",
        "https://cdn.example.com/pic.PNG?token=xyz",
        "https://cdn.example.com/pic.jpg#frag",
    ])
    def test_image_extensions_recognized(self, url):
        assert _url_has_image_extension(url) is True

    @pytest.mark.parametrize("url", [
        "/.attachments/invoice.pdf",
        "/.attachments/report.PDF",
        "/.attachments/doc.docx",
        "/.attachments/data.xlsx",
        "/.attachments/deck.pptx",
        "/.attachments/plain.txt",
        "/.attachments/archive.zip",
        "/.attachments/no-extension",
        "",
    ])
    def test_non_image_extensions_rejected(self, url):
        assert _url_has_image_extension(url) is False

    def test_covers_all_declared_extensions(self):
        for ext in _IMAGE_EXTENSIONS:
            assert _url_has_image_extension(f"/.attachments/x{ext}") is True


def _make_stub_wrapper(process_attachment_return, repos_wrapper=None):
    """Build a minimal stub with just enough surface to call `_process_images`
    as an unbound method — avoids the real `__init__` that needs Azure creds.
    """
    stub = SimpleNamespace()
    stub.llm = MagicMock()
    stub._DEFAULT_WORKERS = 1
    stub._index_workers = 1
    stub._get_repos_wrapper = MagicMock(return_value=repos_wrapper or MagicMock())
    stub.process_attachment = MagicMock(return_value=process_attachment_return)
    return stub


class TestProcessImagesAttachmentClassification:
    """Behavioural tests for `_process_images` via the class-level function."""

    def test_pdf_attachment_via_image_syntax_is_left_untouched(self):
        stub = _make_stub_wrapper(process_attachment_return="SHOULD NOT BE USED")
        content = "Intro.\n![invoice.pdf](/.attachments/abc.pdf)\nOutro."

        result = AzureDevOpsApiWrapper._process_images(stub, content, "wiki-1")

        # Non-image extension: markdown stays exactly as-is.
        assert result == content
        # The image path must NOT have called process_attachment for the PDF.
        stub.process_attachment.assert_not_called()

    def test_real_image_attachment_is_described(self):
        stub = _make_stub_wrapper(process_attachment_return="A screenshot of the login page")
        content = "![screen.png](/.attachments/xyz.png)"

        result = AzureDevOpsApiWrapper._process_images(stub, content, "wiki-1")

        assert result == "![screen.png](A screenshot of the login page)"
        stub.process_attachment.assert_called_once()

    def test_tool_exception_return_is_not_spliced_into_markdown(self):
        """Regression: `parse_file_content` returns `ToolException` instead of
        raising. The old code spliced its str() into the page markdown; the fix
        must detect the ToolException instance and skip the substitution."""
        err = ToolException("Not supported type of files entered. ...")
        stub = _make_stub_wrapper(process_attachment_return=err)
        content = "![screen.png](/.attachments/xyz.png)"

        result = AzureDevOpsApiWrapper._process_images(stub, content, "wiki-1")

        assert result == content  # untouched
        # And crucially: the error text is NOT inside the returned content.
        assert "Not supported type" not in result

    def test_mixed_image_and_pdf_only_image_gets_described(self):
        stub = _make_stub_wrapper(process_attachment_return="Login screenshot")
        content = (
            "Header\n"
            "![screen.png](/.attachments/abc.png)\n"
            "![invoice.pdf](/.attachments/def.pdf)\n"
            "Footer"
        )

        result = AzureDevOpsApiWrapper._process_images(stub, content, "wiki-1")

        assert "![screen.png](Login screenshot)" in result
        # PDF markdown is untouched.
        assert "![invoice.pdf](/.attachments/def.pdf)" in result
        # Only ONE call to process_attachment (for the image, not the PDF).
        assert stub.process_attachment.call_count == 1

    def test_no_images_in_content_returns_unchanged(self):
        stub = _make_stub_wrapper(process_attachment_return="unused")
        content = "Just a plain wiki page with no attachments."

        result = AzureDevOpsApiWrapper._process_images(stub, content, "wiki-1")

        assert result == content
        stub.process_attachment.assert_not_called()
        stub._get_repos_wrapper.assert_not_called()
