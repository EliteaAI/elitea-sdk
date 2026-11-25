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

    def test_tool_exception_raise_is_not_spliced_into_markdown(self):
        """Regression: `parse_file_content` raises `ToolException` on failure.
        The old code spliced its str() into the page markdown; the fix must
        catch the exception and skip the substitution."""
        err = ToolException("Not supported type of files entered. ...")
        stub = _make_stub_wrapper(process_attachment_return="unused")
        stub.process_attachment = MagicMock(side_effect=err)
        content = "![screen.png](/.attachments/xyz.png)"

        result = AzureDevOpsApiWrapper._process_images(stub, content, "wiki-1")

        # The markdown is untouched; the error is only named in the trailing note.
        assert result.startswith(content + "\n\n[1 of 1 images not described: ")
        assert "![screen.png](Not supported type" not in result

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


# --- #7755: code wikis, size suffix, visible failures ---------------------------------

from unittest.mock import patch  # noqa: E402

from elitea_sdk.tools.ado.wiki import ado_wrapper as wiki_module  # noqa: E402
from elitea_sdk.tools.ado.wiki.ado_wrapper import _strip_image_size_suffix  # noqa: E402


def _wiki(type_=None, versions=None, mapped_path=None):
    return SimpleNamespace(
        repository_id="repo-1",
        type=type_,
        versions=[SimpleNamespace(version=v) for v in versions] if versions is not None else None,
        mapped_path=mapped_path,
    )


def _real_wrapper(wiki):
    wrapper = AzureDevOpsApiWrapper.model_construct()
    object.__setattr__(wrapper, "organization_url", "https://dev.azure.com/org")
    object.__setattr__(wrapper, "project", "Digital")
    object.__setattr__(wrapper, "token", SimpleNamespace(get_secret_value=lambda: "t"))
    object.__setattr__(wrapper, "llm", MagicMock())
    object.__setattr__(wrapper, "_index_workers", 1)
    object.__setattr__(wrapper, "_image_cache", None)
    object.__setattr__(wrapper, "_client", MagicMock())
    wrapper._client.get_wiki.return_value = wiki
    return wrapper


class TestWikiBranch:

    @pytest.mark.parametrize("wiki, branch", [
        (_wiki("codeWiki", ["main"], "/"), "main"),
        (_wiki("codeWiki", ["release/2.0"], "/docs"), "release/2.0"),
        (_wiki("projectWiki", ["wikiMaster"], "/"), "wikiMaster"),
        (_wiki("projectWiki", ["main"], "/"), "wikiMaster"),
        (_wiki("codeWiki", [], "/"), "wikiMaster"),
        (_wiki("codeWiki", None, "/"), "wikiMaster"),
        (SimpleNamespace(repository_id="repo-1"), "wikiMaster"),
    ])
    def test_repos_wrapper_uses_the_published_branch(self, wiki, branch):
        wrapper = _real_wrapper(wiki)
        with patch.object(wiki_module, "ReposApiWrapper") as repos_cls:
            wrapper._get_repos_wrapper("w")
        assert repos_cls.call_args.kwargs["base_branch"] == branch
        assert repos_cls.call_args.kwargs["active_branch"] == branch

    def test_failed_lookup_is_cached_with_its_reason(self):
        wrapper = _real_wrapper(_wiki("projectWiki"))
        boom = Exception("Neither the base branch 'wikiMaster' nor the active branch exist")
        with patch.object(wiki_module, "ReposApiWrapper", side_effect=boom) as repos_cls:
            assert wrapper._get_repos_wrapper("w") is None
            assert wrapper._get_repos_wrapper("w") is None
        assert repos_cls.call_count == 1
        assert wrapper._client.get_wiki.call_count == 1
        assert "wikiMaster" in wrapper._repos_wrapper_error("w")


class TestCodeWikiImages:

    def _run(self, wiki, content):
        wrapper = _real_wrapper(wiki)
        repos = MagicMock()
        repos.download_file.return_value = b"\x89PNG"
        with patch.object(wiki_module, "ReposApiWrapper", return_value=repos), \
                patch.object(wiki_module, "parse_file_content", return_value="A diagram") as parse:
            result = wrapper._process_images(content, wiki_identified="w")
        return result, repos, parse

    def test_code_wiki_on_main_describes_attachments(self):
        result, repos, _ = self._run(
            _wiki("codeWiki", ["main"], "/"), "Intro\n![image.png](/.attachments/image-1.png)\n")
        assert result == "Intro\n![image.png](A diagram)\n"
        repos.download_file.assert_called_once_with(path=".attachments/image-1.png")

    @pytest.mark.parametrize("mapped_path", ["/docs", "docs/", "/docs/"])
    def test_code_wiki_subfolder_prefixes_the_path(self, mapped_path):
        _, repos, _ = self._run(_wiki("codeWiki", ["main"], mapped_path), "![a](/.attachments/a%20b.png)")
        repos.download_file.assert_called_once_with(path="docs/.attachments/a b.png")

    def test_project_wiki_ignores_mapped_path(self):
        _, repos, _ = self._run(_wiki("projectWiki", ["wikiMaster"], "/docs"), "![a](/.attachments/a.png)")
        repos.download_file.assert_called_once_with(path=".attachments/a.png")

    @pytest.mark.parametrize("suffix", [" =250x", " =x300", " =250x300", "  =250x"])
    def test_size_suffix_is_described_and_removed(self, suffix):
        content = f"![image.png](/.attachments/image-1.png{suffix})"
        result, repos, _ = self._run(_wiki("codeWiki", ["main"], "/"), content)
        assert result == "![image.png](A diagram)"
        repos.download_file.assert_called_once_with(path=".attachments/image-1.png")


class TestImageFailureNote:

    def test_all_images_fail_when_the_branch_is_missing(self):
        wrapper = _real_wrapper(_wiki("projectWiki"))
        content = "Body\n![a](/.attachments/a.png)\n![b](/.attachments/b.png =250x)\n![doc](/.attachments/d.pdf)"
        boom = Exception("Neither the base branch 'wikiMaster' nor the active branch 'wikiMaster' exist")
        with patch.object(wiki_module, "ReposApiWrapper", side_effect=boom):
            result = wrapper._process_images(content, wiki_identified="w")
        assert result.startswith(content)
        note = result[len(content):]
        assert note.startswith("\n\n[2 of 2 images not described: wiki repository is unavailable: Neither")
        assert note.count("Neither") == 1  # the same reason is listed once

    def test_partial_failure_counts_only_failed_images(self):
        stub = _make_stub_wrapper(process_attachment_return="unused")
        stub.process_attachment = MagicMock(side_effect=["Described", ToolException("LLM error: 429")])
        content = "![a](/.attachments/a.png)\n![b](/.attachments/b.png)"

        result = AzureDevOpsApiWrapper._process_images(stub, content, "wiki-1")

        assert result == "![a](Described)\n![b](/.attachments/b.png)\n\n[1 of 2 images not described: LLM error: 429]"

    def test_non_image_attachments_are_not_counted(self):
        stub = _make_stub_wrapper(process_attachment_return="unused")
        content = "![invoice.pdf](/.attachments/abc.pdf)"
        assert AzureDevOpsApiWrapper._process_images(stub, content, "wiki-1") == content

    def test_relative_image_failure_is_reported(self):
        stub = _make_stub_wrapper(process_attachment_return="unused")
        content = "![x](images/x.png)"
        with patch.object(wiki_module.requests, "get", side_effect=Exception("Invalid URL")):
            result = AzureDevOpsApiWrapper._process_images(stub, content, "wiki-1")
        assert result == content + "\n\n[1 of 1 images not described: could not fetch 'images/x.png': Invalid URL]"

    def test_indexing_path_does_not_append_the_note(self):
        stub = _make_stub_wrapper(process_attachment_return="unused")
        stub.process_attachment = MagicMock(side_effect=ToolException("boom"))
        content = "![a](/.attachments/a.png)"
        result = AzureDevOpsApiWrapper._process_images(stub, content, "wiki-1", report_failures=False)
        assert result == content


class TestIncludeAttachmentsPath:

    def test_code_wiki_subfolder_downloads_from_the_prefixed_path_but_keeps_ids(self):
        from langchain_core.documents import Document

        wrapper = _real_wrapper(_wiki("codeWiki", ["main"], "/docs"))
        object.__setattr__(wrapper, "_index_include_attachments", True)
        object.__setattr__(wrapper, "_index_wiki_identifier", "w")
        repos = MagicMock()
        repos.download_file.return_value = b"\x89PNG"
        parent = Document(page_content="body", metadata={
            "id": "page-1", "path": "/Home", "_ado_wiki_attachments": ["/.attachments/diagram.png"],
        })
        with patch.object(wiki_module, "ReposApiWrapper", return_value=repos) as repos_cls, \
                patch.object(AzureDevOpsApiWrapper, "_matches_extension_filter", lambda self, name: True):
            emitted = list(wrapper._process_document(parent))

        assert repos_cls.call_args.kwargs["active_branch"] == "main"
        repos.download_file.assert_called_once_with(path="docs/.attachments/diagram.png")
        assert emitted[0].metadata["id"] == "page-1::.attachments/diagram.png"


@pytest.mark.parametrize("url, clean", [
    ("/.attachments/x.png =250x", "/.attachments/x.png"),
    ("/.attachments/x.png =x300", "/.attachments/x.png"),
    ("/.attachments/x.png =250x300", "/.attachments/x.png"),
    ("/.attachments/x.png", "/.attachments/x.png"),
    ("/.attachments/x=1.png", "/.attachments/x=1.png"),
    ("", ""),
])
def test_strip_image_size_suffix(url, clean):
    assert _strip_image_size_suffix(url) == clean
