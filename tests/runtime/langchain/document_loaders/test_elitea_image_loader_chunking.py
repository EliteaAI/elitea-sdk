"""Regression tests for EliteAImageLoader max_tokens chunking.

Reference: EliteaAI/elitea_issues#4259 — loader ignored max_tokens and always
returned a single document. These tests avoid LLM/OCR dependencies by mocking
``get_content`` and verify the chunking behavior directly.
"""

from unittest.mock import patch

import pytest

from elitea_sdk.runtime.langchain.document_loaders.EliteAImageLoader import EliteAImageLoader


def _make_loader(**kwargs):
    return EliteAImageLoader(file_path="/tmp/fake_image.png", **kwargs)


def _long_markdown_content(paragraphs: int = 40) -> str:
    """Build markdown content large enough to exceed small max_tokens budgets."""
    block = (
        "This is a synthetic paragraph used to exercise the token-based "
        "chunking logic in EliteAImageLoader. It contains enough filler text "
        "so that tiktoken length exceeds a small max_tokens threshold. "
    )
    sections = []
    for idx in range(paragraphs):
        sections.append(f"## Section {idx}\n\n{block * 3}\n")
    return "\n".join(sections)


def test_load_respects_max_tokens_and_splits_into_multiple_chunks():
    content = _long_markdown_content()
    loader = _make_loader(max_tokens=128)
    with patch.object(EliteAImageLoader, "get_content", return_value=content):
        docs = loader.load()
    assert len(docs) > 1, "Expected content to be split into multiple chunks when max_tokens is small"
    for doc in docs:
        assert "chunk_id" in doc.metadata
        assert doc.metadata["source"] == "/tmp/fake_image.png"


def test_load_returns_single_doc_when_content_fits_max_tokens():
    content = "# Title\n\nShort description."
    loader = _make_loader(max_tokens=512)
    with patch.object(EliteAImageLoader, "get_content", return_value=content):
        docs = loader.load()
    assert len(docs) == 1
    # When not actually split, metadata must stay clean (no chunker-added keys).
    assert "chunk_id" not in docs[0].metadata
    assert docs[0].metadata.get("processing_method") == "ocr"


@pytest.mark.parametrize("disabled_value", [-1, 0, None])
def test_load_skips_chunking_when_max_tokens_disabled(disabled_value):
    content = _long_markdown_content()
    loader = _make_loader(max_tokens=disabled_value)
    with patch.object(EliteAImageLoader, "get_content", return_value=content):
        docs = loader.load()
    assert len(docs) == 1
    assert "chunk_id" not in docs[0].metadata
