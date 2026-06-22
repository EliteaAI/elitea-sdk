"""
Tests for the `index_name` None-guard in `BaseIndexerToolkit.index_data`.

Regression coverage for the bug where a missing `index_name` propagated through
`index_meta_init`, which formats `f"{INDEX_META_TYPE}_{index_name}"` into the
document's page_content. With `index_name=None`, this produced the literal
string "index_meta_None" — orphan meta rows accumulated in pgvector with no
associated collection. Guard at the entry point: fail fast with a clear error
instead of silently corrupting the store.
"""
import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit


class TestIndexNameGuard:

    def _wrapper(self):
        # Bypass full Pydantic validation — we only need the method bound.
        return BaseIndexerToolkit.model_construct()

    def test_raises_when_index_name_missing(self):
        with pytest.raises(ToolException, match="index_name"):
            self._wrapper().index_data()

    def test_raises_when_index_name_none(self):
        with pytest.raises(ToolException, match="index_name"):
            self._wrapper().index_data(index_name=None)

    def test_raises_when_index_name_empty_string(self):
        with pytest.raises(ToolException, match="index_name"):
            self._wrapper().index_data(index_name="")

    def test_raises_when_index_name_only_whitespace(self):
        # Whitespace-only would also stringify into a meaningless meta page_content.
        with pytest.raises(ToolException, match="index_name"):
            self._wrapper().index_data(index_name="   ")
