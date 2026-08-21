"""Regression test for IndexerNode.invoke's consumer of self.index_tool.invoke
(indexer_tool.py, formerly line 75).

self.index_tool is a plain BaseTool built with handle_tool_error=False, so a
ToolException raised from its _run propagates straight out of .invoke() —
the `isinstance(index_results, ToolException)` check used to compensate for
index tools that *returned* a ToolException instead of raising; now that all
producers raise, the check was dead code and has been removed. This test
proves the raise still lands in the same fallback message path.
"""
from unittest.mock import patch

from langchain_core.tools import BaseTool, ToolException

from elitea_sdk.runtime.tools.indexer_tool import IndexerNode


class _DocTool(BaseTool):
    name: str = "doc_tool"
    description: str = "Produces a document"

    def _run(self, *args, **kwargs):
        return "some document text"


class _FailingIndexTool(BaseTool):
    name: str = "index_tool"
    description: str = "Indexes a document"

    def _run(self, *args, **kwargs):
        raise ToolException("Index backend unavailable")


def test_index_tool_raise_propagates_into_fallback_message():
    node = IndexerNode(
        tool=_DocTool(), index_tool=_FailingIndexTool(), input_mapping={}, input_variables=[]
    )

    with patch("elitea_sdk.runtime.tools.indexer_tool.dispatch_custom_event"):
        result = node.invoke({})

    content = result["messages"][0]["content"]
    assert "Index backend unavailable" in content
