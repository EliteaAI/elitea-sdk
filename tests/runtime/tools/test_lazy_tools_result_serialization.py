"""Lazy-tool invocation returns JSON, not a Python repr (#6532).

`InvokeToolTool` is the meta-tool bound in lazy_tools_mode: the model calls it to
invoke any tool by name. It returns a string, so whatever it produces is what the
model reads -- the LLMNode boundary can no longer help. Both the sync and async
paths must serialize the same way.
"""

import asyncio
import json

from langchain_core.tools import StructuredTool

from elitea_sdk.runtime.tools.lazy_tools import InvokeToolTool, ToolRegistry


class OwnRendering:
    """A toolkit object whose __str__ is its documented rendering."""

    def __str__(self):
        return "page text\n-----\nimage description"


def _invoke_tool_for(result):
    tool = StructuredTool.from_function(
        func=lambda: result,
        name="get_issues",
        description="Run get_issues",
        metadata={"toolkit_name": "gitlab", "tool_name": "get_issues"},
    )
    return InvokeToolTool(registry=ToolRegistry.from_tools([tool]))


def _run_sync(result):
    return _invoke_tool_for(result)._run(toolkit="gitlab", tool="get_issues", arguments={})


def _run_async(result):
    return asyncio.run(
        _invoke_tool_for(result)._arun(toolkit="gitlab", tool="get_issues", arguments={})
    )


class TestSyncInvocation:
    def test_records_reach_the_model_as_json(self):
        payload = [{"id": 1, "title": "Bob's bug", "author": None}]

        output = _run_sync(payload)

        assert json.loads(output) == payload
        assert "'id'" not in output

    def test_non_ascii_is_not_escaped(self):
        output = _run_sync([{"title": "привет"}])

        assert "привет" in output
        assert "\\u" not in output

    def test_string_result_is_unchanged(self):
        assert _run_sync("plain tool output") == "plain tool output"

    def test_object_keeps_its_own_rendering(self):
        assert _run_sync(OwnRendering()) == "page text\n-----\nimage description"

    def test_empty_result_keeps_its_notice(self):
        assert _run_sync(None) == "Tool executed successfully (no output)"


class TestAsyncInvocation:
    def test_records_reach_the_model_as_json(self):
        payload = [{"id": 2, "title": "Async bug"}]

        output = _run_async(payload)

        assert json.loads(output) == payload
        assert "'id'" not in output

    def test_empty_result_keeps_its_notice(self):
        assert _run_async(None) == "Tool executed successfully (no output)"
