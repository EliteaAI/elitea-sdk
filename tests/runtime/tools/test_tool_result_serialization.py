"""Tool results reach the LLM as JSON, not a Python repr (#6532).

Also pins the behaviour that must NOT change: strings pass through untouched,
objects keep their own __str__ rendering, and multimodal content-block lists
stay structured.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool

from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime.tools.tool import ToolNode


class OnenoteLikeItems:
    """Stands in for sharepoint's OnenotePageItems, whose __str__ is its contract."""

    def __str__(self):
        return "page text\n-----\nimage description"


def _tool(result):
    return StructuredTool.from_function(
        func=lambda: result,
        name="get_issues",
        description="Run get_issues",
        metadata={"toolkit_id": 1, "toolkit_name": "gitlab", "tool_name": "get_issues"},
    )


class _SingleToolCallClient:
    def bind_tools(self, tools, **_kwargs):
        return self

    def invoke(self, messages, config=None):
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="done")
        return AIMessage(
            content="",
            tool_calls=[{"name": "get_issues", "args": {}, "id": "issues-call"}],
        )


def _run_tool_returning(result):
    tool = _tool(result)
    node = LLMNode(
        client=_SingleToolCallClient(),
        available_tools=[tool],
        lazy_tools_mode=False,
        input_mapping={},
        output_variables=["messages"],
    )

    state = node.invoke({"messages": [HumanMessage(content="list issues")]})

    return next(
        message for message in state["messages"] if isinstance(message, ToolMessage)
    )


class TestCollectionResults:
    def test_list_of_dicts_reaches_llm_as_json(self):
        payload = [{"number": 7, "state": "opened", "author": None}]

        message = _run_tool_returning(payload)

        assert json.loads(message.content) == payload
        assert "'number'" not in message.content

    def test_dict_reaches_llm_as_json(self):
        payload = {"total": 2, "items": [{"key": "PROJ-1"}]}

        message = _run_tool_returning(payload)

        assert json.loads(message.content) == payload

    def test_non_ascii_payload_is_not_escaped(self):
        message = _run_tool_returning([{"title": "привет"}])

        assert "привет" in message.content
        assert "\\u" not in message.content


class TestPreservedBehaviour:
    def test_string_result_passes_through_unchanged(self):
        message = _run_tool_returning("Found nothing to do")

        assert message.content == "Found nothing to do"

    def test_empty_string_result_stays_empty(self):
        message = _run_tool_returning("")

        assert message.content == ""

    def test_object_result_keeps_its_own_rendering(self):
        message = _run_tool_returning(OnenoteLikeItems())

        assert message.content == "page text\n-----\nimage description"

    def test_content_block_list_stays_structured(self):
        blocks = [
            {"type": "text", "text": "here is the page"},
            {"type": "image_url", "image_url": {"url": "https://example/x.png"}},
        ]

        message = _run_tool_returning(blocks)

        assert isinstance(message.content, list)
        assert message.content[0]["type"] == "text"

    def test_bytes_bearing_blocks_are_serialized_without_dumping_binary(self):
        blocks = [{"type": "image", "raw_bytes": b"\x89PNG\x00\x01"}]

        message = _run_tool_returning(blocks)

        assert isinstance(message.content, str)
        assert json.loads(message.content) == [{"type": "image", "raw_bytes": "<6 bytes>"}]

    def test_unserializable_result_never_breaks_the_turn(self):
        payload = {"name": "root"}
        payload["self"] = payload

        message = _run_tool_returning(payload)

        assert isinstance(message.content, str)
        assert message.content


class _ArgumentGeneratingClient:
    def invoke(self, messages, config=None):
        return AIMessage(content="{}")


class TestToolNodeResults:
    def test_tool_node_message_content_is_json(self):
        node = ToolNode(
            client=_ArgumentGeneratingClient(),
            tool=_tool({"ключ": "значение"}),
            input_variables=["messages"],
            output_variables=[],
        )

        # ToolNode dispatches a custom event, which requires an enclosing run.
        runner = RunnableLambda(lambda state, config: node.invoke(state, config=config))
        state = runner.invoke({"messages": [HumanMessage(content="run it")]})

        content = state["messages"][0]["content"]
        assert json.loads(content) == {"ключ": "значение"}
        assert "значение" in content
