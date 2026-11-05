"""A failed model call marks the run-level error holder; a normal answer leaves it empty.

LLMNode turns a provider exception into an error AIMessage so the graph can finish. That message
reads like an answer, so the worker needs the holder to end a non-interactive run as failed.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from elitea_sdk.runtime.tool_outcome import run_error_flag
from elitea_sdk.runtime.tools.llm import LLMNode, _llm_error_message


class FakeBadRequestError(Exception):
    """Stands in for the provider's HTTP 400 error."""

    status_code = 400


@pytest.fixture()
def llm_node():
    client = MagicMock()
    client.model_name = "test-model"
    client._llm_type = "fake"
    return LLMNode(name="test_llm_node", description="test", client=client)


def test_provider_400_sets_the_run_error_holder(llm_node):
    with run_error_flag() as holder, patch.object(
        LLMNode, "_invoke_llm_internal", side_effect=FakeBadRequestError("invalid request: bad parameter")
    ):
        result = llm_node.invoke({"messages": []})

    assert holder["error"]["toolkit"] == "LLM", holder
    assert holder["error"]["tool"] == "model call", holder
    assert "invalid request" in holder["error"]["reason"], holder
    assert result["messages"][-1].response_metadata["llm_node_error"] is True


def test_normal_answer_leaves_the_holder_empty(llm_node):
    with run_error_flag() as holder, patch.object(
        LLMNode, "_invoke_llm_internal", return_value={"messages": [AIMessage(content="all good")]}
    ):
        result = llm_node.invoke({"messages": []})

    assert holder == {}
    assert "llm_node_error" not in result["messages"][-1].response_metadata


def test_error_without_an_open_holder_still_returns_the_tagged_message(llm_node):
    with patch.object(LLMNode, "_invoke_llm_internal", side_effect=FakeBadRequestError("boom")):
        result = llm_node.invoke({"messages": []})

    last = result["messages"][-1]
    assert last.response_metadata["llm_node_error"] is True
    assert last.response_metadata["llm_node_exception_type"] == "FakeBadRequestError"


def test_error_message_helper_tags_the_message_and_the_holder():
    client = MagicMock()
    with run_error_flag() as holder:
        message = _llm_error_message("Failed to retry after truncation: x", FakeBadRequestError("x"), client)

    assert message.content == "Failed to retry after truncation: x"
    assert message.response_metadata["llm_node_error"] is True
    assert message.response_metadata["llm_node_exception_type"] == "FakeBadRequestError"
    assert holder["error"]["tool"] == "model call"


def test_first_model_error_stays_in_the_holder():
    client = MagicMock()
    with run_error_flag() as holder:
        _llm_error_message("first", FakeBadRequestError("first failure"), client)
        _llm_error_message("second", FakeBadRequestError("second failure"), client)

    assert "first failure" in holder["error"]["reason"]


class ReadTimeout(Exception):
    """Same class name as the httpx error that marks a stalled stream."""


class RemoteProtocolError(Exception):
    """Same class name as the httpx error that marks a cut stream."""


class LoopedClient:
    """Asks for one tool call, then raises on every later model call."""

    def __init__(self, error):
        self.error = error
        self.model_name = "test-model"
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "a"}, "id": "call_1"}])
        raise self.error


def _looped_node(client):
    tool = StructuredTool.from_function(
        func=lambda path="x": "file contents",
        name="read_file",
        description="read_file tool.",
        metadata={"toolkit_type": "fs", "toolkit_name": "fs", "tool_name": "read_file"},
    )
    return LLMNode(
        client=client,
        available_tools=[tool],
        tool_names=["read_file"],
        lazy_tools_mode=False,
        input_mapping={},
        output_variables=["messages"],
    )


def _run_looped(error):
    client = LoopedClient(error)
    with run_error_flag() as holder:
        result = _looped_node(client).invoke({"messages": [HumanMessage(content="go")]})
    return holder, result["messages"][-1]


@pytest.mark.parametrize(
    "error, reason_part",
    [
        pytest.param(FakeBadRequestError("upstream exploded"), "upstream exploded", id="generic-error"),
        pytest.param(FakeBadRequestError("invalid api key"), "invalid api key", id="non-recoverable"),
        pytest.param(ReadTimeout("read timed out"), "LLM stream stalled", id="stalled-stream"),
        pytest.param(RemoteProtocolError("peer closed"), "LLM stream interrupted", id="interrupted-stream"),
    ],
)
def test_model_error_in_the_tool_loop_sets_the_run_error_holder(error, reason_part):
    holder, last = _run_looped(error)

    assert holder["error"]["tool"] == "model call", holder
    assert holder["error"]["toolkit"] == "LLM", holder
    assert reason_part in holder["error"]["reason"], holder
    assert last.response_metadata["llm_node_error"] is True


def test_context_error_whose_retry_fails_again_sets_the_holder():
    holder, last = _run_looped(FakeBadRequestError("maximum context length reached"))

    assert holder["error"]["tool"] == "model call", holder
    assert "maximum context length reached" in holder["error"]["reason"], holder
    assert last.response_metadata["llm_node_error"] is True


def test_context_error_with_no_tool_message_sets_the_run_error_holder():
    client = LoopedClient(FakeBadRequestError("unused"))
    node = _looped_node(client)
    error = FakeBadRequestError("maximum context length reached")

    def drop_tool_messages_and_fail(messages, *args, **kwargs):
        # The overflow handler looks for the last tool message; none is left to truncate.
        messages[:] = [m for m in messages if not isinstance(m, ToolMessage)]
        raise error

    with run_error_flag() as holder, patch.object(
        LLMNode, "_enforce_context_floor", side_effect=drop_tool_messages_and_fail
    ):
        result = node.invoke({"messages": [HumanMessage(content="go")]})

    last = result["messages"][-1]
    assert last.content.startswith("Context window exceeded"), last.content
    assert last.response_metadata["llm_node_error"] is True
    assert holder["error"]["tool"] == "model call", holder
    assert "maximum context length reached" in holder["error"]["reason"], holder


def test_normal_tool_loop_leaves_the_holder_empty():
    class AnsweringClient(LoopedClient):
        def invoke(self, messages, config=None):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "a"}, "id": "call_1"}])
            return AIMessage(content="done")

    client = AnsweringClient(None)
    with run_error_flag() as holder:
        result = _looped_node(client).invoke({"messages": [HumanMessage(content="go")]})

    assert holder == {}
    assert result["messages"][-1].content == "done"
