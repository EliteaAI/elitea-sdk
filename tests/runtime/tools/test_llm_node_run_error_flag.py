"""A failed model call marks the run-level error holder; a normal answer leaves it empty.

LLMNode turns a provider exception into an error AIMessage so the graph can finish. That message
reads like an answer, so the worker needs the holder to end a non-interactive run as failed.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

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
