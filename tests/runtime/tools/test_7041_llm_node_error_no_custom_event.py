"""LLMNode must tag error AIMessages with llm_node_error metadata on failure (#7041).

When the LLM raises (e.g. AuthenticationError 401) and routing_scope is None, LLMNode.invoke()
catches the exception and converts it to an AIMessage so the graph can continue. The AIMessage
is tagged with response_metadata['llm_node_error'] = True so that Application._run() can detect
it downstream and re-raise as RuntimeError, causing BaseTool.run() to fire on_tool_error and
the UI Application chip to show an error state instead of closing as "complete".

This test also verifies that no on_custom_event("llm_node_error") is dispatched — the old code
dispatched it with a random run_id, which in a parallel nested-agent scenario picked the
still-running sibling's run_id and falsely closed its UI spinner with an error state.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage


class _TrackingCallback(BaseCallbackHandler):
    """Records every on_custom_event call."""

    def __init__(self):
        super().__init__()
        self.custom_events = []

    def on_custom_event(self, name, data, *, run_id, **kwargs):
        self.custom_events.append({"name": name, "run_id": run_id, "data": data})


@pytest.fixture()
def llm_node():
    """Build a minimal LLMNode with a mock client."""
    from elitea_sdk.runtime.tools.llm import LLMNode

    client = MagicMock()
    client.model_name = "broken-model"
    client._llm_type = "fake"

    node = LLMNode(
        name="test_llm_node",
        description="test",
        client=client,
    )
    return node


def _config_with_callback(cb):
    return {"callbacks": [cb]}


def test_llm_node_error_tags_response_metadata(llm_node):
    """When the LLM raises with routing_scope=None, invoke() returns a tagged error AIMessage (#7041)."""
    from elitea_sdk.runtime.tools.llm import LLMNode

    with patch.object(LLMNode, "_invoke_llm_internal", side_effect=RuntimeError("connection refused")):
        result = llm_node.invoke({"messages": []})

    messages = result.get("messages", [])
    assert messages, "Expected at least one message in the result"
    last_msg = messages[-1]
    assert isinstance(last_msg, AIMessage), f"Expected AIMessage, got {type(last_msg)}"
    assert last_msg.response_metadata.get("llm_node_error") is True, (
        "Expected response_metadata['llm_node_error'] = True on error AIMessage"
    )
    assert last_msg.response_metadata.get("llm_node_exception_type") == "RuntimeError", (
        f"Expected llm_node_exception_type='RuntimeError', got "
        f"{last_msg.response_metadata.get('llm_node_exception_type')}"
    )


def test_llm_node_error_does_not_call_on_custom_event(llm_node):
    """on_custom_event must not be called with 'llm_node_error' when the LLM raises (#7041)."""
    from elitea_sdk.runtime.tools.llm import LLMNode

    cb = _TrackingCallback()

    with patch.object(LLMNode, "_invoke_llm_internal", side_effect=RuntimeError("connection refused")):
        llm_node.invoke({"messages": []}, config=_config_with_callback(cb))

    llm_node_error_events = [e for e in cb.custom_events if e["name"] == "llm_node_error"]
    assert llm_node_error_events == [], (
        f"Expected no llm_node_error custom events but got: {llm_node_error_events}"
    )


def test_llm_node_error_no_custom_event_with_callback_manager(llm_node):
    """Same guarantee holds when callbacks are wrapped in a CallbackManager."""
    from langchain_core.callbacks import CallbackManager
    from elitea_sdk.runtime.tools.llm import LLMNode

    cb = _TrackingCallback()
    mgr = CallbackManager(handlers=[cb])

    with patch.object(LLMNode, "_invoke_llm_internal", side_effect=RuntimeError("connection refused")):
        llm_node.invoke({"messages": []}, config={"callbacks": mgr})

    llm_node_error_events = [e for e in cb.custom_events if e["name"] == "llm_node_error"]
    assert llm_node_error_events == [], (
        f"Expected no llm_node_error custom events but got: {llm_node_error_events}"
    )
