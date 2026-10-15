"""LLMNode must not dispatch on_custom_event("llm_node_error") on failure (#7041).

Before the fix, a failed LLMNode called on_custom_event("llm_node_error", ...) with a
fresh random run_id on every registered callback. In a parent agent running two nested
agents in parallel, this spurious dispatch picked the still-running sibling's run_id
from pending_llm_requests and falsely closed its UI spinner with an error state.

on_llm_error fires via LangChain's normal callback mechanism with the exact matching
run_id and already handles the error correctly, so the on_custom_event dispatch was
entirely redundant and harmful.

This test verifies that on_custom_event is never called by LLMNode when the LLM raises.
"""

from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler


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


def test_llm_node_error_does_not_call_on_custom_event(llm_node):
    """on_custom_event must not be called when the LLM raises (#7041)."""
    from elitea_sdk.runtime.tools.llm import LLMNode

    cb = _TrackingCallback()

    # Patch _invoke_llm_internal to raise directly, bypassing unrelated
    # setup code (input_mapping propagation etc.) so we exercise the exact
    # exception-handling block in invoke() where the dispatch was removed.
    with patch.object(LLMNode, "_invoke_llm_internal", side_effect=RuntimeError("connection refused")):
        with pytest.raises(Exception):
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
        with pytest.raises(Exception):
            llm_node.invoke({"messages": []}, config={"callbacks": mgr})

    llm_node_error_events = [e for e in cb.custom_events if e["name"] == "llm_node_error"]
    assert llm_node_error_events == [], (
        f"Expected no llm_node_error custom events but got: {llm_node_error_events}"
    )
