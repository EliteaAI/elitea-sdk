"""LLMNode must tag error AIMessages with llm_node_error metadata on failure (#7041).

When the LLM raises (e.g. AuthenticationError 401) and routing_scope is None, LLMNode.invoke()
catches the exception and converts it to an AIMessage so the graph can continue. The AIMessage
is tagged with response_metadata['llm_node_error'] = True so that Application._run() can detect
it downstream and re-raise as LlmNodeError, causing BaseTool.run() to fire on_tool_error and
the UI Application chip to show an error state instead of closing as "complete".

Application._run() checks only the *final* non-Human message so that a tagged error from a
previous call (already committed to the child's checkpoint) does not cause every subsequent
call to that sub-agent to raise.

This test also verifies that no on_custom_event("llm_node_error") is dispatched — the old code
dispatched it with a random run_id, which in a parallel nested-agent scenario picked the
still-running sibling's run_id and falsely closed its UI spinner with an error state.

Additionally, tests that LlmNodeError raised by a sub-agent's Application._run() propagates
through the parent agent's __perform_tool_calling loop instead of being swallowed as a
ToolMessage(status="error") (#7041 end-to-end propagation gap).
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage


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


# ---------------------------------------------------------------------------
# Application._run() boundary tests
# ---------------------------------------------------------------------------

def _tagged_error(text="Error: auth failed", exc_type="AuthenticationError"):
    msg = AIMessage(content=text)
    msg.response_metadata["llm_node_error"] = True
    msg.response_metadata["llm_node_exception_type"] = exc_type
    return msg


def test_application_run_raises_when_final_message_is_tagged():
    """A tagged final non-Human message must raise LlmNodeError (#7041)."""
    from elitea_sdk.runtime.tools.application import LlmNodeError, _raise_if_llm_node_error

    response = {"messages": [HumanMessage(content="go"), _tagged_error()]}
    with pytest.raises(LlmNodeError, match="Nested agent LLM failed"):
        _raise_if_llm_node_error(response)


def test_application_run_does_not_raise_when_tagged_message_is_not_final():
    """An older tagged message followed by a successful reply must NOT raise.

    This prevents a tagged error from a previous call (committed to the child's
    checkpoint history) from poisoning every later call to the same sub-agent.
    """
    from elitea_sdk.runtime.tools.application import _raise_if_llm_node_error

    successful_reply = AIMessage(content="all good now")
    response = {
        "messages": [
            HumanMessage(content="first call"),
            _tagged_error("Error: 401"),
            HumanMessage(content="second call"),
            successful_reply,
        ]
    }
    # Should not raise; the final non-Human message is the successful reply.
    _raise_if_llm_node_error(response)


def test_application_run_does_not_raise_when_current_run_has_no_ai_message():
    """A child pipeline whose result is in state/output (not a final AIMessage) must NOT raise.

    If the current run added a HumanMessage but no AI or Tool reply, the scan walks backwards
    and hits the HumanMessage first. Stopping there prevents it from crossing into a tagged
    error from a *previous* run.
    """
    from elitea_sdk.runtime.tools.application import _raise_if_llm_node_error

    response = {
        "messages": [
            HumanMessage(content="first call"),
            _tagged_error("Error: 401"),
            HumanMessage(content="second call"),
            # current run produced no AI/Tool message — result is in output/state
        ]
    }
    # Must not raise: the scan hits the current run's HumanMessage and stops.
    _raise_if_llm_node_error(response)


# ---------------------------------------------------------------------------
# Propagation through __perform_tool_calling (#7041 end-to-end gap)
# ---------------------------------------------------------------------------

def test_llm_node_error_propagates_through_perform_tool_calling():
    """LlmNodeError from a sub-agent must not be swallowed by the parent's tool-calling loop.

    Before this fix, the generic `except Exception as e` block in __perform_tool_calling
    converted the LlmNodeError into a ToolMessage(status="error") and continued the loop,
    causing the parent to finish successfully and the UI to show "complete" instead of error.

    This test drives __perform_tool_calling directly via asyncio rather than going through
    the full invoke() -> _invoke_llm_internal() stack, which requires extensive mock setup
    (input_mapping, message filtering, tool binding, etc.).
    """
    import asyncio
    from elitea_sdk.runtime.tools.application import Application, LlmNodeError
    from elitea_sdk.runtime.tools.llm import LLMNode
    from langchain_core.messages import AIMessage, HumanMessage

    client = MagicMock()
    client.model_name = "parent-model"
    client._llm_type = "fake"

    node = LLMNode(name="parent", description="parent", client=client)

    tool_call = {"name": "nested_agent", "args": {"task": "do something"}, "id": "call_abc", "type": "tool_call"}
    ai_with_tool_call = AIMessage(content="", tool_calls=[tool_call])

    err = LlmNodeError("Nested agent LLM failed (NotFoundError): model not found")

    # Must spec=Application so isinstance(tool_to_execute, Application) is True in the
    # __perform_tool_calling path — only Application tools use .invoke(), the non-Application
    # branch calls .ainvoke() which would swallow the exception before hitting our guard.
    fake_tool = MagicMock(spec=Application)
    fake_tool.name = "nested_agent"
    fake_tool.invoke.side_effect = err

    messages = [HumanMessage(content="call the agent")]

    # Access the name-mangled async method directly.
    perform_tc = node._LLMNode__perform_tool_calling  # noqa: SLF001

    with patch.object(LLMNode, "_resolve_tool_to_execute", return_value=fake_tool):
        with pytest.raises(LlmNodeError, match="Nested agent LLM failed"):
            asyncio.run(
                perform_tc(
                    ai_with_tool_call,
                    messages,
                    client,
                    config=None,
                )
            )
