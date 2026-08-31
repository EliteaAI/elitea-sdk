"""The handle_tool_errors callable wired into the swarm ToolNodes (#6172).

ToolNode's own default re-raises anything but a ToolInvocationError, which takes the
whole swarm graph down; plain True would swallow the four signal exceptions the
middleware deliberately re-raises. These tests pin both halves of that contract.
"""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphInterrupt
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
import pytest

from elitea_sdk.runtime.exceptions import BudgetExceededError
from elitea_sdk.runtime.middleware.tool_exception_handler import swarm_handle_tool_errors
from elitea_sdk.runtime.tool_outcome import ToolErrorClass, outcome_sink
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired


def test_generic_exception_becomes_prose():
    assert swarm_handle_tool_errors(RuntimeError("boom")) == "Error executing tool: boom"


def test_generic_exception_records_an_error_outcome():
    with outcome_sink() as recorded:
        swarm_handle_tool_errors(TimeoutError("upstream timed out"))

    assert len(recorded) == 1
    outcome = recorded[0]
    assert outcome.status == "error"
    assert outcome.error_class is ToolErrorClass.INFRASTRUCTURE
    assert outcome.retriable is True
    assert outcome.exception_type == "TimeoutError"


@pytest.mark.parametrize("signal", [
    GraphInterrupt(),
    McpAuthorizationRequired("auth needed", "https://mcp.example"),
    NotImplementedError("no sync path"),
    BudgetExceededError("project budget exhausted"),
])
def test_signal_exceptions_are_reraised(signal):
    """These four are control flow, not tool failures — turning them into prose would
    silently drop an MCP auth interrupt or feed a budget rejection back to the model."""
    with pytest.raises(type(signal)):
        swarm_handle_tool_errors(signal)


def _graph_with(tool):
    """ToolNode needs the graph runtime around it: invoked bare it raises over a missing
    config key before the tool ever runs, which would test nothing."""
    node = ToolNode([tool], handle_tool_errors=swarm_handle_tool_errors)
    builder = StateGraph(MessagesState)
    builder.add_node("tools", node)
    builder.set_entry_point("tools")
    builder.set_finish_point("tools")
    return builder.compile()


def _call(name):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": "call_1"}])


def test_tool_node_produces_an_error_status_message():
    """End to end through ToolNode, so _infer_handled_types actually reads the
    Exception annotation on the callable rather than us trusting it does."""
    def _explode() -> str:
        raise RuntimeError("tool blew up")

    tool = StructuredTool.from_function(func=_explode, name="explode", description="x")

    result = _graph_with(tool).invoke({"messages": [_call("explode")]})

    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert "tool blew up" in message.content


def test_tool_node_lets_a_signal_escape():
    def _needs_auth() -> str:
        raise McpAuthorizationRequired("auth needed", "https://mcp.example")

    tool = StructuredTool.from_function(func=_needs_auth, name="mcp_tool", description="x")

    with pytest.raises(McpAuthorizationRequired):
        _graph_with(tool).invoke({"messages": [_call("mcp_tool")]})
