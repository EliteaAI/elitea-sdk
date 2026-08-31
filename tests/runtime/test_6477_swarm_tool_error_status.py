"""Swarm tool failures must ship ToolMessage(status="error") (#6477).

On the swarm path the tools handed to ToolNode are already wrapped by
ToolExceptionHandlerMiddleware, whose wrappers *return* the error prose instead of raising
it. ToolNode therefore treats the call as a success and builds a success-status message
over error content; handle_tool_errors (#6172) never fires because nothing was raised.
The wrap_tool_call hooks read the recorded ToolOutcome envelope and stamp the status —
structurally, never by looking at the message text.
"""

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from elitea_sdk.runtime.middleware.strategies import LoggingStrategy
from elitea_sdk.runtime.middleware.tool_exception_handler import (
    ToolExceptionHandlerMiddleware,
    swarm_awrap_tool_call,
    swarm_handle_tool_errors,
    swarm_wrap_tool_call,
)
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired


def _wrapped(tool):
    """Same middleware the swarm path applies, with a strategy that only observes."""
    middleware = ToolExceptionHandlerMiddleware(strategies=[LoggingStrategy()])
    return middleware.wrap_tool(tool)


def _graph_with(tool):
    """ToolNode needs the graph runtime around it: invoked bare it raises over a missing
    config key before the tool ever runs."""
    node = ToolNode(
        [tool],
        handle_tool_errors=swarm_handle_tool_errors,
        wrap_tool_call=swarm_wrap_tool_call,
        awrap_tool_call=swarm_awrap_tool_call,
    )
    builder = StateGraph(MessagesState)
    builder.add_node("tools", node)
    builder.set_entry_point("tools")
    builder.set_finish_point("tools")
    return builder.compile()


def _call(name):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": "call_1"}])


def _last(result):
    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    return message


def test_wrapped_failing_tool_reports_error_status():
    def _explode() -> str:
        raise RuntimeError("tool blew up")

    tool = _wrapped(StructuredTool.from_function(func=_explode, name="explode", description="x"))

    message = _last(_graph_with(tool).invoke({"messages": [_call("explode")]}))

    assert message.status == "error"
    assert "tool blew up" in message.content


@pytest.mark.asyncio
async def test_wrapped_failing_tool_reports_error_status_async():
    async def _explode() -> str:
        raise RuntimeError("async blew up")

    tool = _wrapped(StructuredTool.from_function(
        coroutine=_explode, name="explode_async", description="x",
    ))

    message = _last(await _graph_with(tool).ainvoke({"messages": [_call("explode_async")]}))

    assert message.status == "error"
    assert "async blew up" in message.content


def test_success_returning_error_looking_text_stays_success():
    """The whole point of reading the envelope: prose-sniffing would fail this one."""
    def _no_results() -> str:
        return "Error executing tool: nothing found"

    tool = _wrapped(StructuredTool.from_function(func=_no_results, name="search", description="x"))

    message = _last(_graph_with(tool).invoke({"messages": [_call("search")]}))

    assert message.status == "success"


def test_content_and_artifact_failure_reports_error_status():
    """_shape_error_output returns a two-tuple for these; the status must still land."""
    def _explode() -> str:
        raise RuntimeError("artifact tool blew up")

    tool = _wrapped(StructuredTool.from_function(
        func=_explode, name="explode_artifact", description="x",
        response_format="content_and_artifact",
    ))

    message = _last(_graph_with(tool).invoke({"messages": [_call("explode_artifact")]}))

    assert message.status == "error"


def test_signal_from_a_wrapped_tool_still_escapes():
    def _needs_auth() -> str:
        raise McpAuthorizationRequired("auth needed", "https://mcp.example")

    tool = _wrapped(StructuredTool.from_function(func=_needs_auth, name="mcp_tool", description="x"))

    with pytest.raises(McpAuthorizationRequired):
        _graph_with(tool).invoke({"messages": [_call("mcp_tool")]})


def test_unwrapped_raising_tool_is_unaffected():
    """The #6172 path: nothing wrapped it, so handle_tool_errors is what fires."""
    def _explode() -> str:
        raise RuntimeError("bare tool blew up")

    tool = StructuredTool.from_function(func=_explode, name="bare", description="x")

    message = _last(_graph_with(tool).invoke({"messages": [_call("bare")]}))

    assert message.status == "error"
    assert "bare tool blew up" in message.content
