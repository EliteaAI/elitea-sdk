"""
Smoke test for issue #4935: duplicate tool names reaching bind_tools on the
toolnode-react path when lazy_tools_mode=True.

Anthropic's API rejects requests with duplicate tool names (HTTP 400). OpenAI
silently accepts them, so the bug was only visible on the Anthropic path with
smart tool selection (lazy_tools_mode=True) and subagent execution
(Application tool present, routes to _create_toolnode_react_agent).

The fix: deduplicate_tool_names() is now called in _create_toolnode_react_agent
before bind_tools(), mirroring the existing call at swarm-main line ~1132.
"""
from unittest.mock import MagicMock, patch
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.tools.application import Application


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyEliteaRuntime:
    def get_mcp_toolkits(self):
        return []


class _EmptyArgs(BaseModel):
    pass


class _RealBaseTool(BaseTool):
    """Minimal concrete BaseTool subclass. Avoids MagicMock recursion in ToolNode."""
    name: str
    description: str = "test tool"
    args_schema: type[BaseModel] = _EmptyArgs

    def _run(self, *args, **kwargs):
        return "ok"


def _make_tool(name: str) -> _RealBaseTool:
    return _RealBaseTool(name=name)


def _make_application_tool(name: str) -> Application:
    subapp = MagicMock()
    subapp.invoke.return_value = {"output": "done"}
    return Application(
        name=name,
        description="A subagent application tool",
        application=subapp,
        return_type="str",
        client=None,
        is_subgraph=True,
    )


class CapturingLLM:
    """Fake LLM that records the tool list passed to bind_tools."""

    def __init__(self):
        self.bound_tool_names: list[str] = []
        self.temperature = 0
        self.max_tokens = 1000

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [t.name for t in tools]
        return _BoundCapturingLLM()

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}


class _BoundCapturingLLM:
    def invoke(self, messages, config=None):
        from langchain_core.messages import AIMessage
        return AIMessage(content="task complete")


def _build_assistant(tools, lazy_tools_mode: bool, llm) -> Assistant:
    return Assistant(
        elitea=DummyEliteaRuntime(),
        data={"instructions": "Use tools", "tools": [], "meta": {}},
        client=llm,
        tools=tools,
        memory=MemorySaver(),
        app_type="predict",
        lazy_tools_mode=lazy_tools_mode,
    )


# ---------------------------------------------------------------------------
# Tests: toolnode-react path (the fix)
#
# We patch ToolNode so it accepts any tools list without doing Pydantic schema
# introspection — that keeps tests isolated to the dedup concern. The patch
# target is the name imported inside the method body.
# ---------------------------------------------------------------------------

def _patch_toolnode():
    """Return a context manager that replaces ToolNode with a no-op constructor.

    ToolNode is imported with `from langgraph.prebuilt import ToolNode` inside the
    method body, so it is not a module-level name on assistant.py. The correct
    target is the source location: langgraph.prebuilt.ToolNode.
    """
    mock_toolnode_cls = MagicMock(return_value=MagicMock())
    return patch("langgraph.prebuilt.ToolNode", mock_toolnode_cls)


def test_toolnode_react_dedup_fires_before_bind_tools_lazy_mode():
    """
    Core regression: lazy_tools_mode=True skips init-time dedup.
    After the fix, bind_tools receives a list with no duplicate names.
    """
    llm = CapturingLLM()
    tools = [
        _make_application_tool("search_data"),
        _make_tool("index_data"),
        _make_tool("index_data"),  # duplicate — would cause Anthropic HTTP 400
    ]

    assistant = _build_assistant(tools, lazy_tools_mode=True, llm=llm)
    with _patch_toolnode():
        assistant.runnable()

    assert len(llm.bound_tool_names) == len(set(llm.bound_tool_names)), (
        f"Duplicate tool names reached bind_tools: {llm.bound_tool_names}"
    )
    # Verify dedup happened (duplicate was renamed, not dropped)
    assert len(llm.bound_tool_names) == 3


def test_toolnode_react_dedup_fires_before_bind_tools_non_lazy_mode():
    """
    Non-lazy path: init-time dedup already ran. The second dedup call in
    _create_toolnode_react_agent must be a no-op and must not error.
    """
    llm = CapturingLLM()
    tools = [
        _make_application_tool("search_data"),
        _make_tool("index_data"),
        _make_tool("index_data"),  # init-time dedup handles this
    ]

    assistant = _build_assistant(tools, lazy_tools_mode=False, llm=llm)
    with _patch_toolnode():
        assistant.runnable()

    assert len(llm.bound_tool_names) == len(set(llm.bound_tool_names)), (
        f"Duplicate tool names reached bind_tools: {llm.bound_tool_names}"
    )
    assert len(llm.bound_tool_names) == 3


def test_toolnode_react_dedup_handles_three_way_collision():
    """
    Stress: three tools all named 'run_query'.
    After dedup: ['run_query', 'run_query_1', 'run_query_2'] — all unique.
    """
    llm = CapturingLLM()
    tools = [
        _make_application_tool("execute_subagent"),
        _make_tool("run_query"),
        _make_tool("run_query"),
        _make_tool("run_query"),
    ]

    assistant = _build_assistant(tools, lazy_tools_mode=True, llm=llm)
    with _patch_toolnode():
        assistant.runnable()

    assert len(llm.bound_tool_names) == len(set(llm.bound_tool_names)), (
        f"Duplicate tool names reached bind_tools: {llm.bound_tool_names}"
    )
    assert "run_query" in llm.bound_tool_names
    assert "run_query_1" in llm.bound_tool_names
    assert "run_query_2" in llm.bound_tool_names


# ---------------------------------------------------------------------------
# Regression guards: verify the fix and its parallel (swarm) are both present
# ---------------------------------------------------------------------------

def test_toolnode_react_path_calls_deduplicate_tool_names():
    """
    Source-level guard: _create_toolnode_react_agent must call
    deduplicate_tool_names (the fix for #4935).
    """
    import inspect
    import elitea_sdk.runtime.langchain.assistant as mod

    src = inspect.getsource(mod.Assistant._create_toolnode_react_agent)
    assert "deduplicate_tool_names" in src, (
        "_create_toolnode_react_agent does not call deduplicate_tool_names — "
        "the fix for #4935 is missing."
    )


def test_swarm_path_still_calls_deduplicate_tool_names():
    """
    Regression guard: _create_swarm_agent must still call
    deduplicate_tool_names (the parallel fix introduced before #4935).
    """
    import inspect
    import elitea_sdk.runtime.langchain.assistant as mod

    src = inspect.getsource(mod.Assistant._create_swarm_agent)
    assert "deduplicate_tool_names" in src, (
        "_create_swarm_agent no longer calls deduplicate_tool_names — "
        "the swarm regression guard has broken."
    )
