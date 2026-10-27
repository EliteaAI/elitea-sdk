"""Regression tests for #7193: stale-checkpoint cleanup racing another write to the same thread.

A new turn reads the checkpoint, then clears it. If another run rewrites the thread
in between (summarization, or a second new turn clearing first), per-ID RemoveMessage
for the stale IDs used to raise ValueError from the add_messages reducer.
"""
from types import SimpleNamespace
from typing import Annotated, Optional, TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

from elitea_sdk.runtime.langchain.langraph_agent import LangGraphAgentRunnable


class _State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    hitl_decisions: Optional[dict]
    context_info: Optional[dict]


def _graph_with_finished_turn():
    builder = StateGraph(_State)
    builder.add_node("agent", lambda state: {"messages": [AIMessage(content="done")]})
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke({"messages": [HumanMessage(content="first turn")]}, config)
    return graph, config


def _clear(graph, config, checkpoint_state):
    # Only update_state is used by the method, so the compiled graph stands in for self
    LangGraphAgentRunnable._clear_stale_checkpoint(graph, config, checkpoint_state)


def _summarize(graph, config):
    # Shape the summarization middleware writes: drop everything, keep a new tail
    graph.update_state(config, {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES),
                                             AIMessage(content="preserved tail")]})


def _append(graph, config):
    graph.update_state(config, {"messages": [AIMessage(content="another tool step")]})


def _concurrent_new_turn(graph, config):
    _clear(graph, config, graph.get_state(config))


@pytest.mark.parametrize("other_write", [_summarize, _append, _concurrent_new_turn],
                         ids=["summarized", "appended", "concurrent_new_turn"])
def test_cleanup_survives_checkpoint_changed_after_read(other_write):
    graph, config = _graph_with_finished_turn()
    stale = graph.get_state(config)

    other_write(graph, config)
    _clear(graph, config, stale)

    # Messages written after the read must be cleared too, not leak into the new turn
    assert graph.get_state(config).values["messages"] == []


def test_new_input_runs_on_cleared_thread():
    graph, config = _graph_with_finished_turn()
    stale = graph.get_state(config)
    _summarize(graph, config)
    _clear(graph, config, stale)

    result = graph.invoke({"messages": [HumanMessage(content="second turn")]}, config)

    assert [m.content for m in result["messages"]] == ["second turn", "done"]


def test_flags_cleared_in_one_update():
    graph, config = _graph_with_finished_turn()
    graph.update_state(config, {"hitl_decisions": {"a": 1}, "context_info": {"b": 2}})
    stale = graph.get_state(config)

    _clear(graph, config, stale)

    values = graph.get_state(config).values
    assert values["messages"] == []
    assert values.get("hitl_decisions") is None
    assert values.get("context_info") is None


def test_empty_checkpoint_writes_nothing():
    from unittest.mock import MagicMock

    fake_self = MagicMock()
    LangGraphAgentRunnable._clear_stale_checkpoint(fake_self, {}, SimpleNamespace(values={}))

    fake_self.update_state.assert_not_called()
