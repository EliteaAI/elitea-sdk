"""
#7128: every run of a pipeline Agent node gets its own ``parent_agent_call_id``.

The UI groups a sub-agent's Tool Call History entry by that id. An Agent node
calls the agent directly (no LLM tool_call_id), so before the fix every run of
the node carried no id and all loop passes merged into one entry.
"""
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from elitea_sdk.runtime.tools.application import Application
from elitea_sdk.runtime.tools.function import FunctionTool


class RecordingApplication:
    """Child runnable that records the config each run sees."""

    def __init__(self, pause_on_call=None):
        self.calls = []
        self.pause_on_call = pause_on_call

    def invoke(self, payload, config=None):
        self.calls.append({'payload': payload, 'config': config})
        if self.pause_on_call is not None and len(self.calls) == self.pause_on_call:
            interrupt('approve?')
        return {'output': 'ok'}


class LoopState(TypedDict, total=False):
    messages: list
    task: str
    n: int


def _agent_node(child, name='agent_node'):
    application = Application(
        name='child_agent',
        description='Nested agent',
        application=child,
        return_type='str',
        client=None,
        is_subgraph=True,
        metadata={'display_name': 'child_agent'},
    )
    return FunctionTool(
        tool=application, name=name, return_type='str',
        input_variables=['task'],
        input_mapping={'task': {'type': 'variable', 'value': 'task'}},
        output_variables=['messages'],
    )


def _loop_graph(node, passes):
    def agent(state, config):
        node.invoke(state, config)
        return {'n': state.get('n', 0) + 1}

    builder = StateGraph(LoopState)
    builder.add_node('agent_node', agent)
    builder.set_entry_point('agent_node')
    builder.add_conditional_edges(
        'agent_node', lambda s: END if s['n'] >= passes else 'agent_node')
    return builder.compile(checkpointer=MemorySaver())


def _call_ids(child):
    return [c['config']['metadata'].get('parent_agent_call_id') for c in child.calls]


def test_each_loop_pass_gets_its_own_call_id():
    child = RecordingApplication()
    graph = _loop_graph(_agent_node(child), passes=3)

    graph.invoke({'task': 'go', 'n': 0, 'messages': []},
                 {'configurable': {'thread_id': 'loop-7128'}})

    ids = _call_ids(child)
    assert len(ids) == 3
    assert all(i and i.startswith('pipeline:agent_node:') for i in ids)
    assert len(set(ids)) == 3


def test_resumed_run_keeps_its_call_id():
    child = RecordingApplication(pause_on_call=1)
    graph = _loop_graph(_agent_node(child), passes=2)
    config = {'configurable': {'thread_id': 'loop-7128-hitl'}}

    graph.invoke({'task': 'go', 'n': 0, 'messages': []}, config)
    graph.invoke(Command(resume='approved'), config)

    ids = _call_ids(child)
    # Pass 1 ran, paused, then re-ran on resume; pass 2 ran once.
    assert len(ids) == 3
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]


def test_inherited_call_id_is_overridden_by_the_node():
    child = RecordingApplication()
    node = _agent_node(child)

    node.invoke({'task': 'go', 'messages': []}, {
        'metadata': {'parent_agent_call_id': 'toolu_outer'},
        'configurable': {'thread_id': 't', '__pregel_task_id': 'task-1'},
    })

    assert _call_ids(child) == ['pipeline:agent_node:task-1']


def test_tool_call_invocation_still_uses_the_llm_call_id():
    child = RecordingApplication()
    application = _agent_node(child).tool

    application.invoke(
        {'type': 'tool_call', 'id': 'call-A', 'name': 'child_agent',
         'args': {'task': 'go', 'chat_history': []}},
        config={'configurable': {'thread_id': 't'}},
    )

    assert _call_ids(child) == ['call-A']
