"""#6886: a sub-agent's input-guardrail block stops the parent run.

Without the re-raise sites, the child's InputGuardrailBlocked is caught by the parent
tool loop and turned into a ToolMessage ("Error: ..."), and the parent model happily
continues. These go through the real Assistant graph so every catch on the way is hit.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.exceptions import InputGuardrailBlocked
from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.tools.application import Application


class _Runtime:
    def get_mcp_toolkits(self):
        return []


class _BlockedChild:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload, config=None):
        self.calls += 1
        raise InputGuardrailBlocked(categories=['credentials'], rules=['credentials.github_token'],
                                    segments=['subagent_task'])


class _OkChild:
    def invoke(self, payload, config=None):
        return {'output': 'fine'}


class _Bound:
    def __init__(self, root):
        self.root = root

    def invoke(self, messages, config=None):
        self.root.calls.append(list(messages))
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content='parent carried on')
        return AIMessage(content='', tool_calls=[
            {'name': name, 'args': {'task': 'Run'}, 'id': f'call-{i}', 'type': 'tool_call'}
            for i, name in enumerate(self.root.tool_names)
        ])


class _ParentLLM:
    def __init__(self, *tool_names):
        self.tool_names = tool_names
        self.calls = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _Bound(self)

    def invoke(self, messages, config=None):
        return _Bound(self).invoke(messages, config=config)


def _app(name, application):
    return Application(name=name, description=f'{name} worker', application=application,
                       return_type='str', client=None, is_subgraph=True)


def _run(llm, tools):
    runnable = Assistant(
        elitea=_Runtime(), data={'instructions': 'Use tools', 'tools': [], 'meta': {}},
        client=llm, tools=tools, memory=MemorySaver(), app_type='predict',
    ).runnable()
    return runnable.invoke({'messages': [HumanMessage(content='delegate')]},
                           config={'configurable': {'thread_id': 'guard-6886'}})


def test_blocked_subagent_stops_the_parent_sequential_loop():
    child = _BlockedChild()
    llm = _ParentLLM('child')

    with pytest.raises(InputGuardrailBlocked) as excinfo:
        _run(llm, [_app('child', child)])

    assert child.calls == 1
    # The parent model never saw a tool result for the blocked call
    assert len(llm.calls) == 1
    assert excinfo.value.segments == ['subagent_task']


def test_blocked_subagent_stops_the_parent_parallel_fan_out():
    llm = _ParentLLM('child_a', 'child_b')

    with pytest.raises(InputGuardrailBlocked):
        _run(llm, [_app('child_a', _OkChild()), _app('child_b', _BlockedChild())])

    assert len(llm.calls) == 1


def test_block_raised_by_a_plain_tool_is_not_tool_prose():
    def guarded(task: str) -> str:
        raise InputGuardrailBlocked(categories=['credentials'])

    tool = StructuredTool.from_function(func=guarded, name='guarded', description='guarded tool')
    llm = _ParentLLM('guarded')

    with pytest.raises(InputGuardrailBlocked):
        _run(llm, [tool])

    assert len(llm.calls) == 1
