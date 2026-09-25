"""tool_execution_timeout bounds a nested agent's tool loop, and must be configurable.

A node invoked from inside a running event loop (the agent-as-tool shape) runs its
tool-calling loop in a worker thread joined with tool_execution_timeout. Before the
fix this was a hardcoded 900s for agents: long sub-agents were cut at exactly 15
minutes and the parent received "Error: Async operation in thread timed out".
"""

import asyncio
import time
from unittest.mock import patch

import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.tools.llm import (
    DEFAULT_TOOL_EXECUTION_TIMEOUT,
    TOOL_EXECUTION_TIMEOUT_ENV,
    LLMNode,
    default_tool_execution_timeout,
    normalize_tool_execution_timeout,
)

SLOW_TOOL_SECONDS = 1.5


def _slow_tool():
    def slow(path='x'):
        time.sleep(SLOW_TOOL_SECONDS)
        return 'ok'

    return StructuredTool.from_function(
        func=slow,
        name='slow_tool',
        description='Slow tool.',
        metadata={'toolkit_type': 't', 'toolkit_name': 't', 'tool_name': 'slow_tool'},
    )


class OneToolCallLLM:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content='', tool_calls=[{'name': 'slow_tool', 'args': {}, 'id': 'c1'}])
        return AIMessage(content='final answer')


def _node(**kwargs):
    return LLMNode(
        client=OneToolCallLLM(),
        available_tools=[_slow_tool()],
        tool_names=['slow_tool'],
        lazy_tools_mode=False,
        input_mapping={},
        output_variables=['messages'],
        **kwargs,
    )


def _invoke_inside_running_loop(node):
    async def run():
        return node.invoke({'messages': [HumanMessage(content='go')]})

    return asyncio.run(run())


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(TOOL_EXECUTION_TIMEOUT_ENV, raising=False)


# --- resolution -----------------------------------------------------------

def test_default_is_900_without_env():
    assert default_tool_execution_timeout() == DEFAULT_TOOL_EXECUTION_TIMEOUT == 900
    assert LLMNode(client=None).tool_execution_timeout == 900


@pytest.mark.parametrize('raw, expected', [
    ('3600', 3600.0),
    ('0', None),
    ('none', None),
    ('', None),
    ('garbage', DEFAULT_TOOL_EXECUTION_TIMEOUT),
])
def test_env_overrides_default(monkeypatch, raw, expected):
    monkeypatch.setenv(TOOL_EXECUTION_TIMEOUT_ENV, raw)
    assert default_tool_execution_timeout() == expected
    assert LLMNode(client=None).tool_execution_timeout == expected


@pytest.mark.parametrize('value, expected', [
    (None, None),
    (0, None),
    (-5, None),
    (30, 30.0),
    ('45', 45.0),
])
def test_normalize(value, expected):
    assert normalize_tool_execution_timeout(value) == expected


# --- behaviour in the nested (running-loop) path --------------------------

def test_nested_loop_is_cut_at_configured_timeout():
    started = time.time()
    result = _invoke_inside_running_loop(_node(tool_execution_timeout=0.3))
    elapsed = time.time() - started

    assert elapsed < SLOW_TOOL_SECONDS
    content = result['messages'][-1].content
    assert 'timed out after 0.3s' in content


@pytest.mark.parametrize('timeout', [None, 0])
def test_nested_loop_without_limit_runs_to_completion(timeout):
    result = _invoke_inside_running_loop(_node(tool_execution_timeout=timeout))
    assert result['messages'][-1].content == 'final answer'


def test_nested_loop_honours_env_default(monkeypatch):
    monkeypatch.setenv(TOOL_EXECUTION_TIMEOUT_ENV, '0.3')
    result = _invoke_inside_running_loop(_node())
    assert 'timed out after 0.3s' in result['messages'][-1].content


# --- agent wiring ---------------------------------------------------------

class _DummyElitea:
    def get_mcp_toolkits(self):
        return []


def _agent_node_schema(meta):
    with patch('elitea_sdk.runtime.langchain.langraph_agent.create_graph') as create_graph:
        Assistant(
            elitea=_DummyElitea(),
            data={'instructions': 'x', 'tools': [], 'meta': meta},
            client=OneToolCallLLM(),
            app_type='predict',
        ).runnable()
    schema = yaml.safe_load(create_graph.call_args.kwargs['yaml_schema'])
    return schema['nodes'][0]


@pytest.mark.parametrize('value', [3600, None, 0])
def test_agent_meta_timeout_reaches_llm_node(value):
    assert _agent_node_schema({'tool_execution_timeout': value})['tool_execution_timeout'] == value


def test_agent_without_meta_timeout_uses_deployment_default():
    assert 'tool_execution_timeout' not in _agent_node_schema({})


def _built_agent_llm_node(meta):
    built = []

    class RecordingLLMNode(LLMNode):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            built.append(self)

    with patch('elitea_sdk.runtime.langchain.langraph_agent.LLMNode', RecordingLLMNode):
        Assistant(
            elitea=_DummyElitea(),
            data={'instructions': 'x', 'tools': [], 'meta': meta},
            client=OneToolCallLLM(),
            tools=[_slow_tool()],
            app_type='predict',
            lazy_tools_mode=False,
        ).runnable()
    assert len(built) == 1
    return built[0]


def test_built_agent_node_uses_env_default(monkeypatch):
    monkeypatch.setenv(TOOL_EXECUTION_TIMEOUT_ENV, '1234')
    assert _built_agent_llm_node({}).tool_execution_timeout == 1234.0


def test_built_agent_node_meta_beats_env(monkeypatch):
    monkeypatch.setenv(TOOL_EXECUTION_TIMEOUT_ENV, '1234')
    assert _built_agent_llm_node({'tool_execution_timeout': None}).tool_execution_timeout is None
