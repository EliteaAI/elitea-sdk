"""The run-level error holder in the agent tool loop: a missing tool counts as a failure, a retried tool does not."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.middleware.strategies import LoggingStrategy
from elitea_sdk.runtime.middleware.tool_exception_handler import ToolExceptionHandlerMiddleware
from elitea_sdk.runtime.tool_outcome import run_error_flag


class _Runtime:
    def get_mcp_toolkits(self):
        return []


class _ScriptedLLM:
    """Asks for the given tool names one per turn, then answers."""
    temperature = 0
    max_tokens = 1000
    _get_model_default_parameters = {'temperature': 0, 'max_tokens': 1000}

    def __init__(self, tool_names):
        self.tool_names = list(tool_names)

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        done = sum(isinstance(m, ToolMessage) for m in messages)
        if done >= len(self.tool_names):
            return AIMessage(content='FINAL')
        return AIMessage(content='', tool_calls=[
            {'name': self.tool_names[done], 'args': {'item': 'x'}, 'id': f'call-{done}'},
        ])


def _tool(func, name):
    return StructuredTool.from_function(
        func=func, name=name, description='A tool.',
        metadata={'toolkit_type': 'jira', 'toolkit_name': 'jira', 'tool_name': name},
    )


def _run(tool_names, tools):
    assistant = Assistant(
        elitea=_Runtime(),
        data={'instructions': 'Use tools', 'tools': [], 'meta': {}},
        client=_ScriptedLLM(tool_names),
        tools=tools,
        memory=MemorySaver(),
        app_type='predict',
        middleware=[ToolExceptionHandlerMiddleware(strategies=[LoggingStrategy()])],
    )
    with run_error_flag() as holder:
        result = assistant.runnable().invoke(
            {'messages': [HumanMessage(content='go')]}, config={'configurable': {'thread_id': 't'}},
        )
    return holder, result


def test_a_call_to_a_tool_that_is_not_there_marks_the_run():
    holder, result = _run(['ghost'], [_tool(lambda item: 'ok', 'real')])

    assert result['output'] == 'FINAL'
    assert holder['error']['tool'] == 'ghost'
    assert 'not available' in holder['error']['reason']


def test_a_tool_that_fails_once_and_then_works_leaves_the_run_clean():
    calls = []

    def flaky(item):
        calls.append(item)
        if len(calls) == 1:
            raise RuntimeError('first try failed')
        return 'ok'

    holder, result = _run(['flaky', 'flaky'], [_tool(flaky, 'flaky')])

    assert len(calls) == 2
    assert result['output'] == 'FINAL'
    assert holder == {}


def test_a_tool_that_keeps_failing_marks_the_run():
    def broken(item):
        raise RuntimeError('still broken')

    holder, _ = _run(['broken', 'broken'], [_tool(broken, 'broken')])

    assert holder['error']['tool'] == 'broken'
    assert 'still broken' in holder['error']['reason']


def test_a_failure_of_one_tool_is_not_cleared_by_another_tool():
    def broken(item):
        raise RuntimeError('boom')

    holder, _ = _run(['broken', 'fine'], [_tool(broken, 'broken'), _tool(lambda item: 'ok', 'fine')])

    assert holder['error']['tool'] == 'broken'
