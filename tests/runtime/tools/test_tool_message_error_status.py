"""ToolMessage.status on the agent tool-calling path (#6172).

Before this, every tool failure on the agent path shipped with the default
status="success" — the one field on the wire that exists to mark a failed tool call
said the call succeeded. Each test drives the real loop to the site that builds the
message, rather than asserting on a constructor in isolation.

status is Literal["success", "error"], so a declined sensitive tool and a TRUNCATED
result cannot be expressed in it; those two sites deliberately stay "success" and the
AST test at the bottom pins that decision.
"""

import asyncio

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, ToolException
import pytest

from elitea_sdk.runtime.tool_outcome import (
    ToolOutcome,
    ToolResultStatus,
    record_outcome,
    recorded_outcomes,
)
from elitea_sdk.runtime.tools.llm import LLMNode


def _tool(name='read_file', func=None):
    return StructuredTool.from_function(
        func=func or (lambda path='x': 'ok'),
        name=name,
        description=f'{name} tool.',
        metadata={'toolkit_type': 'fs', 'toolkit_name': 'fs', 'tool_name': name},
    )


class _Client:
    """Requests `rounds` tool calls for `call_name`, then answers."""

    def __init__(self, call_name='read_file', rounds=1):
        self.call_name = call_name
        self.rounds = rounds
        self.round = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        self.round += 1
        if self.round <= self.rounds:
            return AIMessage(content='', tool_calls=[
                {'name': self.call_name, 'args': {}, 'id': f'c{self.round}'},
            ])
        return AIMessage(content='done')


def _node(client, tools=None):
    return LLMNode(
        client=client,
        available_tools=tools if tools is not None else [_tool()],
        tool_names=['read_file'],
        lazy_tools_mode=False,
        input_mapping={},
        output_variables=['messages'],
    )


def _tool_messages(messages):
    return [m for m in messages if isinstance(m, ToolMessage)]


def _config(thread_id):
    return {'configurable': {'thread_id': thread_id}}


def test_generic_tool_exception_sets_error_status():
    def _boom(path='x'):
        raise RuntimeError('kaboom')

    result = _node(_Client(), [_tool(func=_boom)]).invoke(
        {'messages': [HumanMessage(content='go')]}, config=_config('t-error'),
    )

    message = _tool_messages(result['messages'])[-1]
    assert message.status == 'error'
    assert 'kaboom' in message.content


def test_missing_tool_sets_error_status():
    result = _node(_Client(call_name='nope')).invoke(
        {'messages': [HumanMessage(content='go')]}, config=_config('t-missing'),
    )

    message = _tool_messages(result['messages'])[-1]
    assert message.status == 'error'
    assert "not available" in message.content


def test_step_limit_placeholder_sets_error_status():
    """The placeholder stands in for a call that never ran, so it is a failure from
    the model's point of view — it must not read as a successful empty result."""
    node = _node(_Client(rounds=5))
    node.steps_limit = 1

    result = node.invoke(
        {'messages': [HumanMessage(content='go')]}, config=_config('t-limit'),
    )

    messages = _tool_messages(result['messages'])
    placeholder = messages[-1]
    assert 'step limit' in placeholder.content
    assert placeholder.status == 'error'
    # The call that did execute keeps its success status.
    assert messages[0].status == 'success'


def test_parallel_subagent_failure_sets_error_status():
    class _ExplodingChild:
        name = 'app_a'

        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError('child exploded')

        def invoke(self, *args, **kwargs):
            raise RuntimeError('child exploded')

    node = _node(_Client(rounds=0))
    completion = AIMessage(content='', tool_calls=[
        {'name': 'app_a', 'args': {}, 'id': 'c1'},
        {'name': 'app_b', 'args': {}, 'id': 'c2'},
    ])

    with pytest.MonkeyPatch.context() as mp:
        import elitea_sdk.runtime.tools.llm as llm_module

        # No parent run outside a graph, so the progress event would raise first.
        mp.setattr(llm_module, 'dispatch_custom_event', lambda *a, **k: None)
        mp.setattr(
            type(node), '_collect_parallel_application_specs',
            lambda self, *a, **k: [
                ('app_a', {}, 'c1', _ExplodingChild()),
                ('app_b', {}, 'c2', _ExplodingChild()),
            ],
        )
        messages, _ = asyncio.run(
            node._LLMNode__perform_tool_calling(
                completion, [HumanMessage(content='go')], node.client,
                _config('t-parallel'),
            )
        )

    failures = _tool_messages(messages)
    assert len(failures) == 2
    assert {m.status for m in failures} == {'error'}
    assert all('child exploded' in m.content for m in failures)


def _middleware_tool(func):
    from elitea_sdk.runtime.middleware.strategies import TransformErrorStrategy
    from elitea_sdk.runtime.middleware.tool_exception_handler import (
        ToolExceptionHandlerMiddleware,
    )

    middleware = ToolExceptionHandlerMiddleware(strategies=[TransformErrorStrategy(llm=None)])
    return middleware.wrap_tool(_tool(func=func))


def test_middleware_caught_exception_sets_error_status():
    """#6809: the middleware returns prose instead of raising, so without the sink the
    loop built a success message over a failed call and the eval trace scored it ok."""
    def _missing(path='x'):
        raise ToolException('File not found: no-slash.txt')

    result = _node(_Client(), [_middleware_tool(_missing)]).invoke(
        {'messages': [HumanMessage(content='go')]}, config=_config('t-middleware'),
    )

    message = _tool_messages(result['messages'])[-1]
    assert message.status == 'error'


def test_truncated_outcome_keeps_success_status():
    def _big(path='x'):
        record_outcome(ToolOutcome(status=ToolResultStatus.TRUNCATED, message='cut'))
        return 'partial'

    result = _node(_Client(), [_tool(func=_big)]).invoke(
        {'messages': [HumanMessage(content='go')]}, config=_config('t-truncated'),
    )

    assert _tool_messages(result['messages'])[-1].status == 'success'


def test_run_callback_sees_the_error_outcome_at_tool_end():
    """The indexer's on_tool_end fires inside the call and reads recorded_outcomes()."""
    seen = []

    class _Recorder(BaseCallbackHandler):
        def on_tool_end(self, output, **kwargs):
            seen.append([o.status for o in recorded_outcomes()])

    def _missing(path='x'):
        raise ToolException('File not found')

    config = {**_config('t-callback'), 'callbacks': [_Recorder()]}
    _node(_Client(), [_middleware_tool(_missing)]).invoke(
        {'messages': [HumanMessage(content='go')]}, config=config,
    )

    assert seen == [[ToolResultStatus.ERROR]]
    # Read outside any call: nothing leaks from the one that just ran.
    assert recorded_outcomes() == []


def test_only_known_failure_sites_carry_error_status():
    """Pin both directions: every explicit status belongs to a known failure path.

    The five current sites cover a generic tool failure, an unavailable tool, a
    step-limit placeholder, a parallel sub-agent failure, and exhausted output
    continuation in a parallel child. Blocked and truncated results deliberately
    retain the default status: ToolMessage.status cannot represent those states.
    """
    import ast
    import inspect

    from elitea_sdk.runtime.tools import llm as llm_module

    tree = ast.parse(inspect.getsource(llm_module))
    statuses = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'ToolMessage'):
            continue
        for keyword in node.keywords:
            if keyword.arg == 'status':
                assert isinstance(keyword.value, ast.Constant)
                statuses.append(keyword.value.value)

    assert statuses == ['error'] * 5
