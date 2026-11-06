"""One aggregate pause for the sensitive tool calls of one AI message.

The tests run the real graph (Assistant.runnable() on a shared MemorySaver). Every
phase builds a fresh runnable, as every HTTP request does in production.
"""
import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.clients.client import EliteAClient
from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.middleware.sensitive_tool_guard import SensitiveToolGuardMiddleware
from elitea_sdk.runtime.middleware.strategies import LoggingStrategy
from elitea_sdk.runtime.middleware.tool_exception_handler import ToolExceptionHandlerMiddleware
from elitea_sdk.runtime.toolkits.security import configure_sensitive_tools, reset_sensitive_tools

PENDING_KEYS = {'interrupt_id', 'tool_call_id', 'toolkit_name', 'tool_name'}


def setup_function() -> None:
    reset_sensitive_tools()


def teardown_function() -> None:
    reset_sensitive_tools()


class _DummyRuntime:
    def get_mcp_toolkits(self):
        return []


class _Recorder:
    """Shared side-effect log: one entry for every executed tool."""

    def __init__(self):
        self.executed: list[str] = []


class _BatchLLM:
    """Emits one AI message with all given calls, then a final answer.

    With ``later_calls`` it emits a second AI message after the first round ran.
    """

    def __init__(self, calls: list[tuple[str, str]], later_calls: list[tuple[str, str]] | None = None):
        self.calls = calls
        self.later_calls = later_calls or []
        self.invocations: list[list] = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        self.invocations.append(list(messages))
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if len(results) >= len(self.calls) + len(self.later_calls):
            return AIMessage(content='done: ' + '|'.join(str(m.content) for m in results))
        wanted = self.calls if len(results) < len(self.calls) else self.later_calls
        return AIMessage(
            content='',
            tool_calls=[
                {'name': name, 'args': {'path': f'/{name}'}, 'id': call_id}
                for name, call_id in wanted
            ],
        )


def _make_tools(recorder: _Recorder, async_unsupported: bool = False):
    def _tool(name):
        def _run(path=''):
            recorder.executed.append(name)
            return f'ran:{name}'

        async def _arun(path=''):
            raise NotImplementedError('no async path')

        return StructuredTool.from_function(
            func=_run,
            coroutine=_arun if async_unsupported else None,
            name=name,
            description=f'{name} tool',
            metadata={'toolkit_type': 'art', 'toolkit_name': 'Artifact', 'tool_name': name},
        )
    return [_tool(n) for n in ('create_file', 'delete_file', 'update_file', 'read_file')]


def _runnable(memory, llm, recorder, batch: bool = True, async_unsupported: bool = False):
    # Production order: the exception handler first, the sensitive guard last.
    middleware = [SensitiveToolGuardMiddleware(batch_approvals=batch)]
    if async_unsupported:
        middleware.insert(0, ToolExceptionHandlerMiddleware(
            strategies=[LoggingStrategy()], conversation_id='conv',
        ))
    assistant = Assistant(
        elitea=_DummyRuntime(),
        data={'instructions': 'Use tools.', 'tools': [], 'meta': {}},
        client=llm,
        tools=_make_tools(recorder, async_unsupported),
        memory=memory,
        app_type='predict',
        middleware=middleware,
    )
    return assistant.runnable()


def _start(calls, batch=True, sensitive=('create_file', 'delete_file', 'update_file'), thread='t',
           later_calls=None, async_unsupported=False):
    configure_sensitive_tools({'art': list(sensitive)})
    memory = MemorySaver()
    recorder = _Recorder()
    cfg = {'configurable': {'thread_id': thread}}
    llm = _BatchLLM(calls, later_calls)
    first = _runnable(memory, llm, recorder, batch, async_unsupported).invoke(
        {'messages': [HumanMessage(content='go')]}, config=dict(cfg),
    )
    return memory, recorder, cfg, first


def _resume(memory, recorder, cfg, calls, decisions, batch=True, later_calls=None,
            async_unsupported=False):
    llm = _BatchLLM(calls, later_calls)
    payload = {'hitl_resume': True, 'hitl_decisions': decisions}
    if not batch:
        payload = {'hitl_resume': True, 'hitl_action': decisions, 'hitl_value': ''}
    result = _runnable(memory, llm, recorder, batch, async_unsupported).invoke(
        payload, config=dict(cfg),
    )
    return result, llm


def _decide(pending, action='approve', only=None, value=''):
    return [
        {'interrupt_id': p['interrupt_id'], 'action': action, 'value': value}
        for p in pending
        if only is None or p['tool_name'] in only
    ]


CALLS2 = [('create_file', 'c1'), ('delete_file', 'c2')]
CALLS3 = [('create_file', 'c1'), ('delete_file', 'c2'), ('update_file', 'c3')]


class TestBatchPause:
    def test_two_sensitive_calls_raise_one_aggregate_pause(self):
        _, recorder, _, first = _start(CALLS2)

        assert first['execution_finished'] is False
        assert recorder.executed == []
        pending = first['hitl_interrupts']
        assert [p['tool_name'] for p in pending] == ['create_file', 'delete_file']
        assert [p['tool_call_id'] for p in pending] == ['c1', 'c2']
        assert first['hitl_interrupt']['interrupt_id'] == pending[0]['interrupt_id']

    def test_three_sensitive_calls_raise_one_aggregate_pause(self):
        _, recorder, _, first = _start(CALLS3)

        assert [p['tool_name'] for p in first['hitl_interrupts']] == [
            'create_file', 'delete_file', 'update_file',
        ]
        assert recorder.executed == []

    def test_pending_entries_carry_names_only(self):
        _, _, _, first = _start(CALLS2)

        for entry in first['hitl_interrupts']:
            assert PENDING_KEYS <= set(entry)
            assert entry['toolkit_name'] == 'Artifact'
            assert 'tool_args' not in entry
            assert '/create_file' not in json.dumps(entry)
        ids = {p['interrupt_id'] for p in first['hitl_interrupts']}
        assert len(ids) == 2

    def test_aggregate_payload_keys(self):
        memory, _, cfg, first = _start(CALLS2)
        snapshot = _runnable(memory, _BatchLLM(CALLS2), _Recorder()).get_state(dict(cfg))
        raw = [
            i.value for t in snapshot.tasks for i in t.interrupts if isinstance(i.value, dict)
        ]

        assert len(raw) == 1
        aggregate = raw[0]
        assert aggregate['type'] == 'hitl'
        assert aggregate['guardrail_type'] == 'parallel_sensitive_tools'
        assert 'batch' not in aggregate
        assert isinstance(aggregate['pending'], list) and len(aggregate['pending']) == 2
        for entry in aggregate['pending']:
            assert PENDING_KEYS <= set(entry)

    def test_all_approve_runs_calls_in_message_order(self):
        memory, recorder, cfg, first = _start(CALLS3)

        result, llm = _resume(memory, recorder, cfg, CALLS3, _decide(first['hitl_interrupts']))

        assert result['execution_finished'] is True
        assert recorder.executed == ['create_file', 'delete_file', 'update_file']
        assert 'hitl_interrupts' not in result or not result['hitl_interrupts']
        assert len(llm.invocations) == 1

    def test_all_reject_blocks_every_call(self):
        memory, recorder, cfg, first = _start(CALLS2)

        result, llm = _resume(
            memory, recorder, cfg, CALLS2, _decide(first['hitl_interrupts'], action='reject'),
        )

        assert result['execution_finished'] is True
        assert recorder.executed == []
        blocked = [m for m in llm.invocations[-1] if isinstance(m, ToolMessage)]
        assert len(blocked) == 2
        assert all(SensitiveToolGuardMiddleware.BLOCKED_TOOL_RESULT_TYPE in str(m.content) for m in blocked)

    def test_mixed_decisions_run_only_approved_call(self):
        memory, recorder, cfg, first = _start(CALLS3)
        pending = first['hitl_interrupts']
        decisions = (
            _decide(pending, only={'create_file'})
            + _decide(pending, action='reject', only={'delete_file'})
            + _decide(pending, only={'update_file'})
        )

        result, llm = _resume(memory, recorder, cfg, CALLS3, decisions)

        assert result['execution_finished'] is True
        assert recorder.executed == ['create_file', 'update_file']
        results = [m for m in llm.invocations[-1] if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in results] == ['c1', 'c2', 'c3']
        assert SensitiveToolGuardMiddleware.BLOCKED_TOOL_RESULT_TYPE in str(results[1].content)

    def test_non_sensitive_call_runs_once_and_keeps_order(self):
        calls = [('read_file', 'c0'), ('create_file', 'c1'), ('delete_file', 'c2')]
        memory, recorder, cfg, first = _start(calls)

        assert [p['tool_name'] for p in first['hitl_interrupts']] == ['create_file', 'delete_file']
        assert recorder.executed == []

        result, _ = _resume(memory, recorder, cfg, calls, _decide(first['hitl_interrupts']))

        assert result['execution_finished'] is True
        assert recorder.executed == ['read_file', 'create_file', 'delete_file']

    def test_partial_decisions_ask_again_only_for_open_calls(self):
        memory, recorder, cfg, first = _start(CALLS3)
        pending = first['hitl_interrupts']

        second, _ = _resume(
            memory, recorder, cfg, CALLS3, _decide(pending, only={'create_file'}),
        )

        assert second['execution_finished'] is False
        assert recorder.executed == []
        again = second['hitl_interrupts']
        assert [p['tool_name'] for p in again] == ['delete_file', 'update_file']

        final, _ = _resume(memory, recorder, cfg, CALLS3, _decide(again))

        assert final['execution_finished'] is True
        assert recorder.executed == ['create_file', 'delete_file', 'update_file']

    def test_partial_resume_keeps_earlier_reject(self):
        memory, recorder, cfg, first = _start(CALLS2)
        pending = first['hitl_interrupts']

        second, _ = _resume(
            memory, recorder, cfg, CALLS2, _decide(pending, action='reject', only={'create_file'}),
        )
        assert [p['tool_name'] for p in second['hitl_interrupts']] == ['delete_file']

        final, _ = _resume(memory, recorder, cfg, CALLS2, _decide(second['hitl_interrupts']))

        assert final['execution_finished'] is True
        assert recorder.executed == ['delete_file']

    def test_partial_resume_hides_batch_state_from_ui_payload(self):
        memory, recorder, cfg, first = _start(CALLS3)

        second, _ = _resume(
            memory, recorder, cfg, CALLS3, _decide(first['hitl_interrupts'], only={'create_file'}),
        )

        hidden = {'_batch_decided', '_pending_messages'}
        assert second['execution_finished'] is False
        assert not hidden & set(second['hitl_interrupt'])
        for entry in second['hitl_interrupts']:
            assert not hidden & set(entry)

    def test_repeated_call_id_keeps_the_per_call_pause(self):
        calls = [('create_file', 'dup'), ('delete_file', 'dup')]
        _, recorder, _, first = _start(calls)

        assert first['execution_finished'] is False
        assert first['hitl_interrupt']['guardrail_type'] == 'sensitive_tool'
        assert len(first.get('hitl_interrupts') or []) <= 1
        assert recorder.executed == []

    @pytest.mark.parametrize('order', ['approve_first', 'reject_first'])
    def test_two_answers_for_one_call_block_it(self, order):
        memory, recorder, cfg, first = _start(CALLS2)
        pending = first['hitl_interrupts']
        approve = _decide(pending, only={'create_file'})
        reject = _decide(pending, action='reject', only={'create_file'})
        both = approve + reject if order == 'approve_first' else reject + approve

        final, _ = _resume(
            memory, recorder, cfg, CALLS2, both + _decide(pending, only={'delete_file'}),
        )

        assert final['execution_finished'] is True
        assert recorder.executed == ['delete_file']

    def test_single_sensitive_call_keeps_single_pause(self):
        calls = [('read_file', 'c0'), ('create_file', 'c1')]
        _, _, _, first = _start(calls)

        assert first['execution_finished'] is False
        assert first['hitl_interrupt']['guardrail_type'] == 'sensitive_tool'
        assert not first.get('hitl_interrupts') or len(first['hitl_interrupts']) == 1


class TestFlagOff:
    def test_flag_off_resume_pauses_for_second_call(self):
        configure_sensitive_tools({'art': ['create_file', 'delete_file']})
        memory = MemorySaver()
        recorder = _Recorder()
        cfg = {'configurable': {'thread_id': 'off'}}
        first = _runnable(memory, _BatchLLM(CALLS2), recorder, False).invoke(
            {'messages': [HumanMessage(content='go')]}, config=dict(cfg),
        )
        second = _runnable(memory, _BatchLLM(CALLS2), recorder, False).invoke(
            {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''}, config=dict(cfg),
        )

        assert first['hitl_interrupt']['tool_name'] == 'create_file'
        assert second['execution_finished'] is False
        assert second['hitl_interrupt']['tool_name'] == 'delete_file'
        assert recorder.executed == ['create_file']

    def test_flag_is_off_by_default(self):
        assert SensitiveToolGuardMiddleware().batch_approvals is False
        assert SensitiveToolGuardMiddleware(batch_approvals='yes').batch_approvals is False
        assert SensitiveToolGuardMiddleware(batch_approvals=True).batch_approvals is True


LATER2 = [('create_file', 'c3'), ('delete_file', 'c4')]


class TestNoStaleDecision:
    def test_approved_decision_does_not_answer_a_later_model_turn(self):
        memory, recorder, cfg, first = _start(CALLS2, later_calls=LATER2)
        assert recorder.executed == []

        second, _ = _resume(
            memory, recorder, cfg, CALLS2, _decide(first['hitl_interrupts']), later_calls=LATER2,
        )

        assert second['execution_finished'] is False
        assert [p['tool_call_id'] for p in second['hitl_interrupts']] == ['c3', 'c4']
        assert recorder.executed == ['create_file', 'delete_file']

        final, _ = _resume(
            memory, recorder, cfg, CALLS2, _decide(second['hitl_interrupts']), later_calls=LATER2,
        )

        assert final['execution_finished'] is True
        assert recorder.executed == ['create_file', 'delete_file', 'create_file', 'delete_file']

    def test_reused_call_ids_in_a_later_turn_run_nothing_again(self):
        memory, recorder, cfg, first = _start(CALLS2, later_calls=CALLS2)

        final, _ = _resume(
            memory, recorder, cfg, CALLS2, _decide(first['hitl_interrupts']), later_calls=CALLS2,
        )

        assert final['execution_finished'] is True
        assert not final.get('hitl_interrupts')
        assert recorder.executed == ['create_file', 'delete_file']

    def test_block_with_comment_keeps_note_in_blocked_result(self):
        memory, recorder, cfg, first = _start(CALLS2, later_calls=LATER2)
        pending = first['hitl_interrupts']
        decisions = (
            _decide(pending, action='block_with_comment', only={'create_file'}, value='use a copy')
            + _decide(pending, only={'delete_file'})
        )

        second, llm = _resume(memory, recorder, cfg, CALLS2, decisions, later_calls=LATER2)

        assert recorder.executed == ['delete_file']
        assert second['execution_finished'] is False
        assert [p['tool_call_id'] for p in second['hitl_interrupts']] == ['c3', 'c4']

        final, llm = _resume(
            memory, recorder, cfg, CALLS2, _decide(second['hitl_interrupts'], action='reject'),
            later_calls=LATER2,
        )

        assert final['execution_finished'] is True
        blocked = [m for m in llm.invocations[-1] if isinstance(m, ToolMessage)]
        assert blocked[0].tool_call_id == 'c1'
        assert SensitiveToolGuardMiddleware.BLOCKED_TOOL_RESULT_TYPE in str(blocked[0].content)
        assert 'use a copy' in str(blocked[0].content)
        assert 'use a copy' not in str(blocked[2].content)

    def test_guard_reads_decision_once(self):
        from elitea_sdk.runtime.tools.llm import _RESUMED_SENSITIVE_DECISION

        guard = SensitiveToolGuardMiddleware(batch_approvals=True)
        configure_sensitive_tools({'art': ['create_file']})
        tool = _make_tools(_Recorder())[0]
        context = guard.pending_review(tool, {'path': '/x'})
        token = _RESUMED_SENSITIVE_DECISION.set({'action': 'approve', 'value': ''})
        try:
            first = guard._review_sensitive_tool_call(context)
            assert _RESUMED_SENSITIVE_DECISION.get(None) is None
            assert first == {'action': 'approve', 'value': ''}
        finally:
            _RESUMED_SENSITIVE_DECISION.reset(token)


class TestSyncFallbackKeepsApproval:
    """A tool without an async path runs through the sync fallback of the agent loop.

    The approval given for the call covers the fallback: one review, no second pause.
    """

    @pytest.fixture
    def reviews(self, monkeypatch):
        seen: list[str] = []
        original = SensitiveToolGuardMiddleware._review_sensitive_tool_call

        def counted(self, ctx):
            seen.append(ctx['tool_name'])
            return original(self, ctx)

        monkeypatch.setattr(SensitiveToolGuardMiddleware, '_review_sensitive_tool_call', counted)
        return seen

    @pytest.mark.parametrize('batch', [False, True], ids=['single', 'batch'])
    def test_approved_call_is_reviewed_once_and_runs_once(self, batch, reviews):
        memory, recorder, cfg, first = _start(
            [('create_file', 'c1')], batch=batch, async_unsupported=True,
        )
        assert first['execution_finished'] is False
        reviews.clear()

        decisions = _decide(first['hitl_interrupts']) if batch else 'approve'
        final, _ = _resume(
            memory, recorder, cfg, [('create_file', 'c1')], decisions, batch=batch,
            async_unsupported=True,
        )

        assert final['execution_finished'] is True, final.get('hitl_interrupt')
        assert not final.get('hitl_interrupt')
        assert recorder.executed == ['create_file']
        assert reviews == ['create_file']

    @pytest.mark.parametrize('batch', [False, True], ids=['single', 'batch'])
    def test_second_distinct_call_still_pauses(self, batch, reviews):
        calls = [('create_file', 'c1')]
        later = [('delete_file', 'c2')]
        memory, recorder, cfg, first = _start(
            calls, batch=batch, later_calls=later, async_unsupported=True,
        )
        decisions = _decide(first['hitl_interrupts']) if batch else 'approve'

        second, _ = _resume(
            memory, recorder, cfg, calls, decisions, batch=batch, later_calls=later,
            async_unsupported=True,
        )

        assert second['execution_finished'] is False
        assert second['hitl_interrupt']['tool_name'] == 'delete_file'
        assert recorder.executed == ['create_file']

    @pytest.mark.parametrize('batch', [False, True], ids=['single', 'batch'])
    def test_rejected_call_never_runs(self, batch, reviews):
        memory, recorder, cfg, first = _start(
            [('create_file', 'c1')], batch=batch, async_unsupported=True,
        )
        decisions = _decide(first['hitl_interrupts'], action='reject') if batch else 'reject'

        final, _ = _resume(
            memory, recorder, cfg, [('create_file', 'c1')], decisions, batch=batch,
            async_unsupported=True,
        )

        assert final['execution_finished'] is True
        assert recorder.executed == []


class TestGuardReview:
    def test_pending_review_returns_context_for_sensitive_tool(self):
        configure_sensitive_tools({'art': ['create_file']})
        guard = SensitiveToolGuardMiddleware(batch_approvals=True)
        tools = {t.name: t for t in _make_tools(_Recorder())}

        sensitive = guard.pending_review(tools['create_file'], {'path': '/x'})
        plain = guard.pending_review(tools['read_file'], {'path': '/x'})

        assert sensitive is not None
        assert sensitive['tool_name'] == 'create_file'
        assert plain is None

    def test_pending_review_skips_auto_approved_tool(self):
        configure_sensitive_tools({'art': ['create_file']})
        guard = SensitiveToolGuardMiddleware(
            batch_approvals=True, auto_approve=[('art', 'create_file'), ('Artifact', 'create_file')],
        )
        tools = {t.name: t for t in _make_tools(_Recorder())}

        assert guard.pending_review(tools['create_file'], {'path': '/x'}) is None


class TestClientFlag:
    def test_inject_passes_flag_to_guard(self):
        configure_sensitive_tools({'art': ['create_file']})
        client = EliteAClient.__new__(EliteAClient)
        for flag in (True, False):
            middleware: list[Any] = []
            client._inject_sensitive_tool_guard(middleware, 'conv', False, flag)
            guards = [m for m in middleware if isinstance(m, SensitiveToolGuardMiddleware)]
            assert [g.batch_approvals for g in guards] == [flag]

    def test_inject_default_keeps_flag_off(self):
        configure_sensitive_tools({'art': ['create_file']})
        client = EliteAClient.__new__(EliteAClient)
        middleware: list[Any] = []
        client._inject_sensitive_tool_guard(middleware, 'conv')

        assert [m.batch_approvals for m in middleware] == [False]
