"""Preserve distinct sensitive calls with identical arguments across resume."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.base import message_to_dict
from langchain_core.messages.utils import messages_from_dict
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.langchain.langraph_agent import LangGraphAgentRunnable
from elitea_sdk.runtime.middleware.sensitive_tool_guard import SensitiveToolGuardMiddleware
from elitea_sdk.runtime.toolkits.security import configure_sensitive_tools, reset_sensitive_tools
from elitea_sdk.runtime.tools.llm import LLMNode


TOOL_NAME = 'fixture_status'


def _original(second_args=None, content=''):
    return AIMessage(
        content=content,
        tool_calls=[
            {'name': TOOL_NAME, 'args': {'value': 'same'}, 'id': 'call-first'},
            {
                'name': TOOL_NAME,
                'args': {'value': 'same'} if second_args is None else second_args,
                'id': 'call-second',
            },
        ],
    )


def _restored_history(original, results):
    pending = [message_to_dict(original), *[message_to_dict(result) for result in results]]
    trimmed = LangGraphAgentRunnable._trim_pending_messages(pending)
    return LLMNode._filter_orphaned_tool_calls(messages_from_dict(trimmed))


def _unfinished_ids(completion, history):
    return [
        call['id'] for call in completion.tool_calls
        if not LLMNode._tool_call_already_completed(call['id'], history)
    ]


@pytest.mark.parametrize('second_args', [{'value': 'same'}, {'value': 'different'}])
@pytest.mark.parametrize('content', ['', [{'type': 'thinking', 'thinking': 'fixture', 'signature': 'fixture'}]])
def test_resume_selects_unfinished_call_after_completed_sibling_filtering(second_args, content):
    original = _original(second_args, content)
    history = _restored_history(original, [ToolMessage(content='receipt-first', tool_call_id='call-first')])
    assert history[0].tool_calls[0]['id'] == 'call-first'
    assert len(history[0].tool_calls) == 1
    context = {
        'tool_name': TOOL_NAME,
        'tool_args': second_args,
        'tool_call_id': 'resume-placeholder',
        'original_ai_message': message_to_dict(original),
    }

    completion = LLMNode._build_resume_completion(context, history)

    assert completion is not None
    assert context['tool_call_id'] == 'call-second'
    assert completion.content == content
    assert [call['id'] for call in completion.tool_calls] == ['call-first', 'call-second']
    assert _unfinished_ids(completion, history) == ['call-second']


def test_first_resume_preserves_first_call_and_both_original_identities():
    original = _original()
    context = {
        'tool_name': TOOL_NAME,
        'tool_args': {'value': 'same'},
        'original_ai_message': message_to_dict(original),
    }

    completion = LLMNode._build_resume_completion(context, [])

    assert completion is not None
    assert context['tool_call_id'] == 'call-first'
    assert _unfinished_ids(completion, []) == ['call-first', 'call-second']


def test_completed_matches_reuse_history_without_creating_an_invocation():
    original = _original()
    history = _restored_history(original, [
        ToolMessage(content='receipt-first', tool_call_id='call-first'),
        ToolMessage(content='receipt-second', tool_call_id='call-second'),
    ])
    context = {
        'tool_name': TOOL_NAME,
        'tool_args': {'value': 'same'},
        'original_ai_message': message_to_dict(original),
    }

    completion = LLMNode._build_resume_completion(context, history)

    assert completion is history[0]
    assert context['tool_call_id'] == 'call-first'
    assert _unfinished_ids(completion, history) == []


@pytest.mark.parametrize('context', [
    {},
    {'original_ai_message': None},
    {'original_ai_message': message_to_dict(HumanMessage(content='fixture'))},
    {'tool_name': 'other_fixture_status', 'tool_args': {'value': 'same'},
     'original_ai_message': message_to_dict(_original())},
    {'tool_name': TOOL_NAME, 'tool_args': {'value': 'unmatched'},
     'original_ai_message': message_to_dict(_original())},
])
def test_missing_or_ambiguous_resume_identity_keeps_existing_fallback(context):
    assert LLMNode._build_resume_completion(context, []) is None


def test_completed_call_from_another_toolkit_does_not_match_by_operation_name():
    original = _original()
    original.tool_calls[0]['name'] = 'other_fixture_status'
    history = _restored_history(original, [ToolMessage(content='other-receipt', tool_call_id='call-first')])
    context = {
        'tool_name': TOOL_NAME,
        'tool_args': {'value': 'same'},
        'original_ai_message': message_to_dict(original),
    }

    completion = LLMNode._build_resume_completion(context, history)

    assert completion is not None
    assert context['tool_call_id'] == 'call-second'
    assert _unfinished_ids(completion, history) == ['call-second']


class _RepeatedCallsLLM:
    temperature = 0
    max_tokens = 1000

    def __init__(self, observed):
        self.observed = observed

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, _tools, **_kwargs):
        return self

    def invoke(self, messages, config=None):
        results = [(message.tool_call_id, message.content) for message in messages if isinstance(message, ToolMessage)]
        if results:
            self.observed.append(results)
            # Like the provider fixture, return the results actually supplied.
            # A missing sibling must fail the final receipt assertion.
            return AIMessage(content=';'.join(f'{call_id}={content}' for call_id, content in results))
        return _original()


def _runnable(memory, executed, observed):
    def status(value: str):
        executed.append(value)
        return 'receipt'

    return Assistant(
        elitea=SimpleNamespace(get_mcp_toolkits=lambda: []),
        data={'instructions': 'Use tools', 'tools': [], 'meta': {}},
        client=_RepeatedCallsLLM(observed),
        tools=[StructuredTool.from_function(
            func=status,
            name=TOOL_NAME,
            description='Read a synthetic status.',
            metadata={'toolkit_type': 'fixture', 'toolkit_name': 'fixture', 'tool_name': TOOL_NAME},
        )],
        memory=memory,
        app_type='predict',
        middleware=[SensitiveToolGuardMiddleware()],
    ).runnable()


def test_identical_calls_resume_separately_without_replaying_first_call():
    configure_sensitive_tools({'fixture': [TOOL_NAME]})
    try:
        memory = MemorySaver()
        executed = []
        observed = []
        config = {'configurable': {'thread_id': 'identical-sensitive-call-fixture'}}
        first = _runnable(memory, executed, observed).invoke(
            {'messages': [HumanMessage(content='Read the same status twice.')]}, config=config,
        )
        assert first['execution_finished'] is False
        first_interrupt = first['hitl_interrupt']['interrupt_id']
        assert executed == []

        second = _runnable(memory, executed, observed).invoke(
            {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''}, config=config,
        )
        assert second['execution_finished'] is False
        assert second['hitl_interrupt']['interrupt_id'] != first_interrupt
        assert executed == ['same']

        final = _runnable(memory, executed, observed).invoke(
            {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''}, config=config,
        )
        assert final['execution_finished'] is True
        assert final['output'] == 'call-first=receipt;call-second=receipt'
        assert executed == ['same', 'same']
        assert observed == [[('call-first', 'receipt'), ('call-second', 'receipt')]]
    finally:
        reset_sensitive_tools()
