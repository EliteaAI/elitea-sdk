"""Escalation signals sent with the Auto resolve request (#6722). The SDK reports; it never selects."""
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from elitea_sdk.runtime.clients.routing import AutoChatModel, PIN
from elitea_sdk.runtime.clients import routing_escalation as esc

SENTINEL = 'ZXQ-SENTINEL-REQUEST-TEXT'
SCOPE = None


def run(messages, configurable_extra=None):
    requests = []
    owner = SimpleNamespace(base_url='https://unit.invalid', headers={}, project_id=7)

    def request(method, url, **kwargs):
        body = kwargs['json']
        requests.append(body)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: {
            'action': 'generate', 'config': {'model_name': 'fixture', 'max_tokens': 8000, 'context_window': 200000},
            'invocation_id': body['invocation_id'], 'scope_id': body['scope_id'],
            'pin': 'signed', 'state_token': 'state', 'expires_at': time.time() + 3600})
    owner._request = request
    owner.get_llm = Mock(return_value=Mock())
    auto = AutoChatModel(owner=owner, settings={'selection': {'mode': 'auto'}})
    config = {'configurable': {'thread_id': 't1', 'elitea_routing_run_id': 'run-2', **(configurable_extra or {})}}
    auto._materialize(messages, config)
    return requests[0]


def scope_id():
    from elitea_sdk.runtime.clients.routing import routing_scope_id
    return routing_scope_id({'configurable': {'thread_id': 't1'}})


def pinned(content, **metadata):
    return AIMessage(content=content, response_metadata={PIN: {'scope_id': scope_id(), 'invocation_id': 'run-1', 'pin': 'p'}, **metadata})


def history(answer, extra=()):
    return [HumanMessage(content='first question'), answer, *extra, HumanMessage(content=SENTINEL)]


def test_absent_fields_leave_request_keys_unchanged():
    body = run([HumanMessage(content='hello')])
    assert set(body) == {'selection', 'surface', 'messages', 'tools', 'generation_input_bytes', 'runtime_context',
                         'runtime_context_token', 'output_schema', 'invocation_id', 'scope_id', 'state_token',
                         'observation', 'prior_pin'}


@pytest.mark.parametrize('event', ['regenerate', 'upgrade_control'])
def test_escalation_event_forwarded_with_message_id(event):
    body = run([HumanMessage(content='hello')],
               {esc.ESCALATION_KEY: {'event': event, 'from_message_id': 'msg-42'}})
    assert body['escalation'] == {'event': event, 'from_message_id': 'msg-42'}
    assert 'last_response' not in body


def test_escalation_without_message_id_is_null():
    body = run([HumanMessage(content='hello')], {esc.ESCALATION_KEY: {'event': 'regenerate'}})
    assert body['escalation'] == {'event': 'regenerate', 'from_message_id': None}


@pytest.mark.parametrize('value', [None, 'regenerate', {}, {'event': 'bogus'}, {'event': None}, ['regenerate']])
def test_malformed_escalation_is_ignored(value):
    body = run([HumanMessage(content='hello')], {esc.ESCALATION_KEY: value})
    assert 'escalation' not in body


def test_last_response_finish_reason_from_previous_pinned_message():
    body = run(history(pinned('The answer is 4.', finish_reason='length')))
    assert body['last_response'] == {'finish_reason': 'length', 'refusal': False, 'tool_failure': False, 'inability': False}


def test_anthropic_stop_reason_is_normalised_and_refusal_detected():
    body = run(history(pinned('', stop_reason='refusal')))
    assert body['last_response']['finish_reason'] == 'refusal'
    assert body['last_response']['refusal'] is True
    assert run(history(pinned('ok', stop_reason='max_tokens')))['last_response']['finish_reason'] == 'length'


def test_provider_refusal_field_counts_as_refusal():
    answer = pinned('text')
    answer.additional_kwargs['refusal'] = 'policy'
    assert run(history(answer))['last_response']['refusal'] is True


@pytest.mark.parametrize('text', ["I cannot read that file.", "Sorry, I can't do that.", "I don’t have access to your repo."
                                  .replace('’', "'"), "I am unable to open the link.", "I CAN'T find it."])
def test_inability_rule_positive(text):
    assert run(history(pinned(text)))['last_response']['inability'] is True


def test_inability_rule_curly_apostrophe_and_blocks():
    assert run(history(pinned("I can’t verify this.")))['last_response']['inability'] is True
    blocks = [{'type': 'text', 'text': 'I cannot do it.'}]
    assert run(history(pinned(blocks)))['last_response']['inability'] is True


def test_inability_rule_negative_and_window():
    assert run(history(pinned('Here is the report you asked for.')))['last_response']['inability'] is False
    assert run(history(pinned('x' * esc.RULE_WINDOW + ' I cannot do it.')))['last_response']['inability'] is False
    assert run(history(pinned('Sure. Note that you cannot undo this.')))['last_response']['inability'] is False


def test_refusal_rule_from_content():
    assert run(history(pinned("I can't help with that request.")))['last_response']['refusal'] is True
    assert run(history(pinned('Done.')))['last_response']['refusal'] is False


def test_tool_failure_from_error_tool_message_after_answer():
    call = AIMessage(content='', tool_calls=[{'id': 'c1', 'name': 'tool', 'args': {}}],
                     response_metadata={PIN: {'scope_id': scope_id(), 'invocation_id': 'run-1', 'pin': 'p'}})
    messages = [HumanMessage(content='first'), call, ToolMessage(content='boom', tool_call_id='c1', status='error'),
                HumanMessage(content=SENTINEL)]
    assert run(messages)['last_response']['tool_failure'] is True


def test_tool_failure_false_when_cycle_ended_with_success_or_no_tools():
    call = AIMessage(content='', tool_calls=[{'id': 'c1', 'name': 'tool', 'args': {}}],
                     response_metadata={PIN: {'scope_id': scope_id(), 'invocation_id': 'run-1', 'pin': 'p'}})
    ok = [HumanMessage(content='first'), call, ToolMessage(content='fine', tool_call_id='c1'),
          pinned('Done.', finish_reason='stop'), HumanMessage(content=SENTINEL)]
    assert run(ok)['last_response']['tool_failure'] is False
    assert run(history(pinned('Done.')))['last_response']['tool_failure'] is False


def test_tool_failure_recovered_before_answer_is_not_a_failure():
    call = lambda i: AIMessage(content='', tool_calls=[{'id': i, 'name': 'tool', 'args': {}}],
                               response_metadata={PIN: {'scope_id': scope_id(), 'invocation_id': 'run-1', 'pin': 'p'}})
    messages = [HumanMessage(content='first'), call('a'), ToolMessage(content='boom', tool_call_id='a', status='error'),
                call('b'), ToolMessage(content='ok', tool_call_id='b'), pinned('Done.'), HumanMessage(content=SENTINEL)]
    assert run(messages)['last_response']['tool_failure'] is False


def test_finish_reason_falls_back_to_receipt_observation_when_history_has_no_metadata():
    assert esc.last_response_signal(history(AIMessage(content='plain')), 2, {'finish_reason': 'length'})['finish_reason'] == 'length'
    assert esc.last_response_signal(history(AIMessage(content='plain')), 2)['finish_reason'] is None


def test_no_assistant_message_means_no_last_response():
    assert esc.last_response_signal([HumanMessage(content='hello')], 0) is None


def test_no_request_or_answer_text_in_signals():
    answer = pinned(f"I cannot do {SENTINEL}", finish_reason='stop')
    body = run(history(answer), {esc.ESCALATION_KEY: {'event': 'regenerate', 'from_message_id': 'msg-1'}})
    signals = json.dumps({'escalation': body['escalation'], 'last_response': body['last_response']})
    assert SENTINEL not in signals
    assert set(body['last_response']) == {'finish_reason', 'refusal', 'tool_failure', 'inability'}
    assert set(body['escalation']) == {'event', 'from_message_id'}
