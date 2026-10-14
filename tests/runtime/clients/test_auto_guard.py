"""Guarded mode (#6722, PLAN 13.4): the SDK asks the Gateway to guard a final answer before delivery."""
import asyncio
import hashlib
import json
import time
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import requests as http
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from elitea_sdk.runtime.clients.routing import AutoChatModel, PIN, GUARD_META, GUARD_ANSWER_LIMIT

SENTINEL = 'ZXQ-SENTINEL-ANSWER-TEXT'
FIRST = 'Cheap model answer.'
SECOND = 'Escalated model answer.'
TARGET = {'config': {'model_name': 'escalated', 'max_tokens': 8000, 'context_window': 200000},
          'pin': 'pin-escalated', 'state_token': 'state-escalated'}
FAIL = {'verdict': 'fail', 'code': 'omitted_part', 'clause': 'asked for two items, gave one', 'escalate_to': TARGET}


class Harness:
    def __init__(self, guard=None, settings=None, answers=None):
        self.calls, self.generated = [], []
        self.guard = guard if guard is not None else {'verdict': 'pass'}
        self.answers = answers or {'cheap': AIMessage(content=FIRST, response_metadata={'finish_reason': 'stop'}),
                                   'escalated': AIMessage(content=SECOND, response_metadata={'finish_reason': 'stop'})}
        owner = SimpleNamespace(base_url='https://unit.invalid', headers={}, project_id=7)
        owner._request = self.request
        owner.get_llm = lambda name, config: ChatOpenAI(model=name, api_key='fixture', max_tokens=8000)
        self.auto = AutoChatModel(owner=owner, settings={'selection': {'mode': 'auto'}, **(settings or {})})
        self.sink = {}
        self.config = {'configurable': {'thread_id': 't1', 'elitea_routing_run_id': 'run-1', 'elitea_routing_sink': self.sink}}

    def request(self, method, url, **kwargs):
        body = kwargs['json']
        self.calls.append((url.rsplit('/', 1)[1], body))
        if url.endswith('/resolve'):
            payload = {'action': 'generate', 'config': {'model_name': 'cheap', 'max_tokens': 8000, 'context_window': 200000},
                       'invocation_id': body['invocation_id'], 'scope_id': body['scope_id'], 'pin': 'pin-cheap',
                       'state_token': 'state-cheap', 'expires_at': time.time() + 3600}
        elif isinstance(self.guard, Exception):
            raise self.guard
        else:
            payload = self.guard
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: payload)

    def guard_calls(self):
        return [body for name, body in self.calls if name == 'guard']

    def invoke(self, messages, run=None):
        def native(model, *args, **kwargs):
            self.generated.append(model.model_name)
            if isinstance(self.answers[model.model_name], Exception):
                raise self.answers[model.model_name]
            return self.answers[model.model_name]
        with patch.object(ChatOpenAI, 'invoke', autospec=True, side_effect=native):
            return (run or self.auto.invoke)(messages, self.config)


def ask(text='Name two colours.'):
    return [HumanMessage(content=text)]


def test_pass_path_is_unchanged():
    h = Harness()
    result = h.invoke(ask())
    assert result.content == FIRST and h.generated == ['cheap']
    assert GUARD_META not in result.response_metadata
    assert result.response_metadata[PIN]['pin'] == 'pin-cheap'
    assert len(h.guard_calls()) == 1


def test_guard_request_contract():
    h = Harness()
    h.invoke(ask())
    body = h.guard_calls()[0]
    assert set(body) == {'pin', 'scope_id', 'invocation_id', 'request_text_projection', 'answer_text', 'last_response'}
    assert body['pin'] == 'pin-cheap' and body['answer_text'] == FIRST
    assert body['request_text_projection'] == [{'role': 'user', 'content': 'Name two colours.'}]
    assert body['last_response'] == {'finish_reason': 'stop', 'refusal': False, 'tool_failure': False, 'inability': False}
    resolve = next(b for n, b in h.calls if n == 'resolve')
    assert (body['scope_id'], body['invocation_id']) == (resolve['scope_id'], resolve['invocation_id'])


def test_answer_text_is_capped():
    h = Harness(answers={'cheap': AIMessage(content='x' * (GUARD_ANSWER_LIMIT + 500))})
    h.invoke(ask())
    assert len(h.guard_calls()[0]['answer_text']) == GUARD_ANSWER_LIMIT


def test_fail_path_regenerates_once_on_returned_config_and_stamps():
    h = Harness(guard=FAIL)
    result = h.invoke(ask())
    assert result.content == SECOND and h.generated == ['cheap', 'escalated']
    binding = result.response_metadata[PIN]
    assert binding['pin'] == 'pin-escalated' and binding['state_token'] == 'state-escalated'
    assert binding['config']['model_name'] == 'escalated'
    assert result.response_metadata[GUARD_META] == {
        'verdict': 'fail', 'code': 'omitted_part', 'clause': 'asked for two items, gave one',
        'first_answer_sha256': hashlib.sha256(FIRST.encode()).hexdigest()}
    assert len(h.guard_calls()) == 1  # the escalated answer is not guarded again
    assert h.sink['binding']['pin'] == 'pin-escalated'


def test_fail_path_regenerates_from_the_same_messages():
    h = Harness(guard=FAIL)
    seen = []

    def native(model, messages, *args, **kwargs):
        seen.append((model.model_name, [m.content for m in messages]))
        return h.answers[model.model_name]
    with patch.object(ChatOpenAI, 'invoke', autospec=True, side_effect=native):
        h.auto.invoke(ask(), h.config)
    assert seen[0][1] == seen[1][1] == ['Name two colours.']


def test_guard_error_delivers_first_answer():
    for failure in (http.ConnectionError('down'), http.Timeout('slow'), RuntimeError('500')):
        h = Harness(guard=failure)
        result = h.invoke(ask())
        assert result.content == FIRST and h.generated == ['cheap']
        assert GUARD_META not in result.response_metadata
        assert result.response_metadata[PIN]['pin'] == 'pin-cheap'


@pytest.mark.parametrize('payload', [None, [], {'verdict': 'maybe'}, {'verdict': 'fail'},
                                     {'verdict': 'fail', 'escalate_to': {'pin': 'p'}},
                                     {'verdict': 'fail', 'escalate_to': {'config': {}, 'pin': 7}}])
def test_malformed_guard_response_delivers_first_answer(payload):
    h = Harness(guard=payload)
    assert h.invoke(ask()).content == FIRST and h.generated == ['cheap']


def test_failed_escalated_generation_delivers_first_answer():
    h = Harness(guard=FAIL, answers={'cheap': AIMessage(content=FIRST), 'escalated': RuntimeError('provider down')})
    result = h.invoke(ask())
    assert result.content == FIRST and result.response_metadata[PIN]['pin'] == 'pin-cheap'
    assert GUARD_META not in result.response_metadata


def test_unadmittable_escalated_config_delivers_first_answer():
    bad = {**FAIL, 'escalate_to': {**TARGET, 'config': {**TARGET['config'], 'context_window': 10}}}
    h = Harness(guard=bad)
    assert h.invoke(ask()).content == FIRST and h.generated == ['cheap']


def test_tool_cycle_turns_skip_the_guard():
    call = AIMessage(content='', tool_calls=[{'id': 'c1', 'name': 'search', 'args': {}}],
                     response_metadata={'finish_reason': 'tool_calls'})
    h = Harness(guard=FAIL, answers={'cheap': call})
    result = h.invoke(ask())
    assert result.tool_calls and h.guard_calls() == [] and GUARD_META not in result.response_metadata


def test_only_final_answer_of_a_tool_cycle_is_guarded_once():
    call = AIMessage(content='', tool_calls=[{'id': 'c1', 'name': 'search', 'args': {}}],
                     response_metadata={'finish_reason': 'tool_calls'})
    h = Harness()
    h.answers['cheap'] = call
    first = h.invoke(ask())
    assert h.guard_calls() == []
    h.answers['cheap'] = AIMessage(content=FIRST, response_metadata={'finish_reason': 'stop'})
    final = h.invoke([*ask(), first, ToolMessage(content='ok', tool_call_id='c1')])
    assert final.content == FIRST and len(h.guard_calls()) == 1
    assert h.guard_calls()[0]['last_response']['tool_failure'] is False
    # A further model call inside the same invocation is not guarded again.
    again = h.invoke([*ask(), first, ToolMessage(content='ok', tool_call_id='c1')])
    assert len(h.guard_calls()) == 1 and again.content == FIRST


def test_tool_failure_flag_reaches_the_guard():
    call = AIMessage(content='', tool_calls=[{'id': 'c1', 'name': 'search', 'args': {}}],
                     response_metadata={PIN: {}, 'finish_reason': 'tool_calls'})
    h = Harness()
    h.answers['cheap'] = call
    first = h.invoke(ask())
    h.answers['cheap'] = AIMessage(content="I can't reach the tool.", response_metadata={'finish_reason': 'stop'})
    h.invoke([*ask(), first, ToolMessage(content='boom', tool_call_id='c1', status='error')])
    flags = h.guard_calls()[0]['last_response']
    assert flags['tool_failure'] is True and flags['inability'] is True


def test_guard_can_be_switched_off():
    h = Harness(guard=FAIL, settings={'routing_guard': False})
    assert h.invoke(ask()).content == FIRST
    assert h.guard_calls() == []


def test_async_fail_path():
    h = Harness(guard=FAIL)

    async def native(model, *args, **kwargs):
        h.generated.append(model.model_name)
        return h.answers[model.model_name]
    with patch.object(ChatOpenAI, 'ainvoke', autospec=True, side_effect=native):
        result = asyncio.run(h.auto.ainvoke(ask(), h.config))
    assert result.content == SECOND and h.generated == ['cheap', 'escalated']
    assert result.response_metadata[GUARD_META]['verdict'] == 'fail'


def test_guard_metadata_carries_no_answer_text():
    h = Harness(guard=FAIL, answers={'cheap': AIMessage(content=f'{FIRST} {SENTINEL}', response_metadata={'finish_reason': 'stop'}),
                                     'escalated': AIMessage(content=SECOND)})
    result = h.invoke(ask())
    assert SENTINEL not in json.dumps(result.response_metadata[GUARD_META])
    assert set(result.response_metadata[GUARD_META]) == {'verdict', 'code', 'clause', 'first_answer_sha256'}
    assert SENTINEL not in json.dumps(result.response_metadata[PIN], default=str)


def test_streaming_path_is_not_guarded():
    h = Harness(guard=FAIL)
    from langchain_core.messages import AIMessageChunk
    with patch.object(ChatOpenAI, 'stream', autospec=True, return_value=iter([AIMessageChunk(content=FIRST)])):
        list(h.auto.stream(ask(), h.config))
    assert h.guard_calls() == []
