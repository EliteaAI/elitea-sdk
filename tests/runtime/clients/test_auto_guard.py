"""Record-only guard (#6722, PLAN 13.5): the delivered final answer is judged off the request path; the turn never changes."""
import asyncio
import hashlib
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import requests as http
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from elitea_sdk.runtime.clients.routing import AutoChatModel, PIN, GUARD_ANSWER_LIMIT

GUARD_META = 'elitea_routing_guard'

SENTINEL = 'ZXQ-SENTINEL-ANSWER-TEXT'
FIRST = 'Cheap model answer.'
SECOND = 'Escalated model answer.'
TARGET = {'config': {'model_name': 'escalated', 'max_tokens': 8000, 'context_window': 200000},
          'pin': 'pin-escalated', 'state_token': 'state-escalated'}
FAIL = {'verdict': 'fail', 'code': 'omitted_part', 'clause': 'asked for two items, gave one', 'escalate_to': TARGET}  # escalate_to must be ignored


class Harness:
    def __init__(self, guard=None, settings=None, answers=None):
        self.calls, self.generated = [], []
        self.release, self.guard_threads = threading.Event(), []
        self.release.set()
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
        else:
            self.guard_threads.append(threading.current_thread())
            assert self.release.wait(5)
            if isinstance(self.guard, Exception):
                raise self.guard
            payload = self.guard
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: payload)

    @staticmethod
    def settle():
        for thread in [t for t in threading.enumerate() if t.name == 'auto-routing-guard']:
            thread.join(5)

    def guard_calls(self):
        self.settle()
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


def test_fail_verdict_is_recorded_and_never_changes_the_turn():
    h = Harness(guard=FAIL)
    result = h.invoke(ask())
    h.settle()
    assert result.content == FIRST and h.generated == ['cheap']  # no second generation
    assert GUARD_META not in result.response_metadata
    assert result.response_metadata[PIN]['pin'] == 'pin-cheap'
    assert result.response_metadata[PIN]['config']['model_name'] == 'cheap'
    assert h.sink['binding']['pin'] == 'pin-cheap'
    assert len(h.guard_calls()) == 1


def test_invoke_does_not_wait_for_the_guard():
    h = Harness()
    h.release.clear()  # the Gateway guard call hangs until released
    started = time.monotonic()
    result = h.invoke(ask())
    assert time.monotonic() - started < 3 and result.content == FIRST
    assert [t for t in threading.enumerate() if t.name == 'auto-routing-guard']  # still in flight
    h.release.set()
    assert len(h.guard_calls()) == 1


def test_guard_thread_is_a_daemon_with_the_guard_timeout():
    h = Harness()
    seen = {}
    original = h.request

    def spy(method, url, **kwargs):
        seen.update(timeout=kwargs.get('timeout'), daemon=threading.current_thread().daemon)
        return original(method, url, **kwargs)
    h.auto.owner._request = spy
    h.invoke(ask())
    h.settle()
    assert seen['timeout'] == (3, 30) and seen['daemon'] is True


@pytest.mark.parametrize('failure', [http.ConnectionError('down'), http.Timeout('slow'), RuntimeError('500')])
def test_guard_errors_are_swallowed(failure):
    h = Harness(guard=failure)
    result = h.invoke(ask())
    h.settle()
    assert result.content == FIRST and h.generated == ['cheap']
    assert GUARD_META not in result.response_metadata


def test_guard_failing_status_is_swallowed():
    h = Harness()
    original = h.request

    def bad_status(method, url, **kwargs):
        response = original(method, url, **kwargs)
        if url.endswith('/guard'):
            response.raise_for_status = lambda: (_ for _ in ()).throw(http.HTTPError('503'))
        return response
    h.auto.owner._request = bad_status
    assert h.invoke(ask()).content == FIRST
    h.settle()


def test_thread_start_failure_never_breaks_the_turn():
    h = Harness()
    with patch('elitea_sdk.runtime.clients.routing.threading.Thread', side_effect=RuntimeError('cannot start')):
        assert h.invoke(ask()).content == FIRST
    assert h.guard_calls() == []


def test_tool_cycle_turns_skip_the_guard():
    call = AIMessage(content='', tool_calls=[{'id': 'c1', 'name': 'search', 'args': {}}],
                     response_metadata={'finish_reason': 'tool_calls'})
    h = Harness(guard=FAIL, answers={'cheap': call})
    result = h.invoke(ask())
    assert result.tool_calls and h.guard_calls() == [] and GUARD_META not in result.response_metadata


def test_one_guard_per_invocation():
    call = AIMessage(content='', tool_calls=[{'id': 'c1', 'name': 'search', 'args': {}}],
                     response_metadata={'finish_reason': 'tool_calls'})
    h = Harness()
    h.answers['cheap'] = call
    first = h.invoke(ask())
    assert h.guard_calls() == []
    h.answers['cheap'] = AIMessage(content=FIRST, response_metadata={'finish_reason': 'stop'})
    history = [*ask(), first, ToolMessage(content='ok', tool_call_id='c1')]
    final = h.invoke(history)
    assert final.content == FIRST and len(h.guard_calls()) == 1
    assert final.response_metadata[PIN]['guarded'] is True
    again = h.invoke(history)  # a further model call inside the same invocation
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


def test_async_path_records_without_changing_the_turn():
    h = Harness(guard=FAIL)

    async def native(model, *args, **kwargs):
        h.generated.append(model.model_name)
        return h.answers[model.model_name]
    with patch.object(ChatOpenAI, 'ainvoke', autospec=True, side_effect=native):
        result = asyncio.run(h.auto.ainvoke(ask(), h.config))
    h.settle()
    assert result.content == FIRST and h.generated == ['cheap']
    assert GUARD_META not in result.response_metadata and len(h.guard_calls()) == 1


def test_delivered_message_carries_no_answer_text_beyond_the_answer():
    h = Harness(guard=FAIL, answers={'cheap': AIMessage(content=f'{FIRST} {SENTINEL}', response_metadata={'finish_reason': 'stop'})})
    result = h.invoke(ask())
    assert SENTINEL in result.content
    assert SENTINEL not in json.dumps(result.response_metadata, default=str)
    assert h.guard_calls()[0]['answer_text'].endswith(SENTINEL)  # the Gateway receives it, the metadata does not


def test_streaming_path_is_unaffected():
    h = Harness(guard=FAIL)
    from langchain_core.messages import AIMessageChunk
    with patch.object(ChatOpenAI, 'stream', autospec=True, return_value=iter([AIMessageChunk(content=FIRST)])):
        chunks = list(h.auto.stream(ask(), h.config))
    assert h.guard_calls() == [] and chunks[0].content == FIRST
