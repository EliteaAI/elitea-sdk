"""Verify signed Auto reasoning contracts at the actual HTTP serialization boundary."""
import json
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from elitea_sdk.runtime.clients.client import EliteAClient


OPENAI_MODELS = ['global.openai.gpt-5.6-terra', 'global.openai.gpt-5.6-sol']
ANTHROPIC_MODELS = ['eu.anthropic.claude-opus-5', 'eu.anthropic.claude-opus-4-8',
                    'eu.anthropic.claude-opus-4-7']


def client():
    result = EliteAClient.__new__(EliteAClient)
    result.base_url = 'http://unit.invalid'
    result.allm_path = '/anthropic'
    result.llm_path = '/openai'
    result.auth_token = 'synthetic-token'
    result.project_id = '1'
    return result


def contract(name, effort):
    native = name in ANTHROPIC_MODELS
    fields = {} if effort is None else (
        {'thinking': {'type': 'adaptive', 'display': 'summarized'},
         'output_config': {'effort': effort}} if native else {'reasoning': {'effort': effort}})
    return {'model_name': name, 'max_tokens': 32000, 'streaming': False,
            'reasoning_effort': effort, 'routing_reasoning_fields': fields,
            'routing_transport': 'anthropic_messages' if native else 'chat_completions',
            'routing_total_output_cap': True, 'routing_pin': 'signed-fixture',
            'routing_invocation_id': 'a'*64}


@pytest.mark.parametrize('name', OPENAI_MODELS + ANTHROPIC_MODELS)
@pytest.mark.parametrize('effort', [None, 'low', 'medium', 'high'])
def test_exact_measured_reasoning_reaches_http_body(name, effort, monkeypatch):
    # Worker preferences deliberately conflict with the measured Chat Completions path.
    worker = SimpleNamespace(descriptor=SimpleNamespace(config={
        'use_responses_api_for': ['gpt-5.6'], 'reasoning_in_body_for': []}))
    monkeypatch.setitem(sys.modules, 'tools', SimpleNamespace(this=SimpleNamespace(for_module=lambda _: worker)))
    requests = []
    is_native = name in ANTHROPIC_MODELS

    def handle(request):
        requests.append(request)
        result = ({'id': 'fixture', 'type': 'message', 'role': 'assistant', 'model': name,
                   'content': [{'type': 'text', 'text': 'done'}], 'stop_reason': 'end_turn',
                   'usage': {'input_tokens': 1, 'output_tokens': 1}} if is_native else {
                       'id': 'fixture', 'object': 'chat.completion', 'created': 1, 'model': name,
                       'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'done'},
                                    'finish_reason': 'stop'}],
                       'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})
        return httpx.Response(200, json=result)

    settings = contract(name, effort)
    original = deepcopy(settings)
    cls = ChatAnthropic if is_native else ChatOpenAI
    symbol = 'ChatAnthropic' if is_native else 'ChatOpenAI'
    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client, patch(
            'langchain_anthropic.chat_models._get_default_httpx_client', return_value=http_client), patch(
            'elitea_sdk.runtime.clients.client.' + symbol,
            side_effect=lambda **kwargs: cls(**kwargs) if is_native else cls(http_client=http_client, **kwargs)):
        model = client().get_llm(name, settings)
        result = model.invoke([HumanMessage(content='synthetic task')])
    assert result.content == 'done'
    assert len(requests) == 1
    body = json.loads(requests[0].content)
    actual = {key: body[key] for key in ('thinking', 'output_config', 'reasoning', 'reasoning_effort') if key in body}
    assert actual == settings['routing_reasoning_fields']
    assert body.get('max_tokens', body.get('max_completion_tokens')) == 32000
    assert requests[0].url.path.endswith('/messages' if is_native else '/chat/completions')
    assert requests[0].headers['X-Elitea-Routing-Pin'] == 'signed-fixture'
    assert requests[0].headers['X-Elitea-Routing-Invocation'] == 'a'*64
    assert settings == original


@pytest.mark.parametrize('change', [
    {'routing_reasoning_fields': None},
    {'routing_reasoning_fields': []},
    {'routing_reasoning_fields': {}},
    {'routing_reasoning_fields': {'reasoning_effort': 'high'}},
    {'routing_reasoning_fields': {'reasoning': {'effort': 'low'}}},
    {'routing_reasoning_fields': {'reasoning': {'effort': 'high', 'summary': 'auto'}}},
    {'reasoning_effort': 'max'},
    {'reasoning_effort': None},
    {'routing_transport': None},
])
def test_malformed_or_contradictory_signed_contract_rejected_before_construction(change):
    settings = {**contract(OPENAI_MODELS[0], 'high'), **change}
    with patch('elitea_sdk.runtime.clients.client.ChatOpenAI') as constructor:
        with pytest.raises(ValueError, match='measured contract|measured reasoning contract'):
            client().get_llm(OPENAI_MODELS[0], settings)
    constructor.assert_not_called()


@pytest.mark.parametrize('thinking', [
    {'type': 'disabled'}, {'type': 'adaptive'},
    {'type': 'adaptive', 'display': 'summarized', 'budget_tokens': 1024},
])
def test_native_contract_rejects_unmeasured_thinking_shape(thinking):
    settings = contract(ANTHROPIC_MODELS[0], 'high')
    settings['routing_reasoning_fields']['thinking'] = thinking
    with patch('elitea_sdk.runtime.clients.client.ChatAnthropic') as constructor:
        with pytest.raises(ValueError, match='measured contract'):
            client().get_llm(ANTHROPIC_MODELS[0], settings)
    constructor.assert_not_called()


@pytest.mark.parametrize('name,compatible', [('gpt-5.4', False), ('gpt-5-mini', False),
                                         (ANTHROPIC_MODELS[0], True)])
def test_manual_reasoning_ignores_unsigned_routing_fields(name, compatible):
    with patch('elitea_sdk.runtime.clients.client.ChatOpenAI') as constructor:
        client().get_llm(name, {'max_tokens': 8000, 'reasoning_effort': 'medium',
                              'openai_compatible': compatible,
                              'routing_reasoning_fields': {'arbitrary': 'ignored'}})
    kwargs = constructor.call_args.kwargs
    assert kwargs['reasoning_effort'] == 'medium'
    assert 'extra_body' not in kwargs
    assert 'thinking' not in kwargs


@pytest.mark.parametrize('name', ['gpt-5.4', 'gpt-5-mini'])
def test_legacy_auto_effort_still_uses_existing_reasoning_parameter(name):
    with patch('elitea_sdk.runtime.clients.client.ChatOpenAI') as constructor:
        client().get_llm(name, {'max_tokens': 8000, 'reasoning_effort': 'high',
                              'routing_pin': 'old-signed-fixture', 'routing_invocation_id': 'a'*64})
    assert constructor.call_args.kwargs['reasoning_effort'] == 'high'
    assert 'extra_body' not in constructor.call_args.kwargs
