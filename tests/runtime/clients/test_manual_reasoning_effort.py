"""Manual reasoning effort must reach Chat Completions as a top-level reasoning_effort."""
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from elitea_sdk.runtime.clients.client import EliteAClient


def client():
    result = EliteAClient.__new__(EliteAClient)
    result.base_url = 'http://unit.invalid'
    result.allm_path = '/anthropic'
    result.llm_path = '/openai'
    result.auth_token = 'synthetic-token'
    result.project_id = '1'
    return result


def worker(monkeypatch, config):
    module = SimpleNamespace(descriptor=SimpleNamespace(config=config))
    monkeypatch.setitem(sys.modules, 'tools', SimpleNamespace(this=SimpleNamespace(for_module=lambda _: module)))


def request_body(name, settings):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={
            'id': 'fixture', 'object': 'chat.completion', 'created': 1, 'model': name,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'done'},
                         'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client, patch(
            'elitea_sdk.runtime.clients.client.ChatOpenAI',
            side_effect=lambda **kwargs: ChatOpenAI(http_client=http_client, **kwargs)):
        result = client().get_llm(name, settings).invoke([HumanMessage(content='synthetic task')])
    assert result.content == 'done'
    assert len(requests) == 1
    assert requests[0].url.path.endswith('/chat/completions')
    return json.loads(requests[0].content)


# Bedrock-hosted GPT models reject a nested `reasoning` object since LiteLLM 1.108
# (#7574, #7620, #7638). A stale worker opt-in must not bring it back.
@pytest.mark.parametrize('name', ['global.openai.gpt-6-luna', 'global.openai.gpt-6.1-sol',
                                  'global.openai.gpt-5.6-luna', 'gpt-5.4-mini'])
@pytest.mark.parametrize('config', [{}, {'reasoning_in_body_for': ['gpt-5.6', 'gpt-6']}])
@pytest.mark.parametrize('effort', ['low', 'Medium', 'high'])
def test_reasoning_effort_is_top_level(name, config, effort, monkeypatch):
    worker(monkeypatch, config)
    body = request_body(name, {'model_name': name, 'max_tokens': 1000, 'streaming': False,
                               'reasoning_effort': effort})
    assert body['reasoning_effort'] == effort.lower()
    assert 'reasoning' not in body
    assert 'thinking' not in body


@pytest.mark.parametrize('effort', [None, ''])
def test_no_reasoning_fields_without_effort(effort, monkeypatch):
    worker(monkeypatch, {'reasoning_in_body_for': ['gpt-6']})
    body = request_body('global.openai.gpt-6-luna', {
        'model_name': 'global.openai.gpt-6-luna', 'max_tokens': 1000, 'streaming': False,
        'reasoning_effort': effort})
    assert not {'reasoning', 'reasoning_effort', 'thinking'} & set(body)


def test_responses_api_keeps_reasoning_object(monkeypatch):
    worker(monkeypatch, {'use_responses_api_for': ['gpt-5.6'], 'reasoning_in_body_for': ['gpt-5.6']})
    with patch('elitea_sdk.runtime.clients.client.ChatOpenAI') as chat:
        client().get_llm('gpt-5.6-terra', {'model_name': 'gpt-5.6-terra', 'max_tokens': 1000,
                                           'streaming': False, 'reasoning_effort': 'high'})
    kwargs = chat.call_args.kwargs
    assert kwargs['reasoning'] == {'effort': 'high', 'summary': 'auto'}
    assert kwargs['use_responses_api'] is True
    assert 'reasoning_effort' not in kwargs
    assert 'reasoning' not in (kwargs.get('extra_body') or {})
