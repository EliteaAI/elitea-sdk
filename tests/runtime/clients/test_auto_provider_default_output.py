"""Auto pins that carry no output cap (provider default), EliteaAI/elitea_issues#6722 and #6826."""
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage

from elitea_sdk.runtime.clients.client import EliteAClient
from elitea_sdk.runtime.clients.routing import AutoChatModel, effective_output_limit
from elitea_sdk.runtime.tools.llm import LLMNode

CONTEXT = 200_000
MAX_OUT = 64_000


class Native:
    def __init__(self):
        self.bound = None

    def _get_request_payload(self, messages, **kwargs):
        return {'messages': [m.model_dump(mode='json') for m in messages], **kwargs}

    def invoke(self, messages, config=None, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content='ok')


def make_auto(max_tokens, settings_cap=None, payload_bytes=0):
    native = Native()
    if payload_bytes:
        native._get_request_payload = lambda *a, **k: {'system': 'x' * payload_bytes}
    requests = []
    owner = SimpleNamespace(base_url='https://unit.invalid', headers={}, project_id=7)

    def request(method, url, **kwargs):
        body = kwargs['json']
        requests.append(body)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'action': 'generate',
            'config': {'model_name': 'native', 'max_tokens': max_tokens, 'max_output_tokens': MAX_OUT,
                       'context_window': CONTEXT, 'routing_total_output_cap': max_tokens is not None},
            'pin': 'signed-fixture', 'scope_id': body['scope_id'], 'state_token': 'state-fixture',
            'invocation_id': body['invocation_id'], 'expires_at': time.time() + 3600})

    owner._request = request
    owner.get_llm = Mock(return_value=native)
    settings = {'selection': {'mode': 'auto'}, 'routing_surface': 'agent'}
    if settings_cap is not None:
        settings['max_tokens'] = settings_cap
    return AutoChatModel(owner=owner, settings=settings), owner, requests


def cfg():
    return {'configurable': {'thread_id': 't', 'checkpoint_ns': 'agent', 'elitea_routing_run_id': 'r'}}


def client():
    c = EliteAClient.__new__(EliteAClient)
    c.base_url, c.allm_path, c.llm_path = 'http://proxy', '/anthropic', '/openai'
    c.auth_token, c.project_id, c.api_extra_headers = 'tok', 1, {}
    return c


def pinned_config(max_tokens, **extra):
    return {'max_tokens': max_tokens, 'max_output_tokens': MAX_OUT, 'context_window': CONTEXT,
            'routing_total_output_cap': max_tokens is not None, **extra}


# (1) None pin -> no max-token field reaches the provider
def test_none_pin_reaches_openai_client_without_a_token_field():
    with patch('elitea_sdk.runtime.clients.client.ChatOpenAI') as openai:
        client().get_llm('gpt-5.4', pinned_config(None))
    kwargs = openai.call_args.kwargs
    assert not {'max_tokens', 'max_completion_tokens', 'max_output_tokens'} & set(kwargs)


def test_none_pin_through_auto_hands_none_to_get_llm_and_sends_no_output_cap():
    auto, owner, requests = make_auto(None)
    auto.invoke([HumanMessage(content='task')], cfg())
    assert 'output_cap' not in requests[0]
    assert owner.get_llm.call_args.args[1]['max_tokens'] is None


def test_none_pin_anthropic_uses_required_model_maximum_not_a_synthetic_cap():
    with patch('elitea_sdk.runtime.clients.client.ChatAnthropic') as anthropic:
        client().get_llm('claude-sonnet-5-5', pinned_config(None, reasoning_effort='high'))
    # Anthropic requires the field; the provider default is the model maximum and no thinking padding is added.
    assert anthropic.call_args.kwargs['max_tokens'] == MAX_OUT


# (2) int pin unchanged
def test_int_pin_is_still_sent_to_the_provider():
    with patch('elitea_sdk.runtime.clients.client.ChatOpenAI') as openai:
        client().get_llm('gpt-5.4', pinned_config(8000))
    assert openai.call_args.kwargs['max_tokens'] == 8000


def test_int_pin_through_auto_is_not_replaced_by_chat_setting():
    auto, owner, _ = make_auto(8000, settings_cap=2048)
    auto.invoke([HumanMessage(content='task')], cfg())
    assert owner.get_llm.call_args.args[1]['max_tokens'] == 8000


# (3) context admission
def test_effective_output_limit_prefers_int_pin_then_model_maximum():
    assert effective_output_limit({'max_tokens': 8000, 'max_output_tokens': MAX_OUT}) == 8000
    assert effective_output_limit({'max_tokens': None, 'max_output_tokens': MAX_OUT}) == MAX_OUT
    with pytest.raises(ValueError):
        effective_output_limit({'max_tokens': None})


def test_admission_uses_model_maximum_when_pin_has_no_cap():
    # payload ~ CONTEXT - MAX_OUT - slack passes; one that only exceeds with MAX_OUT must refuse.
    fits = CONTEXT - MAX_OUT - 5_000
    auto, _, _ = make_auto(None, payload_bytes=fits)
    auto.invoke([HumanMessage(content='Hi')], cfg())
    over = CONTEXT - MAX_OUT + 1_000
    auto, _, _ = make_auto(None, payload_bytes=over)
    with pytest.raises(ValueError, match='conservative context allowance'):
        auto.invoke([HumanMessage(content='Hi')], cfg())


def test_same_payload_passes_with_a_small_int_pin():
    over = CONTEXT - MAX_OUT + 1_000
    auto, _, _ = make_auto(8000, payload_bytes=over)
    auto.invoke([HumanMessage(content='Hi')], cfg())


# (4) explicit chat cap
def test_explicit_chat_cap_is_sent_and_reaches_provider_when_pin_has_none():
    auto, owner, requests = make_auto(None, settings_cap=2048)
    auto.invoke([HumanMessage(content='task')], cfg())
    assert requests[0]['output_cap'] == 2048
    assert owner.get_llm.call_args.args[1]['max_tokens'] == 2048


def test_explicit_chat_cap_never_exceeds_model_maximum_in_provider_call():
    auto, owner, _ = make_auto(None, settings_cap=MAX_OUT * 2)
    auto.invoke([HumanMessage(content='task')], cfg())
    assert owner.get_llm.call_args.args[1]['max_tokens'] == MAX_OUT


@pytest.mark.parametrize('cap', [None, -1])
def test_unset_chat_cap_leaves_provider_default(cap):
    auto, owner, requests = make_auto(None, settings_cap=cap)
    auto.invoke([HumanMessage(content='task')], cfg())
    assert 'output_cap' not in requests[0]
    assert owner.get_llm.call_args.args[1]['max_tokens'] is None


# (5) continuation budget
def continuation_node(llm):
    return LLMNode(client=llm, name='agent', available_tools=[])


def test_continuation_budget_falls_back_to_model_maximum_for_none_pinned_anthropic():
    with patch('elitea_sdk.runtime.clients.client.ChatAnthropic', SimpleNamespace):
        llm = client().get_llm('claude-sonnet-5-5', pinned_config(None))
    node = continuation_node(llm)
    assert node._continuation_max_tokens(remaining_words=100, anchor='a', closure_only=False) == 158
    huge = node._continuation_max_tokens(remaining_words=10_000_000, anchor='a', closure_only=False)
    assert huge == MAX_OUT


def test_continuation_budget_for_none_pinned_openai_sends_no_cap():
    with patch('elitea_sdk.runtime.clients.client.ChatOpenAI', SimpleNamespace):
        llm = client().get_llm('gpt-5.4', pinned_config(None))
    node = continuation_node(llm)
    assert node._continuation_max_tokens(remaining_words=100, anchor='a', closure_only=False) is None


def test_continuation_budget_for_auto_wrapper_with_none_pin_sends_no_cap():
    auto, _, _ = make_auto(None)
    node = continuation_node(auto)
    assert node._continuation_max_tokens(remaining_words=100, anchor='a', closure_only=False) is None


# Input-size proxy must reflect what the provider receives, not routing metadata
def _history(metadata_bytes):
    from langchain_core.messages import AIMessage, ToolMessage
    from elitea_sdk.runtime.clients.routing import PIN, routing_scope_id
    scope = routing_scope_id(cfg())
    messages = [HumanMessage(content='first question ' * 40)]
    for i in range(6):
        meta = {PIN: {'pin': 'p' * 50, 'scope_id': scope, 'invocation_id': f'old{i}', 'trace': 'x' * metadata_bytes}} if metadata_bytes else {}
        messages.append(AIMessage(content=f'answer {i} ' * 60, response_metadata=meta,
                                  usage_metadata={'input_tokens': 5, 'output_tokens': 5, 'total_tokens': 10},
                                  tool_calls=[{'id': f'c{i}', 'name': 'fetch', 'args': {'q': 'v'}}]))
        messages.append(ToolMessage(content='result ' * 30, tool_call_id=f'c{i}'))
    messages.append(HumanMessage(content='follow up'))
    return messages


def _input_bytes(messages):
    auto, _, requests = make_auto(None)
    auto.invoke(messages, cfg())
    return requests[0]['generation_input_bytes']


def test_generation_input_bytes_excludes_routing_metadata():
    plain = _input_bytes(_history(0))
    heavy = _input_bytes(_history(10_000))
    assert heavy == plain  # 6 assistant turns x 10 KB of metadata contribute nothing
    assert abs(heavy - plain) <= plain * 0.10


def test_generation_input_bytes_plain_conversation_covers_content_and_framing():
    messages = _history(0)
    size = _input_bytes(messages)
    content = sum(len(str(m.content).encode()) for m in messages)
    assert size >= content + 64 * len(messages)
    assert size < content * 1.5 + 64 * len(messages) + 2_000
