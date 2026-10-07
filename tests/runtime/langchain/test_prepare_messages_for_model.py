"""Provider-bound ToolMessage content contract (#6353)."""

import copy
import json

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from elitea_sdk.runtime.langchain.utils import (
    EMPTY_ERROR_TOOL_RESULT_CONTENT,
    EMPTY_SUCCESSFUL_TOOL_RESULT_CONTENT,
    prepare_messages_for_model,
)


def test_empty_tool_content_is_replaced_without_mutating_history():
    empty_success = ToolMessage(
        content='',
        tool_call_id='call-success',
        name='empty_tool',
        artifact={'raw': 'artifact'},
        additional_kwargs={'source': 'test'},
        response_metadata={'provider': 'strict'},
        id='message-1',
    )
    empty_error = ToolMessage(
        content='   ',
        tool_call_id='call-error',
        status='error',
    )
    messages = [HumanMessage(content='go'), empty_success, empty_error]

    prepared = prepare_messages_for_model(messages)

    assert empty_success.content == ''
    assert empty_error.content == '   '
    assert prepared[1].content == EMPTY_SUCCESSFUL_TOOL_RESULT_CONTENT
    assert prepared[2].content == EMPTY_ERROR_TOOL_RESULT_CONTENT
    assert prepared[1].tool_call_id == 'call-success'
    assert prepared[1].name == 'empty_tool'
    assert prepared[1].artifact == {'raw': 'artifact'}
    assert prepared[1].additional_kwargs == {'source': 'test'}
    assert prepared[1].response_metadata == {'provider': 'strict'}
    assert prepared[1].id == 'message-1'
    assert prepared[2].status == 'error'


def test_empty_content_block_list_is_replaced():
    empty_list = ToolMessage(content=[], tool_call_id='call-list')
    legacy_none = ToolMessage.model_construct(
        content=None,
        tool_call_id='call-none',
    )

    prepared = prepare_messages_for_model([empty_list, legacy_none])

    assert prepared[0].content == EMPTY_SUCCESSFUL_TOOL_RESULT_CONTENT
    assert prepared[1].content == EMPTY_SUCCESSFUL_TOOL_RESULT_CONTENT
    assert empty_list.content == []
    assert legacy_none.content is None


def test_non_empty_messages_are_preserved_exactly():
    assistant_tool_call = AIMessage(
        content='',
        tool_calls=[{'name': 'lookup', 'args': {}, 'id': 'call-1'}],
    )
    text_result = ToolMessage(content='result', tool_call_id='call-1')
    structured_result = ToolMessage(
        content=[{'type': 'text', 'text': 'result'}],
        tool_call_id='call-2',
    )
    messages = [assistant_tool_call, text_result, structured_result]

    prepared = prepare_messages_for_model(messages)

    assert prepared == messages
    assert all(actual is original for actual, original in zip(prepared, messages))


def test_preparation_is_idempotent():
    prepared = prepare_messages_for_model([
        ToolMessage(content='', tool_call_id='call-1'),
    ])

    prepared_again = prepare_messages_for_model(prepared)

    assert prepared_again == prepared
    assert prepared_again[0] is prepared[0]


@pytest.fixture(params=[
    ('anthropic', 'claude-sonnet-4-5'),
    ('anthropic', 'global.bedrock.claude-haiku-4-5'),
    ('openai', 'gpt-4.1'),
    ('openai', 'gpt-5.4'),
    ('responses', 'gpt-5.4'),
    ('openai', 'global.google.gemini-2.5-pro'),
    ('openai', 'global.meta.llama-3.3'),
    ('openai', 'claude-sonnet-openai-compatible'),
], ids=['anthropic', 'bedrock-messages', 'openai', 'openai-reasoning',
        'openai-responses', 'gemini-chat', 'llama-chat', 'claude-chat'])
def provider_client(request):
    transport, model_name = request.param
    if transport == 'anthropic':
        return ChatAnthropic(model=model_name, api_key='offline', max_tokens=1024)
    return ChatOpenAI(model=model_name, api_key='offline', use_responses_api=transport == 'responses')


@pytest.mark.parametrize('content', [
    'useful result', '[]', '{}', 'null',
    [{'type': 'text', 'text': 'useful result'}],
    [{'type': 'text', 'text': 'useful result'},
     {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,aW1hZ2U='}}],
], ids=['text', 'json-array', 'json-object', 'json-null', 'text-block', 'text-and-image'])
@pytest.mark.parametrize('status', ['success', 'error'])
def test_other_provider_wire_payload_is_unchanged_for_nonempty_results(provider_client, content, status):
    """Compare real serializers, including image data and Responses tool output."""
    history = [HumanMessage(content='go'), AIMessage(content='', tool_calls=[{
        'id': 'call-1', 'name': 'lookup', 'args': {},
    }]), ToolMessage(content=copy.deepcopy(content), tool_call_id='call-1', status=status,
                    name='lookup', id='result-1', artifact={'raw': 'audit'},
                    additional_kwargs={'source': 'test'}, response_metadata={'trace': 'test'})]
    snapshot = copy.deepcopy(history)
    before = provider_client._get_request_payload(history)
    projected = prepare_messages_for_model(history, model=provider_client.bind_tools([]))
    after = provider_client._get_request_payload(projected)
    assert after == before
    assert history == snapshot
    assert all(original is prepared for original, prepared in zip(history, projected))


@pytest.mark.parametrize('content', [
    [{'type': 'text', 'text': ''}], [{'type': 'text', 'text': '\n'}], ['', '\n'],
])
@pytest.mark.parametrize('status', ['success', 'error'])
def test_structured_empty_result_serializes_with_matching_id_for_other_providers(provider_client, content, status):
    history = [HumanMessage(content='go'), AIMessage(content='', tool_calls=[{
        'id': 'call-1', 'name': 'lookup', 'args': {},
    }]), ToolMessage(content=copy.deepcopy(content), tool_call_id='call-1', status=status)]
    snapshot = copy.deepcopy(history)
    projected = prepare_messages_for_model(history, model=provider_client)
    wire = provider_client._get_request_payload(projected)
    expected = EMPTY_ERROR_TOOL_RESULT_CONTENT if status == 'error' else EMPTY_SUCCESSFUL_TOOL_RESULT_CONTENT
    assert projected[-1].content == expected
    assert expected in json.dumps(wire)
    assert 'call-1' in json.dumps(wire)
    assert history == snapshot


def test_complete_parallel_and_repeated_calls_are_unchanged_for_other_providers(provider_client):
    from elitea_sdk.runtime.tools.llm import LLMNode
    history = [HumanMessage(content='go')]
    for iteration in (1, 2):
        calls = [{'id': f'call-{iteration}-{i}', 'name': 'lookup', 'args': {'index': i}} for i in (1, 2)]
        history.append(AIMessage(content='', tool_calls=calls, response_metadata={'trace': iteration}))
        history.extend(ToolMessage(content=f'result-{call["id"]}', tool_call_id=call['id']) for call in calls)
    before = provider_client._get_request_payload(history)
    filtered = LLMNode._filter_orphaned_tool_calls(history)
    projected = prepare_messages_for_model(filtered, model=provider_client)
    assert provider_client._get_request_payload(projected) == before
    assert all(original is prepared for original, prepared in zip(history, projected))
