"""SDK-side regressions for #6830; original checkpoint content stays intact."""

import copy
import json

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from elitea_sdk.runtime.langchain.utils import (
    EMPTY_ERROR_TOOL_RESULT_CONTENT,
    EMPTY_SUCCESSFUL_TOOL_RESULT_CONTENT,
    prepare_messages_for_model,
)
from elitea_sdk.runtime.tools.llm import LLMNode


DOCUMENT = {'type': 'document', 'source': {
    'type': 'text', 'media_type': 'text/plain', 'data': 'important report',
}}
SEARCH_RESULT = {'type': 'search_result', 'source': 'kb:1', 'title': 'Result',
                 'content': [{'type': 'text', 'text': 'important finding'}]}
IMAGE = {'type': 'image', 'source': {
    'type': 'base64', 'media_type': 'image/png', 'data': 'aW1hZ2U=',
}}


@pytest.mark.parametrize('content', [
    [{'type': 'text', 'text': ''}],
    [{'type': 'text', 'text': '\n'}],
    ['', '\n', {'type': 'text', 'text': ' '}],
])
@pytest.mark.parametrize('status', ['success', 'error'])
def test_structured_blank_result_uses_placeholder_without_changing_history(content, status):
    original = ToolMessage(content=content, tool_call_id='call-1', status=status,
                           id='message-1', name='lookup', artifact={'raw': 'retained'},
                           response_metadata={'source': 'test'})
    before = original.model_dump()
    prepared = prepare_messages_for_model([original])[0]
    expected = EMPTY_ERROR_TOOL_RESULT_CONTENT if status == 'error' else EMPTY_SUCCESSFUL_TOOL_RESULT_CONTENT
    assert prepared.content == expected
    assert prepared.model_dump(exclude={'content'}) == original.model_dump(exclude={'content'})
    assert original.model_dump() == before


@pytest.mark.parametrize('block', [DOCUMENT, SEARCH_RESULT, {'type': 'json', 'json': {'answer': 42}}])
def test_databricks_fallback_preserves_unsupported_block_as_json_text(block):
    client = ChatAnthropic(model='global.databricks.claude-haiku-4-5', api_key='offline')
    original = ToolMessage(content=[block], tool_call_id='call-1')
    before = copy.deepcopy(original.content)
    prepared = prepare_messages_for_model([original], model=client.bind_tools([]))[0]
    assert json.loads(prepared.content[0]['text']) == block
    assert prepared.content[0]['type'] == 'text'
    assert original.content == before
    assert prepare_messages_for_model([prepared], model=client)[0] is prepared


def test_databricks_mixed_result_retains_text_image_and_document_data():
    original = ToolMessage(content=[{'type': 'text', 'text': '\n'},
                                    {'type': 'text', 'text': 'summary'}, IMAGE, DOCUMENT],
                           tool_call_id='call-1')
    prepared = prepare_messages_for_model([original], model='databricks/claude')[0]
    assert prepared.content[:2] == [{'type': 'text', 'text': 'summary'}, IMAGE]
    assert json.loads(prepared.content[2]['text']) == DOCUMENT


@pytest.mark.parametrize('model', [None, 'claude-sonnet-4-5', 'bedrock/claude', 'gpt-4.1'])
def test_other_providers_preserve_native_rich_content(model):
    original = ToolMessage(content=[DOCUMENT, SEARCH_RESULT, IMAGE], tool_call_id='call-1')
    assert prepare_messages_for_model([original], model=model)[0] is original


def test_json_looking_strings_are_unchanged_for_databricks():
    messages = [ToolMessage(content=value, tool_call_id=f'call-{i}')
                for i, value in enumerate(['[]', '{}', 'null'])]
    assert all(a is b for a, b in zip(messages, prepare_messages_for_model(messages, model='databricks/claude')))


@pytest.mark.parametrize('parsed_calls', [[], [{'id': 'call-1', 'name': 'lookup', 'args': {}}]])
def test_raw_orphan_is_removed_even_if_parsed_calls_have_no_orphans(parsed_calls):
    original = AIMessage(content=[{'type': 'thinking', 'thinking': 'plan', 'signature': 'sig'},
                                  {'type': 'tool_use', 'id': 'call-invalid', 'name': 'lookup', 'input': {}},
                                  *([{'type': 'tool_use', 'id': 'call-1', 'name': 'lookup', 'input': {}}]
                                    if parsed_calls else [])],
                         tool_calls=parsed_calls,
                         invalid_tool_calls=[{'id': 'call-invalid', 'name': 'lookup',
                                              'args': '{}{}', 'error': 'invalid JSON'}],
                         id='assistant-1', response_metadata={'source': 'test'})
    before = original.model_dump()
    history = [HumanMessage(content='go'), original]
    if parsed_calls:
        history.append(ToolMessage(content='ok', tool_call_id='call-1'))
    prepared = LLMNode._filter_orphaned_tool_calls(history)
    assistant = prepared[1]
    assert assistant.content[0] == original.content[0]
    assert [b['id'] for b in assistant.content if b['type'] == 'tool_use'] == (
        ['call-1'] if parsed_calls else [])
    assert assistant.invalid_tool_calls == []
    assert assistant.id == original.id
    assert assistant.response_metadata == original.response_metadata
    assert original.model_dump() == before


def test_raw_call_with_immediately_following_result_is_preserved():
    assistant = AIMessage(content=[{'type': 'tool_use', 'id': 'call-1', 'name': 'lookup', 'input': {}}])
    history = [assistant, ToolMessage(content='ok', tool_call_id='call-1')]
    assert LLMNode._filter_orphaned_tool_calls(history) == history


def test_auto_routing_prepares_results_after_resolving_native_databricks_model():
    from elitea_sdk.runtime.clients.routing import AutoChatModel
    client = ChatAnthropic(model='global.databricks.claude-haiku-4-5', api_key='offline')
    original = ToolMessage(content=[DOCUMENT], tool_call_id='call-1')
    projected = AutoChatModel._native_messages(client.bind_tools([]), [original], {'invocation_id': 'test'})
    assert json.loads(projected[0].content[0]['text']) == DOCUMENT
    assert original.content == [DOCUMENT]


def test_partial_blank_content_is_removed_without_losing_native_blocks():
    original = ToolMessage(content=['', {'type': 'text', 'text': '\n'}, IMAGE,
                                    {'type': 'text', 'text': 'useful'}], tool_call_id='call-1')
    prepared = prepare_messages_for_model([original])[0]
    assert prepared.content == [IMAGE, {'type': 'text', 'text': 'useful'}]
    assert len(original.content) == 4


@pytest.mark.parametrize('block', [DOCUMENT, SEARCH_RESULT])
def test_native_anthropic_serialization_retains_rich_block(block):
    client = ChatAnthropic(model='claude-sonnet-4-5', api_key='offline')
    history = [HumanMessage(content='go'), AIMessage(content='', tool_calls=[{
        'id': 'call-1', 'name': 'lookup', 'args': {},
    }]), ToolMessage(content=[block], tool_call_id='call-1')]
    payload = client._get_request_payload(prepare_messages_for_model(history, model=client))
    assert payload['messages'][-1]['content'][0]['content'] == [block]
