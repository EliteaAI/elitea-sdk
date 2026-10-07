"""Offline wire-contract checks for #6830; no Databricks credentials required.

Execute the real OpenAPI toolkit and LLMNode through real LangChain clients.
Only the HTTP transports are replaced. The pairing oracle checks the serialized
request, including immediate adjacency, rather than a union of IDs in history.
This proves the SDK's request contract, not acceptance by a live gateway/model.
The non-empty content check emulates #6353's reported Databricks restriction;
native Anthropic also permits empty tool results when the block is present.

References:
https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference
https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
"""

import copy
import json

import anthropic
import httpx
import pytest
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from elitea_sdk.runtime.langchain.utils import EMPTY_SUCCESSFUL_TOOL_RESULT_CONTENT
from elitea_sdk.runtime.tools import llm as llm_module
from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.tools.openapi import EliteAOpenAPIToolkit


SPEC = {
    'openapi': '3.0.0',
    'info': {'title': 'Offline results API', 'version': '1.0'},
    'servers': [{'url': 'https://openapi.invalid'}],
    'paths': {'/results': {'get': {
        'operationId': 'lookup',
        'description': 'Look up results.',
        'responses': {'200': {'description': 'Result'}, '204': {'description': 'No content'}},
    }}},
}


def _assert_tool_pairs(messages, dialect):
    """Every result must match this turn's calls, before any intervening turn."""
    pending = set()
    seen = set()
    results = []
    for message in messages:
        content = message.get('content') or []
        blocks = content if isinstance(content, list) else []
        if dialect == 'anthropic':
            calls = [block['id'] for block in blocks if block['type'] == 'tool_use']
            tool_results = [
                (block['tool_use_id'], block.get('content'))
                for block in blocks if block['type'] == 'tool_result'
            ]
        else:
            calls = [call['id'] for call in message.get('tool_calls', [])]
            tool_results = (
                [(message['tool_call_id'], message.get('content'))]
                if message['role'] == 'tool' else []
            )
        if tool_results:
            assert message['role'] == ('user' if dialect == 'anthropic' else 'tool')
            for call_id, result in tool_results:
                assert call_id in pending, 'Unmatched or duplicate tool result'
                assert (
                    isinstance(result, str) and result.strip()
                    or isinstance(result, list) and any(
                        isinstance(block, dict) and (
                            block.get('type') == 'text' and str(block.get('text', '')).strip()
                            or block.get('type') in {'image', 'image_url'}
                        ) for block in result
                    )
                ), 'Empty tool content'
                pending.remove(call_id)
                results.append((call_id, result))
            if dialect == 'anthropic':
                assert not pending, 'Missing sibling tool result'
                assert all(block['type'] == 'tool_result' for block in blocks[:len(tool_results)])
        else:
            assert not pending, 'Tool use without tool result immediately after'
            assert all(isinstance(call_id, str) and call_id for call_id in calls)
            assert len(calls) == len(set(calls)), 'Duplicate tool call ID'
            assert not seen.intersection(calls), 'Replayed tool call ID'
            pending.update(calls)
            seen.update(calls)
    assert not pending, 'Tool use without tool result immediately after'
    return results


class _ApiSession:
    def __init__(self, body, status):
        self.headers = {}
        self.body = body
        self.status = status
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url))
        response = requests.Response()
        response.status_code = self.status
        response._content = self.body
        return response


class _WireProvider:
    def __init__(self, dialect, sibling_count, tool_args):
        self.dialect = dialect
        self.sibling_count = sibling_count
        self.tool_args = tool_args
        self.requests = []

    def respond(self, request):
        payload = json.loads(request.content)
        self.requests.append(payload)
        try:
            _assert_tool_pairs(payload['messages'], self.dialect)
        except AssertionError as error:
            return httpx.Response(400, json={
                'type': 'error',
                'error': {'type': 'invalid_request_error', 'message': str(error)},
            })
        round_number = len(self.requests)
        calls = [
            {'id': f'call-{round_number}-{index}', 'name': 'lookup', 'args': self.tool_args}
            for index in range(self.sibling_count)
        ] if round_number <= 2 else []
        if payload.get('stream'):
            return self._stream_response(payload, round_number, calls)
        if self.dialect == 'anthropic':
            content = [
                {'type': 'tool_use', 'id': call['id'], 'name': call['name'], 'input': call['args']}
                for call in calls
            ] or [{'type': 'text', 'text': 'done'}]
            response = {
                'id': f'msg-{round_number}', 'type': 'message', 'role': 'assistant',
                'model': payload['model'], 'content': content,
                'stop_reason': 'tool_use' if calls else 'end_turn', 'stop_sequence': None,
                'usage': {'input_tokens': 10, 'output_tokens': 10},
            }
        else:
            message = {'role': 'assistant', 'content': None if calls else 'done'}
            if calls:
                message['tool_calls'] = [
                    {'id': call['id'], 'type': 'function', 'function': {
                        'name': call['name'], 'arguments': json.dumps(call['args']),
                    }} for call in calls
                ]
            response = {
                'id': f'completion-{round_number}', 'object': 'chat.completion',
                'created': 1, 'model': payload['model'],
                'choices': [{'index': 0, 'message': message,
                             'finish_reason': 'tool_calls' if calls else 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20},
            }
        return httpx.Response(200, json=response)

    def _stream_response(self, payload, round_number, calls):
        if self.dialect == 'anthropic':
            events = [('message_start', {'type': 'message_start', 'message': {
                'id': f'msg-{round_number}', 'type': 'message', 'role': 'assistant',
                'model': payload['model'], 'content': [], 'stop_reason': None,
                'stop_sequence': None, 'usage': {'input_tokens': 10, 'output_tokens': 0},
            }})]
            for index, call in enumerate(calls):
                events.extend([
                    ('content_block_start', {'type': 'content_block_start', 'index': index,
                     'content_block': {'type': 'tool_use', 'id': call['id'],
                                       'name': call['name'], 'input': {}}}),
                    ('content_block_delta', {'type': 'content_block_delta', 'index': index,
                     'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(call['args'])}}),
                    ('content_block_stop', {'type': 'content_block_stop', 'index': index}),
                ])
            if not calls:
                events.extend([
                    ('content_block_start', {'type': 'content_block_start', 'index': 0,
                     'content_block': {'type': 'text', 'text': ''}}),
                    ('content_block_delta', {'type': 'content_block_delta', 'index': 0,
                     'delta': {'type': 'text_delta', 'text': 'done'}}),
                    ('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
                ])
            events.extend([
                ('message_delta', {'type': 'message_delta',
                 'delta': {'stop_reason': 'tool_use' if calls else 'end_turn', 'stop_sequence': None},
                 'usage': {'output_tokens': 10}}),
                ('message_stop', {'type': 'message_stop'}),
            ])
            content = ''.join(f'event: {name}\ndata: {json.dumps(data)}\n\n' for name, data in events)
        else:
            def chunk(delta, finish_reason=None):
                return json.dumps({
                    'id': f'completion-{round_number}', 'object': 'chat.completion.chunk',
                    'created': 1, 'model': payload['model'],
                    'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}],
                })

            chunks = [chunk({'role': 'assistant'})]
            for index, call in enumerate(calls):
                chunks.extend([
                    chunk({'tool_calls': [{'index': index, 'id': call['id'], 'type': 'function',
                                          'function': {'name': call['name'], 'arguments': ''}}]}),
                    chunk({'tool_calls': [{'index': index,
                                          'function': {'arguments': json.dumps(call['args'])}}]}),
                ])
            if not calls:
                chunks.append(chunk({'content': 'done'}))
            chunks.append(chunk({}, 'tool_calls' if calls else 'stop'))
            content = ''.join(f'data: {data}\n\n' for data in chunks) + 'data: [DONE]\n\n'
        return httpx.Response(200, text=content, headers={'content-type': 'text/event-stream'})


def _invoke_openapi_loop(dialect, streaming, sibling_count, body, status, tool_args):
    toolkit = EliteAOpenAPIToolkit.get_toolkit(
        toolkit_name='offline', openapi_configuration={'spec': SPEC},
    )
    session = _ApiSession(body, status)
    wrapper = toolkit.request_session
    wrapper._client._requestor = session
    wrapper._client._collect_operations()
    provider = _WireProvider(dialect, sibling_count, tool_args)

    with httpx.Client(transport=httpx.MockTransport(provider.respond)) as http_client:
        if dialect == 'anthropic':
            client = ChatAnthropic(
                model='claude-haiku-4-5', api_key='unused', max_retries=0,
                streaming=streaming, max_tokens=1024,
            )
            client.__dict__['_client'] = anthropic.Anthropic(
                api_key='unused', http_client=http_client, max_retries=0,
            )
        else:
            client = ChatOpenAI(
                model='databricks-claude-haiku-4-5', api_key='unused', max_retries=0,
                base_url='https://provider.invalid/v1', http_client=http_client,
                streaming=streaming,
            )
        node = LLMNode(
            client=client, available_tools=toolkit.get_tools(), tool_names=['lookup'],
            lazy_tools_mode=False, input_mapping={}, output_variables=['messages'],
        )
        result = node.invoke(
            {'messages': [HumanMessage(content='Look up results twice.')]},
            config={'configurable': {'thread_id': 'offline-6830'}},
        )
    return result, provider, session


@pytest.mark.parametrize('dialect', ['anthropic', 'openai'])
@pytest.mark.parametrize('streaming', [False, True], ids=['sync', 'stream'])
@pytest.mark.parametrize('sibling_count', [1, 2])
@pytest.mark.parametrize('body,status,tool_args,expected', [
    (b'[]', 200, {}, '[]'),
    (b'{}', 200, {}, '{}'),
    (b'null', 200, {}, 'null'),
    (b'{"value": []}', 200, {}, '{"value": []}'),
    (b'', 204, {}, EMPTY_SUCCESSFUL_TOOL_RESULT_CONTENT),
    (b' \n', 200, {}, EMPTY_SUCCESSFUL_TOOL_RESULT_CONTENT),
    (b'[]', 200, {'regexp': r'.+'}, EMPTY_SUCCESSFUL_TOOL_RESULT_CONTENT),
], ids=['empty-array', 'empty-object', 'json-null', 'nested-empty-array',
        'http-204', 'whitespace', 'regexp-removes-output'])
def test_openapi_results_preserve_two_iterations_of_wire_tool_pairs(
    dialect, streaming, sibling_count, body, status, tool_args, expected,
):
    result, provider, session = _invoke_openapi_loop(
        dialect, streaming, sibling_count, body, status, tool_args,
    )

    assert result['messages'][-1].text == 'done'
    assert len(provider.requests) == 3
    assert len(session.calls) == 2 * sibling_count
    wire_results = _assert_tool_pairs(provider.requests[-1]['messages'], dialect)
    assert wire_results == [
        (f'call-{round_number}-{index}', expected)
        for round_number in (1, 2) for index in range(sibling_count)
    ]
    raw_results = [message.content for message in result['messages'] if isinstance(message, ToolMessage)]
    raw = '' if tool_args else body.decode()
    assert raw_results == [raw] * (2 * sibling_count)


@pytest.mark.parametrize('dialect', ['anthropic', 'openai'])
@pytest.mark.parametrize('body', [b'[]', b'{}', b''])
def test_without_6353_projection_only_truly_blank_content_fails(monkeypatch, dialect, body):
    # Reintroduce only #6353's missing projection. Empty JSON still has content;
    # a blank body fails despite both IDs being correctly paired on the wire.
    monkeypatch.setattr(llm_module, 'prepare_messages_for_model', lambda messages, **kwargs: list(messages))
    result, provider, _ = _invoke_openapi_loop(dialect, False, 1, body, 200, {})
    if body:
        assert result['messages'][-1].text == 'done'
        assert len(provider.requests) == 3
    else:
        assert 'Empty tool content' in result['messages'][-1].text
        assert len(provider.requests) == 2
        messages = copy.deepcopy(provider.requests[-1]['messages'])
        if dialect == 'anthropic':
            messages[-1]['content'][0]['content'] = '[]'
        else:
            messages[-1]['content'] = '[]'
        assert _assert_tool_pairs(messages, dialect) == [('call-1-0', '[]')]


@pytest.mark.parametrize('defect', ['missing-result', 'wrong-id', 'intervening-message', 'empty-content'])
def test_pairing_oracle_rejects_corrupt_anthropic_requests(defect):
    messages = [
        {'role': 'assistant', 'content': [
            {'type': 'tool_use', 'id': 'call-1', 'name': 'lookup', 'input': {}},
        ]},
        {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'call-1', 'content': '[]'},
        ]},
    ]
    assert _assert_tool_pairs(messages, 'anthropic') == [('call-1', '[]')]
    broken = copy.deepcopy(messages)
    if defect == 'missing-result':
        broken.pop()
    elif defect == 'wrong-id':
        broken[1]['content'][0]['tool_use_id'] = 'unmatched'
    elif defect == 'intervening-message':
        broken.insert(1, {'role': 'user', 'content': 'Wait.'})
    else:
        broken[1]['content'][0]['content'] = ''
    with pytest.raises(AssertionError):
        _assert_tool_pairs(broken, 'anthropic')
