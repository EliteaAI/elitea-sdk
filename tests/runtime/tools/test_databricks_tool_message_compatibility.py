"""#6830: real SDK tool loop -> ChatAnthropic -> LiteLLM -> Databricks payload.

No provider credentials or network are used. Set ELITEA_DATABRICKS_COMPAT_TESTS=1
and install databricks-compatibility-requirements.txt in an isolated overlay to
enable this suite; LiteLLM is not an SDK dependency. CI runs this separately.
The HTTP transport supplies deterministic model responses. Both adapter stages
are the real pinned implementation, not a fixture of their expected output.
Passing proves pairing/content preservation at the Databricks request boundary,
not server-side acceptance of synthetic thinking signatures or image bytes.
"""

import copy
import json
import os
from importlib.metadata import version
from unittest.mock import patch

import anthropic
import httpx
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from elitea_sdk.runtime.langchain.utils import prepare_messages_for_model
from elitea_sdk.runtime.tools.llm import LLMNode
from tests.runtime.langchain.test_databricks_tool_results import DOCUMENT, IMAGE, SEARCH_RESULT
from tests.runtime.tools.test_openapi_tool_message_provider_contract import _assert_tool_pairs, _WireProvider


pytestmark = pytest.mark.skipif(
    os.environ.get('ELITEA_DATABRICKS_COMPAT_TESTS') != '1',
    reason='Run the isolated pinned Databricks compatibility suite explicitly',
)


@pytest.fixture(scope='module')
def gateway():
    # LiteLLM imports otherwise fetch its cost map, unrelated to these adapters.
    with patch.dict(os.environ, {'LITELLM_LOCAL_MODEL_COST_MAP': 'True'}):
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter
        from litellm.llms.databricks.chat.transformation import DatabricksConfig
    assert version('litellm') == '1.83.14', 'Run this regression against the pinned customer adapter'

    def convert(payload):
        messages = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
            messages=copy.deepcopy(payload['messages']), model='databricks/claude-haiku-4-5',
        )
        return DatabricksConfig().transform_request(
            model='claude-haiku-4-5', messages=messages, optional_params={},
            litellm_params={}, headers={},
        )['messages']
    return convert


RESULTS = [
    pytest.param('[]', '[]', id='json-array-string'),
    pytest.param('{}', '{}', id='json-object-string'),
    pytest.param('', 'returned no content', id='empty-string'),
    pytest.param([{'type': 'text', 'text': ''}], 'returned no content', id='empty-text-block'),
    pytest.param([{'type': 'text', 'text': '\n'}], 'returned no content', id='whitespace-text-block'),
    pytest.param([SEARCH_RESULT], 'important finding', id='search-result'),
    pytest.param([DOCUMENT], 'important report', id='document'),
    pytest.param([{'type': 'text', 'text': 'summary'}, DOCUMENT], 'important report', id='mixed-document'),
    pytest.param([{'type': 'text', 'text': 'summary'}, IMAGE], 'summary', id='text-and-image'),
]


@pytest.mark.parametrize('content,expected', RESULTS)
@pytest.mark.parametrize('streaming', [False, True], ids=['sync', 'stream'])
@pytest.mark.parametrize('sibling_count', [1, 2], ids=['sequential', 'parallel'])
def test_real_tool_loop_preserves_pairs_and_content_through_gateway(gateway, content, expected, streaming, sibling_count):
    def lookup() -> object:
        """Return the deterministic compatibility probe result."""
        return copy.deepcopy(content)

    provider = _WireProvider('anthropic', sibling_count, {})
    converted_requests = []

    def respond(request):
        converted = gateway(json.loads(request.content))
        converted_requests.append(converted)
        try:
            _assert_tool_pairs(converted, 'openai')
        except AssertionError as error:
            return httpx.Response(400, json={'type': 'error', 'error': {
                'type': 'invalid_request_error', 'message': str(error),
            }})
        return provider.respond(request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = ChatAnthropic(model='global.databricks.claude-haiku-4-5', api_key='offline',
                               streaming=streaming, max_tokens=1024, max_retries=0)
        client.__dict__['_client'] = anthropic.Anthropic(api_key='offline', http_client=http_client, max_retries=0)
        node = LLMNode(client=client, available_tools=[StructuredTool.from_function(lookup)],
                       tool_names=['lookup'], lazy_tools_mode=False, input_mapping={}, output_variables=['messages'])
        result = node.invoke({'messages': [HumanMessage(content='Look up results twice.')]},
                             config={'configurable': {'thread_id': 'offline-6830-compat'}})

    assert result['messages'][-1].text == 'done'
    assert len(converted_requests) == 3  # Two real tool-loop iterations.
    results = _assert_tool_pairs(converted_requests[-1], 'openai')
    assert len(results) == 2 * sibling_count
    assert all(expected in json.dumps(value, ensure_ascii=False) for _, value in results)
    raw_results = [message.content for message in result['messages'] if isinstance(message, ToolMessage)]
    assert raw_results == [content] * (2 * sibling_count)
    if content == [SEARCH_RESULT] or content == [DOCUMENT]:
        assert all(json.loads(value)['type'] == content[0]['type'] for _, value in results)
    if content == [{'type': 'text', 'text': 'summary'}, IMAGE]:
        assert all(any(block['type'] == 'image_url' for block in value) for _, value in results)


@pytest.mark.parametrize('content', [
    [{'type': 'text', 'text': ''}], [{'type': 'text', 'text': '\n'}], [SEARCH_RESULT], [DOCUMENT],
])
def test_negative_controls_reproduce_missing_result_then_pass_after_preparation(gateway, content):
    client = ChatAnthropic(model='databricks/claude-haiku-4-5', api_key='offline')
    history = [HumanMessage(content='go'), AIMessage(content='', tool_calls=[{
        'id': 'call-1', 'name': 'lookup', 'args': {},
    }]), ToolMessage(content=content, tool_call_id='call-1')]
    raw = gateway(client._get_request_payload(history))
    with pytest.raises(AssertionError, match='Tool use without tool result immediately after'):
        _assert_tool_pairs(raw, 'openai')
    repaired = prepare_messages_for_model(history, model=client)
    assert len(_assert_tool_pairs(gateway(client._get_request_payload(repaired)), 'openai')) == 1


def test_raw_orphan_negative_control_then_filtered_history_passes(gateway):
    client = ChatAnthropic(model='databricks/claude-haiku-4-5', api_key='offline')
    assistant = AIMessage(content=[{'type': 'thinking', 'thinking': 'plan', 'signature': 'sig'},
                                   {'type': 'tool_use', 'id': 'call-1', 'name': 'lookup', 'input': {}},
                                   {'type': 'tool_use', 'id': 'call-invalid', 'name': 'lookup', 'input': {}}],
                          tool_calls=[{'id': 'call-1', 'name': 'lookup', 'args': {}}])
    history = [HumanMessage(content='go'), assistant, ToolMessage(content='ok', tool_call_id='call-1')]
    with pytest.raises(AssertionError, match='Tool use without tool result immediately after'):
        _assert_tool_pairs(gateway(client._get_request_payload(history)), 'openai')
    cleaned = LLMNode._filter_orphaned_tool_calls(history)
    assert _assert_tool_pairs(gateway(client._get_request_payload(cleaned)), 'openai') == [('call-1', 'ok')]


@pytest.mark.parametrize('content', [
    ['', '\n'], [{'type': 'json', 'json': {'answer': 42}}],
    [{'type': 'text', 'content': 'nonstandard text field'}], [IMAGE],
])
def test_prebuilt_checkpoint_results_keep_pairing_after_projection(gateway, content):
    client = ChatAnthropic(model='databricks/claude-haiku-4-5', api_key='offline')
    assistant = AIMessage(content=[{'type': 'redacted_thinking', 'data': 'fixture'},
                                   {'type': 'tool_use', 'id': 'call-1', 'name': 'lookup', 'input': {}}],
                          tool_calls=[{'id': 'call-1', 'name': 'lookup', 'args': {}}])
    history = [HumanMessage(content='go'), assistant, ToolMessage(content=content, tool_call_id='call-1')]
    projected = prepare_messages_for_model(history, model=client)
    assert len(_assert_tool_pairs(gateway(client._get_request_payload(projected)), 'openai')) == 1


@pytest.mark.parametrize('messages', [
    [{'role': 'assistant', 'tool_calls': [{'id': 'a'}]}, {'role': 'tool', 'tool_call_id': 'b', 'content': 'ok'}],
    [{'role': 'assistant', 'tool_calls': [{'id': 'a'}]}, {'role': 'user', 'content': 'intervening'},
     {'role': 'tool', 'tool_call_id': 'a', 'content': 'ok'}],
    [{'role': 'assistant', 'tool_calls': [{'id': 'a'}]}, {'role': 'tool', 'tool_call_id': 'a', 'content': 'ok'},
     {'role': 'tool', 'tool_call_id': 'a', 'content': 'duplicate'}],
])
def test_pairing_oracle_rejects_wrong_id_intervening_message_and_duplicate(messages):
    with pytest.raises(AssertionError):
        _assert_tool_pairs(messages, 'openai')
