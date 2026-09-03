"""End-to-end LLMNode regression for strict provider tool-message validation."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from elitea_sdk.runtime.tools.llm import LLMNode


class _StrictOpenAICompatibleClient:
    """Reject the malformed request shape reported by Databricks in #6353."""

    def __init__(self):
        self.calls = []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None, **kwargs):
        self.calls.append(list(messages))
        requested_ids = {
            tool_call['id']
            for message in messages
            if isinstance(message, AIMessage)
            for tool_call in (message.tool_calls or [])
        }
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            if not isinstance(message.tool_call_id, str) or not message.tool_call_id:
                raise RuntimeError('DatabricksException: BAD_REQUEST: Invalid tool_call_id.')
            if message.tool_call_id not in requested_ids:
                raise RuntimeError('DatabricksException: BAD_REQUEST: Unmatched tool_call_id.')
            if not message.content or (
                isinstance(message.content, str) and not message.content.strip()
            ):
                raise RuntimeError(
                    'DatabricksException: BAD_REQUEST: Missing content in the tool message.'
                )

        if len(self.calls) == 1:
            return AIMessage(
                content='',
                tool_calls=[{'name': 'empty_result', 'args': {}, 'id': 'call-1'}],
            )
        return AIMessage(content='done')


def _empty_result():
    """Complete successfully without producing output."""
    return ''


def test_empty_tool_result_is_projected_to_non_empty_provider_content():
    client = _StrictOpenAICompatibleClient()
    tool = StructuredTool.from_function(
        func=_empty_result,
        name='empty_result',
        description='Return no output.',
        metadata={
            'toolkit_type': 'test',
            'toolkit_name': 'test',
            'tool_name': 'empty_result',
        },
    )
    node = LLMNode(
        client=client,
        available_tools=[tool],
        tool_names=['empty_result'],
        lazy_tools_mode=False,
        input_mapping={},
        output_variables=['messages'],
    )

    result = node.invoke(
        {'messages': [HumanMessage(content='go')]},
        config={'configurable': {'thread_id': 'strict-provider-empty-result'}},
    )

    assert result['messages'][-1].content == 'done'
    provider_tool_message = next(
        message for message in client.calls[1]
        if isinstance(message, ToolMessage)
    )
    assert provider_tool_message.tool_call_id == 'call-1'
    assert provider_tool_message.content.strip()

    # Provider compatibility is a projection: persisted/audited tool output remains raw.
    history_tool_message = next(
        message for message in result['messages']
        if isinstance(message, ToolMessage)
    )
    assert history_tool_message.content == ''
