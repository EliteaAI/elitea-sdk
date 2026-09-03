"""Provider-bound ToolMessage content contract (#6353)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
