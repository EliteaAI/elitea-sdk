from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage

from elitea_sdk.runtime.tools.llm import LLMNode


def _nested_config():
    return {
        'metadata': {
            'parent_agent_path': [
                {'name': 'General Purpose', 'call_id': 'call-1'},
            ],
        },
    }


def test_nested_length_completion_is_finished_inside_the_same_llm_node():
    client = Mock()
    client.invoke.return_value = AIMessage(
        content='34. item-34\n35. item-35',
        response_metadata={'finish_reason': 'stop'},
    )
    node = LLMNode(client=client)
    first = AIMessage(
        content='33. item-33',
        response_metadata={'finish_reason': 'length'},
    )

    result = node._continue_nested_output(
        messages=[HumanMessage(content='List the items')],
        completion=first,
        config=_nested_config(),
    )

    assert result.content == '33. item-33\n34. item-34\n35. item-35'
    client.invoke.assert_called_once()
    continuation_messages = client.invoke.call_args.args[0]
    assert continuation_messages[-3].content == '33. item-33'
    assert 'output only the missing continuation' in continuation_messages[-2].content.lower()


def test_direct_chat_length_completion_remains_user_controlled():
    client = Mock()
    node = LLMNode(client=client)
    first = AIMessage(
        content='partial answer',
        response_metadata={'finish_reason': 'length'},
    )

    result = node._continue_nested_output(
        messages=[HumanMessage(content='Write an answer')],
        completion=first,
        config={'metadata': {}},
    )

    assert result is first
    client.invoke.assert_not_called()


def test_continuation_merge_preserves_output_limit_seams():
    assert LLMNode._merge_continuation_text('64. item-64\n65', '. item-65') == (
        '64. item-64\n65. item-65'
    )
    assert LLMNode._merge_continuation_text('97. item', '-97\n98. item-98') == (
        '97. item-97\n98. item-98'
    )
    assert LLMNode._merge_continuation_text('33. item-33', '34. item-34') == (
        '33. item-33\n34. item-34'
    )
    assert LLMNode._merge_continuation_text('33. item-33', '\n34. item-34') == (
        '33. item-33\n34. item-34'
    )


def test_reasoning_only_nested_exhaustion_retries_for_visible_output():
    client = Mock()
    client.invoke.return_value = AIMessage(
        content='visible answer',
        response_metadata={'status': 'completed'},
    )
    node = LLMNode(client=client)
    first = AIMessage(
        content=[{'type': 'reasoning', 'reasoning': 'hidden reasoning'}],
        response_metadata={
            'status': 'incomplete',
            'incomplete_details': {'reason': 'max_output_tokens'},
        },
    )

    result = node._continue_nested_output(
        messages=[HumanMessage(content='Solve the task')],
        completion=first,
        config=_nested_config(),
    )

    assert result.content == 'visible answer'
    continuation_messages = client.invoke.call_args.args[0]
    assert not any(
        isinstance(message, AIMessage) and message.content == 'hidden reasoning'
        for message in continuation_messages
    )
