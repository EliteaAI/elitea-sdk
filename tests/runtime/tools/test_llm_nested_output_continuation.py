from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.langraph_agent import create_graph
from elitea_sdk.runtime.tools.llm import LLMNode


def _nested_config():
    return {
        'metadata': {
            'parent_agent_path': [
                {'name': 'General Purpose', 'call_id': 'call-1'},
            ],
        },
    }


def _pipeline_config():
    return {
        'metadata': {
            'langgraph_node': 'LLM1',
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


def test_direct_pipeline_llm_node_is_finished_automatically():
    client = Mock()
    client.invoke.return_value = AIMessage(
        content='missing ending',
        response_metadata={'finish_reason': 'stop'},
    )
    node = LLMNode(client=client)
    first = AIMessage(
        content='partial answer ',
        response_metadata={'finish_reason': 'length'},
    )

    result = node._continue_nested_output(
        messages=[HumanMessage(content='Write an answer')],
        completion=first,
        config=_pipeline_config(),
    )

    assert result.content == 'partial answer missing ending'
    client.invoke.assert_called_once()


class _PipelineLLM:
    def __init__(self):
        self.calls = []
        self.responses = [
            AIMessage(
                content='first part ',
                response_metadata={'finish_reason': 'length'},
            ),
            AIMessage(
                content='second part',
                response_metadata={'finish_reason': 'stop'},
            ),
            AIMessage(
                content='downstream result',
                response_metadata={'finish_reason': 'stop'},
            ),
        ]

    def invoke(self, messages, config=None):
        self.calls.append(list(messages))
        return self.responses.pop(0)


def test_assembled_output_is_mapped_to_state_before_the_next_pipeline_node():
    client = _PipelineLLM()
    pipeline = create_graph(
        client=client,
        yaml_schema='''
name: continuation-state
state:
  input:
    type: str
  draft:
    type: str
  final:
    type: str
  messages:
    type: list
nodes:
  - id: LLM1
    type: llm
    input_mapping:
      system:
        type: fixed
        value: Write the draft.
      task:
        type: variable
        value: input
    input: [input]
    output: [draft]
    transition: LLM2
  - id: LLM2
    type: llm
    input_mapping:
      system:
        type: fixed
        value: Use the complete draft.
      task:
        type: variable
        value: draft
    input: [draft]
    output: [final]
    transition: END
entry_point: LLM1
''',
        tools=[],
        memory=MemorySaver(),
    )
    config = {'configurable': {'thread_id': 'direct-pipeline-continuation'}}

    pipeline.invoke({'input': 'start'}, config=config)

    state = pipeline.get_state(config).values
    assert state['draft'] == 'first part second part'
    assert state['final'] == 'downstream result'
    assert client.calls[2][-1].content == 'first part second part'


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
