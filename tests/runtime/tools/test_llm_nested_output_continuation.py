from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.exceptions import OutputContinuationExhausted
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
        content='33. item-33\n34. item-34\n35. item-35',
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
    assert continuation_messages[-2].content == '33. item-33'
    assert 'original task' in continuation_messages[-1].content.lower()


def test_standalone_model_call_remains_user_controlled():
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


def test_single_node_chat_graph_is_finished_automatically():
    client = Mock()
    client.invoke.return_value = AIMessage(
        content='partial answer missing ending',
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
        config={'metadata': {'langgraph_node': 'agent'}},
    )

    assert result.content == 'partial answer missing ending'
    client.invoke.assert_called_once()


def test_unanchored_meta_response_is_rejected_without_corrupting_partial_output():
    client = Mock()
    refusal = AIMessage(
        content=(
            "I can continue, but I can't reliably resume from the exact cutoff "
            "without seeing the last sentence."
        ),
        response_metadata={'finish_reason': 'stop'},
    )
    client.invoke.side_effect = [refusal, refusal]
    node = LLMNode(client=client)
    first = AIMessage(
        content="That afternoon, while the fog pressed in and the town's une",
        response_metadata={'finish_reason': 'length'},
    )

    with pytest.raises(OutputContinuationExhausted) as raised:
        node._continue_nested_output(
            messages=[HumanMessage(content='Write a 1200-word story.')],
            completion=first,
            config={'metadata': {'langgraph_node': 'agent'}},
        )

    assert raised.value.attempts == 2
    assert raised.value.failure_reason == 'invalid_continuation'
    assert raised.value.partial_output == first.content
    assert 'verified output boundary' in raised.value.user_message
    assert client.invoke.call_count == 2
    assert 'boundary repair' in (
        client.invoke.call_args.args[0][-1].content.lower()
    )


def test_invalid_seam_gets_one_strict_retry_without_merging_rewritten_text():
    first_text = 'The raven spoke. She'
    client = Mock()
    client.invoke.side_effect = [
        AIMessage(
            content='“should have known,” Mara said.',
            response_metadata={'finish_reason': 'length'},
        ),
        AIMessage(
            content='The raven spoke. She should have known.',
            response_metadata={'finish_reason': 'stop'},
        ),
    ]
    node = LLMNode(client=client)

    result = node._continue_nested_output(
        messages=[HumanMessage(content='Write the story.')],
        completion=AIMessage(
            content=first_text,
            response_metadata={'finish_reason': 'length'},
        ),
        config={'metadata': {'langgraph_node': 'agent'}},
    )

    assert result.content == 'The raven spoke. She should have known.'
    assert '“should have known,”' not in result.content
    assert client.invoke.call_count == 2


def test_each_continuation_uses_fresh_context_with_accumulated_output_and_progress():
    client = Mock()
    client.invoke.side_effect = [
        AIMessage(
            content='Part one. Part two.',
            response_metadata={'finish_reason': 'length'},
        ),
        AIMessage(
            content='Part one. Part two. Done.',
            response_metadata={'finish_reason': 'stop'},
        ),
    ]
    original_messages = [HumanMessage(content='Write a three-part answer.')]
    node = LLMNode(client=client)
    first = AIMessage(
        content='Part one.',
        response_metadata={'finish_reason': 'length'},
    )

    result = node._continue_nested_output(
        messages=original_messages,
        completion=first,
        config={'metadata': {'langgraph_node': 'agent'}},
    )

    assert result.content == 'Part one. Part two. Done.'
    assert client.invoke.call_count == 2
    first_request = client.invoke.call_args_list[0].args[0]
    second_request = client.invoke.call_args_list[1].args[0]
    assert len(first_request) == len(original_messages) + 2
    assert len(second_request) == len(original_messages) + 2
    assert first_request[-2].content == 'Part one.'
    assert second_request[-2].content == 'Part one. Part two.'
    assert '2 words' in first_request[-1].content
    assert '4 words' in second_request[-1].content
    assert 'original task' in second_request[-1].content.lower()


@pytest.mark.parametrize(
    ('prompt', 'expected'),
    [
        ('Write a complete 1200-word story now.', 1200),
        ('Explain the design in approximately 900 words.', 900),
        ('Please tell me in detail how to build a Rust worker system.', None),
        ('Review this 1200-word story and identify plot holes.', None),
        ('Summarize the attached 800-word report.', None),
    ],
)
def test_requested_output_word_target_is_confident_and_optional(prompt, expected):
    assert LLMNode._requested_output_word_target(
        [HumanMessage(content=prompt)]
    ) == expected


def test_explicit_word_target_bounds_the_next_continuation_and_stops_at_boundary():
    initial_words = [f'word-{index}' for index in range(70)]
    initial = ' '.join(initial_words)
    completed = f"{initial} {' '.join(f'word-{index}' for index in range(70, 100))}."
    client = Mock()
    client.max_tokens = 1000
    client.invoke.return_value = AIMessage(
        content=completed,
        response_metadata={'finish_reason': 'length'},
    )
    node = LLMNode(client=client)

    result = node._continue_nested_output(
        messages=[HumanMessage(content='Write a complete 100-word story.')],
        completion=AIMessage(
            content=initial,
            response_metadata={'finish_reason': 'length'},
        ),
        config={'metadata': {'langgraph_node': 'agent'}},
    )

    assert result.content == completed
    client.invoke.assert_called_once()
    continuation_prompt = client.invoke.call_args.args[0][-1].content
    assert '100 words' in continuation_prompt
    assert '30 words remain' in continuation_prompt
    assert 45 < client.invoke.call_args.kwargs['max_tokens'] < client.max_tokens


def test_word_target_already_reached_at_safe_boundary_needs_no_continuation():
    completed = f"{'word ' * 99}done."
    client = Mock()
    client.max_tokens = 1000
    node = LLMNode(client=client)

    result = node._continue_nested_output(
        messages=[HumanMessage(content='Write a 100-word story.')],
        completion=AIMessage(
            content=completed,
            response_metadata={'finish_reason': 'length'},
        ),
        config={'metadata': {'langgraph_node': 'agent'}},
    )

    assert result.content == completed
    client.invoke.assert_not_called()


def test_word_target_uses_one_bounded_closure_when_cut_mid_sentence():
    initial = f"{'word ' * 100}unfinished"
    client = Mock()
    client.max_tokens = 1000
    client.invoke.return_value = AIMessage(
        content=f'{initial} ending.',
        response_metadata={'finish_reason': 'length'},
    )
    node = LLMNode(client=client)

    result = node._continue_nested_output(
        messages=[HumanMessage(content='Write a 100-word story.')],
        completion=AIMessage(
            content=initial,
            response_metadata={'finish_reason': 'length'},
        ),
        config={'metadata': {'langgraph_node': 'agent'}},
    )

    assert result.content == f'{initial} ending.'
    client.invoke.assert_called_once()
    continuation_prompt = client.invoke.call_args.args[0][-1].content
    assert 'closure-only mode' in continuation_prompt.lower()
    assert client.invoke.call_args.kwargs['max_tokens'] == 128


def test_direct_pipeline_llm_node_is_finished_automatically():
    client = Mock()
    client.invoke.return_value = AIMessage(
        content='partial answer missing ending',
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
                content='first part second part',
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


def test_anchored_merge_is_exact_bounded_and_handles_a_midword_cut():
    existing = f"{'prefix ' * 40}the town was une"
    anchor = LLMNode._continuation_anchor(existing)

    assert len(anchor) == 160
    assert LLMNode._merge_anchored_continuation(
        existing,
        f'{anchor}asy throughout the night.',
        anchor,
    ) == f'{existing}asy throughout the night.'
    assert LLMNode._merge_anchored_continuation(
        existing,
        'I can continue if you paste the last paragraph.',
        anchor,
    ) is None


def test_anchored_merge_accepts_a_strong_exact_suffix_overlap():
    existing = 'Keep the hot path small.\n\n## 13) Make'
    anchor = LLMNode._continuation_anchor(existing)

    assert LLMNode._merge_anchored_continuation(
        existing,
        '13) Make observability first-class',
        anchor,
    ) == 'Keep the hot path small.\n\n## 13) Make observability first-class'


def test_anchored_merge_accepts_an_exact_repeated_partial_line():
    existing = 'Keep messages compact:\n- keep'
    anchor = LLMNode._continuation_anchor(existing)

    assert LLMNode._merge_anchored_continuation(
        existing,
        '- keep payloads small',
        anchor,
    ) == 'Keep messages compact:\n- keep payloads small'


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


def test_nested_continuation_exhaustion_raises_with_accumulated_partial_output():
    client = Mock()
    client.invoke.side_effect = [
        AIMessage(
            content='partial' + ''.join(
                f' part-{part}' for part in range(1, index + 1)
            ),
            response_metadata={'finish_reason': 'length'},
        )
        for index in range(1, 5)
    ]
    node = LLMNode(client=client)
    first = AIMessage(
        content='partial',
        response_metadata={'finish_reason': 'length'},
    )

    with pytest.raises(OutputContinuationExhausted) as raised:
        node._continue_nested_output(
            messages=[HumanMessage(content='Write the answer')],
            completion=first,
            config=_nested_config(),
        )

    assert raised.value.attempts == 4
    assert raised.value.failure_reason == 'attempt_limit'
    assert raised.value.stop_reason == 'length'
    assert raised.value.partial_output == 'partial part-1 part-2 part-3 part-4'
    assert client.invoke.call_count == 4


def test_nested_continuation_stops_when_the_model_makes_no_progress():
    client = Mock()
    client.invoke.return_value = AIMessage(
        content='same partial output',
        response_metadata={'finish_reason': 'length'},
    )
    node = LLMNode(client=client)
    first = AIMessage(
        content='same partial output',
        response_metadata={'finish_reason': 'length'},
    )

    with pytest.raises(OutputContinuationExhausted) as raised:
        node._continue_nested_output(
            messages=[HumanMessage(content='Write the answer')],
            completion=first,
            config=_nested_config(),
        )

    assert raised.value.attempts == 1
    assert raised.value.failure_reason == 'no_progress'
    assert raised.value.partial_output == 'same partial output'
    client.invoke.assert_called_once()


def test_nested_continuation_wraps_provider_failure_without_losing_partial_output():
    client = Mock()
    client.invoke.side_effect = RuntimeError('provider unavailable')
    node = LLMNode(client=client)
    first = AIMessage(
        content='partial answer',
        response_metadata={'finish_reason': 'length'},
    )

    with pytest.raises(OutputContinuationExhausted) as raised:
        node._continue_nested_output(
            messages=[HumanMessage(content='Write the answer')],
            completion=first,
            config=_nested_config(),
        )

    assert raised.value.failure_reason == 'provider_error'
    assert raised.value.partial_output == 'partial answer'
    assert isinstance(raised.value.__cause__, RuntimeError)


class _ExhaustedPipelineLLM:
    def __init__(self):
        self.calls = []
        self.responses = [
            AIMessage(content='first', response_metadata={'finish_reason': 'length'}),
            *[
                AIMessage(
                    content='first' + ''.join(
                        f' part-{part}' for part in range(1, index + 1)
                    ),
                    response_metadata={'finish_reason': 'length'},
                )
                for index in range(1, 5)
            ],
        ]

    def invoke(self, messages, config=None):
        self.calls.append(list(messages))
        return self.responses.pop(0)


def test_exhausted_pipeline_output_never_reaches_the_downstream_node():
    client = _ExhaustedPipelineLLM()
    pipeline = create_graph(
        client=client,
        yaml_schema='''
name: exhausted-continuation
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

    with pytest.raises(OutputContinuationExhausted):
        pipeline.invoke(
            {'input': 'start'},
            config={'configurable': {'thread_id': 'exhausted-pipeline'}},
        )

    assert len(client.calls) == 5
