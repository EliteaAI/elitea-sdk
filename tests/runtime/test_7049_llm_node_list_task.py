"""A "list" pipeline variable mapped to an LLM node's task must reach the model as text.

LLMNode passed any list task straight into ``HumanMessage.content``, treating it as
content blocks. A pipeline list like ``["a.pdf", "b.pdf"]`` is data, not blocks:
LiteLLM's Bedrock converter failed on its items and the gateway answered
``BedrockException - {}`` with a 400 (#7049).
"""
import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from elitea_sdk.runtime.tools.llm import LLMNode


class _RecordingChat(GenericFakeChatModel):
    seen: list = []

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _sent_task(task):
    llm = _RecordingChat(messages=iter([AIMessage(content='done')]), seen=[])
    node = LLMNode(client=llm, name='llm', available_tools=[],
                   input_mapping={'system': {'type': 'fixed', 'value': 'You help.'},
                                  'task': {'type': 'variable', 'value': 'items'}},
                   input_variables=['items'], output_variables=['messages'], return_type='dict')
    node.invoke({'items': task, 'messages': []}, {'configurable': {}})
    sent = llm.seen[-1][-1]
    assert isinstance(sent, HumanMessage)
    return sent.content


@pytest.mark.parametrize('task', [
    # The state default of a "list" variable: an empty user message is a 400 on Anthropic.
    [],
    ['a.pdf', 'b.pdf'],
    [{'name': 'a.pdf', 'filepath': '/x/a.pdf'}],
    [1, 2, 3],
])
def test_list_variable_task_is_sent_as_json_text(task):
    assert _sent_task(task) == json.dumps(task, ensure_ascii=False)


def test_content_block_task_is_kept():
    blocks = [{'type': 'text', 'text': 'Describe it'},
              {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAA'}}]
    assert _sent_task(blocks) == blocks


@pytest.mark.parametrize('task,expected', [
    (None, ''),
    ('plain', 'plain'),
    ([], '[]'),
    (42, '42'),
    ({'k': 'v'}, "{'k': 'v'}"),
    (['ünï'], '["ünï"]'),
    # Bare strings beside typed blocks are text blocks, not a data list.
    (['Look:', {'type': 'image_url', 'image_url': {'url': 'u'}}],
     [{'type': 'text', 'text': 'Look:'}, {'type': 'image_url', 'image_url': {'url': 'u'}}]),
    # One untyped dict among blocks makes the whole list data.
    ([{'type': 'text', 'text': 't'}, {'name': 'x'}],
     '[{"type": "text", "text": "t"}, {"name": "x"}]'),
])
def test_task_message_content(task, expected):
    assert LLMNode._task_message_content(task) == expected
