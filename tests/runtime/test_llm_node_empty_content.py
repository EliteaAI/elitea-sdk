"""An empty content-block list from the model must not become a literal "[]" answer.

Sonnet can reply ``AIMessage(content=[])`` with no tool calls. LLMNode's raw-content
fallback used ``str(content)``, so the agent's final message read "[]" and eval scored
it ``ok`` instead of ``empty``.
"""
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from elitea_sdk.runtime.tools.llm import LLMNode


def _invoke(reply_content):
    llm = GenericFakeChatModel(messages=iter([AIMessage(content=reply_content)]))
    node = LLMNode(client=llm, name='agent', available_tools=[],
                   input_mapping={'messages': {'type': 'variable', 'value': 'messages'}},
                   input_variables=['messages'], output_variables=['messages'], return_type='dict')
    return node.invoke({'messages': [HumanMessage(content='hi')]}, {'configurable': {}})


def test_empty_content_list_reply_is_blank():
    result = _invoke([])
    assert result['messages'][-1].content == ''


def test_text_reply_unchanged():
    result = _invoke('the answer')
    assert result['messages'][-1].content == 'the answer'


@pytest.mark.parametrize('content,expected', [
    ([], ''),
    (None, ''),
    ('', ''),
    ('text', 'text'),
    ([{'type': 'image', 'url': 'x'}], "[{'type': 'image', 'url': 'x'}]"),
])
def test_raw_content_fallback(content, expected):
    assert LLMNode._raw_content_fallback(content) == expected
