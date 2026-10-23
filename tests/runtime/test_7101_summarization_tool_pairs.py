"""#7101: summarization never leaves a tool result without its tool call."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from elitea_sdk.runtime.middleware.summarization import SummarizationMiddleware
from elitea_sdk.runtime.tools.llm import LLMNode


def _call(call_id):
    return AIMessage(content='', tool_calls=[{'id': call_id, 'name': 'lookup', 'args': {}}])


def _middleware(trigger_tokens):
    model = GenericFakeChatModel(messages=iter([AIMessage(content='summary')]))
    middleware = SummarizationMiddleware(model=model, trigger=('tokens', trigger_tokens), keep=('messages', 2))
    middleware.accountant.count = lambda messages: 100 * len(messages)
    return middleware


def test_cutoff_landing_on_a_tool_result_moves_back_to_its_call():
    messages = [HumanMessage(content='a'), HumanMessage(content='b'), _call('call-1'),
                ToolMessage(content='ok', tool_call_id='call-1'), HumanMessage(content='c'), AIMessage(content='d')]
    middleware = _middleware(trigger_tokens=300)

    cutoff = middleware._determine_cutoff_index(messages)

    assert cutoff == 2
    assert middleware._last_fitting_count == 2


def test_cutoff_not_on_a_tool_result_is_unchanged():
    messages = [HumanMessage(content='a'), _call('call-1'), ToolMessage(content='ok', tool_call_id='call-1'),
                HumanMessage(content='b'), HumanMessage(content='c'), AIMessage(content='d')]
    middleware = _middleware(trigger_tokens=300)

    assert middleware._determine_cutoff_index(messages) == 3
    assert middleware._last_fitting_count == 1


def test_a_leading_orphaned_tool_result_is_removed():
    history = [ToolMessage(content='ok', tool_call_id='call-gone'), HumanMessage(content='go'),
               _call('call-1'), ToolMessage(content='ok', tool_call_id='call-1')]

    assert LLMNode._filter_orphaned_tool_calls(history) == history[1:]


def test_a_tool_result_for_another_call_is_removed():
    history = [HumanMessage(content='go'), _call('call-1'), ToolMessage(content='ok', tool_call_id='call-1'),
               ToolMessage(content='stale', tool_call_id='call-other')]

    assert LLMNode._filter_orphaned_tool_calls(history) == history[:3]


def test_a_tool_result_after_a_plain_message_is_removed():
    history = [HumanMessage(content='go'), AIMessage(content='hi'), ToolMessage(content='ok', tool_call_id='call-1')]

    assert LLMNode._filter_orphaned_tool_calls(history) == history[:2]
