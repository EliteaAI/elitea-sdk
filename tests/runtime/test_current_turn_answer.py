"""An empty reply must not re-send the previous turn's answer (chat repeated its first answer on every turn)."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from elitea_sdk.runtime.langchain.langraph_agent import extract_current_turn_answer


def test_empty_reply_does_not_return_the_previous_turn_answer():
    messages = [
        HumanMessage(content='Create an agent'),
        AIMessage(content='I created the agent.'),
        HumanMessage(content='do it'),
        AIMessage(content=[]),
    ]
    assert extract_current_turn_answer(messages) is None


def test_empty_synthesis_after_tools_uses_the_turn_text():
    messages = [
        HumanMessage(content='Find it'),
        AIMessage(content='Searching now.'),
        ToolMessage(content='{"ok": true}', tool_call_id='t1'),
        AIMessage(content=[]),
    ]
    assert extract_current_turn_answer(messages) == '{"ok": true}'


def test_returns_the_latest_reply_of_the_turn():
    messages = [
        HumanMessage(content='first'),
        AIMessage(content='old answer'),
        HumanMessage(content='second'),
        AIMessage(content=[{'type': 'text', 'text': 'new answer'}]),
    ]
    assert extract_current_turn_answer(messages) == 'new answer'


def test_no_messages():
    assert extract_current_turn_answer([]) is None
    assert extract_current_turn_answer(None) is None
