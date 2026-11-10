"""#7460: pylon_main replays recent tool calls as {role: assistant, tool_calls} + {role: tool} dicts.

convert_dict_to_message used to rebuild only role+content, so the stubs reached the provider
as an empty assistant turn and a stray user message — the model never saw that a tool ran.
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from elitea_sdk.runtime.langchain.langraph_agent import convert_dict_to_message


def test_assistant_stub_keeps_tool_calls():
    msg = convert_dict_to_message({'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'hist_1', 'name': 'create_file', 'args': {'filename': 'a.md'}, 'type': 'tool_call'}]})
    assert isinstance(msg, AIMessage)
    assert msg.tool_calls[0]['id'] == 'hist_1'
    assert msg.tool_calls[0]['args'] == {'filename': 'a.md'}


def test_tool_stub_becomes_tool_message():
    msg = convert_dict_to_message({'role': 'tool', 'content': 'ok', 'tool_call_id': 'hist_1'})
    assert isinstance(msg, ToolMessage)
    assert msg.tool_call_id == 'hist_1' and msg.content == 'ok'


def test_plain_assistant_text_unchanged():
    msg = convert_dict_to_message({'role': 'assistant', 'content': 'hi'})
    assert isinstance(msg, AIMessage) and msg.content == 'hi' and msg.tool_calls == []


def test_tool_role_without_id_stays_human_fallback():
    # A ToolMessage without its call id would be rejected by every provider; keep the old fallback
    assert isinstance(convert_dict_to_message({'role': 'tool', 'content': 'x'}), HumanMessage)
