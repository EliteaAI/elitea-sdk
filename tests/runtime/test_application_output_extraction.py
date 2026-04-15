from langchain_core.messages import AIMessage, HumanMessage

from elitea_sdk.runtime.langchain.constants import ELITEA_RS, PRINTER_NODE_RS
from elitea_sdk.runtime.tools.application import extract_application_response_output


def test_extract_application_response_output_prefers_langgraph_style_output_fields():
    assert extract_application_response_output({'output': 'final-output'}) == 'final-output'
    assert extract_application_response_output({ELITEA_RS: 'elitea-output'}) == 'elitea-output'
    assert extract_application_response_output({PRINTER_NODE_RS: 'printer-output'}) == 'printer-output'


def test_extract_application_response_output_falls_back_to_last_non_human_message():
    response = {
        'messages': [
            HumanMessage(content='delegate this task'),
            AIMessage(content=[
                {'type': 'tool_use', 'name': 'child_agent', 'id': 'tool-1'},
                {'type': 'text', 'text': 'child finished successfully'},
            ]),
        ]
    }

    assert extract_application_response_output(response) == 'child finished successfully'


def test_extract_application_response_output_returns_empty_string_for_tool_only_payloads():
    response = {
        'messages': [
            HumanMessage(content='delegate this task'),
            AIMessage(content=[
                {'type': 'tool_use', 'name': 'child_agent', 'id': 'tool-1'},
                {'type': 'tool_result', 'tool_use_id': 'tool-1', 'content': []},
            ]),
        ]
    }

    assert extract_application_response_output(response) == ''


def test_extract_application_response_output_preserves_multiblock_text_boundaries():
    response = {
        'output': [
            {'type': 'text', 'text': 'First paragraph.'},
            {'type': 'thinking', 'thinking': 'hidden chain of thought'},
            {'type': 'text', 'text': 'Second paragraph.'},
        ]
    }

    assert extract_application_response_output(response) == 'First paragraph.\n\nSecond paragraph.'