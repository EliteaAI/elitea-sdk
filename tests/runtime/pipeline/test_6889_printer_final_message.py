"""A finished run that ends on the printer's reply shows the printed value only (#6889).

The printer pauses the run with its value plus a resume hint ("How to proceed? ...").
When the user resumes and nothing after the printer answers, the run's output is the
printer's reply from the paused turn; the hint must not be part of it.
"""

import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.langraph_agent import PrinterNode, create_graph


@tool
def echo(text: str) -> str:
    """Return the text."""
    return 'ECHO:' + text


echo.metadata = {'toolkit_name': 'kit', 'toolkit_type': 'kit'}

HINT = PrinterNode.DEFAULT_FINAL_MSG


def toolkit(node_id, text, transition, output=None):
    return {'id': node_id, 'type': 'toolkit', 'toolkit_name': 'kit', 'tool': 'echo',
            'input_mapping': {'text': {'type': 'fixed', 'value': text}}, 'output': output or [],
            'transition': transition}


def printer(transition, **extra):
    return {'id': 'p', 'type': 'printer', 'transition': transition,
            'input_mapping': {'printer': {'type': 'fstring', 'value': 'Value: {x}'}}, **extra}


STATE_MODIFIER = {'id': 'sm', 'type': 'state_modifier', 'template': 'z', 'variables_to_clean': [],
                  'output': [], 'transition': 'END'}


def run_two_turns(nodes):
    doc = {'entry_point': 't1', 'state': {'messages': {'type': 'list'}, 'x': {'type': 'str'}}, 'nodes': nodes}
    graph = create_graph(client=None, yaml_schema=yaml.safe_dump(doc), tools=[echo], memory=MemorySaver())
    config = {'configurable': {'thread_id': 't'}}
    paused = graph.invoke({'messages': [HumanMessage('go')]}, config=config)
    # The platform resends the history, including the printer's reply, with the new message.
    history = [HumanMessage('go'), AIMessage(paused['output']), HumanMessage('continue')]
    return paused, graph.invoke({'messages': history}, config=config)


@pytest.mark.parametrize('after_printer', [
    pytest.param([], id='printer -> END'),
    pytest.param([STATE_MODIFIER], id='printer -> node without output -> END'),
])
def test_finished_run_ending_on_the_printer_shows_the_value_only(after_printer):
    transition = after_printer[0]['id'] if after_printer else 'END'
    paused, finished = run_two_turns([toolkit('t1', 'one', 'p', ['x']), printer(transition), *after_printer])

    assert paused['execution_finished'] is False
    assert paused['output'] == f'Value: ECHO:one\n\n-----\n*{HINT}*'
    assert finished['execution_finished'] is True
    assert finished['output'] == 'Value: ECHO:one'


def test_custom_final_message_is_stripped_too():
    paused, finished = run_two_turns([toolkit('t1', 'one', 'p', ['x']), printer('END', final_message='Next?')])

    assert paused['output'].endswith('*Next?*')
    assert finished['output'] == 'Value: ECHO:one'


def test_a_node_answering_after_the_printer_is_the_output():
    _, finished = run_two_turns([toolkit('t1', 'one', 'p', ['x']), printer('t2'),
                                 toolkit('t2', 'two', 'END', ['messages'])])

    assert finished['execution_finished'] is True
    assert finished['output'] == 'ECHO:two'
