"""validate_pipeline agrees with create_graph (#6388 / #7223).

Every spec here is built with the real create_graph. A spec the runtime refuses must be
rejected by the validator; a spec the validator accepts must build. The validator may be
stricter than the build only for shapes that fail or silently misbehave once they run,
and each such case states why.
"""

import pytest
import yaml
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.langraph_agent import create_graph
from elitea_sdk.runtime.pipeline import validate_pipeline


@tool
def echo(text: str) -> str:
    """Return the text."""
    return 'ECHO:' + text


echo.metadata = {'toolkit_name': 'kit', 'toolkit_type': 'kit'}


def fixed(value):
    return {'type': 'fixed', 'value': value}


def build(doc):
    return create_graph(client=None, yaml_schema=yaml.safe_dump(doc), tools=[echo], memory=MemorySaver())


def builds(doc) -> bool:
    try:
        build(doc)
        return True
    except Exception:
        return False


TOOLKIT = {'id': 't', 'type': 'toolkit', 'toolkit_name': 'kit', 'tool': 'echo',
           'input_mapping': {'text': fixed('x')}, 'transition': 'END'}


def _routed_by(first, *more):
    """`first`, then toolkit nodes `a`, `b` and `c` that each echo their own letter and end."""
    return {'entry_point': first['id'], 'nodes': [first, *more] + [
        dict(TOOLKIT, id=n, input_mapping={'text': fixed(n.upper())}) for n in ('a', 'b', 'c')]}


def _condition_node(**condition):
    return {k: v for k, v in TOOLKIT.items() if k != 'transition'} | {'condition': condition}

BOTH_ACCEPT = {
    'llm -> router -> llm': {
        'entry_point': 'a', 'state': {'messages': {'type': 'list'}, 'intent': {'type': 'str'}},
        'nodes': [
            {'id': 'a', 'type': 'llm', 'input_mapping': {'task': fixed('x')}, 'output': ['intent'],
             'structured_output': True, 'transition': 'r'},
            {'id': 'r', 'type': 'router', 'input': ['intent'], 'condition': 'b', 'routes': ['b'],
             'default_output': 'END'},
            {'id': 'b', 'type': 'llm', 'input_mapping': {'task': fixed('y')}, 'output': ['messages'],
             'transition': 'END'},
        ]},
    'toolkit with tool and toolkit_name': {'entry_point': 't', 'nodes': [TOOLKIT]},
    'code node returning a dict': {
        'entry_point': 'c', 'state': {'x': {'type': 'int'}, 'messages': {'type': 'list'}},
        'nodes': [{'id': 'c', 'type': 'code', 'code': fixed("{'x': 1}"), 'input': ['x'], 'output': ['x'],
                   'structured_output': True, 'transition': 'END'}]},
    'hitl with approve and reject': {
        'entry_point': 't', 'nodes': [
            dict(TOOLKIT, transition='h'),
            {'id': 'h', 'type': 'hitl', 'user_message': fixed('ok?'), 'routes': {'approve': 'END', 'reject': 'END'}},
        ]},
    # The shape the UI's (hidden) Condition node saves: no transition, so the condition routes.
    'toolkit routed by an inline condition': {
        'entry_point': 't', 'nodes': [
            {k: v for k, v in TOOLKIT.items() if k != 'transition'} | {'condition': {
                'condition_input': ['messages'], 'condition_definition': 'b',
                'conditional_outputs': ['a', 'b'], 'default_output': 'END'}},
            dict(TOOLKIT, id='a', input_mapping={'text': fixed('A')}),
            dict(TOOLKIT, id='b', input_mapping={'text': fixed('B')}),
        ]},
    'router as the entry point': {
        'entry_point': 'r', 'nodes': [
            {'id': 'r', 'type': 'router', 'input': ['input'], 'condition': 'b', 'routes': ['b'],
             'default_output': 'END'},
            dict(TOOLKIT, id='b', input_mapping={'text': fixed('B')}),
        ]},
    'router reading last_tool_outcome': {
        'entry_point': 't', 'nodes': [
            dict(TOOLKIT, transition='r'),
            {'id': 'r', 'type': 'router', 'input': ['last_tool_outcome'], 'condition': 'b', 'routes': ['b'],
             'default_output': 'END'},
            dict(TOOLKIT, id='b', input_mapping={'text': fixed('B')}),
        ]},
}

BOTH_REJECT = {
    'toolkit with name but no tool': {'entry_point': 't', 'nodes': [
        {'id': 't', 'type': 'toolkit', 'name': 'echo', 'toolkit_name': 'kit', 'transition': 'END'}]},
    'no entry_point': {'nodes': [TOOLKIT]},
    'custom node': {'entry_point': 'c', 'nodes': [{'id': 'c', 'type': 'custom', 'transition': 'END'}]},
    'agent without task': {'entry_point': 'g', 'nodes': [
        {'id': 'g', 'type': 'agent', 'tool': 'echo', 'input_mapping': {'chat_history': fixed([])},
         'transition': 'END'}]},
    'printer without transition': {'entry_point': 'p', 'nodes': [
        {'id': 'p', 'type': 'printer', 'input_mapping': {'printer': fixed('hi')}}]},
    'input as a list of objects': {'entry_point': 'a', 'nodes': [
        {'id': 'a', 'type': 'llm', 'input': [{'name': 'input'}], 'input_mapping': {'task': fixed('x')},
         'transition': 'END'}]},
    # Malformed inline conditions on a node that routes by them (review of #890, round 2)
    **{f'inline condition {label}': _routed_by({'id': 't', **{k: v for k, v in TOOLKIT.items()
                                                               if k not in ('id', 'transition')},
                                                 'condition': condition})
       for label, condition in [
           ('as a string', 'b'),
           ('as a list', ['a', 'b']),
           ('with dict outputs', {'condition_definition': 'b', 'conditional_outputs': [{'node': 'a'}, {'node': 'b'}]}),
           ('with null outputs', {'condition_definition': 'b', 'conditional_outputs': None}),
       ]},
}

# Builds, but is wrong once it runs; the validator rejects it on purpose.
STRICTER_THAN_THE_BUILD = {
    'transition to an unknown node (the run silently stops there)': {'entry_point': 't', 'nodes': [
        dict(TOOLKIT, transition='secnd'),
        {'id': 'second', 'type': 'toolkit', 'toolkit_name': 'kit', 'tool': 'echo',
         'input_mapping': {'text': fixed('SECOND')}, 'transition': 'END'}]},
    'code block without type (KeyError at run time)': {
        'entry_point': 'c', 'state': {'x': {'type': 'int'}, 'messages': {'type': 'list'}},
        'nodes': [{'id': 'c', 'type': 'code', 'code': {'value': "{'x': 1}"}, 'output': ['x'],
                   'structured_output': True, 'transition': 'END'}]},
    'llm node with no system or task (sends only the history)': {'entry_point': 'a', 'nodes': [
        {'id': 'a', 'type': 'llm', 'transition': 'END'}]},
    'removed loop node': {'entry_point': 'l', 'nodes': [
        {'id': 'l', 'type': 'loop', 'tool': 'echo', 'transition': 'END'}]},
    # Review of #890, round 2: each builds, then fails or misroutes when run.
    'state_modifier writing tool_outcomes (the dict reducer fails the run)': {
        'entry_point': 's', 'state': {'messages': {'type': 'list'}}, 'nodes': [
            {'id': 's', 'type': 'state_modifier', 'template': 'x', 'output': ['tool_outcomes'], 'transition': 'END'}]},
    'hitl editing tool_outcomes (the resumed edit fails the run)': {
        'entry_point': 't', 'nodes': [
            dict(TOOLKIT, transition='h'),
            {'id': 'h', 'type': 'hitl', 'user_message': fixed('ok?'), 'edit_state_key': 'tool_outcomes',
             'routes': {'approve': 'END', 'edit': 't2'}},
            dict(TOOLKIT, id='t2')]},
    'inline condition reading a bare-string condition_input (reads one letter at a time)': _routed_by(
        _condition_node(condition_input='input', condition_definition='b', conditional_outputs=['a', 'b'],
                        default_output='END')),
}


@pytest.mark.parametrize('name', BOTH_ACCEPT)
def test_accepted_specs_build(name):
    doc = BOTH_ACCEPT[name]
    assert validate_pipeline(doc)['ok'], validate_pipeline(doc)['errors']
    assert builds(doc)


@pytest.mark.parametrize('name', BOTH_REJECT)
def test_specs_the_runtime_refuses_are_rejected(name):
    doc = BOTH_REJECT[name]
    assert not builds(doc)
    assert not validate_pipeline(doc)['ok']


@pytest.mark.parametrize('name', STRICTER_THAN_THE_BUILD)
def test_stricter_cases_build_but_are_rejected(name):
    doc = STRICTER_THAN_THE_BUILD[name]
    assert builds(doc), 'the runtime now refuses this spec; move it to BOTH_REJECT'
    assert not validate_pipeline(doc)['ok']


def test_unknown_transition_really_skips_the_rest_of_the_pipeline():
    graph = build(STRICTER_THAN_THE_BUILD['transition to an unknown node (the run silently stops there)'])
    result = graph.invoke({'input': 'hi', 'messages': []}, config={'configurable': {'thread_id': 'r2'}})
    text = str(result.get('messages'))
    assert 'ECHO:x' in text and 'SECOND' not in text


@pytest.mark.parametrize('name', ['toolkit routed by an inline condition', 'router as the entry point',
                                  'router reading last_tool_outcome'])
def test_accepted_routing_shapes_really_route(name):
    graph = build(BOTH_ACCEPT[name])
    result = graph.invoke({'input': 'hi', 'messages': []}, config={'configurable': {'thread_id': name}})
    text = str(result.get('messages'))
    assert 'ECHO:B' in text and 'ECHO:A' not in text


def test_writing_tool_outcomes_really_fails_the_run():
    graph = build(STRICTER_THAN_THE_BUILD['state_modifier writing tool_outcomes (the dict reducer fails the run)'])
    with pytest.raises(ValueError, match='dictionary update sequence'):
        graph.invoke({'input': 'hi', 'messages': []}, config={'configurable': {'thread_id': 'to'}})


def test_editing_tool_outcomes_in_hitl_really_fails_the_resume():
    graph = build(STRICTER_THAN_THE_BUILD['hitl editing tool_outcomes (the resumed edit fails the run)'])
    config = {'configurable': {'thread_id': 'hitl-to'}}
    graph.invoke({'input': 'hi', 'messages': []}, config=config)
    with pytest.raises(ValueError, match='dictionary update sequence'):
        graph.invoke({'hitl_resume': True, 'hitl_action': 'edit', 'hitl_value': 'text'}, config=config)


# An inline condition with no conditional_outputs routes to whatever the template renders and never
# uses default_output (ConditionalEdge.invoke).
EMPTY_OUTPUTS_WORKS = _routed_by(_condition_node(condition_definition='b', conditional_outputs=[], default_output=''))
EMPTY_OUTPUTS_TYPO = _routed_by(_condition_node(condition_definition='zzz', conditional_outputs=[], default_output='c'))


def test_empty_conditional_outputs_route_by_the_template_and_skip_default_output():
    for doc, expected in ((EMPTY_OUTPUTS_WORKS, 'ECHO:B'), (EMPTY_OUTPUTS_TYPO, None)):
        result = build(doc).invoke({'input': 'hi', 'messages': []}, config={'configurable': {'thread_id': str(id(doc))}})
        text = str(result.get('messages'))
        assert 'ECHO:C' not in text  # default_output is never used
        assert expected is None or expected in text


def test_empty_conditional_outputs_warn_without_blocking():
    # The validator cannot evaluate the template, so both get the same warning and no error.
    for doc in (EMPTY_OUTPUTS_WORKS, EMPTY_OUTPUTS_TYPO):
        result = validate_pipeline(doc)
        assert result['ok'], result['errors']
        assert any(f['field'] == 'condition.conditional_outputs' for f in result['warnings'])


def test_code_block_without_type_fails_when_run():
    graph = build(STRICTER_THAN_THE_BUILD['code block without type (KeyError at run time)'])
    with pytest.raises(KeyError, match='type'):
        graph.invoke({'input': 'hi', 'messages': []}, config={'configurable': {'thread_id': 'r5'}})
