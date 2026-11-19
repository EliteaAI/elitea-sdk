"""validate_pipeline (#6388 / #7223): each rule fires on the broken shape and stays quiet on the
working one. The must-pass cases matter as much as the must-fail ones: a false positive
blocks a pipeline that runs."""

import ast
import copy
from pathlib import Path

import pytest
import yaml

from elitea_sdk.runtime.pipeline import validator
from elitea_sdk.runtime.pipeline.validator import ValidationContext, validate_pipeline

BASE = {
    'entry_point': 'classify',
    'state': {
        'input': {'type': 'str'},
        'messages': {'type': 'list'},
        'intent': {'type': 'str'},
    },
    'nodes': [
        {
            'id': 'classify', 'type': 'llm', 'input': ['input'], 'output': ['intent'],
            'structured_output': True,
            'input_mapping': {
                'system': {'type': 'fixed', 'value': 'Return JSON {"intent": "bug|other"}.'},
                'task': {'type': 'fstring', 'value': 'Message: {input}'},
            },
            'transition': 'route',
        },
        {
            'id': 'route', 'type': 'router', 'input': ['intent'],
            'condition': "{% if intent == 'bug' %}answer{% endif %}",
            'routes': ['answer'], 'default_output': 'END',
        },
        {
            'id': 'answer', 'type': 'llm', 'input': ['input', 'intent', 'messages'], 'output': ['messages'],
            'input_mapping': {'task': {'type': 'fstring', 'value': 'Intent {intent}: {input}'}},
            'transition': 'END',
        },
    ],
}


def spec(**changes):
    doc = copy.deepcopy(BASE)
    doc.update(changes)
    return doc


def with_nodes(*nodes, entry=None, state=None):
    doc = {'entry_point': entry or nodes[0]['id'], 'nodes': list(nodes)}
    if state is not None:
        doc['state'] = state
    return doc


def rules(result, severity='errors'):
    return [f['rule'] for f in result[severity]]


def assert_clean(result):
    assert result['ok'], result['errors']
    assert not result['warnings'], result['warnings']


# ---- the result shape and the happy path ---------------------------------------------------------

def test_a_working_pipeline_is_clean():
    assert_clean(validate_pipeline(BASE))


def test_yaml_text_and_json_text_validate_the_same():
    import json
    assert_clean(validate_pipeline(yaml.safe_dump(BASE)))
    assert_clean(validate_pipeline(json.dumps(BASE)))


def test_every_finding_names_rule_message_and_suggestion():
    result = validate_pipeline(with_nodes({'id': 'a', 'type': 'toolkit', 'transition': 'END'}))
    finding = result['errors'][0]
    assert set(finding) == {'rule', 'node_id', 'field', 'message', 'suggestion'}
    assert finding['suggestion']


def test_validator_never_raises(monkeypatch):
    monkeypatch.setattr(validator, '_r29_messages_discipline', lambda *a: 1 / 0)
    result = validate_pipeline(BASE)
    assert result['ok']
    assert rules(result, 'warnings') == ['R00']


# ---- R20 parsing and shape -----------------------------------------------------------------------------

@pytest.mark.parametrize('bad', ['nodes: [', '- just a list', {'nodes': []}, {'entry_point': 'a'},
                                 {'nodes': 'a'}, {'nodes': ['a']}, {'nodes': [{'id': 'a'}], 'state': []}])
def test_r20_shape(bad):
    assert rules(validate_pipeline(bad)) == ['R20']


# ---- R21, R22 ids and entry point -------------------------------------------------------------------------

def test_r21_duplicate_and_missing_ids():
    doc = spec()
    doc['nodes'].append(copy.deepcopy(doc['nodes'][2]))
    doc['nodes'].append({'type': 'llm', 'input_mapping': {'task': {'type': 'fixed', 'value': 'x'}}})
    assert rules(validate_pipeline(doc)).count('R21') == 2


def test_r22_missing_entry_point_suggests_the_legacy_flag_node():
    doc = spec()
    del doc['entry_point']
    doc['nodes'][2]['entrypoint'] = True
    finding = validate_pipeline(doc)['errors'][0]
    assert finding['rule'] == 'R22' and 'entry_point: answer' in finding['suggestion']


@pytest.mark.parametrize('entry', ['END', 'missing'])
def test_r22_bad_entry_point(entry):
    assert 'R22' in rules(validate_pipeline(spec(entry_point=entry)))


def test_r22_routing_node_as_entry_point_is_only_a_warning():
    # It builds and runs; the UI lets a user pick a router as the entry.
    result = validate_pipeline(spec(entry_point='route'))
    assert result['ok'] and 'R22' in rules(result, 'warnings')


# ---- R23 status from the registry --------------------------------------------------------------------------

@pytest.mark.parametrize('node_type', ['loop', 'loop_from_tool', 'function', 'tool', 'pipeline', 'subgraph', 'custom'])
def test_r23_removed_and_ui_only_types_are_errors(node_type):
    result = validate_pipeline(with_nodes({'id': 'a', 'type': node_type, 'tool': 'x', 'transition': 'END'}))
    assert 'R23' in rules(result)
    assert 'Use instead' in result['errors'][0]['suggestion']


def test_r23_deprecated_printer_is_a_warning():
    result = validate_pipeline(with_nodes(
        {'id': 'a', 'type': 'printer', 'input_mapping': {'printer': {'type': 'fixed', 'value': 'hi'}},
         'transition': 'END'}))
    assert result['ok']
    assert rules(result, 'warnings') == ['R23']


def test_r23_status_change_in_the_registry_changes_the_finding(monkeypatch):
    from elitea_sdk.runtime.pipeline import registry
    data = copy.deepcopy(registry._load())
    data['nodes']['router']['status'] = 'deprecated'
    data['nodes']['router']['replacement'] = 'a decision node'
    monkeypatch.setattr(registry, '_load', lambda: data)
    result = validate_pipeline(BASE)
    assert result['ok'] and [f['node_id'] for f in result['warnings']] == ['route']


@pytest.mark.parametrize('node', [{'id': 'a', 'transition': 'END'}, {'id': 'a', 'type': 'banana', 'transition': 'END'}])
def test_r23_missing_or_unknown_type(node):
    assert rules(validate_pipeline(with_nodes(node))) == ['R23']


def test_r23_type_end_node_is_rejected_with_the_fix():
    doc = spec()
    doc['nodes'][2]['transition'] = 'finish'
    doc['nodes'].append({'id': 'finish', 'type': 'END'})
    finding = next(f for f in validate_pipeline(doc)['errors'] if f['rule'] == 'R23')
    assert 'transition: END' in finding['suggestion']


CONDITION = {'condition_input': ['intent'], 'condition_definition': "{% if intent == 'bug' %}a{% else %}b{% endif %}",
             'conditional_outputs': ['a', 'b'], 'default_output': 'END'}


def _condition_only(outputs=('a', 'b')):
    """A toolkit node routed only by an inline condition, the shape the UI's Condition node saves."""
    condition = dict(CONDITION, conditional_outputs=list(outputs))
    task = {'type': 'fixed', 'value': 'x'}
    return with_nodes(
        {'id': 't', 'type': 'toolkit', 'tool': 'get_issue', 'toolkit_name': 'jira', 'condition': condition},
        {'id': 'a', 'type': 'llm', 'input_mapping': {'task': task}, 'transition': 'END'},
        {'id': 'b', 'type': 'llm', 'input_mapping': {'task': task}, 'transition': 'END'},
        state={'messages': {'type': 'list'}, 'intent': {'type': 'str'}})


def test_r23_inline_condition_that_routes_is_a_warning():
    result = validate_pipeline(_condition_only())
    assert result['ok'], result['errors']
    assert [f['rule'] for f in result['warnings']] == ['R23']  # and no false R25 for a or b


def test_r23_inline_condition_next_to_a_transition_is_an_error():
    doc = spec()
    doc['nodes'][2]['condition'] = {'condition_definition': '1 == 2', 'conditional_outputs': []}
    finding = next(f for f in validate_pipeline(doc)['errors'] if f['rule'] == 'R23')
    assert 'ignored' in finding['message']


def test_r23_inline_condition_on_a_hitl_node_is_an_error():
    doc = with_nodes(
        {'id': 'a', 'type': 'llm', 'input_mapping': {'task': {'type': 'fixed', 'value': 'x'}}, 'transition': 'h'},
        {'id': 'h', 'type': 'hitl', 'user_message': {'type': 'fixed', 'value': 'ok?'},
         'routes': {'approve': 'END'}, 'condition': dict(CONDITION, conditional_outputs=[])})
    assert 'R23' in rules(validate_pipeline(doc))


def test_r2_reports_a_mistyped_condition_target():
    result = validate_pipeline(_condition_only(outputs=('aa', 'b')))
    assert any(f['rule'] == 'R2' and f['field'] == 'condition.conditional_outputs[0]' for f in result['errors'])


def test_inline_decision_is_a_warning():
    doc = spec()
    doc['nodes'][2]['decision'] = {'nodes': ['classify'], 'description': 'loop back'}
    del doc['nodes'][2]['transition']
    result = validate_pipeline(doc)
    assert result['ok'] and 'R23' in rules(result, 'warnings')


# ---- R24 required fields (no hollow nodes) ------------------------------------------------------------------

@pytest.mark.parametrize('node,field', [
    ({'id': 'a', 'type': 'llm', 'transition': 'END'}, 'input_mapping.system'),
    ({'id': 'a', 'type': 'llm', 'input_mapping': {'system': {'type': 'fixed', 'value': '  '}}, 'transition': 'END'},
     'input_mapping.system'),
    ({'id': 'a', 'type': 'toolkit', 'transition': 'END'}, 'tool'),
    ({'id': 'a', 'type': 'mcp', 'transition': 'END'}, 'tool'),
    ({'id': 'a', 'type': 'agent', 'tool': 'Child', 'transition': 'END'}, 'input_mapping.task'),
    ({'id': 'a', 'type': 'code', 'code': {'type': 'fixed', 'value': ''}, 'transition': 'END'}, 'code.value'),
    ({'id': 'a', 'type': 'state_modifier', 'template': '{{ x }}', 'transition': 'END'}, 'output'),
])
def test_r24_hollow_nodes(node, field):
    findings = [f for f in validate_pipeline(with_nodes(node))['errors'] if f['rule'] == 'R24']
    assert [f['field'] for f in findings] == [field]


def test_r24_llm_needs_only_one_of_system_and_task():
    node = {'id': 'a', 'type': 'llm', 'input_mapping': {'task': {'type': 'fixed', 'value': 'Say hi'}},
            'transition': 'END'}
    assert 'R24' not in rules(validate_pipeline(with_nodes(node)))


@pytest.mark.parametrize('node_type', ['toolkit', 'mcp', 'agent'])
def test_r24_name_instead_of_tool_says_the_runtime_reads_tool(node_type):
    node = {'id': 'a', 'type': node_type, 'name': 'Jira', 'transition': 'END',
            'input_mapping': {'task': {'type': 'fixed', 'value': 'x'}}}
    finding = next(f for f in validate_pipeline(with_nodes(node))['errors'] if f['rule'] == 'R24')
    assert 'reads `tool`, not `name`' in finding['suggestion'] and 'tool: Jira' in finding['suggestion']


def test_r24_router_and_decision_and_hitl():
    nodes = [
        {'id': 'start', 'type': 'llm', 'input_mapping': {'task': {'type': 'fixed', 'value': 'x'}}, 'transition': 'r'},
        {'id': 'r', 'type': 'router', 'default_output': 'd'},
        {'id': 'd', 'type': 'decision', 'default_output': 'h'},
        {'id': 'h', 'type': 'hitl'},
    ]
    fields = sorted(f['field'] for f in validate_pipeline(with_nodes(*nodes))['errors'] if f['rule'] == 'R24')
    assert fields == ['condition', 'description', 'nodes', 'routes', 'routes', 'user_message']


# ---- R2 and R25 graph -------------------------------------------------------------------------------------

def test_r2_unknown_targets_in_every_routing_field():
    doc = spec()
    doc['nodes'][0]['transition'] = 'nowhere'
    doc['nodes'][1]['routes'] = ['answer', 'gone']
    doc['nodes'][1]['default_output'] = 'lost'
    fields = sorted(f['field'] for f in validate_pipeline(doc)['errors'] if f['rule'] == 'R2')
    assert fields == ['default_output', 'routes[1]', 'transition']


def test_r2_end_is_case_insensitive_and_not_a_node():
    doc = spec()
    doc['nodes'][2]['transition'] = 'end'
    assert validate_pipeline(doc)['ok']


def test_r25_no_path_to_end():
    doc = spec()
    doc['nodes'][2]['transition'] = 'classify'
    doc['nodes'][1]['default_output'] = 'answer'
    assert 'R25' in rules(validate_pipeline(doc))


def test_r25_orphan_node_is_a_warning():
    doc = spec()
    doc['nodes'].append({'id': 'lonely', 'type': 'llm', 'input_mapping': {'task': {'type': 'fixed', 'value': 'x'}},
                         'transition': 'END'})
    result = validate_pipeline(doc)
    assert result['ok'] and [f['node_id'] for f in result['warnings'] if f['rule'] == 'R25'] == ['lonely']


def test_r25_a_node_without_transition_ends_the_run():
    doc = spec()
    del doc['nodes'][2]['transition']
    assert validate_pipeline(doc)['ok']


def test_r25_loop_with_an_exit_is_fine():
    doc = with_nodes(
        {'id': 'work', 'type': 'code', 'input': ['i'], 'output': ['i'], 'structured_output': True,
         'code': {'type': 'fixed', 'value': "i = elitea_state.get('i', 0)\n{'i': i + 1}"}, 'transition': 'check'},
        {'id': 'check', 'type': 'router', 'input': ['i'], 'condition': "{% if i < 3 %}work{% endif %}",
         'routes': ['work'], 'default_output': 'END'},
        state={'i': {'type': 'int', 'value': 0}, 'messages': {'type': 'list'}},
    )
    assert_clean(validate_pipeline(doc))


# ---- R1, R4, R26 state ---------------------------------------------------------------------------------------

def test_r4_undeclared_state_variables():
    doc = spec()
    doc['nodes'][0]['output'] = ['intent', 'confidence']
    doc['nodes'][2]['input_mapping']['task'] = {'type': 'variable', 'value': 'draft'}
    fields = sorted(f['field'] for f in validate_pipeline(doc)['errors'] if f['rule'] == 'R4')
    assert fields == ['input_mapping.task', 'output']


def test_r4_runtime_channels_need_no_declaration():
    doc = spec()
    del doc['state']
    doc['nodes'][0]['output'] = ['messages']
    doc['nodes'][0]['structured_output'] = False
    doc['nodes'][1]['input'] = ['router_output']
    doc['nodes'][1]['condition'] = 'answer'
    doc['nodes'][2]['input'] = ['input', 'messages']
    doc['nodes'][2]['input_mapping']['task']['value'] = '{input}'
    assert validate_pipeline(doc)['ok']


def test_r4_tool_outcomes_need_no_declaration():
    # create_state always defines them (#6171); a router branching on the last tool call works.
    doc = with_nodes(
        {'id': 't', 'type': 'toolkit', 'tool': 'get_issue', 'toolkit_name': 'jira', 'transition': 'r'},
        {'id': 'r', 'type': 'router', 'input': ['last_tool_outcome', 'tool_outcomes'],
         'condition': "{% if last_tool_outcome.status == 'ok' %}done{% endif %}", 'routes': ['done'],
         'default_output': 'END'},
        {'id': 'done', 'type': 'llm', 'input_mapping': {'task': {'type': 'fixed', 'value': 'x'}}, 'transition': 'END'},
        state={'messages': {'type': 'list'}})
    assert validate_pipeline(doc)['ok']


def test_r4_writing_a_runtime_variable_is_a_warning():
    doc = spec()
    doc['nodes'][0]['output'] = ['intent', 'last_tool_outcome']
    result = validate_pipeline(doc)
    assert result['ok'] and any(f['rule'] == 'R4' and f['node_id'] == 'classify' for f in result['warnings'])


def test_runtime_state_vars_are_all_defined_by_the_runtime():
    from elitea_sdk.runtime.langchain.utils import create_state
    assert validator.RUNTIME_STATE_VARS <= set(create_state({}).__annotations__)


@pytest.mark.parametrize('value', ['input', [{'name': 'input'}]])
def test_r4_input_must_be_a_list_of_names(value):
    doc = spec()
    doc['nodes'][0]['input'] = value
    assert 'R4' in rules(validate_pipeline(doc))


def test_r1_unused_state_warns_and_reference_surfaces_count():
    doc = spec()
    doc['state'].update({'unused': {'type': 'str'}, 'in_template': {'type': 'str'}, 'cleaned': {'type': 'str'}})
    doc['nodes'][1]['condition'] = "{% if intent == 'bug' and in_template %}answer{% endif %}"
    doc['nodes'][2]['variables_to_clean'] = ['cleaned']
    result = validate_pipeline(doc)
    assert result['ok'] and [f['field'] for f in result['warnings'] if f['rule'] == 'R1'] == ['state.unused']


def test_r26_custom_state_without_messages_warns():
    doc = spec()
    del doc['state']['messages']
    doc['nodes'][2]['input'] = ['input', 'intent']
    result = validate_pipeline(doc)
    assert result['ok'] and rules(result, 'warnings') == ['R26']


# ---- R5 mapping blocks ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('mapping,field', [
    ({'task': 'just text'}, 'input_mapping.task'),
    ({'task': {'type': 'template', 'value': 'x'}}, 'input_mapping.task.type'),
    ({'task': {'type': 'fixed'}}, 'input_mapping.task.value'),
    ({'task': {'type': 'fixed', 'value': 'x'}, 'transition': 'END'}, 'input_mapping.transition'),
])
def test_r5_input_mapping_entries(mapping, field):
    doc = spec()
    doc['nodes'][2]['input_mapping'] = mapping
    assert field in [f['field'] for f in validate_pipeline(doc)['errors'] if f['rule'] == 'R5']


@pytest.mark.parametrize('code,field', [("{'x': 1}", 'code'), ({'value': "{'x': 1}"}, 'code.type')])
def test_r5_code_block_shape(code, field):
    node = {'id': 'a', 'type': 'code', 'code': code, 'transition': 'END'}
    assert field in [f['field'] for f in validate_pipeline(with_nodes(node))['errors'] if f['rule'] == 'R5']


# ---- R6, R12 outputs ---------------------------------------------------------------------------------------------

def test_r6_structured_llm_output_must_not_list_messages():
    doc = spec()
    doc['nodes'][0]['output'] = ['intent', 'messages']
    assert 'R6' in rules(validate_pipeline(doc))


def test_r12_state_modifier_writes_one_variable():
    node = {'id': 'a', 'type': 'state_modifier', 'template': '{{ input }}', 'input': ['input'],
            'output': ['a', 'b'], 'transition': 'END'}
    state = {'a': {'type': 'str'}, 'b': {'type': 'str'}, 'messages': {'type': 'list'}}
    assert 'R12' in rules(validate_pipeline(with_nodes(node, state=state)))


# ---- R7, R27, R28 code nodes ----------------------------------------------------------------------------------------

def code_node(source, structured=True):
    node = {'id': 'n', 'type': 'code', 'input': ['input'], 'output': ['out'], 'structured_output': structured,
            'code': {'type': 'fixed', 'value': source}, 'transition': 'END'}
    return with_nodes(node, state={'out': {'type': 'dict'}, 'messages': {'type': 'list'}})


R7_MUST_FAIL = {
    'result-only-inside-a-branch': 'v = elitea_state.get("input")\nif v:\n    {"out": v}\nelse:\n    {"out": ""}',
    'result-returned-from-a-def': 'def build():\n    return {"out": "v"}\nx = build()',
    'trailing-unrelated-name': 'unrelated = 1\nif True:\n    d = {"out": "hidden"}\nunrelated',
    'result-in-a-comprehension': 'vals = [{"out": str(i)} for i in range(3)]\nlen(vals)',
    'trailing-fstring': 'd = {"out": 1}\nf"{d}"',
}

R7_MUST_PASS = {
    'bare-trailing-dict': 'v = elitea_state.get("input", "")\n{"out": v.upper()}',
    'trailing-var-set-on-every-branch': 'if True:\n    r = {"out": 1}\nelse:\n    r = {"out": 2}\nr',
    'trailing-var-from-a-call': 'def build():\n    return {"out": "v"}\nresult = build()\nresult',
    'attribute-call': 'base = {"out": "v"}\nbase.copy()',
    'json-round-trip': 'import json\nraw = {"out": "v"}\njson.loads(json.dumps(raw))',
    'conditional-expression': 'a = {"out": "a"}\nb = {"out": "b"}\na if True else b',
    'boolean-or': 'a = {}\nb = {"out": "v"}\na or b',
    'dict-merge': 'a = {"out": "v"}\nb = {}\na | b',
    'call-with-kwargs': 'def build(**kw):\n    return kw\nargs = {"out": "v"}\nbuild(**args)',
    'lambda-called': '(lambda: {"out": "v"})()',
    'next-on-generator': 'def gen():\n    yield {"out": "v"}\nnext(gen())',
    'shadowed-builtin': 'def len(x):\n    return {"out": "s"}\nlen([1])',
    'subscript': 'pair = ({"out": "v"}, 1)\npair[0]',
    'top-level-await': 'import asyncio\nawait asyncio.sleep(0)\n{"out": 1}',
    'no-dict-at-all': 'x = 1\nx',
}


@pytest.mark.parametrize('name', R7_MUST_FAIL)
def test_r7_result_not_captured(name):
    assert 'R7' in rules(validate_pipeline(code_node(R7_MUST_FAIL[name])))


@pytest.mark.parametrize('name', R7_MUST_PASS)
def test_r7_no_false_positive(name):
    result = validate_pipeline(code_node(R7_MUST_PASS[name]))
    assert 'R7' not in rules(result) and result['ok'], result['errors']


def test_r7_ignores_nodes_without_structured_output():
    assert 'R7' not in rules(validate_pipeline(code_node(R7_MUST_FAIL['result-returned-from-a-def'], structured=False)))


def test_r27_top_level_return_is_a_syntax_error_with_the_fix():
    finding = validate_pipeline(code_node('return {"out": 1}'))['errors'][0]
    assert finding['rule'] == 'R27' and 'return' in finding['suggestion']


def test_r28_code_reads_state_it_does_not_list():
    result = validate_pipeline(code_node('x = elitea_state.get("secret")\ny = alita_state["other"]\n{"out": x}'))
    assert result['ok'] and sorted(f['message'].split('`')[3] for f in result['warnings'] if f['rule'] == 'R28') \
        == ['other', 'secret']


def test_r28_quiet_when_all_state_is_injected():
    doc = code_node('x = elitea_state.get("anything")\n{"out": x}')
    doc['nodes'][0]['input'] = ['messages']
    assert 'R28' not in rules(validate_pipeline(doc), 'warnings')


# ---- R29, R31, R33, R35, R36 -----------------------------------------------------------------------------------------

def test_r29_many_message_writers_warns():
    doc = spec()
    for node in doc['nodes'][:1]:
        node['structured_output'] = False
        node['output'] = ['messages']
    doc['nodes'].insert(2, {'id': 'extra', 'type': 'llm', 'input_mapping': {'task': {'type': 'fixed', 'value': 'x'}},
                            'output': ['messages'], 'transition': 'answer'})
    doc['nodes'][1]['routes'] = ['extra']
    doc['nodes'][1]['condition'] = 'extra'
    assert 'R29' in rules(validate_pipeline(doc), 'warnings')


def test_r31_blocking_node_on_a_schedule():
    draft = {'id': 'draft', 'type': 'llm', 'input_mapping': {'task': {'type': 'fixed', 'value': 'x'}},
             'transition': 'h'}
    node = {'id': 'h', 'type': 'hitl', 'user_message': {'type': 'fixed', 'value': 'ok?'},
            'routes': {'approve': 'END', 'reject': 'END'}}
    assert validate_pipeline(with_nodes(draft, node), ValidationContext(trigger='chat'))['ok']
    assert rules(validate_pipeline(with_nodes(draft, node), ValidationContext(trigger='schedule'))) == ['R31']


def test_r33_fstring_variable_not_in_input():
    doc = spec()
    doc['nodes'][2]['input'] = ['input']
    result = validate_pipeline(doc)
    assert result['ok'] and 'R33' in rules(result, 'warnings')


def test_r33_ignores_jinja_and_escaped_braces():
    doc = spec()
    doc['nodes'][2]['input_mapping']['task']['value'] = 'Return {{"a": 1}} for {input} {intent}'
    assert 'R33' not in rules(validate_pipeline(doc), 'warnings')


@pytest.mark.parametrize('routes,edit_key,expected', [
    ({'approve': 'END', 'edit': 'END'}, 'intent', 'R35'),
    ({'approve': 'END', 'edit': 'classify'}, None, 'R35'),
    ({'approve': 'END', 'edit': 'classify'}, 'undeclared', 'R35'),
    ({'approve': 'END', 'retry': 'classify'}, None, 'R35'),
])
def test_r35_hitl_routes(routes, edit_key, expected):
    doc = spec()
    node = {'id': 'review', 'type': 'hitl', 'user_message': {'type': 'fstring', 'value': '{intent}'},
            'input': ['intent'], 'routes': routes}
    if edit_key:
        node['edit_state_key'] = edit_key
    doc['nodes'].append(node)
    doc['nodes'][2]['transition'] = 'review'
    assert expected in rules(validate_pipeline(doc))


def test_r36_tool_parameters_against_the_schema():
    node = {'id': 'get', 'type': 'toolkit', 'toolkit_name': 'jira', 'tool': 'get_issue', 'transition': 'END',
            'input_mapping': {'issue': {'type': 'fixed', 'value': 'X-1'}}}
    schemas = {'jira': {'get_issue': {'properties': {'issue_key': {}, 'fields': {}}, 'required': ['issue_key']}}}
    result = validate_pipeline(with_nodes(node), ValidationContext(tool_schemas=schemas))
    assert rules(result) == ['R36'] and rules(result, 'warnings') == ['R36']
    node['input_mapping'] = {'issue_key': {'type': 'fixed', 'value': 'X-1'}}
    assert_clean(validate_pipeline(with_nodes(node), ValidationContext(tool_schemas=schemas)))


def test_validator_module_imports_only_the_registry():
    tree = ast.parse(Path(validator.__file__).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            assert node.module == 'registry', ast.dump(node)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or '').startswith('elitea_sdk')
