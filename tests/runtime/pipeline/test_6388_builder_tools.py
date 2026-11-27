"""What the Builder's pipeline tools return and how they describe themselves (#6388 / #7222)."""

import ast
from pathlib import Path

from elitea_sdk.runtime.pipeline import builder
from elitea_sdk.runtime.pipeline.builder import (
    MAX_TOPICS_PER_CALL,
    VALIDATE_TOOL_DESCRIPTION,
    builder_reference,
    reference_tool_description,
    validate_for_builder,
)
from elitea_sdk.runtime.pipeline.knowledge import list_topics

# The two descriptions ride along on every model call while the Builder is enabled.
DESCRIPTION_BUDGET = 1500


def test_descriptions_fit_the_budget():
    assert len(reference_tool_description()) + len(VALIDATE_TOOL_DESCRIPTION) < DESCRIPTION_BUDGET


def test_reference_description_lists_every_topic_so_no_turn_is_spent_discovering_them():
    description = reference_tool_description()
    for topic in list_topics():
        assert topic['key'] in description


def test_descriptions_carry_no_schema():
    """Node fields belong in the reference result, not in text sent on every call."""
    for text in (reference_tool_description(), VALIDATE_TOOL_DESCRIPTION):
        for field in ('input_mapping', 'toolkit_name', 'default_output', 'structured_output'):
            assert field not in text


def test_no_topic_gives_the_overview():
    assert builder_reference().startswith('# Pipeline YAML: the shape')
    assert builder_reference([]) == builder_reference()


def test_comma_separated_string_is_accepted():
    assert builder_reference('overview, node:llm') == builder_reference(['overview', 'node:llm'])


def test_topics_per_call_are_capped():
    keys = [t['key'] for t in list_topics()][:MAX_TOPICS_PER_CALL + 2]
    text = builder_reference(keys)
    assert text.count('\n---\n') == MAX_TOPICS_PER_CALL
    assert f'Not loaded (at most {MAX_TOPICS_PER_CALL} topics per call): {keys[-2]}, {keys[-1]}' in text


def test_validate_summary_for_errors_warnings_and_clean():
    broken = validate_for_builder('entry_point: a\nnodes:\n  - {id: a, type: toolkit, name: Jira, transition: END}')
    assert broken['valid'] is False and broken['summary'].startswith('1 error(s) to fix before saving')
    assert 'reads `tool`, not `name`' in broken['errors'][0]['suggestion']

    deprecated = validate_for_builder(
        'entry_point: a\nnodes:\n  - {id: a, type: printer, transition: END,'
        ' input_mapping: {printer: {type: fixed, value: hi}}}')
    assert deprecated['valid'] is True and deprecated['summary'].startswith('No errors. 1 warning(s)')

    clean = validate_for_builder(
        'entry_point: a\nnodes:\n  - {id: a, type: llm, transition: END,'
        ' input_mapping: {task: {type: fixed, value: hi}}}')
    assert clean == {'valid': True, 'summary': 'Valid. No errors or warnings.', 'errors': [], 'warnings': []}


def test_validate_passes_the_trigger_through():
    spec = ('entry_point: a\nnodes:\n'
            '  - {id: a, type: llm, transition: h, input_mapping: {task: {type: fixed, value: x}}}\n'
            '  - {id: h, type: hitl, user_message: {type: fixed, value: "ok?"}, routes: {approve: END}}')
    assert validate_for_builder(spec)['valid']
    assert not validate_for_builder(spec, trigger='schedule')['valid']


def test_builder_module_imports_only_this_package():
    tree = ast.parse(Path(builder.__file__).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 1 or not (node.module or '').startswith('elitea_sdk'), ast.dump(node)
