"""Pipeline authoring knowledge (#6388 / #7221).

The pack is what a model reads before writing a pipeline, so it must not drift from the
runtime: every node type has an entry, every YAML example passes the validator, no entry
recommends a retired node, and the facts the registry owns come from the registry.
"""

import copy
import re

import pytest
import yaml

from elitea_sdk.runtime.pipeline import knowledge, registry
from elitea_sdk.runtime.pipeline.knowledge import get_reference, list_topics, render_entry
from elitea_sdk.runtime.pipeline.registry import node_types
from elitea_sdk.runtime.pipeline.validator import validate_pipeline

ENTRIES = knowledge._entries()
YAML_BLOCK = re.compile(r'```yaml\n(.*?)```', re.DOTALL)
# Sizes keep a loaded topic cheap; a Builder turn usually loads overview plus two or three nodes.
SIZE_LIMIT = {'overview': 4500, 'node': 3000, 'pattern': 3500}
DEFAULT_SIZE_LIMIT = 4000


def yaml_examples():
    for key, entry in sorted(ENTRIES.items()):
        for index, block in enumerate(YAML_BLOCK.findall(entry.body)):
            yield pytest.param(key, block, id=f'{key}#{index}')


def as_pipeline(block: str) -> dict:
    """Wrap a node or node-list snippet into a pipeline the validator can check on its own.

    State variables the snippet uses are declared. A single-node snippet's targets (the "next
    node" it points at) become stub llm nodes; a snippet of several nodes is a wiring example,
    so a target it names but does not define stays an error, as a typo would.
    """
    doc = yaml.safe_load(block)
    if isinstance(doc, dict) and 'nodes' in doc:
        pipeline = copy.deepcopy(doc)
    else:
        nodes = doc if isinstance(doc, list) else [doc]
        pipeline = {'nodes': copy.deepcopy(nodes)}
    nodes = pipeline['nodes']
    if 'entry_point' not in pipeline:
        first = nodes[0]
        if registry.node_spec(first.get('type', ''))['routing'] == 'own':
            # a lone routing node cannot start a pipeline; give it a plain predecessor
            nodes.insert(0, {'id': '_start', 'type': 'llm', 'transition': first['id'],
                             'input_mapping': {'task': {'type': 'fixed', 'value': 'stub'}}})
        pipeline['entry_point'] = nodes[0]['id']
    ids = {n['id'] for n in nodes}
    referenced = set()
    snippet = [n for n in nodes if n['id'] != '_start']
    if len(snippet) == 1:
        referenced.update(_targets(snippet[0]))
    for missing in sorted(referenced - ids - {'END'}):
        nodes.append({'id': missing, 'type': 'llm', 'transition': 'END',
                      'input_mapping': {'task': {'type': 'fixed', 'value': 'stub'}}})
    state = pipeline.setdefault('state', {})
    state.setdefault('messages', {'type': 'list'})
    for node in nodes:
        for name in list(node.get('input') or []) + list(node.get('output') or []):
            state.setdefault(name, {'type': 'str'})
        for entry in (node.get('input_mapping') or {}).values():
            if isinstance(entry, dict) and entry.get('type') == 'variable':
                state.setdefault(entry['value'], {'type': 'str'})
    return pipeline


def _targets(node):
    out = []
    for field in ('transition', 'default_output'):
        if isinstance(node.get(field), str):
            out.append(node[field])
    routes = node.get('routes')
    out += list(routes.values()) if isinstance(routes, dict) else list(routes or [])
    out += list(node.get('nodes') or [])
    return out


# ---- coverage and shape -------------------------------------------------------------------------------

def test_every_offered_or_retiring_node_type_has_an_entry():
    for name in node_types('current', 'deprecated', 'ui_only'):
        assert f'node:{name}' in ENTRIES, f'node type {name!r} has no knowledge/node/{name}.md'


def test_no_entry_for_a_node_type_the_registry_does_not_know():
    for key in ENTRIES:
        if key.startswith('node:'):
            assert registry.node_spec(key[5:]) is not None, key


def test_core_topics_exist():
    for key in ('overview', 'design', 'constraints', 'troubleshooting'):
        assert key in ENTRIES
    assert len([k for k in ENTRIES if k.startswith('pattern:')]) >= 8


def test_file_name_matches_key():
    for key, entry in ENTRIES.items():
        expected = key.split(':', 1)[1] if ':' in key else key
        assert entry.path.stem == expected, (key, entry.path)


def test_depends_on_names_registry_entries():
    known = set(node_types()) | set(registry._load().get('inline_constructs') or {})
    for key, entry in ENTRIES.items():
        unknown = set(entry.depends_on) - known
        assert not unknown, f'{key}: depends_on lists unknown entries {sorted(unknown)}'
        if key.startswith('node:'):
            assert key[5:] in entry.depends_on, key


@pytest.mark.parametrize('key', sorted(ENTRIES))
def test_entry_size_budget(key):
    kind = key.split(':', 1)[0]
    limit = SIZE_LIMIT.get(kind, SIZE_LIMIT.get(key, DEFAULT_SIZE_LIMIT))
    assert len(render_entry(key)) <= limit, f'{key} is {len(render_entry(key))} chars; keep it under {limit}'


# ---- content stays true to the runtime ---------------------------------------------------------------------

@pytest.mark.parametrize('key,block', list(yaml_examples()))
def test_every_yaml_example_passes_the_validator(key, block):
    result = validate_pipeline(as_pipeline(block))
    retired_node_page = key.startswith('node:') and registry.node_spec(key[5:])['status'] != 'current'
    errors = [f for f in result['errors'] if not (retired_node_page and f['rule'] == 'R23')]
    assert not errors, errors
    allowed = {'R1', 'R25'} | ({'R23'} if retired_node_page else set())
    unexpected = [f for f in result['warnings'] if f['rule'] not in allowed]
    assert not unexpected, unexpected


def test_examples_use_only_current_node_types_outside_their_own_page():
    current = set(node_types('current'))
    for key, entry in ENTRIES.items():
        own = key[5:] if key.startswith('node:') else None
        for block in YAML_BLOCK.findall(entry.body):
            for node in as_pipeline(block)['nodes']:
                assert node['type'] in current or node['type'] == own, (key, node['type'])


def test_prose_never_recommends_a_retired_node_type():
    """A retired type may be named only to say it is retired, on its own page or next to that word."""
    retired = node_types('deprecated', 'removed', 'ui_only')
    # `tool` is also the field every tool node sets, so only `type: tool` counts for it
    names = '|'.join(re.escape(n) for n in retired if n != 'tool')
    pattern = re.compile(r'`(?:type:\s*)?(' + names + r')`|`type:\s*(tool)`')
    for key, entry in ENTRIES.items():
        own = key[5:] if key.startswith('node:') else None
        prose = YAML_BLOCK.sub('', entry.body)
        for line in prose.splitlines():
            for name in (a or b for a, b in pattern.findall(line)):
                if name == own:
                    continue
                assert re.search(r'remove|deprecated|not a runtime type|editor|instead|do not|never|is not',
                                 line, re.IGNORECASE), f'{key}: mentions `{name}` without saying it is retired: {line!r}'


def test_no_legacy_names_or_line_citations():
    for key, entry in ENTRIES.items():
        assert 'alita_state' not in entry.body and 'alita_client' not in entry.body, key
        assert not re.search(r'langraph_agent\.py:\d', entry.body), key


def test_prose_does_not_restate_required_fields():
    """Required fields come from the registry; a hand-written 'Required:' line would drift."""
    for key, entry in ENTRIES.items():
        assert '**Required:**' not in entry.body, key


# ---- serving --------------------------------------------------------------------------------------------------

def test_node_entries_carry_registry_facts():
    text = render_entry('node:toolkit')
    assert '**Status:** current' in text and '**Required:** `tool`' in text


def test_retired_node_entry_names_its_replacement():
    text = render_entry('node:printer')
    assert 'deprecated' in text and '**Use instead:**' in text and 'Waits for a person' in text


def test_registry_change_shows_up_without_editing_the_file(monkeypatch):
    data = copy.deepcopy(registry._load())
    data['nodes']['router']['status'] = 'deprecated'
    data['nodes']['router']['replacement'] = 'a decision node'
    monkeypatch.setattr(registry, '_load', lambda: data)
    assert '**Use instead:** a decision node' in render_entry('node:router')
    assert '`router`' not in render_entry('overview').split('Do not use')[0].split('## Node types')[1]


def test_overview_lists_current_types_and_names_retired_ones():
    text = render_entry('overview')
    for name in node_types('current'):
        assert f'- `{name}`:' in text
    for name in node_types('deprecated', 'removed', 'ui_only'):
        assert f'`{name}`' in text.split('Do not use in new pipelines:')[1]


def test_get_reference_joins_topics_and_reports_unknown_ones():
    text = get_reference(['overview', 'node:llm', 'nonsense'])
    assert text.count('\n---\n') == 2
    assert 'Unknown topic(s): nonsense' in text and 'node:llm' in text.split('Valid topics:')[1]


def test_list_topics_starts_with_overview():
    assert [t['key'] for t in list_topics()][:2] == ['overview', 'design']
