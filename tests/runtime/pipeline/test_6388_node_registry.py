"""Pipeline node registry (#6388 / #7544).

The registry is the single source of truth for node types; these tests keep it honest
against the runtime that actually builds the graph.
"""

import ast
import json
from pathlib import Path

import pytest

from elitea_sdk.runtime import pipeline
from elitea_sdk.runtime.pipeline import registry

# Read as source, not imported: importing it pulls in the whole runtime.
_LANGRAPH_AGENT = Path(registry.__file__).parents[1] / 'langchain' / 'langraph_agent.py'

# The type create_graph falls back to when a node has no `type`; it is a removed type,
# so the registry must still know it.
_DEFAULT_NODE_TYPE = 'function'


def _types_dispatched_by_create_graph() -> set:
    """Every string create_graph compares `node_type` against, read from its source."""
    tree = ast.parse(_LANGRAPH_AGENT.read_text(encoding='utf-8'))
    create_graph = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'create_graph'
    )
    found = set()
    for node in ast.walk(create_graph):
        if not (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == 'node_type'):
            continue
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(right, ast.Constant):
                found.add(right.value)
            elif isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.List, ast.Tuple, ast.Set)):
                found.update(e.value for e in right.elts if isinstance(e, ast.Constant))
    return found


def test_registry_lists_every_node_type_the_runtime_dispatches():
    dispatched = _types_dispatched_by_create_graph() | {_DEFAULT_NODE_TYPE}
    assert dispatched, 'found no node_type comparisons in create_graph; the AST scan is broken'
    missing = dispatched - set(pipeline.node_types())
    assert not missing, f'create_graph dispatches node types the registry does not list: {sorted(missing)}'


def test_registry_lists_no_node_type_the_runtime_does_not_know():
    unknown = set(pipeline.node_types()) - _types_dispatched_by_create_graph() - {_DEFAULT_NODE_TYPE}
    assert not unknown, f'registry lists node types create_graph never dispatches: {sorted(unknown)}'


def test_custom_is_the_types_the_runtime_names_as_valid():
    """create_graph's error for `custom` lists the types a user may pick; they must all be offered
    (or, for printer, deliberately deprecated) and none of them removed."""
    source = _LANGRAPH_AGENT.read_text(encoding='utf-8')
    tree = ast.parse(source)
    valid = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'valid_types' for t in node.targets):
            valid = [e.value for e in node.value.elts]
    assert valid, 'valid_types list not found in create_graph'
    for name in valid:
        assert pipeline.node_spec(name)['status'] in ('current', 'deprecated'), name


@pytest.mark.parametrize('name,status', [
    ('llm', 'current'), ('agent', 'current'), ('toolkit', 'current'), ('mcp', 'current'),
    ('code', 'current'), ('router', 'current'), ('decision', 'current'), ('hitl', 'current'),
    ('state_modifier', 'current'),
    ('printer', 'deprecated'), ('indexer', 'deprecated'),
    ('custom', 'ui_only'),
    ('function', 'removed'), ('tool', 'removed'), ('loop', 'removed'), ('loop_from_tool', 'removed'),
    ('pipeline', 'removed'), ('subgraph', 'removed'),
])
def test_initial_statuses(name, status):
    assert pipeline.node_spec(name)['status'] == status


def test_nine_current_executable_types():
    assert sorted(pipeline.node_types('current')) == sorted([
        'llm', 'agent', 'toolkit', 'mcp', 'code', 'router', 'decision', 'hitl', 'state_modifier',
    ])


def test_routing_kind_matches_how_create_graph_wires_the_node():
    """Types that `continue` past the shared transition block install their own edges."""
    assert {n for n in pipeline.node_types('current', 'deprecated') if pipeline.node_spec(n)['routing'] == 'own'} \
        == {'router', 'decision', 'hitl'}


def test_tool_nodes_require_tool_not_name():
    """The runtime reads `tool` (and optionally `toolkit_name`); a `name` field is ignored."""
    for name in ('toolkit', 'mcp', 'agent'):
        flat = [p for alternatives in pipeline.required_paths(name) for p in alternatives]
        assert 'tool' in flat, name
        assert 'name' not in flat, name


def test_required_alternatives_are_split():
    assert pipeline.required_paths('llm') == [['input_mapping.system', 'input_mapping.task']]
    assert pipeline.required_paths('unknown-type') == []


def test_inline_constructs():
    assert pipeline.inline_construct('condition')['status'] == 'deprecated'  # it still routes
    assert pipeline.inline_construct('decision')['status'] == 'deprecated'


def test_get_node_registry_is_json_and_a_copy():
    data = pipeline.get_node_registry()
    json.dumps(data)
    data['nodes']['llm']['status'] = 'removed'
    assert pipeline.node_spec('llm')['status'] == 'current'


def test_registry_hash_changes_only_with_the_named_entries(monkeypatch):
    before_router = pipeline.registry_entries_hash(['router'])
    before_llm = pipeline.registry_entries_hash(['llm'])
    data = json.loads(json.dumps(registry._load()))
    data['nodes']['router']['summary'] = 'changed'
    monkeypatch.setattr(registry, '_load', lambda: data)
    assert pipeline.registry_entries_hash(['router']) != before_router
    assert pipeline.registry_entries_hash(['llm']) == before_llm


@pytest.mark.parametrize('mutate,message', [
    (lambda d: d['nodes']['llm'].update(status='gone'), 'invalid status'),
    (lambda d: d['nodes']['llm'].update(routing='sideways'), 'invalid routing'),
    (lambda d: d['nodes']['printer'].pop('replacement'), 'names no replacement'),
])
def test_malformed_registry_is_rejected(mutate, message):
    data = json.loads(json.dumps(registry._load()))
    mutate(data)
    with pytest.raises(ValueError, match=message):
        registry._check(data)


def test_registry_module_imports_nothing_from_the_sdk():
    """The platform keeps a byte-identical copy of registry.py; it cannot import the SDK."""
    tree = ast.parse(Path(registry.__file__).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and not (node.module or '').startswith('elitea_sdk'), ast.dump(node)
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith('elitea_sdk') for a in node.names)
