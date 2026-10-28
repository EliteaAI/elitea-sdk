"""#7241: an Agent node whose Task field is mapped to a list-typed variable
must not fail ``task: str`` schema validation on the Application tool."""

import json

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.runtime.tools.application import Application, _coerce_task_to_str
from elitea_sdk.runtime.toolkits.application import build_dynamic_application_schema


class StaticApplication:
    def __init__(self):
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append(payload)
        return {'output': 'ok'}


def _app(child, args_schema=None):
    kwargs = dict(name='Yoda', description='child', application=child,
                  return_type='str', client=None, is_subgraph=True)
    if args_schema is not None:
        kwargs['args_schema'] = args_schema
    return Application(**kwargs)


@pytest.mark.parametrize('schema', [None, build_dynamic_application_schema([], 'Yoda')])
def test_list_task_is_serialized_to_json(schema):
    child = StaticApplication()
    _app(child, schema).invoke({'task': ['a', {'b': 1}], 'list': []})
    assert child.calls[0]['input'] == json.dumps(['a', {'b': 1}])


def test_dict_task_is_serialized_to_json():
    child = StaticApplication()
    _app(child).invoke({'task': {'k': 'v'}})
    assert child.calls[0]['input'] == '{"k": "v"}'


def test_empty_list_task_raises_task_required_not_validation_error():
    """The exact repro payload from the issue: task == []."""
    child = StaticApplication()
    with pytest.raises(ToolException, match='Task is required'):
        _app(child).invoke({'task': [], 'input_attachments': [], 'list': [], 'output': ''})
    assert not child.calls


def test_string_and_missing_task_untouched():
    payload = {'task': 'hello', 'x': 1}
    assert _coerce_task_to_str(payload) is payload
    payload = {'x': 1}
    assert _coerce_task_to_str(payload) is payload


@pytest.mark.parametrize('value, expected', [(5, '5'), (1.5, '1.5'), (True, 'true'), ({'a': [1, 2]}, '{"a": [1, 2]}')])
def test_scalar_and_nested_tasks_are_serialized(value, expected):
    child = StaticApplication()
    _app(child).invoke({'task': value})
    assert child.calls[0]['input'] == expected


def test_none_task_still_fails_schema_validation_as_before():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _app(StaticApplication()).invoke({'task': None})


def test_caller_input_is_not_mutated():
    payload = {'task': ['a']}
    _app(StaticApplication()).invoke(payload)
    assert payload == {'task': ['a']}


def test_non_task_variables_pass_through_unchanged():
    """Only task is coerced; extra state variables reach the child with their type."""
    child = StaticApplication()
    _app(child).invoke({'task': 'go', 'items': [1, 2], 'cfg': {'k': 1}})
    assert child.calls[0]['input'] == 'go'
    assert child.calls[0]['items'] == [1, 2]
    assert child.calls[0]['cfg'] == {'k': 1}


def test_tool_call_format_list_task():
    """Orchestrator-LLM path: ToolCall dict whose args.task is a list."""
    child = StaticApplication()
    result = _app(child).invoke({'type': 'tool_call', 'id': 'c1', 'name': 'Yoda', 'args': {'task': ['x']}})
    assert child.calls[0]['input'] == '["x"]'
    assert result.tool_call_id == 'c1'
