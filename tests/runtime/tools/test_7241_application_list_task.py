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
