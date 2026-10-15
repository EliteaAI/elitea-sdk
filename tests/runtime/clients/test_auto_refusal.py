"""A Gateway refusal of Auto reaches the user as a readable message, not a generic failure."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
from langchain_core.messages import HumanMessage

from elitea_sdk.runtime.exceptions import AutoRoutingRefused, PipelineConfigurationError

path = Path(__file__).resolve().parents[3]/'elitea_sdk/runtime/clients/routing.py'
spec = importlib.util.spec_from_file_location('auto_refusal_contract', path)
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

ADMIN_HINT = 'Ask a project admin to choose a classifier in Project Settings → General → Chat configuration.'


def refusing_model(status, body=None, json_error=False):
    def raise_for_status():
        raise requests.HTTPError(f'{status} Error', response=SimpleNamespace(status_code=status))
    def parse():
        if json_error:
            raise ValueError('not json')
        return body
    owner = SimpleNamespace(base_url='https://unit.invalid', headers={}, project_id=7)
    owner._request = lambda method, url, **kwargs: SimpleNamespace(
        status_code=status, raise_for_status=raise_for_status, json=parse)
    owner.get_llm = Mock()
    return m.AutoChatModel(owner=owner, settings={'selection': {'mode': 'auto'}, 'routing_surface': 'agent'})


def invoke(model):
    config = {'configurable': {'thread_id': 't', 'checkpoint_ns': 'agent', 'elitea_routing_run_id': 'r'}}
    return model.invoke([HumanMessage(content='hello')], config)


@pytest.mark.parametrize('reason', ['CLASSIFIER_UNAVAILABLE', 'CLASSIFIER_NOT_CONFIGURED', 'CLASSIFIER_PRICE_UNAVAILABLE'])
def test_classifier_refusal_tells_user_to_ask_an_admin(reason):
    with pytest.raises(AutoRoutingRefused) as caught:
        invoke(refusing_model(422, {'error': 'The Auto classifier is unavailable', 'reason': reason}))
    error = caught.value
    assert str(error) == f'Auto model selection cannot run: The Auto classifier is unavailable. {ADMIN_HINT}'
    assert (error.reason, error.status_code) == (reason, 422)
    # The indexer already shows this type's own message to the user.
    assert isinstance(error, PipelineConfigurationError)


@pytest.mark.parametrize('body', [{'error': 'No model qualifies', 'reason': None},
                                  {'error': 'No model qualifies', 'reason': 'NO_QUALIFIED_MODELS'},
                                  {'error': 'No model qualifies'}])
def test_other_refusal_has_no_admin_hint(body):
    with pytest.raises(AutoRoutingRefused) as caught:
        invoke(refusing_model(422, body))
    assert str(caught.value) == 'Auto model selection cannot run: No model qualifies.'
    assert caught.value.reason == body.get('reason')
    assert caught.value.status_code == 422


def test_server_error_keeps_http_error():
    with pytest.raises(requests.HTTPError):
        invoke(refusing_model(500, {'error': 'boom', 'reason': 'CLASSIFIER_UNAVAILABLE'}))


@pytest.mark.parametrize('body,json_error', [(None, True), ({'detail': 'nope'}, False), ({'error': {'message': 'x'}}, False), ({'error': ''}, False)])
def test_client_error_without_readable_error_keeps_http_error(body, json_error):
    with pytest.raises(requests.HTTPError):
        invoke(refusing_model(422, body, json_error))
