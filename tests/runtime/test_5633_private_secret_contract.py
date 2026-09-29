"""Contract tests for GitHub issue #5633: get_private_project_secret's error mapping.

The platform endpoint reports why it refused under the JSON key `error`. Status alone is
not enough to tell a code node what went wrong: `no_personal_project` and `not_found` are
both 404, and the two mean very different things to whoever wrote the pipeline. These
tests pin the bodies the endpoint actually sends to the exceptions users see, so a rename
on either side fails here instead of surfacing as a confusing runtime error.
"""
from unittest.mock import patch

import pytest

from elitea_sdk.runtime.clients.sandbox_client import (
    NoPersonalProject,
    PrivateSecretError,
    PrivateSecretNotFound,
    PrivateSecretNotShared,
    SandboxClient,
)


class _FakeResponse:
    def __init__(self, status_code, payload, reason='Fake'):
        self.status_code = status_code
        self.ok = status_code < 400
        self.reason = reason
        self._payload = payload

    def json(self):
        if self._payload is _NOT_JSON:
            raise ValueError('no json body')
        return self._payload


_NOT_JSON = object()


def _client():
    return SandboxClient(base_url='http://elitea.invalid', project_id=7, auth_token='token')


def _call(response, secret_name='TOKEN', **kwargs):
    client = _client()
    with patch.object(client, '_request', return_value=response) as request:
        return client.get_private_project_secret(secret_name, **kwargs), request


# --- the wire contract ----------------------------------------------------------
#
# (status, error) pairs are exactly what plugins/secrets/api/v2/private_secret.py returns.


@pytest.mark.parametrize(('status', 'error', 'expected'), [
    (404, 'no_personal_project', NoPersonalProject),
    (403, 'not_shared', PrivateSecretNotShared),
    (404, 'not_found', PrivateSecretNotFound),
    (400, 'default_secret', PrivateSecretError),
    (400, 'invalid_secret_name', PrivateSecretError),
    (401, 'unidentified_caller', PrivateSecretError),
])
def test_each_refusal_maps_to_its_exception(status, error, expected):
    with pytest.raises(expected):
        _call(_FakeResponse(status, {'error': error}))


def test_the_two_404s_do_not_collapse_into_one_exception():
    """The whole reason the body is read rather than the status."""
    with pytest.raises(NoPersonalProject):
        _call(_FakeResponse(404, {'error': 'no_personal_project'}))
    with pytest.raises(PrivateSecretNotFound):
        _call(_FakeResponse(404, {'error': 'not_found'}))


def test_error_is_the_authoritative_key():
    """An earlier design sketch used `code`; the endpoint settled on `error`."""
    with pytest.raises(PrivateSecretError) as raised:
        _call(_FakeResponse(403, {'code': 'not_shared'}))

    assert not isinstance(raised.value, PrivateSecretNotShared)


def test_an_unrecognized_reason_still_raises_the_base_error():
    with pytest.raises(PrivateSecretError):
        _call(_FakeResponse(500, {'error': 'something_new'}))


def test_a_non_json_body_does_not_mask_the_failure():
    with pytest.raises(PrivateSecretError):
        _call(_FakeResponse(502, _NOT_JSON))


# --- success path ---------------------------------------------------------------


def test_returns_the_value_on_success():
    value, _request = _call(_FakeResponse(200, {'value': 'personal-value'}))

    assert value == 'personal-value'


def test_the_secret_name_is_encoded_into_a_single_path_segment():
    """An unencoded name could otherwise address a different endpoint."""
    _value, request = _call(_FakeResponse(200, {'value': 'v'}), secret_name='a/b')

    url = request.call_args[0][1]
    assert url.endswith('/secrets/private_secret/7/a%2Fb')


# --- default ---------------------------------------------------------------------


@pytest.mark.parametrize('error', ['no_personal_project', 'not_shared', 'not_found'])
def test_an_explicit_default_replaces_any_refusal(error):
    value, _request = _call(_FakeResponse(404, {'error': error}), default='fallback')

    assert value == 'fallback'


def test_a_none_default_is_honored_rather_than_treated_as_absent():
    value, _request = _call(_FakeResponse(403, {'error': 'not_shared'}), default=None)

    assert value is None


def test_not_shared_message_also_covers_a_missing_secret():
    """The endpoint answers not_shared for a missing secret too, so the message must say so."""
    with pytest.raises(PrivateSecretNotShared) as raised:
        _call(_FakeResponse(403, {'error': 'not_shared'}), secret_name='AQA_TOKEN')

    assert 'is not shared or does not exist' in str(raised.value)
