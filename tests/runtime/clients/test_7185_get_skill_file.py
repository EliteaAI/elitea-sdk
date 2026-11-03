"""EliteAClient.get_skill_file fetches one bundled skill file with the run's token, once per run (#7185)."""
from unittest.mock import MagicMock, patch

import pytest

from elitea_sdk.runtime.clients.client import EliteAClient
from elitea_sdk.runtime.exceptions import SkillBodyUnavailable, SkillFileUnavailable


def _client():
    return EliteAClient(base_url='https://platform.example.com', project_id=42, auth_token='tok')


def _response(status_code, content=b''):
    return MagicMock(status_code=status_code, ok=200 <= status_code < 300, history=[], content=content)


def test_get_skill_file_calls_the_file_endpoint_of_the_skills_project_with_the_run_token():
    with patch('requests.Session.request') as request:
        request.return_value = _response(200, b'# Forms')

        assert _client().get_skill_file(1, 20, 120, 'scripts/fill form.py') == b'# Forms'

    args, kwargs = request.call_args
    assert args == (
        'get', 'https://platform.example.com/api/v2/elitea_core/skill_file/prompt_lib/1/20/120/scripts/fill%20form.py',
    )
    assert kwargs['headers']['Authorization'] == 'Bearer tok'
    assert kwargs['timeout'] == (5, 30)


def test_an_attached_skill_without_a_project_reads_from_the_run_project():
    with patch('requests.Session.request') as request:
        request.return_value = _response(200, b'x')

        _client().get_skill_file(None, 20, 120, 'forms.md')

    assert request.call_args[0][1].endswith('/skill_file/prompt_lib/42/20/120/forms.md')


def test_a_file_is_fetched_once_per_run_and_per_version():
    client = _client()
    with patch('requests.Session.request') as request:
        request.return_value = _response(200, b'x')

        client.get_skill_file(1, 20, 120, 'forms.md')
        client.get_skill_file(1, 20, 120, 'forms.md')
        client.get_skill_file(1, 20, 121, 'forms.md')

    assert request.call_count == 2


@pytest.mark.parametrize('status, denied', [(403, True), (404, True), (500, False), (502, False)])
def test_get_skill_file_maps_http_failures_to_denied_or_transient(status, denied):
    client = _client()
    with patch('requests.Session.request') as request:
        request.return_value = _response(status)
        with pytest.raises(SkillFileUnavailable) as error:
            client.get_skill_file(1, 20, 120, 'forms.md')
        request.return_value = _response(200, b'later')

        assert client.get_skill_file(1, 20, 120, 'forms.md') == b'later'

    assert error.value.denied is denied
    assert isinstance(error.value, SkillBodyUnavailable)
