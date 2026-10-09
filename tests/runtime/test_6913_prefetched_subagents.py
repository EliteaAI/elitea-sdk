"""Issue #6913 — sub-agent version details pre-expanded by the platform.

The platform now ships ``{"app_id:version_id": {name, description, version_details}}`` with the
predict task. ``ApplicationToolkit.get_toolkit`` must use that entry instead of the
``GET application`` + ``PATCH version`` round trips, and fall back to them on a miss so standalone
SDK use (no platform payload) is unchanged. ``get_mcp_toolkits`` is fetched once per client.
"""

from unittest.mock import MagicMock

from langchain_core.runnables import RunnableLambda

from elitea_sdk.runtime.clients.client import EliteAClient
from elitea_sdk.runtime.toolkits.application import ApplicationToolkit


def _version_details(name='child'):
    # llm_settings left empty so get_toolkit uses fallback_llm and never calls client.get_llm
    return {
        'id': 22, 'name': 'base', 'agent_type': 'openai', 'llm_settings': {},
        'instructions': f'{name} instructions', 'variables': [], 'meta': {}, 'tools': [],
    }


class _RecordingClient:
    """Same-project client stub that records every HTTP-backed call get_toolkit makes."""

    project_id = 1

    def __init__(self, prefetched=None):
        self.prefetched_version_details = prefetched or {}
        self.http_calls = []
        self.application_calls = []

    # real implementation under test, borrowed from EliteAClient
    get_prefetched_app = EliteAClient.get_prefetched_app

    def get_app_details(self, application_id):
        self.http_calls.append(('GET application', application_id))
        return {'name': 'fetched-name', 'description': 'fetched description'}

    def get_app_version_details(self, application_id, application_version_id):
        self.http_calls.append(('PATCH version', application_id, application_version_id))
        return _version_details('fetched')

    def application(self, application_id, application_version_id, **kwargs):
        self.application_calls.append(kwargs['version_details'])
        return RunnableLambda(lambda x: x)


def _build(client):
    return ApplicationToolkit.get_toolkit(
        client, application_id=11, application_version_id=22, selected_tools=[],
        project_id=1, fallback_llm=MagicMock(),
    ).get_tools()[0]


def test_prefetched_hit_skips_both_round_trips():
    client = _RecordingClient({'11:22': {
        'name': 'Prefetched Child', 'description': 'from payload',
        'version_details': _version_details('prefetched'),
    }})

    tool = _build(client)

    assert client.http_calls == []
    assert tool.name == 'PrefetchedChild'  # sanitized from the prefetched name, not 'fetched-name'
    assert client.application_calls[0]['instructions'] == 'prefetched instructions'


def test_miss_falls_back_to_get_and_patch():
    # Prefetched data exists, but for a different version — must not be used.
    client = _RecordingClient({'11:99': {'name': 'x', 'description': '', 'version_details': _version_details()}})

    _build(client)

    assert client.http_calls == [('GET application', 11), ('PATCH version', 11, 22)]
    assert client.application_calls[0]['instructions'] == 'fetched instructions'


def test_client_without_prefetch_support_uses_http():
    # Older/alternative clients (e.g. sandbox) have no get_prefetched_app at all.
    client = _RecordingClient()
    client.get_prefetched_app = None

    _build(client)

    assert len(client.http_calls) == 2


def test_prefetched_entry_is_not_mutated_across_builds():
    entry_details = _version_details('prefetched')
    entry_details['tools'] = [{'type': 'mcp', 'name': 'm', 'settings': {'url': 'u'}}]
    client = _RecordingClient({'11:22': {'name': 'n', 'description': '', 'version_details': entry_details}})

    _build(client)
    client.application_calls[0]['instructions'] = 'mutated by first build'
    _build(client)

    assert client.application_calls[1]['instructions'] == 'prefetched instructions'
    assert client.prefetched_version_details['11:22']['version_details']['instructions'] == 'prefetched instructions'


def test_malformed_entry_is_treated_as_miss():
    client = _RecordingClient({'11:22': {'name': 'n', 'version_details': None}})

    _build(client)

    assert len(client.http_calls) == 2


def test_mcp_toolkits_fetched_once_per_client():
    client = EliteAClient.__new__(EliteAClient)
    client._mcp_toolkits_cache = None
    client.mcp_tools_list = 'http://x/tools_list/1'
    client.headers = {}
    response = MagicMock()
    response.json.return_value = [{'name': 'tk'}]
    client._request = MagicMock(return_value=response)

    first = client.get_mcp_toolkits()
    first.append({'name': 'caller mutation'})
    second = client.get_mcp_toolkits()

    assert client._request.call_count == 1
    assert second == [{'name': 'tk'}]


def test_mcp_toolkits_error_response_is_not_cached():
    client = EliteAClient.__new__(EliteAClient)
    client._mcp_toolkits_cache = None
    client.mcp_tools_list = 'http://x/tools_list/1'
    client.headers = {}
    response = MagicMock()
    response.json.return_value = {'error': 'boom'}
    client._request = MagicMock(return_value=response)

    client.get_mcp_toolkits()
    client.get_mcp_toolkits()

    assert client._request.call_count == 2
