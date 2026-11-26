"""Issue #7759 — nested agents are built lazily, on first call, not when the parent loads.

Before: ``ApplicationToolkit.get_toolkit`` called ``client.application`` for every child
(recursively), and ``Application._run`` then rebuilt the child on each call anyway, so the
load-time build was always thrown away. Now load only reads version details and builds the
tool's schema/description; the per-call build in ``_run`` is the only one.

The cycle/depth guard (#5680) used to run during that recursive load. With lazy builds the
recursion happens at call time instead, so the tool carries its load path and restores it
while building — these tests pin that a cycle across lazy builds is still cut.
"""
import logging
from unittest.mock import MagicMock

import pytest

from elitea_sdk.runtime.credential_slots import GovernedCredentialError
from elitea_sdk.runtime.toolkits import tools as tools_mod
from elitea_sdk.runtime.toolkits.application import ApplicationToolkit
from elitea_sdk.runtime.toolkits.tools import get_tools
from elitea_sdk.runtime.tools.application import Application
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired


class _BuildStop(Exception):
    """Raised by the fake client.application so a test can inspect the build without running it."""


def _app_tool_cfg(app_id, ver_id):
    return {'type': 'application', 'name': f'app_{app_id}', 'project_id': 1, 'agent_type': 'agent',
            'settings': {'application_id': app_id, 'application_version_id': ver_id}}


def _version(tools=None):
    return {'id': 0, 'name': 'base', 'agent_type': 'openai', 'llm_settings': {},
            'instructions': 'x', 'variables': [], 'meta': {}, 'tools': tools or []}


class _Client:
    """Same-project client: version details from a registry; application() records each build."""

    project_id = 1

    def __init__(self, registry, on_build=None):
        self.registry = registry  # (app_id, ver_id) -> list of tool configs
        self.builds = []
        self.on_build = on_build

    def get_prefetched_app(self, application_id, application_version_id):
        return None

    def get_app_details(self, application_id):
        return {'name': f'app_{application_id}', 'description': ''}

    def get_app_version_details(self, application_id, application_version_id):
        return _version(self.registry.get((application_id, application_version_id), []))

    def get_mcp_toolkits(self):
        return []

    def application(self, application_id, application_version_id, **kwargs):
        self.builds.append(((application_id, application_version_id), tools_mod._APP_LOAD_STACK.get()))
        if self.on_build:
            return self.on_build(self, application_id, application_version_id, kwargs)
        raise _BuildStop()


def _load(client, app_id, ver_id):
    return ApplicationToolkit.get_toolkit(
        client, application_id=app_id, application_version_id=ver_id, selected_tools=[],
        project_id=1, fallback_llm=MagicMock(),
    ).get_tools()[0]


def test_loading_a_sub_agent_does_not_build_it():
    client = _Client({(2, 20): [_app_tool_cfg(3, 30)]})

    tool = _load(client, 2, 20)

    assert client.builds == []
    assert tool.application is None
    assert tool.args_runnable['application_id'] == 2


def test_nested_children_are_not_built_when_the_parent_loads():
    # Parent loads its tool list through the real loader; neither child nor grandchild is built.
    client = _Client({(2, 20): [_app_tool_cfg(3, 30)], (3, 30): []})

    loaded = get_tools([_app_tool_cfg(2, 20)], elitea_client=client, llm=MagicMock())

    assert [t.name for t in loaded] == ['app_2']
    assert client.builds == []


def test_each_call_builds_once_with_the_load_path_restored():
    client = _Client({})
    tool = _load(client, 2, 20)
    tool.load_path = frozenset({(1, 10), (2, 20)})

    for _ in range(2):
        with pytest.raises(_BuildStop):
            tool._run(task='do it')

    assert client.builds == [((2, 20), frozenset({(1, 10), (2, 20)}))] * 2
    assert tools_mod._APP_LOAD_STACK.get() == frozenset()


def test_cycle_across_lazy_builds_is_still_cut(caplog):
    # A -> B -> A: building B for a call must skip its reference back to A.
    loaded_tools_per_build = {}

    def build_by_loading_tools(client, app_id, ver_id, kwargs):
        tools = kwargs['version_details']['tools']
        loaded_tools_per_build[(app_id, ver_id)] = get_tools(tools, elitea_client=client, llm=MagicMock())
        raise _BuildStop()

    client = _Client({(1, 10): [_app_tool_cfg(2, 20)], (2, 20): [_app_tool_cfg(1, 10)]},
                     on_build=build_by_loading_tools)

    with caplog.at_level(logging.WARNING):
        [tool_a] = get_tools([_app_tool_cfg(1, 10)], elitea_client=client, llm=MagicMock())
        with pytest.raises(_BuildStop):
            tool_a._run(task='go')
        [tool_b] = loaded_tools_per_build[(1, 10)]
        with pytest.raises(_BuildStop):
            tool_b._run(task='go')

    assert loaded_tools_per_build[(2, 20)] == []
    assert [r for r in caplog.records if 'Circular application reference' in r.message]


def test_tool_without_runnable_or_build_config_fails_readably():
    tool = Application(name='orphan', description='d', client=None)

    with pytest.raises(Exception, match="no runnable configuration"):
        tool._run(task='x')


def _raising_build(exc):
    def build(client, app_id, ver_id, kwargs):
        raise exc
    return build


def test_mcp_authorization_from_a_lazy_build_reaches_the_caller_unchanged():
    # Used to surface while the parent loaded; the tool-call loop in llm.py now catches it at call time.
    err = McpAuthorizationRequired('auth needed', 'https://example/mcp')
    tool = _load(_Client({}, on_build=_raising_build(err)), 2, 20)

    with pytest.raises(McpAuthorizationRequired) as raised:
        tool._run(task='go')

    assert raised.value is err


def test_governed_refusal_from_a_lazy_build_reaches_the_caller_unchanged():
    err = GovernedCredentialError('governed_slot_unresolved', 'child')
    tool = _load(_Client({}, on_build=_raising_build(err)), 2, 20)

    with pytest.raises(GovernedCredentialError) as raised:
        tool._run(task='go')

    assert raised.value is err


def test_sub_agent_that_cannot_be_read_is_still_dropped_at_load():
    # Deleted / inaccessible sub-agents fail on the version-details read, which still happens at load.
    client = _Client({})
    client.get_app_version_details = MagicMock(side_effect=RuntimeError('404'))

    loaded = get_tools([_app_tool_cfg(2, 20)], elitea_client=client, llm=MagicMock())

    assert loaded == []
