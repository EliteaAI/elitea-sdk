"""Issue #5680 — application-tool cycle / runaway-depth guard.

Loading an ``application``-type tool recurses into ``ApplicationToolkit.get_toolkit`` ->
``LangChainAssistant`` -> ``get_tools`` for the nested agent. Without a guard, a self-reference
(A->A) or a loop (A->B->A) recurses until Python raises ``maximum recursion depth exceeded``.

These tests patch ``ApplicationToolkit.get_toolkit`` with a fake that re-enters ``get_tools``
using a small in-memory registry of agent configs, reproducing the recursion shape without any
real client, then assert the guard stops it with exactly one clear log line.
"""

from unittest.mock import patch

from langchain_core.tools import StructuredTool

from elitea_sdk.runtime.toolkits import tools as tools_mod
from elitea_sdk.runtime.toolkits.tools import get_tools, _MAX_APP_NESTING_BACKSTOP


class _FakeClient:
    """Minimal elitea_client stub — only the MCP-toolkit hook get_tools touches at the end."""

    def get_mcp_toolkits(self):
        return []


def _client():
    return _FakeClient()


def _app_tool(app_id: int, version_id: int, name: str = None) -> dict:
    """Build an ``application``-type tool config as the backend would send it."""
    return {
        'type': 'application',
        'name': name or f'app_{app_id}',
        'project_id': 1,
        'agent_type': 'agent',
        'settings': {
            'application_id': app_id,
            'application_version_id': version_id,
        },
    }


def _leaf_tool(name: str) -> StructuredTool:
    """A real terminal tool an agent might expose (so a loaded agent yields something)."""
    return StructuredTool.from_function(
        func=lambda: 'ok',
        name=name,
        description=f'leaf tool {name}',
    )


def _make_fake_get_toolkit(registry: dict):
    """Return a fake ApplicationToolkit.get_toolkit bound to ``registry``.

    ``registry`` maps (app_id, version_id) -> list of tool configs that agent exposes.
    The returned toolkit's ``get_tools`` re-enters the real ``get_tools`` for that agent's
    tools — this is exactly the recursion the guard must bound.
    """

    class _FakeToolkit:
        def __init__(self, app_id, version_id):
            self._key = (app_id, version_id)

        def get_tools(self):
            nested = registry.get(self._key, [])
            # Re-enter the real loader for the nested agent's own tools.
            return get_tools(nested, elitea_client=_client())

    def _fake_get_toolkit(client, application_id, application_version_id, **kwargs):
        return _FakeToolkit(application_id, application_version_id)

    return _fake_get_toolkit


def test_self_reference_is_skipped_without_recursion(caplog):
    """A->A: agent 1 references itself. One warning, no bound tool, no RecursionError."""
    registry = {(1, 1): [_app_tool(1, 1, 'self')]}
    with patch.object(tools_mod.ApplicationToolkit, 'get_toolkit',
                      _make_fake_get_toolkit(registry)):
        with caplog.at_level('WARNING'):
            result = get_tools([_app_tool(1, 1, 'self')], elitea_client=_client())

    assert result == []
    cyclic = [r for r in caplog.records if 'Circular application reference' in r.message]
    assert len(cyclic) == 1, f"expected one cycle warning, got {[r.message for r in caplog.records]}"


def test_two_hop_cycle_is_skipped(caplog):
    """A->B->A: mutual references terminate at the second entry into A."""
    registry = {
        (1, 1): [_app_tool(2, 2, 'b')],
        (2, 2): [_app_tool(1, 1, 'a')],
    }
    with patch.object(tools_mod.ApplicationToolkit, 'get_toolkit',
                      _make_fake_get_toolkit(registry)):
        with caplog.at_level('WARNING'):
            result = get_tools([_app_tool(1, 1, 'a')], elitea_client=_client())

    assert result == []
    cyclic = [r for r in caplog.records if 'Circular application reference' in r.message]
    assert len(cyclic) == 1


def test_acyclic_chain_loads_with_no_warnings(caplog):
    """A->B->C (distinct, acyclic) loads fully — the guard must not touch legit nesting."""
    registry = {
        (1, 1): [_app_tool(2, 2, 'b')],
        (2, 2): [_app_tool(3, 3, 'c')],
        (3, 3): [{'type': '__leaf__'}],  # sentinel handled below
    }

    # For the terminal agent, return a real leaf tool instead of another application.
    class _FakeToolkit:
        def __init__(self, app_id, version_id):
            self._key = (app_id, version_id)

        def get_tools(self):
            if self._key == (3, 3):
                return [_leaf_tool('final_leaf')]
            return get_tools(registry[self._key], elitea_client=_client())

    def _fake(client, application_id, application_version_id, **kwargs):
        return _FakeToolkit(application_id, application_version_id)

    with patch.object(tools_mod.ApplicationToolkit, 'get_toolkit', _fake):
        with caplog.at_level('WARNING'):
            result = get_tools([_app_tool(1, 1, 'a')], elitea_client=_client())

    assert [t.name for t in result] == ['final_leaf']
    warnings = [r for r in caplog.records
                if 'Circular application reference' in r.message
                or 'nesting depth' in r.message]
    assert warnings == [], f"legit acyclic chain must not warn: {[r.message for r in warnings]}"


def test_deep_noncyclic_chain_trips_backstop(caplog):
    """A chain of distinct apps longer than the backstop is cut cleanly (no stack overflow).

    Each agent references the next distinct app id, so there is never a cycle — only the
    depth backstop can stop it.
    """
    depth = _MAX_APP_NESTING_BACKSTOP + 5
    registry = {(i, i): [_app_tool(i + 1, i + 1, f'app_{i + 1}')] for i in range(1, depth + 2)}

    with patch.object(tools_mod.ApplicationToolkit, 'get_toolkit',
                      _make_fake_get_toolkit(registry)):
        with caplog.at_level('WARNING'):
            result = get_tools([_app_tool(1, 1, 'a')], elitea_client=_client())

    assert result == []
    backstop = [r for r in caplog.records if 'backstop' in r.message]
    assert len(backstop) == 1, f"expected one backstop warning, got {[r.message for r in caplog.records]}"
    # And crucially: no cycle warning, since the chain is acyclic.
    assert not [r for r in caplog.records if 'Circular application reference' in r.message]


def test_contextvars_restored_after_load(caplog):
    """The load-path ContextVars must return to their defaults after get_tools completes."""
    registry = {(1, 1): [_app_tool(2, 2, 'b')], (2, 2): []}
    with patch.object(tools_mod.ApplicationToolkit, 'get_toolkit',
                      _make_fake_get_toolkit(registry)):
        get_tools([_app_tool(1, 1, 'a')], elitea_client=_client())

    assert tools_mod._APP_LOAD_STACK.get() == frozenset()
    assert tools_mod._APP_LOAD_DEPTH.get() == 0
