"""Issue #7759 — failures from a sub-agent built lazily, on its first call.

Before #7759 a sub-agent was built while the parent loaded, so its MCP-authorization
request or governed-credential refusal (#7229) surfaced before the run, and any other
build failure silently dropped the sub-agent. Now the build happens inside the parent's
tool call, so these tests go through the real Assistant graph (sequential and parallel
fan-out) to pin what each failure turns into there:

- governed refusal  -> ends the run readably, naming the refused sub-agent (as at load before),
                       and the parent model never reasons about it as a tool result;
- MCP authorization -> the durable mcp_auth pause, from the sequential loop and from a
                       parallel fan-out sibling alike, never tool prose;
- anything else     -> a readable tool error naming the sub-agent; the parent continues.

The last tests pin the cycle / depth guard (#5680) across lazy builds, with the root
agent itself on the load path.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.clients.client import _on_app_load_path
from elitea_sdk.runtime.credential_slots import GovernedCredentialError
from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.toolkits import tools as tools_mod
from elitea_sdk.runtime.toolkits.tools import _MAX_APP_NESTING_BACKSTOP, get_tools
from elitea_sdk.runtime.tools.application import Application
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired


class _Runtime:
    def get_mcp_toolkits(self):
        return []


class _FailingBuildClient:
    """client.application raises, as a sub-agent build does when its toolkit cannot load."""

    def __init__(self, exc):
        self.exc = exc
        self.builds = 0

    def application(self, application_id, application_version_id, **kwargs):
        self.builds += 1
        raise self.exc


class _OkChild:
    def invoke(self, payload, config=None):
        return {'output': 'fine'}


class _Bound:
    def __init__(self, root):
        self.root = root

    def invoke(self, messages, config=None):
        self.root.calls.append(list(messages))
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content='parent carried on')
        return AIMessage(content='', tool_calls=[
            {'name': name, 'args': {'task': 'Run'}, 'id': f'call-{i}', 'type': 'tool_call'}
            for i, name in enumerate(self.root.tool_names)
        ])


class _ParentLLM:
    def __init__(self, *tool_names):
        self.tool_names = tool_names
        self.calls = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _Bound(self)

    def invoke(self, messages, config=None):
        return _Bound(self).invoke(messages, config=config)


def _lazy_app(name, client):
    # Shaped like ApplicationToolkit.get_toolkit output: no prebuilt runnable, build args only.
    return Application(name=name, description=f'{name} worker', return_type='str', client=client,
                       is_subgraph=True, args_runnable={'application_id': 7, 'application_version_id': 70})


def _eager_app(name, application):
    return Application(name=name, description=f'{name} worker', application=application,
                       return_type='str', client=None, is_subgraph=True)


def _run(llm, tools):
    runnable = Assistant(
        elitea=_Runtime(), data={'instructions': 'Use tools', 'tools': [], 'meta': {}},
        client=llm, tools=tools, memory=MemorySaver(), app_type='predict',
    ).runnable()
    return runnable.invoke({'messages': [HumanMessage(content='delegate')]},
                           config={'configurable': {'thread_id': 'lazy-7759'}})


def _tool_messages(result):
    return [m for m in result['messages'] if isinstance(m, ToolMessage)]


def _auth_error():
    return McpAuthorizationRequired('auth needed', 'https://example/mcp')


# --- governed refusal: ends the run, naming the sub-agent ---------------------------

def _ended_with_refusal(result, llm, name):
    assert result['execution_finished'] is True
    assert f'sub-agent \u201c{name}\u201d' in result['output']
    assert len(llm.calls) == 1  # the parent model never saw the refusal as a tool result


def test_governed_refusal_in_a_lazy_build_ends_the_sequential_run():
    llm = _ParentLLM('refused_child')
    client = _FailingBuildClient(GovernedCredentialError('governed_slot_unresolved'))

    result = _run(llm, [_lazy_app('refused_child', client)])

    _ended_with_refusal(result, llm, 'refused_child')


def test_governed_refusal_keeps_the_innermost_sub_agent_name():
    llm = _ParentLLM('outer')
    client = _FailingBuildClient(GovernedCredentialError('governed_slot_unresolved', 'inner'))

    result = _run(llm, [_lazy_app('outer', client)])

    _ended_with_refusal(result, llm, 'inner')


def test_governed_refusal_in_a_lazy_build_ends_the_parallel_run():
    llm = _ParentLLM('ok_child', 'refused_child')
    client = _FailingBuildClient(GovernedCredentialError('governed_slot_unresolved'))

    result = _run(llm, [_eager_app('ok_child', _OkChild()), _lazy_app('refused_child', client)])

    _ended_with_refusal(result, llm, 'refused_child')


# --- MCP authorization: the durable pause, never tool prose ---------------------------

def _paused_for_auth(result, llm):
    assert (result.get('hitl_interrupt') or {}).get('guardrail_type') == 'mcp_auth'
    assert result['execution_finished'] is False
    assert not any('auth needed' in str(m.content) for m in _tool_messages(result))
    assert len(llm.calls) == 1


def test_mcp_authorization_in_a_lazy_build_pauses_the_sequential_run():
    llm = _ParentLLM('oauth_child')

    result = _run(llm, [_lazy_app('oauth_child', _FailingBuildClient(_auth_error()))])

    _paused_for_auth(result, llm)


def test_mcp_authorization_from_a_parallel_sibling_pauses_the_run():
    # Before the fix the parallel loop wrapped it as "Error executing ...: auth needed".
    llm = _ParentLLM('ok_child', 'oauth_child')
    tools = [_eager_app('ok_child', _OkChild()), _lazy_app('oauth_child', _FailingBuildClient(_auth_error()))]

    result = _run(llm, tools)

    _paused_for_auth(result, llm)
    assert result['hitl_interrupt']['tool_name'] == 'oauth_child'


# --- anything else: readable tool error, parent continues -----------------------------

def test_unexpected_build_failure_is_a_readable_tool_error():
    llm = _ParentLLM('broken_child')
    client = _FailingBuildClient(RuntimeError('toolkit settings invalid'))

    result = _run(llm, [_lazy_app('broken_child', client)])

    [message] = _tool_messages(result)
    assert "Sub-agent 'broken_child' could not be loaded" in message.content
    assert 'toolkit settings invalid' in message.content
    assert result['messages'][-1].content == 'parent carried on'
    assert client.builds == 1


# --- cycle / depth guard across lazy builds -------------------------------------------

class _RegistryClient:
    """Builds an agent the way EliteAClient.application does: load its tools through get_tools."""

    project_id = 1

    def __init__(self, registry):
        self.registry = registry  # (app_id, ver_id) -> list of application tool configs
        self.built = []

    def get_prefetched_app(self, application_id, application_version_id):
        return None

    def get_app_details(self, application_id):
        return {'name': f'app_{application_id}', 'description': ''}

    def get_app_version_details(self, application_id, application_version_id):
        return {'id': application_version_id, 'name': 'base', 'agent_type': 'openai', 'llm_settings': {},
                'instructions': 'x', 'variables': [], 'meta': {},
                'tools': self.registry.get((application_id, application_version_id), [])}

    def get_mcp_toolkits(self):
        return []

    @_on_app_load_path
    def application(self, application_id, application_version_id, **kwargs):
        self.built.append((application_id, application_version_id))
        tools = self.registry.get((application_id, application_version_id), [])
        return get_tools(tools, elitea_client=self, llm=object())


def _app_cfg(app_id):
    return {'type': 'application', 'name': f'app_{app_id}', 'project_id': 1, 'agent_type': 'agent',
            'settings': {'application_id': app_id, 'application_version_id': app_id * 10}}


def _call(tool, client):
    # Run the tool's lazy build and return the child's loaded tools (what _run would invoke).
    return tool._build_child(dict(tool.args_runnable), {})


def test_root_agent_is_on_the_load_path_so_a_cycle_back_to_it_is_cut(caplog):
    # A (root) -> B -> A: building B must skip A, which is the root.
    client = _RegistryClient({(1, 10): [_app_cfg(2)], (2, 20): [_app_cfg(1)]})

    [tool_b] = client.application(1, 10)
    b_children = _call(tool_b, client)

    assert b_children == []
    assert [r for r in caplog.records if 'Circular application reference' in r.message]
    assert tools_mod._APP_LOAD_STACK.get() == frozenset()


def test_long_chain_of_lazy_builds_still_trips_the_depth_backstop(caplog):
    # 1 -> 2 -> ... -> N, all distinct: walking the chain call by call must stop at the backstop.
    n = _MAX_APP_NESTING_BACKSTOP + 5
    client = _RegistryClient({(i, i * 10): [_app_cfg(i + 1)] for i in range(1, n)})

    level = client.application(1, 10)
    depth = 1
    while level:
        [tool] = level
        level = _call(tool, client)
        depth += 1

    assert depth <= _MAX_APP_NESTING_BACKSTOP + 1
    assert [r for r in caplog.records if 'reached backstop' in r.message]
    assert tools_mod._APP_LOAD_STACK.get() == frozenset()
