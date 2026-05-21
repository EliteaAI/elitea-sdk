"""Contract tests for the industry-standard Task toolkit (issue #4960).

Resolves:
- #4961: Smart Tool Selection no longer disabled when Application tools attached.
- #4949: Swarm sub-agent gets isolated thread_id derived from parent + tool_call_id.
- G2: Application result collapses to output string in parent's ToolMessage.
- TASK_DELEGATION_ADDON injection: prompt mentions sub-agent delegation when
  agent_tools are present.
- Per-tool failure isolation in __perform_tool_calling for Application tools.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.langchain.constants import TASK_DELEGATION_ADDON
from elitea_sdk.runtime.tools.application import Application


class DummyEliteARuntime:
    def get_mcp_toolkits(self):
        return []


class StaticApplication:
    def __init__(self, output='child-complete'):
        self.output = output
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append({'payload': payload, 'config': config})
        return {'output': self.output}


class FailingApplication:
    def invoke(self, payload, config=None):
        raise RuntimeError('child blew up')


class _ParentLLMBound:
    def __init__(self, root, tools):
        self.root = root
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        self.root.invoke_calls.append(list(messages))
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_messages:
            return AIMessage(content=f'parent saw: {tool_messages[-1].content}')
        return AIMessage(
            content='',
            tool_calls=[
                {
                    'name': self.root.target_tool_name,
                    'args': {'task': 'Run the child task'},
                    'id': self.root.tool_call_id,
                    'type': 'tool_call',
                }
            ],
        )


class ParentLLM:
    def __init__(self, target_tool_name='child_app', tool_call_id='call-child-1'):
        self.target_tool_name = target_tool_name
        self.tool_call_id = tool_call_id
        self.invoke_calls = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _ParentLLMBound(self, tools)

    def invoke(self, messages, config=None):
        return _ParentLLMBound(self, []).invoke(messages, config=config)


def _build_assistant(llm, tools, *, lazy_tools_mode=False, app_type='predict', instructions='Use tools'):
    return Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': instructions, 'tools': [], 'meta': {}},
        client=llm,
        tools=tools,
        memory=MemorySaver(),
        app_type=app_type,
        lazy_tools_mode=lazy_tools_mode,
    )


# --- #4961: Smart Tool Selection co-exists with Application tools -------------


def test_smart_tool_selection_works_with_application_tools_attached():
    """With Application tool attached AND enough other tools to clear the lazy-mode
    threshold, the resulting runnable must have a populated tool_registry AND the
    LLMNode-based graph topology. Pre-fix Application-bearing agents routed through
    _create_toolnode_react_agent which hardcoded tool_registry=None and produced
    a model/tools graph instead of agent/StateDefaultNode."""
    from langchain_core.tools import StructuredTool

    regular_tools = [
        StructuredTool.from_function(
            func=lambda x, _name=f'tool_{i}': f'{_name}: {x}',
            name=f'tool_{i}',
            description=f'utility tool number {i}',
        )
        for i in range(20)
    ]
    app_tool = Application(
        name='child_app',
        description='delegated worker',
        application=StaticApplication(),
        return_type='str',
        client=None,
        is_subgraph=True,
    )

    assistant = _build_assistant(ParentLLM(), regular_tools + [app_tool], lazy_tools_mode=True)
    runnable = assistant.runnable()

    assert runnable.tool_registry is not None, (
        'Smart Tool Selection disabled when Application is attached — #4961 regression'
    )
    nodes = runnable.get_graph().nodes
    assert 'agent' in nodes
    assert 'model' not in nodes and 'tools' not in nodes, (
        'Routing fell back into the deleted _create_toolnode_react_agent path'
    )


# --- #4949: Quarantined child thread_id ---------------------------------------


def test_swarm_subagent_gets_isolated_thread_id_derived_from_tool_call_id():
    """Application._run must derive child thread_id as f"{parent}:{tool_call_id}"
    so child checkpoints don't collide with parent's MemorySaver on second-turn
    swarm invocation. Resolves #4949."""
    captured = StaticApplication(output='ok')
    app = Application(
        name='child',
        description='child',
        application=captured,
        return_type='str',
        client=None,
        is_subgraph=True,
    )

    parent_config = {
        'configurable': {
            'thread_id': 'parent-thread-42',
            '_tool_call_id': 'call-child-7',
        }
    }
    app._run(task='do work', config=parent_config)

    assert captured.calls, 'child application was not invoked'
    child_config = captured.calls[0]['config']
    assert child_config['configurable']['thread_id'] == 'parent-thread-42:call-child-7'
    assert '_tool_call_id' not in child_config['configurable']


def test_swarm_subagent_falls_back_to_parent_thread_id_without_tool_call_id():
    """When invoked from a non-tool-calling path (e.g. pipeline FunctionTool),
    tool_call_id is None and the child should use the parent's thread_id directly.
    Preserves existing pipeline semantics."""
    captured = StaticApplication(output='ok')
    app = Application(
        name='child',
        description='child',
        application=captured,
        return_type='str',
        client=None,
        is_subgraph=True,
    )

    parent_config = {'configurable': {'thread_id': 'parent-thread-only'}}
    app._run(task='do work', config=parent_config)

    child_config = captured.calls[0]['config']
    assert child_config['configurable']['thread_id'] == 'parent-thread-only'


# --- G2: result collapse via LLMNode ------------------------------------------


def test_application_result_collapses_to_output_string_via_llmnode():
    """Parent's ToolMessage.content for an Application call must be the child's
    `output` string, not a stringified AgentResponse dict. Pre-fix LLMNode
    bypassed Application.invoke's collapse path. Verifies G2."""
    parent_llm = ParentLLM(target_tool_name='child_app', tool_call_id='call-collapse-1')
    app_tool = Application(
        name='child_app',
        description='child',
        application=StaticApplication(output='child-graph-complete'),
        return_type='str',
        client=None,
        is_subgraph=True,
    )

    runnable = _build_assistant(parent_llm, [app_tool]).runnable()
    runnable.invoke(
        {'messages': [HumanMessage(content='Delegate this task')]},
        config={'configurable': {'thread_id': 'collapse-thread'}},
    )

    follow_up_messages = [
        msgs for msgs in parent_llm.invoke_calls
        if any(isinstance(m, ToolMessage) for m in msgs)
    ]
    assert follow_up_messages, 'parent LLM was never called with a ToolMessage in scope'
    tool_messages = [m for m in follow_up_messages[0] if isinstance(m, ToolMessage)]
    assert tool_messages[0].content == 'child-graph-complete'


# --- Failure isolation --------------------------------------------------------


def test_subagent_exception_yields_error_toolmessage_and_parent_continues():
    """When a sub-agent raises, the parent receives an error ToolMessage
    (not a propagated exception) and the LLM is re-invoked so it can react.
    Validates the existing for-loop's per-tool exception handling
    (tools/llm.py:2274-2283) still works through the LLMNode path post-Step-3."""
    parent_llm = ParentLLM(target_tool_name='broken_child', tool_call_id='call-fail-1')
    app_tool = Application(
        name='broken_child',
        description='child that fails',
        application=FailingApplication(),
        return_type='str',
        client=None,
        is_subgraph=True,
    )

    runnable = _build_assistant(parent_llm, [app_tool]).runnable()
    runnable.invoke(
        {'messages': [HumanMessage(content='Delegate this task')]},
        config={'configurable': {'thread_id': 'fail-thread'}},
    )

    follow_up = [
        msgs for msgs in parent_llm.invoke_calls
        if any(isinstance(m, ToolMessage) for m in msgs)
    ]
    assert follow_up, 'parent LLM did not see the error ToolMessage'
    error_msg = next(m for m in follow_up[0] if isinstance(m, ToolMessage))
    assert 'broken_child' in error_msg.content or 'Error' in error_msg.content
    assert error_msg.tool_call_id == 'call-fail-1'


# --- #4949 (chat / nested-agent path): swarm-as-subagent input normalization --


def test_swarm_subagent_invoked_via_application_input_translates_to_messages():
    """When a swarm-mode child agent is invoked as an Application tool from a parent,
    Application._run/formulate_query passes {"input": [HumanMessage(...)]}. The swarm
    graph uses MessagesState (key "messages") — without translation, state["messages"]
    stays empty and the peer's agent_node calls Anthropic with [SystemMessage] only,
    raising 500 'messages is required'. This is the failure mode reported in #4949
    (chat-execution + nested-agent paths)."""
    from unittest.mock import MagicMock
    from langgraph.checkpoint.memory import MemorySaver

    child_app = Application(
        name='child',
        description='child',
        application=StaticApplication(),
        return_type='str',
        client=None,
        is_subgraph=True,
    )

    swarm_assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={
            'instructions': 'You are the main agent',
            'tools': [],
            'meta': {'internal_tools': ['swarm']},
            'internal_tools': ['swarm'],
        },
        client=MagicMock(),
        tools=[child_app],
        memory=MemorySaver(),
        app_type='agent',
    )

    adapter = swarm_assistant.runnable()

    captured = {}
    real_graph = adapter._graph

    def capturing_invoke(input, config=None, **kwargs):
        captured['input'] = input
        # Don't actually run the swarm graph — just capture and return a minimal result
        return {'messages': [HumanMessage(content='captured')]}

    adapter._graph = MagicMock()
    adapter._graph.invoke = capturing_invoke
    adapter._graph.get_state = real_graph.get_state

    adapter.invoke(
        {'input': [HumanMessage(content='Run the child task')]},
        config={'configurable': {'thread_id': 'swarm-as-subagent-thread'}},
    )

    assert captured['input'].get('messages'), (
        'SwarmResultAdapter did not translate input["input"] to input["messages"] — '
        'swarm peer agent_node would receive empty state["messages"] and Anthropic '
        'would crash with "messages is required" (#4949 chat/nested path)'
    )
    msgs = captured['input']['messages']
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == 'Run the child task'


def test_swarm_subagent_input_translation_handles_string_task():
    """When Application is_subgraph=True (pipeline-shaped child), formulate_query
    passes input as a plain string. The adapter must wrap it in a HumanMessage."""
    from unittest.mock import MagicMock
    from langgraph.checkpoint.memory import MemorySaver

    child_app = Application(
        name='child', description='child', application=StaticApplication(),
        return_type='str', client=None, is_subgraph=True,
    )
    swarm_assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'main', 'tools': [], 'meta': {'internal_tools': ['swarm']},
              'internal_tools': ['swarm']},
        client=MagicMock(),
        tools=[child_app],
        memory=MemorySaver(),
        app_type='agent',
    )
    adapter = swarm_assistant.runnable()

    captured = {}
    adapter._graph = MagicMock()
    adapter._graph.invoke = lambda inp, cfg=None, **k: captured.setdefault('input', inp) or {'messages': []}
    adapter._graph.get_state = lambda cfg: None

    adapter.invoke({'input': 'plain task string'}, config={'configurable': {'thread_id': 't'}})

    msgs = captured['input'].get('messages') or []
    assert msgs, 'string input was not translated to a HumanMessage list'
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == 'plain task string'


# --- TASK_DELEGATION_ADDON injection ------------------------------------------


def test_task_delegation_addon_injected_when_agent_tools_present():
    """The Sub-agent delegation prompt section must appear in the compiled system
    prompt when the agent has any Application tool attached, and must be absent
    when no Application tools are attached."""
    from langchain_core.tools import StructuredTool
    regular_tool = StructuredTool.from_function(
        func=lambda x: x, name='regular', description='regular tool',
    )
    app_tool = Application(
        name='child', description='child', application=StaticApplication(),
        return_type='str', client=None, is_subgraph=True,
    )

    captured = {}

    class CapturingLLM(ParentLLM):
        def bind_tools(self, tools, **kwargs):
            return _CapturingBound(self, tools)

    class _CapturingBound(_ParentLLMBound):
        def invoke(self, messages, config=None):
            for m in messages:
                if isinstance(m, SystemMessage):
                    captured.setdefault('system_prompts', []).append(m.content)
            return super().invoke(messages, config)

    # With Application tool → addon present
    assistant_with = _build_assistant(
        CapturingLLM(target_tool_name='child', tool_call_id='id-1'),
        [regular_tool, app_tool],
        instructions='You are a coordinator.',
    )
    assistant_with.runnable().invoke(
        {'messages': [HumanMessage(content='go')]},
        config={'configurable': {'thread_id': 'addon-with'}},
    )
    assert any(
        'Sub-agent delegation' in p and 'task: str' in p
        for p in captured.get('system_prompts', [])
    ), 'TASK_DELEGATION_ADDON missing when Application tool is attached'

    # Without Application tool → addon absent
    captured.clear()
    assistant_without = _build_assistant(
        CapturingLLM(target_tool_name='regular', tool_call_id='id-2'),
        [regular_tool],
        instructions='You are a coordinator.',
    )
    assistant_without.runnable().invoke(
        {'messages': [HumanMessage(content='go')]},
        config={'configurable': {'thread_id': 'addon-without'}},
    )
    assert not any(
        'Sub-agent delegation' in p
        for p in captured.get('system_prompts', [])
    ), 'TASK_DELEGATION_ADDON should not appear when no Application tools are attached'
