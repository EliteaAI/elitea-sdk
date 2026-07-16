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


class DictBridgeInterruptingApplication:
    """Child whose inner graph ABSORBS the sensitive-tool interrupt and RETURNS
    it in state (the dict-bridge path that a real standalone
    LangGraphAgentRunnable takes), rather than letting interrupt() raise.

    This is the only child shape compatible with the parallel deferred-interrupt
    aggregation (issue #4993): the executor thread must get a returned sentinel,
    not a raised GraphInterrupt that asyncio.gather would capture.
    """

    def __init__(self, output, tool_name):
        self.output = output
        self.tool_name = tool_name
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append({'payload': payload, 'config': config})
        if isinstance(payload, dict) and payload.get('hitl_resume'):
            return {'output': self.output, 'execution_finished': True}
        return {
            'output': 'Need approval',
            'execution_finished': False,
            'hitl_interrupt': {
                'type': 'hitl',
                'guardrail_type': 'sensitive_tool',
                'message': f'approve {self.tool_name}?',
                'tool_name': self.tool_name,
            },
        }


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


def test_subagent_gets_isolated_thread_id_derived_from_child_name():
    """Application._run derives child thread_id as f"{parent}:{child_name}".
    Stable across parent turns (multi-turn child history works) and namespace-
    isolated from parent (no stale-mixing — #4949)."""
    captured = StaticApplication(output='ok')
    app = Application(
        name='analyst',
        description='child',
        application=captured,
        return_type='str',
        client=None,
        is_subgraph=True,
    )

    parent_config = {'configurable': {'thread_id': 'parent-chat-42'}}
    app._run(task='do work', config=parent_config)

    assert captured.calls, 'child application was not invoked'
    child_config = captured.calls[0]['config']
    assert child_config['configurable']['thread_id'] == 'parent-chat-42:analyst'


def test_subagent_thread_id_stable_across_parent_turns():
    """Multi-turn parent conversation must produce the SAME child thread_id on every
    turn — otherwise a swarm child loses its conversation history between parent
    turns (the reason we don't key on tool_call_id)."""
    captured = StaticApplication(output='ok')
    app = Application(
        name='researcher',
        description='child',
        application=captured,
        return_type='str',
        client=None,
        is_subgraph=True,
    )
    parent_config = {'configurable': {'thread_id': 'parent-chat-42'}}

    # Two separate parent turns, two separate _run invocations
    app._run(task='turn 1 task', config=parent_config)
    app._run(task='turn 2 task', config=parent_config)

    thread_ids = [call['config']['configurable']['thread_id'] for call in captured.calls]
    assert thread_ids == ['parent-chat-42:researcher', 'parent-chat-42:researcher'], (
        'child thread_id must be stable across parent turns to preserve multi-turn '
        'history (especially for swarm children)'
    )


def test_two_different_subagents_get_isolated_thread_ids():
    """Two different Application children attached to the same parent must get
    distinct thread_ids — they are independent agents and shouldn't share state."""
    a = StaticApplication(output='a')
    b = StaticApplication(output='b')
    app_a = Application(name='analyst', description='', application=a,
                        return_type='str', client=None, is_subgraph=True)
    app_b = Application(name='writer', description='', application=b,
                        return_type='str', client=None, is_subgraph=True)
    parent_config = {'configurable': {'thread_id': 'shared-parent'}}

    app_a._run(task='analyze', config=parent_config)
    app_b._run(task='write', config=parent_config)

    assert a.calls[0]['config']['configurable']['thread_id'] == 'shared-parent:analyst'
    assert b.calls[0]['config']['configurable']['thread_id'] == 'shared-parent:writer'


def test_sequential_application_call_stamps_non_null_ancestry_call_id():
    """A single Application call still carries its wrapper tool-call identity.

    The ancestry contract must not depend on the parallel-only private config
    key; sequential nested UI events need the same stable invocation key.
    """
    captured = StaticApplication(output='ok')
    app = Application(
        name='analyst', description='child', application=captured,
        return_type='str', client=None, is_subgraph=True,
    )

    app.invoke(
        {'type': 'tool_call', 'name': 'analyst', 'id': 'call-sequential',
         'args': {'task': 'analyze'}},
        config={'configurable': {'thread_id': 'parent-thread'}},
    )

    metadata = captured.calls[0]['config']['metadata']
    assert metadata['parent_agent_call_id'] == 'call-sequential'
    assert metadata['parent_agent_path'] == [{
        'name': 'analyst', 'call_id': 'call-sequential',
    }]


def test_parallel_sibling_ordinal_is_consumed_into_path_not_inherited():
    captured = StaticApplication(output='ok')
    app = Application(
        name='analyst', description='child', application=captured,
        return_type='str', client=None, is_subgraph=True,
    )

    app.invoke(
        {'type': 'tool_call', 'name': 'analyst', 'id': 'call-parallel',
         'args': {'task': 'analyze'}},
        config={
            'configurable': {'thread_id': 'parent-thread'},
            'metadata': {'sibling_ordinal': 3},
        },
    )

    metadata = captured.calls[0]['config']['metadata']
    assert metadata['parent_agent_path'][-1]['sibling_ordinal'] == 3
    assert 'sibling_ordinal' not in metadata


def test_application_rebuilds_runnable_per_invocation_without_shared_mutation():
    """Dynamic child variables stay invocation-local under concurrent use."""
    from concurrent.futures import ThreadPoolExecutor
    import threading

    created = []
    created_lock = threading.Lock()

    class InvocationRunnable:
        def __init__(self, task):
            self.task = task

        def invoke(self, payload, config=None):
            return {'output': self.task}

    class DynamicClient:
        def application(self, **kwargs):
            task = kwargs['application_variables']['task']['value']
            runnable = InvocationRunnable(task)
            with created_lock:
                created.append(runnable)
            return runnable

    original = StaticApplication(output='must-remain-installed')
    app = Application(
        name='dynamic_child', description='child', application=original,
        return_type='str', client=DynamicClient(), is_subgraph=True,
        args_runnable={'application_id': 1},
    )
    tasks = [f'task-{index}' for index in range(16)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(
            lambda task: app._run(
                task=task,
                config={'configurable': {'thread_id': f'parent-{task}'}},
            )['output'],
            tasks,
        ))

    assert outputs == tasks
    assert len(created) == len(tasks)
    assert app.application is original


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


# --- Swarm handoff tool: empty-args tolerance (Anthropic compatibility) -------


def test_swarm_handoff_tool_schema_tolerates_missing_task():
    """Anthropic production crash: LLM emits transfer_to_* tool_call with empty
    args. After LangGraph injects state and tool_call_id, BaseTool._parse_input
    runs args_schema.model_validate({state, tool_call_id}) — no task. Pre-fix
    that raised ValidationError; post-fix task defaults to '' and validation
    passes. The receiving peer's _extract_task_from_assigning_tool_call still
    reads whatever the LLM emitted on the assigning AIMessage."""
    from elitea_sdk.runtime.langchain.assistant import _create_swarm_handoff_tool

    handoff = _create_swarm_handoff_tool(agent_name='GuidedDeveloper')
    schema = handoff.args_schema

    # Exact production failure shape: state + tool_call_id injected, task missing.
    instance = schema.model_validate({
        'state': {'messages': [HumanMessage(content='go')]},
        'tool_call_id': 'L3BqyxmeaWn6Vf0gIrKclJ',
    })
    assert getattr(instance, 'task', None) == ''

    # OpenAI / well-behaved LLM: task is provided as expected.
    instance = schema.model_validate({
        'state': {'messages': []},
        'tool_call_id': 'call-with-task-1',
        'task': 'Find papers about X and return DOIs as JSON',
    })
    assert instance.task == 'Find papers about X and return DOIs as JSON'


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
    addon_prompts = [
        p for p in captured.get('system_prompts', [])
        if 'Sub-agent delegation' in p and 'task: str' in p
    ]
    assert addon_prompts, 'TASK_DELEGATION_ADDON missing when Application tool is attached'
    # Anti-bias guidance: the LLM should prefer toolkit tools for direct actions
    # and delegate only when the sub-agent's domain matches. Without this nudge,
    # an LLM with a heterogeneous surface (registry-gated toolkit + direct-bound
    # sub-agent) tends to over-delegate because the sub-agent path is 1-step.
    assert any('Prefer toolkit tools' in p for p in addon_prompts), (
        'TASK_DELEGATION_ADDON should steer the LLM toward toolkit tools for direct '
        'actions and delegate only when the sub-agent description matches'
    )

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


# --- #4993: parallel sub-agent fan-out + multi-interrupt HITL -----------------


class _MultiAppParentBound:
    def __init__(self, root, tools):
        self.root = root
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        self.root.invoke_calls.append(list(messages))
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_messages:
            joined = ' | '.join(str(m.content) for m in tool_messages)
            return AIMessage(content=f'parent saw: {joined}')
        return AIMessage(
            content='',
            tool_calls=[
                {'name': self.root.tool_a, 'args': {'task': 'Run A'},
                 'id': self.root.id_a, 'type': 'tool_call'},
                {'name': self.root.tool_b, 'args': {'task': 'Run B'},
                 'id': self.root.id_b, 'type': 'tool_call'},
            ],
        )


class MultiAppParentLLM:
    """Parent LLM that fans out: emits TWO Application tool_calls in one
    assistant message, then finishes once both ToolMessages are in scope."""

    def __init__(self, tool_a='child_a', tool_b='child_b',
                 id_a='call-A', id_b='call-B'):
        self.tool_a = tool_a
        self.tool_b = tool_b
        self.id_a = id_a
        self.id_b = id_b
        self.invoke_calls = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _MultiAppParentBound(self, tools)

    def invoke(self, messages, config=None):
        return _MultiAppParentBound(self, []).invoke(messages, config=config)


def _app(name, application, **kw):
    return Application(
        name=name, description=kw.pop('description', f'{name} worker'),
        application=application, return_type='str', client=None,
        is_subgraph=True, **kw,
    )


# --- deferred-interrupt mode (the parallel-safe child contract) ---------------


def test_deferred_mode_returns_sentinel_instead_of_raising():
    """In deferred mode a paused child must RETURN a sentinel (not call
    interrupt()): inside an asyncio.gather executor thread a raised
    GraphInterrupt is captured as a plain exception and the pause is lost."""
    child = DictBridgeInterruptingApplication(output='ok', tool_name='create_file')
    app = _app('worker', child)

    result = app._run(
        task='do work',
        config={'configurable': {
            'thread_id': 'parent-1',
            '__hitl_deferred_mode__': True,
            '__hitl_parallel_call_id__': 'call-X',
        }},
    )

    assert isinstance(result, dict) and result.get('__hitl_deferred__') is True
    assert result['tool_call_id'] == 'call-X'
    assert result['hitl_interrupt']['tool_name'] == 'create_file'
    assert result['hitl_interrupt']['_parent_tool_name'] == 'worker'
    assert 'nested_config' in result


def test_deferred_resume_invokes_child_with_resume_payload():
    """On the parallel resume path the child is re-invoked from its checkpoint
    with {hitl_resume, hitl_action, hitl_value} instead of a fresh task."""
    child = DictBridgeInterruptingApplication(output='approved-result', tool_name='create_file')
    app = _app('worker', child)

    result = app._run(
        task='do work',
        config={'configurable': {
            'thread_id': 'parent-1',
            '__hitl_parallel_call_id__': 'call-X',
            '__hitl_parallel_resume__': {'action': 'approve', 'value': 'go'},
        }},
    )

    # Child got the resume envelope, not a fresh task.
    resume_payload = child.calls[-1]['payload']
    assert resume_payload.get('hitl_resume') is True
    assert resume_payload.get('hitl_action') == 'approve'
    assert resume_payload.get('hitl_value') == 'go'
    assert result['output'] == 'approved-result'


def test_parallel_call_id_isolates_same_name_sibling_thread_ids():
    """Two parallel calls to the SAME sub-agent must get distinct child
    thread_ids — the parallel path suffixes with tool_call_id so the two
    checkpoints don't collide. The sequential/single path is unchanged
    (covered by the existing :name-only thread_id tests)."""
    captured = StaticApplication(output='ok')
    app = _app('analyst', captured)

    app._run(task='a', config={'configurable': {
        'thread_id': 'parent-chat-42', '__hitl_parallel_call_id__': 'call-A'}})
    app._run(task='b', config={'configurable': {
        'thread_id': 'parent-chat-42', '__hitl_parallel_call_id__': 'call-B'}})

    thread_ids = [c['config']['configurable']['thread_id'] for c in captured.calls]
    assert thread_ids == ['parent-chat-42:analyst:call-A',
                          'parent-chat-42:analyst:call-B']


# --- _collect_parallel_application_specs partition ----------------------------


def _bare_llmnode(tools):
    from elitea_sdk.runtime.tools.llm import LLMNode
    return LLMNode(client=None, available_tools=list(tools), lazy_tools_mode=False)


def test_collect_specs_pure_application_batch_returns_specs():
    node = _bare_llmnode([_app('child_a', StaticApplication()),
                          _app('child_b', StaticApplication())])
    tool_calls = [
        {'name': 'child_a', 'args': {'task': 'x'}, 'id': 'call-A'},
        {'name': 'child_b', 'args': {'task': 'y'}, 'id': 'call-B'},
    ]
    specs = node._collect_parallel_application_specs(
        tool_calls, [], {'configurable': {}})
    assert specs is not None and len(specs) == 2
    assert {s[2] for s in specs} == {'call-A', 'call-B'}


def test_collect_specs_mixed_batch_stays_sequential():
    """A turn mixing an Application call with a regular toolkit call must NOT
    fan out — returns None so the sequential loop runs."""
    from langchain_core.tools import StructuredTool
    regular = StructuredTool.from_function(
        func=lambda x: x, name='regular', description='regular tool')
    node = _bare_llmnode([_app('child_a', StaticApplication()), regular])
    tool_calls = [
        {'name': 'child_a', 'args': {'task': 'x'}, 'id': 'call-A'},
        {'name': 'regular', 'args': {'x': '1'}, 'id': 'call-R'},
    ]
    assert node._collect_parallel_application_specs(
        tool_calls, [], {'configurable': {}}) is None


def test_collect_specs_single_application_stays_sequential():
    node = _bare_llmnode([_app('child_a', StaticApplication())])
    tool_calls = [{'name': 'child_a', 'args': {'task': 'x'}, 'id': 'call-A'}]
    assert node._collect_parallel_application_specs(
        tool_calls, [], {'configurable': {}}) is None


def test_collect_specs_resume_single_remaining_with_decision():
    """Resume edge case: one sibling completed (its ToolMessage restored), the
    other paused. The lone remaining call was checkpointed under the suffixed
    thread_id, so it MUST stay on the parallel path even as a 1-spec batch —
    detected by matching the resume hitl_decisions."""
    node = _bare_llmnode([_app('child_a', StaticApplication()),
                          _app('child_b', StaticApplication())])
    tool_calls = [
        {'name': 'child_a', 'args': {'task': 'x'}, 'id': 'call-A'},
        {'name': 'child_b', 'args': {'task': 'y'}, 'id': 'call-B'},
    ]
    completed = [ToolMessage(content='A-done', tool_call_id='call-A')]
    specs = node._collect_parallel_application_specs(
        tool_calls, completed, {'configurable': {}},
        hitl_decisions=[{'tool_call_id': 'call-B', 'action': 'approve'}],
    )
    assert specs is not None and len(specs) == 1 and specs[0][2] == 'call-B'


# --- end-to-end fan-out through the full parent graph -------------------------


def test_two_application_calls_fan_out_and_both_complete():
    """Two independent Application calls in one turn run concurrently and both
    results return to the parent. Distinct suffixed child thread_ids confirm
    the parallel path (not the sequential :name-only path) executed."""
    child_a = StaticApplication(output='A-done')
    child_b = StaticApplication(output='B-done')
    app_a = _app('child_a', child_a)
    app_b = _app('child_b', child_b)

    parent = MultiAppParentLLM(tool_a='child_a', tool_b='child_b',
                               id_a='call-A', id_b='call-B')
    runnable = _build_assistant(parent, [app_a, app_b]).runnable()
    runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both')]},
        config={'configurable': {'thread_id': 'fan-out-thread'}},
    )

    assert child_a.calls and child_b.calls, 'both children must run'
    assert child_a.calls[0]['config']['configurable']['thread_id'] == \
        'fan-out-thread:child_a:call-A'
    assert child_b.calls[0]['config']['configurable']['thread_id'] == \
        'fan-out-thread:child_b:call-B'

    follow_up = [msgs for msgs in parent.invoke_calls
                 if any(isinstance(m, ToolMessage) for m in msgs)]
    assert follow_up, 'parent must be re-invoked with both ToolMessages'
    contents = {m.content for m in follow_up[0] if isinstance(m, ToolMessage)}
    assert contents == {'A-done', 'B-done'}


def test_parallel_fan_out_runs_children_concurrently():
    """Concurrency proof via a shared barrier: each child blocks until BOTH
    have entered. Sequential execution would never release the barrier (the
    first child waits forever) → BrokenBarrierError → error ToolMessages.
    Concurrent execution lets both pass and return their outputs."""
    import threading

    barrier = threading.Barrier(2, timeout=8)

    class BarrierApplication:
        def __init__(self, output):
            self.output = output
            self.calls = []

        def invoke(self, payload, config=None):
            self.calls.append({'payload': payload, 'config': config})
            barrier.wait()  # only returns if a sibling reached it concurrently
            return {'output': self.output}

    child_a = BarrierApplication('A-done')
    child_b = BarrierApplication('B-done')
    parent = MultiAppParentLLM(tool_a='child_a', tool_b='child_b',
                               id_a='call-A', id_b='call-B')
    runnable = _build_assistant(
        parent, [_app('child_a', child_a), _app('child_b', child_b)]).runnable()
    runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both')]},
        config={'configurable': {'thread_id': 'concurrent-thread'}},
    )

    follow_up = [msgs for msgs in parent.invoke_calls
                 if any(isinstance(m, ToolMessage) for m in msgs)]
    assert follow_up, 'children did not run concurrently (barrier never released)'
    contents = {m.content for m in follow_up[0] if isinstance(m, ToolMessage)}
    assert contents == {'A-done', 'B-done'}


def test_one_child_fails_does_not_cancel_sibling():
    """When one parallel sub-agent raises, the sibling still completes and the
    parent receives an error ToolMessage for the failure + the good result for
    the sibling."""
    good = StaticApplication(output='B-done')
    parent = MultiAppParentLLM(tool_a='child_a', tool_b='child_b',
                               id_a='call-A', id_b='call-B')
    runnable = _build_assistant(
        parent, [_app('child_a', FailingApplication()),
                 _app('child_b', good)]).runnable()
    runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both')]},
        config={'configurable': {'thread_id': 'fail-isolation-thread'}},
    )

    assert good.calls, 'sibling must still run when one child fails'
    follow_up = [msgs for msgs in parent.invoke_calls
                 if any(isinstance(m, ToolMessage) for m in msgs)]
    assert follow_up
    by_id = {m.tool_call_id: m for m in follow_up[0] if isinstance(m, ToolMessage)}
    assert by_id['call-B'].content == 'B-done'
    assert 'Error' in by_id['call-A'].content or 'child blew up' in by_id['call-A'].content
