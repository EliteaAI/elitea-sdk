from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.langchain.langraph_agent import LangGraphAgentRunnable
from elitea_sdk.runtime.middleware.sensitive_tool_guard import SensitiveToolGuardMiddleware
from elitea_sdk.runtime.tools.application import Application
from elitea_sdk.runtime.toolkits.application import ApplicationToolkit
from elitea_sdk.runtime.toolkits.security import configure_sensitive_tools, reset_sensitive_tools


class DummyEliteARuntime:
    def get_mcp_toolkits(self):
        return []


class ParentLLM:
    def __init__(self, target_tool_name='child_two'):
        self.target_tool_name = target_tool_name
        self.calls = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _ParentLLMBound(self, tools)

    def invoke(self, messages, config=None):
        return _ParentLLMBound(self, []).invoke(messages, config=config)


class _ParentLLMBound:
    def __init__(self, root, tools):
        self.root = root
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        tool_contents = [str(message.content) for message in tool_messages]
        self.root.calls.append(
            {
                'tool_contents': tool_contents,
                'bound_tools': [tool.name for tool in self.tools],
            }
        )

        if 'child-two-complete' in tool_contents:
            return AIMessage(content='Parent task completed')

        return AIMessage(
            content='',
            tool_calls=[
                {
                    'name': self.root.target_tool_name,
                    'args': {'task': 'Run the child task'},
                    'id': 'call-child-two',
                    'type': 'tool_call',
                }
            ],
        )


class InterruptingApplication:
    def __init__(self, output='child-two-complete'):
        self.output = output
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append({'payload': payload, 'config': config})
        review = interrupt(
            {
                'type': 'hitl',
                'guardrail_type': 'sensitive_tool',
                'message': 'Need approval',
                'tool_name': 'create_file',
            }
        )

        if isinstance(review, dict) and review.get('action') == 'approve':
            return {'output': self.output}

        return {'output': f'unexpected review payload: {review}'}


class StaticApplication:
    def __init__(self, output='static-output'):
        self.output = output
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append({'payload': payload, 'config': config})
        return {'output': self.output}


class FakeToolkitClient:
    def __init__(self):
        self.project_id = 7
        self.application_calls = []

    def get_app_details(self, application_id):
        return {'name': f'Child {application_id}', 'description': 'child app'}

    def get_app_version_details(self, application_id, application_version_id):
        return {
            'variables': [],
            'meta': {},
            'llm_settings': {
                'model_name': 'fake-model',
                'max_tokens': 1000,
                'temperature': 0,
                'reasoning_effort': None,
            },
        }

    def get_llm(self, model_name, model_settings):
        return {'model_name': model_name, 'model_settings': model_settings}

    def application(self, *args, **kwargs):
        self.application_calls.append({'args': args, 'kwargs': dict(kwargs)})
        return StaticApplication(output='nested-child-output')


class ParentResultAwareLLM:
    def __init__(self, target_tool_name='child_graph'):
        self.target_tool_name = target_tool_name
        self.calls = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _ParentResultAwareLLMBound(self, tools)

    def invoke(self, messages, config=None):
        return _ParentResultAwareLLMBound(self, []).invoke(messages, config=config)


class _ParentResultAwareLLMBound:
    def __init__(self, root, tools):
        self.root = root
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        tool_contents = [str(message.content) for message in tool_messages]
        self.root.calls.append(tool_contents)

        if tool_contents == ['child-graph-complete']:
            return AIMessage(content='parent-graph-complete')

        if tool_contents:
            return AIMessage(content=f'unexpected child result: {tool_contents[0]}')

        return AIMessage(
            content='',
            tool_calls=[
                {
                    'name': self.root.target_tool_name,
                    'args': {'task': 'Run the child task'},
                    'id': 'call-child-graph',
                    'type': 'tool_call',
                }
            ],
        )


class ChildToolCallingLLM:
    temperature = 0
    max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _ChildToolCallingLLMBound(tools)

    def invoke(self, messages, config=None):
        return _ChildToolCallingLLMBound([]).invoke(messages, config=config)


class _ChildToolCallingLLMBound:
    def __init__(self, tools):
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content='child-graph-complete')

        return AIMessage(
            content='',
            tool_calls=[
                {
                    'name': 'create_file',
                    'args': {'path': '/tmp/test.txt'},
                    'id': 'call-child-tool',
                    'type': 'tool_call',
                }
            ],
        )


class RejectAwareParentLLM:
    def __init__(self):
        self.calls = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _RejectAwareParentLLMBound(self, tools)

    def invoke(self, messages, config=None):
        return _RejectAwareParentLLMBound(self, []).invoke(messages, config=config)


class _RejectAwareParentLLMBound:
    def __init__(self, root, tools):
        self.root = root
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        tool_contents = [str(message.content) for message in tool_messages]
        self.root.calls.append(tool_contents)

        if tool_contents:
            return AIMessage(content=f'parent-sees:{tool_contents[-1]}')

        return AIMessage(
            content='',
            tool_calls=[
                {
                    'name': self.tools[0].name,
                    'args': {'task': 'Run the child task'},
                    'id': 'call-child-graph-reject',
                    'type': 'tool_call',
                }
            ],
        )


class RejectAwareChildLLM:
    temperature = 0
    max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _RejectAwareChildLLMBound(tools)

    def invoke(self, messages, config=None):
        return _RejectAwareChildLLMBound([]).invoke(messages, config=config)


class _RejectAwareChildLLMBound:
    def __init__(self, tools):
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        blocked_messages = [
            str(message.content)
            for message in messages
            if isinstance(message, ToolMessage)
        ]
        if blocked_messages:
            return AIMessage(content='child-reject-finished')

        return AIMessage(
            content='',
            tool_calls=[
                {
                    'name': 'create_file',
                    'args': {'path': '/tmp/test.txt'},
                    'id': 'call-child-tool-reject',
                    'type': 'tool_call',
                }
            ],
        )


class PendingAwareParentLLM:
    def __init__(self):
        self.calls = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _PendingAwareParentLLMBound(self, tools)

    def invoke(self, messages, config=None):
        return _PendingAwareParentLLMBound(self, []).invoke(messages, config=config)


class _PendingAwareParentLLMBound:
    def __init__(self, root, tools):
        self.root = root
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        tool_contents = [str(message.content) for message in tool_messages]
        self.root.calls.append(tool_contents)

        if tool_contents == ['child-finished']:
            return AIMessage(content='parent-done')

        if tool_contents:
            return AIMessage(content=f'unexpected parent tool result: {tool_contents[-1]}')

        return AIMessage(
            content='',
            tool_calls=[
                {
                    'name': self.tools[0].name,
                    'args': {'task': 'Run the child task'},
                    'id': 'call-child-graph-pending',
                    'type': 'tool_call',
                }
            ],
        )


class PendingAwareChildLLM:
    temperature = 0
    max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _PendingAwareChildLLMBound(tools)

    def invoke(self, messages, config=None):
        return _PendingAwareChildLLMBound([]).invoke(messages, config=config)


class _PendingAwareChildLLMBound:
    def __init__(self, tools):
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        tool_contents = [str(message.content) for message in tool_messages]

        if not tool_messages:
            return AIMessage(
                content='',
                tool_calls=[
                    {
                        'name': 'list_files',
                        'args': {},
                        'id': 'call-safe-1',
                        'type': 'tool_call',
                    },
                    {
                        'name': 'get_issues',
                        'args': {},
                        'id': 'call-safe-2',
                        'type': 'tool_call',
                    },
                    {
                        'name': 'search_issues',
                        'args': {},
                        'id': 'call-safe-3',
                        'type': 'tool_call',
                    },
                ],
            )

        if tool_contents == ['safe-list', 'safe-issues', 'safe-search']:
            return AIMessage(
                content='',
                tool_calls=[
                    {
                        'name': 'create_issue',
                        'args': {},
                        'id': 'call-sensitive-1',
                        'type': 'tool_call',
                    },
                    {
                        'name': 'create_issue',
                        'args': {},
                        'id': 'call-sensitive-2',
                        'type': 'tool_call',
                    },
                    {
                        'name': 'create_issue',
                        'args': {},
                        'id': 'call-sensitive-3',
                        'type': 'tool_call',
                    },
                ],
            )

        if 'created-issue' in tool_contents:
            return AIMessage(content='child-finished')

        return AIMessage(content=f'unexpected child tool history: {tool_contents}')



def _build_parent_runnable(memory, llm, tools):
    assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'Use tools', 'tools': [], 'meta': {}},
        client=llm,
        tools=tools,
        memory=memory,
        app_type='predict',
    )
    return assistant.runnable()



def test_multiple_application_tools_use_toolnode_runtime_and_resume_hitl():
    parent_memory = MemorySaver()
    child_one = StaticApplication(output='child-one-complete')
    child_two = InterruptingApplication(output='child-two-complete')

    tools = [
        Application(
            name='child_one',
            description='First child agent',
            application=child_one,
            return_type='str',
            client=None,
            is_subgraph=True,
        ),
        Application(
            name='child_two',
            description='Second child agent',
            application=child_two,
            return_type='str',
            client=None,
            is_subgraph=True,
        ),
    ]

    thread_config = {'configurable': {'thread_id': 'toolnode-nested-hitl-thread'}}

    initial_llm = ParentLLM(target_tool_name='child_two')
    initial_runnable = _build_parent_runnable(parent_memory, initial_llm, tools)
    graph = initial_runnable.get_graph()

    assert 'agent' in graph.nodes

    initial_result = initial_runnable.invoke(
        {'messages': [HumanMessage(content='Delegate this task')]},
        config=thread_config,
    )

    assert initial_result['execution_finished'] is False
    assert initial_result['output'] == 'Need approval'
    assert initial_result['hitl_interrupt']['tool_name'] == 'create_file'
    assert child_one.calls == []
    assert len(child_two.calls) == 1

    resumed_llm = ParentLLM(target_tool_name='child_two')
    resumed_runnable = _build_parent_runnable(parent_memory, resumed_llm, tools)
    resume_result = resumed_runnable.invoke(
        {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''},
        config=thread_config,
    )

    assert resume_result['execution_finished'] is True
    assert resume_result['output'] == 'Parent task completed'
    assert len(child_two.calls) == 2



def test_application_run_forwards_parent_checkpoint_context():
    nested = StaticApplication(output='ok')
    application_tool = Application(
        name='child_agent',
        description='Nested agent',
        application=nested,
        return_type='str',
        client=None,
        is_subgraph=True,
        metadata={'display_name': 'child_agent'},
    )

    application_tool.invoke(
        {'task': 'Run nested app', 'chat_history': []},
        config={
            'metadata': {'origin': 'parent'},
            'configurable': {
                'thread_id': 'parent-thread',
                'checkpoint_ns': 'parent-ns',
                'checkpoint_id': 'parent-cp',
                'selected_tools': ['should-be-removed'],
            },
        },
    )

    assert len(nested.calls) == 1
    nested_config = nested.calls[0]['config']
    assert nested_config['metadata']['origin'] == 'parent'
    assert nested_config['metadata']['parent_agent_name'] == 'child_agent'
    # Child gets its own thread_id namespace derived from parent + child name —
    # stable across parent turns (multi-turn child history works), isolated
    # from parent (no stale-mixing — #4949). See test_application_task_toolkit.
    assert nested_config['configurable']['thread_id'] == 'parent-thread:child_agent'
    assert nested_config['configurable']['checkpoint_ns'] == 'parent-ns'
    assert nested_config['configurable']['checkpoint_id'] == 'parent-cp'
    assert 'selected_tools' not in nested_config['configurable']



def test_application_toolkit_passes_parent_memory_and_subgraph_flag():
    client = FakeToolkitClient()
    parent_memory = object()

    toolkit = ApplicationToolkit.get_toolkit(
        client=client,
        application_id=1,
        application_version_id=2,
        is_subgraph=True,
        memory=parent_memory,
    )

    assert len(client.application_calls) == 1
    initial_call = client.application_calls[0]['kwargs']
    assert initial_call['memory'] is parent_memory
    assert initial_call['is_subgraph'] is True

    tool = toolkit.get_tools()[0]
    assert tool.is_subgraph is True
    assert tool.args_runnable['memory'] is parent_memory
    assert tool.args_runnable['is_subgraph'] is True



def test_nested_child_graph_result_is_collapsed_to_output_for_parent_toolnode():
    reset_sensitive_tools()
    configure_sensitive_tools({'filesystem': ['create_file']})

    executed = []

    def create_file(**kwargs):
        executed.append(kwargs)
        return 'file-created'

    child_tool = StructuredTool.from_function(
        func=create_file,
        name='create_file',
        description='create file',
        metadata={
            'toolkit_type': 'filesystem',
            'toolkit_name': 'filesystem',
            'tool_name': 'create_file',
        },
    )

    child_assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'child', 'tools': [], 'meta': {}},
        client=ChildToolCallingLLM(),
        tools=[child_tool],
        memory=None,
        app_type='predict',
        is_subgraph=True,
        middleware=[SensitiveToolGuardMiddleware()],
    )
    child_runnable = child_assistant.runnable()

    parent_tool = Application(
        name='child_graph',
        description='Nested child graph',
        application=child_runnable,
        return_type='str',
        client=None,
        is_subgraph=True,
    )

    parent_memory = MemorySaver()
    thread_config = {'configurable': {'thread_id': 'nested-child-graph-thread'}}

    initial_llm = ParentResultAwareLLM(target_tool_name='child_graph')
    initial_runnable = _build_parent_runnable(parent_memory, initial_llm, [parent_tool])
    initial_result = initial_runnable.invoke(
        {'messages': [HumanMessage(content='Delegate this task')]},
        config=thread_config,
    )

    assert initial_result['execution_finished'] is False
    assert initial_result['hitl_interrupt']['tool_name'] == 'create_file'

    resumed_llm = ParentResultAwareLLM(target_tool_name='child_graph')
    resumed_runnable = _build_parent_runnable(parent_memory, resumed_llm, [parent_tool])
    resume_result = resumed_runnable.invoke(
        {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''},
        config=thread_config,
    )

    assert len(executed) == 1
    assert resumed_llm.calls[-1] == ['child-graph-complete']
    assert resume_result['execution_finished'] is True
    assert resume_result['output'] == 'parent-graph-complete'

    reset_sensitive_tools()



def test_nested_child_graph_reject_path_returns_normalized_result_to_parent_toolnode():
    reset_sensitive_tools()
    configure_sensitive_tools({'filesystem': ['create_file']})

    executed = []

    def create_file(**kwargs):
        executed.append(kwargs)
        return 'file-created'

    child_tool = StructuredTool.from_function(
        func=create_file,
        name='create_file',
        description='create file',
        metadata={
            'toolkit_type': 'filesystem',
            'toolkit_name': 'filesystem',
            'tool_name': 'create_file',
        },
    )

    child_assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'child', 'tools': [], 'meta': {}},
        client=RejectAwareChildLLM(),
        tools=[child_tool],
        memory=None,
        app_type='predict',
        is_subgraph=True,
        middleware=[SensitiveToolGuardMiddleware()],
    )
    child_runnable = child_assistant.runnable()

    parent_tool = Application(
        name='child_graph',
        description='Nested child graph',
        application=child_runnable,
        return_type='str',
        client=None,
        is_subgraph=True,
    )

    parent_memory = MemorySaver()
    thread_config = {'configurable': {'thread_id': 'nested-child-graph-reject-thread'}}

    initial_llm = RejectAwareParentLLM()
    initial_runnable = _build_parent_runnable(parent_memory, initial_llm, [parent_tool])
    initial_result = initial_runnable.invoke(
        {'messages': [HumanMessage(content='Delegate this task')]},
        config=thread_config,
    )

    assert initial_result['execution_finished'] is False
    assert initial_result['hitl_interrupt']['tool_name'] == 'create_file'

    resumed_llm = RejectAwareParentLLM()
    resumed_runnable = _build_parent_runnable(parent_memory, resumed_llm, [parent_tool])
    resume_result = resumed_runnable.invoke(
        {'hitl_resume': True, 'hitl_action': 'reject', 'hitl_value': ''},
        config=thread_config,
    )

    assert executed == []
    assert resumed_llm.calls[-1] == ['child-reject-finished']
    assert resume_result['execution_finished'] is True
    assert resume_result['output'] == 'parent-sees:child-reject-finished'

    reset_sensitive_tools()



def test_nested_child_graph_resume_restores_pending_messages_locally():
    reset_sensitive_tools()
    configure_sensitive_tools({'github': ['create_issue']})

    executed = []

    def make_tool(name, return_value):
        def tool(**kwargs):
            executed.append((name, kwargs))
            return return_value

        return StructuredTool.from_function(
            func=tool,
            name=name,
            description=name,
            metadata={
                'toolkit_type': 'github',
                'toolkit_name': 'elitea_testing',
                'tool_name': name,
            },
        )

    child_tools = [
        make_tool('list_files', 'safe-list'),
        make_tool('get_issues', 'safe-issues'),
        make_tool('search_issues', 'safe-search'),
        make_tool('create_issue', 'created-issue'),
    ]

    child_assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'child', 'tools': [], 'meta': {}},
        client=PendingAwareChildLLM(),
        tools=child_tools,
        memory=None,
        app_type='predict',
        is_subgraph=True,
        middleware=[SensitiveToolGuardMiddleware()],
    )
    child_runnable = child_assistant.runnable()

    parent_tool = Application(
        name='child_graph',
        description='Nested child graph',
        application=child_runnable,
        return_type='str',
        client=None,
        is_subgraph=True,
    )

    parent_memory = MemorySaver()
    thread_config = {'configurable': {'thread_id': 'nested-child-graph-pending-thread'}}

    initial_llm = PendingAwareParentLLM()
    initial_runnable = _build_parent_runnable(parent_memory, initial_llm, [parent_tool])
    initial_result = initial_runnable.invoke(
        {'messages': [HumanMessage(content='Delegate this task')]},
        config=thread_config,
    )

    assert initial_result['execution_finished'] is False
    assert initial_result['hitl_interrupt']['tool_name'] == 'create_issue'

    # The child emits create_issue x3 in one AI message.  Under #5245 each
    # sensitive call re-prompts (no batch auto-approve), so we approve once
    # per interrupt until the run finishes.  Pending-message restore must keep
    # working across every resume.
    resume_result = initial_result
    resumed_llm = None
    interrupts = 1  # the initial interrupt already counted above
    for _ in range(10):
        resumed_llm = PendingAwareParentLLM()
        resumed_runnable = _build_parent_runnable(parent_memory, resumed_llm, [parent_tool])
        resume_result = resumed_runnable.invoke(
            {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''},
            config=thread_config,
        )
        if resume_result['execution_finished']:
            break
        assert resume_result['hitl_interrupt']['tool_name'] == 'create_issue', (
            'Each create_issue invocation must re-prompt (per-call, #5245)'
        )
        interrupts += 1

    # Three distinct create_issue invocations → three separate prompts
    # (per-call #5245; no batch auto-approve carry-over).  Exact tool-exec
    # counts are not asserted here because the child subgraph has no
    # checkpointer and replays from scratch on each bubble-resume cycle —
    # this test's invariant is per-call prompting + clean completion + the
    # pending-message restore exercised by the child LLM's history checks.
    assert interrupts == 3, f'Expected one prompt per create_issue call; got {interrupts}'
    assert ('create_issue', {}) in executed
    assert resumed_llm.calls[-1] == ['child-finished']
    assert resume_result['execution_finished'] is True
    assert resume_result['output'] == 'parent-done'

    reset_sensitive_tools()



class FakeApplicationClient:
    """Minimal client whose .application() returns a pre-built child runnable.

    Mirrors elitea_sdk.runtime.clients.client.Client.application: returns the
    SAME runnable on every call so the child's checkpointer survives the
    parent's pause/resume cycle (which re-enters Application._run and rebuilds
    the child via client.application()).

    The child must be a root LangGraphAgentRunnable (not a CompiledStateGraph
    subgraph), because Application._run strips ``__pregel_task_id`` from
    nested_config in the rebuild branch — root graphs need the strip so their
    interrupt() raises GraphInterrupt; subgraphs need the task-id present for
    parent-pregel to track them. Production's client.application(is_subgraph=
    False) returns a root LangGraphAgentRunnable, so the test mirrors that.
    """

    def __init__(self, child_runnable):
        self.child_runnable = child_runnable
        self.application_calls = []

    def application(self, *args, **kwargs):
        self.application_calls.append({'args': args, 'kwargs': dict(kwargs)})
        return self.child_runnable


def test_standalone_application_path_bubbles_hitl_through_rebuild_cycle():
    """Standalone (`client + args_runnable`) path — the production path that
    every UI/indexer-deployed agent takes — must bubble HITL interrupts and
    resume cleanly. Prior tests all used `client=None, application=<prebuilt>`,
    skipping the rebuild branch in `Application._run` (lines 289–316). This
    test exercises the full real-langgraph cycle on the rebuild path:

      1. parent (Assistant) → calls child Application tool
      2. Application._run sees client+args_runnable → rebuilds child via
         client.application(is_subgraph=False, ...)
      3. child sensitive_tool guard fires → child returns hitl_interrupt
      4. Application._run calls interrupt() → bubbles GraphInterrupt to parent
      5. parent checkpoint stored; resume invocation feeds Command(resume=...)
      6. Application._run re-runs, interrupt() returns the resume value
         (positional scratchpad.resume[0] semantics)
      7. child re-invoked with hitl_resume=True → unblocks the sensitive tool
      8. side-effecting tool runs exactly once (no double-execution)

    Also asserts gap-2 fix: parent's `hitl_decisions` audit trail records the
    parent Application tool name (`child_graph`), NOT the child leaf tool
    name (`create_file`) which the parent graph does not own.
    """
    reset_sensitive_tools()
    configure_sensitive_tools({'filesystem': ['create_file']})

    executed = []

    def create_file(**kwargs):
        executed.append(kwargs)
        return 'file-created'

    child_tool = StructuredTool.from_function(
        func=create_file,
        name='create_file',
        description='create file',
        metadata={
            'toolkit_type': 'filesystem',
            'toolkit_name': 'filesystem',
            'tool_name': 'create_file',
        },
    )

    # Build child as a ROOT LangGraphAgentRunnable (is_subgraph=False).
    # This matches production: Application._run's rebuild branch forces
    # is_subgraph=False on runnable_args, so client.application() builds a
    # root-graph child whose interrupt() raises GraphInterrupt cleanly when
    # __pregel_task_id is stripped from nested_config (the #5046 fix).
    child_assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'child', 'tools': [], 'meta': {}},
        client=ChildToolCallingLLM(),
        tools=[child_tool],
        memory=None,
        app_type='predict',
        is_subgraph=False,
        middleware=[SensitiveToolGuardMiddleware()],
    )
    child_runnable = child_assistant.runnable()

    fake_client = FakeApplicationClient(child_runnable)

    parent_tool = Application(
        name='child_graph',
        description='Nested child graph (rebuilt per _run via client)',
        application=child_runnable,  # initial; the rebuild branch overwrites
        return_type='str',
        client=fake_client,           # truthy → rebuild branch fires
        is_subgraph=True,             # parent's registered flag (toolkit-side)
        args_runnable={
            'application_id': 99,
            'application_version_id': 1,
            'is_subgraph': True,        # registered True; _run forces False
        },
    )

    parent_memory = MemorySaver()
    thread_config = {
        'configurable': {'thread_id': 'standalone-rebuild-thread'}
    }

    # Capture every update_state call so we can verify hitl_decisions
    # attribution before the post-run cleanup wipes the field.
    captured_decisions = []
    original_update_state = LangGraphAgentRunnable.update_state

    def capturing_update_state(self, config, values, *args, **kwargs):
        # Only capture decisions from the PARENT graph (not child's own internal decisions)
        cfg_thread = (config or {}).get('configurable', {}).get('thread_id', '')
        is_parent = cfg_thread == 'standalone-rebuild-thread'
        if is_parent and isinstance(values, dict):
            decisions = values.get('hitl_decisions')
            if decisions:
                captured_decisions.extend(decisions)
        return original_update_state(self, config, values, *args, **kwargs)

    initial_llm = ParentResultAwareLLM(target_tool_name='child_graph')
    initial_runnable = _build_parent_runnable(
        parent_memory, initial_llm, [parent_tool]
    )
    initial_result = initial_runnable.invoke(
        {'messages': [HumanMessage(content='Delegate this task')]},
        config=thread_config,
    )

    # The rebuild branch fired exactly once on the initial invocation, and
    # is_subgraph was forced False before the call (so the child is built as
    # a root LangGraphAgentRunnable, not a CompiledStateGraph subgraph).
    assert len(fake_client.application_calls) == 1
    assert fake_client.application_calls[0]['kwargs']['is_subgraph'] is False

    assert initial_result['execution_finished'] is False
    assert initial_result['hitl_interrupt']['tool_name'] == 'create_file'

    # The child has not actually run the sensitive tool yet — the guard
    # paused before execution.
    assert executed == []

    resumed_llm = ParentResultAwareLLM(target_tool_name='child_graph')
    resumed_runnable = _build_parent_runnable(
        parent_memory, resumed_llm, [parent_tool]
    )
    with patch.object(
        LangGraphAgentRunnable, 'update_state', capturing_update_state
    ):
        resume_result = resumed_runnable.invoke(
            {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''},
            config=thread_config,
        )

    # Rebuild branch fires again on resume (Application._run re-runs).
    assert len(fake_client.application_calls) == 2
    assert fake_client.application_calls[1]['kwargs']['is_subgraph'] is False

    # Side-effecting tool ran exactly once across the pause/resume cycle.
    assert len(executed) == 1

    # Resume completes through the parent's LLM with the child's output.
    assert resume_result['execution_finished'] is True
    assert resume_result['output'] == 'parent-graph-complete'
    assert resumed_llm.calls[-1] == ['child-graph-complete']

    # Audit-trail attribution: the bubbled-up decision references the parent
    # Application tool, not the child leaf. Recording 'create_file' would
    # poison the parent's blocked-tool set (parent has no such tool) and
    # produce a misleading audit history.
    bubbled_decisions = [
        d for d in captured_decisions
        if d.get('action') in ('approve', 'reject')
    ]
    assert bubbled_decisions, (
        'expected at least one hitl_decisions entry to be persisted on resume'
    )
    assert all(
        d['tool_name'] == 'child_graph' for d in bubbled_decisions
    ), (
        f"bubbled decision must reference parent Application tool 'child_graph', "
        f"not child leaf; got: {bubbled_decisions}"
    )
    assert all(
        d['tool_name'] != 'create_file' for d in bubbled_decisions
    )

    reset_sensitive_tools()


class TwoSensitiveChildLLM:
    """Child LLM that calls two DISTINCT sensitive tools in sequence.

    First turn (no tool results) → call ``create_file``.
    After ``create_file`` ran     → call the SECOND distinct tool ``delete_file``.
    After ``delete_file`` ran      → finish with ``child-graph-complete``.

    This exercises Issue 1: a single subagent triggering a SECOND distinct
    sensitive-tool approval after the first was approved. The second dialog
    must NOT be swallowed by the standalone (dict-bridge) path in
    ``Application._run``.
    """

    temperature = 0
    max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _TwoSensitiveChildLLMBound(tools)

    def invoke(self, messages, config=None):
        return _TwoSensitiveChildLLMBound([]).invoke(messages, config=config)


class _TwoSensitiveChildLLMBound:
    def __init__(self, tools):
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        contents = [
            str(message.content)
            for message in messages
            if isinstance(message, ToolMessage)
        ]
        if 'file-deleted' in contents:
            return AIMessage(content='child-graph-complete')
        if 'file-created' in contents:
            return AIMessage(
                content='',
                tool_calls=[
                    {
                        'name': 'delete_file',
                        'args': {'path': '/tmp/test.txt'},
                        'id': 'call-child-tool-2',
                        'type': 'tool_call',
                    }
                ],
            )
        return AIMessage(
            content='',
            tool_calls=[
                {
                    'name': 'create_file',
                    'args': {'path': '/tmp/test.txt'},
                    'id': 'call-child-tool-1',
                    'type': 'tool_call',
                }
            ],
        )


def test_standalone_subagent_second_distinct_sensitive_tool_is_not_swallowed():
    """Issue 1 regression — one subagent, two DISTINCT sensitive tools.

    After the first sensitive tool (``create_file``) is approved, the same
    subagent calls a SECOND distinct sensitive tool (``delete_file``). The
    dict-bridge path in ``Application._run`` must surface a fresh interrupt
    for the second tool rather than silently swallowing it (the bug fixed by
    converting the single-shot ``if`` into a ``while`` loop).

    Flow:
      1. initial invoke           → pause at create_file (interrupt #1)
      2. resume(approve) #1        → create_file runs, pause at delete_file (#2)
      3. resume(approve) #2        → delete_file runs, child completes
    Both side-effecting tools execute exactly once.
    """
    reset_sensitive_tools()
    configure_sensitive_tools({'filesystem': ['create_file', 'delete_file']})

    created = []
    deleted = []

    def create_file(**kwargs):
        created.append(kwargs)
        return 'file-created'

    def delete_file(**kwargs):
        deleted.append(kwargs)
        return 'file-deleted'

    create_tool = StructuredTool.from_function(
        func=create_file,
        name='create_file',
        description='create file',
        metadata={
            'toolkit_type': 'filesystem',
            'toolkit_name': 'filesystem',
            'tool_name': 'create_file',
        },
    )
    delete_tool = StructuredTool.from_function(
        func=delete_file,
        name='delete_file',
        description='delete file',
        metadata={
            'toolkit_type': 'filesystem',
            'toolkit_name': 'filesystem',
            'tool_name': 'delete_file',
        },
    )

    child_assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'child', 'tools': [], 'meta': {}},
        client=TwoSensitiveChildLLM(),
        tools=[create_tool, delete_tool],
        memory=None,
        app_type='predict',
        is_subgraph=False,
        middleware=[SensitiveToolGuardMiddleware()],
    )
    child_runnable = child_assistant.runnable()

    fake_client = FakeApplicationClient(child_runnable)

    parent_tool = Application(
        name='child_graph',
        description='Nested child graph (rebuilt per _run via client)',
        application=child_runnable,
        return_type='str',
        client=fake_client,
        is_subgraph=True,
        args_runnable={
            'application_id': 99,
            'application_version_id': 1,
            'is_subgraph': True,
        },
    )

    parent_memory = MemorySaver()
    thread_config = {
        'configurable': {'thread_id': 'two-sensitive-standalone-thread'}
    }

    # --- 1. initial invoke: pause at the FIRST sensitive tool ---
    initial_llm = ParentResultAwareLLM(target_tool_name='child_graph')
    initial_runnable = _build_parent_runnable(
        parent_memory, initial_llm, [parent_tool]
    )
    initial_result = initial_runnable.invoke(
        {'messages': [HumanMessage(content='Delegate this task')]},
        config=thread_config,
    )

    assert initial_result['execution_finished'] is False
    assert initial_result['hitl_interrupt']['tool_name'] == 'create_file'
    assert created == []
    assert deleted == []

    # --- 2. resume #1 (approve create_file): pause at the SECOND tool ---
    resume1_llm = ParentResultAwareLLM(target_tool_name='child_graph')
    resume1_runnable = _build_parent_runnable(
        parent_memory, resume1_llm, [parent_tool]
    )
    resume1_result = resume1_runnable.invoke(
        {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''},
        config=thread_config,
    )

    # The second distinct sensitive tool must surface its OWN interrupt —
    # this is the bug: previously it was swallowed and execution finished.
    assert resume1_result['execution_finished'] is False, (
        'second distinct sensitive tool was swallowed instead of pausing'
    )
    assert resume1_result['hitl_interrupt']['tool_name'] == 'delete_file'
    # First tool ran once; second has not run yet (still pending approval).
    assert len(created) == 1
    assert deleted == []

    # --- 3. resume #2 (approve delete_file): child completes ---
    resume2_llm = ParentResultAwareLLM(target_tool_name='child_graph')
    resume2_runnable = _build_parent_runnable(
        parent_memory, resume2_llm, [parent_tool]
    )
    resume2_result = resume2_runnable.invoke(
        {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''},
        config=thread_config,
    )

    assert resume2_result['execution_finished'] is True
    assert resume2_result['output'] == 'parent-graph-complete'
    # Both side-effecting tools ran exactly once across the whole cycle.
    assert len(created) == 1
    assert len(deleted) == 1

    reset_sensitive_tools()


# NOTE: The former ``test_create_retry_auto_approves_then_distinct_delete_
# still_interrupts`` validated the now-removed within-batch auto-approve
# carry-over in the nested-subgraph path.  Under #5245 every sensitive call
# prompts individually; the nested Application path here has no child
# checkpointer, so the child cannot durably re-pause for a second sensitive
# call within one parent resume.  Per-call prompting + replay-safety is
# covered by tests/runtime/test_sensitive_tool_guard.py::
# test_5245_same_tool_prompts_every_call_across_resumes (single-graph,
# checkpointed) instead.  Durable nested/parallel multi-prompt HITL is
# tracked separately (parallel HITL dispatch redesign).


def test_swarm_result_adapter_hitl_interrupt_and_resume():
    """SwarmResultAdapter detects HITL interrupts and resumes correctly.

    Exercises the swarm-mode path: a compiled graph with a node that calls
    interrupt() has the interrupt detected by SwarmResultAdapter.invoke()
    via get_state(), and resumes with Command(resume=...) when hitl_resume
    is passed.
    """
    from langgraph.graph import StateGraph, START, END, MessagesState
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import interrupt, Command

    def interrupting_node(state: MessagesState):
        review = interrupt({
            'type': 'hitl',
            'guardrail_type': 'sensitive_tool',
            'message': 'Approve this action?',
            'tool_name': 'dangerous_tool',
            'toolkit_name': 'test_toolkit',
            'toolkit_type': 'test',
            'action_label': 'test_toolkit.dangerous_tool',
            'available_actions': ['approve', 'reject'],
        })
        return {"messages": [AIMessage(content=f"Approved: {review.get('action', 'unknown')}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("work", interrupting_node)
    builder.add_edge(START, "work")
    builder.add_edge("work", END)

    checkpointer = MemorySaver()
    compiled = builder.compile(checkpointer=checkpointer)

    class TestSwarmAdapter:
        """Minimal reproduction of SwarmResultAdapter HITL logic."""
        def __init__(self, graph):
            self._graph = graph

        def invoke(self, input, config=None, **kwargs):
            if isinstance(input, dict) and input.get('hitl_resume'):
                resume_value = {
                    'action': input.get('hitl_action', 'approve'),
                    'value': input.get('hitl_value', ''),
                }
                result = self._graph.invoke(
                    Command(resume=resume_value), config, **kwargs
                )
            else:
                if isinstance(input, dict) and not input.get("messages"):
                    raw_input = input.get("input")
                    if isinstance(raw_input, list) and raw_input:
                        input = {**input, "messages": list(raw_input)}
                    elif isinstance(raw_input, str) and raw_input:
                        input = {**input, "messages": [HumanMessage(content=raw_input)]}
                result = self._graph.invoke(input, config, **kwargs)

            try:
                state_snapshot = self._graph.get_state(config)
                hitl_interrupt = None
                if hasattr(state_snapshot, 'tasks') and state_snapshot.tasks:
                    for task in state_snapshot.tasks:
                        if hasattr(task, 'interrupts') and task.interrupts:
                            for intr in task.interrupts:
                                if hasattr(intr, 'value') and isinstance(intr.value, dict):
                                    if intr.value.get('type') == 'hitl':
                                        hitl_interrupt = intr.value
                                        break
                        if hitl_interrupt:
                            break
                if hitl_interrupt:
                    return {
                        'output': hitl_interrupt.get('message', 'Awaiting review'),
                        'messages': result.get('messages', []) if isinstance(result, dict) else [],
                        'execution_finished': False,
                        'hitl_interrupt': hitl_interrupt,
                    }
            except Exception:
                pass

            messages = result.get("messages", [])
            output = ""
            for msg in reversed(messages):
                if not hasattr(msg, "content") or isinstance(msg, HumanMessage):
                    continue
                text = msg.content.strip() if isinstance(msg.content, str) else str(msg.content)
                if text:
                    output = text
                    break

            return {
                'output': output,
                'messages': messages,
                'execution_finished': True,
            }

    adapter = TestSwarmAdapter(compiled)
    config = {'configurable': {'thread_id': 'swarm-hitl-test'}}

    # First invoke: should pause at interrupt
    result = adapter.invoke(
        {'messages': [HumanMessage(content='Do the dangerous thing')]},
        config=config,
    )

    assert result['execution_finished'] is False, (
        f"Expected paused, got: execution_finished={result.get('execution_finished')}"
    )
    assert result.get('hitl_interrupt') is not None
    assert result['hitl_interrupt']['tool_name'] == 'dangerous_tool'
    assert result['hitl_interrupt']['type'] == 'hitl'

    # Resume with approval
    resume_result = adapter.invoke(
        {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''},
        config=config,
    )

    assert resume_result['execution_finished'] is True, (
        f"Expected completed after resume, got: {resume_result}"
    )
    assert 'Approved: approve' in resume_result['output']


def test_swarm_peer_subgraph_with_application_tool_hitl():
    """End-to-end: interrupt() inside Application tool propagates through
    peer subgraph → swarm pregel → SwarmResultAdapter detects it → resume works.

    This simulates the production flow where build_direct_invocation_subgraph
    calls application_tool.invoke() which internally calls interrupt() via
    the dict-bridge HITL bubble-up pattern.
    """
    from langgraph.graph import StateGraph, START, END, MessagesState
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import interrupt, Command
    from langgraph.errors import GraphBubbleUp
    from langgraph_swarm import create_swarm

    # Simulate an Application tool that internally calls interrupt()
    call_count = [0]

    def peer_node(state: MessagesState):
        """Simulates invoke_application calling application_tool.invoke()
        which eventually calls interrupt() via the bubble-up path."""
        call_count[0] += 1
        # On first call, interrupt (simulating Application._run bubble-up)
        review = interrupt({
            'type': 'hitl',
            'guardrail_type': 'sensitive_tool',
            'message': 'Peer agent needs approval for filesystem.create_file',
            'tool_name': 'create_file',
            'toolkit_name': 'filesystem',
            'toolkit_type': 'filesystem',
            'action_label': 'filesystem.create_file',
            'available_actions': ['approve', 'reject'],
        })
        # After resume
        action = review.get('action', 'unknown') if isinstance(review, dict) else 'unknown'
        return {"messages": [AIMessage(content=f"Peer completed with {action}")]}

    # Build main agent: simple node that just outputs
    def main_node(state: MessagesState):
        messages = state.get("messages", [])
        last_content = ""
        for msg in reversed(messages):
            if hasattr(msg, 'content') and not isinstance(msg, HumanMessage):
                last_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if last_content:
                    break
        if last_content and "Peer completed" in last_content:
            return {"messages": [AIMessage(content=f"Main: {last_content}")]}
        # First call — handoff to peer via tool_call
        return {"messages": [AIMessage(
            content="Handing off to peer",
            tool_calls=[{"name": "transfer_to_peer_agent", "args": {}, "id": "handoff-1"}]
        )]}

    # Build peer subgraph
    peer_builder = StateGraph(MessagesState)
    peer_builder.add_node("work", peer_node)
    peer_builder.add_edge(START, "work")
    peer_builder.add_edge("work", END)
    peer_graph = peer_builder.compile(name="peer_agent")

    # Build main subgraph using ToolNode for handoff
    from langgraph_swarm import create_handoff_tool
    handoff_to_peer = create_handoff_tool(
        agent_name="peer_agent",
        description="Hand off to the peer agent"
    )

    def main_agent_node(state: MessagesState):
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, 'content') and not isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else ''
                if content and "Peer completed" in content:
                    return {"messages": [AIMessage(content=f"Final: {content}")]}
        return {"messages": [AIMessage(
            content="Delegating to peer",
            tool_calls=[{"name": "transfer_to_peer_agent", "args": {}, "id": "call-peer"}]
        )]}

    def route_main(state: MessagesState):
        msgs = state.get("messages", [])
        if msgs and hasattr(msgs[-1], 'tool_calls') and msgs[-1].tool_calls:
            return "tools"
        return END

    main_builder = StateGraph(MessagesState)
    main_builder.add_node("agent", main_agent_node)
    main_builder.add_node("tools", ToolNode([handoff_to_peer]))
    main_builder.add_edge(START, "agent")
    main_builder.add_conditional_edges("agent", route_main, {"tools": "tools", END: END})
    main_builder.add_edge("tools", "agent")
    main_graph = main_builder.compile(name="main_agent")

    # Create swarm
    swarm = create_swarm(
        [main_graph, peer_graph],
        default_active_agent="main_agent"
    )
    checkpointer = MemorySaver()
    compiled_swarm = swarm.compile(checkpointer=checkpointer)

    # Use same SwarmResultAdapter logic as production
    class SwarmAdapter:
        def __init__(self, graph):
            self._graph = graph

        def invoke(self, input, config=None, **kwargs):
            if isinstance(input, dict) and input.get('hitl_resume'):
                resume_value = {
                    'action': input.get('hitl_action', 'approve'),
                    'value': input.get('hitl_value', ''),
                }
                result = self._graph.invoke(
                    Command(resume=resume_value), config, **kwargs
                )
            else:
                result = self._graph.invoke(input, config, **kwargs)

            try:
                state_snapshot = self._graph.get_state(config)
                hitl_interrupt = None
                if hasattr(state_snapshot, 'tasks') and state_snapshot.tasks:
                    for task in state_snapshot.tasks:
                        if hasattr(task, 'interrupts') and task.interrupts:
                            for intr in task.interrupts:
                                if hasattr(intr, 'value') and isinstance(intr.value, dict):
                                    if intr.value.get('type') == 'hitl':
                                        hitl_interrupt = intr.value
                                        break
                        if hitl_interrupt:
                            break
                if hitl_interrupt:
                    return {
                        'output': hitl_interrupt.get('message', 'Awaiting review'),
                        'messages': result.get('messages', []) if isinstance(result, dict) else [],
                        'execution_finished': False,
                        'hitl_interrupt': hitl_interrupt,
                    }
            except Exception:
                pass

            messages = result.get("messages", [])
            output = ""
            for msg in reversed(messages):
                if not hasattr(msg, "content") or isinstance(msg, HumanMessage):
                    continue
                text = msg.content.strip() if isinstance(msg.content, str) else str(msg.content)
                if text:
                    output = text
                    break
            return {'output': output, 'messages': messages, 'execution_finished': True}

    adapter = SwarmAdapter(compiled_swarm)
    config = {'configurable': {'thread_id': 'swarm-peer-hitl-test'}}

    # First invoke: main agent hands off to peer, peer calls interrupt()
    result = adapter.invoke(
        {'messages': [HumanMessage(content='Do the task')]},
        config=config,
    )

    assert result['execution_finished'] is False, (
        f"Expected paused at peer HITL, got: {result}"
    )
    assert result['hitl_interrupt']['tool_name'] == 'create_file'
    assert call_count[0] == 1

    # Resume
    resume_result = adapter.invoke(
        {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''},
        config=config,
    )

    assert resume_result['execution_finished'] is True, (
        f"Expected completed after resume, got: {resume_result}"
    )
    assert 'Peer completed with approve' in resume_result['output']
    assert call_count[0] == 2


# ─────────────────────────────────────────────────────────────────────
# Bug #5046 follow-up — Bug 2: second sequential subagent loses
# parent's intermediate messages when its child triggers a HITL
# interrupt that bubbles up via Application._run dict-bridge.
# ─────────────────────────────────────────────────────────────────────


class _TwoSubagentParentLLMBound:
    """Parent LLM that calls subagent_A first, then subagent_B sequentially."""

    def __init__(self, root, tools):
        self.root = root
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        tool_contents = [str(m.content) for m in tool_messages]
        self.root.calls.append(tool_contents)

        # Both subagents finished — final answer.
        if 'subagent-B-result' in tool_contents:
            return AIMessage(content='parent-done')

        # subagent_A finished — call subagent_B next.
        if tool_contents == ['subagent-A-result']:
            return AIMessage(
                content='',
                tool_calls=[{
                    'name': 'subagent_B',
                    'args': {'task': 'Run B'},
                    'id': 'call-subagent-B',
                    'type': 'tool_call',
                }],
            )

        # No tool messages yet — call subagent_A first.
        if not tool_contents:
            return AIMessage(
                content='',
                tool_calls=[{
                    'name': 'subagent_A',
                    'args': {'task': 'Run A'},
                    'id': 'call-subagent-A',
                    'type': 'tool_call',
                }],
            )

        return AIMessage(content=f'unexpected:{tool_contents}')


class TwoSubagentParentLLM:
    temperature = 0
    max_tokens = 1000

    def __init__(self):
        self.calls = []

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _TwoSubagentParentLLMBound(self, tools)

    def invoke(self, messages, config=None):
        return _TwoSubagentParentLLMBound(self, []).invoke(messages, config=config)


class _SafeChildLLMBound:
    """Subagent_A's LLM: just produces a final answer (no tool calls)."""

    def __init__(self, tools):
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        return AIMessage(content='subagent-A-result')


class SafeChildLLM:
    temperature = 0
    max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _SafeChildLLMBound(tools)

    def invoke(self, messages, config=None):
        return _SafeChildLLMBound([]).invoke(messages, config=config)


class _SensitiveChildLLMBound:
    """Subagent_B's LLM: calls a sensitive tool (which fires guard), then completes."""

    def __init__(self, tools):
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content='subagent-B-result')
        return AIMessage(
            content='',
            tool_calls=[{
                'name': 'sensitive_op',
                'args': {'payload': 'x'},
                'id': 'call-sensitive-B',
                'type': 'tool_call',
            }],
        )


class SensitiveChildLLM:
    temperature = 0
    max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _SensitiveChildLLMBound(tools)

    def invoke(self, messages, config=None):
        return _SensitiveChildLLMBound([]).invoke(messages, config=config)


def test_second_sequential_subagent_preserves_parent_pending_on_hitl_resume():
    """Bug #5046 follow-up — Bug 2.

    When a parent invokes two Application-tool subagents sequentially and the
    SECOND subagent triggers a HITL interrupt (sensitive tool), the parent's
    intermediate messages (the first subagent's tool_call + tool_result)
    must survive the pause/resume cycle so the parent's LLM does not re-plan
    from scratch and re-invoke the first subagent.

    Before the fix, ``Application._run`` bubbled up the CHILD's
    ``_pending_messages`` (which describe the child's internal state, not the
    parent's) — the parent's resume saw only ``[Human]`` and re-invoked
    ``subagent_A`` from scratch.
    """
    reset_sensitive_tools()
    configure_sensitive_tools({'demo_kit': ['sensitive_op']})

    sensitive_executions = []

    def sensitive_op(**kwargs):
        sensitive_executions.append(kwargs)
        return 'sensitive-op-done'

    sensitive_tool = StructuredTool.from_function(
        func=sensitive_op,
        name='sensitive_op',
        description='Sensitive op',
        metadata={
            'toolkit_type': 'demo_kit',
            'toolkit_name': 'demo_kit',
            'tool_name': 'sensitive_op',
        },
    )

    # Subagent A: no sensitive tools, produces 'subagent-A-result'.
    child_a_assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'A', 'tools': [], 'meta': {}},
        client=SafeChildLLM(),
        tools=[],
        memory=None,
        app_type='predict',
        is_subgraph=True,
        middleware=[SensitiveToolGuardMiddleware()],
    )
    child_a_runnable = child_a_assistant.runnable()

    # Subagent B: has a sensitive tool, child guard will fire.
    child_b_assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'B', 'tools': [], 'meta': {}},
        client=SensitiveChildLLM(),
        tools=[sensitive_tool],
        memory=None,
        app_type='predict',
        is_subgraph=True,
        middleware=[SensitiveToolGuardMiddleware()],
    )
    child_b_runnable = child_b_assistant.runnable()

    subagent_a_tool = Application(
        name='subagent_A',
        description='First sequential subagent',
        application=child_a_runnable,
        return_type='str',
        client=None,
        is_subgraph=True,
    )
    subagent_b_tool = Application(
        name='subagent_B',
        description='Second sequential subagent',
        application=child_b_runnable,
        return_type='str',
        client=None,
        is_subgraph=True,
    )

    parent_memory = MemorySaver()
    thread_config = {
        'configurable': {'thread_id': 'two-sequential-subagents-thread'}
    }

    # Initial run — should pause at subagent_B's sensitive tool.
    initial_llm = TwoSubagentParentLLM()
    initial_runnable = _build_parent_runnable(
        parent_memory, initial_llm, [subagent_a_tool, subagent_b_tool]
    )
    initial_result = initial_runnable.invoke(
        {'messages': [HumanMessage(content='Run both')]},
        config=thread_config,
    )

    assert initial_result['execution_finished'] is False, (
        f'Expected pause, got: {initial_result}'
    )
    assert initial_result['hitl_interrupt']['tool_name'] == 'sensitive_op'
    # Sensitive tool must NOT have run yet.
    assert sensitive_executions == []

    # The bubbled interrupt must carry PARENT's intermediates (call to
    # subagent_A + its result) so the parent's history survives the resume.
    # ``_pending_messages`` is intentionally stripped from the UI-facing
    # ``initial_result['hitl_interrupt']`` copy, but the full value is
    # persisted in the checkpoint — that is what the resume path reads.
    def _persisted_interrupt_value(runnable, cfg):
        snapshot = runnable.get_state(cfg)
        for task in getattr(snapshot, 'tasks', None) or []:
            for intr in getattr(task, 'interrupts', None) or []:
                value = getattr(intr, 'value', None)
                if isinstance(value, dict) and value.get('type') == 'hitl':
                    return value
        return {}

    persisted_interrupt = _persisted_interrupt_value(initial_runnable, thread_config)
    bubbled_pending = persisted_interrupt.get('_pending_messages') or []
    pending_tool_names = []
    pending_tool_contents = []
    for msg in bubbled_pending:
        msg_type = msg.get('type', '') if isinstance(msg, dict) else ''
        data = msg.get('data', {}) if isinstance(msg, dict) else {}
        if msg_type == 'ai':
            for tc in data.get('tool_calls') or []:
                pending_tool_names.append(tc.get('name'))
        elif msg_type == 'tool':
            pending_tool_contents.append(str(data.get('content', '')))

    assert 'subagent_A' in pending_tool_names, (
        f'Bubbled _pending_messages must contain the AIMessage that '
        f'invoked subagent_A so the parent LLM can see preceding work. '
        f'Got tool calls in pending: {pending_tool_names}'
    )
    assert 'subagent-A-result' in pending_tool_contents, (
        f'Bubbled _pending_messages must contain the ToolMessage with '
        f'subagent_A\'s result so the parent LLM can see what A returned. '
        f'Got tool contents in pending: {pending_tool_contents}'
    )

    # Resume.
    resumed_llm = TwoSubagentParentLLM()
    resumed_runnable = _build_parent_runnable(
        parent_memory, resumed_llm, [subagent_a_tool, subagent_b_tool]
    )
    resume_result = resumed_runnable.invoke(
        {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''},
        config=thread_config,
    )

    # Sensitive tool must have run exactly once.
    assert len(sensitive_executions) == 1, (
        f'Sensitive tool should run exactly once on approve; got '
        f'{len(sensitive_executions)} executions'
    )

    # Parent must complete with the expected final answer.
    assert resume_result['execution_finished'] is True
    assert resume_result['output'] == 'parent-done'

    # The parent LLM on resume must NOT see an empty tool history (which
    # would mean subagent_A's preceding work was lost). The first turn on
    # resume should already see subagent_A's result, then proceed straight
    # to call subagent_B.
    last_call_tool_contents = resumed_llm.calls[-1]
    assert 'subagent-B-result' in last_call_tool_contents, (
        f'Parent LLM final turn should observe subagent_B result. '
        f'Got resumed_llm.calls={resumed_llm.calls}'
    )
    # On resume, the very first parent LLM turn must already have access to
    # subagent_A's prior result. If parent's pending was lost, the LLM would
    # see [] and re-issue the call to subagent_A, doubling its execution.
    # Capturing every parent LLM invocation:
    first_resume_call = resumed_llm.calls[0] if resumed_llm.calls else []
    assert 'subagent-A-result' in first_resume_call, (
        f'Parent LLM first resume turn must see subagent_A\'s result in its '
        f'tool history (otherwise the parent re-plans from scratch and '
        f're-invokes subagent_A). Got first resume call tool history: '
        f'{first_resume_call}'
    )

    reset_sensitive_tools()


# --- #4993: parallel sub-agent fan-out + aggregated multi-interrupt HITL ------


class _MultiAppParentBound:
    def __init__(self, root, tools):
        self.root = root
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        tool_contents = [str(m.content) for m in tool_messages]
        self.root.calls.append(tool_contents)
        if tool_messages:
            return AIMessage(content='parent-done')
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
    """Parent LLM that fans out two Application tool_calls in one turn (#4993)."""

    temperature = 0
    max_tokens = 1000

    def __init__(self, tool_a='child_a', tool_b='child_b',
                 id_a='call-A', id_b='call-B'):
        self.tool_a = tool_a
        self.tool_b = tool_b
        self.id_a = id_a
        self.id_b = id_b
        self.calls = []

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _MultiAppParentBound(self, tools)

    def invoke(self, messages, config=None):
        return _MultiAppParentBound(self, []).invoke(messages, config=config)


class DictBridgeInterruptingApplication:
    """Child whose inner graph ABSORBS the sensitive-tool interrupt and RETURNS
    it in state (the dict-bridge path a real standalone LangGraphAgentRunnable
    takes), so the parallel deferred-aggregation can collect a sentinel rather
    than catching a raised GraphInterrupt. Records the resume action so routing
    can be asserted."""

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


def _subagent(name, application):
    return Application(
        name=name, description=f'{name} worker', application=application,
        return_type='str', client=None, is_subgraph=True,
    )


def test_two_parallel_children_pause_aggregate_into_one_interrupt():
    """Both fanned-out children pause on a sensitive tool → ONE aggregated
    parent interrupt (guardrail_type=parallel_sensitive_tools) whose unpacked
    hitl_interrupts list holds one entry per paused child, each keyed by its
    parent Application tool_call_id."""
    parent_memory = MemorySaver()
    child_a = DictBridgeInterruptingApplication('A-done', 'create_file')
    child_b = DictBridgeInterruptingApplication('B-done', 'delete_file')
    tools = [_subagent('child_a', child_a), _subagent('child_b', child_b)]

    llm = MultiAppParentLLM(tool_a='child_a', tool_b='child_b',
                            id_a='call-A', id_b='call-B')
    runnable = _build_parent_runnable(parent_memory, llm, tools)
    result = runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both')]},
        config={'configurable': {'thread_id': 'parallel-pause-thread'}},
    )

    assert result['execution_finished'] is False
    interrupts = result['hitl_interrupts']
    assert len(interrupts) == 2, f'expected 2 stacked interrupts, got {interrupts}'
    by_id = {i['tool_call_id']: i for i in interrupts}
    assert set(by_id) == {'call-A', 'call-B'}
    assert by_id['call-A']['tool_name'] == 'create_file'
    assert by_id['call-B']['tool_name'] == 'delete_file'
    # Each card is labelled with the sub-agent it originated from so the UI can
    # group N stacked approvals by sub-agent name (issue #4993).
    assert by_id['call-A']['parent_agent_name'] == 'child_a'
    assert by_id['call-B']['parent_agent_name'] == 'child_b'
    # Internal-only keys must be stripped before reaching the UI/transport.
    for entry in interrupts:
        assert '_pending_messages' not in entry
        assert 'nested_config' not in entry


def test_parallel_resume_routes_decisions_to_correct_children():
    """A single resume carrying a hitl_decisions map routes approve→A and
    reject→B to the right children (each resumes from its own checkpoint), both
    ToolMessages return, and the parent completes."""
    parent_memory = MemorySaver()
    child_a = DictBridgeInterruptingApplication('A-done', 'create_file')
    child_b = DictBridgeInterruptingApplication('B-done', 'delete_file')
    tools = [_subagent('child_a', child_a), _subagent('child_b', child_b)]
    thread_config = {'configurable': {'thread_id': 'parallel-resume-thread'}}

    initial_llm = MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B')
    initial_runnable = _build_parent_runnable(parent_memory, initial_llm, tools)
    initial_result = initial_runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both')]},
        config=thread_config,
    )
    assert initial_result['execution_finished'] is False
    assert len(initial_result['hitl_interrupts']) == 2

    resumed_llm = MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B')
    resumed_runnable = _build_parent_runnable(parent_memory, resumed_llm, tools)
    resume_result = resumed_runnable.invoke(
        {'hitl_decisions': [
            {'tool_call_id': 'call-A', 'action': 'approve', 'value': ''},
            {'tool_call_id': 'call-B', 'action': 'reject', 'value': 'no'},
        ]},
        config=thread_config,
    )

    # Each child got resumed with ITS decision.
    assert child_a.calls[-1]['payload'].get('hitl_action') == 'approve'
    assert child_b.calls[-1]['payload'].get('hitl_action') == 'reject'

    # Both ToolMessages reached the parent's final LLM turn, and it completed.
    assert resume_result['execution_finished'] is True
    assert resume_result['output'] == 'parent-done'
    final_contents = resumed_llm.calls[-1]
    assert 'A-done' in final_contents and 'B-done' in final_contents


def test_one_parallel_child_completes_other_pauses():
    """Mixed fan-out outcome: one child finishes, the other pauses. The single
    aggregated interrupt holds ONLY the paused child; the completed sibling's
    ToolMessage is preserved in the interrupt's _pending_messages for restore."""
    parent_memory = MemorySaver()
    child_a = StaticApplication(output='A-done')                       # completes
    child_b = DictBridgeInterruptingApplication('B-done', 'delete_file')  # pauses
    tools = [_subagent('child_a', child_a), _subagent('child_b', child_b)]

    llm = MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B')
    runnable = _build_parent_runnable(parent_memory, llm, tools)
    result = runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both')]},
        config={'configurable': {'thread_id': 'one-pause-thread'}},
    )

    assert result['execution_finished'] is False
    interrupts = result['hitl_interrupts']
    assert len(interrupts) == 1, f'only the paused child should surface; got {interrupts}'
    assert interrupts[0]['tool_call_id'] == 'call-B'
    assert interrupts[0]['tool_name'] == 'delete_file'


# ---------------------------------------------------------------------------
# Multi-round parallel HITL (issue #4993 follow-up)
#
# The single-round parallel design assumed: all children pause once -> one
# aggregated interrupt -> resume once -> done. But each child is a full agent
# that can pause AGAIN after its decision (its LLM picks a DIFFERENT sensitive
# tool on the next turn). These tests pin the multi-round behaviour: a resumed
# child that re-pauses must re-aggregate into a FRESH parent interrupt instead
# of (a) losing the pause inside the gather executor thread or (b) having the
# parent interrupt() return a stale positional-replay value instead of raising.
# ---------------------------------------------------------------------------


class MultiRoundDivergingApplication:
    """Round 1: pauses on ``first_tool``. When that round is REJECTED the child's
    model diverges to ``second_tool`` (also sensitive) and pauses AGAIN. On the
    next (approve) resume it completes. Models the real "block tool X, the LLM
    then tries tool Y" multi-round flow that the single-round design swallowed."""

    def __init__(self, output, first_tool, second_tool):
        self.output = output
        self.first_tool = first_tool
        self.second_tool = second_tool
        self.calls = []
        self._diverged = False

    def _pause(self, tool_name):
        return {
            'output': 'need approval',
            'execution_finished': False,
            'hitl_interrupt': {
                'type': 'hitl',
                'guardrail_type': 'sensitive_tool',
                'message': f'approve {tool_name}?',
                'tool_name': tool_name,
            },
        }

    def invoke(self, payload, config=None):
        self.calls.append({'payload': payload, 'config': config})
        is_resume = isinstance(payload, dict) and payload.get('hitl_resume')
        if not is_resume:
            return self._pause(self.first_tool)
        action = payload.get('hitl_action', 'approve') if isinstance(payload, dict) else 'approve'
        if action == 'reject' and not self._diverged:
            self._diverged = True
            return self._pause(self.second_tool)
        return {'output': self.output, 'execution_finished': True}


def test_parallel_reject_round1_both_diverge_into_second_aggregate():
    """Both children pause on the SAME sensitive tool (round 1). After the user
    BLOCKS both, each child's model diverges to a DIFFERENT sensitive tool and
    re-pauses. A SECOND aggregated parent interrupt MUST fire
    (guardrail_type=parallel_sensitive_tools) with one entry per still-pending
    child — the multi-round case the single-round design dropped."""
    parent_memory = MemorySaver()
    child_a = MultiRoundDivergingApplication('A-done', 'create_file', 'edit_file')
    child_b = MultiRoundDivergingApplication('B-done', 'create_file', 'delete_file')
    tools = [_subagent('child_a', child_a), _subagent('child_b', child_b)]
    thread_config = {'configurable': {'thread_id': 'parallel-multiround-thread'}}

    # Round 1: fan out, both pause on create_file.
    r1 = _build_parent_runnable(parent_memory, MultiAppParentLLM(), tools).invoke(
        {'messages': [HumanMessage(content='Delegate both')]},
        config=thread_config,
    )
    assert r1['execution_finished'] is False
    by_id_1 = {i['tool_call_id']: i for i in r1['hitl_interrupts']}
    assert set(by_id_1) == {'call-A', 'call-B'}
    assert by_id_1['call-A']['tool_name'] == 'create_file'
    assert by_id_1['call-B']['tool_name'] == 'create_file'

    # Reject BOTH -> each child diverges to a distinct sensitive tool and
    # re-pauses -> a fresh aggregated interrupt is raised (NOT swallowed).
    r2 = _build_parent_runnable(parent_memory, MultiAppParentLLM(), tools).invoke(
        {'hitl_decisions': [
            {'tool_call_id': 'call-A', 'action': 'reject', 'value': 'no'},
            {'tool_call_id': 'call-B', 'action': 'reject', 'value': 'no'},
        ]},
        config=thread_config,
    )

    assert r2['execution_finished'] is False, (
        'round-2 divergent sensitive tools must re-fire HITL, not run/lose silently'
    )
    interrupts_2 = r2['hitl_interrupts']
    assert len(interrupts_2) == 2, f'expected a SECOND 2-card aggregate, got {interrupts_2}'
    by_id_2 = {i['tool_call_id']: i for i in interrupts_2}
    assert by_id_2['call-A']['tool_name'] == 'edit_file'
    assert by_id_2['call-B']['tool_name'] == 'delete_file'
    assert by_id_2['call-A']['parent_agent_name'] == 'child_a'
    assert by_id_2['call-B']['parent_agent_name'] == 'child_b'


def test_parallel_round2_holds_only_still_pending_child():
    """Round 1: both pause. On resume one child is APPROVED (completes in the
    background) while the other is REJECTED and diverges to a new sensitive tool.
    The second aggregated interrupt holds ONLY the still-pending child; the
    completed sibling does not resurface."""
    parent_memory = MemorySaver()
    child_a = DictBridgeInterruptingApplication('A-done', 'create_file')  # completes on resume
    child_b = MultiRoundDivergingApplication('B-done', 'create_file', 'delete_file')
    tools = [_subagent('child_a', child_a), _subagent('child_b', child_b)]
    thread_config = {'configurable': {'thread_id': 'parallel-mixed-multiround-thread'}}

    r1 = _build_parent_runnable(parent_memory, MultiAppParentLLM(), tools).invoke(
        {'messages': [HumanMessage(content='Delegate both')]},
        config=thread_config,
    )
    assert len(r1['hitl_interrupts']) == 2

    r2 = _build_parent_runnable(parent_memory, MultiAppParentLLM(), tools).invoke(
        {'hitl_decisions': [
            {'tool_call_id': 'call-A', 'action': 'approve', 'value': ''},
            {'tool_call_id': 'call-B', 'action': 'reject', 'value': 'no'},
        ]},
        config=thread_config,
    )

    assert r2['execution_finished'] is False
    interrupts_2 = r2['hitl_interrupts']
    assert len(interrupts_2) == 1, f'only the re-paused child should surface; got {interrupts_2}'
    assert interrupts_2[0]['tool_call_id'] == 'call-B'
    assert interrupts_2[0]['tool_name'] == 'delete_file'


def test_parallel_multi_round_resolves_to_completion():
    """Full multi-round drive on a real MemorySaver: round-1 reject -> round-2
    fires (proves the stale positional-replay value was consumed so the second
    interrupt RAISES instead of returning) -> round-2 approve -> both children
    complete and the parent finishes with both outputs."""
    parent_memory = MemorySaver()
    child_a = MultiRoundDivergingApplication('A-done', 'create_file', 'edit_file')
    child_b = MultiRoundDivergingApplication('B-done', 'create_file', 'delete_file')
    tools = [_subagent('child_a', child_a), _subagent('child_b', child_b)]
    thread_config = {'configurable': {'thread_id': 'parallel-multiround-complete-thread'}}

    _build_parent_runnable(parent_memory, MultiAppParentLLM(), tools).invoke(
        {'messages': [HumanMessage(content='Delegate both')]},
        config=thread_config,
    )

    r2 = _build_parent_runnable(parent_memory, MultiAppParentLLM(), tools).invoke(
        {'hitl_decisions': [
            {'tool_call_id': 'call-A', 'action': 'reject', 'value': 'no'},
            {'tool_call_id': 'call-B', 'action': 'reject', 'value': 'no'},
        ]},
        config=thread_config,
    )
    assert r2['execution_finished'] is False
    assert len(r2['hitl_interrupts']) == 2

    final_llm = MultiAppParentLLM()
    r3 = _build_parent_runnable(parent_memory, final_llm, tools).invoke(
        {'hitl_decisions': [
            {'tool_call_id': 'call-A', 'action': 'approve', 'value': ''},
            {'tool_call_id': 'call-B', 'action': 'approve', 'value': ''},
        ]},
        config=thread_config,
    )

    assert r3['execution_finished'] is True, (
        'after both rounds resolve the parent must complete (stale-replay '
        'consumption let round-2 raise and round-3 resume route correctly)'
    )
    assert r3['output'] == 'parent-done'
    final_contents = final_llm.calls[-1]
    assert 'A-done' in final_contents and 'B-done' in final_contents
