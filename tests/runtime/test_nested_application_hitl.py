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

    resumed_llm = PendingAwareParentLLM()
    resumed_runnable = _build_parent_runnable(parent_memory, resumed_llm, [parent_tool])
    resume_result = resumed_runnable.invoke(
        {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''},
        config=thread_config,
    )

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
