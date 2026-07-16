from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.langchain.langraph_agent import LangGraphAgentRunnable
from elitea_sdk.runtime.middleware.sensitive_tool_guard import SensitiveToolGuardMiddleware
from elitea_sdk.runtime.tools.application import Application
from elitea_sdk.runtime.tools.llm import LLMNode
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



def test_application_run_propagates_parent_agent_call_id():
    """#5386: the parent tool_call_id is stamped onto the child's event metadata
    as ``parent_agent_call_id`` so the UI can tell two invocations of the SAME
    sub-agent apart (same display name, and on the in-process path the same
    derived thread_id) and render one accordion per invocation instead of
    merging their activity into one.

    The discriminator must be unique PER invocation: a second sequential call to
    the same sub-agent carries the second tool_call_id, not the first.
    """
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

    def _invoke(call_id):
        # Invoked the way LangGraph's ToolNode does: a ToolCall envelope carrying
        # the parent's tool_call id.
        application_tool.invoke(
            {
                'type': 'tool_call',
                'id': call_id,
                'name': 'child_agent',
                'args': {'task': 'Run nested app', 'chat_history': []},
            },
            config={
                'metadata': {'origin': 'parent'},
                'configurable': {'thread_id': 'parent-thread'},
            },
        )

    _invoke('call-A')
    _invoke('call-B')

    assert len(nested.calls) == 2
    # The child's inner events inherit the discriminator via nested_metadata, so
    # asserting it on the child config confirms it was stamped into the shared
    # config metadata (which the wrapper tool event reads too).
    first_meta = nested.calls[0]['config']['metadata']
    second_meta = nested.calls[1]['config']['metadata']
    assert first_meta['parent_agent_call_id'] == 'call-A'
    assert second_meta['parent_agent_call_id'] == 'call-B'
    # Sub-agent display name is unchanged — only the per-invocation id differs.
    assert first_meta['parent_agent_name'] == 'child_agent'
    assert second_meta['parent_agent_name'] == 'child_agent'



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
    """Minimal client whose .application() returns its configured child runnable.

    Tests may replace ``child_runnable`` with a freshly compiled graph sharing
    the same checkpointer to model a process/worker reconstruction. This mirrors
    production's ``Client.application`` rebuild on every Application._run call.

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
    leaked_child_dispatcher = object()

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
            # Simulate a future args-forwarding refactor. The nested rebuild
            # must still enforce the supported hybrid execution boundary.
            'child_dispatcher': leaked_child_dispatcher,
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
    assert fake_client.application_calls[0]['kwargs']['child_dispatcher'] is None

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
    assert fake_client.application_calls[1]['kwargs']['child_dispatcher'] is None

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
# #5778 depth-3: two-level parallel HITL (container-of-containers)
#
# A tier-2 "container" Application can ITSELF fan out over its own tier-3
# leaves via `_run_parallel_application_calls`. When a leaf pauses, the
# container's own dict-bridge cycle produces a `parallel_sensitive_tools`
# aggregate (see application.py `_run`'s `while ... response['hitl_interrupt']`
# loop) — this is exactly the shape a real container-of-containers bubbles up
# to the root as ITS OWN `hitl_interrupt` sentinel. `NestedContainerApplication`
# mocks that shape directly (mirrors `DictBridgeInterruptingApplication`, but
# its first-pause payload is a nested aggregate with one `pending` leaf entry,
# not a flat `sensitive_tool` interrupt).
# ---------------------------------------------------------------------------


class NestedContainerApplication:
    """Mock tier-2 container whose OWN prior fan-out already paused a tier-3
    leaf. Its first invoke() returns a `parallel_sensitive_tools` aggregate
    (one pending leaf) instead of a flat `sensitive_tool` interrupt — the
    shape the root's `_run_parallel_application_calls` must flatten (#5778)
    instead of collapsing to one entry keyed by the container's own id."""

    def __init__(self, output, leaf_tool_call_id, leaf_tool_name):
        self.output = output
        self.leaf_tool_call_id = leaf_tool_call_id
        self.leaf_tool_name = leaf_tool_name
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append({'payload': payload, 'config': config})
        if isinstance(payload, dict) and payload.get('hitl_resume'):
            # Regrouped resume: Application._run's `_grandchild_decisions`
            # branch passes `hitl_decisions` (list) here instead of a single
            # action/value pair. Record which decision the leaf received.
            decisions = payload.get('hitl_decisions') or []
            for d in decisions:
                if d.get('tool_call_id') == self.leaf_tool_call_id:
                    self.calls[-1]['leaf_decision'] = d
            return {'output': self.output, 'execution_finished': True}
        return {
            'output': 'Need approval (nested)',
            'execution_finished': False,
            'hitl_interrupt': {
                'type': 'hitl',
                'guardrail_type': 'parallel_sensitive_tools',
                'message': 'container leaf(s) awaiting approval',
                'pending': [{
                    'type': 'hitl',
                    'guardrail_type': 'sensitive_tool',
                    'message': f'approve {self.leaf_tool_name}?',
                    'tool_name': self.leaf_tool_name,
                    'tool_call_id': self.leaf_tool_call_id,
                }],
            },
        }


def test_two_level_parallel_both_grandchildren_pause_aggregate():
    """Root fans out 2 containers; each container's OWN single leaf paused.
    The root's aggregate must surface exactly 2 pending cards at LEAF
    granularity (not 2 container-level cards) — each keeps its leaf's own
    tool_call_id and receives a public aggregate-unique interrupt_id. Private
    routing remains checkpoint-only."""
    parent_memory = MemorySaver()
    container_x = NestedContainerApplication('X-done', 'leaf-1', 'create_file')
    container_y = NestedContainerApplication('Y-done', 'leaf-2', 'delete_file')
    tools = [_subagent('container_x', container_x), _subagent('container_y', container_y)]

    llm = MultiAppParentLLM(tool_a='container_x', tool_b='container_y',
                            id_a='call-X', id_b='call-Y')
    runnable = _build_parent_runnable(parent_memory, llm, tools)
    result = runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both containers')]},
        config={'configurable': {'thread_id': 'two-level-pause-thread'}},
    )

    assert result['execution_finished'] is False
    interrupts = result['hitl_interrupts']
    assert len(interrupts) == 2, (
        f'expected 2 leaf-level pending cards (not container-level), got: {interrupts}'
    )
    by_id = {i['tool_call_id']: i for i in interrupts}
    # Leaf ids are preserved verbatim — NOT overwritten with the container's
    # own call id ('call-X'/'call-Y'). This is the core #5778 bug fix.
    assert set(by_id) == {'leaf-1', 'leaf-2'}, (
        f'pending entries must be keyed by the LEAF tool_call_id, got: {set(by_id)}'
    )
    assert by_id['leaf-1']['tool_name'] == 'create_file'
    assert by_id['leaf-2']['tool_name'] == 'delete_file'
    assert len({entry['interrupt_id'] for entry in interrupts}) == 2
    assert all('_via_call_id' not in entry for entry in interrupts)

    raw_interrupt = runnable._get_hitl_interrupt(runnable.get_state(
        {'configurable': {'thread_id': 'two-level-pause-thread'}},
    ))
    raw_by_id = {entry['tool_call_id']: entry for entry in raw_interrupt['pending']}
    assert raw_by_id['leaf-1']['_via_call_id'] == 'call-X'
    assert raw_by_id['leaf-2']['_via_call_id'] == 'call-Y'


def test_two_level_parallel_resume_routes_to_correct_grandchild():
    """Resume with UI-shaped decisions keyed by public interrupt_id
    (approve leaf-1 under container_x, reject leaf-2 under container_y). Each
    decision must reach the correct leaf under the correct container, and the
    run completes without a lingering interrupt."""
    parent_memory = MemorySaver()
    container_x = NestedContainerApplication('X-done', 'leaf-1', 'create_file')
    container_y = NestedContainerApplication('Y-done', 'leaf-2', 'delete_file')
    tools = [_subagent('container_x', container_x), _subagent('container_y', container_y)]
    thread_config = {'configurable': {'thread_id': 'two-level-resume-thread'}}

    initial_llm = MultiAppParentLLM('container_x', 'container_y', 'call-X', 'call-Y')
    initial_runnable = _build_parent_runnable(parent_memory, initial_llm, tools)
    initial_result = initial_runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both containers')]},
        config=thread_config,
    )
    assert initial_result['execution_finished'] is False
    assert len(initial_result['hitl_interrupts']) == 2
    interrupts_by_tool = {
        entry['tool_call_id']: entry
        for entry in initial_result['hitl_interrupts']
    }

    resumed_llm = MultiAppParentLLM('container_x', 'container_y', 'call-X', 'call-Y')
    resumed_runnable = _build_parent_runnable(parent_memory, resumed_llm, tools)
    resume_result = resumed_runnable.invoke(
        {'hitl_decisions': [
            {'interrupt_id': interrupts_by_tool['leaf-1']['interrupt_id'],
             'tool_call_id': 'leaf-1', 'action': 'approve', 'value': ''},
            {'interrupt_id': interrupts_by_tool['leaf-2']['interrupt_id'],
             'tool_call_id': 'leaf-2', 'action': 'reject', 'value': 'no'},
        ]},
        config=thread_config,
    )

    # Each container was resumed with the regrouped hitl_decisions list
    # (Application._run's `_grandchild_decisions` branch), and the decision
    # for its OWN leaf is the one it received (not the sibling's).
    assert container_x.calls[-1]['leaf_decision']['action'] == 'approve'
    assert container_y.calls[-1]['leaf_decision']['action'] == 'reject'

    # Both containers completed, and the run finished with no lingering pause.
    assert resume_result['execution_finished'] is True
    assert resume_result['output'] == 'parent-done'
    final_contents = resumed_llm.calls[-1]
    assert 'X-done' in final_contents and 'Y-done' in final_contents


def test_duplicate_leaf_tool_call_ids_route_by_public_interrupt_id():
    """Two sibling containers may reuse the same graph-local leaf call id.

    The public interrupt ids remain distinct and are sufficient for the server
    to hydrate each private container route from the parked checkpoint.
    """
    parent_memory = MemorySaver()
    container_x = NestedContainerApplication('X-done', 'leaf-shared', 'create_file')
    container_y = NestedContainerApplication('Y-done', 'leaf-shared', 'delete_file')
    tools = [_subagent('container_x', container_x), _subagent('container_y', container_y)]
    config = {'configurable': {'thread_id': 'duplicate-leaf-id-thread'}}

    initial = _build_parent_runnable(
        parent_memory,
        MultiAppParentLLM('container_x', 'container_y', 'call-X', 'call-Y'),
        tools,
    ).invoke({'messages': [HumanMessage(content='Delegate both')]}, config=config)

    assert [entry['tool_call_id'] for entry in initial['hitl_interrupts']] == [
        'leaf-shared', 'leaf-shared',
    ]
    by_tool_name = {entry['tool_name']: entry for entry in initial['hitl_interrupts']}
    assert by_tool_name['create_file']['interrupt_id'] != \
        by_tool_name['delete_file']['interrupt_id']
    assert all('_via_call_id' not in entry for entry in initial['hitl_interrupts'])

    resumed = _build_parent_runnable(
        parent_memory,
        MultiAppParentLLM('container_x', 'container_y', 'call-X', 'call-Y'),
        tools,
    ).invoke({'hitl_decisions': [
        {'interrupt_id': by_tool_name['create_file']['interrupt_id'],
         'tool_call_id': 'leaf-shared', 'action': 'approve', 'value': ''},
        {'interrupt_id': by_tool_name['delete_file']['interrupt_id'],
         'tool_call_id': 'leaf-shared', 'action': 'reject', 'value': 'no'},
    ]}, config=config)

    assert resumed['execution_finished'] is True
    assert container_x.calls[-1]['leaf_decision']['action'] == 'approve'
    assert container_y.calls[-1]['leaf_decision']['action'] == 'reject'


def test_nested_public_interrupt_id_is_stable_when_sibling_order_changes():
    stable_first = LLMNode._parallel_interrupt_id(
        'container-call', 'nested-interrupt-uuid', None,
    )
    stable_after_sibling_retired = LLMNode._parallel_interrupt_id(
        'container-call', 'nested-interrupt-uuid', None,
    )
    fallback_first = LLMNode._parallel_interrupt_id(
        'container-call', 'legacy-leaf-call', 0,
    )
    fallback_after_sibling_retired = LLMNode._parallel_interrupt_id(
        'container-call', 'legacy-leaf-call', 1,
    )

    assert stable_first == stable_after_sibling_retired
    assert fallback_first == fallback_after_sibling_retired


def test_parallel_decision_hydration_trusts_only_checkpoint_routes():
    checkpoint_interrupt = {
        'pending': [
            {'interrupt_id': 'public-a', 'tool_call_id': 'leaf-shared',
             '_via_call_id': 'checkpoint-route-a',
             '_nested_interrupt_id': 'nested-public-a'},
            {'interrupt_id': 'public-b', 'tool_call_id': 'leaf-shared',
             '_via_call_id': 'checkpoint-route-b'},
        ],
    }

    hydrated = LangGraphAgentRunnable._hydrate_parallel_hitl_decisions([
        {'interrupt_id': 'public-a', 'tool_call_id': 'leaf-shared',
         '_via_call_id': 'client-forged-route',
         '_nested_interrupt_id': 'client-forged-nested', 'action': 'approve'},
        {'interrupt_id': 'unknown', 'tool_call_id': 'leaf-shared',
         '_via_call_id': 'client-forged-route', 'action': 'approve'},
        {'tool_call_id': 'leaf-shared', '_via_call_id': 'client-forged-route',
         'action': 'approve'},
        {'tool_call_id': 'unknown-leaf',
         '_via_call_id': 'client-forged-route', 'action': 'approve'},
    ], checkpoint_interrupt)

    assert hydrated == [{
        'interrupt_id': 'public-a',
        'tool_call_id': 'leaf-shared',
        '_via_call_id': 'checkpoint-route-a',
        '_nested_interrupt_id': 'nested-public-a',
        'action': 'approve',
    }]


def test_two_level_ancestry_chain_stamped():
    """A grandchild's nested_config['metadata']['parent_agent_path'] must
    have 2 entries (root->container, container->leaf) with distinct call_ids,
    while `parent_agent_name` still equals only the IMMEDIATE parent
    (backward compat, #5778).

    `_hitl_parallel_call_id` (the id stamped into `parent_agent_path`) is only
    populated on the PARALLEL fan-out path (`__hitl_parallel_call_id__`,
    llm.py's `_run_one`, set only inside `_run_parallel_application_calls`) —
    a plain sequential single tool_call never sets it. So this tree must
    exercise 2-way parallel dispatch at BOTH levels (root -> 2 containers in
    parallel; the container under test -> 2 leaves in parallel) to actually
    populate distinct call_ids, matching the real mechanism rather than
    asserting on a code path that never sets `_hitl_parallel_call_id`.
    """
    captured_nested_configs = []

    class RecordingLeaf:
        def __init__(self, tag):
            self.tag = tag

        def invoke(self, payload, config=None):
            captured_nested_configs.append((self.tag, config))
            return {'output': f'{self.tag}-done', 'execution_finished': True}

    # Level 2: container fans out to 2 leaves in parallel (one turn, two
    # tool_calls) so `_run_parallel_application_calls` stamps a distinct
    # `__hitl_parallel_call_id__` for each leaf.
    leaf_1_tool = Application(
        name='leaf_1', description='leaf worker 1',
        application=RecordingLeaf('leaf_1'), return_type='str', client=None,
        is_subgraph=True,
    )
    leaf_2_tool = Application(
        name='leaf_2', description='leaf worker 2',
        application=RecordingLeaf('leaf_2'), return_type='str', client=None,
        is_subgraph=True,
    )

    container_llm = MultiAppParentLLM(tool_a='leaf_1', tool_b='leaf_2',
                                      id_a='call-leaf-1', id_b='call-leaf-2')
    container_assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'container', 'tools': [], 'meta': {}},
        client=container_llm,
        tools=[leaf_1_tool, leaf_2_tool],
        memory=None,
        app_type='predict',
        is_subgraph=True,
    )
    container_runnable = container_assistant.runnable()

    container_x_tool = Application(
        name='container_x', description='tier-2 container X',
        application=container_runnable, return_type='str', client=None,
        is_subgraph=True,
    )

    # Root fans out to a SECOND container in parallel too, purely so the
    # root->container hop also exercises the parallel path (distinct call_id
    # per root-level branch), matching how #5778 actually arises in practice
    # (root fanning out over containers that themselves fan out over leaves).
    container_y_tool = Application(
        name='container_y', description='tier-2 container Y (unused branch)',
        application=StaticApplication(output='Y-done'), return_type='str',
        client=None, is_subgraph=True,
    )

    root_llm = MultiAppParentLLM(tool_a='container_x', tool_b='container_y',
                                 id_a='call-container-x', id_b='call-container-y')
    root_memory = MemorySaver()
    root_runnable = _build_parent_runnable(
        root_memory, root_llm, [container_x_tool, container_y_tool]
    )
    result = root_runnable.invoke(
        {'messages': [HumanMessage(content='Go')]},
        config={'configurable': {'thread_id': 'ancestry-chain-thread'}},
    )
    assert result['execution_finished'] is True

    assert len(captured_nested_configs) == 2
    by_tag = dict(captured_nested_configs)
    leaf_1_nested_config = by_tag['leaf_1']

    path = leaf_1_nested_config['metadata']['parent_agent_path']
    assert len(path) == 2, f'expected root->container->leaf chain of 2 entries, got: {path}'
    assert path[0]['name'] == 'container_x'
    assert path[1]['name'] == 'leaf_1'
    assert path[0]['call_id'] == 'call-container-x'
    assert path[1]['call_id'] == 'call-leaf-1'
    assert path[0]['call_id'] != path[1]['call_id'], (
        f'each ancestry entry must carry its OWN distinct call_id, got: {path}'
    )
    assert path[0]['sibling_ordinal'] == 1
    assert path[1]['sibling_ordinal'] == 1
    # parent_agent_name is unchanged: still ONLY the immediate parent's name
    # (backward compat — existing 1-level consumers read this single field).
    assert leaf_1_nested_config['metadata']['parent_agent_name'] == 'leaf_1'

    leaf_2_nested_config = by_tag['leaf_2']
    leaf_2_path = leaf_2_nested_config['metadata']['parent_agent_path']
    assert leaf_2_path[1]['call_id'] == 'call-leaf-2'
    # Sibling leaves under the SAME container share the container hop but
    # have distinct ids for their own hop.
    assert leaf_2_path[0]['call_id'] == path[0]['call_id']
    assert leaf_2_path[1]['call_id'] != path[1]['call_id']
    assert leaf_2_path[1]['sibling_ordinal'] == 2


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


# ── Track 2 (#4993): durable park-by-returning ──────────────────────────────

def _build_parent_runnable_with_dispatcher(memory, llm, tools, dispatcher):
    assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'Use tools', 'tools': [], 'meta': {}},
        client=llm,
        tools=tools,
        memory=memory,
        app_type='predict',
        child_dispatcher=dispatcher,
    )
    return assistant.runnable()


def test_parallel_fanout_parks_when_child_dispatcher_present():
    """With a child_dispatcher seam injected, a multi-Application fan-out PARKS
    instead of running in-process (Track 2). The runnable returns
    execution_finished=False, parallel_parked=True, and one dispatch spec per
    sub-agent — keyed by tool_call_id with a child_thread_id matching the
    in-process scheme (parent:name:call_id). No child is invoked, and the
    parent thread_id is preserved for the later parallel_reconcile re-invoke."""
    parent_memory = MemorySaver()
    child_a = DictBridgeInterruptingApplication('A-done', 'create_file')
    child_b = DictBridgeInterruptingApplication('B-done', 'delete_file')
    tools = [_subagent('child_a', child_a), _subagent('child_b', child_b)]

    llm = MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B')
    runnable = _build_parent_runnable_with_dispatcher(
        parent_memory, llm, tools, dispatcher=object(),
    )
    result = runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both')]},
        config={'configurable': {'thread_id': 'park-thread'}},
    )

    assert result['execution_finished'] is False
    assert result['parallel_parked'] is True
    specs = {s['tool_call_id']: s for s in result['parallel_dispatch']}
    assert set(specs) == {'call-A', 'call-B'}
    assert specs['call-A']['name'] == 'child_a'
    assert specs['call-B']['name'] == 'child_b'
    assert specs['call-A']['child_thread_id'].startswith('park-thread:dispatch_')
    assert specs['call-A']['child_thread_id'].endswith(':child_a:call-A')
    assert specs['call-B']['child_thread_id'].startswith('park-thread:dispatch_')
    assert specs['call-B']['child_thread_id'].endswith(':child_b:call-B')
    assert specs['call-A']['dispatch_epoch'] == specs['call-B']['dispatch_epoch']
    assert specs['call-A']['input'] == {'task': 'Run A'}
    assert specs['call-B']['index'] == 1
    assert specs['call-A']['sibling_ordinal'] == 1
    assert specs['call-B']['sibling_ordinal'] == 2
    # Parked = NOTHING ran in-process (durable dispatch is pylon_main's job).
    assert child_a.calls == []
    assert child_b.calls == []
    # thread_id survives so the reconcile pass can re-invoke this exact parent.
    assert result['thread_id'] == 'park-thread'


def test_parallel_fanout_falls_back_to_gather_without_dispatcher():
    """dispatcher absent (None) → the Track 1 in-process gather path runs
    unchanged: children execute and a paused child still aggregates into the
    parent interrupt. Guards the back-compat contract (CLI/tests unaffected)."""
    parent_memory = MemorySaver()
    child_a = DictBridgeInterruptingApplication('A-done', 'create_file')
    child_b = DictBridgeInterruptingApplication('B-done', 'delete_file')
    tools = [_subagent('child_a', child_a), _subagent('child_b', child_b)]

    llm = MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B')
    runnable = _build_parent_runnable(parent_memory, llm, tools)
    result = runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both')]},
        config={'configurable': {'thread_id': 'no-dispatcher-thread'}},
    )

    # No park markers; in-process gather produced the aggregated interrupt.
    assert 'parallel_parked' not in result
    assert result['execution_finished'] is False
    assert len(result['hitl_interrupts']) == 2
    # Children actually ran in-process.
    assert child_a.calls and child_b.calls


# ── Track 2 (#4993): reconcile-resume assembly from child checkpoints ───────

class _SimpleAnswerBound:
    def __init__(self, answer):
        self.answer = answer

    def invoke(self, messages, config=None):
        return AIMessage(content=self.answer)


class SimpleAnswerLLM:
    """Standalone child LLM that answers in one turn with no tool calls — used
    to materialise a COMPLETED child checkpoint under a derived child_thread_id,
    the durable source the parent reconcile reads back (#4993)."""

    temperature = 0
    max_tokens = 100

    def __init__(self, answer):
        self.answer = answer

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _SimpleAnswerBound(self.answer)

    def invoke(self, messages, config=None):
        return AIMessage(content=self.answer)


def _materialise_completed_child(memory, child_thread_id, answer):
    """Run a real standalone child agent to completion on the SHARED checkpointer
    under ``child_thread_id`` so the parent can read its result back during
    reconcile — emulating a durable child task that ran in a separate process."""
    child = _build_parent_runnable(memory, SimpleAnswerLLM(answer), [])
    return child.invoke(
        {'messages': [HumanMessage(content='do your part')]},
        config={'configurable': {'thread_id': child_thread_id}},
    )


def test_parallel_reconcile_assembles_child_results_and_completes():
    """End-to-end Track 2 reconcile: a parked parent re-invoked with
    ``parallel_reconcile`` reads EACH child's own checkpoint (durable source,
    not the ephemeral arbiter result), appends one ToolMessage per child keyed
    by the parent Application tool_call_id, and resumes the agent so the LLM
    synthesizes the final answer. Children are NOT re-run in-process. This
    validates the novel ``update_state(as_node=__start__) + invoke(None)``
    re-entry from an END (parked) checkpoint."""
    parent_memory = MemorySaver()
    parent_thread_id = 'reconcile-thread'

    # 1. Parent fans out and PARKS (dispatcher present).
    child_a = DictBridgeInterruptingApplication('unused', 'create_file')
    child_b = DictBridgeInterruptingApplication('unused', 'delete_file')
    tools = [_subagent('child_a', child_a), _subagent('child_b', child_b)]
    park_llm = MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B')
    park_runnable = _build_parent_runnable_with_dispatcher(
        parent_memory, park_llm, tools, dispatcher=object())
    cfg = {'configurable': {'thread_id': parent_thread_id}}
    parked = park_runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both')]}, config=cfg)
    assert parked['parallel_parked'] is True

    # 2. Children run durably under the epoch-scoped ids from the persisted roster.
    specs = {spec['tool_call_id']: spec for spec in parked['parallel_dispatch']}
    a_done = _materialise_completed_child(
        parent_memory, specs['call-A']['child_thread_id'], 'A-RESULT')
    b_done = _materialise_completed_child(
        parent_memory, specs['call-B']['child_thread_id'], 'B-RESULT')
    assert a_done['execution_finished'] and b_done['execution_finished']

    # 3. pylon_main re-invokes the parked parent once both children settled.
    reconcile_llm = MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B')
    reconcile_runnable = _build_parent_runnable_with_dispatcher(
        parent_memory, reconcile_llm, tools, dispatcher=object())
    result = reconcile_runnable.invoke(
        {'parallel_reconcile': parked['dispatch_epoch']}, config=cfg,
    )

    # Parent completed by synthesizing from BOTH child results.
    assert result['execution_finished'] is True
    assert result['output'] == 'parent-done'
    final_tool_contents = reconcile_llm.calls[-1]
    assert 'A-RESULT' in final_tool_contents and 'B-RESULT' in final_tool_contents
    # The fan-out children were NOT executed in-process during reconcile.
    assert child_a.calls == [] and child_b.calls == []
    # Parked marker is cleared — not re-parked, not a spurious interrupt.
    assert 'parallel_parked' not in result
    assert not result.get('hitl_interrupts')

    calls_after_first_reconcile = len(reconcile_llm.calls)
    duplicate = reconcile_runnable.invoke(
        {'parallel_reconcile': parked['dispatch_epoch']}, config=cfg,
    )
    assert duplicate['execution_finished'] is True
    assert duplicate['parallel_dispatch'] == []
    assert len(reconcile_llm.calls) == calls_after_first_reconcile

    # A later turn may receive provider-reused call ids. Its persisted epoch and
    # child checkpoints must still be a new generation.
    fresh_runnable = _build_parent_runnable_with_dispatcher(
        parent_memory,
        MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B'),
        tools,
        dispatcher=object(),
    )
    fresh = fresh_runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both again')]}, config=cfg,
    )
    assert fresh['parallel_parked'] is True
    assert fresh['dispatch_epoch'] != parked['dispatch_epoch']
    assert {
        spec['child_thread_id'] for spec in fresh['parallel_dispatch']
    }.isdisjoint({
        spec['child_thread_id'] for spec in parked['parallel_dispatch']
    })


def test_parallel_reconcile_keeps_parent_parked_when_child_is_missing():
    """Missing durable child state is retryable and cannot finalize the parent."""
    parent_memory = MemorySaver()
    parent_thread_id = 'reconcile-missing-thread'
    child_a = DictBridgeInterruptingApplication('unused', 'create_file')
    child_b = DictBridgeInterruptingApplication('unused', 'delete_file')
    tools = [_subagent('child_a', child_a), _subagent('child_b', child_b)]
    park_llm = MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B')
    park_runnable = _build_parent_runnable_with_dispatcher(
        parent_memory, park_llm, tools, dispatcher=object())
    cfg = {'configurable': {'thread_id': parent_thread_id}}
    parked = park_runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both')]}, config=cfg)
    specs = {spec['tool_call_id']: spec for spec in parked['parallel_dispatch']}
    # Only child_a materialises; child_b's checkpoint is intentionally absent.
    _materialise_completed_child(
        parent_memory, specs['call-A']['child_thread_id'], 'A-RESULT')

    reconcile_llm = MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B')
    reconcile_runnable = _build_parent_runnable_with_dispatcher(
        parent_memory, reconcile_llm, tools, dispatcher=object())
    result = reconcile_runnable.invoke(
        {'parallel_reconcile': parked['dispatch_epoch']}, config=cfg,
    )

    assert result['execution_finished'] is False
    assert result['parallel_parked'] is True
    assert result['parallel_dispatch'] == []
    # No partial ToolMessages are committed and the parent never synthesizes.
    assert reconcile_llm.calls == []
    checkpoint = reconcile_runnable.get_state(cfg)
    assert checkpoint.values['parallel_tasks']['parked'] is True


def test_parked_parent_ignores_duplicate_non_reconcile_delivery():
    """A transport retry cannot re-advertise and relaunch the child roster."""
    memory = MemorySaver()
    thread_id = 'parked-duplicate-thread'
    tools = [
        _subagent('child_a', DictBridgeInterruptingApplication('unused', 'create_file')),
        _subagent('child_b', DictBridgeInterruptingApplication('unused', 'delete_file')),
    ]
    parent_llm = MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B')
    runnable = _build_parent_runnable_with_dispatcher(
        memory, parent_llm, tools, dispatcher=object(),
    )
    config = {'configurable': {'thread_id': thread_id}}
    parked = runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both')]}, config=config,
    )
    calls_before_retry = len(parent_llm.calls)

    duplicate = runnable.invoke(
        {'messages': [HumanMessage(content='Duplicate delivery')]}, config=config,
    )

    assert parked['parallel_dispatch']
    assert duplicate['parallel_waiting'] is True
    assert duplicate['parallel_dispatch'] == []
    assert duplicate['dispatch_epoch'] == parked['dispatch_epoch']
    assert len(parent_llm.calls) == calls_before_retry


def test_stale_reconcile_epoch_waits_without_readvertising_children():
    memory = MemorySaver()
    tools = [
        _subagent('child_a', DictBridgeInterruptingApplication('unused', 'create_file')),
        _subagent('child_b', DictBridgeInterruptingApplication('unused', 'delete_file')),
    ]
    llm = MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B')
    runnable = _build_parent_runnable_with_dispatcher(memory, llm, tools, dispatcher=object())
    config = {'configurable': {'thread_id': 'stale-reconcile-thread'}}
    parked = runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both')]}, config=config,
    )

    stale = runnable.invoke({'parallel_reconcile': 'wrong-epoch'}, config=config)

    assert stale['execution_finished'] is False
    assert stale['parallel_waiting'] is True
    assert stale['parallel_dispatch'] == []
    assert stale['dispatch_epoch'] == parked['dispatch_epoch']


def test_parallel_reconcile_turns_terminal_child_failures_into_tool_results():
    """A failed/missing child cannot leave the already-drained gate parked forever."""
    memory = MemorySaver()
    thread_id = 'reconcile-terminal-errors-thread'
    tools = [
        _subagent('child_a', DictBridgeInterruptingApplication('unused', 'create_file')),
        _subagent('child_b', DictBridgeInterruptingApplication('unused', 'delete_file')),
    ]
    park_llm = MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B')
    config = {'configurable': {'thread_id': thread_id}}
    parked = _build_parent_runnable_with_dispatcher(
        memory, park_llm, tools, dispatcher=object(),
    ).invoke({'messages': [HumanMessage(content='Delegate both')]}, config=config)

    reconcile_llm = MultiAppParentLLM('child_a', 'child_b', 'call-A', 'call-B')
    runnable = _build_parent_runnable_with_dispatcher(
        memory, reconcile_llm, tools, dispatcher=object(),
    )
    result = runnable.invoke({
        'parallel_reconcile': parked['dispatch_epoch'],
        'parallel_terminal_errors': {
            parked['parallel_dispatch'][0]['child_thread_id']: {'error': 'child A failed'},
            parked['parallel_dispatch'][1]['child_thread_id']: {'error': 'child B was not dispatched'},
        },
    }, config=config)

    assert result['execution_finished'] is True
    assert result['output'] == 'parent-done'
    assert any('child A failed' in item for item in reconcile_llm.calls[-1])
    assert any('child B was not dispatched' in item for item in reconcile_llm.calls[-1])


def test_parallel_reconcile_keeps_parent_parked_when_child_is_paused_or_unreadable():
    for status in ('paused', 'unreadable'):
        parent_memory = MemorySaver()
        thread_id = f'reconcile-{status}-thread'
        tools = [
            _subagent('child_a', DictBridgeInterruptingApplication('unused', 'create_file')),
            _subagent('child_b', DictBridgeInterruptingApplication('unused', 'delete_file')),
        ]
        config = {'configurable': {'thread_id': thread_id}}
        parked_runnable = _build_parent_runnable_with_dispatcher(
            parent_memory, MultiAppParentLLM(), tools, dispatcher=object(),
        )
        parked = parked_runnable.invoke(
            {'messages': [HumanMessage(content='Delegate both')]}, config=config,
        )
        reconcile_runnable = _build_parent_runnable_with_dispatcher(
            parent_memory, MultiAppParentLLM(), tools, dispatcher=object(),
        )

        with patch.object(
            LangGraphAgentRunnable, '_read_child_result',
            return_value=(status, None),
        ):
            payload = {'parallel_reconcile': parked['dispatch_epoch']}
            if status == 'paused':
                payload['parallel_terminal_errors'] = {
                    spec['child_thread_id']: {'error': 'stale failure'}
                    for spec in parked['parallel_dispatch']
                }
            result = reconcile_runnable.invoke(payload, config=config)

        assert result['parallel_parked'] is True
        assert result['execution_finished'] is False
        assert result['parallel_dispatch'] == []
        assert reconcile_runnable.get_state(config).values[
            'parallel_tasks'
        ]['parked'] is True


# ── #5778 depth-3: REAL 3-tier graph — container self-restores its own pending ──
#
# The existing depth-3 tests (NestedContainerApplication) MOCK the container's
# hitl_interrupt shape, so the container never has a real LLM node that could
# re-plan from scratch — they cannot detect the pending-messages loss. These
# tests build a GENUINE middle tier: a real Assistant().runnable() container
# whose own LLM fans out two real leaf Applications, one of which completes and
# one of which pauses at a sensitive tool. On resume the container must restore
# the completed leaf's ToolMessage from its own durable raw interrupt payload,
# NOT re-plan and re-invoke the finished leaf.


class _ContainerFanoutBound:
    """Container LLM: fans out leaf_done + leaf_pause in ONE turn, then
    synthesizes a final answer once both leaves have produced ToolMessages."""

    def __init__(self, root, tools):
        self.root = root
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        tool_contents = [str(m.content) for m in tool_messages]
        self.root.calls.append(tool_contents)
        have_done = any('leaf-done-result' in c for c in tool_contents)
        have_pause = any('leaf-pause-result' in c for c in tool_contents)
        if have_done and have_pause:
            return AIMessage(content='container-complete')
        # Fan out BOTH leaves in a single assistant turn.
        return AIMessage(
            content='',
            tool_calls=[
                {'name': 'leaf_done', 'args': {'task': 'Resolve name'},
                 'id': 'call-leaf-done', 'type': 'tool_call'},
                {'name': 'leaf_pause', 'args': {'task': 'Resolve surname'},
                 'id': 'call-leaf-pause', 'type': 'tool_call'},
            ],
        )


class ContainerFanoutLLM:
    temperature = 0
    max_tokens = 1000

    def __init__(self):
        self.calls = []

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _ContainerFanoutBound(self, tools)

    def invoke(self, messages, config=None):
        return _ContainerFanoutBound(self, []).invoke(messages, config=config)


class _RootToContainerBound:
    """Root LLM: single tool call to the container, then final answer."""

    def __init__(self, root, tools):
        self.root = root
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        tool_contents = [str(m.content) for m in tool_messages]
        self.root.calls.append(tool_contents)
        if tool_contents:
            return AIMessage(content='root-complete')
        return AIMessage(
            content='',
            tool_calls=[{
                'name': 'full_name_resolver', 'args': {'task': 'Resolve full name'},
                'id': 'call-container', 'type': 'tool_call',
            }],
        )


class RootToContainerLLM:
    temperature = 0
    max_tokens = 1000

    def __init__(self):
        self.calls = []

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _RootToContainerBound(self, tools)

    def invoke(self, messages, config=None):
        return _RootToContainerBound(self, []).invoke(messages, config=config)


def _build_real_container_runnable(
    leaf_pause_executions, *, memory=None, leaf_done_app=None,
    leaf_pause_app=None,
):
    """Build a REAL tier-2 container Assistant().runnable() whose LLM fans out
    two leaf Applications, using the SAME leaf doubles the working depth-2 tests
    use so this test isolates the NESTING (a container between root and leaves),
    not a different leaf-raise path:

      * leaf_done  → StaticApplication (completes immediately with a result)
      * leaf_pause → DictBridgeInterruptingApplication (RETURNS a sensitive_tool
        hitl_interrupt in its response dict — the dict-bridge shape a real
        standalone child takes — so the container's fan-out collects a deferred
        sentinel and aggregates, exactly like the working root-level case).

    The container has its OWN MemorySaver (thread_id-scoped) so its paused state
    persists across the parent's resume — mirroring production, where
    client.application() rebuilds the container with the indexer's PostgresSaver.
    """
    leaf_done_app = leaf_done_app or StaticApplication(output='leaf-done-result')
    leaf_done_tool = _subagent('leaf_done', leaf_done_app)

    # leaf_pause records each resume so we can assert its sensitive op ran once.
    leaf_pause_app = leaf_pause_app or _RecordingDictBridgeApplication(
        'leaf-pause-result', 'surname_op', leaf_pause_executions)
    leaf_pause_tool = _subagent('leaf_pause', leaf_pause_app)

    container_assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'fan out leaves', 'tools': [], 'meta': {}},
        client=ContainerFanoutLLM(), tools=[leaf_done_tool, leaf_pause_tool],
        memory=memory or MemorySaver(), app_type='predict', is_subgraph=False,
        middleware=[SensitiveToolGuardMiddleware()],
    )
    return (
        container_assistant.runnable(), leaf_done_app, leaf_pause_app,
        container_assistant.memory,
    )


class _RecordingDictBridgeApplication:
    """Like DictBridgeInterruptingApplication but records resume executions so a
    test can assert the sensitive op ran exactly once across pause/resume."""

    def __init__(self, output, tool_name, executions):
        self.output = output
        self.tool_name = tool_name
        self.executions = executions
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append({'payload': payload, 'config': config})
        if isinstance(payload, dict) and payload.get('hitl_resume'):
            self.executions.append(payload)
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


def test_rebuilt_container_restores_pending_messages_from_raw_checkpoint_interrupt():
    """#5778 nested-tier bug: a container (tier-2) whose own fan-out completed
    ONE leaf and paused on a SIBLING leaf must, on resume, restore the completed
    leaf's ToolMessage from its own raw durable checkpoint interrupt — NOT
    re-plan from scratch and re-invoke the already-completed leaf. The resume
    uses a newly compiled runnable sharing only the checkpointer, approximating
    a new worker process. Converges in exactly one resume round.
    """
    reset_sensitive_tools()
    configure_sensitive_tools({'demo_kit': ['surname_op']})

    leaf_pause_executions = []
    (
        container_runnable, leaf_done_app, leaf_pause_app, container_memory,
    ) = _build_real_container_runnable(leaf_pause_executions)
    application_client = FakeApplicationClient(container_runnable)

    container_tool = Application(
        name='full_name_resolver',
        description='Resolves a full name by fanning out to two leaves',
        application=container_runnable,
        return_type='str',
        client=application_client,
        is_subgraph=True,
        args_runnable={'application_id': 42, 'application_version_id': 1,
                       'is_subgraph': True},
    )

    parent_memory = MemorySaver()
    thread_config = {'configurable': {'thread_id': 'depth3-container-thread'}}

    # Initial run: root → container → fan out both leaves; leaf_pause pauses.
    initial_runnable = _build_parent_runnable(
        parent_memory, RootToContainerLLM(), [container_tool])
    initial_result = initial_runnable.invoke(
        {'messages': [HumanMessage(content='Resolve Roman Mitusov')]},
        config=thread_config,
    )

    assert initial_result['execution_finished'] is False, (
        f'expected a pause at the leaf sensitive tool, got: {initial_result}'
    )
    # The sensitive tool must NOT have run yet.
    assert leaf_pause_executions == []
    # leaf_done completed exactly once during the initial fan-out.
    assert len(leaf_done_app.calls) == 1, (
        f'leaf_done should have completed once on the initial run; '
        f'got {len(leaf_done_app.calls)} calls'
    )

    # The container's own raw paused checkpoint must durably carry its completed
    # leaf's ToolMessage in the interrupt payload. No update_state write is made
    # against the paused child: doing so would replace the checkpoint task and
    # make the interrupt unresumable.
    container_cfg = {'configurable': {
        'thread_id': 'depth3-container-thread:full_name_resolver'}}
    container_state = container_runnable.get_state(container_cfg)

    def _tool_contents(dict_msgs):
        out = []
        for msg in dict_msgs or []:
            if isinstance(msg, dict) and msg.get('type') == 'tool':
                out.append(str(msg.get('data', {}).get('content', '')))
            elif isinstance(msg, ToolMessage):
                out.append(str(msg.content))
        return out

    payload_pending = []
    for task in getattr(container_state, 'tasks', None) or []:
        for intr in getattr(task, 'interrupts', None) or []:
            v = getattr(intr, 'value', None)
            if isinstance(v, dict) and v.get('_pending_messages'):
                payload_pending = v['_pending_messages']
    durable_contents = _tool_contents(payload_pending)
    assert any('leaf-done-result' in c for c in durable_contents), (
        f"container raw interrupt must durably hold the completed leaf's "
        f"ToolMessage (payload={payload_pending!r})"
    )

    # Recompile the middle-tier graph with the same checkpointer and leaf test
    # doubles, then make subsequent Application rebuilds return this new object.
    # No in-memory runnable state from the initial execution is relied upon.
    rebuilt_container, _, _, _ = _build_real_container_runnable(
        leaf_pause_executions,
        memory=container_memory,
        leaf_done_app=leaf_done_app,
        leaf_pause_app=leaf_pause_app,
    )
    assert rebuilt_container is not container_runnable
    application_client.child_runnable = rebuilt_container
    # Resume with a per-leaf decision (the shape the UI sends for an aggregate
    # pause) keyed by the paused leaf's tool_call_id.
    resume_runnable = _build_parent_runnable(
        parent_memory, RootToContainerLLM(), [container_tool])
    resume_result = resume_runnable.invoke(
        {'hitl_decisions': [
            {'tool_call_id': 'call-leaf-pause', 'action': 'approve', 'value': ''},
        ]},
        config=thread_config,
    )

    # (a) leaf_done must NOT be re-invoked on resume — still exactly one call.
    assert len(leaf_done_app.calls) == 1, (
        f'leaf_done must not be re-invoked on resume (container must restore its '
        f'result from its own checkpoint, not re-plan from scratch); got '
        f'{len(leaf_done_app.calls)} calls'
    )
    # (b) the sensitive tool ran exactly once (approved).
    assert len(leaf_pause_executions) == 1
    # (c) converges in this single resume call.
    assert resume_result['execution_finished'] is True, (
        f'the container must converge in one resume round; got: {resume_result}'
    )
    assert resume_result['output'] == 'root-complete'

    reset_sensitive_tools()


def _build_single_container_with_two_pausing_leaves(thread_id):
    first_executions = []
    second_executions = []
    first_leaf = _RecordingDictBridgeApplication(
        'leaf-done-result', 'first_sensitive_op', first_executions,
    )
    second_leaf = _RecordingDictBridgeApplication(
        'leaf-pause-result', 'second_sensitive_op', second_executions,
    )
    container_runnable, _, _, _ = _build_real_container_runnable(
        second_executions,
        leaf_done_app=first_leaf,
        leaf_pause_app=second_leaf,
    )
    container_tool = Application(
        name='full_name_resolver',
        description='Resolves a full name by fanning out to two leaves',
        application=container_runnable,
        return_type='str',
        client=FakeApplicationClient(container_runnable),
        is_subgraph=True,
        args_runnable={
            'application_id': 42,
            'application_version_id': 1,
            'is_subgraph': True,
        },
    )
    runnable = _build_parent_runnable(
        MemorySaver(), RootToContainerLLM(), [container_tool],
    )
    return (
        runnable,
        {'configurable': {'thread_id': thread_id}},
        first_executions,
        second_executions,
    )


def test_single_container_bubbles_and_resumes_all_parallel_leaf_interrupts():
    """A -> single B -> two pausing C leaves must preserve B's raw aggregate.

    The child runnable's public ``hitl_interrupt`` is intentionally only the
    first UI card.  Application must read the child's authoritative checkpoint
    even when B itself was invoked sequentially; otherwise A sees one decision,
    resumes one leaf, and repeatedly replays B to discover the sibling.
    """
    (
        initial_runnable, config, first_executions, second_executions,
    ) = _build_single_container_with_two_pausing_leaves(
        'single-container-two-pauses',
    )

    initial = initial_runnable.invoke(
        {'messages': [HumanMessage(content='Resolve John Smith')]},
        config=config,
    )

    assert initial['execution_finished'] is False
    assert len(initial['hitl_interrupts']) == 2
    assert {entry['tool_call_id'] for entry in initial['hitl_interrupts']} == {
        'call-leaf-done', 'call-leaf-pause',
    }
    raw_parent = initial_runnable._get_hitl_interrupt(
        initial_runnable.get_state(config),
    )
    assert raw_parent['guardrail_type'] == 'parallel_sensitive_tools'
    assert len(raw_parent['pending']) == 2

    resumed = initial_runnable.invoke(
        {
            'hitl_decisions': [
                {
                    'interrupt_id': entry['interrupt_id'],
                    'tool_call_id': entry['tool_call_id'],
                    'action': 'approve',
                    'value': '',
                }
                for entry in initial['hitl_interrupts']
            ],
        },
        config=config,
    )

    assert resumed['execution_finished'] is True
    assert resumed['output'] == 'root-complete'
    assert len(first_executions) == 1
    assert len(second_executions) == 1


def test_single_container_partial_decision_keeps_remaining_public_id_stable():
    runnable, config, first_executions, second_executions = (
        _build_single_container_with_two_pausing_leaves(
            'single-container-partial-decisions',
        )
    )
    initial = runnable.invoke(
        {'messages': [HumanMessage(content='Resolve John Smith')]},
        config=config,
    )
    by_call = {
        entry['tool_call_id']: entry
        for entry in initial['hitl_interrupts']
    }

    after_first = runnable.invoke(
        {'hitl_decisions': [{
            'interrupt_id': by_call['call-leaf-done']['interrupt_id'],
            'tool_call_id': 'call-leaf-done',
            'action': 'approve',
            'value': '',
        }]},
        config=config,
    )

    assert after_first['execution_finished'] is False
    assert len(after_first['hitl_interrupts']) == 1
    remaining = after_first['hitl_interrupts'][0]
    assert remaining['tool_call_id'] == 'call-leaf-pause'
    assert remaining['interrupt_id'] == by_call['call-leaf-pause']['interrupt_id']
    assert len(first_executions) == 1
    assert second_executions == []

    finished = runnable.invoke(
        {'hitl_decisions': [{
            'interrupt_id': by_call['call-leaf-pause']['interrupt_id'],
            'tool_call_id': 'call-leaf-pause',
            'action': 'approve',
            'value': '',
        }]},
        config=config,
    )

    assert finished['execution_finished'] is True
    assert len(first_executions) == 1
    assert len(second_executions) == 1


def test_real_parallel_containers_route_duplicate_leaf_ids_through_nested_interrupt_ids():
    """A real A -> (B1, B2) -> C graph keeps interrupt ids scoped per tier.

    Both compiled B graphs use the same graph-local paused leaf call id. A must
    expose distinct root-scoped public ids, then restore each B-scoped id while
    consuming the private routing hop so B's checkpoint hydration accepts it.
    """
    x_executions = []
    y_executions = []
    x_runnable, x_done, _, _ = _build_real_container_runnable(x_executions)
    y_runnable, y_done, _, _ = _build_real_container_runnable(y_executions)

    def _container_tool(name, runnable):
        return Application(
            name=name, description=f'{name} real compiled container',
            application=runnable, return_type='str',
            client=FakeApplicationClient(runnable), is_subgraph=True,
            args_runnable={
                'application_id': name,
                'application_version_id': 1,
                'is_subgraph': True,
            },
        )

    tools = [
        _container_tool('container_x', x_runnable),
        _container_tool('container_y', y_runnable),
    ]
    memory = MemorySaver()
    config = {'configurable': {'thread_id': 'real-duplicate-leaf-thread'}}
    initial_runnable = _build_parent_runnable(
        memory,
        MultiAppParentLLM('container_x', 'container_y', 'call-X', 'call-Y'),
        tools,
    )
    initial = initial_runnable.invoke(
        {'messages': [HumanMessage(content='Delegate both real containers')]},
        config=config,
    )

    interrupts = initial['hitl_interrupts']
    assert initial['execution_finished'] is False
    assert len(interrupts) == 2
    assert [entry['tool_call_id'] for entry in interrupts] == [
        'call-leaf-pause', 'call-leaf-pause',
    ]
    assert len({entry['interrupt_id'] for entry in interrupts}) == 2
    assert all('_via_call_id' not in entry for entry in interrupts)
    assert all('_nested_interrupt_id' not in entry for entry in interrupts)

    raw = initial_runnable._get_hitl_interrupt(
        initial_runnable.get_state(config),
    )
    assert all(entry.get('_nested_interrupt_id') for entry in raw['pending'])
    assert all(
        entry['_nested_interrupt_id'] != entry['interrupt_id']
        for entry in raw['pending']
    )

    resumed = _build_parent_runnable(
        memory,
        MultiAppParentLLM('container_x', 'container_y', 'call-X', 'call-Y'),
        tools,
    ).invoke({'hitl_decisions': [
        {
            'interrupt_id': entry['interrupt_id'],
            'tool_call_id': entry['tool_call_id'],
            'action': 'approve',
            'value': '',
        }
        for entry in interrupts
    ]}, config=config)

    assert resumed['execution_finished'] is True
    assert resumed['output'] == 'parent-done'
    assert len(x_done.calls) == 1 and len(y_done.calls) == 1
    assert len(x_executions) == 1 and len(y_executions) == 1


class _TwoRoundDictBridgeApplication:
    """A leaf that pauses on a FIRST sensitive tool, and after that decision
    diverges to a SECOND distinct sensitive tool (pauses again), then completes.
    Reads the resume action from either the scalar shape or the per-leaf
    hitl_decisions list (depth-3 aggregate resume forwards the list)."""

    def __init__(self, output, first_tool, second_tool, executions):
        self.output = output
        self.first_tool = first_tool
        self.second_tool = second_tool
        self.executions = executions
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
        self.executions.append(payload)
        if not self._diverged:
            self._diverged = True
            return self._pause(self.second_tool)
        return {'output': self.output, 'execution_finished': True}


def test_container_tier_staggered_leaf_pauses_each_restore_independently():
    """#5778 nested-tier: a container's leaf pauses across TWO rounds (it
    diverges to a second sensitive tool) while a SIBLING leaf stays completed
    from round 1. Across both rounds the container must keep restoring the
    completed sibling from its OWN checkpoint — the completed leaf is invoked
    exactly ONCE total, and the run converges after the second approval.
    """
    reset_sensitive_tools()
    configure_sensitive_tools({'demo_kit': ['first_op', 'second_op']})

    leaf_done_app = StaticApplication(output='leaf-done-result')
    leaf_pause_execs = []
    leaf_pause_app = _TwoRoundDictBridgeApplication(
        'leaf-pause-result', 'first_op', 'second_op', leaf_pause_execs)

    container_assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'fan out leaves', 'tools': [], 'meta': {}},
        client=ContainerFanoutLLM(),
        tools=[_subagent('leaf_done', leaf_done_app),
               _subagent('leaf_pause', leaf_pause_app)],
        memory=MemorySaver(), app_type='predict', is_subgraph=False,
        middleware=[SensitiveToolGuardMiddleware()],
    )
    container_runnable = container_assistant.runnable()

    container_tool = Application(
        name='full_name_resolver', description='Resolves a full name',
        application=container_runnable, return_type='str',
        client=FakeApplicationClient(container_runnable), is_subgraph=True,
        args_runnable={'application_id': 42, 'application_version_id': 1,
                       'is_subgraph': True},
    )

    parent_memory = MemorySaver()
    thread_config = {'configurable': {'thread_id': 'depth3-staggered-thread'}}

    # Round 0: initial fan-out. leaf_done completes; leaf_pause pauses on first_op.
    r0 = _build_parent_runnable(
        parent_memory, RootToContainerLLM(), [container_tool]).invoke(
        {'messages': [HumanMessage(content='Resolve Roman Mitusov')]},
        config=thread_config,
    )
    assert r0['execution_finished'] is False
    assert len(leaf_done_app.calls) == 1

    # Round 1: approve first_op → leaf_pause diverges to second_op (pauses again).
    r1 = _build_parent_runnable(
        parent_memory, RootToContainerLLM(), [container_tool]).invoke(
        {'hitl_decisions': [
            {'tool_call_id': 'call-leaf-pause', 'action': 'approve', 'value': ''}]},
        config=thread_config,
    )
    assert r1['execution_finished'] is False, (
        f'container should pause AGAIN on the second sensitive tool; got: {r1}'
    )
    assert r0['hitl_interrupts'][0]['interrupt_id'] != r1['hitl_interrupts'][0]['interrupt_id']
    # leaf_done still NOT re-invoked while the container re-planned round 2.
    assert len(leaf_done_app.calls) == 1, (
        f'leaf_done must stay completed (invoked once) across the staggered '
        f're-pause; got {len(leaf_done_app.calls)} calls'
    )

    # Round 2: approve second_op → leaf_pause completes, run converges.
    r2 = _build_parent_runnable(
        parent_memory, RootToContainerLLM(), [container_tool]).invoke(
        {'hitl_decisions': [
            {'tool_call_id': 'call-leaf-pause', 'action': 'approve', 'value': ''}]},
        config=thread_config,
    )
    assert r2['execution_finished'] is True, (
        f'container must converge after the second approval; got: {r2}'
    )
    # leaf_done invoked exactly once across BOTH rounds — never re-run.
    assert len(leaf_done_app.calls) == 1
    assert r2['output'] == 'root-complete'

    reset_sensitive_tools()
