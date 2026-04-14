from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from elitea_sdk.runtime.langchain.assistant import Assistant
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

    assert 'model' in graph.nodes
    assert 'tools' in graph.nodes

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
    assert nested_config['configurable']['thread_id'] == 'parent-thread'
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
