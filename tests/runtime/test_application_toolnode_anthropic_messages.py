from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.assistant import Assistant
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


class StrictAnthropicParentLLM:
    def __init__(self, target_tool_name='child_app'):
        self.target_tool_name = target_tool_name
        self.calls = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _StrictAnthropicParentLLMBound(self, tools)

    def invoke(self, messages, config=None):
        return _StrictAnthropicParentLLMBound(self, []).invoke(messages, config=config)


class _StrictAnthropicParentLLMBound:
    def __init__(self, root, tools):
        self.root = root
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        self.root.calls.append(list(messages))

        non_system_messages = [message for message in messages if not isinstance(message, SystemMessage)]
        if not non_system_messages:
            raise RuntimeError('anthropic.BadRequestError: messages: at least one message is required')

        if any(isinstance(message, ToolMessage) for message in non_system_messages):
            return AIMessage(content='parent-complete')

        return AIMessage(
            content='',
            tool_calls=[
                {
                    'name': self.tools[0].name,
                    'args': {'task': 'Run the child task'},
                    'id': 'call-child-app',
                    'type': 'tool_call',
                }
            ],
        )


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


def test_application_toolnode_runtime_preserves_current_user_message_for_anthropic_ordering():
    child_tool = Application(
        name='child_app',
        description='Nested child app',
        application=StaticApplication(output='child-complete'),
        return_type='str',
        client=None,
        is_subgraph=True,
    )
    llm = StrictAnthropicParentLLM(target_tool_name='child_app')
    runnable = _build_parent_runnable(MemorySaver(), llm, [child_tool])

    result = runnable.invoke(
        {'messages': [HumanMessage(content='Delegate this task')]},
        config={'configurable': {'thread_id': 'anthropic-toolnode-order-thread'}},
    )

    assert result['execution_finished'] is True
    assert result['output'] == 'parent-complete'
    assert [type(message).__name__ for message in llm.calls[0]] == ['SystemMessage', 'HumanMessage']
    assert llm.calls[0][1].content == 'Delegate this task'
    assert [type(message).__name__ for message in llm.calls[1]] == [
        'SystemMessage',
        'HumanMessage',
        'AIMessage',
        'ToolMessage',
    ]
    assert llm.calls[1][1].content == 'Delegate this task'
    assert llm.calls[1][2].tool_calls[0]['name'] == 'child_app'
    assert llm.calls[1][3].tool_call_id == 'call-child-app'


def test_application_toolnode_runtime_does_not_duplicate_matching_input_and_messages():
    child_tool = Application(
        name='child_app',
        description='Nested child app',
        application=StaticApplication(output='child-complete'),
        return_type='str',
        client=None,
        is_subgraph=True,
    )
    llm = StrictAnthropicParentLLM(target_tool_name='child_app')
    runnable = _build_parent_runnable(MemorySaver(), llm, [child_tool])

    result = runnable.invoke(
        {
            'input': 'Delegate this task',
            'messages': [HumanMessage(content='Delegate this task')],
        },
        config={'configurable': {'thread_id': 'anthropic-toolnode-no-duplicate-thread'}},
    )

    assert result['execution_finished'] is True
    assert result['output'] == 'parent-complete'
    assert [type(message).__name__ for message in llm.calls[0]] == ['SystemMessage', 'HumanMessage']
    assert [message.content for message in llm.calls[0] if isinstance(message, HumanMessage)] == ['Delegate this task']
