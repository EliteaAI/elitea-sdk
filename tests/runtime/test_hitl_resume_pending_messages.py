from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.middleware.sensitive_tool_guard import SensitiveToolGuardMiddleware
from elitea_sdk.runtime.toolkits.security import configure_sensitive_tools, reset_sensitive_tools


def setup_function() -> None:
    reset_sensitive_tools()


def teardown_function() -> None:
    reset_sensitive_tools()


class DummyEliteARuntime:
    def get_mcp_toolkits(self):
        return []


class ResumeReplayLLM:
    def __init__(self):
        self.calls = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _ResumeReplayLLMBound(self, tools, kwargs)

    def invoke(self, messages, config=None):
        return _ResumeReplayLLMBound(self, [], {}).invoke(messages, config=config)


class _ResumeReplayLLMBound:
    def __init__(self, root, tools, kwargs):
        self.root = root
        self.tools = list(tools)
        self.kwargs = dict(kwargs)

    def invoke(self, messages, config=None):
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        tool_contents = [str(message.content) for message in tool_messages]
        ai_tool_calls = [
            getattr(message, 'tool_calls', None)
            for message in messages
            if isinstance(message, AIMessage)
        ]
        self.root.calls.append(
            {
                'bound_tools': [tool.name for tool in self.tools],
                'tool_contents': tool_contents,
                'ai_tool_calls': ai_tool_calls,
            }
        )

        if {'safe1-ok', 'safe2-ok', 'danger-ok'}.issubset(set(tool_contents)):
            return AIMessage(content='FINAL')

        if 'danger-ok' in tool_contents:
            return AIMessage(
                content='',
                tool_calls=[
                    {'name': 'safe1', 'args': {}, 'id': 'redo-safe1'},
                    {'name': 'safe2', 'args': {}, 'id': 'redo-safe2'},
                ],
            )

        return AIMessage(
            content='',
            tool_calls=[
                {'name': 'safe1', 'args': {}, 'id': 'call-safe1'},
                {'name': 'safe2', 'args': {}, 'id': 'call-safe2'},
                {'name': 'danger', 'args': {}, 'id': 'call-danger'},
            ],
        )


def _build_resume_repro_runnable(memory, llm):
    assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'Use tools', 'tools': [], 'meta': {}},
        client=llm,
        tools=[
            StructuredTool.from_function(
                func=lambda: 'safe1-ok',
                name='safe1',
                description='safe1',
                metadata={'toolkit_type': 'dummy', 'toolkit_name': 'dummy', 'tool_name': 'safe1'},
            ),
            StructuredTool.from_function(
                func=lambda: 'safe2-ok',
                name='safe2',
                description='safe2',
                metadata={'toolkit_type': 'dummy', 'toolkit_name': 'dummy', 'tool_name': 'safe2'},
            ),
            StructuredTool.from_function(
                func=lambda: 'danger-ok',
                name='danger',
                description='danger',
                metadata={'toolkit_type': 'dummy', 'toolkit_name': 'dummy', 'tool_name': 'danger'},
            ),
        ],
        memory=memory,
        app_type='predict',
        middleware=[SensitiveToolGuardMiddleware()],
    )
    return assistant.runnable()


class SequentialDangerLLM:
    """Plans sensitive tools one-at-a-time across successive resume cycles.

    danger1 -> danger2 -> danger3 -> FINAL, each interrupting. FINAL is only
    emitted once ALL THREE results are visible. The bug only bites from the
    third interrupt onward: danger3's resume must carry danger1-ok forward,
    and danger1-ok lives in the *restored* region of that cycle (below the
    post-restore message count). If the capture window is anchored there the
    result is shed, the model loses sight of danger1, and it re-plans from the
    beginning instead of finishing (#5245).
    """

    def __init__(self):
        self.calls = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {'temperature': self.temperature, 'max_tokens': self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _SequentialDangerBound(self, tools, kwargs)

    def invoke(self, messages, config=None):
        return _SequentialDangerBound(self, [], {}).invoke(messages, config=config)


class _SequentialDangerBound:
    def __init__(self, root, tools, kwargs):
        self.root = root
        self.tools = list(tools)
        self.kwargs = dict(kwargs)

    def invoke(self, messages, config=None):
        tool_contents = {
            str(m.content) for m in messages if isinstance(m, ToolMessage)
        }
        self.root.calls.append({'tool_contents': set(tool_contents)})

        if {'danger1-ok', 'danger2-ok', 'danger3-ok'}.issubset(tool_contents):
            return AIMessage(content='FINAL')
        if {'danger1-ok', 'danger2-ok'}.issubset(tool_contents):
            return AIMessage(
                content='',
                tool_calls=[{'name': 'danger3', 'args': {}, 'id': 'call-danger3'}],
            )
        if 'danger1-ok' in tool_contents:
            return AIMessage(
                content='',
                tool_calls=[{'name': 'danger2', 'args': {}, 'id': 'call-danger2'}],
            )
        return AIMessage(
            content='',
            tool_calls=[{'name': 'danger1', 'args': {}, 'id': 'call-danger1'}],
        )


def _build_sequential_danger_runnable(memory, llm):
    assistant = Assistant(
        elitea=DummyEliteARuntime(),
        data={'instructions': 'Use tools', 'tools': [], 'meta': {}},
        client=llm,
        tools=[
            StructuredTool.from_function(
                func=lambda: 'danger1-ok',
                name='danger1',
                description='danger1',
                metadata={'toolkit_type': 'dummy', 'toolkit_name': 'dummy', 'tool_name': 'danger1'},
            ),
            StructuredTool.from_function(
                func=lambda: 'danger2-ok',
                name='danger2',
                description='danger2',
                metadata={'toolkit_type': 'dummy', 'toolkit_name': 'dummy', 'tool_name': 'danger2'},
            ),
            StructuredTool.from_function(
                func=lambda: 'danger3-ok',
                name='danger3',
                description='danger3',
                metadata={'toolkit_type': 'dummy', 'toolkit_name': 'dummy', 'tool_name': 'danger3'},
            ),
        ],
        memory=memory,
        app_type='predict',
        middleware=[SensitiveToolGuardMiddleware()],
    )
    return assistant.runnable()


def test_hitl_resume_accumulates_history_across_sequential_sensitive_tools():
    """Approving sequential sensitive tools must not shed earlier results.

    Regression for #5245: each resume cycle's pending capture window was
    anchored at the post-restore message count, so a prior cycle's executed
    tool result (sitting in the restored region) fell below the window and was
    dropped from the next interrupt's pending. The model then lost sight of the
    already-approved tool and re-planned from the beginning.
    """
    configure_sensitive_tools({'dummy': ['danger1', 'danger2', 'danger3']})

    memory = MemorySaver()
    tid = 'seq-danger-thread'

    def _resume_approve():
        runnable = _build_sequential_danger_runnable(memory, SequentialDangerLLM())
        return runnable.invoke(
            {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''},
            config={'configurable': {'thread_id': tid}},
        )

    first = _build_sequential_danger_runnable(memory, SequentialDangerLLM())
    r1 = first.invoke(
        {'messages': [HumanMessage(content='go')]},
        config={'configurable': {'thread_id': tid}},
    )
    assert r1['execution_finished'] is False
    assert r1['hitl_interrupt']['tool_name'] == 'danger1'

    r2 = _resume_approve()  # danger1 executes -> plan danger2 -> interrupt
    assert r2['execution_finished'] is False
    assert r2['hitl_interrupt']['tool_name'] == 'danger2'

    r3 = _resume_approve()  # danger2 executes -> plan danger3 -> interrupt
    assert r3['execution_finished'] is False, (
        "danger3 was not reached — danger1-ok was likely shed and the model "
        "re-planned from the beginning"
    )
    assert r3['hitl_interrupt']['tool_name'] == 'danger3'

    r4 = _resume_approve()  # danger3 executes -> FINAL (needs all three)
    assert r4['execution_finished'] is True
    assert r4['output'] == 'FINAL'


def test_hitl_resume_restores_pending_messages_before_replaying_llm():
    configure_sensitive_tools({'dummy': ['danger']})

    memory = MemorySaver()
    thread_config = {'configurable': {'thread_id': 'resume-repro-thread'}}

    initial_llm = ResumeReplayLLM()
    initial_runnable = _build_resume_repro_runnable(memory, initial_llm)

    initial_result = initial_runnable.invoke(
        {'messages': [HumanMessage(content='do the thing')]},
        config=thread_config,
    )

    assert initial_result['execution_finished'] is False
    assert initial_result['hitl_interrupt']['tool_name'] == 'danger'

    resume_llm = ResumeReplayLLM()
    resumed_runnable = _build_resume_repro_runnable(memory, resume_llm)

    resume_result = resumed_runnable.invoke(
        {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''},
        config={'configurable': {'thread_id': 'resume-repro-thread'}},
    )

    assert resume_result['execution_finished'] is True
    assert resume_result['output'] == 'FINAL'
    assert len(resume_llm.calls) == 1
    assert {'safe1-ok', 'safe2-ok', 'danger-ok'} == set(resume_llm.calls[0]['tool_contents'])
    assert not any(
        tool_call and tool_call[0]['id'].startswith('redo-')
        for tool_call in resume_llm.calls[0]['ai_tool_calls']
        if tool_call
    )