"""A plain approve runs the approved sensitive call once, even when ainvoke falls back to invoke."""
from typing import Any, Optional, Type

import asyncio
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.middleware.sensitive_tool_guard import SensitiveToolGuardMiddleware
from elitea_sdk.runtime.toolkits.security import configure_sensitive_tools, reset_sensitive_tools


def setup_function() -> None:
    reset_sensitive_tools()


def teardown_function() -> None:
    reset_sensitive_tools()


class _DummyRuntime:
    def get_mcp_toolkits(self):
        return []


class _CreateArgs(BaseModel):
    title: str = ''


class _SyncOnlyTool(BaseTool):
    """A toolkit-style tool with a sync path only: its async call raises NotImplementedError."""

    name: str = 'create_issue'
    description: str = 'Create an issue'
    args_schema: Optional[Type[BaseModel]] = _CreateArgs
    runs: list = []

    def _run(self, title: str = '', **kwargs: Any) -> str:
        self.runs.append(title)
        return 'create-result:PROJ-1'

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError('sync only')


class _CallsToolLLM:
    """Asks for one sensitive call, then answers once the tool result is in the messages."""

    def __init__(self):
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        if any(isinstance(m, ToolMessage) and 'create-result:PROJ-1' in str(m.content) for m in messages):
            return AIMessage(content='Done: PROJ-1')
        return AIMessage(
            content='',
            tool_calls=[{'name': 'create_issue', 'args': {'title': 'T'}, 'id': 'call-create-1'}],
        )


def _build_runnable(memory: MemorySaver, tool: BaseTool):
    assistant = Assistant(
        elitea=_DummyRuntime(),
        data={"instructions": "Use tools.", "tools": [], "meta": {}},
        client=_CallsToolLLM(),
        tools=[tool],
        memory=memory,
        app_type="predict",
        middleware=[SensitiveToolGuardMiddleware()],
    )
    return assistant.runnable()


@pytest.fixture
def tool() -> _SyncOnlyTool:
    return _SyncOnlyTool(
        metadata={'toolkit_type': 'jira', 'toolkit_name': 'jira', 'tool_name': 'create_issue'},
        runs=[],
    )


def test_plain_approve_runs_sync_only_tool_once_without_second_pause(tool):
    configure_sensitive_tools({'jira': ['create_issue']})
    memory = MemorySaver()
    thread_cfg = {'configurable': {'thread_id': 'plain-approve-sync-only'}}

    first = _build_runnable(memory, tool).invoke(
        {'messages': [HumanMessage(content='Create a task')]}, config={**thread_cfg},
    )
    assert first['execution_finished'] is False
    assert first['hitl_interrupt']['tool_name'] == 'create_issue'
    assert tool.runs == []

    resumed = _build_runnable(memory, tool).invoke(
        {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''}, config={**thread_cfg},
    )

    assert not resumed.get('hitl_interrupt'), 'the approved call paused again'
    assert resumed['execution_finished'] is True
    assert 'PROJ-1' in resumed.get('output', '')
    assert tool.runs == ['T']


def test_reject_of_sync_only_tool_runs_nothing_and_does_not_pause_again(tool):
    configure_sensitive_tools({'jira': ['create_issue']})
    memory = MemorySaver()
    thread_cfg = {'configurable': {'thread_id': 'reject-sync-only'}}

    _build_runnable(memory, tool).invoke(
        {'messages': [HumanMessage(content='Create a task')]}, config={**thread_cfg},
    )
    rejected = _build_runnable(memory, tool).invoke(
        {'hitl_resume': True, 'hitl_action': 'reject', 'hitl_value': 'no'}, config={**thread_cfg},
    )

    assert not rejected.get('hitl_interrupt')
    assert tool.runs == []


class _ToggleAsyncTool(BaseTool):
    """Async path fails until `async_ok` is set; the sync path always works."""

    name: str = 'create_issue'
    description: str = 'Create an issue'
    args_schema: Optional[Type[BaseModel]] = _CreateArgs
    async_ok: bool = False

    def _run(self, title: str = '', **kwargs: Any) -> str:
        return 'ok'

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        if not self.async_ok:
            raise NotImplementedError('sync only')
        return 'ok-async'


def _wrapped_toggle_tool():
    configure_sensitive_tools({'jira': ['create_issue']})
    tool = _ToggleAsyncTool(metadata={'toolkit_type': 'jira', 'toolkit_name': 'jira', 'tool_name': 'create_issue'})
    return tool, SensitiveToolGuardMiddleware().wrap_tool(tool)


def _run_with_review_count(coro_factory, action='approve'):
    with patch.object(
        SensitiveToolGuardMiddleware, '_review_sensitive_tool_call',
        return_value={'action': action, 'value': ''},
    ) as review:
        asyncio.run(coro_factory())
    return review.call_count


def test_sync_fallback_after_async_failure_reviews_the_call_once():
    _, wrapped = _wrapped_toggle_tool()

    async def scenario():
        try:
            await wrapped.ainvoke({'title': 'T'})
        except NotImplementedError:
            wrapped.invoke({'title': 'T'})

    assert _run_with_review_count(scenario) == 1


def test_next_sync_call_with_same_input_is_reviewed_after_the_fallback():
    _, wrapped = _wrapped_toggle_tool()

    async def scenario():
        try:
            await wrapped.ainvoke({'title': 'T'})
        except NotImplementedError:
            wrapped.invoke({'title': 'T'})
        wrapped.invoke({'title': 'T'})

    assert _run_with_review_count(scenario) == 2


def test_sync_call_with_other_input_is_reviewed_after_async_failure():
    _, wrapped = _wrapped_toggle_tool()

    async def scenario():
        try:
            await wrapped.ainvoke({'title': 'T'})
        except NotImplementedError:
            wrapped.invoke({'title': 'other'})

    assert _run_with_review_count(scenario) == 2


def test_new_async_call_drops_the_approval_left_by_an_earlier_failure():
    _, wrapped = _wrapped_toggle_tool()

    async def scenario():
        try:
            await wrapped.ainvoke({'title': 'T'})
        except NotImplementedError:
            pass
        wrapped.async_ok = True
        await wrapped.ainvoke({'title': 'T'})
        wrapped.invoke({'title': 'T'})

    assert _run_with_review_count(scenario) == 3


def test_rejected_async_call_leaves_no_approval_for_a_sync_call():
    _, wrapped = _wrapped_toggle_tool()

    async def scenario():
        await wrapped.ainvoke({'title': 'T'})
        wrapped.invoke({'title': 'T'})

    assert _run_with_review_count(scenario, action='reject') == 2


class _SyncBodyTool(BaseTool):
    """A toolkit-style tool with a `_run` only: the base class supplies the async path."""

    name: str = 'create_issue'
    description: str = 'Create an issue'
    args_schema: Optional[Type[BaseModel]] = _CreateArgs
    runs: list = []

    def _run(self, title: str = '', **kwargs: Any) -> str:
        self.runs.append(title)
        return 'create-result:PROJ-1'


class _HandlerWithoutRunInline:
    """A callback handler like the one of the audit callback: no `run_inline` attribute.

    langchain reads `run_inline` while it dispatches a tool event on the async path, so the
    async call of the tool raises AttributeError after the guard approved it.
    """

    ignore_llm = ignore_chain = ignore_agent = ignore_retriever = ignore_chat_model = False
    ignore_custom_event = ignore_retry = ignore_tool = False
    raise_error = False

    def __getattr__(self, name: str):
        if name.startswith('on_'):
            return lambda *args, **kwargs: None
        raise AttributeError(name)


def test_plain_approve_with_callback_handler_failing_the_async_path_runs_tool_once():
    configure_sensitive_tools({'jira': ['create_issue']})
    tool = _SyncBodyTool(
        metadata={'toolkit_type': 'jira', 'toolkit_name': 'jira', 'tool_name': 'create_issue'}, runs=[],
    )
    memory = MemorySaver()
    thread_cfg = {'configurable': {'thread_id': 'plain-approve-handler-attribute-error'}}
    run_cfg = {**thread_cfg, 'callbacks': [_HandlerWithoutRunInline()]}

    first = _build_runnable(memory, tool).invoke(
        {'messages': [HumanMessage(content='Create a task')]}, config={**run_cfg},
    )
    resumed = _build_runnable(memory, tool).invoke(
        {'hitl_resume': True, 'hitl_action': 'approve', 'hitl_value': ''}, config={**run_cfg},
    )

    assert first['hitl_interrupt']['tool_name'] == 'create_issue'
    assert not resumed.get('hitl_interrupt'), 'the approved call paused again'
    assert resumed['execution_finished'] is True
    assert tool.runs == ['T']
