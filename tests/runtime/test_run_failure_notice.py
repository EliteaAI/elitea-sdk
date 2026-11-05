"""The SDK tags its own failure reports so a consumer need not read the text.

A failed node, a stopped pipeline and a step limit reach the caller as assistant
messages whose text reads like a normal answer. Each carries
``response_metadata['run_failure_notice'] = True``; a normal answer carries none.
"""

import json
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from elitea_sdk.runtime.langchain.langraph_agent import _make_mcp_auth_skip_node
from elitea_sdk.runtime.tool_outcome import RUN_FAILURE_NOTICE_KEY, run_failure_message
from elitea_sdk.runtime.tools.function import PIPELINE_BLOCKED_KEY, FunctionTool
from elitea_sdk.runtime.tools.indexer_tool import IndexerNode
from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime.tools.loop_output import LoopToolNode
from elitea_sdk.runtime.tools.tool import ToolNode


class _Args(BaseModel):
    issue_number: str = Field(description="The issue number")


def _tool(func, name="update_issue"):
    return StructuredTool.from_function(
        func=func,
        name=name,
        description="Update an issue",
        args_schema=_Args,
        metadata={"toolkit_type": "jira", "toolkit_name": "jira"},
    )


def _function_node(tool, name="issue_node"):
    return FunctionTool(
        name=name,
        tool=tool,
        input_mapping={"issue_number": {"type": "fixed", "value": "42"}},
        input_variables=[],
        output_variables=["issue", "messages"],
    )


def _run(node):
    with patch("elitea_sdk.runtime.tools.function.dispatch_custom_event"):
        return node.invoke({})


def _after_add_messages(messages):
    """What the graph state holds after the reducer converts dicts to messages."""
    return add_messages([], messages)


def _raise_runtime(issue_number):
    raise RuntimeError("connection refused")


def _assert_tagged(messages):
    assert len(messages) == 1
    stored = _after_add_messages(messages)[0]
    assert isinstance(stored, AIMessage)
    assert stored.response_metadata.get(RUN_FAILURE_NOTICE_KEY) is True


class TestRunFailureMessage:
    def test_dict_survives_add_messages_as_ai_message(self):
        stored = _after_add_messages([run_failure_message("boom")])[0]

        assert isinstance(stored, AIMessage)
        assert stored.content == "boom"
        assert stored.response_metadata == {RUN_FAILURE_NOTICE_KEY: True}


class TestFunctionTool:
    def test_exception_branch_message_is_tagged(self):
        result = _run(_function_node(_tool(_raise_runtime)))

        _assert_tagged(result["messages"])
        content = result["messages"][0]["content"]
        assert content.startswith("Tool input to the update_issue with value")
        assert "raised Exception" in content
        assert "connection refused" in content

    def test_successful_call_message_is_not_tagged(self):
        result = _run(_function_node(_tool(lambda issue_number: "done")))

        stored = _after_add_messages(result["messages"])[0]
        assert RUN_FAILURE_NOTICE_KEY not in stored.response_metadata

    def test_blocked_termination_message_is_tagged(self):
        node = _function_node(_tool(lambda issue_number: "x"))
        blocked = json.dumps({
            "type": "sensitive_tool_blocked",
            "blocked_tool_name": "update_issue",
            "blocked_toolkit_type": "jira",
        })

        result = node._build_blocked_termination(blocked)

        _assert_tagged(result["messages"])
        assert result["messages"][0]["content"] == result[PIPELINE_BLOCKED_KEY]
        assert result["messages"][0]["content"].startswith("**Pipeline stopped**")

    def test_mcp_auth_skipped_message_is_tagged(self):
        node = _function_node(_tool(lambda issue_number: "x"))

        result = node._build_mcp_auth_skipped_termination(
            {"tool_name": "update_issue", "toolkit_name": "jira"}
        )

        _assert_tagged(result["messages"])
        assert "was skipped" in result["messages"][0]["content"]

    def test_mcp_auth_refresh_message_is_tagged(self):
        node = _function_node(_tool(lambda issue_number: "x"))

        result = node._build_mcp_auth_refresh_termination(
            {"tool_name": "update_issue", "toolkit_name": "jira"}
        )

        _assert_tagged(result["messages"])
        assert "authorized Toolkit was not available" in result["messages"][0]["content"]


class TestGraphLevelAuthSkipNode:
    def test_skip_node_message_is_tagged(self):
        result = _make_mcp_auth_skip_node("n", "jira", ["issue"])({})

        _assert_tagged(result["messages"])
        assert result["messages"][0]["content"].startswith("**Pipeline stopped**")
        assert result[PIPELINE_BLOCKED_KEY] == result["messages"][0]["content"]


class _JsonClient:
    """Answers the argument-forming call with JSON that fails the tool schema."""

    def invoke(self, messages, config=None):
        return AIMessage(content='{"issue_number": 5}')


def _make_schema_tool():
    return _tool(lambda issue_number: "ok")


class TestValidationFailureNodes:
    def test_tool_node_validation_error_is_tagged(self):
        node = ToolNode(
            client=_JsonClient(), tool=_make_schema_tool(),
            input_variables=["text"], output_variables=["out"],
        )
        with patch("elitea_sdk.runtime.tools.tool.dispatch_custom_event"):
            result = node.invoke({"text": "go"})

        _assert_tagged(result["messages"])
        assert "raised ValidationError" in result["messages"][0]["content"]

    def test_loop_tool_node_validation_error_is_tagged(self):
        node = LoopToolNode(
            client=_JsonClient(), tool=_make_schema_tool(), loop_tool=_make_schema_tool(),
            input_variables=["text"], output_variables=["out"],
        )
        with patch("elitea_sdk.runtime.tools.loop_output.dispatch_custom_event"):
            result = node.invoke({"text": "go"})

        _assert_tagged(result["messages"])
        assert "raised ValidationError" in result["messages"][0]["content"]

    def test_indexer_node_chunking_error_is_tagged(self):
        node = IndexerNode(
            tool=_make_schema_tool(), chunking_tool="no_such_chunker", chunking_config={},
            input_mapping={"issue_number": {"type": "fixed", "value": "1"}},
            input_variables=[], output_variables=["out"],
        )
        with patch("elitea_sdk.runtime.tools.indexer_tool.dispatch_custom_event"):
            result = node.invoke({})

        _assert_tagged(result["messages"])
        assert "no_such_chunker raised an error" in result["messages"][0]["content"]

    def test_indexer_node_index_error_is_tagged(self):
        node = IndexerNode(
            tool=_make_schema_tool(), index_tool=_tool(_raise_runtime, name="index_docs"),
            input_mapping={"issue_number": {"type": "fixed", "value": "1"}},
            input_variables=[], output_variables=["out"],
        )
        with patch("elitea_sdk.runtime.tools.indexer_tool.dispatch_custom_event"):
            result = node.invoke({})

        _assert_tagged(result["messages"])
        assert "raised ValidationError" in result["messages"][0]["content"]


def _read_tool():
    return StructuredTool.from_function(
        func=lambda path='x': 'file contents',
        name='read_file',
        description='read_file tool.',
        metadata={'toolkit_type': 'fs', 'toolkit_name': 'fs', 'tool_name': 'read_file'},
    )


class _ScriptedClient:
    """Plays one scripted step per invoke: an AIMessage is returned, an exception is raised."""

    def __init__(self, steps):
        self.steps = list(steps)

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        return step


def _tool_call(n):
    return AIMessage(content='', tool_calls=[{'name': 'read_file', 'args': {'path': f'f{n}'}, 'id': f'call_{n}'}])


def _llm_node(client, steps_limit=None):
    node = LLMNode(
        client=client,
        available_tools=[_read_tool()],
        tool_names=['read_file'],
        lazy_tools_mode=False,
        input_mapping={},
        output_variables=['messages'],
    )
    if steps_limit is not None:
        node.steps_limit = steps_limit
    return node


def _ai_messages(result):
    return [m for m in result['messages'] if isinstance(m, AIMessage)]


class TestLlmNode:
    def test_step_limit_message_is_tagged(self):
        client = _ScriptedClient([_tool_call(1), _tool_call(2), _tool_call(3)])
        result = _llm_node(client, steps_limit=1).invoke(
            {'messages': [HumanMessage(content='go')]}, config={'configurable': {'thread_id': 't-notice'}},
        )

        limit = [m for m in _ai_messages(result) if 'Maximum tool execution iterations' in str(m.content)]
        assert len(limit) == 1
        assert limit[0].response_metadata.get(RUN_FAILURE_NOTICE_KEY) is True
        assert 'Stopping tool execution.' in limit[0].content

    def test_failed_retry_after_truncation_message_is_tagged(self):
        client = _ScriptedClient([
            _tool_call(1),
            RuntimeError('maximum context length exceeded'),
            RuntimeError('provider down'),
        ])
        result = _llm_node(client).invoke(
            {'messages': [HumanMessage(content='go')]}, config={'configurable': {'thread_id': 't-notice'}},
        )

        failed = [m for m in _ai_messages(result) if str(m.content).startswith('Failed to retry after truncation')]
        assert len(failed) == 1
        assert failed[0].response_metadata.get(RUN_FAILURE_NOTICE_KEY) is True

    def test_normal_final_answer_is_not_tagged(self):
        client = _ScriptedClient([_tool_call(1), AIMessage(content='all done')])
        result = _llm_node(client).invoke(
            {'messages': [HumanMessage(content='go')]}, config={'configurable': {'thread_id': 't-notice'}},
        )

        answers = _ai_messages(result)
        assert answers[-1].content == 'all done'
        assert all(RUN_FAILURE_NOTICE_KEY not in m.response_metadata for m in answers)
