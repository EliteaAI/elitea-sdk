"""The SDK tags its own failure reports so a consumer need not read the text.

A failed node, a stopped pipeline and a step limit reach the caller as assistant
messages whose text reads like a normal answer. Each carries
``response_metadata['run_failure_notice'] = True``; a normal answer carries none.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from elitea_sdk.runtime.langchain.langraph_agent import LangGraphAgentRunnable, _make_mcp_auth_skip_node
from elitea_sdk.runtime.tool_outcome import RUN_FAILURE_NOTICE_KEY, run_failure_message
from elitea_sdk.runtime.tools.agent import AgentNode
from elitea_sdk.runtime.tools.application import Application
from elitea_sdk.runtime.tools.function import PIPELINE_BLOCKED_KEY, FunctionTool
from elitea_sdk.runtime.tools.indexer_tool import IndexerNode
from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime.tools.loop_output import LoopToolNode
from elitea_sdk.runtime.tools.tool import ToolNode


FAILED_NOTICE = "The tool {name} failed. Open the run log for details."


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
    def test_exception_branch_message_hides_arguments_schema_and_error(self):
        result = _run(_function_node(_tool(_raise_runtime)))

        _assert_tagged(result["messages"])
        content = result["messages"][0]["content"]
        assert "update_issue" in content
        assert "connection refused" not in content
        assert "42" not in content
        assert "Tool schema" not in content

    def test_exception_branch_logs_the_detail_but_not_the_arguments(self, caplog):
        with caplog.at_level("ERROR", logger="elitea_sdk.runtime.tools.function"):
            _run(_function_node(_tool(_raise_runtime)))

        failed = [r for r in caplog.records if r.name == "elitea_sdk.runtime.tools.function" and r.exc_info]
        assert len(failed) == 1
        assert "connection refused" in str(failed[0].exc_info[1])
        assert "42" not in failed[0].getMessage()

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

    def test_indexer_node_chunking_error_has_no_traceback(self):
        node = IndexerNode(
            tool=_make_schema_tool(), chunking_tool="no_such_chunker", chunking_config={},
            input_mapping={"issue_number": {"type": "fixed", "value": "1"}},
            input_variables=[], output_variables=["out"],
        )
        with patch("elitea_sdk.runtime.tools.indexer_tool.dispatch_custom_event"):
            result = node.invoke({})

        content = result["messages"][0]["content"]
        assert "Traceback" not in content
        assert "File \"" not in content

    def test_agent_node_validation_error_is_tagged(self):
        node = AgentNode(
            tool=_make_schema_tool(), task="Do {text}",
            input_variables=["text"], output_variables=["out"],
        )
        with patch("elitea_sdk.runtime.tools.agent.dispatch_custom_event"):
            result = node.invoke({"text": "go"})

        _assert_tagged(result["messages"])
        assert result["messages"][0]["content"] == FAILED_NOTICE.format(name="update_issue")

    def test_agent_node_notice_hides_input_schema_and_error_but_logs_detail(self, caplog):
        node = AgentNode(
            tool=_make_schema_tool(), task="Do {text}",
            input_variables=["text"], output_variables=["out"],
        )
        with patch("elitea_sdk.runtime.tools.agent.dispatch_custom_event"):
            with caplog.at_level("ERROR", logger="elitea_sdk.runtime.tools.agent"):
                result = node.invoke({"text": "secret-input"})

        content = result["messages"][0]["content"]
        assert "secret-input" not in content
        assert "Tool schema" not in content
        assert "issue_number" not in content
        assert "validation error" not in content.lower()
        assert any(r.exc_info for r in caplog.records)

    def test_indexer_node_index_error_is_tagged(self):
        node = IndexerNode(
            tool=_make_schema_tool(), index_tool=_tool(_raise_runtime, name="index_docs"),
            input_mapping={"issue_number": {"type": "fixed", "value": "1"}},
            input_variables=[], output_variables=["out"],
        )
        with patch("elitea_sdk.runtime.tools.indexer_tool.dispatch_custom_event"):
            result = node.invoke({})

        _assert_tagged(result["messages"])
        assert result["messages"][0]["content"] == FAILED_NOTICE.format(name="update_issue")

    def test_indexer_node_notice_hides_input_schema_and_error_but_logs_detail(self, caplog):
        node = IndexerNode(
            tool=_tool(lambda issue_number: "secret-document"),
            index_tool=StructuredTool.from_function(
                func=lambda documents: _raise_runtime(documents),
                name="index_docs", description="Index documents"),
            input_mapping={"issue_number": {"type": "fixed", "value": "1"}},
            input_variables=[], output_variables=["out"],
        )
        with patch("elitea_sdk.runtime.tools.indexer_tool.dispatch_custom_event"):
            with caplog.at_level("ERROR", logger="elitea_sdk.runtime.tools.indexer_tool"):
                result = node.invoke({})

        content = result["messages"][0]["content"]
        assert "secret-document" not in content
        assert "connection refused" not in content
        assert "Tool schema" not in content
        assert "connection refused" in caplog.text

    def test_indexer_node_notice_when_the_tool_call_itself_fails(self):
        node = IndexerNode(
            tool=_tool(_raise_runtime),
            input_mapping={"issue_number": {"type": "fixed", "value": "1"}},
            input_variables=[], output_variables=["out"],
        )
        with patch("elitea_sdk.runtime.tools.indexer_tool.dispatch_custom_event"):
            result = node.invoke({})

        _assert_tagged(result["messages"])
        assert result["messages"][0]["content"] == FAILED_NOTICE.format(name="update_issue")


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


class _RaisingCompletionClient(_ScriptedClient):
    """Fails every call with the same exception."""

    def __init__(self, error):
        super().__init__([error])


def _run_llm(node):
    return node.invoke(
        {'messages': [HumanMessage(content='go')]}, config={'configurable': {'thread_id': 't-notice'}},
    )


def _error_messages(result, text):
    return [m for m in _ai_messages(result) if text in str(m.content)]


class TestLlmProviderErrors:
    def test_provider_error_in_the_first_call_carries_both_tags(self):
        result = _run_llm(_llm_node(_RaisingCompletionClient(RuntimeError('provider down'))))

        error = _ai_messages(result)[-1]
        assert error.content.startswith('Error:')
        assert error.response_metadata.get('llm_node_error') is True
        assert error.response_metadata.get(RUN_FAILURE_NOTICE_KEY) is True

    @pytest.mark.parametrize('error, expected', [
        pytest.param(
            httpx.ReadTimeout('timed out'),
            'Error: LLM stream stalled: no data received. The model provider stopped responding; please retry.',
            id='stalled-stream',
        ),
        pytest.param(RuntimeError('Error code: 400 - bad request'), 'Error: Error code: 400 - bad request', id='http-400'),
    ])
    def test_provider_error_answer_keeps_the_text_the_platform_matches(self, error, expected):
        # elitea_core marks a model failure by this exact "Error: " prefix and shows the text after it
        result = _run_llm(_llm_node(_RaisingCompletionClient(error)))

        assert _ai_messages(result)[-1].content == expected

    @pytest.mark.parametrize('error_text, expected', [
        pytest.param('invalid credentials for the model', 'Model configuration error', id='non-recoverable'),
        pytest.param('provider down', 'Error processing tool results', id='other'),
    ])
    def test_provider_error_in_the_tool_loop_carries_both_tags(self, error_text, expected):
        client = _ScriptedClient([_tool_call(1), RuntimeError(error_text)])
        result = _run_llm(_llm_node(client))

        failed = _error_messages(result, expected)
        assert len(failed) == 1
        assert failed[0].response_metadata.get('llm_node_error') is True
        assert failed[0].response_metadata.get(RUN_FAILURE_NOTICE_KEY) is True

    @pytest.mark.parametrize('error_text, expected', [
        pytest.param('maximum context length exceeded', 'Context window exceeded', id='context-window'),
        pytest.param('output token limit reached', 'Model output limit exceeded', id='output-limit'),
    ])
    def test_context_or_output_limit_without_a_tool_message_is_tagged(self, error_text, expected):
        client = _ScriptedClient([_tool_call(1), RuntimeError(error_text)])
        node = _llm_node(client)
        # The follow-up call sees a history with no tool message in it
        with patch.object(
            LLMNode, '_filter_orphaned_tool_calls',
            side_effect=lambda msgs: [m for m in msgs if not isinstance(m, ToolMessage) and not getattr(m, 'tool_calls', None)],
        ):
            result = _run_llm(node)

        failed = _error_messages(result, expected)
        assert len(failed) == 1
        assert failed[0].response_metadata.get(RUN_FAILURE_NOTICE_KEY) is True


class _StubRunnable:
    """Stands in for a child agent or pipeline: returns the scripted response."""

    def __init__(self, response):
        self.response = response

    def invoke(self, payload, config=None, **kwargs):
        return self.response


def _nested_app(response):
    return Application(
        name='child', description='child agent', application=_StubRunnable(response), client=None,
    )


def _notice(text='Pipeline stopped'):
    return AIMessage(content=text, response_metadata={RUN_FAILURE_NOTICE_KEY: True})


class TestNestedApplication:
    @pytest.mark.parametrize('response, expected', [
        pytest.param(
            {'messages': [HumanMessage(content='go'), _notice()], 'output': 'x'}, True, id='last-message-tagged',
        ),
        pytest.param(
            {'messages': [HumanMessage(content='go'), {'role': 'assistant', 'content': 'x', 'response_metadata': {RUN_FAILURE_NOTICE_KEY: True}}]},
            True, id='last-message-tagged-dict',
        ),
        pytest.param(
            {'messages': [HumanMessage(content='go'), AIMessage(content='ok')], PIPELINE_BLOCKED_KEY: 'blocked'},
            True, id='pipeline-blocked',
        ),
        pytest.param(
            {'messages': [HumanMessage(content='go'), AIMessage(content='ok')], 'tool_execution_limit_reached': True},
            True, id='step-limit',
        ),
        pytest.param(
            {'messages': [_notice(), HumanMessage(content='go'), AIMessage(content='ok')]},
            False, id='tag-before-this-run',
        ),
        pytest.param(
            {'messages': [HumanMessage(content='go'), _notice(), AIMessage(content='recovered')]},
            False, id='clean-last-reply',
        ),
        pytest.param({'messages': [], 'output': 'text'}, False, id='no-messages'),
        pytest.param('not a dict', False, id='not-a-dict'),
    ])
    def test_child_failed_reads_the_tag_the_blocked_key_and_the_step_limit(self, response, expected):
        from elitea_sdk.runtime.tools.application import _child_failed

        assert _child_failed(response) is expected

    def test_a_failed_child_reaches_the_parent_as_a_tagged_message(self):
        response = {
            'messages': [HumanMessage(content='go'), _notice('The tool failed')],
            'output': 'The tool failed',
        }

        result = _nested_app(response)._run(task='go')

        _assert_tagged(result['messages'])
        assert result['messages'][0]['content'] == 'The tool failed'

    def test_a_blocked_child_reaches_the_parent_as_a_tagged_message(self):
        response = {
            'messages': [HumanMessage(content='go'), AIMessage(content='stopped')],
            'output': 'stopped',
            PIPELINE_BLOCKED_KEY: 'stopped',
        }

        result = _nested_app(response)._run(task='go')

        _assert_tagged(result['messages'])

    def test_a_clean_child_reply_is_not_tagged(self):
        response = {'messages': [HumanMessage(content='go'), AIMessage(content='all done')], 'output': 'all done'}

        result = _nested_app(response)._run(task='go')

        stored = _after_add_messages(result['messages'])[0]
        assert stored.content == 'all done'
        assert RUN_FAILURE_NOTICE_KEY not in stored.response_metadata

    @pytest.mark.parametrize('flag, value', [
        pytest.param(PIPELINE_BLOCKED_KEY, 'stopped', id='pipeline-blocked'),
        pytest.param('tool_execution_limit_reached', True, id='step-limit'),
    ])
    def test_a_child_that_stopped_reaches_the_parent_as_a_tagged_message(self, flag, value):
        response = {'messages': [HumanMessage(content='go'), AIMessage(content='stopped')], 'output': 'stopped', flag: value}

        result = _nested_app(response)._run(task='go')

        _assert_tagged(result['messages'])


class TestNamesTheWorkerReads:
    """The indexer worker reads these names as plain strings, so a rename must fail a test here."""

    def test_the_tag_and_the_blocked_key_keep_their_names(self):
        assert RUN_FAILURE_NOTICE_KEY == 'run_failure_notice'
        assert PIPELINE_BLOCKED_KEY == '_pipeline_blocked'
        assert run_failure_message('text') == {
            'role': 'assistant', 'content': 'text', 'response_metadata': {'run_failure_notice': True},
        }

    def test_a_run_stopped_at_the_step_limit_carries_the_flag_and_the_output(self):
        fake = SimpleNamespace(get_state=lambda config: SimpleNamespace(values={'messages': [], 'output': 'old'}))

        result = LangGraphAgentRunnable._handle_graph_recursion_error(fake, {}, 'thread-1', 25)

        assert result['tool_execution_limit_reached'] is True
        assert result['execution_finished'] is False
        assert result['output'].startswith('Tool step limit 25 reached')
