"""Issue #6171: the tool outcome is projected onto dedicated pipeline state keys.

Before this, a pipeline author could only detect a tool failure by pattern-matching
the message history (``"error" in messages[-1].content``), which is wrong in both
directions: a successful payload containing the word "error" routes to the failure
branch, and a message-less predecessor makes the expression fail. These tests pin
the typed replacement and, just as importantly, that it is purely additive.
"""
import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool, ToolException
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from elitea_sdk.runtime.langchain.langraph_agent import create_graph
from elitea_sdk.runtime.middleware.base import MiddlewareManager
from elitea_sdk.runtime.middleware.strategies import (
    CircuitBreakerStrategy,
    LoggingStrategy,
    TransformErrorStrategy,
)
from elitea_sdk.runtime.middleware.tool_exception_handler import (
    ToolExceptionHandlerMiddleware,
)
from elitea_sdk.runtime.tool_outcome import ToolErrorClass, ToolResultStatus
from elitea_sdk.runtime.tools.function import (
    LAST_TOOL_OUTCOME_KEY,
    PIPELINE_BLOCKED_KEY,
    TOOL_OUTCOMES_KEY,
    FunctionTool,
)
from elitea_sdk.runtime.utils.evaluate import EvaluateTemplate


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


def _middleware():
    return ToolExceptionHandlerMiddleware(
        strategies=[
            TransformErrorStrategy(llm=None),
            CircuitBreakerStrategy(threshold=5),
            LoggingStrategy(),
        ]
    )


def _node(tool, output_variables=None, name="issue_node"):
    return FunctionTool(
        name=name,
        tool=tool,
        input_mapping={"issue_number": {"type": "fixed", "value": "42"}},
        input_variables=[],
        output_variables=output_variables if output_variables is not None else ["issue"],
    )


def _run(node):
    from unittest.mock import patch

    with patch("elitea_sdk.runtime.tools.function.dispatch_custom_event"):
        return node.invoke({})


def _outcome(result):
    return result[LAST_TOOL_OUTCOME_KEY]


# ─── Additive proof ──────────────────────────────────────────────────


class TestAdditive:
    """The declared output variables must be untouched, in value AND in type."""

    def test_success_output_variable_value_and_type_unchanged(self):
        node = _node(_tool(lambda issue_number: {"id": 7, "ok": True}))
        result = _run(node)

        assert result["issue"] == {"id": 7, "ok": True}
        assert isinstance(result["issue"], dict)

    def test_str_output_stays_str(self):
        node = _node(_tool(lambda issue_number: "plain string"))
        result = _run(node)

        assert result["issue"] == "plain string"
        assert type(result["issue"]) is str

    def test_only_the_two_documented_keys_are_added(self):
        node = _node(_tool(lambda issue_number: "x"))
        result = _run(node)

        assert set(result) == {"issue", TOOL_OUTCOMES_KEY, LAST_TOOL_OUTCOME_KEY}

    def test_outcome_keyed_by_node_id(self):
        node = _node(_tool(lambda issue_number: "x"), name="fetch_step")
        result = _run(node)

        assert set(result[TOOL_OUTCOMES_KEY]) == {"fetch_step"}
        assert result[TOOL_OUTCOMES_KEY]["fetch_step"] == result[LAST_TOOL_OUTCOME_KEY]


# ─── One expression, every failure mode ──────────────────────────────


def _raise_tool_exception(issue_number):
    raise ToolException("connection refused")


def _raise_runtime(issue_number):
    raise RuntimeError("connection refused")


def _raise_value(issue_number):
    raise ValueError("issue_number must be numeric")


def _raise_permission(issue_number):
    raise PermissionError("forbidden")


class TestFailureModes:
    """Every way a tool can fail must land on status == error, wrapped or not."""

    @pytest.mark.parametrize(
        "func",
        [_raise_tool_exception, _raise_runtime, _raise_permission],
        ids=["tool_exception", "runtime_error", "permission_error"],
    )
    def test_wrapped_failures_report_error(self, func):
        node = _node(_middleware().wrap_tool(_tool(func)))
        result = _run(node)

        assert _outcome(result)["status"] == ToolResultStatus.ERROR.value

    @pytest.mark.parametrize(
        "func",
        [_raise_runtime, _raise_permission],
        ids=["runtime_error", "permission_error"],
    )
    def test_unwrapped_failures_report_error(self, func):
        """The middleware skips tools it cannot wrap, so FunctionTool classifies locally."""
        node = _node(_tool(func))
        result = _run(node)

        assert _outcome(result)["status"] == ToolResultStatus.ERROR.value

    def test_error_class_survives_the_channel(self):
        node = _node(_middleware().wrap_tool(_tool(_raise_runtime)))
        result = _run(node)

        outcome = _outcome(result)
        assert outcome["error_class"] == ToolErrorClass.INFRASTRUCTURE.value
        assert outcome["retriable"] is True
        assert outcome["exception_type"] == "RuntimeError"

    def test_policy_failure_is_not_retriable(self):
        node = _node(_middleware().wrap_tool(_tool(_raise_permission)))
        result = _run(node)

        outcome = _outcome(result)
        assert outcome["error_class"] == ToolErrorClass.POLICY.value
        assert outcome["retriable"] is False

    def test_unwrapped_local_classification_matches(self):
        node = _node(_tool(_raise_runtime))
        result = _run(node)

        outcome = _outcome(result)
        assert outcome["error_class"] == ToolErrorClass.INFRASTRUCTURE.value
        assert outcome["exception_type"] == "RuntimeError"

    def test_tool_and_toolkit_identity_present(self):
        node = _node(_middleware().wrap_tool(_tool(_raise_runtime)))
        result = _run(node)

        outcome = _outcome(result)
        assert outcome["tool_name"] == "update_issue"
        assert outcome["toolkit_type"] == "jira"


# ─── The false positive this replaces ────────────────────────────────


class TestNoFalsePositive:
    def test_successful_payload_containing_the_word_error_is_success(self):
        node = _node(_tool(lambda issue_number: "Fix error in login flow"))
        result = _run(node)

        assert _outcome(result)["status"] == ToolResultStatus.SUCCESS.value
        # The very check this replaces would have gone the other way.
        assert "error" in result["issue"].lower()

    def test_error_shaped_dict_returned_not_raised_is_still_success(self):
        """A tool that returns rather than raises has not failed; the payload is data.

        Pinned deliberately: the status reflects whether the CALL failed, and inventing
        a failure from payload shape would reintroduce the guesswork being removed.
        """
        node = _node(_tool(lambda issue_number: {"error": "not found"}))
        result = _run(node)

        assert _outcome(result)["status"] == ToolResultStatus.SUCCESS.value


# ─── No output variable declared ─────────────────────────────────────


class TestNoOutputVariable:
    def test_key_present_when_node_declares_no_output(self):
        node = _node(_tool(lambda issue_number: "x"), output_variables=[])
        result = _run(node)

        assert _outcome(result)["status"] == ToolResultStatus.SUCCESS.value

    def test_key_present_on_failure_with_no_output(self):
        node = _node(_middleware().wrap_tool(_tool(_raise_runtime)), output_variables=[])
        result = _run(node)

        assert _outcome(result)["status"] == ToolResultStatus.ERROR.value

    def test_messages_only_node_still_gets_the_key(self):
        node = _node(_tool(lambda issue_number: "x"), output_variables=["messages"])
        result = _run(node)

        assert "messages" in result
        assert _outcome(result)["status"] == ToolResultStatus.SUCCESS.value


# ─── Blocked is not error ────────────────────────────────────────────


BLOCKED_PAYLOAD = (
    '{"type": "sensitive_tool_blocked", "blocked_tool_name": "update_issue", '
    '"blocked_toolkit_type": "jira", "denial_reason": "denied by user"}'
)


class TestBlockedIsNotError:
    """A decline is a deliberate user choice, so it must be distinguishable from a fault."""

    def test_declined_tool_reports_blocked(self):
        node = _node(_tool(lambda issue_number: BLOCKED_PAYLOAD))
        result = _run(node)

        assert _outcome(result)["status"] == ToolResultStatus.BLOCKED.value
        assert result[PIPELINE_BLOCKED_KEY]

    def test_blocked_is_not_reported_as_error(self):
        node = _node(_tool(lambda issue_number: BLOCKED_PAYLOAD))
        result = _run(node)

        assert _outcome(result)["status"] != ToolResultStatus.ERROR.value

    def test_blocked_message_carries_the_pipeline_stop_prose(self):
        node = _node(_tool(lambda issue_number: BLOCKED_PAYLOAD))
        result = _run(node)

        assert "blocked" in _outcome(result)["message"].lower()

    def test_mcp_auth_skip_reports_blocked(self):
        node = _node(_tool(lambda issue_number: "x"))
        skipped = node._build_mcp_auth_skipped_termination(
            {"tool_name": "update_issue", "toolkit_name": "jira"}
        )
        outcome = node._outcome_for(skipped, [])

        assert outcome.status is ToolResultStatus.BLOCKED


# ─── The state schema must declare the keys ──────────────────────────


class TestStateDeclaration:
    """LangGraph silently DROPS an update for a key the state TypedDict omits — no
    exception — so the declaration is what makes the whole feature work."""

    def test_keys_declared_in_created_state(self):
        from elitea_sdk.runtime.langchain.utils import create_state

        annotations = create_state({"messages": "list[str]"}).__annotations__
        assert TOOL_OUTCOMES_KEY in annotations
        assert LAST_TOOL_OUTCOME_KEY in annotations

    def test_reducer_merges_per_node(self):
        from elitea_sdk.runtime.langchain.utils import _tool_outcomes_reducer

        merged = _tool_outcomes_reducer({"a": {"status": "success"}}, {"b": {"status": "error"}})
        assert merged == {"a": {"status": "success"}, "b": {"status": "error"}}

    def test_reducer_overwrites_a_re_entered_node(self):
        from elitea_sdk.runtime.langchain.utils import _tool_outcomes_reducer

        merged = _tool_outcomes_reducer({"a": {"status": "error"}}, {"a": {"status": "success"}})
        assert merged == {"a": {"status": "success"}}

    def test_reducer_clears_on_none(self):
        from elitea_sdk.runtime.langchain.utils import _tool_outcomes_reducer

        assert _tool_outcomes_reducer({"a": {}}, None) == {}


# ─── The message-less predecessor ────────────────────────────────────


class TestEmptyMessageHistory:
    def test_last_message_is_defined_when_history_is_empty(self):
        from elitea_sdk.runtime.langchain.langraph_agent import ConditionalEdge

        edge = ConditionalEdge(
            condition="{% if 'error' in last_message %}fail{% else %}ok{% endif %}",
            condition_inputs=["last_message"],
            conditional_outputs=["fail", "ok"],
        )
        # Previously undefined rather than empty, so this surfaced as a template error.
        assert edge.invoke({"messages": []}) == "ok"

    def test_undefined_value_is_reported_as_missing_data_not_bad_syntax(self):
        template = EvaluateTemplate("{{ nope.attr }}", {"present": 1})

        with pytest.raises(Exception, match="not available"):
            template.extract()

    def test_syntax_error_still_reads_as_invalid_template(self):
        with pytest.raises(Exception, match="Invalid jinja template"):
            EvaluateTemplate("{% invalid syntax %}", {}).extract()


# ─── End to end: a pipeline routes on the key ────────────────────────


class _FakeLLM:
    temperature = 0
    max_tokens = 100

    @property
    def _get_model_default_parameters(self):
        return {"temperature": 0, "max_tokens": 100}

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        return AIMessage(content="LLM-DONE")


def _routing_pipeline():
    return yaml.dump(
        {
            "name": "outcome-routing",
            "state": {
                "input": {"type": "str"},
                "messages": {"type": "list"},
                "landed": {"type": "str"},
            },
            "entry_point": "call_tool",
            "nodes": [
                {
                    "id": "call_tool",
                    "type": "toolkit",
                    "toolkit_name": "jira",
                    "tool": "update_issue",
                    "input": ["messages"],
                    "output": ["issue"],
                    "condition": {
                        "condition_input": ["last_tool_outcome"],
                        "condition_definition": (
                            "{% if last_tool_outcome.status == 'error' %}on_failure"
                            "{% else %}on_success{% endif %}"
                        ),
                        "conditional_outputs": ["on_failure", "on_success"],
                        "default_output": "END",
                    },
                },
                {
                    "id": "on_failure",
                    "type": "state_modifier",
                    "input": ["messages"],
                    "output": ["landed"],
                    "template": "failure-branch",
                    "transition": "END",
                },
                {
                    "id": "on_success",
                    "type": "state_modifier",
                    "input": ["messages"],
                    "output": ["landed"],
                    "template": "success-branch",
                    "transition": "END",
                },
            ],
        }
    )


def _graph_for(func):
    tool = StructuredTool.from_function(
        func=func,
        name="update_issue",
        description="Update an issue",
        metadata={"toolkit_type": "jira", "toolkit_name": "jira", "tool_name": "update_issue"},
    )
    manager = MiddlewareManager()
    manager.add(_middleware())
    return create_graph(
        client=_FakeLLM(),
        yaml_schema=_routing_pipeline(),
        tools=[manager.wrap_tool(tool)],
        memory=MemorySaver(),
    )


class TestPipelineRouting:
    """The point of the whole change: one condition that routes correctly."""

    def test_failing_tool_routes_to_the_failure_branch(self):
        graph = _graph_for(lambda: (_ for _ in ()).throw(RuntimeError("connection refused")))

        result = graph.invoke(
            {"messages": [HumanMessage(content="go")]},
            config={"configurable": {"thread_id": "outcome-fail"}},
        )

        assert result["last_tool_outcome"]["status"] == "error"
        assert result["landed"] == "failure-branch"

    def test_success_containing_the_word_error_routes_to_success(self):
        graph = _graph_for(lambda: "Fix error in login flow")

        result = graph.invoke(
            {"messages": [HumanMessage(content="go")]},
            config={"configurable": {"thread_id": "outcome-ok"}},
        )

        assert result["last_tool_outcome"]["status"] == "success"
        assert result["landed"] == "success-branch"
