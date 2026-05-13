"""
Tests for FunctionTool agent node output variable handling.

Covers the bug where AgentNode's configured output variable received stale
structured-output content from a preceding LLM node instead of the agent's
actual response.

Two root causes were fixed:
1. extract_application_response_output (application.py) now prefers last AI
   message over elitea_response state variable.
2. FunctionTool.invoke() (function.py) correctly distinguishes react agents
   (is_subgraph=False) from child pipelines (is_subgraph=True) when deciding
   whether to let child-state propagation populate output variables or always
   use the agent's last AI message.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from elitea_sdk.runtime.tools.function import FunctionTool
from elitea_sdk.runtime.langchain.constants import ELITEA_RS, PRINTER_NODE_RS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_application_tool(*, is_subgraph: bool, invoke_return: dict) -> MagicMock:
    """Return a mock tool that looks like an Application with the given is_subgraph flag.

    The class must be named 'Application' so that FunctionTool's is_nested_app
    detection (``type(self.tool).__name__ == 'Application'``) works correctly for
    react-agent tools that have ``is_subgraph=False``.
    """
    # Dynamically create a class named 'Application' so type(tool).__name__ == 'Application'
    ApplicationClass = type("Application", (MagicMock,), {})
    tool = ApplicationClass(spec=BaseTool)
    tool.name = "mock_agent"
    tool.description = "mock"
    tool.is_subgraph = is_subgraph
    tool.invoke.return_value = invoke_return
    tool.args_schema = None
    return tool


def _make_function_tool(tool: MagicMock, output_variables: list) -> FunctionTool:
    """Wrap *tool* in a FunctionTool with the given output_variables."""
    return FunctionTool(
        name="test_node",
        description="test",
        tool=tool,
        return_type="str",
        input_variables=["messages"],
        input_mapping={"task": {"type": "variable", "value": "messages"}},
        output_variables=output_variables,
    )


def _base_state(extra: dict | None = None) -> dict:
    state = {
        "messages": [HumanMessage(content="hi my name is sam")],
        "input": "hi my name is sam",
    }
    if extra:
        state.update(extra)
    return state


def _invoke(ft: FunctionTool, state: dict) -> dict:
    """Invoke FunctionTool with standard patches applied."""
    with patch(
        "elitea_sdk.runtime.tools.function.convert_to_openai_tool",
        return_value={"function": {"parameters": {"properties": {}}}},
    ):
        with patch(
            "elitea_sdk.runtime.tools.function.propagate_the_input_mapping",
            return_value={"task": "hi"},
        ):
            with patch("elitea_sdk.runtime.tools.function.dispatch_custom_event"):
                return ft.invoke(state)


# ---------------------------------------------------------------------------
# FunctionTool — react agent (is_subgraph=False) output variable tests
# ---------------------------------------------------------------------------

class TestFunctionToolReactAgentOutputVariable:
    """
    When is_subgraph=False (react agent), the output variable must always be
    populated from the agent's last AI message, never from stale parent-state
    values echoed back through the child.
    """

    def test_agent_output_written_to_output_variable(self):
        """Happy path: agent response is mapped to the declared output variable."""
        tool = _make_application_tool(
            is_subgraph=False,
            invoke_return={
                "messages": [{"role": "assistant", "content": "Hmm, hello, Sam."}],
            },
        )
        result = _invoke(_make_function_tool(tool, ["agent_r", "messages"]), _base_state())
        assert result["agent_r"] == "Hmm, hello, Sam."

    def test_stale_elitea_response_does_not_pollute_output_var(self):
        """
        Core bug scenario: parent pipeline has elitea_response from a preceding
        structured-output LLM node.  That value must NOT end up in agent_r.

        The child returns agent_r echoed back from parent state and the correct
        answer only in messages.
        """
        stale_value = (
            "Elitea is a modern work companion designed to help teams organize knowledge."
        )
        correct_value = "Hmm, hello, Sam, it is. Greet you, I do."

        tool = _make_application_tool(
            is_subgraph=False,
            invoke_return={
                "messages": [{"role": "assistant", "content": correct_value}],
                "agent_r": stale_value,   # stale parent-state echo
                ELITEA_RS: stale_value,   # stale elitea_response from structured-output LLM
            },
        )
        state = _base_state(extra={"agent_r": stale_value, ELITEA_RS: stale_value, "story": "some story"})

        result = _invoke(_make_function_tool(tool, ["agent_r", "messages"]), state)

        assert result["agent_r"] == correct_value, (
            f"Expected correct agent output but got: {result['agent_r']!r}"
        )
        assert result["agent_r"] != stale_value

    def test_stale_output_var_in_child_state_excluded_from_propagation(self):
        """
        For react agents, declared output_variables must be excluded from the
        child→parent state propagation loop so that stale echoed values never
        land in result_dict before the output-variable mapping runs.
        """
        stale_value = "stale content from structured output LLM node"
        correct_value = "actual agent answer"

        tool = _make_application_tool(
            is_subgraph=False,
            invoke_return={
                "messages": [{"role": "assistant", "content": correct_value}],
                "agent_r": stale_value,  # would overwrite if not excluded
                "other_var": "side-effect value",
            },
        )
        result = _invoke(
            _make_function_tool(tool, ["agent_r", "messages"]),
            _base_state(extra={"agent_r": stale_value}),
        )

        assert result["agent_r"] == correct_value
        # Non-output state variables still propagate normally
        assert result.get("other_var") == "side-effect value"

    def test_other_state_vars_still_propagate_for_react_agent(self):
        """Non-output state variables returned by a react agent propagate to parent."""
        tool = _make_application_tool(
            is_subgraph=False,
            invoke_return={
                "messages": [{"role": "assistant", "content": "done"}],
                "side_effect_var": "computed by agent",
            },
        )
        result = _invoke(_make_function_tool(tool, ["agent_r", "messages"]), _base_state())
        assert result.get("side_effect_var") == "computed by agent"

    def test_no_output_variables_returns_messages_only(self):
        """When no output_variables declared, result contains only messages."""
        tool = _make_application_tool(
            is_subgraph=False,
            invoke_return={"messages": [{"role": "assistant", "content": "response"}]},
        )
        result = _invoke(_make_function_tool(tool, []), _base_state())
        assert "messages" in result
        assert "agent_r" not in result

    def test_output_var_from_langchain_ai_message_object(self):
        """Agent response as LangChain AIMessage objects (not dicts) is handled."""
        correct_value = "Greetings from the force."
        tool = _make_application_tool(
            is_subgraph=False,
            invoke_return={
                "messages": [
                    HumanMessage(content="input"),
                    AIMessage(content=correct_value),
                ],
            },
        )
        result = _invoke(_make_function_tool(tool, ["agent_r", "messages"]), _base_state())
        assert result["agent_r"] == correct_value


# ---------------------------------------------------------------------------
# FunctionTool — child pipeline (is_subgraph=True) output variable tests
# ---------------------------------------------------------------------------

class TestFunctionToolChildPipelineOutputVariable:
    """
    When is_subgraph=True (child pipeline), output variables explicitly set by
    the child pipeline must be respected and not overwritten by the last AI message.
    """

    def test_child_pipeline_explicit_output_var_preserved(self):
        """Child pipeline that explicitly sets an output variable — value kept."""
        child_computed_value = {"status": "ok", "count": 3}
        tool = _make_application_tool(
            is_subgraph=True,
            invoke_return={
                "messages": [{"role": "assistant", "content": "pipeline done"}],
                "result_var": child_computed_value,
            },
        )
        result = _invoke(_make_function_tool(tool, ["result_var", "messages"]), _base_state())
        assert result["result_var"] == child_computed_value

    def test_child_pipeline_output_var_not_overwritten_by_last_message(self):
        """
        A child pipeline sets result_var to a string.  It must not be replaced
        by the plain-text last message content.
        """
        child_value = "structured output from child pipeline"
        last_msg_content = "Pipeline execution complete."

        tool = _make_application_tool(
            is_subgraph=True,
            invoke_return={
                "messages": [{"role": "assistant", "content": last_msg_content}],
                "result_var": child_value,
            },
        )
        result = _invoke(_make_function_tool(tool, ["result_var", "messages"]), _base_state())

        assert result["result_var"] == child_value
        assert result["result_var"] != last_msg_content

    def test_child_pipeline_unset_output_var_falls_back_to_last_message(self):
        """
        If child pipeline does NOT set the declared output variable, fall back
        to the last AI message (same as react agent behaviour).
        """
        last_msg_content = "Nothing was set explicitly."
        tool = _make_application_tool(
            is_subgraph=True,
            invoke_return={
                "messages": [{"role": "assistant", "content": last_msg_content}],
                # result_var intentionally absent
            },
        )
        result = _invoke(_make_function_tool(tool, ["result_var", "messages"]), _base_state())
        assert result["result_var"] == last_msg_content

    def test_child_pipeline_none_output_var_falls_back_to_last_message(self):
        """If child pipeline sets the output variable to None, fall back to last AI message."""
        last_msg_content = "Fallback to message."
        tool = _make_application_tool(
            is_subgraph=True,
            invoke_return={
                "messages": [{"role": "assistant", "content": last_msg_content}],
                "result_var": None,
            },
        )
        result = _invoke(_make_function_tool(tool, ["result_var", "messages"]), _base_state())
        assert result["result_var"] == last_msg_content

    def test_child_pipeline_extra_state_vars_propagate(self):
        """Non-output state variables from child pipeline still propagate to parent."""
        tool = _make_application_tool(
            is_subgraph=True,
            invoke_return={
                "messages": [{"role": "assistant", "content": "done"}],
                "result_var": "child output",
                "extra_computed": "side computation",
            },
        )
        result = _invoke(_make_function_tool(tool, ["result_var", "messages"]), _base_state())
        assert result.get("extra_computed") == "side computation"


# ---------------------------------------------------------------------------
# extract_application_response_output priority tests
# ---------------------------------------------------------------------------

class TestExtractApplicationResponseOutputPriority:
    """
    Verify the new lookup order:
      output  →  last AI message from messages  →  elitea_response  →  PRINTER_NODE_RS
    """

    def setup_method(self):
        from elitea_sdk.runtime.tools.application import extract_application_response_output
        self.extract = extract_application_response_output

    def test_output_key_takes_highest_priority(self):
        response = {
            "output": "explicit output",
            ELITEA_RS: "elitea_response value",
            "messages": [AIMessage(content="ai message")],
        }
        assert self.extract(response) == "explicit output"

    def test_last_ai_message_preferred_over_elitea_response(self):
        """Core bug: elitea_response is stale, messages[-1] is the real agent output."""
        response = {
            ELITEA_RS: "stale structured-output LLM content",
            "messages": [
                HumanMessage(content="hi my name is sam"),
                AIMessage(content="Hmm, hello, Sam, it is. Greet you, I do."),
            ],
        }
        result = self.extract(response)
        assert result == "Hmm, hello, Sam, it is. Greet you, I do."
        assert "stale" not in result

    def test_elitea_response_used_when_messages_empty(self):
        response = {ELITEA_RS: "fallback elitea_response", "messages": []}
        assert self.extract(response) == "fallback elitea_response"

    def test_elitea_response_used_when_messages_absent(self):
        response = {ELITEA_RS: "fallback elitea_response"}
        assert self.extract(response) == "fallback elitea_response"

    def test_elitea_response_used_when_all_messages_are_human(self):
        response = {
            ELITEA_RS: "fallback",
            "messages": [HumanMessage(content="just a question")],
        }
        assert self.extract(response) == "fallback"

    def test_printer_node_rs_used_as_last_resort(self):
        response = {PRINTER_NODE_RS: "printer output"}
        assert self.extract(response) == "printer output"

    def test_output_beats_everything(self):
        response = {
            "output": "winner",
            ELITEA_RS: "loser 1",
            PRINTER_NODE_RS: "loser 2",
            "messages": [AIMessage(content="loser 3")],
        }
        assert self.extract(response) == "winner"

    def test_last_ai_message_beats_elitea_response_and_printer(self):
        response = {
            ELITEA_RS: "stale",
            PRINTER_NODE_RS: "also stale",
            "messages": [
                HumanMessage(content="input"),
                AIMessage(content="correct agent answer"),
            ],
        }
        assert self.extract(response) == "correct agent answer"

    def test_reversed_scan_picks_last_non_human_message(self):
        response = {
            "messages": [
                HumanMessage(content="first"),
                AIMessage(content="first ai"),
                HumanMessage(content="second"),
                AIMessage(content="second ai — this is the answer"),
            ],
        }
        assert self.extract(response) == "second ai — this is the answer"

    def test_dict_message_role_human_skipped(self):
        response = {
            ELITEA_RS: "stale fallback",
            "messages": [
                {"role": "user", "content": "should be skipped"},
                {"role": "assistant", "content": "correct"},
            ],
        }
        assert self.extract(response) == "correct"

    def test_tool_only_ai_message_falls_back_to_elitea_response(self):
        """When last AI message contains only tool_use blocks, skip it and fall back."""
        response = {
            ELITEA_RS: "fallback elitea_response",
            "messages": [
                HumanMessage(content="run tool"),
                AIMessage(content=[{"type": "tool_use", "name": "some_tool", "id": "t1"}]),
            ],
        }
        assert self.extract(response) == "fallback elitea_response"



