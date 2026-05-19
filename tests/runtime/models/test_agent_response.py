"""
Tests for the AgentResponse Pydantic model.

AgentResponse provides a standardized response format for all SDK agent types,
ensuring consistent behavior across LangGraphAgentRunnable, SwarmResultAdapter,
and Application._run().
"""
import pytest
from elitea_sdk.runtime.models.agent_response import AgentResponse


class TestAgentResponseConstruction:
    """Test AgentResponse model construction and defaults."""

    def test_default_values(self):
        """Test that AgentResponse has correct default values."""
        response = AgentResponse()

        assert response.output == ""
        assert response.messages == []
        assert response.thread_id is None
        assert response.execution_finished is True
        assert response.context_info is None
        assert response.hitl_interrupt is None

    def test_explicit_values(self):
        """Test AgentResponse with explicit values."""
        response = AgentResponse(
            output="Task completed",
            messages=[{"role": "assistant", "content": "Task completed"}],
            thread_id="thread-123",
            execution_finished=False,
        )

        assert response.output == "Task completed"
        assert response.messages == [{"role": "assistant", "content": "Task completed"}]
        assert response.thread_id == "thread-123"
        assert response.execution_finished is False

    def test_optional_metadata_fields(self):
        """Test optional metadata fields (context_info, hitl_interrupt)."""
        response = AgentResponse(
            output="Result",
            context_info={"token_count": 100, "message_count": 5},
            hitl_interrupt={"pending": True, "tool_name": "approve"},
        )

        assert response.context_info == {"token_count": 100, "message_count": 5}
        assert response.hitl_interrupt == {"pending": True, "tool_name": "approve"}


class TestAgentResponseExtraFields:
    """Test that AgentResponse allows extra fields (state variables)."""

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed and preserved."""
        response = AgentResponse(
            output="Done",
            custom_var="custom_value",
            pipeline_state={"step": 3},
        )

        assert response.output == "Done"
        assert response.custom_var == "custom_value"
        assert response.pipeline_state == {"step": 3}

    def test_extra_fields_in_to_dict(self):
        """Test that extra fields appear in to_dict() output."""
        response = AgentResponse(
            output="Result",
            my_state_var="state_value",
        )

        result = response.to_dict()

        assert result["output"] == "Result"
        assert result["my_state_var"] == "state_value"


class TestAgentResponseToDict:
    """Test AgentResponse.to_dict() method."""

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict() includes all standard fields."""
        response = AgentResponse(
            output="Output text",
            messages=[{"role": "assistant", "content": "Output text"}],
            thread_id="thread-456",
            execution_finished=True,
        )

        result = response.to_dict()

        assert "output" in result
        assert "messages" in result
        assert "thread_id" in result
        assert "execution_finished" in result
        assert result["output"] == "Output text"
        assert result["thread_id"] == "thread-456"

    def test_to_dict_includes_none_values(self):
        """Test that to_dict() includes None values (not excluded)."""
        response = AgentResponse(output="Test")

        result = response.to_dict()

        # None values should be present, not excluded
        assert "thread_id" in result
        assert result["thread_id"] is None
        assert "context_info" in result
        assert result["context_info"] is None

    def test_to_dict_returns_dict_type(self):
        """Test that to_dict() returns a plain dict."""
        response = AgentResponse(output="Test")

        result = response.to_dict()

        assert isinstance(result, dict)
        assert type(result) is dict  # Not a subclass

    def test_to_dict_with_extra_state_variables(self):
        """Test to_dict() includes extra state variables from child pipelines."""
        response = AgentResponse(
            output="Pipeline result",
            messages=[],
            thread_id=None,
            execution_finished=True,
            # Extra state variables that a child pipeline might return
            user_query="search term",
            results_count=42,
            processed_data={"key": "value"},
        )

        result = response.to_dict()

        # Standard fields
        assert result["output"] == "Pipeline result"
        assert result["execution_finished"] is True

        # Extra state variables
        assert result["user_query"] == "search term"
        assert result["results_count"] == 42
        assert result["processed_data"] == {"key": "value"}


class TestAgentResponseIntegration:
    """Integration-style tests for AgentResponse usage patterns."""

    def test_swarm_adapter_pattern(self):
        """Test the pattern used by SwarmResultAdapter."""
        # Simulate what SwarmResultAdapter does
        output = "Swarm task completed"
        messages = [{"role": "assistant", "content": output}]

        response = AgentResponse(
            output=output,
            messages=messages,
            thread_id=None,
            execution_finished=True,
        ).to_dict()

        assert response["output"] == "Swarm task completed"
        assert response["messages"] == messages
        assert response["thread_id"] is None
        assert response["execution_finished"] is True

    def test_application_run_pattern(self):
        """Test the pattern used by Application._run() with state propagation."""
        # Simulate what Application._run() does
        normalized_output = "Child agent result"
        child_response = {
            "output": normalized_output,
            "thread_id": "child-thread",
            "execution_finished": True,
            "custom_state": "from_child",
        }

        agent_response = AgentResponse(
            output=normalized_output,
            messages=[{"role": "assistant", "content": normalized_output}],
            thread_id=child_response.get("thread_id"),
            execution_finished=child_response.get("execution_finished", True),
        )

        # Merge with extra state (as Application._run() does)
        extra_state = {"custom_state": child_response["custom_state"]}
        result = {**agent_response.to_dict(), **extra_state}

        assert result["output"] == normalized_output
        assert result["thread_id"] == "child-thread"
        assert result["custom_state"] == "from_child"

    def test_empty_output_handling(self):
        """Test that empty output is handled correctly."""
        response = AgentResponse(
            output="",
            messages=[],
        )

        result = response.to_dict()

        assert result["output"] == ""
        assert result["messages"] == []
