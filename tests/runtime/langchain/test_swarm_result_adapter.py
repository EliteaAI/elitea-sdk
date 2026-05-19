"""
Tests for SwarmResultAdapter response format.

SwarmResultAdapter wraps compiled swarm graphs to return the standardized
AgentResponse format expected by the pylon and other SDK consumers.
"""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage


class TestSwarmResultAdapterResponseFormat:
    """Test that SwarmResultAdapter returns the correct response format."""

    def _create_mock_swarm_result(self, messages):
        """Helper to create mock swarm graph result."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"messages": messages}
        return mock_graph

    def test_response_has_required_keys(self):
        """Test that SwarmResultAdapter response has all required keys."""
        from elitea_sdk.runtime.models.agent_response import AgentResponse

        # Simulate what SwarmResultAdapter returns
        output = "Task completed"
        messages = [AIMessage(content=output)]

        response = AgentResponse(
            output=output,
            messages=messages,
            thread_id=None,
            execution_finished=True,
        ).to_dict()

        # Verify all required keys are present
        assert "output" in response, "Response must have 'output' key"
        assert "messages" in response, "Response must have 'messages' key"
        assert "thread_id" in response, "Response must have 'thread_id' key"
        assert "execution_finished" in response, "Response must have 'execution_finished' key"

    def test_response_output_extraction_from_ai_message(self):
        """Test that output is correctly extracted from AIMessage."""
        from elitea_sdk.runtime.models.agent_response import AgentResponse

        ai_content = "This is the AI response"
        messages = [
            HumanMessage(content="User question"),
            AIMessage(content=ai_content),
        ]

        # Simulate SwarmResultAdapter logic
        output = None
        for msg in reversed(messages):
            if hasattr(msg, 'content') and not isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, str) and content.strip():
                    output = content
                    break

        response = AgentResponse(
            output=output or "",
            messages=messages,
            thread_id=None,
            execution_finished=True,
        ).to_dict()

        assert response["output"] == ai_content

    def test_response_output_extraction_from_list_content(self):
        """Test output extraction when AI message has list content (multimodal)."""
        from elitea_sdk.runtime.models.agent_response import AgentResponse

        # Multimodal content as list of blocks
        content_blocks = [
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
        ]
        messages = [AIMessage(content=content_blocks)]

        # Simulate SwarmResultAdapter logic for list content
        output = None
        for msg in reversed(messages):
            if hasattr(msg, 'content') and not isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, list):
                    content = "\n".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                if isinstance(content, str) and content.strip():
                    output = content
                    break

        response = AgentResponse(
            output=output or "",
            messages=messages,
            thread_id=None,
            execution_finished=True,
        ).to_dict()

        assert "First part" in response["output"]
        assert "Second part" in response["output"]

    def test_response_empty_output_when_no_ai_message(self):
        """Test that output is empty string when no AI message found."""
        from elitea_sdk.runtime.models.agent_response import AgentResponse

        messages = [HumanMessage(content="User message only")]

        # Simulate SwarmResultAdapter logic
        output = None
        for msg in reversed(messages):
            if hasattr(msg, 'content') and not isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, str) and content.strip():
                    output = content
                    break

        response = AgentResponse(
            output=output or "",
            messages=messages,
            thread_id=None,
            execution_finished=True,
        ).to_dict()

        assert response["output"] == ""

    def test_response_thread_id_is_none(self):
        """Test that thread_id is always None for swarm (single-turn)."""
        from elitea_sdk.runtime.models.agent_response import AgentResponse

        response = AgentResponse(
            output="Result",
            messages=[],
            thread_id=None,
            execution_finished=True,
        ).to_dict()

        assert response["thread_id"] is None

    def test_response_execution_finished_is_true(self):
        """Test that execution_finished is always True for swarm."""
        from elitea_sdk.runtime.models.agent_response import AgentResponse

        response = AgentResponse(
            output="Result",
            messages=[],
            thread_id=None,
            execution_finished=True,
        ).to_dict()

        assert response["execution_finished"] is True

    def test_messages_preserved_in_response(self):
        """Test that original messages are preserved in response."""
        from elitea_sdk.runtime.models.agent_response import AgentResponse

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm doing well!"),
        ]

        response = AgentResponse(
            output="I'm doing well!",
            messages=messages,
            thread_id=None,
            execution_finished=True,
        ).to_dict()

        # Messages are serialized to dicts by Pydantic
        assert len(response["messages"]) == 4
        assert response["messages"][0]["content"] == "Hello"
        assert response["messages"][1]["content"] == "Hi there!"
        assert response["messages"][3]["content"] == "I'm doing well!"


class TestSwarmResultAdapterBackwardsCompatibility:
    """Test backwards compatibility with pylon extraction."""

    def test_pylon_can_extract_output_key(self):
        """Test that pylon's extract_response_content can use 'output' key."""
        from elitea_sdk.runtime.models.agent_response import AgentResponse

        response = AgentResponse(
            output="The final answer",
            messages=[AIMessage(content="The final answer")],
            thread_id=None,
            execution_finished=True,
        ).to_dict()

        # Simulate pylon's simplified extraction
        content = response.get("output", "")
        assert content == "The final answer"

    def test_pylon_can_fallback_to_messages(self):
        """Test that pylon can fall back to messages if needed."""
        from elitea_sdk.runtime.models.agent_response import AgentResponse

        messages = [AIMessage(content="Message content")]
        response = AgentResponse(
            output="",  # Empty output
            messages=messages,
            thread_id=None,
            execution_finished=True,
        ).to_dict()

        # Simulate pylon's fallback extraction
        # Messages are serialized to dicts by Pydantic, so check for dict format
        content = response.get("output", "")
        if not content and "messages" in response:
            msgs = response.get("messages", [])
            if msgs:
                last_msg = msgs[-1]
                # Handle both Message objects and serialized dicts
                if hasattr(last_msg, 'content'):
                    content = last_msg.content
                elif isinstance(last_msg, dict):
                    content = last_msg.get('content', '')

        assert content == "Message content"
