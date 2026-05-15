"""
Tests for LLMNode structured output handling with Anthropic-style responses.

Covers issue #4890: Anthropic returns valid JSON wrapped in markdown fences
as text content. The LLMNode must extract this JSON and map it to the
structured output Pydantic model.

Tests cover:
1. _extract_structured_from_content — extracts JSON from various content formats
2. _map_parsed_json_to_model — maps parsed JSON to Pydantic model fields
3. _create_fallback_completion — generates safe fallback when parsing fails
"""
import asyncio
import json
import pytest
from unittest.mock import MagicMock, patch
from typing import Any, List

from pydantic import create_model, Field, ValidationError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.exceptions import OutputParserException

from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime.langchain.constants import ELITEA_RS
from elitea_sdk.runtime.langchain.utils import create_pydantic_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_node() -> LLMNode:
    """Create a minimal LLMNode instance for testing mapping methods."""
    return LLMNode(
        name="test_llm",
        description="test",
        input_variables=["messages"],
        input_mapping={},
        output_variables=[],
    )


def _make_struct_model(fields: dict):
    """Create a structured output model with ELITEA_RS included."""
    all_fields = {**fields, ELITEA_RS: {"type": "str", "default": ""}}
    return create_pydantic_model("LLMOutput", all_fields)


def _make_completion(content):
    """Create a mock LLM completion object."""
    mock = MagicMock()
    mock.content = content
    return mock


# ---------------------------------------------------------------------------
# _map_parsed_json_to_model: direct field match
# ---------------------------------------------------------------------------

class TestMapParsedJsonToModelDirect:
    """Test mapping when parsed JSON keys match model field names."""

    def test_dict_with_matching_fields(self):
        node = _make_llm_node()
        model = _make_struct_model({"question": {"type": "list", "description": "Q"}})
        parsed = {"question": [{"id": "q1", "text": "What?"}]}
        result = node._map_parsed_json_to_model(parsed, model)
        assert result.question == [{"id": "q1", "text": "What?"}]

    def test_dict_with_multiple_fields(self):
        node = _make_llm_node()
        model = _make_struct_model({
            "title": {"type": "str"},
            "items": {"type": "list", "description": "Items"},
        })
        parsed = {"title": "Report", "items": [{"name": "item1"}]}
        result = node._map_parsed_json_to_model(parsed, model)
        assert result.title == "Report"
        assert result.items == [{"name": "item1"}]


# ---------------------------------------------------------------------------
# _map_parsed_json_to_model: field name mismatch (Anthropic uses different keys)
# ---------------------------------------------------------------------------

class TestMapParsedJsonToModelMismatch:
    """Test mapping when Anthropic uses different field names than the model."""

    def test_plural_vs_singular_field_name(self):
        """LLM returns 'questions' but model expects 'question'."""
        node = _make_llm_node()
        model = _make_struct_model({"question": {"type": "list", "description": "Q"}})
        parsed = {"questions": [{"id": "q1"}, {"id": "q2"}]}
        result = node._map_parsed_json_to_model(parsed, model)
        assert result.question == [{"id": "q1"}, {"id": "q2"}]

    def test_completely_different_field_name(self):
        """LLM returns 'results' but model expects 'items'."""
        node = _make_llm_node()
        model = _make_struct_model({"items": {"type": "list", "description": "Items"}})
        parsed = {"results": [{"a": 1}, {"a": 2}]}
        result = node._map_parsed_json_to_model(parsed, model)
        assert result.items == [{"a": 1}, {"a": 2}]


# ---------------------------------------------------------------------------
# _map_parsed_json_to_model: array responses
# ---------------------------------------------------------------------------

class TestMapParsedJsonToModelArray:
    """Test mapping when parsed JSON is an array (no wrapping object)."""

    def test_array_maps_to_list_field(self):
        """Anthropic returns bare array — maps to first list-type field."""
        node = _make_llm_node()
        model = _make_struct_model({"question": {"type": "list", "description": "Q"}})
        parsed = [{"id": "q1", "text": "What?"}, {"id": "q2", "text": "How?"}]
        result = node._map_parsed_json_to_model(parsed, model)
        assert result.question == parsed

    def test_array_with_multiple_list_fields_uses_first(self):
        node = _make_llm_node()
        model = _make_struct_model({
            "primary": {"type": "list", "description": "Primary"},
            "secondary": {"type": "list", "description": "Secondary", "default": []},
        })
        parsed = [{"x": 1}]
        result = node._map_parsed_json_to_model(parsed, model)
        assert result.primary == [{"x": 1}]

    def test_array_with_no_list_field_raises(self):
        """If model has no list field, array cannot be mapped."""
        node = _make_llm_node()
        model = _make_struct_model({"name": {"type": "str"}})
        with pytest.raises(ValueError, match="Cannot map parsed JSON"):
            node._map_parsed_json_to_model([1, 2, 3], model)


# ---------------------------------------------------------------------------
# _extract_structured_from_content: end-to-end extraction
# ---------------------------------------------------------------------------

class TestExtractStructuredFromContent:
    """Test full extraction pipeline from LLM completion content."""

    def test_fenced_array_content(self):
        """Exact Anthropic pattern: fenced JSON array in content."""
        node = _make_llm_node()
        model = _make_struct_model({"question": {"type": "list", "description": "Q"}})
        content = '```json\n[{"question_id": "q1", "question_text": "What?"}]\n```'
        completion = _make_completion(content)
        result = node._extract_structured_from_content(completion, model)
        assert result is not None
        assert result.question == [{"question_id": "q1", "question_text": "What?"}]

    def test_fenced_object_content(self):
        node = _make_llm_node()
        model = _make_struct_model({"question": {"type": "list", "description": "Q"}})
        content = '```json\n{"question": [{"id": "q1"}]}\n```'
        completion = _make_completion(content)
        result = node._extract_structured_from_content(completion, model)
        assert result is not None
        assert result.question == [{"id": "q1"}]

    def test_plain_json_content(self):
        node = _make_llm_node()
        model = _make_struct_model({"name": {"type": "str"}})
        content = '{"name": "test value"}'
        completion = _make_completion(content)
        result = node._extract_structured_from_content(completion, model)
        assert result is not None
        assert result.name == "test value"

    def test_list_content_blocks_anthropic_format(self):
        """Anthropic extended thinking: content as list of blocks."""
        node = _make_llm_node()
        model = _make_struct_model({"question": {"type": "list", "description": "Q"}})
        content = [
            {"type": "thinking", "thinking": "Let me analyze..."},
            {"type": "text", "text": '```json\n[{"id": "q1"}]\n```'},
        ]
        completion = _make_completion(content)
        result = node._extract_structured_from_content(completion, model)
        assert result is not None
        assert result.question == [{"id": "q1"}]

    def test_empty_content_returns_none(self):
        node = _make_llm_node()
        model = _make_struct_model({"name": {"type": "str"}})
        completion = _make_completion("")
        result = node._extract_structured_from_content(completion, model)
        assert result is None

    def test_non_json_content_returns_none(self):
        node = _make_llm_node()
        model = _make_struct_model({"name": {"type": "str"}})
        completion = _make_completion("I cannot process this request.")
        result = node._extract_structured_from_content(completion, model)
        assert result is None

    def test_no_content_attribute_returns_none(self):
        node = _make_llm_node()
        model = _make_struct_model({"name": {"type": "str"}})
        completion = "plain string"
        result = node._extract_structured_from_content(completion, model)
        assert result is None


# ---------------------------------------------------------------------------
# _create_fallback_completion: safe fallback generation
# ---------------------------------------------------------------------------

class TestCreateFallbackCompletion:
    """Test fallback completion when all parsing strategies fail."""

    def test_fallback_puts_content_in_elitea_rs(self):
        node = _make_llm_node()
        model = _make_struct_model({"question": {"type": "list", "description": "Q"}})
        result = node._create_fallback_completion("raw text", model)
        assert getattr(result, ELITEA_RS) == "raw text"

    def test_fallback_sets_required_fields_to_none(self):
        node = _make_llm_node()
        model = _make_struct_model({"name": {"type": "str"}})
        result = node._create_fallback_completion("text", model)
        assert result.name is None

    def test_fallback_uses_defaults_for_optional_fields(self):
        node = _make_llm_node()
        model = _make_struct_model({
            "count": {"type": "int", "default": 0},
        })
        result = node._create_fallback_completion("text", model)
        assert result.count == 0


# ---------------------------------------------------------------------------
# Integration: full pipeline simulation
# ---------------------------------------------------------------------------

class TestStructuredOutputIntegration:
    """Integration tests simulating the full Anthropic structured output flow."""

    def test_issue_4890_exact_scenario(self):
        """
        Reproduce issue #4890:
        - Pipeline has list-type state variable 'question'
        - Anthropic returns fenced JSON array
        - Must successfully parse and populate the field
        """
        node = _make_llm_node()
        model = _make_struct_model({"question": {"type": "list", "description": "Questions"}})

        anthropic_response = '''```json
[
  {"question_id": "q1", "question_text": "What is the main topic of the document?"},
  {"question_id": "q2", "question_text": "What are the key findings?"},
  {"question_id": "q3", "question_text": "What methodology was used?"}
]
```'''
        completion = _make_completion(anthropic_response)
        result = node._extract_structured_from_content(completion, model)

        assert result is not None
        assert len(result.question) == 3
        assert result.question[0]["question_id"] == "q1"
        assert result.question[2]["question_text"] == "What methodology was used?"

    def test_openai_direct_dict_still_works(self):
        """OpenAI returns dict matching model fields — no change in behavior."""
        node = _make_llm_node()
        model = _make_struct_model({"question": {"type": "list", "description": "Q"}})
        parsed = {"question": [{"id": "q1"}, {"id": "q2"}]}
        result = node._map_parsed_json_to_model(parsed, model)
        assert result.question == [{"id": "q1"}, {"id": "q2"}]

    def test_anthropic_wrapped_object_with_mismatched_key(self):
        """Anthropic wraps in object with different key name."""
        node = _make_llm_node()
        model = _make_struct_model({"question": {"type": "list", "description": "Q"}})
        content = '```json\n{"questions": [{"id": "q1"}, {"id": "q2"}]}\n```'
        completion = _make_completion(content)
        result = node._extract_structured_from_content(completion, model)
        assert result is not None
        assert len(result.question) == 2


# ---------------------------------------------------------------------------
# Issue #4890 part 2: tool-calling branch of _invoke_with_structured_output
#
# Covers the regression where LLMNode with tools assigned, Anthropic models, and
# extended thinking ENABLED silently returned [] for list-type structured output.
# Root cause: passing the full tool exchange history (AI(tool_calls) + ToolMsg)
# back to with_structured_output caused Anthropic to respond with continuation
# text (markdown-fenced JSON) instead of a fresh tool_use. The fix sanitizes the
# history to "messages + [last clean AIMessage]" before the structured call.
# ---------------------------------------------------------------------------


def _ai_with_tool_calls(text: str = "", tool_calls=None, content_blocks=None):
    """Build an AIMessage carrying tool_calls (with optional thinking blocks)."""
    msg = AIMessage(
        content=content_blocks if content_blocks is not None else text,
        tool_calls=tool_calls or [
            {"id": "tc_1", "name": "search", "args": {"query": "x"}, "type": "tool_call"}
        ],
    )
    return msg


def _ai_clean(text="done", content_blocks=None):
    """Build a clean AIMessage (no tool_calls)."""
    return AIMessage(content=content_blocks if content_blocks is not None else text)


def _tool_msg(content="result", call_id="tc_1"):
    return ToolMessage(content=content, tool_call_id=call_id)


class TestBuildCleanMessagesForStructuredOutput:
    """Helper-level tests for the history sanitizer."""

    def test_drops_tool_use_aimessages_keeps_final(self):
        node = _make_llm_node()
        messages = [HumanMessage(content="ask")]
        new_messages = [
            HumanMessage(content="ask"),
            _ai_with_tool_calls("calling"),
            _tool_msg(),
            _ai_clean("here is the synthesis"),
        ]
        result = node._build_clean_messages_for_structured_output(messages, new_messages)
        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert not getattr(result[1], 'tool_calls', None)
        assert result[1].content == "here is the synthesis"

    def test_picks_last_clean_aimessage_in_multi_iteration_loop(self):
        node = _make_llm_node()
        messages = [HumanMessage(content="task")]
        new_messages = [
            HumanMessage(content="task"),
            _ai_with_tool_calls("first"),
            _tool_msg(call_id="tc_1"),
            _ai_with_tool_calls("second", tool_calls=[
                {"id": "tc_2", "name": "search", "args": {}, "type": "tool_call"}
            ]),
            _tool_msg(call_id="tc_2"),
            _ai_clean("final answer"),
        ]
        result = node._build_clean_messages_for_structured_output(messages, new_messages)
        assert len(result) == 2
        assert result[1].content == "final answer"

    def test_max_iterations_fallback_strips_tool_calls_and_tool_use_blocks(self):
        """When the loop exits with the last AIMessage still carrying tool_calls
        AND Anthropic-shape list content with tool_use blocks (the realistic case
        for thinking-on Anthropic), return a copy with tool_calls cleared AND
        tool_use blocks removed from content so Anthropic does not see unmatched
        tool_use blocks."""
        node = _make_llm_node()
        messages = [HumanMessage(content="task")]
        anthropic_content = [
            {"type": "thinking", "thinking": "I should call search", "signature": "s1"},
            {"type": "text", "text": "Let me search for that"},
            {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"query": "x"}},
        ]
        msg = AIMessage(
            content=anthropic_content,
            tool_calls=[{"id": "tu_1", "name": "search", "args": {"query": "x"}, "type": "tool_call"}],
        )
        new_messages = [HumanMessage(content="task"), msg]
        result = node._build_clean_messages_for_structured_output(messages, new_messages)
        assert len(result) == 2
        cleaned = result[1]
        assert isinstance(cleaned, AIMessage)
        assert not getattr(cleaned, 'tool_calls', None)
        assert isinstance(cleaned.content, list)
        # tool_use stripped, thinking + text kept
        assert all(
            not (isinstance(b, dict) and b.get('type') == 'tool_use')
            for b in cleaned.content
        )
        block_types = {b.get('type') for b in cleaned.content if isinstance(b, dict)}
        assert block_types == {"thinking", "text"}

    def test_max_iterations_fallback_with_string_content(self):
        """Backward-compatible string-content case (non-Anthropic / thinking-off):
        clear tool_calls attribute, leave string content intact."""
        node = _make_llm_node()
        messages = [HumanMessage(content="task")]
        new_messages = [
            HumanMessage(content="task"),
            _ai_with_tool_calls("interrupted"),
        ]
        result = node._build_clean_messages_for_structured_output(messages, new_messages)
        assert len(result) == 2
        assert not getattr(result[1], 'tool_calls', None)
        assert result[1].content == "interrupted"

    def test_preserves_thinking_blocks_in_final_aimessage(self):
        """Anthropic returns content as a list of blocks when thinking is on; the
        helper must keep the structure intact so the API still accepts the turn."""
        node = _make_llm_node()
        messages = [HumanMessage(content="ask")]
        thinking_blocks = [
            {"type": "thinking", "thinking": "Let me think...", "signature": "sig_xyz"},
            {"type": "text", "text": "synthesis"},
        ]
        new_messages = [
            HumanMessage(content="ask"),
            _ai_with_tool_calls("first"),
            _tool_msg(),
            _ai_clean(content_blocks=thinking_blocks),
        ]
        result = node._build_clean_messages_for_structured_output(messages, new_messages)
        assert len(result) == 2
        assert result[1].content == thinking_blocks
        assert result[1].content[0]["type"] == "thinking"
        assert result[1].content[1]["type"] == "text"

    def test_returns_messages_only_when_no_aimessage_in_new_messages(self):
        """Edge case: __perform_tool_calling somehow returns no AIMessage."""
        node = _make_llm_node()
        messages = [HumanMessage(content="ask")]
        new_messages = [HumanMessage(content="ask"), _tool_msg()]
        result = node._build_clean_messages_for_structured_output(messages, new_messages)
        assert result == messages

    def test_strip_tool_use_blocks_passes_through_string_content(self):
        """No-regression: OpenAI/non-Anthropic models return string content. The
        strip helper must be a no-op for strings — same object reference returned."""
        node = _make_llm_node()
        s = "plain text answer"
        result = node._strip_tool_use_blocks(s)
        assert result is s, "string content must be returned unchanged (same reference)"

    def test_strip_tool_use_blocks_passes_through_list_without_tool_use(self):
        """List content with only thinking + text blocks (no tool_use) — returned
        list must be equal to the original. Used to verify that the helper does
        not corrupt thinking-on responses that happen not to contain tool_use."""
        node = _make_llm_node()
        blocks = [
            {"type": "thinking", "thinking": "thinking text", "signature": "s"},
            {"type": "text", "text": "answer"},
        ]
        result = node._strip_tool_use_blocks(blocks)
        assert result == blocks
        assert all(b.get('type') != 'tool_use' for b in result)

    def test_openai_shape_string_content_passes_through_unchanged(self):
        """No-regression: when the last AIMessage has plain string content and no
        tool_calls (the typical OpenAI/GPT shape after __perform_tool_calling
        completes), the helper returns the message unchanged (same reference) —
        no defensive copy, no allocation."""
        node = _make_llm_node()
        messages = [HumanMessage(content="ask")]
        final_ai = AIMessage(content="OpenAI-style string synthesis answer")
        new_messages = [
            HumanMessage(content="ask"),
            _ai_with_tool_calls("planning"),
            _tool_msg(),
            final_ai,
        ]
        result = node._build_clean_messages_for_structured_output(messages, new_messages)
        assert result[1] is final_ai, "string-content AIMessage must be passed through by reference"


class TestInvokeWithStructuredOutputToolCallingBranch:
    """Verify the patched tool-calling branch passes sanitized history and falls
    back via _extract_structured_from_content on parser failure."""

    def _struct_model_with_question_list(self):
        return _make_struct_model({"question": {"type": "list", "description": "Q"}})

    def _build_node_with_mocks(self, tool_call_loop_messages: List, second_call_behavior, plain_call_response=None):
        """Wire mocks so that:
        - llm_client.invoke (first call): returns AIMessage with tool_calls
        - __perform_tool_calling: short-circuited to return tool_call_loop_messages
        - llm.invoke (struct output, second call): determined by second_call_behavior
        - self.client.invoke (fallback plain call, UNBOUND): returns plain_call_response
        """
        node = _make_llm_node()
        struct_model = self._struct_model_with_question_list()

        first_completion = _ai_with_tool_calls("planning", tool_calls=[
            {"id": "tc_1", "name": "search", "args": {}, "type": "tool_call"}
        ])

        # llm_client (tool-bound) — used only for the FIRST invoke
        llm_client = MagicMock()
        llm_client.invoke = MagicMock(return_value=first_completion)

        # self.client (unbound base) — used for the fallback retry
        node.client = MagicMock()
        node.client.invoke = MagicMock(return_value=plain_call_response)

        struct_llm = MagicMock()
        if isinstance(second_call_behavior, Exception):
            struct_llm.invoke = MagicMock(side_effect=second_call_behavior)
        else:
            struct_llm.invoke = MagicMock(return_value=second_call_behavior)

        node._LLMNode__get_struct_output_model = MagicMock(return_value=struct_llm)

        async def _fake_perform_tool_calling(completion, msgs, client, cfg):
            return tool_call_loop_messages, completion
        node._LLMNode__perform_tool_calling = _fake_perform_tool_calling
        node._run_async_in_sync_context = lambda coro: asyncio.new_event_loop().run_until_complete(coro)

        return node, struct_model, llm_client, struct_llm

    def test_uses_sanitized_history_not_full_new_messages(self):
        """The structured-output call must receive messages + [last_clean_ai],
        NOT the full new_messages list."""
        messages = [HumanMessage(content="ask for list")]
        loop_messages = [
            HumanMessage(content="ask for list"),
            _ai_with_tool_calls("calling tool"),
            _tool_msg(),
            _ai_clean("synthesized answer"),
        ]
        success_completion = self._struct_model_with_question_list()(
            question=[{"id": "q1"}, {"id": "q2"}], **{ELITEA_RS: ""}
        )
        node, struct_model, llm_client, struct_llm = self._build_node_with_mocks(
            loop_messages, second_call_behavior=success_completion,
        )

        completion, initial, final = node._invoke_with_structured_output(
            llm_client, messages, struct_model, config={}
        )

        assert struct_llm.invoke.call_count == 1
        passed_messages = struct_llm.invoke.call_args[0][0]
        assert len(passed_messages) == 2, "should be messages + [last_clean_ai], not full loop history"
        assert isinstance(passed_messages[1], AIMessage)
        assert not getattr(passed_messages[1], 'tool_calls', None)
        assert passed_messages[1].content == "synthesized answer"
        assert completion.question == [{"id": "q1"}, {"id": "q2"}]

    def test_returns_full_new_messages_for_checkpoint_preservation(self):
        """final_messages returned upstream must be the full new_messages so
        checkpoint state and audit trail see the real exchange."""
        messages = [HumanMessage(content="ask")]
        loop_messages = [
            HumanMessage(content="ask"),
            _ai_with_tool_calls("calling"),
            _tool_msg(),
            _ai_clean("done"),
        ]
        success_completion = self._struct_model_with_question_list()(
            question=[{"id": "q1"}], **{ELITEA_RS: ""}
        )
        node, struct_model, llm_client, _ = self._build_node_with_mocks(
            loop_messages, second_call_behavior=success_completion,
        )

        _, _, final_messages = node._invoke_with_structured_output(
            llm_client, messages, struct_model, config={}
        )
        assert final_messages == loop_messages
        assert len(final_messages) == 4

    def test_anthropic_thinking_on_markdown_fallback_extracts_list(self):
        """When Anthropic with thinking ON returns markdown-fenced JSON as text
        instead of tool_use, the structured-output parser raises ValidationError,
        the fallback re-issues a plain invoke (on the UNBOUND client to avoid
        another tool call), and _extract_structured_from_content parses the array.
        The plain_completion is returned upstream as initial_completion so
        _format_structured_output_result reads ELITEA_RS fallback text from the
        actually-parsed response, not the original planning AIMessage."""
        messages = [HumanMessage(content="give me a list")]
        loop_messages = [
            HumanMessage(content="give me a list"),
            _ai_with_tool_calls("calling"),
            _tool_msg(content="some data"),
            _ai_clean(content_blocks=[
                {"type": "thinking", "thinking": "...", "signature": "s"},
                {"type": "text", "text": "I'll synthesize the answer."},
            ]),
        ]
        fallback_response = AIMessage(content=[
            {"type": "thinking", "thinking": "Let me format this", "signature": "s2"},
            {"type": "text", "text": "```json\n[{\"question_id\": \"q1\"}, {\"question_id\": \"q2\"}]\n```"},
        ])
        node, struct_model, llm_client, struct_llm = self._build_node_with_mocks(
            loop_messages,
            second_call_behavior=ValidationError.from_exception_data("LLMOutput", []),
            plain_call_response=fallback_response,
        )

        completion, returned_initial, final_messages = node._invoke_with_structured_output(
            llm_client, messages, struct_model, config={}
        )

        assert struct_llm.invoke.call_count == 1, "structured output attempted once"
        assert llm_client.invoke.call_count == 1, "tool-bound client used only for first call"
        assert node.client.invoke.call_count == 1, "fallback uses unbound client"
        assert len(completion.question) == 2
        assert completion.question[0]["question_id"] == "q1"
        assert returned_initial is fallback_response, (
            "initial_completion returned upstream must be plain_completion so "
            "_format_structured_output_result extracts ELITEA_RS from the parsed response"
        )
        assert final_messages == loop_messages, "checkpoint state preserved"

    def test_propagates_exception_when_fallback_also_fails(self):
        """If both structured-output parsing AND content extraction fail, the
        original exception must propagate — never silently return None/empty."""
        messages = [HumanMessage(content="ask")]
        loop_messages = [
            HumanMessage(content="ask"),
            _ai_with_tool_calls("calling"),
            _tool_msg(),
            _ai_clean("done"),
        ]
        unparseable = AIMessage(content="this is not json at all")
        node, struct_model, llm_client, _ = self._build_node_with_mocks(
            loop_messages,
            second_call_behavior=OutputParserException("parse failed"),
            plain_call_response=unparseable,
        )

        with pytest.raises(OutputParserException):
            node._invoke_with_structured_output(llm_client, messages, struct_model, config={})

    def test_thinking_off_anthropic_no_regression(self):
        """Anthropic without extended thinking returns plain string content;
        helper must still produce a valid 2-element history and the structured
        output call must succeed."""
        messages = [HumanMessage(content="ask")]
        loop_messages = [
            HumanMessage(content="ask"),
            _ai_with_tool_calls("calling"),
            _tool_msg(),
            _ai_clean("plain string final answer"),
        ]
        success = self._struct_model_with_question_list()(
            question=[{"id": "a"}], **{ELITEA_RS: ""}
        )
        node, struct_model, llm_client, struct_llm = self._build_node_with_mocks(
            loop_messages, second_call_behavior=success,
        )

        completion, _, _ = node._invoke_with_structured_output(
            llm_client, messages, struct_model, config={}
        )

        passed = struct_llm.invoke.call_args[0][0]
        assert isinstance(passed[1].content, str)
        assert completion.question == [{"id": "a"}]

    def test_openai_no_regression_with_proper_tool_use(self):
        """GPT path returns a valid Pydantic instance directly from
        with_structured_output — fallback must not fire."""
        messages = [HumanMessage(content="ask")]
        loop_messages = [
            HumanMessage(content="ask"),
            _ai_with_tool_calls("calling"),
            _tool_msg(),
            _ai_clean("done"),
        ]
        success = self._struct_model_with_question_list()(
            question=[{"q": 1}, {"q": 2}, {"q": 3}], **{ELITEA_RS: ""}
        )
        node, struct_model, llm_client, struct_llm = self._build_node_with_mocks(
            loop_messages, second_call_behavior=success,
        )

        completion, _, _ = node._invoke_with_structured_output(
            llm_client, messages, struct_model, config={}
        )

        assert struct_llm.invoke.call_count == 1
        assert llm_client.invoke.call_count == 1, "no fallback should fire"
        assert node.client.invoke.call_count == 0, "unbound client must not be touched on success"
        assert len(completion.question) == 3

    def test_openai_happy_path_returns_original_initial_completion_unchanged(self):
        """No-regression contract for _format_structured_output_result consumer:
        on the OpenAI/GPT happy path (structured output succeeds without fallback),
        the initial_completion returned upstream MUST be the original first
        AIMessage with tool_calls — not plain_completion. This preserves the
        pre-fix behavior where _format_structured_output_result reads
        initial_completion.content for ELITEA_RS fallback."""
        messages = [HumanMessage(content="ask")]
        loop_messages = [
            HumanMessage(content="ask"),
            _ai_with_tool_calls("planning"),
            _tool_msg(),
            _ai_clean("done"),
        ]
        success = self._struct_model_with_question_list()(
            question=[{"id": "x"}], **{ELITEA_RS: "user-facing summary"}
        )
        node, struct_model, llm_client, _ = self._build_node_with_mocks(
            loop_messages, second_call_behavior=success,
        )

        # Capture the first_completion the mock returns
        first_completion_returned = llm_client.invoke.return_value

        _, returned_initial, _ = node._invoke_with_structured_output(
            llm_client, messages, struct_model, config={}
        )

        assert returned_initial is first_completion_returned, (
            "happy path must return the original initial_completion — only the "
            "fallback path swaps it for plain_completion"
        )

    def test_openai_value_error_falls_through_to_upstream_handler(self):
        """No-regression: when structured output raises ValueError AND the local
        fallback's content extraction fails (returns None), the original
        ValueError must re-raise so the upstream caller's _handle_structured_output_fallback
        (json_mode → function_calling → text extraction chain) still gets to run.
        Verifies we did not shortcut the existing OpenAI fallback chain."""
        messages = [HumanMessage(content="ask")]
        loop_messages = [
            HumanMessage(content="ask"),
            _ai_with_tool_calls("planning"),
            _tool_msg(),
            _ai_clean("done"),
        ]
        # Fallback response that _extract_structured_from_content cannot parse
        unparseable = AIMessage(content="not json, just words and more words")
        node, struct_model, llm_client, _ = self._build_node_with_mocks(
            loop_messages,
            second_call_behavior=ValueError("structured output failed"),
            plain_call_response=unparseable,
        )

        with pytest.raises(ValueError, match="structured output failed"):
            node._invoke_with_structured_output(llm_client, messages, struct_model, config={})

        # Verify the upstream handler will receive a ValueError it can catch
        # (line 1252 of llm.py only catches ValueError).
        assert node.client.invoke.call_count == 1, "local fallback attempted"
