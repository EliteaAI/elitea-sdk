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

from elitea_sdk.runtime.tools.llm import LLMNode, STRUCTURED_OUTPUT_PREFILL_PROMPT
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
    """Helper-level tests for the history-prep used in the structured-output
    follow-up call.

    Two contracts are enforced:
    1. The full ``new_messages`` history (including matched
       ``tool_call → tool_result`` pairs) is preserved so the model has
       the data the synthesis was based on. Only the **last** ``AIMessage``
       is sanitized, and only when it carries unmatched ``tool_calls`` /
       ``tool_use`` blocks (the max-iterations exit case).
    2. If the cleaned list still ends with an ``AIMessage`` (the synthesis
       turn), a trailing ``HumanMessage`` is appended so the conversation
       satisfies Anthropic's "must end with a user message" invariant.
       This is the prefill-rejection regression that PR #157 missed.
    """

    _PREFILL_PROMPT = STRUCTURED_OUTPUT_PREFILL_PROMPT

    def test_completed_loop_appends_human_message_after_synthesis_ai(self):
        """Normal completion: the synthesis AIMessage has no tool_calls.
        The full tool exchange is preserved AND a trailing HumanMessage
        is appended to satisfy Anthropic's prefill rule. Prior to Fix A,
        the second structured-output invoke ended on the synthesis
        AIMessage and Anthropic returned 400."""
        node = _make_llm_node()
        new_messages = [
            HumanMessage(content="ask"),
            _ai_with_tool_calls("calling"),
            _tool_msg(content="search returned: [a, b, c]"),
            _ai_clean("here is the synthesis"),
        ]
        result = node._build_clean_messages_for_structured_output(new_messages)
        assert len(result) == 5, (
            "tool exchange preserved (4 msgs) + appended HumanMessage prefill prompt"
        )
        # Original four are unchanged and passed by reference
        assert result[0] is new_messages[0]
        assert result[1] is new_messages[1]
        assert result[2] is new_messages[2]
        assert result[3] is new_messages[3], "final AIMessage passed by reference"
        # Trailing HumanMessage closes the conversation for Anthropic
        assert isinstance(result[4], HumanMessage)
        assert result[4].content == self._PREFILL_PROMPT

    def test_multi_iteration_loop_keeps_all_pairs_and_appends_human(self):
        """Two tool iterations + final synthesis: ALL six original messages
        preserved + HumanMessage appended → 7 total."""
        node = _make_llm_node()
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
        result = node._build_clean_messages_for_structured_output(new_messages)
        assert len(result) == 7
        assert result[-2].content == "final answer", "synthesis preserved at index -2"
        assert isinstance(result[-1], HumanMessage)

    def test_max_iter_exit_sanitizes_last_ai_and_appends_human(self):
        """When the loop exits with the last AIMessage still carrying
        tool_calls + tool_use blocks (max-iterations / error exit), strip
        them from THAT message only — preserve every prior message,
        AND append a HumanMessage (the sanitized message is still an
        AIMessage; the prefill rule still applies)."""
        node = _make_llm_node()
        unmatched_anthropic = AIMessage(
            content=[
                {"type": "thinking", "thinking": "I should call search", "signature": "s2"},
                {"type": "text", "text": "Let me search again"},
                {"type": "tool_use", "id": "tu_99", "name": "search", "input": {"query": "x"}},
            ],
            tool_calls=[{"id": "tu_99", "name": "search", "args": {"query": "x"}, "type": "tool_call"}],
        )
        new_messages = [
            HumanMessage(content="task"),
            _ai_with_tool_calls("first"),
            _tool_msg(call_id="tc_1", content="prior tool result"),
            unmatched_anthropic,
        ]
        result = node._build_clean_messages_for_structured_output(new_messages)
        assert len(result) == 5
        # Prior matched pair is preserved by reference
        assert result[0] is new_messages[0]
        assert result[1] is new_messages[1]
        assert result[2] is new_messages[2]
        # Last AIMessage is sanitized: no tool_calls, no tool_use blocks,
        # but thinking + text blocks kept
        cleaned = result[3]
        assert cleaned is not unmatched_anthropic, "must be a sanitized copy"
        assert not getattr(cleaned, 'tool_calls', None)
        assert isinstance(cleaned.content, list)
        block_types = {b.get('type') for b in cleaned.content if isinstance(b, dict)}
        assert block_types == {"thinking", "text"}
        # Trailing HumanMessage closes the conversation for Anthropic
        assert isinstance(result[4], HumanMessage)

    def test_max_iter_exit_string_content_appends_human_message(self):
        """String-content variant of the max-iter exit: clear tool_calls
        attribute, leave string content intact, append HumanMessage."""
        node = _make_llm_node()
        new_messages = [
            HumanMessage(content="task"),
            _ai_with_tool_calls("interrupted"),
        ]
        result = node._build_clean_messages_for_structured_output(new_messages)
        assert len(result) == 3
        assert result[0] is new_messages[0]
        assert not getattr(result[1], 'tool_calls', None)
        assert result[1].content == "interrupted"
        assert isinstance(result[2], HumanMessage)

    def test_thinking_blocks_in_final_aimessage_are_preserved(self):
        """Anthropic returns content as a list of blocks with thinking on.
        The helper must not modify the synthesis message when it has no
        unmatched tool calls — preserve the block structure intact, and
        append the prefill HumanMessage."""
        node = _make_llm_node()
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
        result = node._build_clean_messages_for_structured_output(new_messages)
        assert len(result) == 5
        # Synthesis AIMessage is at index -2 now; HumanMessage is appended at -1
        assert result[-2].content == thinking_blocks
        assert result[-2] is new_messages[-1]
        assert isinstance(result[-1], HumanMessage)

    def test_returns_input_when_no_aimessage_present(self):
        """Edge case: history with no AIMessage. Trailing element is a
        ``ToolMessage`` (which Anthropic accepts as a user-role turn), so
        no HumanMessage append is needed. Return the list as-is."""
        node = _make_llm_node()
        new_messages = [HumanMessage(content="ask"), _tool_msg()]
        result = node._build_clean_messages_for_structured_output(new_messages)
        assert result == new_messages

    def test_strip_tool_use_blocks_passes_through_string_content(self):
        """No-regression: OpenAI/non-Anthropic models return string content. The
        strip helper must be a no-op for strings — same object reference returned."""
        node = _make_llm_node()
        s = "plain text answer"
        result = node._strip_tool_use_blocks(s)
        assert result is s

    def test_strip_tool_use_blocks_passes_through_list_without_tool_use(self):
        """List content with only thinking + text blocks (no tool_use) —
        helper must not corrupt thinking-on responses that happen not to
        contain tool_use."""
        node = _make_llm_node()
        blocks = [
            {"type": "thinking", "thinking": "thinking text", "signature": "s"},
            {"type": "text", "text": "answer"},
        ]
        result = node._strip_tool_use_blocks(blocks)
        assert result == blocks
        assert all(b.get('type') != 'tool_use' for b in result)

    def test_openai_shape_string_content_appends_human_message(self):
        """No-regression: when the last AIMessage has plain string content
        (typical OpenAI/GPT shape), the synthesis message is preserved by
        reference and a HumanMessage is still appended. The HumanMessage
        is harmless for OpenAI (which accepts trailing AIMessage too) but
        is required for Anthropic — applying it uniformly keeps the path
        provider-agnostic."""
        node = _make_llm_node()
        final_ai = AIMessage(content="OpenAI-style string synthesis answer")
        new_messages = [
            HumanMessage(content="ask"),
            _ai_with_tool_calls("planning"),
            _tool_msg(),
            final_ai,
        ]
        result = node._build_clean_messages_for_structured_output(new_messages)
        assert len(result) == 5, "full history (4) + trailing HumanMessage (1)"
        assert result[-2] is final_ai, "string-content AIMessage passed by reference"
        assert isinstance(result[-1], HumanMessage)

    def test_empty_input_returns_empty_list(self):
        """Defensive: empty new_messages → empty list, no append (nothing
        to follow up on)."""
        node = _make_llm_node()
        result = node._build_clean_messages_for_structured_output([])
        assert result == []


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

    def test_passes_full_history_with_tool_exchange_to_structured_output(self):
        """The structured-output call must receive the FULL ``new_messages``
        history including the matched tool_call/tool_result pair — without
        the tool data the model cannot produce correct structured output.
        This was the regression in PR #157 v1: dropping the tool exchange
        made both providers return empty fields.

        After Fix A (Anthropic prefill regression), the helper also
        appends a trailing ``HumanMessage`` when the synthesis turn ends
        in an ``AIMessage``. So the structured-output call receives 5
        messages: the 4-message tool exchange + 1 prefill HumanMessage.
        """
        messages = [HumanMessage(content="ask for list")]
        loop_messages = [
            HumanMessage(content="ask for list"),
            _ai_with_tool_calls("calling tool"),
            _tool_msg(content="search returned: [a, b, c]"),
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
        # Full history (4 msgs) + appended HumanMessage prefill prompt (1)
        assert len(passed_messages) == 5, (
            "full tool exchange must reach the structured-output call AND "
            "a trailing HumanMessage must close the conversation for Anthropic"
        )
        # Tool exchange present
        assert isinstance(passed_messages[1], AIMessage)
        assert isinstance(passed_messages[2], ToolMessage)
        assert "search returned" in passed_messages[2].content
        # Synthesis AIMessage at index 3 (no pending tool_calls)
        assert isinstance(passed_messages[3], AIMessage)
        assert not getattr(passed_messages[3], 'tool_calls', None)
        assert passed_messages[3].content == "synthesized answer"
        # Trailing HumanMessage closes the conversation (Fix A: prefill rule)
        assert isinstance(passed_messages[4], HumanMessage)
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
        """When structured output raises ValueError, _invoke_with_structured_output
        propagates it directly — there is no local recovery. The upstream caller
        catches (ValueError, ValidationError, OutputParserException) and routes
        to _handle_structured_output_fallback, which is the single recovery point."""
        messages = [HumanMessage(content="ask")]
        loop_messages = [
            HumanMessage(content="ask"),
            _ai_with_tool_calls("planning"),
            _tool_msg(),
            _ai_clean("done"),
        ]
        node, struct_model, llm_client, _ = self._build_node_with_mocks(
            loop_messages,
            second_call_behavior=ValueError("structured output failed"),
        )

        with pytest.raises(ValueError, match="structured output failed"):
            node._invoke_with_structured_output(llm_client, messages, struct_model, config={})

        # No local fallback path: the unbound client must NOT be invoked.
        assert node.client.invoke.call_count == 0, (
            "no local recovery — exceptions propagate to the upstream handler"
        )


# ---------------------------------------------------------------------------
# Issue #4890 fix: Anthropic thinking-mode detection and json_schema routing
#
# These tests verify the fix:
# - _is_anthropic_thinking_client detects the thinking flag correctly
# - __get_struct_output_model routes to method='json_schema' for thinking Anthropic
#   and leaves method='function_calling' for non-thinking Anthropic and all other providers
# - _handle_structured_output_fallback skips function_calling retry for thinking Anthropic
# - The schema produced by parse_pydantic_type for "list"/"any" types is accepted by
#   Anthropic's transform_schema directly (verified in test_extract_json_content.py),
#   so no schema patching is needed at this layer.
#
# Uses the realistic Assistant + LLM stub pattern (like test_hitl_resume_real_graph.py)
# for the integration scenario so the real langchain routing is exercised — NOT mocked
# away as in the previous PR's tests.
# ---------------------------------------------------------------------------


# ─── Minimal Anthropic-like client stubs (mirror pattern from test_hitl_resume_real_graph.py) ──


class _AnthropicThinkingLLM:
    """Stub that looks like langchain_anthropic.ChatAnthropic with thinking enabled.

    Carries the .thinking attribute that _is_anthropic_thinking_client reads, and
    a .bound attribute on the tool-bound version so the helper can traverse the
    RunnableBinding wrapper.
    """

    def __init__(self):
        # Matches the ChatAnthropic attribute shape
        self.thinking = {"type": "enabled", "budget_tokens": 1000}
        # Fake module-like identifier — module is checked via type().__module__
        self._module = "langchain_anthropic.chat_models"
        self.invocations: list = []
        self.with_structured_output_calls: list = []

    def bind_tools(self, tools, **kwargs):
        return _AnthropicThinkingLLMBound(self, tools)

    def invoke(self, messages, config=None):
        return _AnthropicThinkingLLMBound(self, []).invoke(messages, config=config)

    def with_structured_output(self, schema, method="function_calling", **kwargs):
        self.with_structured_output_calls.append(method)
        # Return a stub that produces a Pydantic-like result
        return _StructuredOutputStub(schema, method)


class _AnthropicThinkingLLMBound:
    """Tool-bound version wrapping _AnthropicThinkingLLM.

    The .bound attribute mimics the RunnableBinding.bound attribute that
    _is_anthropic_thinking_client inspects.
    """

    def __init__(self, root: "_AnthropicThinkingLLM", tools):
        self.root = root
        self.tools = list(tools)
        # The key: .bound points back to the real LLM (carries .thinking)
        self.bound = root

    def invoke(self, messages, config=None):
        self.root.invocations.append(list(messages))
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        tool_contents = [str(m.content) for m in tool_messages]

        # After tool result present: return clean synthesis AIMessage
        if any("tool-result" in c for c in tool_contents):
            return AIMessage(
                content=[
                    {"type": "thinking", "thinking": "Both tools completed.", "signature": "sig_final"},
                    {"type": "text", "text": "Here is the synthesis result."},
                ]
            )

        # First call: issue one tool call
        return AIMessage(
            content=[
                {
                    "type": "thinking",
                    "thinking": "I need to call the data_fetch tool.",
                    "signature": "sig_planning",
                },
                {"type": "text", "text": ""},
            ],
            tool_calls=[
                {"name": "data_fetch", "args": {"query": "test"}, "id": "call-001"},
            ],
        )

    def with_structured_output(self, schema, method="function_calling", **kwargs):
        self.root.with_structured_output_calls.append(method)
        return _StructuredOutputStub(schema, method)


class _AnthropicNonThinkingLLM:
    """Stub ChatAnthropic WITHOUT thinking enabled (thinking is None / not set)."""

    def __init__(self):
        self.thinking = None
        self._module = "langchain_anthropic.chat_models"
        self.invocations: list = []
        self.with_structured_output_calls: list = []

    def bind_tools(self, tools, **kwargs):
        bound = _AnthropicNonThinkingLLMBound(self, tools)
        return bound

    def with_structured_output(self, schema, method="function_calling", **kwargs):
        self.with_structured_output_calls.append(method)
        return _StructuredOutputStub(schema, method)

    def invoke(self, messages, config=None):
        return AIMessage(content="synthesis answer")


class _AnthropicNonThinkingLLMBound:
    def __init__(self, root, tools):
        self.root = root
        self.bound = root

    def invoke(self, messages, config=None):
        return self.root.invoke(messages, config=config)

    def with_structured_output(self, schema, method="function_calling", **kwargs):
        return self.root.with_structured_output(schema, method=method, **kwargs)


class _OpenAILikeLLM:
    """Stub that does NOT look like langchain_anthropic (no thinking attribute)."""

    def __init__(self):
        self.invocations: list = []
        self.with_structured_output_calls: list = []

    def bind_tools(self, tools, **kwargs):
        return _OpenAILikeLLMBound(self, tools)

    def with_structured_output(self, schema, method="function_calling", **kwargs):
        self.with_structured_output_calls.append(method)
        return _StructuredOutputStub(schema, method)

    def invoke(self, messages, config=None):
        return AIMessage(content="openai answer")


class _OpenAILikeLLMBound:
    def __init__(self, root, tools):
        self.root = root
        # No .bound attribute — mirrors real OpenAI tool-bound RunnableBinding

    def invoke(self, messages, config=None):
        return self.root.invoke(messages, config=config)

    def with_structured_output(self, schema, method="function_calling", **kwargs):
        return self.root.with_structured_output(schema, method=method, **kwargs)


class _AnthropicPrefillRejection(Exception):
    """Mimics ``anthropic.BadRequestError`` raised by the gateway when the
    conversation handed to a structured-output invoke ends with an
    ``AIMessage``. Real production error message:

        "This model does not support assistant message prefill. The
        conversation must end with a user message."

    Three prior PRs against the same code path all merged with mock-only
    validation that ignored this contract — the stub now enforces it so a
    regression cannot reach production again.
    """


class _StructuredOutputStub:
    """A fake structured-output runnable that returns a Pydantic-like instance.

    Accepts both shapes that ``with_structured_output`` can be called
    with: a Pydantic class (default for OpenAI / non-Anthropic) or a
    JSON-schema dict (the Anthropic path, which patches
    ``$defs.JsonValue`` before sending). The stub mirrors the real
    runnable's behavior: dict-shaped invocation returns a dict, class-
    shaped invocation returns the class instance.

    Enforces the Anthropic prefill contract: invoking with a message list
    that ends in ``AIMessage`` raises ``_AnthropicPrefillRejection``. This
    mirrors the real gateway behavior that PR #157 missed because the
    permissive stub silently accepted any input shape.
    """

    def __init__(self, schema, method: str):
        self.schema = schema
        self.method = method

    def invoke(self, messages, config=None):
        if messages and isinstance(messages[-1], AIMessage):
            raise _AnthropicPrefillRejection(
                "This model does not support assistant message prefill. "
                "The conversation must end with a user message."
            )
        # Anthropic path: schema is a dict — return a dict matching the
        # canonical "question + ELITEA_RS" shape used across tests.
        if isinstance(self.schema, dict):
            return {"question": [{"id": "q1", "text": "What?"}], ELITEA_RS: "synthesis"}
        # OpenAI / default path: schema is a Pydantic class.
        try:
            return self.schema(question=[{"id": "q1", "text": "What?"}], rs="synthesis")
        except Exception:
            return self.schema.model_construct()


# ─── Patch type.__module__ so _is_anthropic_thinking_client sees the right module ──────

def _patch_module(cls: type, module_path: str):
    """Monkey-patch __module__ on a class so the module-name check in
    _is_anthropic_thinking_client sees the right string."""
    cls.__module__ = module_path


# Apply the module patches once so the stubs look like langchain_anthropic classes.
_patch_module(_AnthropicThinkingLLM, "langchain_anthropic.chat_models")
_patch_module(_AnthropicThinkingLLMBound, "langchain_anthropic.chat_models")
_patch_module(_AnthropicNonThinkingLLM, "langchain_anthropic.chat_models")
_patch_module(_AnthropicNonThinkingLLMBound, "langchain_anthropic.chat_models")
# _OpenAILikeLLM intentionally NOT patched — stays outside langchain_anthropic


# ─── Helper ──────────────────────────────────────────────────────────────────


def _make_data_fetch_tool():
    """Return a simple StructuredTool that simulates a data fetch."""
    from langchain_core.tools import StructuredTool

    def _fetch(query: str = ""):
        return "tool-result:data"

    return StructuredTool.from_function(
        func=_fetch,
        name="data_fetch",
        description="Fetch data",
    )


# ─── Unit tests: detection helpers ───────────────────────────────────────────


class TestIsAnthropicThinkingClient:
    """Unit tests for LLMNode._is_anthropic_thinking_client."""

    def test_detects_direct_thinking_enabled_client(self):
        """Direct ChatAnthropic stub with thinking=enabled → True."""
        client = _AnthropicThinkingLLM()
        assert LLMNode._is_anthropic_thinking_client(client) is True

    def test_detects_thinking_via_bound_attribute(self):
        """Tool-bound RunnableBinding stub with .bound carrying .thinking → True."""
        llm = _AnthropicThinkingLLM()
        bound = llm.bind_tools([])  # returns _AnthropicThinkingLLMBound
        assert LLMNode._is_anthropic_thinking_client(bound) is True

    def test_non_thinking_anthropic_returns_false(self):
        """ChatAnthropic stub with thinking=None → False."""
        client = _AnthropicNonThinkingLLM()
        assert LLMNode._is_anthropic_thinking_client(client) is False

    def test_openai_like_client_returns_false(self):
        """Non-Anthropic client stub → False."""
        client = _OpenAILikeLLM()
        assert LLMNode._is_anthropic_thinking_client(client) is False

    def test_magic_mock_returns_false(self):
        """MagicMock (used in tests) must not be detected as Anthropic thinking."""
        from unittest.mock import MagicMock
        assert LLMNode._is_anthropic_thinking_client(MagicMock()) is False

    def test_none_returns_false(self):
        assert LLMNode._is_anthropic_thinking_client(None) is False


# ─── Unit tests: __get_struct_output_model routing ────────────────────────────


class TestGetStructOutputModelRouting:
    """Verify that __get_struct_output_model routes to json_schema for
    thinking-enabled Anthropic and leaves function_calling for all other cases."""

    def _make_node_with_client(self, client) -> LLMNode:
        node = _make_llm_node()
        node.client = client
        return node

    def test_anthropic_thinking_routes_to_json_schema(self):
        """Thinking-enabled Anthropic → with_structured_output called with method='json_schema'."""
        llm = _AnthropicThinkingLLM()
        node = self._make_node_with_client(llm)
        struct_model = _make_struct_model({"question": {"type": "list", "description": "Q"}})

        # Call the private method via name-mangling
        node._LLMNode__get_struct_output_model(llm, struct_model, method="function_calling")

        assert "json_schema" in llm.with_structured_output_calls, (
            "Anthropic thinking client must trigger json_schema routing, "
            f"but got: {llm.with_structured_output_calls}"
        )
        assert "function_calling" not in llm.with_structured_output_calls, (
            "function_calling must NOT be called for thinking-enabled Anthropic"
        )

    def test_anthropic_thinking_bound_client_routes_to_json_schema(self):
        """Tool-bound (RunnableBinding) wrapping thinking-enabled Anthropic → json_schema."""
        llm = _AnthropicThinkingLLM()
        node = self._make_node_with_client(llm)
        bound = llm.bind_tools([])  # _AnthropicThinkingLLMBound
        struct_model = _make_struct_model({"question": {"type": "list", "description": "Q"}})

        node._LLMNode__get_struct_output_model(bound, struct_model, method="function_calling")

        # Both node.client (base) and the llm stub share the same call list
        assert "json_schema" in llm.with_structured_output_calls

    def test_non_thinking_anthropic_uses_function_calling(self):
        """Non-thinking Anthropic → original method='function_calling' forwarded unchanged."""
        llm = _AnthropicNonThinkingLLM()
        node = self._make_node_with_client(llm)
        struct_model = _make_struct_model({"question": {"type": "list", "description": "Q"}})

        node._LLMNode__get_struct_output_model(llm, struct_model, method="function_calling")

        assert "function_calling" in llm.with_structured_output_calls, (
            "Non-thinking Anthropic must use function_calling (the default path)"
        )
        assert "json_schema" not in llm.with_structured_output_calls

    def test_openai_uses_function_calling(self):
        """OpenAI-like client → original method='function_calling' forwarded unchanged."""
        llm = _OpenAILikeLLM()
        node = self._make_node_with_client(llm)
        struct_model = _make_struct_model({"question": {"type": "list", "description": "Q"}})

        node._LLMNode__get_struct_output_model(llm, struct_model, method="function_calling")

        assert "function_calling" in llm.with_structured_output_calls
        assert "json_schema" not in llm.with_structured_output_calls

    def test_explicit_json_schema_always_forwarded(self):
        """When caller explicitly requests json_schema (e.g., fallback path), it must be
        forwarded for ALL providers including thinking Anthropic — no extra routing."""
        llm = _AnthropicThinkingLLM()
        node = self._make_node_with_client(llm)
        struct_model = _make_struct_model({"question": {"type": "list", "description": "Q"}})

        # The json_schema → json_schema identity path: Anthropic + thinking but
        # method already is json_schema → falls through to default, still json_schema
        # via the base path (the if-branch only fires for method="function_calling")
        node._LLMNode__get_struct_output_model(llm, struct_model, method="json_schema")
        # json_schema was forwarded (not re-routed from function_calling)
        # The call must appear in the log exactly once
        assert llm.with_structured_output_calls.count("json_schema") == 1

    def test_json_mode_forwarded_for_non_thinking_client(self):
        """json_mode forwarded as-is for non-Anthropic (used in _handle_structured_output_fallback)."""
        llm = _OpenAILikeLLM()
        node = self._make_node_with_client(llm)
        struct_model = _make_struct_model({"name": {"type": "str"}})

        node._LLMNode__get_struct_output_model(llm, struct_model, method="json_mode")

        assert "json_mode" in llm.with_structured_output_calls

    def test_anthropic_receives_patched_schema_dict_not_pydantic_class(self):
        """The whole point of the JsonValue revert: Anthropic gets a
        patched schema **dict** (with concrete ``$defs.JsonValue``),
        not the raw Pydantic class. Without this patch the empty
        ``$defs.JsonValue`` would reach ``transform_schema`` and the
        call would 400. Both thinking and non-thinking Anthropic go
        through the patch — the only difference is the method override
        for thinking mode."""
        llm = _AnthropicThinkingLLM()
        node = self._make_node_with_client(llm)
        struct_model = _make_struct_model({"question": {"type": "list", "description": "Q"}})

        captured_schemas = []
        original = llm.with_structured_output

        def capture(schema, method="function_calling", **kwargs):
            captured_schemas.append(schema)
            return original(schema, method=method, **kwargs)

        llm.with_structured_output = capture
        node._LLMNode__get_struct_output_model(llm, struct_model, method="function_calling")

        assert len(captured_schemas) == 1
        passed = captured_schemas[0]
        assert isinstance(passed, dict), (
            f"Anthropic must receive a JSON-schema dict (patched), got {type(passed).__name__}"
        )
        # The patch concretely defines $defs.JsonValue
        assert "$defs" in passed
        assert "JsonValue" in passed["$defs"]
        assert "anyOf" in passed["$defs"]["JsonValue"], (
            "patched JsonValue must have a concrete anyOf; the empty default is "
            "what transform_schema rejects"
        )

    def test_non_thinking_anthropic_also_receives_patched_dict(self):
        """The patch applies to non-thinking Anthropic too — the empty
        ``$defs.JsonValue`` is rejected regardless of which structured-
        output method is used (function_calling translates it to a tool
        spec; json_schema runs it through transform_schema). Patching
        unconditionally for any Anthropic client is the simplest correct
        rule."""
        llm = _AnthropicNonThinkingLLM()
        node = self._make_node_with_client(llm)
        struct_model = _make_struct_model({"question": {"type": "list"}})

        captured_schemas = []
        original = llm.with_structured_output

        def capture(schema, method="function_calling", **kwargs):
            captured_schemas.append(schema)
            return original(schema, method=method, **kwargs)

        llm.with_structured_output = capture
        node._LLMNode__get_struct_output_model(llm, struct_model, method="function_calling")

        assert len(captured_schemas) == 1
        assert isinstance(captured_schemas[0], dict)
        assert captured_schemas[0]["$defs"]["JsonValue"]["anyOf"]

    def test_openai_receives_pydantic_class_unchanged(self):
        """OpenAI / Azure / Google / etc. continue to receive the
        Pydantic class directly. This is the path that lets reasoning
        models see the permissive ``JsonValue`` schema with empty
        ``$defs.JsonValue`` (which OpenAI's API engine accepts and
        reasoning models interpret correctly) — patching it would
        re-introduce the ``list[list[str]]`` regression that motivated
        the revert."""
        llm = _OpenAILikeLLM()
        node = self._make_node_with_client(llm)
        struct_model = _make_struct_model({"question": {"type": "list"}})

        captured_schemas = []
        original = llm.with_structured_output

        def capture(schema, method="function_calling", **kwargs):
            captured_schemas.append(schema)
            return original(schema, method=method, **kwargs)

        llm.with_structured_output = capture
        node._LLMNode__get_struct_output_model(llm, struct_model, method="function_calling")

        assert captured_schemas[0] is struct_model, (
            "OpenAI must receive the Pydantic class by reference — "
            "no schema patching, no dict translation"
        )


# ─── Integration-style test: full _invoke_with_structured_output with realistic shapes ──────


class TestAnthropicThinkingStructuredOutputIntegration:
    """Integration-level tests using the realistic LLM stub pattern.

    These tests exercise the REAL _invoke_with_structured_output code path
    (not mocked away) with fake Anthropic thinking LLM stubs. They verify:
    1. The full tool-calling → structured output loop works end-to-end.
    2. with_structured_output is called with method='json_schema' (NOT function_calling).
    3. Non-raising path: no OutputParserException.
    4. Non-thinking Anthropic and OpenAI remain on the function_calling path.
    """

    def _make_node_for_anthropic_thinking(self, llm) -> LLMNode:
        """Build a minimal LLMNode wired for structured output with a thinking Anthropic LLM."""
        node = LLMNode(
            name="test_thinking_llm",
            description="test",
            input_variables=["messages"],
            input_mapping={},
            output_variables=["question"],
            structured_output=True,
            structured_output_dict={"question": {"type": "list", "description": "Questions"}},
            available_tools=[_make_data_fetch_tool()],
            tool_names=["data_fetch"],
        )
        node.client = llm
        return node

    def test_anthropic_thinking_with_tool_call_uses_json_schema(self):
        """Full scenario: thinking-enabled Anthropic LLM issues a tool call,
        tool returns result, then structured output is requested. The
        with_structured_output call must use method='json_schema', NOT
        method='function_calling', to avoid _raise_if_no_tool_calls.

        This is the core fix for issue #4890 (second pass).
        """
        llm = _AnthropicThinkingLLM()
        node = self._make_node_for_anthropic_thinking(llm)

        struct_model = _make_struct_model({"question": {"type": "list", "description": "Q"}})

        # Simulate: first LLM call returns a tool-calling AIMessage (thinking blocks)
        tool_call_msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "I'll call data_fetch.", "signature": "sig1"},
                {"type": "text", "text": ""},
            ],
            tool_calls=[{"name": "data_fetch", "args": {"query": "test"}, "id": "tc-001"}],
        )
        # After tool result: clean AIMessage (no tool_calls)
        synthesis_msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "Got results.", "signature": "sig2"},
                {"type": "text", "text": "Here is the answer."},
            ]
        )

        # Wire the tool-bound client: first call = tool_call_msg
        llm_bound = llm.bind_tools([_make_data_fetch_tool()])
        llm_bound_invoke_responses = iter([tool_call_msg, synthesis_msg])

        original_bound_invoke = llm_bound.invoke

        def controlled_invoke(messages, config=None):
            try:
                return next(llm_bound_invoke_responses)
            except StopIteration:
                return synthesis_msg

        llm_bound.invoke = controlled_invoke

        # Track with_structured_output calls
        wso_calls = []
        original_wso = llm_bound.with_structured_output

        def tracked_wso(schema, method="function_calling", **kwargs):
            wso_calls.append(method)
            return original_wso(schema, method=method, **kwargs)

        llm_bound.with_structured_output = tracked_wso

        messages = [HumanMessage(content="Fetch data and give me a structured list")]

        # Run _invoke_with_structured_output
        completion, initial, final_messages = node._invoke_with_structured_output(
            llm_bound, messages, struct_model, config={}
        )

        # CRITICAL ASSERTION: structured output must use json_schema, not function_calling
        assert "json_schema" in wso_calls, (
            f"Anthropic thinking-enabled model must use method='json_schema' for "
            f"structured output, but with_structured_output was called with: {wso_calls}. "
            "This is the root cause of issue #4890 — function_calling goes through "
            "_raise_if_no_tool_calls which raises OutputParserException on synthesis turns."
        )
        assert "function_calling" not in wso_calls, (
            f"function_calling must NOT be used for thinking-enabled Anthropic, "
            f"got: {wso_calls}"
        )

        # No exception was raised — the fix works
        assert completion is not None

    def test_anthropic_prefill_contract_satisfied_by_clean_messages(self):
        """End-to-end check that Fix A satisfies Anthropic's prefill rule.

        The hardened ``_StructuredOutputStub`` raises
        ``_AnthropicPrefillRejection`` when invoked with an
        ``AIMessage``-terminated message list (the real gateway returns
        400). For the tool-calling branch, the synthesis turn is an
        ``AIMessage``; without Fix A appending a trailing
        ``HumanMessage`` in ``_build_clean_messages_for_structured_output``
        this integration would raise.

        The bare-stub demonstration confirms the stub enforces the rule;
        the integration call proves Fix A actually feeds the stub a
        compliant list. Together these prevent the PR-#157 regression
        from recurring (mock that ignored the contract → silently green
        tests → 400 in production).
        """
        # 1. Demonstrate the stub raises on bare AIMessage-terminated input.
        bare_stub = _StructuredOutputStub(
            _make_struct_model({"question": {"type": "list"}}),
            method="json_schema",
        )
        with pytest.raises(_AnthropicPrefillRejection):
            bare_stub.invoke([
                HumanMessage(content="ask"),
                AIMessage(content="synthesis"),
            ])

        # 2. Drive the full tool-calling integration. Fix A appends a
        # HumanMessage so the stub does NOT raise — proving the contract
        # is satisfied at the boundary the real gateway enforces.
        llm = _AnthropicThinkingLLM()
        node = self._make_node_for_anthropic_thinking(llm)
        struct_model = _make_struct_model({"question": {"type": "list", "description": "Q"}})

        tool_call_msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "I'll fetch.", "signature": "s1"},
                {"type": "text", "text": ""},
            ],
            tool_calls=[{"name": "data_fetch", "args": {"query": "x"}, "id": "tc-1"}],
        )
        synthesis_msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "Done.", "signature": "s2"},
                {"type": "text", "text": "Answer."},
            ]
        )

        llm_bound = llm.bind_tools([_make_data_fetch_tool()])
        responses = iter([tool_call_msg, synthesis_msg])
        llm_bound.invoke = lambda msgs, config=None: next(responses, synthesis_msg)

        # Capture the messages handed to the structured-output invoke so
        # we can assert the trailing-HumanMessage shape directly.
        captured_struct_input: list = []
        original_wso = llm_bound.with_structured_output

        def capturing_wso(schema, method="function_calling", **kwargs):
            stub = original_wso(schema, method=method, **kwargs)
            original_invoke = stub.invoke

            def capture_and_invoke(messages, config=None):
                captured_struct_input.append(list(messages))
                return original_invoke(messages, config=config)

            stub.invoke = capture_and_invoke
            return stub

        llm_bound.with_structured_output = capturing_wso

        completion, _, _ = node._invoke_with_structured_output(
            llm_bound,
            [HumanMessage(content="Fetch and structure")],
            struct_model,
            config={},
        )

        assert completion is not None, "structured output succeeded — prefill rule satisfied"
        assert captured_struct_input, "structured-output stub was invoked"
        passed = captured_struct_input[0]
        assert isinstance(passed[-1], HumanMessage), (
            "Fix A must append a HumanMessage so Anthropic does not reject the "
            f"conversation as assistant prefill; got trailing {type(passed[-1]).__name__}"
        )
        # Production contract: ``_invoke_llm_internal`` normalizes
        # ``completion`` to a dict via ``isinstance`` check. The Anthropic
        # path's dict comes through directly; the OpenAI path's Pydantic
        # instance is dumped. Mirror that exact pattern here so this test
        # locks in the contract end-to-end.
        result = completion if isinstance(completion, dict) else completion.model_dump()
        assert isinstance(result, dict)
        assert "question" in result

    def test_non_thinking_anthropic_still_uses_function_calling(self):
        """Non-regression: non-thinking Anthropic must still use function_calling."""
        llm = _AnthropicNonThinkingLLM()
        node = self._make_node_for_anthropic_thinking(llm)

        struct_model = _make_struct_model({"question": {"type": "list", "description": "Q"}})

        # Single call: direct completion without tool calling
        synthesis = AIMessage(content="plain string synthesis")

        llm_bound = llm.bind_tools([_make_data_fetch_tool()])
        llm_bound.invoke = lambda msgs, config=None: synthesis

        wso_calls = []
        original_wso = llm_bound.with_structured_output

        def tracked_wso(schema, method="function_calling", **kwargs):
            wso_calls.append(method)
            return original_wso(schema, method=method, **kwargs)

        llm_bound.with_structured_output = tracked_wso

        messages = [HumanMessage(content="Give me a list")]

        completion, _, _ = node._invoke_with_structured_output(
            llm_bound, messages, struct_model, config={}
        )

        assert "function_calling" in wso_calls, (
            f"Non-thinking Anthropic must use function_calling, got: {wso_calls}"
        )
        assert "json_schema" not in wso_calls, (
            "json_schema must NOT be used for non-thinking Anthropic"
        )

    def test_openai_path_uses_function_calling(self):
        """No-regression: OpenAI-like client must use function_calling (unchanged path)."""
        llm = _OpenAILikeLLM()
        node = self._make_node_for_anthropic_thinking(llm)

        struct_model = _make_struct_model({"question": {"type": "list", "description": "Q"}})

        synthesis = AIMessage(content="openai synthesis answer")
        llm_bound = llm.bind_tools([_make_data_fetch_tool()])
        llm_bound.invoke = lambda msgs, config=None: synthesis

        wso_calls = []
        original_wso = llm_bound.with_structured_output

        def tracked_wso(schema, method="function_calling", **kwargs):
            wso_calls.append(method)
            return original_wso(schema, method=method, **kwargs)

        llm_bound.with_structured_output = tracked_wso

        messages = [HumanMessage(content="Give me a list")]

        node._invoke_with_structured_output(llm_bound, messages, struct_model, config={})

        assert "function_calling" in wso_calls, (
            f"OpenAI path must use function_calling, got: {wso_calls}"
        )
        assert "json_schema" not in wso_calls


# ─── _handle_structured_output_fallback: chain still tries all methods ──────


class TestHandleStructuredOutputFallbackThinkingModel:
    """Verify the fallback chain (json_mode → function_calling → plain text)
    still attempts function_calling. For thinking-Anthropic, function_calling is
    transparently rerouted to json_schema by ``__get_struct_output_model``, so
    the chain is safe — no separate skip-guard is needed."""

    def test_non_thinking_anthropic_still_tries_function_calling_in_fallback(self):
        """Non-regression: non-thinking Anthropic must still try function_calling."""
        llm = _AnthropicNonThinkingLLM()
        node = _make_llm_node()
        node.client = llm

        struct_model = _make_struct_model({"question": {"type": "list", "description": "Q"}})

        method_calls = []

        def patched_get_struct_output(client, model, method="function_calling"):
            method_calls.append(method)
            raise ValueError(f"{method} failed")

        node._LLMNode__get_struct_output_model = patched_get_struct_output

        # Plain client invoke returns parseable JSON
        json_response = AIMessage(content='{"question": [{"id": "q1"}], "rs": ""}')
        llm.invoke = lambda msgs, config=None: json_response

        messages = [HumanMessage(content="list please")]
        node._handle_structured_output_fallback(
            llm, messages, struct_model, {}, ValueError("initial")
        )

        assert "function_calling" in method_calls, (
            "Non-thinking Anthropic fallback must attempt function_calling"
        )
