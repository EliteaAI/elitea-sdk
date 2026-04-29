"""Unit tests for issue #4333 fix-set.

Covers the four PR review concerns:
1. ``__perform_tool_calling`` must not duplicate AIMessage when the same
   tool_call ids already exist in history (multi-tool sibling case).
2. ``__perform_tool_calling`` must skip tool_calls whose results are already
   present in history (prevents non-sensitive sibling re-execution).
3. ``args_match_normalized`` honors the documented semantic equivalences
   (int vs float, missing key vs None, key ordering).
4. Implicit: covered by the side-effect counters in
   ``test_hitl_resume_real_graph.py``.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from elitea_sdk.runtime.langchain.utils import args_match_normalized
from elitea_sdk.runtime.tools.llm import LLMNode


# ─── Issue 3: args_match_normalized semantic equivalences ────────────────────


class TestArgsMatchNormalized:
    def test_int_vs_float_match(self):
        assert args_match_normalized({"x": 1}, {"x": 1.0}) is True

    def test_nested_int_vs_float_match(self):
        assert args_match_normalized(
            {"opts": {"timeout": 30}},
            {"opts": {"timeout": 30.0}},
        ) is True

    def test_missing_key_matches_explicit_none(self):
        assert args_match_normalized(
            {"a": 1, "b": None},
            {"a": 1},
        ) is True

    def test_dict_key_order_irrelevant(self):
        assert args_match_normalized(
            {"a": 1, "b": 2, "c": 3},
            {"c": 3, "a": 1, "b": 2},
        ) is True

    def test_tuple_vs_list_match(self):
        assert args_match_normalized({"items": (1, 2, 3)}, {"items": [1, 2, 3]}) is True

    def test_real_mismatch_rejected(self):
        assert args_match_normalized({"x": 1}, {"x": 2}) is False

    def test_bool_not_coerced_to_int(self):
        # True must NOT match 1.0 — bool semantics distinct from numeric
        assert args_match_normalized({"flag": True}, {"flag": 1}) is False

    def test_empty_args_match(self):
        assert args_match_normalized({}, {}) is True
        assert args_match_normalized(None, None) is True
        assert args_match_normalized(None, {}) is True

    def test_deeply_nested_complex_args_round_trip(self):
        """Mimics issue #4333 args going through JSON checkpoint round-trip."""
        original = {
            "project_key": "PROJ",
            "custom_fields": {"story_points": 5, "sprint_id": 42},
            "labels": ["backend", "urgent"],
            "priority": None,
        }
        round_tripped = {
            "project_key": "PROJ",
            "custom_fields": {"story_points": 5.0, "sprint_id": 42.0},
            "labels": ["backend", "urgent"],
            # `priority` was None in original — checkpoint dropped the key
        }
        assert args_match_normalized(original, round_tripped) is True


# ─── Issue 1: dedup append (no duplicate AIMessage in messages) ──────────────


class TestAppendCompletionDedup:
    def test_appends_when_no_collision(self):
        messages = [HumanMessage(content="hi")]
        completion = AIMessage(
            content="",
            tool_calls=[{"name": "t", "args": {}, "id": "tc-1"}],
        )
        result = LLMNode._append_completion_dedup(messages, completion)
        assert result[-1] is completion
        assert len(result) == 2

    def test_skips_only_on_identity_match(self):
        """Multi-tool sibling case: ``_build_resume_completion`` returns the
        SAME AIMessage object that's already in ``messages`` — identity match
        prevents duplication. Two distinct objects with overlapping tool_calls
        must still both be appended (e.g., subset/superset relationships)."""
        existing_ai = AIMessage(
            content="",
            tool_calls=[{"name": "t", "args": {}, "id": "tc-1"}],
        )
        messages = [HumanMessage(content="hi"), existing_ai]
        # Same object — should be skipped
        result = LLMNode._append_completion_dedup(messages, existing_ai)
        assert len(result) == 2
        assert result[-1] is existing_ai

    def test_skips_when_existing_followed_by_tool_messages(self):
        """Multi-tool sibling case: identity-match dedup must work even when
        ``messages`` ends with sibling ToolMessages (PR review Issue 1)."""
        existing_ai = AIMessage(
            content="",
            tool_calls=[
                {"name": "search", "args": {}, "id": "tc-search"},
                {"name": "create", "args": {}, "id": "tc-create"},
            ],
        )
        messages = [
            HumanMessage(content="hi"),
            existing_ai,
            ToolMessage(content="search-result", tool_call_id="tc-search"),
        ]
        result = LLMNode._append_completion_dedup(messages, existing_ai)
        assert len(result) == 3

    def test_appends_when_tool_calls_overlap_but_objects_differ(self):
        """A freshly deserialized AIMessage with overlapping tool_call_ids must
        still be appended — object identity is the safety net, not tc id sets."""
        existing_ai = AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {}, "id": "tc-search"}],
        )
        messages = [HumanMessage(content="hi"), existing_ai]
        completion = AIMessage(
            content="",
            tool_calls=[
                {"name": "search", "args": {}, "id": "tc-search"},
                {"name": "create", "args": {}, "id": "tc-create"},
            ],
        )
        result = LLMNode._append_completion_dedup(messages, completion)
        assert len(result) == 3
        assert result[-1] is completion


# ─── Issue 2: skip already-completed tool_calls ──────────────────────────────


class TestToolCallAlreadyCompleted:
    def test_returns_false_when_no_tool_message(self):
        messages = [HumanMessage(content="hi")]
        assert LLMNode._tool_call_already_completed("tc-1", messages) is False

    def test_returns_true_when_tool_message_exists(self):
        messages = [
            HumanMessage(content="hi"),
            ToolMessage(content="result", tool_call_id="tc-1"),
        ]
        assert LLMNode._tool_call_already_completed("tc-1", messages) is True

    def test_returns_false_when_id_mismatches(self):
        messages = [
            HumanMessage(content="hi"),
            ToolMessage(content="result", tool_call_id="tc-2"),
        ]
        assert LLMNode._tool_call_already_completed("tc-1", messages) is False

    def test_empty_id_returns_false(self):
        messages = [
            ToolMessage(content="result", tool_call_id=""),
        ]
        assert LLMNode._tool_call_already_completed("", messages) is False
