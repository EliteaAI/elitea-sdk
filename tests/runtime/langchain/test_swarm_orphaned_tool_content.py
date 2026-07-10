"""
Regression tests for issue #5768: orphaned `tool_use` content blocks.

When swarm mode trims an AIMessage's `.tool_calls` (parallel-handoff guard or
the orphan sanitizer), it must ALSO prune the matching `tool_use` blocks from
`.content`. langchain_anthropic serializes the outgoing Anthropic payload from
`.content`, re-emitting any `tool_use` block whose id is absent from
`.tool_calls` — so a "removed" call still reaches the API with no matching
`tool_result`, producing:

    messages.N: `tool_use` ids were found without `tool_result` blocks
    immediately after

These tests pin the invariant that `.content` and `.tool_calls` stay in sync.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from elitea_sdk.runtime.langchain.assistant import _rebuild_ai_message_without_calls


def _tool_use_block(tid, name="transfer_to_x", args=None):
    return {"type": "tool_use", "id": tid, "name": name, "input": args or {}}


def _ai_with_two_transfers():
    """AIMessage as Anthropic returns it: list content with two tool_use blocks."""
    return AIMessage(
        content=[
            {"type": "text", "text": "Consulting both specialists."},
            _tool_use_block("toolu_A", "transfer_to_echo_alpha", {"task": "a"}),
            _tool_use_block("toolu_B", "transfer_to_echo_beta", {"task": "b"}),
        ],
        tool_calls=[
            {"name": "transfer_to_echo_alpha", "args": {"task": "a"}, "id": "toolu_A"},
            {"name": "transfer_to_echo_beta", "args": {"task": "b"}, "id": "toolu_B"},
        ],
    )


class TestRebuildHelper:
    def test_drops_content_block_for_removed_call(self):
        msg = _ai_with_two_transfers()
        rebuilt = _rebuild_ai_message_without_calls(msg, {"toolu_A"})

        kept_ids = {tc["id"] for tc in rebuilt.tool_calls}
        assert kept_ids == {"toolu_A"}

        # The discarded call's tool_use block must be gone from .content too.
        content_ids = {b["id"] for b in rebuilt.content
                       if isinstance(b, dict) and b.get("type") == "tool_use"}
        assert content_ids == {"toolu_A"}

    def test_preserves_text_blocks(self):
        msg = _ai_with_two_transfers()
        rebuilt = _rebuild_ai_message_without_calls(msg, {"toolu_A"})
        texts = [b for b in rebuilt.content
                 if isinstance(b, dict) and b.get("type") == "text"]
        assert texts and texts[0]["text"] == "Consulting both specialists."

    def test_preserves_metadata(self):
        msg = _ai_with_two_transfers()
        msg = msg.model_copy(update={
            "additional_kwargs": {"lc_summarized": True},
            "response_metadata": {"model_name": "claude-sonnet-5"},
        })
        rebuilt = _rebuild_ai_message_without_calls(msg, {"toolu_A"})
        assert rebuilt.additional_kwargs.get("lc_summarized") is True
        assert rebuilt.response_metadata.get("model_name") == "claude-sonnet-5"

    def test_str_content_passthrough(self):
        # OpenAI-shaped AIMessage: str content, tool_calls only. Nothing to prune.
        msg = AIMessage(
            content="hi",
            tool_calls=[
                {"name": "transfer_to_a", "args": {}, "id": "id1"},
                {"name": "transfer_to_b", "args": {}, "id": "id2"},
            ],
        )
        rebuilt = _rebuild_ai_message_without_calls(msg, {"id1"})
        assert rebuilt.content == "hi"
        assert {tc["id"] for tc in rebuilt.tool_calls} == {"id1"}

    def test_keep_all_is_noop_on_content(self):
        msg = _ai_with_two_transfers()
        rebuilt = _rebuild_ai_message_without_calls(msg, {"toolu_A", "toolu_B"})
        content_ids = {b["id"] for b in rebuilt.content
                       if isinstance(b, dict) and b.get("type") == "tool_use"}
        assert content_ids == {"toolu_A", "toolu_B"}


class TestAnthropicSerializationInvariant:
    """End-to-end: the cleaned message must survive langchain_anthropic's
    payload builder without leaving an orphaned tool_use (the exact 400)."""

    def _orphaned_tool_use_ids(self, formatted):
        """Return tool_use ids not immediately followed by a matching tool_result."""
        orphaned = []
        for i, m in enumerate(formatted):
            if m["role"] != "assistant" or not isinstance(m["content"], list):
                continue
            tu_ids = [b["id"] for b in m["content"]
                      if isinstance(b, dict) and b.get("type") == "tool_use"]
            if not tu_ids:
                continue
            nxt = formatted[i + 1] if i + 1 < len(formatted) else None
            result_ids = set()
            if nxt and isinstance(nxt.get("content"), list):
                result_ids = {b.get("tool_use_id") for b in nxt["content"]
                              if isinstance(b, dict) and b.get("type") == "tool_result"}
            orphaned += [t for t in tu_ids if t not in result_ids]
        return orphaned

    def test_unfixed_message_would_orphan(self):
        # Guard against a false-positive test: the RAW (unfixed) message,
        # where .tool_calls is trimmed but .content is not, must orphan.
        from langchain_anthropic.chat_models import _format_messages
        msg = _ai_with_two_transfers()
        # Simulate the OLD buggy trim: keep both blocks in content, one in tool_calls.
        buggy = msg.model_copy(update={
            "tool_calls": [msg.tool_calls[0]],  # only toolu_A
        })
        history = [
            HumanMessage(content="go"),
            buggy,
            ToolMessage(content="ok", tool_call_id="toolu_A", name="transfer_to_echo_alpha"),
        ]
        _system, formatted = _format_messages(history)
        assert "toolu_B" in self._orphaned_tool_use_ids(formatted)

    def test_fixed_message_has_no_orphan(self):
        from langchain_anthropic.chat_models import _format_messages
        msg = _ai_with_two_transfers()
        fixed = _rebuild_ai_message_without_calls(msg, {"toolu_A"})
        history = [
            HumanMessage(content="go"),
            fixed,
            ToolMessage(content="ok", tool_call_id="toolu_A", name="transfer_to_echo_alpha"),
        ]
        _system, formatted = _format_messages(history)
        assert self._orphaned_tool_use_ids(formatted) == []
