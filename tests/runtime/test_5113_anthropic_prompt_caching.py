"""Regression tests for issue #5113 — Anthropic prompt caching.

Changes verified:
1. _resolve_jinja2_variables no longer injects current_time / current_datetime;
   current_date is still present (day-granularity cache rotation is acceptable).

2. SystemMessage.content is a cache_control block-list for Anthropic clients and
   a plain string for all other clients (four sites: task path, lazy-tools rebuild,
   lazy-tools new-inject, swarm agent_node).

3. _inject_tool_index_into_messages handles both str-content and list-content
   (Anthropic-style cache_control) SystemMessages without corrupting the text.

4. The swarm gate fires correctly when the model passed to make_agent_node is a
   bound-tools Anthropic RunnableBinding, not just a raw ChatAnthropic.
"""

import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# ---------------------------------------------------------------------------
# Helpers — fake Anthropic / non-Anthropic clients
#
# The detection code checks `type(candidate).__module__`, NOT `candidate.__class__`.
# MagicMock intercepts __class__ but `type(mock)` always returns MagicMock.
# We therefore build real class instances whose *type* carries the correct
# __module__, rather than assigning __class__ on a MagicMock.
# ---------------------------------------------------------------------------

# Real class whose type().__module__ looks like langchain_anthropic
_FakeAnthropicClass = type(
    "ChatAnthropic",
    (),
    {"__module__": "langchain_anthropic.chat_models"},
)

# Real class whose type().__module__ looks like langchain_openai
_FakeOpenAIClass = type(
    "ChatOpenAI",
    (),
    {"__module__": "langchain_openai.chat_models"},
)

# RunnableBinding-like class (langchain_core module, NOT langchain_anthropic)
_FakeBindingClass = type(
    "RunnableBinding",
    (),
    {"__module__": "langchain_core.runnables.base"},
)


def _make_fake_anthropic_client():
    """Return an instance whose type().__module__ == 'langchain_anthropic.chat_models'."""
    return _FakeAnthropicClass()


def _make_fake_openai_client():
    """Return an instance whose type().__module__ == 'langchain_openai.chat_models'."""
    return _FakeOpenAIClass()


def _make_bound_anthropic_client():
    """Return a RunnableBinding-like instance wrapping a fake ChatAnthropic.

    Simulates model.bind_tools(tools) — the binding's own __module__ is NOT
    langchain_anthropic, but its .bound attribute IS.
    """
    binding = _FakeBindingClass()
    binding.bound = _make_fake_anthropic_client()
    return binding


# ---------------------------------------------------------------------------
# 1. Timestamp-poison fix: _resolve_jinja2_variables context
# ---------------------------------------------------------------------------

class TestJinja2ContextTimestamps:
    """Verify the timestamp-poison root-cause fix."""

    def _get_assistant(self):
        from elitea_sdk.runtime.langchain.assistant import Assistant

        client = MagicMock()
        # Minimal Assistant init — we only need _resolve_jinja2_variables
        assistant = Assistant.__new__(Assistant)
        assistant.client = client
        assistant.prompt_variables = {}
        return assistant

    def test_current_date_present(self):
        assistant = self._get_assistant()
        result_vars = {}

        # Resolve a template that references all three keys
        resolved = assistant._resolve_jinja2_variables(
            "d={{current_date}} t={{current_time}} dt={{current_datetime}}"
        )
        # current_date must resolve to a YYYY-MM-DD value
        import re
        assert re.match(r"d=\d{4}-\d{2}-\d{2}", resolved), (
            f"current_date not resolved correctly: {resolved!r}"
        )

    def test_current_time_not_injected(self):
        assistant = self._get_assistant()
        resolved = assistant._resolve_jinja2_variables("{{current_time}}")
        # DebugUndefined leaves unresolved vars as-is — so if current_time were
        # injected, we'd get HH:MM:SS; if not injected, the token stays literal.
        assert ":" not in resolved or resolved.strip() == "{{current_time}}", (
            f"current_time appears to have been resolved: {resolved!r}"
        )
        # More direct: the literal token should remain unchanged
        assert "current_time" in resolved

    def test_current_datetime_not_injected(self):
        assistant = self._get_assistant()
        resolved = assistant._resolve_jinja2_variables("{{current_datetime}}")
        assert "current_datetime" in resolved, (
            f"current_datetime appears to have been resolved: {resolved!r}"
        )

    def test_no_template_fast_path(self):
        assistant = self._get_assistant()
        text = "No template variables here"
        assert assistant._resolve_jinja2_variables(text) == text

    def test_extra_context_still_works(self):
        assistant = self._get_assistant()
        resolved = assistant._resolve_jinja2_variables(
            "hello {{name}}", extra_context={"name": "world"}
        )
        assert resolved == "hello world"


# ---------------------------------------------------------------------------
# 2. cache_control block-list for Anthropic; plain string for others
# ---------------------------------------------------------------------------

class TestAnthropicSystemContent:
    """Verify _anthropic_system_content (llm.py) and _make_anthropic_system_content (assistant.py)."""

    # --- llm.py helper ---

    def test_anthropic_returns_block_list(self):
        from elitea_sdk.runtime.tools.llm import LLMNode

        anthropic_client = _make_fake_anthropic_client()
        result = LLMNode._anthropic_system_content("Hello Anthropic", anthropic_client)
        assert isinstance(result, list), "Expected list for Anthropic client"
        assert len(result) == 1
        block = result[0]
        assert block["type"] == "text"
        assert block["text"] == "Hello Anthropic"
        assert block.get("cache_control") == {"type": "ephemeral"}

    def test_openai_returns_plain_string(self):
        from elitea_sdk.runtime.tools.llm import LLMNode

        openai_client = _make_fake_openai_client()
        result = LLMNode._anthropic_system_content("Hello OpenAI", openai_client)
        assert result == "Hello OpenAI", "Expected plain string for non-Anthropic client"

    def test_bound_anthropic_returns_block_list(self):
        """Bound-tools wrapper (RunnableBinding) must still be detected as Anthropic."""
        from elitea_sdk.runtime.tools.llm import LLMNode

        bound_client = _make_bound_anthropic_client()
        result = LLMNode._anthropic_system_content("Bound client", bound_client)
        assert isinstance(result, list), (
            "Expected block-list for bound Anthropic client; "
            "gate likely fired on the wrong object"
        )
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    # --- assistant.py helper ---

    def test_assistant_helper_anthropic(self):
        from elitea_sdk.runtime.langchain.assistant import _make_anthropic_system_content

        result = _make_anthropic_system_content("System prompt", _make_fake_anthropic_client())
        assert isinstance(result, list)
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_assistant_helper_openai(self):
        from elitea_sdk.runtime.langchain.assistant import _make_anthropic_system_content

        result = _make_anthropic_system_content("System prompt", _make_fake_openai_client())
        assert result == "System prompt"

    def test_assistant_helper_bound_anthropic(self):
        """Swarm gate: bound-tools Anthropic model must be detected correctly."""
        from elitea_sdk.runtime.langchain.assistant import _make_anthropic_system_content

        bound = _make_bound_anthropic_client()
        result = _make_anthropic_system_content("Swarm prompt", bound)
        assert isinstance(result, list), (
            "_make_anthropic_system_content returned plain string for bound Anthropic model; "
            "swarm gate is broken — it checked model_with_tools instead of model"
        )
        assert result[0]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# 3. _inject_tool_index_into_messages — consumer-contract guard
# ---------------------------------------------------------------------------

class TestInjectToolIndex:
    """Verify the list-content consumer guard in _inject_tool_index_into_messages."""

    def _make_node(self, client):
        from elitea_sdk.runtime.tools.llm import LLMNode

        registry = MagicMock()
        registry.generate_index.return_value = "## Tool Index\n- tool_a\n- tool_b"

        # model_construct bypasses Pydantic __init__ validation but still
        # creates a properly initialised Pydantic model instance.
        node = LLMNode.model_construct(client=client, tool_registry=registry)
        return node

    def test_str_content_system_message(self):
        """Original str-content system message: text must be concatenated correctly."""
        node = self._make_node(_make_fake_openai_client())
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello"),
        ]
        result = node._inject_tool_index_into_messages(messages)
        sys_msg = next(m for m in result if isinstance(m, SystemMessage))
        # Content must be plain string for non-Anthropic
        assert isinstance(sys_msg.content, str)
        assert "You are a helpful assistant." in sys_msg.content
        assert "## Tool Index" in sys_msg.content

    def test_list_content_system_message_anthropic(self):
        """Anthropic-style list-content system message: text must NOT be stringified."""
        node = self._make_node(_make_fake_anthropic_client())
        # Simulate a system message that was already marked with cache_control
        existing_content = [{"type": "text", "text": "You are a helpful assistant.", "cache_control": {"type": "ephemeral"}}]
        messages = [
            SystemMessage(content=existing_content),
            HumanMessage(content="Hello"),
        ]
        result = node._inject_tool_index_into_messages(messages)
        sys_msg = next(m for m in result if isinstance(m, SystemMessage))
        # For Anthropic the result should still be a block list, not a stringified Python list
        assert isinstance(sys_msg.content, list), (
            "Expected block-list content after injection into an Anthropic system message; "
            "got plain string — list was likely stringified"
        )
        combined_text = sys_msg.content[0]["text"]
        assert "You are a helpful assistant." in combined_text, (
            "Original system prompt text was lost during injection"
        )
        assert "## Tool Index" in combined_text, (
            "Tool index was not appended to the system message"
        )
        assert sys_msg.content[0].get("cache_control") == {"type": "ephemeral"}, (
            "cache_control was lost after tool-index injection"
        )

    def test_list_content_not_corrupted_for_openai(self):
        """Even for non-Anthropic, if content somehow arrives as a list, text is extracted."""
        node = self._make_node(_make_fake_openai_client())
        # This is an unusual case but should not crash
        existing_content = [{"type": "text", "text": "Base prompt"}]
        messages = [
            SystemMessage(content=existing_content),
            HumanMessage(content="Hello"),
        ]
        result = node._inject_tool_index_into_messages(messages)
        sys_msg = next(m for m in result if isinstance(m, SystemMessage))
        # Non-Anthropic gets a plain string (not a Python list repr)
        assert isinstance(sys_msg.content, str), (
            "Non-Anthropic should always get plain-string content"
        )
        assert "Base prompt" in sys_msg.content
        assert "## Tool Index" in sys_msg.content

    def test_no_system_message_creates_new_one(self):
        """No existing system message: a new one is prepended."""
        node = self._make_node(_make_fake_anthropic_client())
        messages = [HumanMessage(content="Hello")]
        result = node._inject_tool_index_into_messages(messages)
        assert isinstance(result[0], SystemMessage)
        # Should be a block-list for Anthropic
        assert isinstance(result[0].content, list)
        assert "## Tool Index" in result[0].content[0]["text"]

    def test_no_registry_returns_unchanged(self):
        """No tool_registry: messages returned unchanged."""
        from elitea_sdk.runtime.tools.llm import LLMNode

        node = LLMNode.model_construct(
            client=_make_fake_anthropic_client(),
            tool_registry=None,
        )

        messages = [SystemMessage(content="Original"), HumanMessage(content="Hi")]
        result = node._inject_tool_index_into_messages(messages)
        assert result is messages  # same object returned


# ---------------------------------------------------------------------------
# 4. Swarm gate: bound-tools Anthropic binding detected correctly
# ---------------------------------------------------------------------------

class TestSwarmGateBoundTools:
    """The swarm agent_node gates on `model` (raw), not `model_with_tools` (bound)."""

    def test_is_anthropic_model_detects_bound_binding(self):
        from elitea_sdk.runtime.langchain.assistant import _is_anthropic_model

        bound = _make_bound_anthropic_client()
        assert _is_anthropic_model(bound), (
            "_is_anthropic_model returned False for a RunnableBinding wrapping ChatAnthropic; "
            "the swarm gate would silently skip cache marking for all bound-tools agents"
        )

    def test_is_anthropic_model_detects_raw_anthropic(self):
        from elitea_sdk.runtime.langchain.assistant import _is_anthropic_model

        assert _is_anthropic_model(_make_fake_anthropic_client())

    def test_is_anthropic_model_rejects_openai(self):
        from elitea_sdk.runtime.langchain.assistant import _is_anthropic_model

        assert not _is_anthropic_model(_make_fake_openai_client())

    def test_is_anthropic_model_rejects_bound_openai(self):
        from elitea_sdk.runtime.langchain.assistant import _is_anthropic_model

        inner = _make_fake_openai_client()
        binding = MagicMock()
        BindingClass = type("RunnableBinding", (), {"__module__": "langchain_core.runnables.base"})
        binding.__class__ = BindingClass
        binding.bound = inner
        assert not _is_anthropic_model(binding)
