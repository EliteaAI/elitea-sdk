"""
Tests for LLMNode._extract_content_from_completion.

Covers issue #6234: claude-sonnet-5 (Anthropic adaptive thinking) returns the
final answer as a bare string list item (e.g. content=['', {thinking...},
'answer text']) instead of a {'type': 'text', ...} dict. The extractor must
not silently drop that text.
"""
from unittest.mock import MagicMock

from elitea_sdk.runtime.tools.llm import LLMNode


def _make_completion(content):
    """Create a mock LLM completion object."""
    mock = MagicMock()
    mock.content = content
    return mock


class TestExtractContentFromCompletion:
    def test_plain_string_content(self):
        completion = _make_completion("hello world")
        result = LLMNode._extract_content_from_completion(completion)
        assert result == {"thinking": None, "text": "hello world"}

    def test_standard_dict_blocks(self):
        completion = _make_completion(
            [
                {"type": "thinking", "thinking": "reasoning..."},
                {"type": "text", "text": "final answer"},
            ]
        )
        result = LLMNode._extract_content_from_completion(completion)
        assert result == {"thinking": "reasoning...", "text": "final answer"}

    def test_adaptive_thinking_bare_string_answer(self):
        """Regression test for #6234: Anthropic adaptive thinking mode
        (claude-sonnet-5) returns the answer as a bare string list item
        rather than a {'type': 'text', ...} dict."""
        completion = _make_completion(
            [
                "",
                {"type": "thinking", "thinking": "reasoning about the riddle..."},
                "Let me guess — are you thinking of a **grape**?",
            ]
        )
        result = LLMNode._extract_content_from_completion(completion)
        assert result["thinking"] == "reasoning about the riddle..."
        assert result["text"] == "Let me guess — are you thinking of a **grape**?"

    def test_multiple_bare_string_blocks_are_joined(self):
        completion = _make_completion(["part one", "part two"])
        result = LLMNode._extract_content_from_completion(completion)
        assert result["text"] == "part one\n\npart two"

    def test_empty_bare_strings_are_ignored(self):
        completion = _make_completion(["", "", "real answer"])
        result = LLMNode._extract_content_from_completion(completion)
        assert result["text"] == "real answer"

    def test_no_text_blocks_returns_none(self):
        completion = _make_completion([{"type": "thinking", "thinking": "just thinking"}])
        result = LLMNode._extract_content_from_completion(completion)
        assert result == {"thinking": "just thinking", "text": None}

    def test_missing_content_attribute(self):
        completion = object()
        result = LLMNode._extract_content_from_completion(completion)
        assert result == {"thinking": None, "text": None}
