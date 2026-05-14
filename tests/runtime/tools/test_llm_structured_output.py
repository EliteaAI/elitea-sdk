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
import json
import pytest
from unittest.mock import MagicMock, patch
from typing import Any

from pydantic import create_model, Field

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
