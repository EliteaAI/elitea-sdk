"""
Tests for extract_json_content utility and parse_pydantic_type schema fixes.

Covers issue #4890: Anthropic Claude models return valid JSON wrapped in
markdown code fences. The extract_json_content function strips fences and
handles both object and array responses.

Also verifies that parse_pydantic_type produces schemas compatible with
both Anthropic and OpenAI structured output methods.
"""
import json
import pytest

from elitea_sdk.runtime.langchain.utils import (
    extract_json_content,
    parse_pydantic_type,
    create_pydantic_model,
)
from elitea_sdk.runtime.langchain.constants import ELITEA_RS


# ---------------------------------------------------------------------------
# extract_json_content: markdown fence stripping
# ---------------------------------------------------------------------------

class TestExtractJsonContentFenced:
    """Test extraction from markdown-fenced responses (Anthropic pattern)."""

    def test_fenced_json_object(self):
        text = '```json\n{"key": "value", "count": 3}\n```'
        result = extract_json_content(text)
        assert result == {"key": "value", "count": 3}

    def test_fenced_json_array(self):
        text = '```json\n[{"id": 1}, {"id": 2}, {"id": 3}]\n```'
        result = extract_json_content(text)
        assert result == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_fenced_without_json_tag(self):
        text = '```\n{"name": "test"}\n```'
        result = extract_json_content(text)
        assert result == {"name": "test"}

    def test_fenced_with_surrounding_text(self):
        text = 'Here are the results:\n```json\n[1, 2, 3]\n```\nDone.'
        result = extract_json_content(text)
        assert result == [1, 2, 3]

    def test_fenced_nested_objects(self):
        data = {"questions": [
            {"question_id": "q1", "question_text": "What is AI?"},
            {"question_id": "q2", "question_text": "How does ML work?"},
        ]}
        text = f'```json\n{json.dumps(data)}\n```'
        result = extract_json_content(text)
        assert result == data

    def test_fenced_array_of_objects_issue_4890(self):
        """Exact pattern from issue #4890: array wrapped in fences."""
        text = '''```json
[
  {"question_id": "q1", "question_text": "What is the main topic?"},
  {"question_id": "q2", "question_text": "What are key findings?"},
  {"question_id": "q3", "question_text": "What methodology was used?"}
]
```'''
        result = extract_json_content(text)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["question_id"] == "q1"

    def test_fenced_with_extra_whitespace(self):
        text = '```json\n\n  {"a": 1}  \n\n```'
        result = extract_json_content(text)
        assert result == {"a": 1}


# ---------------------------------------------------------------------------
# extract_json_content: plain JSON (no fences)
# ---------------------------------------------------------------------------

class TestExtractJsonContentPlain:
    """Test extraction from plain JSON without markdown fences."""

    def test_plain_object(self):
        result = extract_json_content('{"key": "value"}')
        assert result == {"key": "value"}

    def test_plain_array(self):
        result = extract_json_content('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_plain_nested(self):
        data = {"outer": {"inner": [1, 2]}}
        result = extract_json_content(json.dumps(data))
        assert result == data

    def test_plain_with_whitespace(self):
        result = extract_json_content('  \n  {"x": true}  \n  ')
        assert result == {"x": True}


# ---------------------------------------------------------------------------
# extract_json_content: embedded JSON in text
# ---------------------------------------------------------------------------

class TestExtractJsonContentEmbedded:
    """Test extraction from text with embedded JSON."""

    def test_text_before_json_object(self):
        text = 'The result is: {"name": "test"} as expected.'
        result = extract_json_content(text)
        assert result == {"name": "test"}

    def test_text_before_json_array(self):
        """When text contains both {} and [], object extraction wins (by design)."""
        text = 'Found items: [{"id": 1}, {"id": 2}]'
        result = extract_json_content(text)
        # _find_json_bounds finds first '{' → extracts the first object
        assert result == {"id": 1}

    def test_pure_array_in_text(self):
        """Array extraction works when no object braces precede it."""
        text = 'Results: [1, 2, 3]'
        result = extract_json_content(text)
        assert result == [1, 2, 3]

    def test_json_object_with_nested_braces(self):
        text = 'Output: {"data": {"nested": "value"}, "count": 1}'
        result = extract_json_content(text)
        assert result == {"data": {"nested": "value"}, "count": 1}


# ---------------------------------------------------------------------------
# extract_json_content: error cases
# ---------------------------------------------------------------------------

class TestExtractJsonContentErrors:
    """Test error handling in extract_json_content."""

    def test_no_json_raises_valueerror(self):
        with pytest.raises(ValueError, match="Cannot extract JSON"):
            extract_json_content("no json content here at all")

    def test_empty_string_raises_valueerror(self):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            extract_json_content("")

    def test_malformed_json_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_json_content('```json\n{invalid json}\n```')

    def test_incomplete_array_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_json_content("[1, 2, 3")


# ---------------------------------------------------------------------------
# parse_pydantic_type: schema compatibility (issue #4890 secondary fix)
# ---------------------------------------------------------------------------

class TestParsePydanticType:
    """Verify parse_pydantic_type produces Anthropic & OpenAI compatible schemas."""

    def test_list_type_accepts_any_elements(self):
        from typing import Any
        t = parse_pydantic_type("list")
        assert t == list[Any]
        # Schema must not have broken $defs (the Anthropic rejection)
        model = create_pydantic_model("Test", {"items": {"type": "list"}})
        schema = model.model_json_schema()
        assert "$defs" not in schema
        # items: {} means any element type (valid JSON Schema)
        assert schema["properties"]["items"]["items"] == {}

    def test_dict_type_produces_object(self):
        t = parse_pydantic_type("dict")
        # Should produce valid schema without empty $defs
        model = create_pydantic_model("Test", {"data": {"type": "dict"}})
        schema = model.model_json_schema()
        data_prop = schema["properties"]["data"]
        assert data_prop["type"] == "object"

    def test_list_type_schema_no_empty_defs(self):
        """Regression: list type must not produce empty $defs (Anthropic rejects this)."""
        model = create_pydantic_model("Test", {"items": {"type": "list"}})
        schema = model.model_json_schema()
        defs = schema.get("$defs", {})
        # If $defs exist, none should be empty
        for def_name, def_schema in defs.items():
            assert def_schema, f"Empty schema definition: {def_name}"

    def test_list_accepts_any_data(self):
        """list type must accept dicts, strings, ints — any JSON value."""
        model = create_pydantic_model("Test", {"items": {"type": "list"}})
        # Dicts
        instance = model(items=[{"key": "val"}, {"another": 123}])
        assert len(instance.items) == 2
        assert instance.items[0] == {"key": "val"}
        # Strings
        instance = model(items=["a", "b", "c"])
        assert instance.items == ["a", "b", "c"]
        # Mixed
        instance = model(items=[1, "two", {"three": 3}])
        assert len(instance.items) == 3

    def test_str_int_float_bool_unchanged(self):
        assert parse_pydantic_type("str") is str
        assert parse_pydantic_type("int") is int
        assert parse_pydantic_type("float") is float
        assert parse_pydantic_type("bool") is bool

    def test_nested_list_type(self):
        t = parse_pydantic_type("list[str]")
        assert t == list[str]

    def test_nested_dict_type(self):
        t = parse_pydantic_type("dict[str,int]")
        assert t == dict[str, int]


# ---------------------------------------------------------------------------
# create_pydantic_model: structured output model creation
# ---------------------------------------------------------------------------

class TestCreatePydanticModel:
    """Test model creation for structured output with list-type state vars."""

    def test_model_with_list_field_and_elitea_rs(self):
        """Pipeline model typically includes ELITEA_RS alongside output fields."""
        model = create_pydantic_model("LLMOutput", {
            "question": {"type": "list", "description": "Questions"},
            ELITEA_RS: {"type": "str", "default": ""},
        })
        assert "question" in model.model_fields
        assert ELITEA_RS in model.model_fields

    def test_model_instantiation_with_list_data(self):
        model = create_pydantic_model("LLMOutput", {
            "question": {"type": "list", "description": "Questions"},
            ELITEA_RS: {"type": "str", "default": ""},
        })
        data = [
            {"question_id": "q1", "question_text": "What?"},
            {"question_id": "q2", "question_text": "How?"},
        ]
        instance = model(question=data)
        assert instance.question == data
        assert getattr(instance, ELITEA_RS) == ""

    def test_model_schema_for_anthropic(self):
        """Schema must be accepted by Anthropic (no empty $defs)."""
        model = create_pydantic_model("LLMOutput", {
            "items": {"type": "list", "description": "Result items"},
        })
        schema = model.model_json_schema()
        # Must have properties with items as array
        assert schema["properties"]["items"]["type"] == "array"
        # No empty $defs (this is what Anthropic rejected)
        assert "$defs" not in schema or not any(
            v == {} for v in schema["$defs"].values()
        )
