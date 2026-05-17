"""
Tests for extract_json_content utility and parse_pydantic_type schema handling.

Covers issue #4890: Anthropic Claude models return valid JSON wrapped in
markdown code fences. The extract_json_content function strips fences and
handles both object and array responses.

Also verifies that parse_pydantic_type uses Pydantic's native ``JsonValue``
(permissive — required for OpenAI reasoning models which hallucinate
``list[list[str]]`` under tighter element unions) and that the Anthropic
patch helper rewrites the empty ``$defs.JsonValue`` Pydantic emits into a
shape ``transform_schema`` accepts.
"""
import json
import pytest

from anthropic.lib._parse._transform import transform_schema
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import JsonValue

from elitea_sdk.runtime.langchain.utils import (
    extract_json_content,
    make_anthropic_compatible_schema,
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
    """Verify parse_pydantic_type emits Pydantic's native ``JsonValue`` for
    permissive types and that the Pydantic-native schema is permissive
    enough for OpenAI (including reasoning models).

    The Anthropic-side patching is verified separately in
    ``TestMakeAnthropicCompatibleSchema`` — provider divergence is
    handled at the structured-output boundary, not at schema construction.
    """

    def test_list_type_uses_json_value(self):
        assert parse_pydantic_type("list") == list[JsonValue]

    def test_any_type_uses_json_value(self):
        assert parse_pydantic_type("any") is JsonValue

    def test_unknown_type_falls_back_to_json_value(self):
        assert parse_pydantic_type("does-not-exist") is JsonValue

    def test_list_field_schema_uses_json_value_ref(self):
        """``list[JsonValue]`` emits a ``$ref`` to the ``JsonValue`` def
        with an empty body — the shape OpenAI accepts and Anthropic needs
        patched. Asserting the ``$ref`` here protects against a Pydantic
        upgrade silently changing the emit shape (which would invalidate
        the Anthropic patch's targeting)."""
        model = create_pydantic_model("Test", {"items": {"type": "list"}})
        schema = model.model_json_schema()
        items_schema = schema["properties"]["items"]["items"]
        assert items_schema == {"$ref": "#/$defs/JsonValue"}
        assert "JsonValue" in schema.get("$defs", {})

    def test_any_field_schema_uses_json_value_ref(self):
        model = create_pydantic_model("Test", {"val": {"type": "any"}})
        schema = model.model_json_schema()
        assert schema["properties"]["val"] == {"$ref": "#/$defs/JsonValue"}

    def test_native_pydantic_schema_fails_anthropic_transform(self):
        """Documents the failure mode the Anthropic patch is for: the
        unpatched Pydantic schema (``$defs.JsonValue: {}``) is rejected
        by ``transform_schema`` with *"Schema must have a 'type', 'anyOf',
        'oneOf', or 'allOf' field"*. If Pydantic ever starts emitting a
        non-empty ``JsonValue`` def this test will start failing — at
        which point the patch can be reconsidered."""
        model = create_pydantic_model("Test", {"items": {"type": "list"}})
        schema = model.model_json_schema()
        with pytest.raises(ValueError, match="anyOf"):
            transform_schema(schema)

    def test_dict_type_produces_object(self):
        t = parse_pydantic_type("dict")
        model = create_pydantic_model("Test", {"data": {"type": "dict"}})
        schema = model.model_json_schema()
        data_prop = schema["properties"]["data"]
        assert data_prop["type"] == "object"

    def test_list_accepts_any_data(self):
        """Runtime semantics preserved: ``JsonValue`` accepts the full
        recursive JSON union — primitives, objects, AND nested lists.
        Nested lists are intentional under ``JsonValue`` semantics; the
        previous attempt to forbid them at the schema level (PR #157
        ``AnyJsonValue``) caused OpenAI reasoning models to mis-shape
        their output."""
        model = create_pydantic_model("Test", {"items": {"type": "list"}})
        instance = model(items=[{"key": "val"}, {"another": 123}])
        assert instance.items[0] == {"key": "val"}
        instance = model(items=["a", "b", "c"])
        assert instance.items == ["a", "b", "c"]
        instance = model(items=[1, "two", {"three": 3}, [4, 5], None, True])
        assert len(instance.items) == 6

    def test_any_field_accepts_any_data(self):
        model = create_pydantic_model("Test", {"val": {"type": "any"}})
        for value in ("string", 42, 3.14, True, None, {"k": "v"}, [1, 2]):
            instance = model(val=value)
            assert instance.val == value

    def test_str_int_float_bool_unchanged(self):
        assert parse_pydantic_type("str") is str
        assert parse_pydantic_type("int") is int
        assert parse_pydantic_type("float") is float
        assert parse_pydantic_type("bool") is bool

    def test_nested_list_type(self):
        assert parse_pydantic_type("list[str]") == list[str]

    def test_nested_dict_type(self):
        assert parse_pydantic_type("dict[str,int]") == dict[str, int]


class TestMakeAnthropicCompatibleSchema:
    """Verify the Anthropic schema patch is correct and idempotent.

    Two contracts:
    1. Patched schemas pass ``transform_schema`` (the validator inside
       ``langchain_anthropic.with_structured_output(method='json_schema')``).
    2. The patch is a no-op for schemas without ``$defs.JsonValue`` — so
       calling it from the structured-output boundary is always safe,
       regardless of which fields the Pydantic model carries.
    """

    def test_patched_list_schema_passes_anthropic_transform(self):
        """End-to-end: the dict that flows into Anthropic's
        ``with_structured_output`` after patching round-trips through
        ``transform_schema``. Without the patch this raises (see
        ``test_native_pydantic_schema_fails_anthropic_transform``)."""
        model = create_pydantic_model("Test", {"items": {"type": "list"}})
        patched = make_anthropic_compatible_schema(model)
        transform_schema(patched)  # should not raise

    def test_patched_any_schema_passes_anthropic_transform(self):
        model = create_pydantic_model("Test", {"val": {"type": "any"}})
        patched = make_anthropic_compatible_schema(model)
        transform_schema(patched)

    def test_patched_def_is_recursive_concrete_union(self):
        """The replacement ``$defs.JsonValue`` must be the recursive
        JSON-value union: primitives + object + array-of-JsonValue. The
        recursion preserves full ``JsonValue`` semantics — Anthropic
        models can emit arbitrarily nested structures."""
        model = create_pydantic_model("Test", {"items": {"type": "list"}})
        patched = make_anthropic_compatible_schema(model)
        json_value_def = patched["$defs"]["JsonValue"]
        assert "anyOf" in json_value_def
        types_present = {b.get("type") for b in json_value_def["anyOf"] if "type" in b}
        # six JSON primitives + object + array
        assert types_present == {"string", "number", "integer", "boolean", "null", "object", "array"}
        # the array branch refs back to JsonValue (recursive — the whole point)
        array_branch = next(b for b in json_value_def["anyOf"] if b.get("type") == "array")
        assert array_branch["items"] == {"$ref": "#/$defs/JsonValue"}

    def test_patch_does_not_mutate_input_model_schema(self):
        """Defensive: the patch must deep-copy. Mutating the cached
        ``model.model_json_schema()`` would leak the Anthropic-shape into
        the OpenAI path on subsequent invocations and break the very
        provider divergence the patch exists to enforce."""
        model = create_pydantic_model("Test", {"items": {"type": "list"}})
        original_def = model.model_json_schema()["$defs"]["JsonValue"]
        patched = make_anthropic_compatible_schema(model)
        patched["$defs"]["JsonValue"]["anyOf"].append({"type": "string"})
        assert model.model_json_schema()["$defs"]["JsonValue"] == original_def

    def test_patch_is_noop_when_no_json_value_def(self):
        """Schemas without ``$defs.JsonValue`` (e.g., a model with only
        primitive fields) pass through structurally unchanged. This
        keeps the structured-output boundary call site simple — the
        patch can be invoked unconditionally for any Anthropic call,
        without inspecting the schema for ``JsonValue`` first."""
        model = create_pydantic_model("Test", {"name": {"type": "str"}})
        patched = make_anthropic_compatible_schema(model)
        assert patched == model.model_json_schema()


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

    def test_patched_model_schema_accepted_by_anthropic(self):
        """The patched schema (what actually flows to Anthropic) must
        have no empty ``$defs`` entries — that's what
        ``transform_schema`` rejects. The unpatched schema *does* have
        the empty ``$defs.JsonValue`` (intentional — needed for OpenAI);
        ``make_anthropic_compatible_schema`` is the boundary that
        translates between the two provider expectations."""
        model = create_pydantic_model("LLMOutput", {
            "items": {"type": "list", "description": "Result items"},
        })
        patched = make_anthropic_compatible_schema(model)
        assert patched["properties"]["items"]["type"] == "array"
        defs = patched.get("$defs", {})
        for def_name, def_schema in defs.items():
            assert def_schema, f"Empty schema definition after patch: {def_name}"
