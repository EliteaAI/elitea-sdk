"""Tests for the shared tool-result serialization contract (#6532)."""

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID

from pydantic import BaseModel

from elitea_sdk.tools.utils.serialization import serialize_tool_result


class Status(Enum):
    OPEN = "opened"


class Issue(BaseModel):
    number: int
    title: str


class Measurement(BaseModel):
    label: str
    value: float


@dataclass
class Row:
    name: str
    value: float


class CustomRendering:
    """Stands in for toolkit objects whose __str__ is their documented rendering."""

    def __str__(self):
        return "item one\n-----\nitem two"


@dataclass
class Comment:
    author: str
    body: str


class Unprintable:
    def __str__(self):
        raise RuntimeError("no string for you")


class TestPassThrough:
    def test_string_is_returned_unchanged(self):
        assert serialize_tool_result("Extracted 2 pages") == "Extracted 2 pages"

    def test_empty_string_stays_empty(self):
        assert serialize_tool_result("") == ""

    def test_non_collection_keeps_str_rendering(self):
        assert serialize_tool_result(CustomRendering()) == "item one\n-----\nitem two"

    def test_scalar_keeps_str_rendering(self):
        assert serialize_tool_result(42) == "42"


class TestJsonSerialization:
    def test_list_of_dicts_is_json(self):
        result = serialize_tool_result([{"number": 7, "author": None, "open": True}])

        assert json.loads(result) == [{"number": 7, "author": None, "open": True}]
        assert "'number'" not in result

    def test_no_indentation_is_added(self):
        assert serialize_tool_result([{"a": 1}, {"b": 2}]) == '[{"a": 1}, {"b": 2}]'

    def test_tuple_and_set_become_arrays(self):
        assert json.loads(serialize_tool_result(("a", "b"))) == ["a", "b"]
        assert json.loads(serialize_tool_result({"a"})) == ["a"]

    def test_non_ascii_stays_literal(self):
        result = serialize_tool_result({"title": "привет"})

        assert "привет" in result
        assert "\\u" not in result


class TestNonJsonNativeValues:
    def test_datetime_becomes_isoformat(self):
        result = serialize_tool_result({"created_at": datetime(2026, 1, 2, 3, 4, 5)})

        assert json.loads(result) == {"created_at": "2026-01-02T03:04:05"}

    def test_uuid_becomes_string(self):
        identifier = UUID("12345678-1234-5678-1234-567812345678")

        assert json.loads(serialize_tool_result({"id": identifier}))["id"] == str(identifier)

    def test_decimal_keeps_exact_value(self):
        assert json.loads(serialize_tool_result({"amount": Decimal("1.10")}))["amount"] == "1.10"

    def test_enum_becomes_value(self):
        assert json.loads(serialize_tool_result({"state": Status.OPEN}))["state"] == "opened"

    def test_pydantic_model_becomes_mapping(self):
        result = serialize_tool_result({"issue": Issue(number=7, title="Bug")})

        assert json.loads(result) == {"issue": {"number": 7, "title": "Bug"}}

    def test_dataclass_nested_in_a_collection_becomes_a_mapping(self):
        result = serialize_tool_result([{"comment": Comment(author="alice", body="looks good")}])

        assert json.loads(result) == [{"comment": {"author": "alice", "body": "looks good"}}]

    def test_utf8_bytes_are_decoded(self):
        assert json.loads(serialize_tool_result({"body": b"hello"}))["body"] == "hello"

    def test_binary_bytes_are_summarized_not_dumped(self):
        result = serialize_tool_result({"image": b"\x89PNG\x00\x01"})

        assert json.loads(result) == {"image": "<6 bytes>"}


class TestValuesJsonDumpsMishandles:
    """These fail at the ENCODER, so `default=` is never consulted for them."""

    def test_nan_becomes_null_and_output_stays_parseable(self):
        result = serialize_tool_result([{"amount": float("nan"), "count": 2}])

        assert json.loads(result) == [{"amount": None, "count": 2}]

    def test_infinities_become_null(self):
        result = serialize_tool_result([{"hi": float("inf"), "lo": float("-inf")}])

        assert json.loads(result) == [{"hi": None, "lo": None}]

    def test_a_string_mentioning_nan_is_untouched(self):
        result = serialize_tool_result([{"note": "value is NaN here"}])

        assert json.loads(result) == [{"note": "value is NaN here"}]

    def test_tuple_keys_do_not_collapse_the_whole_payload(self):
        # A tuple key makes json.dumps raise before `default` runs; the payload
        # used to degrade to a Python repr in its entirety.
        result = serialize_tool_result({("a", "b"): 1, "plain": 2})

        assert json.loads(result) == {"('a', 'b')": 1, "plain": 2}

    def test_non_finite_inside_a_dataclass_is_repaired(self):
        # The dataclass is expanded by `default=` inside json.dumps, i.e. after a
        # collection-only walk has finished, so its NaN used to survive.
        result = serialize_tool_result({"rows": [Row(name="a", value=float("nan"))]})

        assert json.loads(result) == {"rows": [{"name": "a", "value": None}]}

    def test_non_finite_inside_a_pydantic_model_is_repaired(self):
        result = serialize_tool_result({"m": Measurement(label="speed", value=float("inf"))})

        assert json.loads(result) == {"m": {"label": "speed", "value": None}}

    def test_default_object_repr_loses_its_memory_address(self):
        class StreamingBody:
            pass

        result = serialize_tool_result([{"Body": StreamingBody()}])

        assert json.loads(result) == [{"Body": "<StreamingBody>"}]

    def test_an_object_with_its_own_str_keeps_it(self):
        result = serialize_tool_result([{"x": CustomRendering()}])

        assert json.loads(result) == [{"x": "item one\n-----\nitem two"}]


class TestBareValues:
    """A value returned on its own gets the same treatment as a nested one."""

    def test_bare_dataclass(self):
        assert json.loads(serialize_tool_result(Row(name="a", value=1.5))) == {"name": "a", "value": 1.5}

    def test_bare_pydantic_model(self):
        assert json.loads(serialize_tool_result(Issue(number=7, title="Bug"))) == {"number": 7, "title": "Bug"}

    def test_bare_binary_is_summarized_not_mojibake(self):
        assert serialize_tool_result(b"\xff\xfe binary") == "<9 bytes>"

    def test_bare_non_finite_float(self):
        assert serialize_tool_result(float("nan")) == "null"
        assert serialize_tool_result(float("inf")) == "null"

    def test_bare_object_keeps_its_own_rendering(self):
        assert serialize_tool_result(CustomRendering()) == "item one\n-----\nitem two"


class TestNeverRaises:
    def test_circular_reference_falls_back_to_repr(self):
        payload = {"name": "root"}
        payload["self"] = payload

        result = serialize_tool_result(payload)

        assert isinstance(result, str)
        assert "root" in result

    def test_unprintable_value_still_returns_text(self):
        result = serialize_tool_result({"broken": Unprintable()})

        assert isinstance(result, str)
