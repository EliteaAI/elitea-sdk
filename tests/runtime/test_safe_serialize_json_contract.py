"""safe_serialize keeps its always-JSON contract (#6532).

function.py feeds the result into the Pyodide preamble, where the sandbox runs
json.loads on it, so a scalar has to render as a JSON literal rather than the way
str() spells it.
"""
import json

import pytest

from elitea_sdk.runtime.langchain.utils import safe_serialize


@pytest.mark.parametrize("value, expected", [
    (None, None),
    (True, True),
    (False, False),
    (5, 5),
    (1.5, 1.5),
    ("plain", "plain"),
    ({"a": 1}, {"a": 1}),
    ([{"a": 1}], [{"a": 1}]),
])
def test_output_always_parses_as_json(value, expected):
    assert json.loads(safe_serialize(value)) == expected


def test_non_finite_float_becomes_null():
    assert json.loads(safe_serialize(float("nan"))) is None


def test_exotic_values_are_json_not_repr():
    from datetime import datetime
    from decimal import Decimal

    parsed = json.loads(safe_serialize([{"created": datetime(2026, 1, 2), "amount": Decimal("1.10")}]))

    assert parsed == [{"created": "2026-01-02T00:00:00", "amount": "1.10"}]


def test_bare_datetime_and_decimal_are_json():
    from datetime import datetime
    from decimal import Decimal

    # Unwrapped, these used to render as text json.loads rejects, and Decimal lost
    # the precision the nested path deliberately preserves.
    assert json.loads(safe_serialize(datetime(2026, 1, 2))) == "2026-01-02T00:00:00"
    assert json.loads(safe_serialize(Decimal("1.10"))) == "1.10"
