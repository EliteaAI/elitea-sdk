"""Unit + integration tests for OpenAPI toolkit query-param serialization.

Regression guard for the Planisware OData bug: query params were handed to
``requests`` as a dict, which encodes spaces as ``+`` (quote_plus) and ignores the
spec's ``allowReserved``. Servers that require ``%20`` (and reject ``+``) got HTTP 400.
The wrapper now serializes the query string itself: spaces -> %20, ``allowReserved``
honored, and ``requests`` is prevented from re-encoding it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from elitea_sdk.tools.openapi.api_wrapper import (
    _build_query_string,
    _get_query_param_specs,
    _serialize_query_param,
    build_wrapper,
)

FORM_RESERVED = {"style": "form", "explode": False, "allow_reserved": True}
FORM_STRICT = {"style": "form", "explode": False, "allow_reserved": False}


# --------------------------------------------------------------------------- #
# Unit: _serialize_query_param
# --------------------------------------------------------------------------- #

def test_space_becomes_percent20_not_plus():
    # Under allowReserved the '$' name prefix passes through raw (like Postman); space -> %20.
    assert _serialize_query_param("$filter", "duration eq 0", FORM_RESERVED) == "$filter=duration%20eq%200"


def test_allow_reserved_passes_odata_chars_unencoded():
    # Exact shape from the bug report: parens, commas, apostrophes survive; spaces are %20.
    out = _serialize_query_param("$filter", "contains(x,',B,')", FORM_RESERVED)
    assert out == "$filter=contains(x,',B,')"
    assert "+" not in out


def test_name_encoded_when_allow_reserved_false():
    # Without allowReserved the '$' name prefix is percent-encoded (unchanged from prior behavior).
    assert _serialize_query_param("$top", 500, FORM_STRICT) == "%24top=500"


def test_integer_value_unaffected():
    # $top under allowReserved: raw name, numeric value untouched.
    assert _serialize_query_param("$top", 500, FORM_RESERVED) == "$top=500"


def test_allow_reserved_false_encodes_reserved_chars():
    out = _serialize_query_param("status", "a eq b", FORM_STRICT)
    assert out == "status=a%20eq%20b"
    # Reserved chars ARE percent-encoded when allowReserved is absent/false.
    assert _serialize_query_param("f", "a(b)", FORM_STRICT) == "f=a%28b%29"


def test_literal_plus_is_escaped_even_with_allow_reserved():
    # '+' is deliberately not in the safe set, so "C++" cannot be misread as spaces.
    out = _serialize_query_param("q", "a+b", FORM_RESERVED)
    assert out == "q=a%2Bb"


def test_hash_is_escaped_even_with_allow_reserved():
    # '#' would start a URL fragment and truncate the query; always encoded.
    assert _serialize_query_param("$filter", "a#b", FORM_RESERVED) == "$filter=a%23b"


def test_navigation_property_slash_stays_raw():
    # OData navigation property: Address/City eq 'London' — '/' is reserved, stays raw.
    out = _serialize_query_param("$filter", "Address/City eq 'London'", FORM_RESERVED)
    assert out == "$filter=Address/City%20eq%20'London'"


def test_double_quote_encoded_single_quote_raw():
    # Single quote is a sub-delim (raw); double quote is not reserved (encoded).
    assert _serialize_query_param("$filter", "n eq 'a\"b'", FORM_RESERVED) == "$filter=n%20eq%20'a%22b'"


def test_none_value_contributes_nothing():
    assert _serialize_query_param("x", None, FORM_RESERVED) is None


def test_array_explode_true_repeats_key():
    spec = {"style": "form", "explode": True, "allow_reserved": False}
    assert _serialize_query_param("tag", ["a", "b"], spec) == "tag=a&tag=b"


def test_array_explode_false_comma_joins():
    assert _serialize_query_param("tag", ["a", "b"], FORM_STRICT) == "tag=a,b"


def test_array_space_and_pipe_delimited():
    space = {"style": "spaceDelimited", "explode": False, "allow_reserved": False}
    pipe = {"style": "pipeDelimited", "explode": False, "allow_reserved": False}
    assert _serialize_query_param("t", ["a", "b"], space) == "t=a%20b"
    assert _serialize_query_param("t", ["a", "b"], pipe) == "t=a|b"


# --------------------------------------------------------------------------- #
# Unit: _get_query_param_specs (defaults + merge/override)
# --------------------------------------------------------------------------- #

def test_specs_defaults_form_explode_true():
    op_raw = {"parameters": [{"name": "q", "in": "query", "schema": {"type": "string"}}]}
    specs = _get_query_param_specs(op_raw)
    assert specs["q"] == {"style": "form", "explode": True, "allow_reserved": False}


def test_specs_reads_style_explode_allow_reserved():
    op_raw = {
        "parameters": [
            {"name": "$filter", "in": "query", "style": "form", "explode": False, "allowReserved": True}
        ]
    }
    specs = _get_query_param_specs(op_raw)
    assert specs["$filter"] == {"style": "form", "explode": False, "allow_reserved": True}


def test_specs_operation_level_overrides_shared():
    shared = [{"name": "q", "in": "query", "allowReserved": False}]
    op_raw = {"parameters": [{"name": "q", "in": "query", "allowReserved": True}]}
    specs = _get_query_param_specs(op_raw, shared)
    assert specs["q"]["allow_reserved"] is True


def test_build_query_string_unknown_param_uses_defaults():
    # Param not in specs -> form/explode/no-reserved default; space still %20.
    assert _build_query_string({"x": "a b"}, {}) == "x=a%20b"


# --------------------------------------------------------------------------- #
# Integration: full _execute against a stubbed requests.Session
# --------------------------------------------------------------------------- #

def _planisware_spec() -> dict:
    return {
        "openapi": "3.0.3",
        "info": {"title": "t", "version": "1.0.0"},
        "servers": [{"url": "https://example.com/api"}],
        "paths": {
            "/ordo_project": {
                "get": {
                    "operationId": "get_portfolio",
                    "parameters": [
                        {
                            "name": "$filter",
                            "in": "query",
                            "style": "form",
                            "explode": False,
                            "allowReserved": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "$top",
                            "in": "query",
                            "schema": {"type": "integer"},
                        },
                    ],
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }


def _strict_spec() -> dict:
    spec = _planisware_spec()
    # Same operation but without allowReserved on $filter.
    spec["paths"]["/ordo_project"]["get"]["parameters"][0].pop("allowReserved")
    return spec


class _CapturingSession:
    """Minimal stand-in for requests.Session: records the request and returns a 200."""

    def __init__(self):
        self.headers: dict[str, str] = {}
        self.calls: list[dict[str, Any]] = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b'{"value": []}'
        resp.text = '{"value": []}'
        return resp


def _run(spec: dict, session: _CapturingSession, **params) -> None:
    wrapper = build_wrapper(spec)
    wrapper._client._requestor = session
    # Rebind operations to the stubbed requestor so op() uses it.
    wrapper._client._collect_operations()
    wrapper._execute("get_portfolio", **params)


def test_execute_encodes_spaces_as_percent20_and_keeps_reserved():
    session = _CapturingSession()
    _run(
        _planisware_spec(),
        session,
        **{"$filter": "duration eq 0 or contains(x,',B,')", "$top": 500},
    )
    url = session.calls[0]["url"]
    # No dict params leaked to requests (would re-encode with '+').
    assert not session.calls[0]["kwargs"].get("params")
    assert "%20" in url
    assert "+" not in url
    assert "contains(x,',B,')" in url
    assert "%24top=500" in url


def test_execute_strict_spec_still_percent20_no_plus():
    session = _CapturingSession()
    _run(_strict_spec(), session, **{"$filter": "duration eq 0", "$top": 500})
    url = session.calls[0]["url"]
    assert "+" not in url
    assert "duration%20eq%200" in url
    # allowReserved absent -> parens/commas would be encoded if present; spaces still %20.


def test_execute_restores_request_after_success():
    session = _CapturingSession()
    _run(_planisware_spec(), session, **{"$filter": "a eq b", "$top": 500})
    # The temporary wrapper must be gone; the restored method is the plain request().
    assert getattr(session.request, "__name__", "") != "_request_with_manual_query_encoding"
    # A second call still routes through the real request (no leak / no double-wrap).
    session.calls.clear()
    _run(_planisware_spec(), session, **{"$filter": "c eq d", "$top": 500})
    assert "+" not in session.calls[0]["url"]


def test_execute_restores_request_after_error():
    session = _CapturingSession()
    orig = session.request

    def boom(method, url, **kwargs):
        raise ConnectionError("network down")

    session.request = boom
    wrapper = build_wrapper(_planisware_spec())
    wrapper._client._requestor = session
    wrapper._client._collect_operations()
    with pytest.raises(Exception):
        wrapper._execute("get_portfolio", **{"$filter": "a eq b", "$top": 500})
    assert session.request is boom  # restored to what it was before the call
