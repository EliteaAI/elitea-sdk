from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.openapi.api_wrapper import build_wrapper
from elitea_sdk.tools.openapi.response_selection import (
    ResponseSelectionError,
    get_response_collection_paths,
    select_response_content,
)


def _selection(output: str) -> tuple[dict[str, Any], Any]:
    parsed = json.loads(output)
    return parsed["_elitea_response_selection"], parsed["data"]


def test_root_array_supports_positive_negative_and_phrase_search():
    source = json.dumps(
        [
            {"id": 1, "status": "active", "region": "North America"},
            {"id": 2, "status": "active archived", "region": "North America"},
            {"id": 3, "status": "active", "region": "Europe"},
        ]
    )

    metadata, data = _selection(
        select_response_content(source, response_search='active "north america" -archived')
    )

    assert data == [{"id": 1, "status": "active", "region": "North America"}]
    assert metadata == {
        "format": "json",
        "collection_path": "$",
        "collection_kind": "array",
        "total_items": 3,
        "matched_items": 1,
        "returned_items": 1,
        "truncated": False,
        "ranking": "bm25",
        "result_order": "relevance",
    }


def test_common_envelope_preserves_shape_and_source_order():
    source = json.dumps(
        {
            "count": 4,
            "items": [
                {"id": 3, "name": "match c"},
                {"id": 1, "name": "match a"},
                {"id": 2, "name": "miss"},
                {"id": 4, "name": "match d"},
            ],
            "next": "/items?page=2",
        }
    )

    metadata, data = _selection(
        select_response_content(source, response_search="match", response_limit=2)
    )

    assert data == {
        "count": 4,
        "items": [{"id": 3, "name": "match c"}, {"id": 1, "name": "match a"}],
        "next": "/items?page=2",
    }
    assert metadata["collection_path"] == "$.items"
    assert metadata["matched_items"] == 3
    assert metadata["returned_items"] == 2
    assert metadata["truncated"] is True


def test_response_limit_alone_previews_first_items():
    metadata, data = _selection(
        select_response_content(json.dumps({"results": [1, 2, 3]}), response_limit=2)
    )

    assert data == {"results": [1, 2]}
    assert metadata["matched_items"] == 3
    assert metadata["returned_items"] == 2
    assert metadata["ranking"] == "none"
    assert metadata["result_order"] == "source"


def test_bm25_ranks_relevant_items_instead_of_using_source_order():
    source = json.dumps(
        [
            {"id": "source-first", "text": "database " + "noise " * 100},
            {"id": "best", "text": "python database"},
            {"id": "python-only", "text": "python"},
        ]
    )

    metadata, data = _selection(
        select_response_content(source, response_search="python database", response_limit=3)
    )

    assert [item["id"] for item in data] == ["best", "python-only", "source-first"]
    assert metadata["matched_items"] == 3
    assert metadata["ranking"] == "bm25"
    assert metadata["result_order"] == "relevance"


def test_quoted_phrase_is_required_not_just_scored_as_individual_words():
    source = json.dumps(
        [
            {"id": 1, "text": "north team in america"},
            {"id": 2, "text": "north america team"},
        ]
    )

    _, data = _selection(select_response_content(source, response_search='"north america"'))

    assert data == [{"id": 2, "text": "north america team"}]


def test_search_uses_full_tokens_not_raw_substrings():
    source = json.dumps(
        [
            {"id": "substring", "text": "concatenate"},
            {"id": "token", "text": "cat"},
        ]
    )

    _, data = _selection(select_response_content(source, response_search="cat"))

    assert data == [{"id": "token", "text": "cat"}]


def test_search_uses_safe_default_limit():
    source = json.dumps([{"id": index, "value": "match"} for index in range(80)])

    metadata, data = _selection(select_response_content(source, response_search="match"))

    assert len(data) == 50
    assert metadata["matched_items"] == 80
    assert metadata["returned_items"] == 50
    assert metadata["truncated"] is True


def test_nested_unique_collection_is_discovered():
    source = json.dumps({"payload": {"records": [{"id": 1}, {"id": 2}]}, "request_id": "abc"})

    metadata, data = _selection(select_response_content(source, response_search="2"))

    assert metadata["collection_path"] == "$.payload.records"
    assert data["payload"]["records"] == [{"id": 2}]


def test_openapi_response_schema_path_wins_over_runtime_ambiguity():
    spec = {
        "components": {
            "schemas": {
                "Page": {
                    "type": "object",
                    "properties": {
                        "payload": {
                            "type": "object",
                            "properties": {"entries": {"type": "array", "items": {"type": "object"}}},
                        }
                    },
                }
            }
        }
    }
    operation = {
        "responses": {
            200: {
                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Page"}}}
            }
        }
    }
    paths = get_response_collection_paths(spec, operation)
    source = json.dumps(
        {
            "audit": [{"id": "wrong-1"}, {"id": "wrong-2"}],
            "payload": {"entries": [{"id": "right-1"}, {"id": "right-2"}]},
        }
    )

    metadata, data = _selection(
        select_response_content(source, response_limit=1, preferred_collection_paths=paths)
    )

    assert paths == [("payload", "entries")]
    assert metadata["collection_path"] == "$.payload.entries"
    assert data["payload"]["entries"] == [{"id": "right-1"}]
    assert len(data["audit"]) == 2


def test_runtime_map_inference_does_not_hide_schema_declared_descendant():
    preferred_paths = [("payload", "entries")]
    source = json.dumps(
        {
            "payload": {
                "status": "ok",
                "entries": [{"id": "right-1"}, {"id": "right-2"}],
            },
            "shadowA": {"status": "ok", "value": "wrong-a"},
            "shadowB": {"status": "ok", "value": "wrong-b"},
        }
    )

    metadata, data = _selection(
        select_response_content(
            source,
            response_search='"right 2"',
            preferred_collection_paths=preferred_paths,
        )
    )

    assert metadata["collection_path"] == "$.payload.entries"
    assert metadata["collection_kind"] == "array"
    assert data["payload"]["entries"] == [{"id": "right-2"}]


@pytest.mark.parametrize("composition_key", ["allOf", "anyOf", "oneOf"])
def test_multiple_composed_schema_paths_compete_by_query_relevance(composition_key: str):
    spec = {}
    operation = {
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "schema": {
                            composition_key: [
                                {
                                    "type": "object",
                                    "properties": {
                                        "declaredFirst": {
                                            "type": "array",
                                            "items": {"type": "object"},
                                        }
                                    },
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "moreRelevant": {
                                            "type": "array",
                                            "items": {"type": "object"},
                                        }
                                    },
                                },
                            ]
                        }
                    }
                }
            }
        }
    }
    paths = get_response_collection_paths(spec, operation)
    source = json.dumps(
        {
            "declaredFirst": [{"title": "critical routine update"}],
            "moreRelevant": [{"title": "critical database outage"}],
        }
    )

    metadata, data = _selection(
        select_response_content(
            source,
            response_search="critical database",
            preferred_collection_paths=paths,
        )
    )

    assert paths == [("declaredFirst",), ("moreRelevant",)]
    assert metadata["collection_path"] == "$.moreRelevant"
    assert data["moreRelevant"] == [{"title": "critical database outage"}]
    assert data["declaredFirst"] == [{"title": "critical routine update"}]


def test_schema_declared_keyed_object_is_searched_and_preserved_as_a_map():
    spec = {
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "status": {"type": "string"},
                    },
                }
            }
        }
    }
    operation = {
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "usersById": {
                                    "type": "object",
                                    "additionalProperties": {"$ref": "#/components/schemas/User"},
                                }
                            },
                        }
                    }
                }
            }
        }
    }
    paths = get_response_collection_paths(spec, operation)
    source = json.dumps(
        {
            "usersById": {
                "user-alpha": {"name": "Alice", "status": "active"},
                "user-beta": {"name": "Bob", "status": "active"},
            },
            "count": 2,
        }
    )

    metadata, data = _selection(
        select_response_content(
            source,
            response_search='"user beta"',
            preferred_collection_paths=paths,
        )
    )

    assert paths == [("usersById",)]
    assert metadata["collection_path"] == "$.usersById"
    assert metadata["collection_kind"] == "object_map"
    assert metadata["total_items"] == 2
    assert data == {
        "usersById": {"user-beta": {"name": "Bob", "status": "active"}},
        "count": 2,
    }


def test_homogeneous_runtime_keyed_object_is_discovered_without_schema_hint():
    source = json.dumps(
        {
            "recordsById": {
                "record-1": {"title": "routine update", "status": "active"},
                "record-2": {"title": "critical database outage", "status": "active"},
                "record-3": {"title": "archived notice", "status": "archived"},
            },
            "request": {"id": "request-1"},
        }
    )

    metadata, data = _selection(
        select_response_content(source, response_search="critical database")
    )

    assert metadata["collection_path"] == "$.recordsById"
    assert metadata["collection_kind"] == "object_map"
    assert data["recordsById"] == {
        "record-2": {"title": "critical database outage", "status": "active"}
    }


def test_query_relevance_selects_the_matching_json_subcorpus():
    source = json.dumps(
        {
            "users": [{"name": "alpha"}, {"name": "beta"}],
            "tickets": [{"title": "critical database outage"}, {"title": "routine update"}],
        }
    )

    metadata, data = _selection(
        select_response_content(source, response_search="critical database", response_limit=1)
    )

    assert metadata["collection_path"] == "$.tickets"
    assert data["tickets"] == [{"title": "critical database outage"}]
    assert len(data["users"]) == 2


def test_ambiguous_collections_return_compact_guidance_not_original_data():
    source = json.dumps(
        {
            "left": [{"payload": "x"}, {"payload": "x"}],
            "right": [{"payload": "x"}, {"payload": "x"}],
        }
    )

    metadata, data = _selection(select_response_content(source, response_search="x"))

    assert metadata["status"] == "ambiguous_collection"
    assert metadata["candidate_paths"] == ["$.left", "$.right"]
    assert data is None
    assert len(json.dumps({"metadata": metadata, "data": data})) < len(source) + 500


def test_non_json_response_uses_line_based_matching():
    source = "alpha active\nbeta archived active\ngamma active\ndelta"

    metadata, data = _selection(
        select_response_content(source, response_search="active -archived", response_limit=1)
    )

    assert metadata["format"] == "text"
    assert metadata["collection_path"] == "$segments"
    assert metadata["total_items"] == 4
    assert metadata["matched_items"] == 2
    assert metadata["truncated"] is True
    assert data == "alpha active"


def test_non_json_paragraphs_are_ranked_with_bm25():
    source = "python\n\npython database\n\nunrelated paragraph"

    metadata, data = _selection(
        select_response_content(source, response_search="python database", response_limit=1)
    )

    assert data == "python database"
    assert metadata["format"] == "text"
    assert metadata["ranking"] == "bm25"
    assert metadata["result_order"] == "relevance"


def test_zero_matches_returns_valid_json_with_empty_collection():
    metadata, data = _selection(
        select_response_content(json.dumps({"items": [{"name": "alpha"}]}), response_search="missing")
    )

    assert metadata["matched_items"] == 0
    assert metadata["returned_items"] == 0
    assert data == {"items": []}


def test_size_bound_reduces_returned_items_without_slicing_json():
    source = json.dumps({"items": [{"id": index, "text": "x" * 120} for index in range(5)]})

    output = select_response_content(source, response_limit=5, max_serialized_chars=500)
    metadata, data = _selection(output)

    assert len(output) <= 500
    assert 0 < metadata["returned_items"] < 5
    assert metadata["truncated"] is True
    assert len(data["items"]) == metadata["returned_items"]


def test_oversized_single_item_returns_structured_error():
    source = json.dumps({"items": [{"text": "x" * 2_000}]})

    output = select_response_content(source, response_limit=1, max_serialized_chars=450)
    metadata, data = _selection(output)

    assert len(output) <= 450
    assert metadata["status"] == "content_too_large"
    assert metadata["returned_items"] == 0
    assert data is None


@pytest.mark.parametrize("search", ['"unterminated', "-"])
def test_invalid_search_expression_is_rejected(search):
    with pytest.raises(ResponseSelectionError):
        select_response_content("[]", response_search=search)


OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Selection test", "version": "1.0.0"},
    "servers": [{"url": "https://example.com"}],
    "paths": {
        "/items": {
            "get": {
                "operationId": "list_items",
                "responses": {
                    "200": {
                        "description": "ok",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "items": {"type": "array", "items": {"type": "object"}}
                                    },
                                }
                            }
                        },
                    }
                },
            }
        }
    },
}


class _ResponseSession:
    def __init__(self, content: bytes):
        self.headers: dict[str, str] = {}
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})

        class Response:
            status_code = 200
            content = self.content

        return Response()


def _wrapper_with_response(content: bytes):
    wrapper = build_wrapper(OPENAPI_SPEC)
    session = _ResponseSession(content)
    wrapper._client._requestor = session
    wrapper._client._collect_operations()
    return wrapper, session


def test_generated_schema_exposes_only_two_optional_selection_fields():
    wrapper = build_wrapper(OPENAPI_SPEC)
    schema = wrapper.get_available_tools()[0]["args_schema"].model_json_schema()

    assert schema["properties"]["response_search"]["default"] is None
    assert schema["properties"]["response_limit"]["default"] is None
    integer_schema = next(
        option
        for option in schema["properties"]["response_limit"]["anyOf"]
        if option.get("type") == "integer"
    )
    assert integer_schema["minimum"] == 1
    assert integer_schema["maximum"] == 200
    assert not {
        "response_filter_enabled",
        "response_filter_type",
        "response_filter_pattern",
        "response_filter_scope",
        "response_filter_mode",
    }.intersection(schema["properties"])


def test_execute_without_selection_preserves_exact_legacy_output():
    raw = b'{ "items": [ {"id": 1} ] }\n'
    wrapper, _ = _wrapper_with_response(raw)

    assert wrapper._execute("list_items") == raw.decode()


def test_execute_applies_selection_without_forwarding_sdk_arguments():
    wrapper, session = _wrapper_with_response(b'{"items":[{"id":1},{"id":2}]}')

    metadata, data = _selection(
        wrapper._execute("list_items", response_search="2", response_limit=1)
    )

    assert data == {"items": [{"id": 2}]}
    assert metadata["collection_path"] == "$.items"
    assert "response_search" not in session.calls[0]["kwargs"]
    assert "response_limit" not in session.calls[0]["kwargs"]


def test_legacy_regexp_behavior_is_preserved_when_used_alone():
    wrapper, _ = _wrapper_with_response(b'{"items":[{"secret":"remove-me"}]}')

    assert wrapper._execute("list_items", regexp="remove-me") == '{"items":[{"secret":""}]}'


def test_regexp_cannot_be_combined_with_structured_selection():
    wrapper, session = _wrapper_with_response(b'{"items":[]}')

    with pytest.raises(ToolException, match="cannot be combined"):
        wrapper._execute("list_items", regexp="x", response_search="x")

    assert session.calls == []
