"""Unit tests for ``elitea_sdk.tools.aha.api_wrapper``.

Covers transport plumbing (REST + GraphQL), pagination, reference-number
validation, output shaping, and error surfacing. HTTP is stubbed at the
``requests.Session`` level — no network calls.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException
from pydantic import SecretStr

from elitea_sdk.tools.aha import AhaToolkit, get_tools as get_aha_tools
from elitea_sdk.tools.aha.api_wrapper import (
    AhaAddCommentInput,
    AhaApiWrapper,
    AhaAttachFileInput,
    AhaCreateRecordLinkInput,
    AhaCreateRecordInput,
    AhaDeleteRecordInput,
    AhaListCommentsInput,
    AhaListRequirementsInput,
    AhaManageRecordInput,
    AhaUpdateRecordInput,
    _FEATURE_REF_RE,
    _PAGE_REF_RE,
    _QUERY_GET_PAGE,
    _REQUIREMENT_REF_RE,
)


def _wrapper(
    base_url: str = "https://example.aha.io",
    **kwargs,
) -> AhaApiWrapper:
    return AhaApiWrapper(
        base_url=base_url,
        api_key=SecretStr("token"),
        **kwargs,
    )


def _rest_stub(payload: Dict[str, Any], *, ok: bool = True, status: int = 200, text: str = ""):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status
    body = json.dumps(payload).encode() if payload else b""
    resp.content = body
    resp.text = text or json.dumps(payload) if payload else text
    resp.json = lambda payload=payload: payload
    return resp


class TestValidation:
    def test_missing_base_url(self):
        with pytest.raises(ToolException, match="base_url is required"):
            AhaApiWrapper(base_url="", api_key=SecretStr("t"))

    def test_bad_scheme(self):
        with pytest.raises(ToolException, match="must start with http"):
            AhaApiWrapper(base_url="ftp://x", api_key=SecretStr("t"))

    def test_missing_api_key(self):
        with pytest.raises(ToolException, match="api_key is required"):
            AhaApiWrapper(base_url="https://example.aha.io", api_key="")

    def test_trailing_slash_stripped(self):
        w = _wrapper("https://example.aha.io/")
        assert w.base_url == "https://example.aha.io"
        assert w._rest_url == "https://example.aha.io/api/v1"
        assert w._graphql_url == "https://example.aha.io/api/v2/graphql"

    def test_session_auth_header(self):
        w = _wrapper()
        assert w._session.headers["Authorization"] == "Bearer token"
        assert w._session.headers["Content-Type"] == "application/json"


class TestRest:
    def test_get_feature_url_and_projection(self):
        w = _wrapper()
        payload = {"feature": {"id": 1, "reference_num": "DEVELOP-1", "name": "X", "extra": "e"}}
        resp = _rest_stub(payload)

        with patch.object(w._session, "request", return_value=resp) as req:
            result = w.get_feature("DEVELOP-1", fields=["id", "name"])

        method, url = req.call_args[0][:2]
        assert method == "GET"
        assert url == "https://example.aha.io/api/v1/features/DEVELOP-1"
        assert result == {"id": 1, "name": "X"}

    def test_rest_http_error_surfaces_status_and_body(self):
        w = _wrapper()
        resp = MagicMock()
        resp.ok = False
        resp.status_code = 404
        resp.text = '{"error":"not found"}'
        resp.content = resp.text.encode()

        with patch.object(w._session, "request", return_value=resp):
            with pytest.raises(ToolException, match=r"failed \(404\).*not found"):
                w.get_feature("DEVELOP-999")

    def test_rest_network_error(self):
        w = _wrapper()
        import requests

        with patch.object(
            w._session, "request", side_effect=requests.exceptions.ConnectionError("boom")
        ):
            with pytest.raises(ToolException, match="network error"):
                w.get_feature("DEVELOP-1")

    def test_none_params_filtered(self):
        w = _wrapper()
        resp = _rest_stub({"products": [], "pagination": {"current_page": 1, "total_pages": 1}})

        with patch.object(w._session, "request", return_value=resp) as req:
            w.list_products(updated_since=None, per_page=10, max_records=5)

        params = req.call_args.kwargs["params"]
        assert "updated_since" not in params  # None filtered out
        assert params["per_page"] == 10
        assert params["page"] == 1

    def test_nested_list_features_by_release(self):
        w = _wrapper()
        resp = _rest_stub({"features": [], "pagination": {"current_page": 1, "total_pages": 1}})

        with patch.object(w._session, "request", return_value=resp) as req:
            w.list_features(release_id="REL-1")

        url = req.call_args[0][1]
        assert url.endswith("/releases/REL-1/features")

    def test_nested_list_features_by_product(self):
        w = _wrapper()
        resp = _rest_stub({"features": [], "pagination": {"current_page": 1, "total_pages": 1}})

        with patch.object(w._session, "request", return_value=resp) as req:
            w.list_features(product_id="PROD-1")

        url = req.call_args[0][1]
        assert url.endswith("/products/PROD-1/features")

    def test_release_id_takes_precedence_over_product(self):
        w = _wrapper()
        resp = _rest_stub({"features": [], "pagination": {"current_page": 1, "total_pages": 1}})

        with patch.object(w._session, "request", return_value=resp) as req:
            w.list_features(product_id="P", release_id="R")

        url = req.call_args[0][1]
        assert "releases/R/features" in url
        assert "products/P" not in url

    def test_list_requirements_schema_requires_feature_id(self):
        assert AhaListRequirementsInput.model_fields["feature_id"].is_required()

    @pytest.mark.parametrize("feature_id", [None, "", "   "])
    def test_list_requirements_rejects_missing_feature_id_before_request(
        self, feature_id
    ):
        w = _wrapper()

        with patch.object(w._session, "request") as req:
            with pytest.raises(
                ToolException,
                match="list_requirements: feature_id is required",
            ):
                w.list_requirements(feature_id=feature_id)

        req.assert_not_called()


class TestPagination:
    def test_walks_pages_until_total(self):
        w = _wrapper()
        pages = [
            {"features": [{"id": 1}, {"id": 2}], "pagination": {"current_page": 1, "total_pages": 3}},
            {"features": [{"id": 3}, {"id": 4}], "pagination": {"current_page": 2, "total_pages": 3}},
            {"features": [{"id": 5}], "pagination": {"current_page": 3, "total_pages": 3}},
        ]

        def side_effect(method, url, params=None, **_):
            return _rest_stub(pages[params["page"] - 1])

        with patch.object(w._session, "request", side_effect=side_effect):
            out = w.list_features(max_records=100)
        assert [r["id"] for r in out] == [1, 2, 3, 4, 5]

    def test_max_records_stops_early(self):
        w = _wrapper()
        pages = [
            {"features": [{"id": 1}, {"id": 2}], "pagination": {"current_page": 1, "total_pages": 3}},
            {"features": [{"id": 3}, {"id": 4}], "pagination": {"current_page": 2, "total_pages": 3}},
            {"features": [{"id": 5}], "pagination": {"current_page": 3, "total_pages": 3}},
        ]
        seen: List[int] = []

        def side_effect(method, url, params=None, **_):
            seen.append(params["page"])
            return _rest_stub(pages[params["page"] - 1])

        with patch.object(w._session, "request", side_effect=side_effect):
            out = w.list_features(max_records=3)
        assert len(out) == 3
        # Should not have fetched page 3 (would exceed max)
        assert seen == [1, 2]

    def test_non_paginated_payload_yields_once(self):
        w = _wrapper()
        # search endpoint might return a single dict wrapped body — we still
        # want the paginator to short-circuit rather than loop forever.
        resp = _rest_stub({"summary": {"total": 0}})
        with patch.object(w._session, "request", return_value=resp) as req:
            out = list(w._paginate("weird"))
        assert out == [{"summary": {"total": 0}}]
        assert req.call_count == 1


class TestGraphQL:
    def test_body_shape_and_variables(self):
        w = _wrapper()
        resp = _rest_stub({"data": {"page": {"id": "1"}}})

        with patch.object(w._session, "post", return_value=resp) as post:
            w.get_page("ABC-N-1", include_parent=True)

        assert post.call_args[0][0] == "https://example.aha.io/api/v2/graphql"
        body = post.call_args.kwargs["json"]
        assert "query" in body
        assert "GetPage" in body["query"]
        assert body["variables"] == {"id": "ABC-N-1", "includeParent": True}

    def test_errors_field_raises(self):
        w = _wrapper()
        resp = _rest_stub({"data": None, "errors": [{"message": "boom"}]})

        with patch.object(w._session, "post", return_value=resp):
            with pytest.raises(ToolException, match="GraphQL errors"):
                w.get_page("ABC-N-1")

    def test_http_error_surfaces(self):
        w = _wrapper()
        resp = MagicMock()
        resp.ok = False
        resp.status_code = 401
        resp.text = "Unauthorized"
        with patch.object(w._session, "post", return_value=resp):
            with pytest.raises(ToolException, match=r"GraphQL failed \(401\)"):
                w.get_page("ABC-N-1")

    def test_search_documents_returns_nodes(self):
        w = _wrapper()
        resp = _rest_stub(
            {
                "data": {
                    "searchDocuments": {
                        "nodes": [
                            {"name": "Doc", "url": "u", "searchableId": "1", "searchableType": "Page"}
                        ]
                    }
                }
            }
        )
        with patch.object(w._session, "post", return_value=resp) as post:
            out = w.search_documents("hello", searchable_type=None)
        assert out == [
            {"name": "Doc", "url": "u", "searchableId": "1", "searchableType": "Page"}
        ]
        # Default searchable_type is "Page" when None passed; Aha GraphQL
        # expects a list ([String!]), not a scalar.
        assert post.call_args.kwargs["json"]["variables"]["searchableType"] == ["Page"]

    def test_search_documents_empty_query_rejected(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="query is required"):
            w.search_documents("")


class TestReferenceValidation:
    @pytest.mark.parametrize(
        "ref",
        ["DEVELOP-1", "ABC-42", "A1-123"],
    )
    def test_valid_feature_refs(self, ref):
        assert _FEATURE_REF_RE.match(ref)

    @pytest.mark.parametrize(
        "ref",
        ["develop-1", "1DEV-1", "DEVELOP", "-1", "DEVELOP-N-1"],
    )
    def test_invalid_feature_refs(self, ref):
        assert not _FEATURE_REF_RE.match(ref)

    def test_requirement_ref_needs_two_numeric_segments(self):
        assert _REQUIREMENT_REF_RE.match("ADT-123-1")
        assert not _REQUIREMENT_REF_RE.match("ADT-123")

    def test_page_ref_needs_N_segment(self):
        assert _PAGE_REF_RE.match("ABC-N-213")
        assert not _PAGE_REF_RE.match("ABC-213")

    def test_bad_reference_rejected_before_http(self):
        w = _wrapper()
        with patch.object(w._session, "post") as post:
            with pytest.raises(ToolException, match="not a valid Aha! feature"):
                w.get_feature_gql("bad-ref")
        post.assert_not_called()


class TestOutputFormat:
    def test_json_passthrough(self):
        w = _wrapper()
        assert w._format_output([{"a": 1}], "json") == [{"a": 1}]

    def test_unknown_format_raises(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="Unsupported output_format"):
            w._format_output([{"a": 1}], "xml")

    def test_csv_output(self):
        w = _wrapper()
        out = w._format_output([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], "csv")
        assert "a,b" in out
        assert "1,x" in out and "2,y" in out

    def test_markdown_output(self):
        w = _wrapper()
        out = w._format_output([{"a": 1}, {"a": 2}], "markdown")
        # tabulate rendering uses pipes
        assert "|" in out
        assert "1" in out and "2" in out

    def test_csv_falls_back_to_data_on_non_list(self):
        w = _wrapper()
        data = {"single": "record"}
        assert w._format_output(data, "csv") == data

    def test_projection_only_keeps_allowlist(self):
        w = _wrapper()
        records = [{"id": 1, "name": "A", "extra": "x"}, {"id": 2, "name": "B"}]
        projected = w._project_records(records, ["id", "name"])
        assert projected == [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]


class TestEmptyResults:
    @pytest.mark.parametrize(
        ("method_name", "kwargs", "expected"),
        [
            ("list_products", {}, "Aha! API returned no products."),
            (
                "list_products",
                {"updated_since": "2026-07-01T00:00:00Z"},
                "Aha! API returned no products updated since "
                "'2026-07-01T00:00:00Z'.",
            ),
            ("list_features", {}, "Aha! API returned no features."),
            (
                "list_features",
                {
                    "product_id": "EL",
                    "release_id": "EL-R-1",
                    "q": "missing feature",
                    "updated_since": "2026-07-01T00:00:00Z",
                },
                "Aha! API returned no features for release 'EL-R-1' "
                "matching query 'missing feature' updated since "
                "'2026-07-01T00:00:00Z'.",
            ),
            (
                "list_features",
                {"product_id": "EL"},
                "Aha! API returned no features for product 'EL'.",
            ),
            (
                "list_requirements",
                {"feature_id": "EL-1"},
                "Aha! API returned no requirements for feature 'EL-1'.",
            ),
            (
                "list_requirements",
                {"feature_id": "EL-1", "q": "missing requirement"},
                "Aha! API returned no requirements for feature 'EL-1' "
                "matching query 'missing requirement'.",
            ),
            ("list_releases", {}, "Aha! API returned no releases."),
            (
                "list_releases",
                {"product_id": "EL", "parking_lot": False},
                "Aha! API returned no releases for product 'EL' "
                "with parking_lot=false.",
            ),
            (
                "list_initiatives",
                {"product_id": "EL"},
                "Aha! API returned no initiatives for product 'EL'.",
            ),
            (
                "list_epics",
                {"release_id": "EL-R-1"},
                "Aha! API returned no epics for release 'EL-R-1'.",
            ),
            (
                "list_epics",
                {"product_id": "EL"},
                "Aha! API returned no epics for product 'EL'.",
            ),
        ],
    )
    def test_list_methods_return_detailed_message(
        self, method_name, kwargs, expected
    ):
        w = _wrapper()

        with patch.object(w, "_collect", return_value=[]):
            out = getattr(w, method_name)(**kwargs)

        assert out == expected

    @pytest.mark.parametrize(
        ("record_type", "expected"),
        [
            (None, "Aha! API returned no records for query 'missing record'."),
            (
                "feature",
                "Aha! API returned no feature records for query 'missing record'.",
            ),
        ],
    )
    def test_search_returns_detailed_message(self, record_type, expected):
        w = _wrapper()
        resp = _rest_stub(
            {"records": [], "pagination": {"current_page": 1, "total_pages": 1}}
        )

        with patch.object(w._session, "request", return_value=resp):
            out = w.search("missing record", type=record_type)

        assert out == expected

    @pytest.mark.parametrize(
        ("product_id", "query", "expected"),
        [
            (None, None, "Aha! API returned no ideas."),
            (
                "EL",
                None,
                "Aha! API returned no ideas for product 'EL'.",
            ),
            (
                None,
                "missing idea",
                "Aha! API returned no ideas matching query 'missing idea'.",
            ),
            (
                "EL",
                "missing idea",
                "Aha! API returned no ideas for product 'EL' matching query 'missing idea'.",
            ),
        ],
    )
    def test_list_ideas_returns_detailed_message(
        self, product_id, query, expected
    ):
        w = _wrapper()
        resp = _rest_stub(
            {"ideas": [], "pagination": {"current_page": 1, "total_pages": 1}}
        )

        with patch.object(w._session, "request", return_value=resp):
            out = w.list_ideas(product_id=product_id, q=query)

        assert out == expected


class TestToolRegistry:
    def test_registry_exposes_all_tools(self):
        w = _wrapper()
        tools = w.get_available_tools()
        names = {t["name"] for t in tools}
        # 19 M2 read tools + 14 M3 write/dispatcher tools = 33
        assert len(tools) == 33
        # Spot-check every category
        assert {"get_feature", "list_features", "search", "get_page", "get_feature_gql"} <= names
        assert {"find_project", "search_records", "read_records"} <= names
        assert {
            "add_comment",
            "list_comments",
            "manage_record",
            "create_record",
            "update_record",
            "delete_record",
            "create_record_link",
            "copy_record",
            "fields_metadata",
            "field_options_metadata",
            "attach_file",
        } <= names

    def test_each_tool_has_required_shape(self):
        w = _wrapper()
        for t in w.get_available_tools():
            assert "name" in t and "description" in t and "args_schema" in t and "ref" in t
            assert callable(t["ref"])

    def test_action_specific_tool_can_be_enabled_independently(self):
        toolkit = AhaToolkit.get_toolkit(
            selected_tools=["delete_record"],
            aha_configuration={
                "base_url": "https://example.aha.io",
                "api_key": SecretStr("token"),
            },
        )

        assert [tool.name for tool in toolkit.get_tools()] == ["delete_record"]

    def test_runtime_elitea_client_is_injected_into_aha_wrapper(self):
        elitea = MagicMock()
        tools = get_aha_tools(
            {
                "settings": {
                    "selected_tools": ["attach_file"],
                    "aha_configuration": {
                        "base_url": "https://example.aha.io",
                        "api_key": SecretStr("token"),
                    },
                    "elitea": elitea,
                },
                "toolkit_name": "Aha",
            }
        )

        assert len(tools) == 1
        assert tools[0].api_wrapper.elitea is elitea


class TestComments:
    def test_schemas_expose_canonical_resource_type_dropdown(self):
        expected = [
            "feature",
            "requirement",
            "idea",
            "release",
            "release_phase",
            "epic",
            "initiative",
            "goal",
            "page",
            "to_do",
        ]

        for schema in (AhaAddCommentInput, AhaListCommentsInput):
            resource_type = schema.model_json_schema()["properties"][
                "resource_type"
            ]
            assert resource_type["enum"] == expected
            assert "to_do" in resource_type["description"]
            assert "todo" not in resource_type["enum"]
            assert "to-do" not in resource_type["enum"]

    def test_add_comment_posts_to_correct_url(self):
        w = _wrapper()
        resp = _rest_stub({"comment": {"id": 1, "body": "hi"}})

        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.add_comment("feature", "DEVELOP-1", "hi")

        method, url = req.call_args[0][:2]
        assert method == "POST"
        assert url.endswith("/features/DEVELOP-1/comments")
        assert req.call_args.kwargs["json"] == {"comment": {"body": "hi"}}
        assert out == {"id": 1, "body": "hi"}

    @pytest.mark.parametrize(
        "resource_type",
        ["to_do", "todo", "to-do", "to-dos", "to_dos", "task", "tasks"],
    )
    def test_add_comment_routes_todo_aliases_to_tasks(self, resource_type):
        w = _wrapper()
        resp = _rest_stub({"comment": {"id": 2}})

        with patch.object(w._session, "request", return_value=resp) as req:
            w.add_comment(resource_type, "EL-TODO-3", "note")

        assert req.call_args[0][1].endswith("/tasks/EL-TODO-3/comments")

    def test_add_comment_rejects_empty_body(self):
        w = _wrapper()
        with patch.object(w._session, "request") as req:
            with pytest.raises(ToolException, match="body is required"):
                w.add_comment("feature", "DEVELOP-1", "  ")
        req.assert_not_called()

    def test_add_comment_rejects_unsupported_resource(self):
        w = _wrapper()
        with pytest.raises(
            ToolException,
            match="Unsupported Aha comment resource type",
        ):
            w.add_comment("sprint", "S-1", "hi")

    def test_list_comments_paginated(self):
        w = _wrapper()
        resp = _rest_stub(
            {"comments": [{"id": 1}, {"id": 2}], "pagination": {"current_page": 1, "total_pages": 1}}
        )
        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.list_comments("feature", "DEVELOP-1", max_records=10)
        assert req.call_args[0][1].endswith("/features/DEVELOP-1/comments")
        assert out == [{"id": 1}, {"id": 2}]

    def test_list_comments_returns_detailed_message(self):
        w = _wrapper()
        resp = _rest_stub(
            {"comments": [], "pagination": {"current_page": 1, "total_pages": 1}}
        )

        with patch.object(w._session, "request", return_value=resp):
            out = w.list_comments("epic", "EL-E-1")

        assert out == "Aha! API returned no comments for epic 'EL-E-1'."

    def test_list_todo_comments_returns_detailed_empty_message(self):
        w = _wrapper()
        resp = _rest_stub(
            {"comments": [], "pagination": {"current_page": 1, "total_pages": 0}}
        )

        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.list_comments("to_do", "EL-TODO-3")

        assert req.call_args[0][1].endswith("/tasks/EL-TODO-3/comments")
        assert out == "Aha! API returned no comments for to_do 'EL-TODO-3'."

    def test_todo_html_404_is_sanitized(self):
        w = _wrapper()
        html_error = (
            "<!DOCTYPE html><html><head>"
            "<title>Aha! | Record not found (404)</title>"
            "<style>.wrapper-500 { color: #666; }</style>"
            "</head><body>Record not found</body></html>"
        )
        resp = _rest_stub({}, ok=False, status=404, text=html_error)

        with patch.object(w._session, "request", return_value=resp):
            with pytest.raises(ToolException) as exc_info:
                w.list_comments("to_do", "EL-TODO-404")

        message = str(exc_info.value)
        assert "Aha! | Record not found (404)" in message
        assert "<html" not in message
        assert "<style>" not in message


class TestManageRecord:
    def test_action_specific_schemas_only_expose_relevant_fields(self):
        assert set(AhaCreateRecordInput.model_fields) == {
            "record_type",
            "parent_id",
            "properties",
        }
        assert set(AhaUpdateRecordInput.model_fields) == {
            "record_type",
            "record_id",
            "parent_id",
            "properties",
        }
        assert set(AhaDeleteRecordInput.model_fields) == {
            "record_type",
            "record_id",
            "parent_id",
        }

    @pytest.mark.parametrize(
        "schema",
        [AhaManageRecordInput, AhaCreateRecordInput, AhaUpdateRecordInput],
    )
    def test_properties_schema_normalizes_legacy_empty_array(self, schema):
        values = {
            "record_type": "feature",
            "record_id": "DEVELOP-1",
            "parent_id": "DEVELOP-R-1",
            "properties": [],
        }
        if schema is AhaManageRecordInput:
            values["action"] = "delete"

        assert schema.model_validate(values).properties == {}

    def test_properties_schema_still_rejects_nonempty_array(self):
        with pytest.raises(ValueError, match="valid dictionary"):
            AhaUpdateRecordInput.model_validate(
                {
                    "record_type": "feature",
                    "record_id": "DEVELOP-1",
                    "properties": ["name"],
                }
            )

    def test_delete_record_does_not_require_properties(self):
        parsed = AhaDeleteRecordInput.model_validate(
            {"record_type": "feature", "record_id": "DEVELOP-1"}
        )

        assert parsed.record_id == "DEVELOP-1"

    def test_dedicated_create_record_uses_create_route(self):
        w = _wrapper()
        resp = _rest_stub({"feature": {"id": 9}})
        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.create_record(
                record_type="feature",
                parent_id="DEVELOP-R-1",
                properties={"name": "New feature"},
            )

        assert req.call_args[0][0] == "POST"
        assert req.call_args[0][1].endswith(
            "/releases/DEVELOP-R-1/features"
        )
        assert out == {"id": 9}

    def test_dedicated_delete_record_uses_delete_route(self):
        w = _wrapper()
        with patch.object(
            w._session, "request", return_value=_rest_stub({})
        ) as req:
            out = w.delete_record("feature", "DEVELOP-1")

        assert req.call_args[0][0] == "DELETE"
        assert req.call_args[0][1].endswith("/features/DEVELOP-1")
        assert out == {
            "deleted": True,
            "record_type": "feature",
            "record_id": "DEVELOP-1",
        }

    def test_update_feature_puts_to_features_url(self):
        w = _wrapper()
        resp = _rest_stub({"feature": {"id": 1, "name": "new"}})
        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.manage_record(
                action="update",
                record_type="feature",
                record_id="DEVELOP-1",
                properties={"name": "new"},
            )
        method, url = req.call_args[0][:2]
        assert method == "PUT"
        assert url.endswith("/features/DEVELOP-1")
        assert req.call_args.kwargs["json"] == {"feature": {"name": "new"}}
        assert out == {"id": 1, "name": "new"}

    def test_create_feature_requires_parent(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="parent_id is required"):
            w.manage_record(action="create", record_type="feature", properties={"name": "x"})

    def test_create_feature_posts_under_release(self):
        w = _wrapper()
        resp = _rest_stub({"feature": {"id": 9}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.manage_record(
                action="create",
                record_type="feature",
                parent_id="REL-1",
                properties={"name": "x"},
            )
        method, url = req.call_args[0][:2]
        assert method == "POST"
        assert url.endswith("/releases/REL-1/features")
        assert req.call_args.kwargs["json"] == {"feature": {"name": "x"}}

    def test_create_requirement_posts_under_feature(self):
        w = _wrapper()
        resp = _rest_stub({"requirement": {"id": 8}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.manage_record(
                action="create",
                record_type="requirement",
                parent_id="DEVELOP-1",
                properties={"name": "req1"},
            )
        assert req.call_args[0][1].endswith("/features/DEVELOP-1/requirements")

    def test_update_requires_record_id(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="record_id is required"):
            w.manage_record(action="update", record_type="feature", properties={"name": "x"})

    def test_bad_record_type_rejected(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="does not support record_type"):
            w.manage_record(action="update", record_type="product", record_id="P-1", properties={})

    def test_bad_action_rejected(self):
        w = _wrapper()
        with pytest.raises(
            ToolException, match="action must be `create`, `update`, or `delete`"
        ):
            w.manage_record(action="patch", record_type="feature")

    # ----- delete action -----

    def test_delete_feature_hits_features_url(self):
        w = _wrapper()
        resp = _rest_stub({})
        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.manage_record(
                action="delete", record_type="feature", record_id="DEVELOP-1"
            )
        method, url = req.call_args[0][:2]
        assert method == "DELETE"
        assert url.endswith("/features/DEVELOP-1")
        assert out["deleted"] is True
        assert out["record_type"] == "feature"
        assert out["record_id"] == "DEVELOP-1"

    def test_delete_page_hits_pages_url(self):
        w = _wrapper()
        resp = _rest_stub({})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.manage_record(action="delete", record_type="page", record_id="ABC-N-1")
        method, url = req.call_args[0][:2]
        assert method == "DELETE"
        assert url.endswith("/pages/ABC-N-1")

    def test_delete_requires_record_id(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="record_id is required"):
            w.manage_record(action="delete", record_type="feature")

    # ----- create: new record types -----

    def test_create_release_posts_under_product(self):
        w = _wrapper()
        resp = _rest_stub({"release": {"id": 10}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.manage_record(
                action="create",
                record_type="release",
                parent_id="DEVELOP",
                properties={"name": "R1"},
            )
        method, url = req.call_args[0][:2]
        assert method == "POST"
        assert url.endswith("/products/DEVELOP/releases")
        assert req.call_args.kwargs["json"] == {"release": {"name": "R1"}}

    def test_create_initiative_posts_under_product(self):
        w = _wrapper()
        resp = _rest_stub({"initiative": {"id": 11}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.manage_record(
                action="create",
                record_type="initiative",
                parent_id="DEVELOP",
                properties={"name": "Init1"},
            )
        assert req.call_args[0][1].endswith("/products/DEVELOP/initiatives")
        assert req.call_args.kwargs["json"] == {"initiative": {"name": "Init1"}}

    def test_create_epic_posts_under_release(self):
        w = _wrapper()
        resp = _rest_stub({"epic": {"id": 12}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.manage_record(
                action="create",
                record_type="epic",
                parent_id="DEVELOP-R-1",
                properties={"name": "E1"},
            )
        assert req.call_args[0][1].endswith("/releases/DEVELOP-R-1/epics")

    def test_create_page_posts_under_product(self):
        w = _wrapper()
        resp = _rest_stub({"page": {"id": 13}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.manage_record(
                action="create",
                record_type="page",
                parent_id="DEVELOP",
                properties={"name": "Notes"},
            )
        assert req.call_args[0][1].endswith("/products/DEVELOP/pages")
        assert req.call_args.kwargs["json"] == {"page": {"name": "Notes"}}

    def test_create_release_requires_parent(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="parent_id is required"):
            w.manage_record(action="create", record_type="release", properties={"name": "x"})

    def test_create_page_requires_parent(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="parent_id is required"):
            w.manage_record(action="create", record_type="page", properties={"name": "x"})

    # ----- update: new record types -----

    def test_update_release_puts_to_product_scoped_url(self):
        w = _wrapper()
        resp = _rest_stub({"release": {"id": 1, "name": "renamed"}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.update_record(
                record_type="release",
                record_id="DEVELOP-R-1",
                parent_id="DEVELOP",
                properties={"name": "renamed"},
            )
        method, url = req.call_args[0][:2]
        assert method == "PUT"
        assert url.endswith("/products/DEVELOP/releases/DEVELOP-R-1")

    def test_update_release_requires_product_scope(self):
        w = _wrapper()

        with pytest.raises(
            ToolException,
            match=r"parent_id is required .*Aha scopes this endpoint by product",
        ):
            w.update_record(
                record_type="release",
                record_id="DEVELOP-R-1",
                properties={"name": "renamed"},
            )

    def test_update_epic_puts_to_epics_url(self):
        w = _wrapper()
        resp = _rest_stub({"epic": {"id": 2}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.manage_record(
                action="update",
                record_type="epic",
                record_id="DEVELOP-E-1",
                properties={"name": "n"},
            )
        assert req.call_args[0][1].endswith("/epics/DEVELOP-E-1")

    def test_update_initiative_puts_to_product_scoped_url(self):
        w = _wrapper()
        resp = _rest_stub({"initiative": {"id": 3}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.update_record(
                record_type="initiative",
                record_id="DEVELOP-S-1",
                parent_id="DEVELOP",
                properties={"description": "d"},
            )
        assert req.call_args[0][1].endswith(
            "/products/DEVELOP/initiatives/DEVELOP-S-1"
        )

    def test_delete_initiative_uses_product_scoped_url(self):
        w = _wrapper()
        with patch.object(
            w._session, "request", return_value=_rest_stub({})
        ) as req:
            w.delete_record(
                record_type="initiative",
                record_id="DEVELOP-S-1",
                parent_id="DEVELOP",
            )

        assert req.call_args[0][0] == "DELETE"
        assert req.call_args[0][1].endswith(
            "/products/DEVELOP/initiatives/DEVELOP-S-1"
        )

    def test_update_page_puts_to_pages_url(self):
        w = _wrapper()
        resp = _rest_stub({"page": {"id": 4}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.manage_record(
                action="update",
                record_type="page",
                record_id="ABC-N-1",
                properties={"name": "n"},
            )
        assert req.call_args[0][1].endswith("/pages/ABC-N-1")


class TestCreateRecordLink:
    def test_schema_exposes_documented_record_and_link_types(self):
        schema = AhaCreateRecordLinkInput.model_json_schema()

        assert set(schema["properties"]["from_record_type"]["enum"]) == {
            "feature",
            "requirement",
            "release",
            "idea",
            "epic",
            "release_phase",
            "initiative",
            "page",
            "goal",
        }
        assert set(schema["properties"]["to_record_type"]["enum"]) == {
            "feature",
            "release",
            "idea",
            "epic",
            "release_phase",
            "initiative",
            "page",
            "goal",
        }
        assert schema["properties"]["link_type"]["enum"] == [
            10,
            20,
            30,
            40,
            50,
            60,
            80,
        ]
        assert "link_type" in schema["required"]

    def test_resolves_references_and_posts_documented_payload(self):
        w = _wrapper()
        responses = [
            _rest_stub({"feature": {"id": "1001", "reference_num": "PROD-5"}}),
            _rest_stub({"epic": {"id": "2002", "reference_num": "PROD-E-1"}}),
            _rest_stub({}),
        ]

        with patch.object(w._session, "request", side_effect=responses) as req:
            out = w.create_record_link(
                from_record_type="feature",
                from_id="PROD-5",
                to_record_type="epic",
                to_id="PROD-E-1",
                link_type=10,
            )

        assert [call.args[0] for call in req.call_args_list] == [
            "GET",
            "GET",
            "POST",
        ]
        assert req.call_args_list[0].args[1].endswith("/features/PROD-5")
        assert req.call_args_list[1].args[1].endswith("/epics/PROD-E-1")
        assert req.call_args_list[2].args[1].endswith(
            "/features/1001/record_links"
        )
        assert req.call_args_list[2].kwargs["json"] == {
            "record_link": {
                "record_type": "epic",
                "record_id": 2002,
                "link_type": 10,
            }
        }
        assert out == {
            "created": True,
            "from_record_type": "feature",
            "from_reference_or_id": "PROD-5",
            "from_record_id": "1001",
            "to_record_type": "epic",
            "to_reference_or_id": "PROD-E-1",
            "to_record_id": "2002",
            "link_type": 10,
            "link_type_name": "relates to",
        }

    def test_numeric_ids_post_without_resolution_requests(self):
        w = _wrapper()
        resp = _rest_stub({"record_link": {"id": "3003"}})

        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.create_record_link(
                from_record_type="page",
                from_id="1001",
                to_record_type="feature",
                to_id="2002",
                link_type=20,
            )

        assert req.call_count == 1
        assert req.call_args.args[0] == "POST"
        assert req.call_args.args[1].endswith("/pages/1001/record_links")
        assert req.call_args.kwargs["json"] == {
            "record_link": {
                "record_type": "feature",
                "record_id": 2002,
                "link_type": 20,
            }
        }
        assert out == {"id": "3003"}

    @pytest.mark.parametrize(
        ("argument", "value", "message"),
        [
            ("from_record_type", "product", "unsupported from_record_type"),
            ("to_record_type", "requirement", "unsupported to_record_type"),
            ("link_type", 70, "unsupported link_type"),
        ],
    )
    def test_rejects_unsupported_contract_values(
        self, argument, value, message
    ):
        w = _wrapper()
        kwargs = {
            "from_record_type": "feature",
            "from_id": "1001",
            "to_record_type": "epic",
            "to_id": "2002",
            "link_type": 10,
        }
        kwargs[argument] = value

        with patch.object(w._session, "request") as req:
            with pytest.raises(ToolException, match=message):
                w.create_record_link(**kwargs)

        req.assert_not_called()

    def test_requirement_is_supported_as_source(self):
        w = _wrapper()
        responses = [
            _rest_stub(
                {
                    "requirement": {
                        "id": "1001",
                        "reference_num": "PROD-5-1",
                    }
                }
            ),
            _rest_stub({}),
        ]

        with patch.object(w._session, "request", side_effect=responses) as req:
            w.create_record_link(
                from_record_type="requirement",
                from_id="PROD-5-1",
                to_record_type="epic",
                to_id="2002",
                link_type=10,
            )

        assert req.call_args_list[0].args[1].endswith(
            "/requirements/PROD-5-1"
        )
        assert req.call_args_list[1].args[1].endswith(
            "/requirements/1001/record_links"
        )

    def test_resolves_goal_reference_before_link_creation(self):
        w = _wrapper()
        responses = [
            _rest_stub(
                {
                    "goals": [
                        {
                            "id": "3003",
                            "reference_num": "PROD-G-2",
                        }
                    ],
                    "pagination": {
                        "current_page": 1,
                        "total_pages": 1,
                    },
                }
            ),
            _rest_stub({"epic": {"id": "2002", "reference_num": "PROD-E-1"}}),
            _rest_stub({}),
        ]

        with patch.object(w._session, "request", side_effect=responses) as req:
            out = w.create_record_link(
                from_record_type="goal",
                from_id="PROD-G-2",
                to_record_type="epic",
                to_id="PROD-E-1",
                link_type=10,
            )

        assert [call.args[0] for call in req.call_args_list] == [
            "GET",
            "GET",
            "POST",
        ]
        assert req.call_args_list[0].args[1].endswith("/goals")
        assert req.call_args_list[0].kwargs["params"] == {
            "page": 1,
            "per_page": 100,
        }
        assert req.call_args_list[1].args[1].endswith("/epics/PROD-E-1")
        assert req.call_args_list[2].args[1].endswith(
            "/goals/3003/record_links"
        )
        assert req.call_args_list[2].kwargs["json"] == {
            "record_link": {
                "record_type": "epic",
                "record_id": 2002,
                "link_type": 10,
            }
        }
        assert out["from_reference_or_id"] == "PROD-G-2"
        assert out["from_record_id"] == "3003"

    def test_resolves_initiative_reference_through_collection(self):
        w = _wrapper()
        responses = [
            _rest_stub(
                {
                    "initiatives": [
                        {
                            "id": "4004",
                            "reference_num": "PROD-S-2",
                        }
                    ],
                    "pagination": {
                        "current_page": 1,
                        "total_pages": 1,
                    },
                }
            ),
            _rest_stub({}),
        ]

        with patch.object(w._session, "request", side_effect=responses) as req:
            w.create_record_link(
                from_record_type="initiative",
                from_id="PROD-S-2",
                to_record_type="epic",
                to_id="2002",
                link_type=10,
            )

        assert req.call_args_list[0].args[1].endswith("/initiatives")
        assert req.call_args_list[1].args[1].endswith(
            "/initiatives/4004/record_links"
        )

    def test_resolves_page_reference_through_graphql(self):
        w = _wrapper()
        response = _rest_stub({})

        with (
            patch.object(
                w,
                "_gql",
                return_value={"page": {"id": "5005"}},
            ) as gql,
            patch.object(w._session, "request", return_value=response) as req,
        ):
            w.create_record_link(
                from_record_type="page",
                from_id="PROD-N-2",
                to_record_type="epic",
                to_id="2002",
                link_type=10,
            )

        gql.assert_called_once_with(
            _QUERY_GET_PAGE,
            {"id": "PROD-N-2", "includeParent": False},
        )
        assert req.call_args.args[1].endswith("/pages/5005/record_links")

    def test_release_phase_reference_requires_numeric_id(self):
        w = _wrapper()

        with patch.object(w._session, "request") as req:
            with pytest.raises(
                ToolException,
                match="target release_phase requires a numeric ID",
            ):
                w.create_record_link(
                    from_record_type="feature",
                    from_id="1001",
                    to_record_type="release_phase",
                    to_id="PHASE-1",
                    link_type=10,
                )

        req.assert_not_called()


class TestCopyRecord:
    def test_release_duplicate(self):
        w = _wrapper()
        resp = _rest_stub({"release": {"id": 99}})
        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.copy_record("release", "REL-1")
        method, url = req.call_args[0][:2]
        assert method == "POST"
        assert url.endswith("/releases/REL-1/duplicate")
        assert out == {"id": 99}

    def test_non_release_rejected(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="only supports duplicating releases"):
            w.copy_record("feature", "DEVELOP-1")


class TestFieldsMetadata:
    def test_fields_metadata_hits_custom_fields(self):
        w = _wrapper()
        resp = _rest_stub(
            {"custom_fields": [{"id": 1, "name": "Sprint"}], "pagination": {"current_page": 1, "total_pages": 1}}
        )
        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.fields_metadata()
        assert req.call_args[0][1].endswith("/custom_fields")
        assert out == [{"id": 1, "name": "Sprint"}]

    def test_field_options_metadata(self):
        w = _wrapper()
        resp = _rest_stub({"options": [{"id": 1, "value": "A"}, {"id": 2, "value": "B"}]})
        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.field_options_metadata("42")
        assert req.call_args[0][1].endswith("/custom_fields/42/options")
        assert out == [{"id": 1, "value": "A"}, {"id": 2, "value": "B"}]

    def test_field_options_requires_id(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="field_id is required"):
            w.field_options_metadata("")


class TestAttachFile:
    def test_schema_advertises_standard_artifact_filepath(self):
        description = AhaAttachFileInput.model_json_schema()["properties"][
            "filepath"
        ]["description"]

        assert "/{bucket}/{filename}" in description
        assert "artifact://" not in description

    def test_uploads_artifact_as_multipart(self):
        elitea = MagicMock()
        artifact_client = elitea.artifact.return_value
        artifact_client.get_raw_content_by_filepath.return_value = (
            b"hi",
            "hello.txt",
        )
        w = _wrapper(elitea=elitea)

        resp = MagicMock()
        resp.ok = True
        resp.status_code = 201
        resp.content = b'{"attachment": {"id": 5}}'
        resp.json = lambda: {"attachment": {"id": 5}}
        record_resp = _rest_stub(
            {"idea": {"description": {"id": "793547626"}}}
        )

        with (
            patch.object(
                w._session,
                "request",
                return_value=record_resp,
            ) as request,
            patch.object(w._session, "post", return_value=resp) as post,
        ):
            out = w.attach_file(
                "idea",
                "PROD-I-1",
                "/generated/hello.txt",
            )

        assert request.call_args[0][0] == "GET"
        assert request.call_args[0][1].endswith("/ideas/PROD-I-1")
        url = post.call_args[0][0]
        assert url.endswith("/notes/793547626/attachments")
        assert "files" in post.call_args.kwargs
        assert post.call_args.kwargs["files"]["attachment[data]"][0] == "hello.txt"
        assert post.call_args.kwargs["files"]["attachment[data]"][1] == b"hi"
        assert post.call_args.kwargs["headers"] == {"Content-Type": None}
        assert post.call_args.kwargs["timeout"] == 60
        elitea.artifact.assert_called_once_with("__temp__")
        artifact_client.get_raw_content_by_filepath.assert_called_once_with(
            "/generated/hello.txt"
        )
        assert out == {"id": 5}

    def test_filename_override_is_used(self):
        elitea = MagicMock()
        elitea.artifact.return_value.get_raw_content_by_filepath.return_value = (
            b"content",
            "source.txt",
        )
        w = _wrapper(elitea=elitea)
        response = _rest_stub({"attachment": {"id": 6}}, status=201)
        record_response = _rest_stub(
            {"feature": {"description": {"id": "12345"}}}
        )

        with (
            patch.object(
                w._session,
                "request",
                return_value=record_response,
            ),
            patch.object(w._session, "post", return_value=response) as post,
        ):
            w.attach_file(
                "feature",
                "DEVELOP-1",
                "/generated/source.txt",
                filename="renamed.txt",
            )

        assert (
            post.call_args.kwargs["files"]["attachment[data]"][0]
            == "renamed.txt"
        )

    def test_to_do_upload_uses_task_attachment_endpoint(self):
        elitea = MagicMock()
        elitea.artifact.return_value.get_raw_content_by_filepath.return_value = (
            b"todo",
            "todo.txt",
        )
        w = _wrapper(elitea=elitea)
        response = _rest_stub({"attachment": {"id": 7}})

        with (
            patch.object(w._session, "request") as request,
            patch.object(w._session, "post", return_value=response) as post,
        ):
            out = w.attach_file("to_do", "1041191038", "/bucket/todo.txt")

        request.assert_not_called()
        assert post.call_args[0][0].endswith(
            "/tasks/1041191038/attachments"
        )
        assert out == {"id": 7}

    def test_record_without_description_id_fails_before_upload(self):
        elitea = MagicMock()
        elitea.artifact.return_value.get_raw_content_by_filepath.return_value = (
            b"data",
            "file.txt",
        )
        w = _wrapper(elitea=elitea)
        record_response = _rest_stub({"idea": {"description": {}}})

        with (
            patch.object(
                w._session,
                "request",
                return_value=record_response,
            ),
            patch.object(w._session, "post") as post,
            pytest.raises(
                ToolException,
                match=r"response does not contain description.id",
            ),
        ):
            w.attach_file("idea", "PROD-I-1", "/bucket/file.txt")

        post.assert_not_called()

    def test_missing_elitea_client_raises_detailed_error(self):
        w = _wrapper()

        with pytest.raises(
            ToolException,
            match=(
                r"failed to retrieve artifact '/bucket/file.txt': "
                r"EliteA client is required"
            ),
        ):
            w.attach_file("feature", "DEVELOP-1", "/bucket/file.txt")


class TestDispatchers:
    def test_find_project_uses_products_endpoint(self):
        w = _wrapper()
        resp = _rest_stub({"products": [{"id": "P-1"}], "pagination": {"current_page": 1, "total_pages": 1}})
        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.find_project(q="alpha")
        assert req.call_args[0][1].endswith("/products")
        assert req.call_args.kwargs["params"]["q"] == "alpha"
        assert out == [{"id": "P-1"}]

    def test_search_records_dispatches_feature(self):
        w = _wrapper()
        resp = _rest_stub({"features": [{"id": 1}], "pagination": {"current_page": 1, "total_pages": 1}})
        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.search_records(record_type="feature", release_id="R-1", q="foo")
        assert "releases/R-1/features" in req.call_args[0][1]
        assert out == [{"id": 1}]

    def test_search_records_dispatches_requirement_with_feature_id(self):
        w = _wrapper()
        resp = _rest_stub(
            {
                "requirements": [{"id": 1}],
                "pagination": {"current_page": 1, "total_pages": 1},
            }
        )

        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.search_records(
                record_type="requirement",
                feature_id="DEVELOP-1",
                q="foo",
            )

        assert "features/DEVELOP-1/requirements" in req.call_args[0][1]
        assert out == [{"id": 1}]

    def test_search_records_rejects_bad_type(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="unsupported record_type"):
            w.search_records(record_type="story")

    def test_read_records_dispatches_feature(self):
        w = _wrapper()
        resp = _rest_stub({"feature": {"id": 1, "name": "F"}})
        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.read_records(record_type="feature", reference_or_id="DEVELOP-1")
        assert req.call_args[0][1].endswith("/features/DEVELOP-1")
        assert out == {"id": 1, "name": "F"}

    def test_read_records_page_uses_graphql(self):
        w = _wrapper()
        resp = _rest_stub({"data": {"page": {"id": "1", "name": "P"}}})
        with patch.object(w._session, "post", return_value=resp) as post:
            out = w.read_records(record_type="page", reference_or_id="ABC-N-1")
        assert post.call_args[0][0].endswith("/api/v2/graphql")
        assert out == {"id": "1", "name": "P"}

    def test_read_records_rejects_bad_type(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="unsupported record_type"):
            w.read_records(record_type="sprint", reference_or_id="S-1")
