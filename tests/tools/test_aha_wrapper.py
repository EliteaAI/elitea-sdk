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

from elitea_sdk.tools.aha.api_wrapper import (
    AhaApiWrapper,
    _FEATURE_REF_RE,
    _PAGE_REF_RE,
    _REQUIREMENT_REF_RE,
)


def _wrapper(base_url: str = "https://example.aha.io") -> AhaApiWrapper:
    return AhaApiWrapper(base_url=base_url, api_key=SecretStr("token"))


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


class TestToolRegistry:
    def test_registry_exposes_all_tools(self):
        w = _wrapper()
        tools = w.get_available_tools()
        names = {t["name"] for t in tools}
        # 19 M2 read tools + 11 M3 write/dispatcher tools + 5 report/analysis tools = 35
        assert len(tools) == 35
        # Spot-check every category
        assert {"get_feature", "list_features", "search", "get_page", "get_feature_gql"} <= names
        assert {"find_project", "search_records", "read_records"} <= names
        assert {
            "add_comment",
            "list_comments",
            "manage_record",
            "create_record_link",
            "copy_record",
            "fields_metadata",
            "field_options_metadata",
            "attach_file",
        } <= names
        assert {
            "get_report_data",
            "get_report_columns_and_filters",
            "get_report_filter_options",
            "manage_report",
            "analyze_records",
        } <= names

    def test_each_tool_has_required_shape(self):
        w = _wrapper()
        for t in w.get_available_tools():
            assert "name" in t and "description" in t and "args_schema" in t and "ref" in t
            assert callable(t["ref"])


class TestComments:
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

    def test_add_comment_supports_todo_alias(self):
        w = _wrapper()
        resp = _rest_stub({"comment": {"id": 2}})

        with patch.object(w._session, "request", return_value=resp) as req:
            w.add_comment("todo", "42", "note")
        assert req.call_args[0][1].endswith("/to_dos/42/comments")

    def test_add_comment_rejects_empty_body(self):
        w = _wrapper()
        with patch.object(w._session, "request") as req:
            with pytest.raises(ToolException, match="body is required"):
                w.add_comment("feature", "DEVELOP-1", "  ")
        req.assert_not_called()

    def test_add_comment_rejects_unsupported_resource(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="Unsupported Aha resource type"):
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


class TestManageRecord:
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
            ToolException, match="action must be 'create', 'update', or 'delete'"
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

    def test_update_release_puts_to_releases_url(self):
        w = _wrapper()
        resp = _rest_stub({"release": {"id": 1, "name": "renamed"}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.manage_record(
                action="update",
                record_type="release",
                record_id="DEVELOP-R-1",
                properties={"name": "renamed"},
            )
        method, url = req.call_args[0][:2]
        assert method == "PUT"
        assert url.endswith("/releases/DEVELOP-R-1")

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

    def test_update_initiative_puts_to_initiatives_url(self):
        w = _wrapper()
        resp = _rest_stub({"initiative": {"id": 3}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.manage_record(
                action="update",
                record_type="initiative",
                record_id="DEVELOP-I-1",
                properties={"description": "d"},
            )
        assert req.call_args[0][1].endswith("/initiatives/DEVELOP-I-1")

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
    def test_from_feature_to_feature(self):
        w = _wrapper()
        resp = _rest_stub({"record_link": {"id": 1}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.create_record_link(
                from_record_type="feature",
                from_id="DEVELOP-1",
                to_record_type="feature",
                to_id="DEVELOP-2",
                link_type="blocks",
            )
        method, url = req.call_args[0][:2]
        assert method == "POST"
        assert url.endswith("/features/DEVELOP-1/record_links")
        body = req.call_args.kwargs["json"]
        assert body["record_link"]["linkable_type"] == "Feature"
        assert body["record_link"]["linkable_id"] == "DEVELOP-2"
        assert body["record_link"]["link_type"] == "blocks"

    def test_rejects_non_feature_source(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="from features"):
            w.create_record_link(
                from_record_type="idea",
                from_id="I-1",
                to_record_type="feature",
                to_id="F-1",
            )


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
    def test_uploads_multipart(self, tmp_path):
        w = _wrapper()
        f = tmp_path / "hello.txt"
        f.write_bytes(b"hi")

        resp = MagicMock()
        resp.ok = True
        resp.status_code = 201
        resp.content = b'{"attachment": {"id": 5}}'
        resp.json = lambda: {"attachment": {"id": 5}}

        with patch.object(w._session, "post", return_value=resp) as post:
            out = w.attach_file("feature", "DEVELOP-1", str(f))

        url = post.call_args[0][0]
        assert url.endswith("/features/DEVELOP-1/attachments")
        assert "files" in post.call_args.kwargs
        assert post.call_args.kwargs["files"]["attachment[file]"][0] == "hello.txt"
        assert out == {"id": 5}

    def test_missing_file_raises(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="cannot read"):
            w.attach_file("feature", "DEVELOP-1", "/no/such/file.txt")


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


class TestReports:
    def test_get_report_data_list_view_url(self):
        w = _wrapper()
        payload = {"columns": [{"name": "Name"}], "rows": [{"Name": "F1"}]}
        resp = _rest_stub(payload)
        with patch.object(w._session, "request", return_value=resp) as req:
            out = w.get_report_data("123")
        method, url = req.call_args[0][:2]
        assert method == "GET"
        assert url.endswith("/custom_reports/123/list_view")
        assert out == payload

    def test_get_report_data_pivot_view_url(self):
        w = _wrapper()
        resp = _rest_stub({"pivot": {"rows": []}})
        with patch.object(w._session, "request", return_value=resp) as req:
            w.get_report_data("42", view="pivot")
        assert req.call_args[0][1].endswith("/custom_reports/42/pivot_view")

    def test_get_report_data_rejects_bad_view(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="Unsupported report view"):
            w.get_report_data("42", view="chart")

    def test_get_report_data_requires_id(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="report_id is required"):
            w.get_report_data("  ")

    def test_get_report_columns_and_filters_extracts_sections(self):
        w = _wrapper()
        payload = {
            "columns": [{"name": "Name"}, {"name": "Status"}],
            "filters": [{"name": "Workflow status", "options": ["Open", "Closed"]}],
            "rows": [{"Name": "F1"}],
        }
        resp = _rest_stub(payload)
        with patch.object(w._session, "request", return_value=resp):
            out = w.get_report_columns_and_filters("7")
        assert out == {
            "columns": [{"name": "Name"}, {"name": "Status"}],
            "filters": [{"name": "Workflow status", "options": ["Open", "Closed"]}],
        }

    def test_get_report_columns_and_filters_defaults_empty(self):
        w = _wrapper()
        resp = _rest_stub({"rows": []})
        with patch.object(w._session, "request", return_value=resp):
            out = w.get_report_columns_and_filters("7")
        assert out == {"columns": [], "filters": []}

    def test_get_report_filter_options_returns_options(self):
        w = _wrapper()
        payload = {
            "filters": [
                {"name": "Workflow status", "options": ["Open", "In progress", "Closed"]},
                {"name": "Owner", "options": [{"id": 1, "name": "Alice"}]},
            ]
        }
        resp = _rest_stub(payload)
        with patch.object(w._session, "request", return_value=resp):
            out = w.get_report_filter_options("7", "workflow status")
        assert out == ["Open", "In progress", "Closed"]

    def test_get_report_filter_options_case_insensitive_match(self):
        w = _wrapper()
        resp = _rest_stub({"filters": [{"name": "Owner", "choices": ["Alice", "Bob"]}]})
        with patch.object(w._session, "request", return_value=resp):
            out = w.get_report_filter_options("7", "OWNER")
        assert out == ["Alice", "Bob"]

    def test_get_report_filter_options_missing_filter_raises(self):
        w = _wrapper()
        resp = _rest_stub({"filters": [{"name": "Status"}]})
        with patch.object(w._session, "request", return_value=resp):
            with pytest.raises(ToolException, match="No filter named"):
                w.get_report_filter_options("7", "Owner")

    def test_get_report_filter_options_non_enumerable_raises(self):
        w = _wrapper()
        resp = _rest_stub({"filters": [{"name": "Search"}]})  # no options key
        with patch.object(w._session, "request", return_value=resp):
            with pytest.raises(ToolException, match="not enumerable"):
                w.get_report_filter_options("7", "Search")

    def test_get_report_filter_options_requires_filter_name(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="filter_name is required"):
            w.get_report_filter_options("7", "  ")

    def test_manage_report_rejects_unsupported_action(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="unsupported action"):
            w.manage_report(action="patch")

    def test_manage_report_create_raises_not_supported(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="does not support report management"):
            w.manage_report(action="create", properties={"name": "R"})

    def test_manage_report_update_raises_not_supported(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="does not support report management"):
            w.manage_report(action="update", report_id="7", properties={"name": "R"})

    def test_manage_report_delete_raises_not_supported(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="does not support report management"):
            w.manage_report(action="delete", report_id="7")


class TestAnalyzeRecords:
    def test_analyze_aggregates_types_status_and_assignee(self):
        w = _wrapper()
        # Two features + one release; two share a workflow_status name.
        records = {
            "features/DEVELOP-1": {
                "feature": {
                    "id": 1,
                    "workflow_status": {"name": "In progress"},
                    "assigned_to_user": {"name": "Alice"},
                    "updated_at": "2026-06-01T00:00:00Z",
                    "tags": ["ml", "urgent"],
                }
            },
            "features/DEVELOP-2": {
                "feature": {
                    "id": 2,
                    "workflow_status": {"name": "Closed"},
                    "updated_at": "2026-06-05T00:00:00Z",
                    "tags": [{"name": "ml"}, {"name": "backlog"}],
                }
            },
            "releases/DEVELOP-R-1": {
                "release": {
                    "id": 100,
                    "workflow_status": "In progress",
                    "updated_at": "2026-05-20T00:00:00Z",
                }
            },
        }

        def fake_request(method, url, **kwargs):
            for key, payload in records.items():
                if url.endswith(f"/{key}"):
                    return _rest_stub(payload)
            return _rest_stub({}, ok=False, status=404, text="not found")

        with patch.object(w._session, "request", side_effect=fake_request):
            out = w.analyze_records(
                references=[
                    {"record_type": "feature", "reference_or_id": "DEVELOP-1"},
                    {"record_type": "feature", "reference_or_id": "DEVELOP-2"},
                    {"record_type": "release", "reference_or_id": "DEVELOP-R-1"},
                ]
            )

        assert out["count"] == 3
        assert out["by_record_type"] == {"feature": 2, "release": 1}
        assert out["by_workflow_status"] == {"In progress": 2, "Closed": 1}
        assert out["by_assigned_to"] == {"Alice": 1}
        assert out["updated_at"]["min"] == "2026-05-20T00:00:00Z"
        assert out["updated_at"]["max"] == "2026-06-05T00:00:00Z"
        assert out["tags"] == ["backlog", "ml", "urgent"]
        assert out["errors"] == []

    def test_analyze_records_records_fetch_errors(self):
        w = _wrapper()

        def fake_request(method, url, **kwargs):
            if url.endswith("/features/GOOD-1"):
                return _rest_stub({"feature": {"id": 1}})
            return _rest_stub({}, ok=False, status=404, text='{"error":"not found"}')

        with patch.object(w._session, "request", side_effect=fake_request):
            out = w.analyze_records(
                references=[
                    {"record_type": "feature", "reference_or_id": "GOOD-1"},
                    {"record_type": "feature", "reference_or_id": "MISSING-1"},
                ]
            )

        assert out["count"] == 1
        assert out["by_record_type"] == {"feature": 1}
        assert len(out["errors"]) == 1
        assert "MISSING-1" in out["errors"][0]["reference"]

    def test_analyze_records_rejects_empty_list(self):
        w = _wrapper()
        with pytest.raises(ToolException, match="references list is empty"):
            w.analyze_records(references=[])

    def test_analyze_records_flags_malformed_entries(self):
        w = _wrapper()
        with patch.object(w._session, "request") as req:
            out = w.analyze_records(
                references=[
                    {"record_type": "feature"},  # missing ref
                    {"reference_or_id": "X"},  # missing type
                    "not-a-dict",  # type: ignore[list-item]
                ]
            )
        req.assert_not_called()
        assert out["count"] == 0
        assert len(out["errors"]) == 3
