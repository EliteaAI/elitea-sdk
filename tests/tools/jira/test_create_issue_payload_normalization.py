"""
Tests for issue #5856: Jira create_issue/update_issue reliability for
escaped/malformed LLM-generated JSON payloads.
"""
import json

import pytest
from langchain_core.tools import ToolException
from requests import Response
from requests.exceptions import HTTPError

from elitea_sdk.tools.jira.api_wrapper import (
    JiraApiWrapper,
    normalize_and_parse_issue_json,
    normalize_rich_text_fields,
    normalize_rich_text_operations,
)


class TestNormalizeAndParseIssueJson:
    def test_valid_raw_json_unchanged(self):
        payload = {"fields": {"project": {"key": "PROJ"}, "summary": "Issue title",
                               "issuetype": {"name": "Task"}}}
        assert normalize_and_parse_issue_json(json.dumps(payload)) == payload

    def test_fenced_json(self):
        payload = {"fields": {"project": {"key": "PROJ"}, "summary": "Issue title"}}
        fenced = f"```json\n{json.dumps(payload)}\n```"
        assert normalize_and_parse_issue_json(fenced) == payload

    def test_fenced_json_without_language_tag(self):
        payload = {"fields": {"project": {"key": "PROJ"}, "summary": "Issue title"}}
        fenced = f"```\n{json.dumps(payload)}\n```"
        assert normalize_and_parse_issue_json(fenced) == payload

    def test_prose_wrapped_json(self):
        payload = {"fields": {"project": {"key": "PROJ"}, "summary": "Issue title"}}
        wrapped = f"Sure, here is the JSON payload:\n{json.dumps(payload)}\nLet me know if you need anything else."
        assert normalize_and_parse_issue_json(wrapped) == payload

    def test_double_encoded_json_string(self):
        payload = {"fields": {"summary": "Bug", "description": "a\nb"}}
        double_encoded = json.dumps(json.dumps(payload))
        assert normalize_and_parse_issue_json(double_encoded) == payload

    def test_multiline_plain_text_description_preserved(self):
        payload = {"fields": {"project": {"key": "PROJ"}, "summary": "QA check",
                               "description": "Summary:\nVerify story creation works.\n\n"
                                              "Acceptance Criteria:\n- Issue is created"}}
        result = normalize_and_parse_issue_json(json.dumps(payload))
        assert result["fields"]["description"] == payload["fields"]["description"]

    def test_multiline_adf_description_preserved(self):
        payload = {
            "fields": {
                "project": {"key": "PROJ"},
                "summary": "QA check",
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {"type": "paragraph", "content": [
                            {"type": "text", "text": "Summary:"},
                            {"type": "hardBreak"},
                            {"type": "text", "text": "Verify story creation works."},
                        ]},
                    ],
                },
            }
        }
        result = normalize_and_parse_issue_json(json.dumps(payload))
        assert result["fields"]["description"] == payload["fields"]["description"]

    def test_malformed_non_json_raises_tool_exception(self):
        with pytest.raises(ToolException):
            normalize_and_parse_issue_json("not json at all {")

    def test_json_array_raises_tool_exception(self):
        with pytest.raises(ToolException):
            normalize_and_parse_issue_json("[1, 2, 3]")

    def test_non_string_input_raises_tool_exception(self):
        with pytest.raises(ToolException):
            normalize_and_parse_issue_json({"already": "a dict"})


@pytest.fixture
def wrapper(monkeypatch):
    instance = JiraApiWrapper.model_construct(labels=None)
    mock_client = type("MockClient", (), {})()
    mock_client.url = "https://test.atlassian.net/"
    mock_client.create_issue = lambda fields, update=None: {"key": "PROJ-1", **fields}
    mock_client.update_issue = lambda issue_key, update: {"key": issue_key, **update}
    mock_client.resource_url = lambda resource: f"rest/api/2/{resource}"
    mock_client.put_calls = []

    def fake_put(path, data=None):
        mock_client.put_calls.append((path, data))
        return {"key": path.rsplit("/", 1)[-1]}

    mock_client.put = fake_put
    monkeypatch.setattr(instance, "_get_client", lambda: mock_client)
    return instance


class TestCreateIssueRecovery:
    def test_create_issue_accepts_fenced_json(self, wrapper):
        payload = {"fields": {"project": {"key": "PROJ"}, "summary": "Issue title",
                               "issuetype": {"name": "Task"}}}
        fenced = f"```json\n{json.dumps(payload)}\n```"
        result = wrapper.create_issue(fenced)
        assert "created successfully" in result

    def test_create_issue_accepts_double_encoded_json(self, wrapper):
        payload = {"fields": {"project": {"key": "PROJ"}, "summary": "Issue title"}}
        double_encoded = json.dumps(json.dumps(payload))
        result = wrapper.create_issue(double_encoded)
        assert "created successfully" in result

    def test_create_issue_rejects_missing_fields(self, wrapper):
        with pytest.raises(ToolException):
            wrapper.create_issue(json.dumps({"notfields": {}}))

    def test_create_issue_rejects_malformed_json(self, wrapper):
        with pytest.raises(ToolException):
            wrapper.create_issue("here is your ticket: not really json")


class TestUpdateIssueRecovery:
    def test_update_issue_accepts_fenced_json(self, wrapper):
        payload = {"key": "PROJ-1", "fields": {"summary": "Updated title"}}
        fenced = f"```json\n{json.dumps(payload)}\n```"
        result = wrapper.update_issue(fenced)
        assert "updated successfully" in result

    def test_update_issue_returns_tool_exception_on_malformed_json(self, wrapper):
        result = wrapper.update_issue("not valid json {{{")
        assert isinstance(result, ToolException)

    def test_update_issue_returns_tool_exception_on_missing_key(self, wrapper):
        result = wrapper.update_issue(json.dumps({"fields": {"summary": "x"}}))
        assert isinstance(result, ToolException)


ADF_DESCRIPTION = {
    "type": "doc",
    "version": 1,
    "content": [
        {"type": "paragraph", "content": [
            {"type": "text", "text": "Retest line 1"},
            {"type": "hardBreak"},
            {"type": "text", "text": "Retest line 2"},
        ]},
    ],
}


class TestRichTextNormalization:
    """Issue #6476: nested/ADF description values crashed update_issue with an HTTPError."""

    def test_adf_flattened_to_text_for_v2(self):
        assert normalize_rich_text_fields({"description": ADF_DESCRIPTION}, "2") == {
            "description": "Retest line 1\nRetest line 2"
        }

    def test_plain_text_wrapped_into_adf_for_v3(self):
        result = normalize_rich_text_fields({"description": "line 1\nline 2"}, "3")
        assert result["description"]["type"] == "doc"
        assert {"type": "hardBreak"} in result["description"]["content"][0]["content"]

    def test_adf_preserved_for_v3(self):
        assert normalize_rich_text_fields({"description": ADF_DESCRIPTION}, "3") == {
            "description": ADF_DESCRIPTION
        }

    def test_scalar_fields_untouched(self):
        fields = {"summary": "plain title", "customfield_1": {"value": "x"}}
        assert normalize_rich_text_fields(fields, "2") == fields

    def test_update_operations_normalized(self):
        result = normalize_rich_text_operations(
            {"description": [{"set": ADF_DESCRIPTION}], "labels": [{"add": "x"}]}, "2"
        )
        assert result == {
            "description": [{"set": "Retest line 1\nRetest line 2"}],
            "labels": [{"add": "x"}],
        }

    def test_update_issue_with_adf_description_succeeds_on_v2(self, wrapper):
        wrapper.api_version = "2"
        payload = {"key": "PROJ-1", "fields": {"description": ADF_DESCRIPTION}}
        result = wrapper.update_issue(json.dumps(payload))
        assert "updated successfully" in result
        _, body = wrapper._get_client().put_calls[-1]
        assert body["fields"]["description"] == "Retest line 1\nRetest line 2"

    def test_update_issue_uses_version_aware_endpoint(self, wrapper):
        wrapper.api_version = "3"
        client = wrapper._get_client()
        client.resource_url = lambda resource: f"rest/api/3/{resource}"
        wrapper.update_issue(json.dumps({"key": "PROJ-1", "fields": {"description": "hello"}}))
        path, body = client.put_calls[-1]
        assert path == "rest/api/3/issue/PROJ-1"
        assert body["fields"]["description"]["type"] == "doc"

    def test_update_issue_http_error_has_no_traceback(self, wrapper):
        wrapper.api_version = "2"
        response = Response()
        response.status_code = 400
        response._content = json.dumps(
            {"errorMessages": [], "errors": {"description": "Operation value must be a string"}}
        ).encode()

        def failing_put(path, data=None):
            raise HTTPError("400 Client Error", response=response)

        wrapper._get_client().put = failing_put
        result = wrapper.update_issue(json.dumps({"key": "PROJ-1", "fields": {"summary": "x"}}))
        assert "Traceback" not in result
        assert "api_wrapper.py" not in result
        assert "Operation value must be a string" in result
