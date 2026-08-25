"""
Tests for issue #5856: Jira create_issue/update_issue reliability for
escaped/malformed LLM-generated JSON payloads.
"""
import json

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.jira.api_wrapper import (
    JiraApiWrapper,
    normalize_and_parse_issue_json,
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
