"""
Tests for `JiraApiWrapper.set_issue_status`.

The underlying `atlassian.Jira.set_issue_status` resolves the status name to a
transition id and posts `{"transition": {"id": None}}` when no transition leads
to that status, which Jira rejects with an opaque "'transition' identifier must
be an integer". The wrapper checks the available transitions first so callers
get an actionable error, and passes `fields` / `update` blocks through as given.
"""
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.jira.api_wrapper import JiraApiWrapper

_TRANSITIONS = [
    {"id": 11, "name": "To Do", "to": "To Do"},
    {"id": 21, "name": "Start", "to": "In Progress"},
]


def _wrapper() -> JiraApiWrapper:
    w = JiraApiWrapper.model_construct()
    client = MagicMock(name="jira_client")
    client.url = "https://example.atlassian.net/"
    client.get_issue_transitions.return_value = _TRANSITIONS
    w._client = client
    w._get_client = lambda: client
    w.labels = []
    return w


def test_unknown_status_raises_clear_error_without_posting():
    w = _wrapper()

    with pytest.raises(ToolException) as exc:
        w.set_issue_status("AT-1", "INVALID_STATUS_XYZ", "{}")

    message = str(exc.value)
    assert "Status 'INVALID_STATUS_XYZ' is not available for issue AT-1" in message
    assert "In Progress" in message and "To Do" in message
    w._client.set_issue_status.assert_not_called()


def test_known_status_is_case_insensitive_and_returns_browse_url():
    w = _wrapper()

    result = w.set_issue_status("AT-1", "in progress", "{}")

    w._client.set_issue_status.assert_called_once_with(
        issue_key="AT-1", status_name="in progress", fields=None, update=None)
    assert "https://example.atlassian.net/browse/AT-1" in result


def test_fields_and_update_blocks_are_passed_separately():
    w = _wrapper()

    w.set_issue_status(
        "AT-1", "In Progress",
        '{"fields": {"resolution": {"name": "Done"}}, "update": {"comment": [{"add": {"body": "x"}}]}}')

    w._client.set_issue_status.assert_called_once_with(
        issue_key="AT-1", status_name="In Progress",
        fields={"resolution": {"name": "Done"}},
        update={"comment": [{"add": {"body": "x"}}]})
