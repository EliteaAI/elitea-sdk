"""A failed Jira call must read as one line, with the traceback kept for the expandable details.

Re-breaks if a Jira wrapper method pastes `format_exc()` into its error again, or if the
toolkit test runner stops sending the exception chain as `debug_error`.
"""
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.runtime.clients.client import EliteAClient
from elitea_sdk.tools.jira.api_wrapper import JiraApiWrapper, parse_payload_params

JIRA_MESSAGE = "Issue does not exist or you do not have permission to see it."


def _wrapper(client) -> JiraApiWrapper:
    w = JiraApiWrapper.model_construct()
    client.url = "https://example.atlassian.net/"
    w._client = client
    w._get_client = lambda: client
    w.labels = []
    return w


def _failing_client(method_name):
    client = MagicMock(name="jira_client")
    getattr(client, method_name).side_effect = Exception(JIRA_MESSAGE)
    return client


def _assert_readable(message, *expected):
    assert "Traceback" not in message
    assert 'File "' not in message
    assert JIRA_MESSAGE in message
    for part in expected:
        assert part in message


@pytest.mark.parametrize(
    "client_method, call, expected",
    [
        pytest.param("issue_add_comment", lambda w: w.add_comments("NOPE-1", "hi"),
                     "Error adding comment to Jira issue NOPE-1", id="add_comments"),
        pytest.param("projects", lambda w: w.list_projects(),
                     "Error listing Jira projects", id="list_projects"),
        pytest.param("get_issue_transitions", lambda w: w.set_issue_status("NOPE-1", "Done", "{}"),
                     "Error setting status for Jira issue NOPE-1", id="set_issue_status"),
    ],
)
def test_raised_errors_carry_the_jira_message_not_the_traceback(client_method, call, expected):
    w = _wrapper(_failing_client(client_method))

    with pytest.raises(ToolException) as exc:
        call(w)

    _assert_readable(str(exc.value), expected)
    assert str(exc.value.__cause__) == JIRA_MESSAGE


def test_list_comments_returns_a_readable_error():
    w = _wrapper(_failing_client("issue_get_comments"))

    message = w.list_comments("NOPE-1")

    _assert_readable(message, "comments for issue NOPE-1")


def test_invalid_payload_params_name_the_json_problem():
    with pytest.raises(ToolException) as exc:
        parse_payload_params("{not json")

    message = str(exc.value)
    assert "not valid JSON" in message
    assert "Traceback" not in message


class _FailingTool:
    name = "add_comments"

    def invoke(self, params, config=None):
        try:
            raise Exception(JIRA_MESSAGE)
        except Exception as e:
            raise ToolException(f"Error adding comment to Jira issue NOPE-1: {e}") from e


def test_runner_sends_the_exception_chain_as_debug_error(monkeypatch):
    import elitea_sdk.runtime.utils.toolkit_utils as toolkit_utils

    client = EliteAClient.__new__(EliteAClient)
    monkeypatch.setattr(client, "_validate_toolkit_config", lambda config: config, raising=False)
    monkeypatch.setattr(client, "get_llm", lambda model, config: object(), raising=False)
    monkeypatch.setattr(toolkit_utils, "instantiate_toolkit_with_client",
                        lambda *args, **kwargs: [_FailingTool()])

    outcome = client.test_toolkit_tool(
        toolkit_config={"toolkit_name": "jira", "settings": {}},
        tool_name="add_comments",
        tool_params={"issue_key": "NOPE-1", "comment": "hi"},
        llm_model="stub-model",
    )

    assert outcome["success"] is False
    _assert_readable(outcome["error"], "Error adding comment to Jira issue NOPE-1")
    assert "Traceback" in outcome["debug_error"]
    assert "direct cause" in outcome["debug_error"]
    assert outcome["debug_error"] != outcome["error"]
