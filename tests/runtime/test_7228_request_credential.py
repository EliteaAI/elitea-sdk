"""C-1009 #7228: the shape-only ``request_credential`` tool.

The model may only name a credential type, a title and a purpose; the tool pauses the run with a ``credential_form``
interrupt that carries those three fields and nothing else, and turns the server's record of the created credential
into a fixed sentence (any other key dropped). It is given to the model only when Main stamped ``credential_form`` on
an internal Elitea MCP toolkit the run token applies to. Every value here is made up.
"""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from elitea_sdk.runtime.langchain.langraph_agent import LangGraphAgentRunnable
from elitea_sdk.runtime.tools import request_credential as rc

INTERNAL_URL = "http://pylon_main:8080/app/6/mcp/elitea_core/applications"
SECRET = "made-up-value-7228-not-a-secret"


def run_tool(monkeypatch, resume_value, **args):
    captured = {}

    def fake_interrupt(payload):
        captured["payload"] = payload
        return resume_value

    monkeypatch.setattr(rc, "interrupt", fake_interrupt)
    result = rc.RequestCredentialTool().invoke(args or {"credential_type": "github", "title": "GitHub",
                                                       "purpose": "list repositories"})
    return result, captured["payload"]


@pytest.mark.parametrize("args", [
    {"credential_type": "github", "title": "GitHub", "value": SECRET},
    {"credential_type": "github", "title": "GitHub", "fields": {"token": SECRET}},
    {"credential_type": "github", "title": "GitHub", "purpose": "x" * 201},
    {"credential_type": "", "title": "GitHub"},
    {"title": "GitHub"},
])
def test_the_schema_refuses_anything_but_the_shape(args):
    with pytest.raises(ValidationError):
        rc.RequestCredentialInput(**args)


def test_a_value_cannot_be_smuggled_through_the_tool_call(monkeypatch):
    monkeypatch.setattr(rc, "interrupt", lambda payload: pytest.fail("the tool must not pause"))
    with pytest.raises(Exception):
        rc.RequestCredentialTool().invoke({"credential_type": "github", "title": "GitHub", "token": SECRET})


def test_the_interrupt_carries_the_shape_only(monkeypatch):
    _, payload = run_tool(monkeypatch, {"action": "cancel"})
    assert payload["guardrail_type"] == "credential_form" and payload["node_name"] == "request_credential"
    assert (payload["credential_type"], payload["title"], payload["purpose"]) == ("github", "GitHub",
                                                                                   "list repositories")
    assert payload["available_actions"] == ["created", "cancel"]
    assert payload["tool_args"] == {"credential_type": "github", "title": "GitHub", "purpose": "list repositories"}
    assert "tool_args_raw" not in payload and payload["interrupt_id"].startswith("hitl_")


def test_created_reads_as_the_fixed_sentence_built_from_the_servers_record(monkeypatch):
    record = {"reference_id": "33333333-3333-4333-8333-333333333333", "title": "GitHub", "credential_type": "github",
              "state": "validated", "configuration_uuid": "44444444-4444-4444-8444-444444444444",
              "configuration_title": "github_ab12cd", "value": SECRET, "token": SECRET}
    result, _ = run_tool(monkeypatch, {"action": "created", "value": record})
    assert result.startswith(
        "Credential 'GitHub' (github) is registered as reference 33333333-3333-4333-8333-333333333333, state "
        "validated; a grant for you is requested and an administrator must approve it before the agent can use it.")
    assert "github_ab12cd" in result and SECRET not in result


@pytest.mark.parametrize("resume_value", [{"action": "cancel"}, {"action": "created", "value": {}},
                                          {"action": "approve"}, "text", None])
def test_anything_but_a_created_record_reads_as_not_provided(monkeypatch, resume_value):
    result, _ = run_tool(monkeypatch, resume_value)
    assert result == "The user did not provide the credential."


def test_both_new_actions_are_a_resume():
    assert LangGraphAgentRunnable._is_hitl_resume({"hitl_action": "created"})
    assert LangGraphAgentRunnable._is_hitl_resume({"hitl_action": "cancel"})


# --- given only with Main's stamp and the run token ----------------------------------------------------------

def client(token="eio1.made-up", origins=("http://pylon_main:8080",)):
    return SimpleNamespace(internal_run_auth=SimpleNamespace(token=token, origins=origins, identity="who"))


def stamped(url=INTERNAL_URL, stamp=True):
    return {"type": "mcp", "name": "Elitea", "settings": {"url": url, "credential_form": stamp}}


@pytest.mark.parametrize("tools, run_client, expected", [
    ([stamped()], client(), True),
    ([stamped(stamp=False)], client(), False),                                   # no stamp
    ([stamped()], SimpleNamespace(), False),                                     # no run token (off, legacy)
    ([stamped()], client(token=None), False),
    ([stamped("https://mcp.example.com/app/6/mcp")], client(), False),           # another origin
    ([stamped("http://pylon_main:8080/other/route")], client(), False),          # not the internal MCP
    ([{"type": "internal_tool", "name": "ask_user", "settings": {"credential_form": True}}], client(), False),
])
def test_the_tool_is_given_only_with_the_stamp_on_an_internal_toolkit_of_the_run_token(tools, run_client, expected):
    assert rc.credential_form_requested(tools, run_client) is expected


def test_get_tools_adds_it_once_and_the_stamp_never_reaches_the_mcp_toolkit(monkeypatch):
    from elitea_sdk.runtime.toolkits import tools as toolkit_tools
    seen = []

    def fake_get_toolkit(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(get_tools=lambda: [])

    monkeypatch.setattr(toolkit_tools.McpToolkit, "get_toolkit", staticmethod(fake_get_toolkit))
    found = toolkit_tools.get_tools([stamped(), stamped(INTERNAL_URL + "2")], elitea_client=client())
    names = [getattr(tool, "name", None) for tool in found]
    assert names.count("request_credential") == 1
    assert seen and all("credential_form" not in kwargs for kwargs in seen)
    without = toolkit_tools.get_tools([stamped()], elitea_client=SimpleNamespace())
    assert "request_credential" not in [getattr(tool, "name", None) for tool in without]


def test_a_resumed_credential_form_is_not_shown_again_at_the_end_of_its_turn():
    interrupt = {"guardrail_type": "credential_form", "interrupt_id": "hitl_1", "tool_name": "request_credential"}
    assert LangGraphAgentRunnable._is_stale_sensitive_tool_interrupt(interrupt, {"interrupt_id": "hitl_1"}, True)
    assert not LangGraphAgentRunnable._is_stale_sensitive_tool_interrupt(interrupt, {"interrupt_id": "hitl_2"}, True)


def test_the_credential_form_resumes_through_the_clarifying_question_branch():
    """One branch, no new mechanism: the graph resume handler treats both guardrails alike, re-issuing the paused tool."""
    import inspect
    source = inspect.getsource(LangGraphAgentRunnable)
    assert "elif guardrail_type in ('clarifying_question', 'credential_form'):" in source
    assert "form_tool = 'request_credential' if guardrail_type == 'credential_form' else 'ask_user'" in source
