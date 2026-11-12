"""A Dataverse tool call without a saved login reaches the chat as an mcp_auth interrupt.

The test sends the real Dataverse tool through the agent graph, so the tool node
(llm.py) turns the sign-in signal into the payload the Login popup reads. The resume
tests show that the run after "Authorize" reads the new token because the platform
builds the toolkit again for every run, and that a toolkit built before the login
asks again.
"""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.configurations import dataverse as dv
from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.tools import _inject_toolkit_id
from elitea_sdk.tools.dataverse import DataverseToolkit

_TOOLKIT_ID = 4711
_TENANT = "11111111-2222-3333-4444-555555555555"
_AUTHORITY = f"https://login.microsoftonline.com/{_TENANT}"
_SETTINGS = {
    "url": "https://onecrmepam.crm4.dynamics.com",
    "tenant_id": _TENANT,
    "client_id": "client-abc",
    "client_secret": "secret-not-sent",
    "refresh_token": None,
    "configuration_uuid": "uuid-1",
}


class _CallsWhoAmI:
    temperature = 0
    max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        return AIMessage(content="", tool_calls=[{
            "name": "who_am_i", "args": {}, "id": "call-dataverse", "type": "tool_call",
        }])


class _Runtime:
    def get_mcp_toolkits(self):
        return []


def _who_am_i(**toolkit_kwargs):
    kit = DataverseToolkit.get_toolkit(dataverse_configuration=dict(_SETTINGS), toolkit_name="crm", **toolkit_kwargs)
    loaded = kit.get_tools()
    _inject_toolkit_id({"id": _TOOLKIT_ID, "type": "dataverse"}, loaded)
    return next(t for t in loaded if t.name == "who_am_i")


def _runnable(client, who_am_i, memory):
    return Assistant(
        elitea=_Runtime(),
        data={"instructions": "Use tools", "tools": [], "meta": {}},
        client=client,
        tools=[who_am_i],
        memory=memory,
        app_type="predict",
    ).runnable()


def _run_to_interrupt():
    return _runnable(_CallsWhoAmI(), _who_am_i(), MemorySaver()).invoke(
        {"messages": [HumanMessage(content="Who am I in Dataverse?")]},
        config={"configurable": {"thread_id": "dataverse-sign-in"}},
    )


def test_dataverse_call_without_a_login_interrupts_for_sign_in():
    result = _run_to_interrupt()
    assert result["execution_finished"] is False
    interrupt = result["hitl_interrupt"]
    assert interrupt["guardrail_type"] == "mcp_auth"
    assert interrupt["available_actions"] == ["authorize", "skip"]
    assert interrupt["tool_name"] == "who_am_i"
    assert interrupt["toolkit_type"] == "dataverse"
    assert interrupt["toolkit_id"] == _TOOLKIT_ID
    assert interrupt["server_url"] == _AUTHORITY
    metadata = interrupt["resource_metadata"]
    assert metadata["oauth_authorization_server"]["token_endpoint"] == f"{_AUTHORITY}/oauth2/v2.0/token"
    assert metadata["configuration_uuid"] == "uuid-1"
    assert metadata["provided_settings"]["mcp_client_id"] == "client-abc"


def test_dataverse_interrupt_payload_holds_no_secret():
    interrupt = _run_to_interrupt()["hitl_interrupt"]
    assert "secret-not-sent" not in str(interrupt)


class _RetriesAfterSignIn(_CallsWhoAmI):
    """Asks for who_am_i, calls it again once the sign-in is authorized, then answers."""

    def invoke(self, messages, config=None):
        last = messages[-1]
        if isinstance(last, ToolMessage) and '"authorized"' in str(last.content):
            return AIMessage(content="", tool_calls=[{
                "name": "who_am_i", "args": {}, "id": "call-retry", "type": "tool_call",
            }])
        if isinstance(last, ToolMessage):
            return AIMessage(content=f"signed in: {last.content}")
        return super().invoke(messages, config)


class _Response:
    status_code = 200
    ok = True

    def json(self):
        return {"UserId": "user-1"}


def _resume(runnable, thread_id):
    return runnable.invoke(
        {"mcp_auth_resume": True, "mcp_auth_action": "authorize"},
        config={"configurable": {"thread_id": thread_id}},
    )


def test_resume_after_authorize_reads_the_new_token(monkeypatch):
    api_calls = []
    monkeypatch.setattr(dv.requests, "request", lambda method, url, headers=None, **kw: api_calls.append(
        {"url": url, "authorization": headers["Authorization"]}) or _Response())
    memory = MemorySaver()
    thread = "dataverse-resume-new-token"
    paused = _runnable(_RetriesAfterSignIn(), _who_am_i(), memory).invoke(
        {"messages": [HumanMessage(content="Who am I in Dataverse?")]},
        config={"configurable": {"thread_id": thread}},
    )
    assert paused["hitl_interrupt"]["guardrail_type"] == "mcp_auth"
    assert api_calls == []

    token_key = f"{_SETTINGS['configuration_uuid']}:{_AUTHORITY}"
    rebuilt = _who_am_i(tokens={token_key: {"access_token": "token-after-login"}})
    resumed = _resume(_runnable(_RetriesAfterSignIn(), rebuilt, memory), thread)

    assert resumed["execution_finished"] is True
    assert "hitl_interrupt" not in resumed or not resumed["hitl_interrupt"]
    assert json.loads(resumed["output"].removeprefix("signed in: ")) == {"UserId": "user-1"}
    assert [c["authorization"] for c in api_calls] == ["Bearer token-after-login"]


def test_resume_with_the_toolkit_built_before_the_login_asks_again(monkeypatch):
    api_calls = []
    monkeypatch.setattr(dv.requests, "request", lambda *a, **kw: api_calls.append(a) or _Response())
    memory = MemorySaver()
    thread = "dataverse-resume-old-toolkit"
    old_tool = _who_am_i()
    _runnable(_RetriesAfterSignIn(), old_tool, memory).invoke(
        {"messages": [HumanMessage(content="Who am I in Dataverse?")]},
        config={"configurable": {"thread_id": thread}},
    )

    again = _resume(_runnable(_RetriesAfterSignIn(), old_tool, memory), thread)

    assert again["execution_finished"] is False
    assert again["hitl_interrupt"]["guardrail_type"] == "mcp_auth"
    assert api_calls == []
