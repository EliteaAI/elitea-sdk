"""A Dataverse tool call without a saved login reaches the chat as an mcp_auth interrupt.

The test sends the real Dataverse tool through the agent graph, so the tool node
(llm.py) turns the sign-in signal into the payload the Login popup reads.
"""
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
    "configuration_project_id": 77,
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
        # The authorize decision closes the first call with a tool message; the model then asks again
        if sum(isinstance(m, ToolMessage) for m in messages) >= 2:
            return AIMessage(content="signed-in")
        return AIMessage(content="", tool_calls=[{
            "name": "who_am_i", "args": {}, "id": f"call-dataverse-{len(messages)}", "type": "tool_call",
        }])


class _Runtime:
    def get_mcp_toolkits(self):
        return []


def _runnable(memory, tokens=None):
    """What one request builds: toolkit from the saved config plus the request tokens, then the agent graph."""
    kit = DataverseToolkit.get_toolkit(
        dataverse_configuration=dict(_SETTINGS), toolkit_name="crm", tokens=tokens or {})
    loaded = kit.get_tools()
    _inject_toolkit_id({"id": _TOOLKIT_ID, "type": "dataverse"}, loaded)
    who_am_i = next(t for t in loaded if t.name == "who_am_i")
    return Assistant(
        elitea=_Runtime(),
        data={"instructions": "Use tools", "tools": [], "meta": {}},
        client=_CallsWhoAmI(),
        tools=[who_am_i],
        memory=memory,
        app_type="predict",
    ).runnable()


_CONFIG = {"configurable": {"thread_id": "dataverse-sign-in"}}
_ASK = {"messages": [HumanMessage(content="Who am I in Dataverse?")]}


def _run_to_interrupt():
    return _runnable(MemorySaver()).invoke(_ASK, config=_CONFIG)


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
    assert metadata["configuration_project_id"] == 77
    assert metadata["provided_settings"]["mcp_client_id"] == "client-abc"
    assert metadata["provided_settings"]["mcp_client_secret"] == "****"
    assert metadata["provided_settings"]["credential_type"] == "dataverse"


def test_dataverse_interrupt_payload_holds_no_secret():
    interrupt = _run_to_interrupt()["hitl_interrupt"]
    assert "secret-not-sent" not in str(interrupt)


def test_authorize_resume_uses_the_token_of_the_resumed_request(monkeypatch):
    calls = []

    class _Reply:
        status_code = 200
        ok = True
        text = ""

        def json(self):
            return {"UserId": "u-1"}

    def fake_request(method, url, headers=None, json=None, timeout=None):
        calls.append(headers["Authorization"])
        return _Reply()

    monkeypatch.setattr(dv.requests, "request", fake_request)
    dv._token_cache.clear()
    memory = MemorySaver()
    paused = _runnable(memory).invoke(_ASK, config=_CONFIG)
    assert paused["execution_finished"] is False
    assert calls == []

    tokens = {f"uuid-1:{_AUTHORITY}": {"access_token": "signed-in-at"}}
    resumed = _runnable(memory, tokens).invoke(
        {"mcp_auth_resume": True, "mcp_auth_action": "authorize"}, config=_CONFIG)

    assert resumed["execution_finished"] is True
    assert "hitl_interrupt" not in resumed or not resumed["hitl_interrupt"]
    assert resumed["output"] == "signed-in"
    assert calls == ["Bearer signed-in-at"]
    dv._token_cache.clear()


def test_authorize_resume_without_a_token_asks_again():
    memory = MemorySaver()
    assert _runnable(memory).invoke(_ASK, config=_CONFIG)["execution_finished"] is False
    again = _runnable(memory).invoke(
        {"mcp_auth_resume": True, "mcp_auth_action": "authorize"}, config=_CONFIG)
    assert again["execution_finished"] is False
    assert again["hitl_interrupt"]["guardrail_type"] == "mcp_auth"
