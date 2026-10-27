"""C-1009 (#7229 item 1, C9-SEC-30): a governed run's only platform identity is its run token.

- The sandbox preamble never hands the run token to generated code (it never leaves the worker); a personal token or
  session is handed over as before.
- A sub-agent version the platform refuses to a governed run (``403 {code, message}``) ends the run with that governed
  reason instead of being skipped; a plain 403 is still the editor-access error.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from elitea_sdk.runtime.clients.client import EliteAClient
from elitea_sdk.runtime.credential_slots import GovernedCredentialError
from elitea_sdk.runtime.exceptions import AgentAccessDeniedError
from elitea_sdk.runtime.toolkits import tools as tools_mod
from elitea_sdk.runtime.toolkits.tools import get_tools
from elitea_sdk.runtime.tools.sandbox import PyodideSandboxTool

RUN_TOKEN = "eio1." + "a" * 40 + "." + "b" * 43


def _sandbox(auth_token, run_token=None):
    client = SimpleNamespace(base_url="http://pylon_main:8080", project_id=2, auth_token=auth_token, auth_session=None,
                             session_cookie_name=None)
    if run_token:
        client.internal_run_auth = SimpleNamespace(token=run_token, origins=("http://pylon_main:8080",), identity="i")
    return PyodideSandboxTool.model_construct(elitea_client=client)


def test_the_run_token_is_never_written_into_generated_code_and_the_client_says_why():
    prepared = _sandbox(RUN_TOKEN, RUN_TOKEN)._prepare_pyodide_input(
        "try:\n    elitea_client.list_artifacts('b')\nexcept PermissionError as exc:\n    said = str(exc)\n")
    assert RUN_TOKEN not in prepared and "SandboxClient(base_url" not in prepared
    namespace = {}
    exec(prepared, namespace)  # pylint: disable=W0122
    assert "not available to code in governed runs" in namespace["said"]


def test_a_personal_token_is_still_handed_to_the_sandbox_client_as_before():
    prepared = _sandbox("pat-value", RUN_TOKEN)._prepare_pyodide_input("print(elitea_client)")
    assert "auth_token='pat-value'" in prepared and RUN_TOKEN not in prepared


class _Response:
    def __init__(self, status, body):
        self.status_code, self._body, self.ok, self.text = status, body, 200 <= status < 300, str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _client(response):
    client = EliteAClient(base_url="http://pylon_main:8080", project_id=1, auth_token=RUN_TOKEN)
    client.internal_run_auth = SimpleNamespace(token=RUN_TOKEN, origins=("http://pylon_main:8080",), identity="i")
    client._request = lambda *args, **kwargs: response  # pylint: disable=W0212
    return client


@pytest.mark.parametrize("code", ["DENIED_DEVELOPMENT_CHILD", "CREDENTIAL_REFERENCE_REQUIRED"])
def test_a_governed_refusal_of_a_fetched_version_carries_its_reason(code):
    with pytest.raises(GovernedCredentialError) as err:
        _client(_Response(403, {"code": code, "message": "anything Main says"})).get_app_version_details(8, 80)
    assert err.value.reason_code == code and "the run was not started" in err.value.message


@pytest.mark.parametrize("body", [{"error": "no access"}, ValueError("not json"), {"code": "lower_case"}, ["x"]])
def test_any_other_403_is_still_the_editor_access_error(body):
    with pytest.raises(AgentAccessDeniedError):
        _client(_Response(403, body)).get_app_version_details(8, 80)


class _FakeClient:
    def get_mcp_toolkits(self):
        return []


def test_a_sub_agent_refused_by_governance_ends_the_run_instead_of_being_skipped():
    def refused(client, application_id, application_version_id, **kwargs):
        raise GovernedCredentialError("DENIED_DEVELOPMENT_CHILD")

    tool = {"type": "application", "name": "child", "project_id": 1, "agent_type": "agent",
            "settings": {"application_id": 8, "application_version_id": 80}}
    with patch.object(tools_mod.ApplicationToolkit, "get_toolkit", side_effect=refused):
        with pytest.raises(GovernedCredentialError) as err:
            get_tools([tool], elitea_client=_FakeClient())
    assert err.value.reason_code == "DENIED_DEVELOPMENT_CHILD"


def test_a_sub_agent_that_fails_otherwise_is_still_skipped():
    def broken(client, application_id, application_version_id, **kwargs):
        raise RuntimeError("deleted agent")

    tool = {"type": "application", "name": "child", "project_id": 1, "agent_type": "agent",
            "settings": {"application_id": 8, "application_version_id": 80}}
    with patch.object(tools_mod.ApplicationToolkit, "get_toolkit", side_effect=broken):
        assert get_tools([tool], elitea_client=_FakeClient()) == []


def test_a_legacy_client_keeps_the_editor_access_error_for_any_403():
    client = EliteAClient(base_url="http://pylon_main:8080", project_id=1, auth_token="pat-value")
    client._request = lambda *args, **kwargs: _Response(403, {"code": "DENIED_DEVELOPMENT_CHILD"})  # pylint: disable=W0212
    with pytest.raises(AgentAccessDeniedError):
        client.get_app_version_details(8, 80)


def test_a_governed_refusal_inside_a_tool_call_is_a_signal_not_prose():
    from elitea_sdk.runtime.middleware.tool_exception_handler import reraise_signal_exceptions
    with pytest.raises(GovernedCredentialError):
        reraise_signal_exceptions(GovernedCredentialError("DENIED_DEVELOPMENT_CHILD"))


def test_a_governed_refusal_is_not_skipped_in_debug_mode_either():
    def refused(client, application_id, application_version_id, **kwargs):
        raise GovernedCredentialError("DENIED_DEVELOPMENT_CHILD")

    tool = {"type": "application", "name": "child", "project_id": 1, "agent_type": "agent",
            "settings": {"application_id": 8, "application_version_id": 80}}
    with patch.object(tools_mod.ApplicationToolkit, "get_toolkit", side_effect=refused):
        with pytest.raises(GovernedCredentialError):
            get_tools([tool], elitea_client=_FakeClient(), debug_mode=True)



def test_a_public_sub_agent_refused_to_a_governed_run_is_said_not_dropped():
    with pytest.raises(GovernedCredentialError) as err:
        _client(_Response(403, {"error": "INTERNAL_TOKEN_INVALID"})).get_public_app_details(12)
    assert err.value.reason_code == "PUBLIC_AGENT_NOT_ADMITTED" and "another project" in err.value.message
    legacy = EliteAClient(base_url="http://pylon_main:8080", project_id=1, auth_token="pat-value")
    legacy._request = lambda *args, **kwargs: _Response(403, {"error": "x"})  # pylint: disable=W0212
    from elitea_sdk.runtime.clients.client import ApiDetailsRequestError
    with pytest.raises(ApiDetailsRequestError):
        legacy.get_public_app_details(12)


def test_the_refusal_names_the_sub_agent_refused_and_survives_a_task_boundary():
    import pickle  # pylint: disable=C0415

    def refused(client, application_id, application_version_id, **kwargs):
        raise GovernedCredentialError("PUBLIC_AGENT_NOT_ADMITTED")

    tool = {"type": "application", "name": "public helper", "project_id": 9, "agent_type": "agent",
            "settings": {"application_id": 12, "application_version_id": 120}}
    with patch.object(tools_mod.ApplicationToolkit, "get_toolkit", side_effect=refused):
        with pytest.raises(GovernedCredentialError) as err:
            get_tools([tool], elitea_client=_FakeClient())
    assert err.value.subagent == "public helper" and "(sub-agent \u201cpublic helper\u201d)" in err.value.message
    again = pickle.loads(pickle.dumps(err.value))
    assert (again.reason_code, again.subagent, again.message) == (err.value.reason_code, "public helper", err.value.message)


class _Refused(Exception):
    """The shape of a model client's 4xx error (openai / anthropic: ``status_code`` and ``body``)."""

    def __init__(self, status_code, code):
        super().__init__(f"Error code: {status_code} - {{'error': '{code}'}}")
        self.status_code, self.body = status_code, {"error": code}


@pytest.mark.parametrize("status_code,code", [(403, "CONTEXT_NOT_LIVE"), (401, "INTERNAL_TOKEN_EXPIRED"),
                                              (403, "INTERNAL_TOKEN_INVALID")])
def test_a_pipeline_model_call_refused_on_the_run_token_ends_the_run(status_code, code):
    from elitea_sdk.runtime.tools.llm import LLMNode  # pylint: disable=C0415
    from elitea_sdk.runtime.utils.internal_run_auth import run_token_refusal  # pylint: disable=C0415

    error = _Refused(status_code, code)
    assert run_token_refusal(error) == code
    node = LLMNode(name="llm_1", description="test", client=SimpleNamespace(model_name="m", _llm_type="fake"))
    with patch.object(LLMNode, "_invoke_llm_internal", side_effect=error):
        with pytest.raises(_Refused):
            node.invoke({"messages": []})


def test_other_model_errors_still_become_the_node_answer():
    from elitea_sdk.runtime.tools.llm import LLMNode  # pylint: disable=C0415
    from elitea_sdk.runtime.utils.internal_run_auth import run_token_refusal  # pylint: disable=C0415

    assert run_token_refusal(_Refused(500, "CONTEXT_NOT_LIVE")) is None
    assert run_token_refusal(_Refused(403, "budget")) is None
    node = LLMNode(name="llm_1", description="test", client=SimpleNamespace(model_name="m", _llm_type="fake"))
    with patch.object(LLMNode, "_invoke_llm_internal", side_effect=_Refused(403, "PROVIDER_DENIED")):
        result = node.invoke({"messages": []})
    assert result["messages"][-1].response_metadata.get("llm_node_error") is True
