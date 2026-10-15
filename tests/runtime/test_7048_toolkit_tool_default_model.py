"""A toolkit tool run that names no model must use the project's default LLM, never a hardcoded one.

Re-breaks if test_toolkit_tool falls back to a fixed model name again (#7048: scheduled re-index
described images with gpt-4o-mini, which most projects do not have, and lost them silently).
"""

import pytest

from elitea_sdk.runtime.clients.client import EliteAClient

PROJECT_DEFAULT = {
    "name": "project-default-model",
    "project_id": 7,
    "openai_compatible": True,
    "default": True,
}
OTHER_MODEL = {"name": "other-model", "project_id": 7, "openai_compatible": False, "default": False}


class _StubTool:
    name = "index_data"

    def invoke(self, params, config=None):
        return {"status": "ok", "message": "1 file indexed"}


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def llm_calls(monkeypatch):
    import elitea_sdk.runtime.utils.toolkit_utils as toolkit_utils

    monkeypatch.setattr(
        toolkit_utils, "instantiate_toolkit_with_client", lambda *args, **kwargs: [_StubTool()]
    )
    return []


def make_client(monkeypatch, llm_calls, models_response):
    client = EliteAClient.__new__(EliteAClient)
    client.models_url = "http://test/api/v2/configurations/models/7?include_shared=true"
    client.headers = {}
    monkeypatch.setattr(client, "_validate_toolkit_config", lambda config: config, raising=False)
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: models_response, raising=False)

    def get_llm(model_name, model_config):
        llm_calls.append((model_name, dict(model_config)))
        return object()

    monkeypatch.setattr(client, "get_llm", get_llm, raising=False)
    return client


def run(client, **kwargs):
    return client.test_toolkit_tool(
        toolkit_config={"toolkit_name": "artifact", "settings": {}},
        tool_name="index_data",
        tool_params={"index_name": "demo"},
        **kwargs,
    )


def test_no_model_uses_the_project_default(monkeypatch, llm_calls):
    client = make_client(monkeypatch, llm_calls, _Response({"items": [OTHER_MODEL, PROJECT_DEFAULT]}))

    outcome = run(client)

    assert outcome["success"] is True
    assert outcome["llm_model"] == "project-default-model"
    model_name, model_config = llm_calls[0]
    assert model_name == "project-default-model"
    assert model_config["model_project_id"] == 7
    assert model_config["openai_compatible"] is True


def test_settings_without_a_model_keep_their_values(monkeypatch, llm_calls):
    client = make_client(monkeypatch, llm_calls, _Response({"items": [PROJECT_DEFAULT]}))

    run(client, llm_config={"max_tokens": 2048, "reasoning_effort": "low"})

    _, model_config = llm_calls[0]
    assert model_config["max_tokens"] == 2048
    assert model_config["reasoning_effort"] == "low"
    assert model_config["model_project_id"] == 7


def test_an_explicit_model_is_not_replaced(monkeypatch, llm_calls):
    client = make_client(monkeypatch, llm_calls, _Response({"items": [PROJECT_DEFAULT]}))

    outcome = run(client, llm_model="chosen-model")

    assert outcome["success"] is True
    assert llm_calls[0][0] == "chosen-model"


def test_auto_selection_is_not_given_a_concrete_model(monkeypatch, llm_calls):
    client = make_client(monkeypatch, llm_calls, _Response({"items": [PROJECT_DEFAULT]}))

    run(client, llm_config={"selection": {"mode": "auto"}})

    assert llm_calls[0][0] is None


def test_a_project_without_models_fails_visibly(monkeypatch, llm_calls):
    client = make_client(monkeypatch, llm_calls, _Response({"items": []}))

    outcome = run(client)

    assert outcome["success"] is False
    assert "no default LLM model is available" in outcome["error"]
    assert llm_calls == []


def test_a_failed_model_lookup_fails_visibly(monkeypatch, llm_calls):
    client = make_client(monkeypatch, llm_calls, _Response({}, status_code=503))

    outcome = run(client)

    assert outcome["success"] is False
    assert "Could not look up the project's default LLM model" in outcome["error"]
    assert llm_calls == []


def test_no_settings_keep_the_low_temperature_for_a_plain_default(monkeypatch, llm_calls):
    client = make_client(monkeypatch, llm_calls, _Response({"items": [PROJECT_DEFAULT]}))

    run(client)

    _, model_config = llm_calls[0]
    assert model_config["temperature"] == 0.1
    assert model_config["max_tokens"] == 1024


def test_a_reasoning_default_gets_no_custom_temperature(monkeypatch, llm_calls):
    reasoning_default = {**PROJECT_DEFAULT, "supports_reasoning": True}
    client = make_client(monkeypatch, llm_calls, _Response({"items": [reasoning_default]}))

    run(client)

    _, model_config = llm_calls[0]
    assert "temperature" not in model_config
    assert model_config["max_tokens"] == 1024
