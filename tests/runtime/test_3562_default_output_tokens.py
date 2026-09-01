"""Default output-token behavior for EliteaAI/elitea_issues#3562."""

from unittest.mock import patch

import pytest

from elitea_sdk.runtime.clients.client import EliteAClient


def _make_client():
    client = EliteAClient.__new__(EliteAClient)
    client.base_url = "http://proxy"
    client.allm_path = "/anthropic"
    client.llm_path = "/openai"
    client.auth_token = "tok"
    client.project_id = 1
    client.api_extra_headers = {}
    return client


@pytest.mark.parametrize("default_value", [None, -1])
def test_openai_compatible_default_omits_provider_output_limit(default_value):
    client = _make_client()
    with patch("elitea_sdk.runtime.clients.client.ChatOpenAI") as mock_openai:
        client.get_llm("gpt-4o", {"max_tokens": default_value})

    assert "max_tokens" not in mock_openai.call_args.kwargs


def test_openai_compatible_custom_limit_is_preserved():
    client = _make_client()
    with patch("elitea_sdk.runtime.clients.client.ChatOpenAI") as mock_openai:
        client.get_llm("gpt-4o", {"max_tokens": 1234})

    assert mock_openai.call_args.kwargs["max_tokens"] == 1234


def test_custom_proxy_default_preserves_provider_and_session_headers():
    client = _make_client()
    client.auth_token = None
    client.auth_session = "session-ref"
    client.session_cookie_name = "auth_session_id"
    client.api_extra_headers = {"X-Databricks-Scope": "workspace"}

    with patch("elitea_sdk.runtime.clients.client.ChatOpenAI") as mock_openai:
        client.get_llm(
            "custom-databricks-model",
            {"max_tokens": -1, "openai_compatible": True},
        )

    kwargs = mock_openai.call_args.kwargs
    assert "max_tokens" not in kwargs
    assert kwargs["api_key"] == "session"
    assert kwargs["default_headers"] == {
        "X-Databricks-Scope": "workspace",
        "Cookie": "auth_session_id=session-ref",
    }


def test_native_anthropic_default_uses_configured_model_maximum():
    client = _make_client()
    with patch("elitea_sdk.runtime.clients.client.ChatAnthropic") as mock_anthropic:
        client.get_llm(
            "claude-sonnet-4-6",
            {"max_tokens": -1, "max_output_tokens": 64000},
        )

    assert mock_anthropic.call_args.kwargs["max_tokens"] == 64000


def test_native_anthropic_default_resolves_missing_capability_from_model_configuration():
    client = _make_client()
    client.get_available_models = lambda: [
        {"name": "claude-sonnet-4-6", "project_id": 7, "max_output_tokens": 32000},
    ]
    with patch("elitea_sdk.runtime.clients.client.ChatAnthropic") as mock_anthropic:
        client.get_llm(
            "claude-sonnet-4-6",
            {"max_tokens": None, "model_project_id": 7},
        )

    assert mock_anthropic.call_args.kwargs["max_tokens"] == 32000


def test_native_anthropic_default_fails_clearly_without_model_capability():
    client = _make_client()
    client.get_available_models = lambda: []

    with pytest.raises(ValueError, match="configured max_output_tokens"):
        client.get_llm("claude-sonnet-4-6", {"max_tokens": -1})


def test_native_reasoning_default_does_not_pad_past_model_maximum():
    client = _make_client()
    with patch("elitea_sdk.runtime.clients.client.ChatAnthropic") as mock_anthropic:
        client.get_llm(
            "eu.anthropic.claude-sonnet-5",
            {
                "max_tokens": -1,
                "max_output_tokens": 64000,
                "reasoning_effort": "medium",
            },
        )

    assert mock_anthropic.call_args.kwargs["max_tokens"] == 64000
