"""Selected model owner reaches the gateway (EliteaAI/elitea_issues#6999)."""

from unittest.mock import patch

from elitea_sdk.runtime.clients.client import EliteAClient, MODEL_PROJECT_HEADER


def _make_client():
    client = EliteAClient.__new__(EliteAClient)
    client.base_url = "http://proxy"
    client.allm_path = "/anthropic"
    client.llm_path = "/openai"
    client.auth_token = "tok"
    client.project_id = 112
    client.api_extra_headers = {}
    return client


def test_openai_sends_the_selected_owner():
    client = _make_client()
    with patch("elitea_sdk.runtime.clients.client.ChatOpenAI") as mock_openai:
        client.get_llm("gpt-4o", {"model_project_id": 1})

    assert mock_openai.call_args.kwargs["default_headers"][MODEL_PROJECT_HEADER] == "1"


def test_anthropic_sends_the_selected_owner():
    client = _make_client()
    with patch("elitea_sdk.runtime.clients.client.ChatAnthropic") as mock_anthropic:
        client.get_llm("anthropic.claude-sonnet-5", {"model_project_id": 1, "max_tokens": 1024})

    assert mock_anthropic.call_args.kwargs["default_headers"][MODEL_PROJECT_HEADER] == "1"


def test_no_owner_sends_no_header():
    client = _make_client()
    with patch("elitea_sdk.runtime.clients.client.ChatOpenAI") as mock_openai:
        client.get_llm("gpt-4o", {"max_tokens": 10})
    with patch("elitea_sdk.runtime.clients.client.ChatAnthropic") as mock_anthropic:
        client.get_llm("anthropic.claude-sonnet-5", {"max_tokens": 1024})

    assert MODEL_PROJECT_HEADER not in (mock_openai.call_args.kwargs.get("default_headers") or {})
    assert MODEL_PROJECT_HEADER not in mock_anthropic.call_args.kwargs["default_headers"]
