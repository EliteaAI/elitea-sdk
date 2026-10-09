"""#6819: get_llm takes the Anthropic thinking mode from the model row's thinking_type.

The name tuple that used to decide adaptive-vs-enabled is only the fallback for rows
whose field is still null, so an admin's stored choice is what reaches the provider
(a Fable row marked always_on no longer gets the budget shape that returns 400).
"""
from unittest.mock import patch

import pytest

from elitea_sdk.runtime.clients.client import EliteAClient, resolve_thinking_type

ADAPTIVE = {"type": "adaptive", "display": "summarized"}


def _client():
    client = EliteAClient.__new__(EliteAClient)
    client.base_url = "http://proxy"
    client.allm_path = "/anthropic"
    client.llm_path = "/openai"
    client.auth_token = "tok"
    client.project_id = "1"
    return client


def _anthropic_kwargs(model_name, model_config):
    with patch("elitea_sdk.runtime.clients.client.ChatAnthropic") as chat_anthropic:
        _client().get_llm(model_name, model_config)
    return chat_anthropic.call_args.kwargs


class TestResolveThinkingType:
    @pytest.mark.parametrize("configured", ["adaptive", "enabled", "always_on"])
    def test_configured_value_wins_over_the_name(self, configured):
        assert resolve_thinking_type("claude-haiku-4-5", configured) == configured
        assert resolve_thinking_type("claude-opus-5", configured) == configured

    @pytest.mark.parametrize("model_name,expected", [
        ("eu.anthropic.claude-opus-4-7", "adaptive"),
        ("claude-opus-4.8", "adaptive"),
        ("global.anthropic.claude-sonnet-5", "adaptive"),
        ("anthropic.claude_opus_5", "adaptive"),
        ("claude-sonnet-4-6", "enabled"),
        ("eu.anthropic.claude-haiku-4-5-20251001-v1:0", "enabled"),
        ("claude-fable-5-1", "enabled"),
    ])
    @pytest.mark.parametrize("configured", [None, "", "something-else"])
    def test_null_field_falls_back_to_the_name_tuple(self, model_name, expected, configured):
        assert resolve_thinking_type(model_name, configured) == expected


class TestGetLlmUsesConfiguredThinkingType:
    @pytest.mark.parametrize("thinking_type", ["adaptive", "always_on"])
    def test_fable_row_with_stored_type_gets_adaptive_thinking(self, thinking_type):
        kwargs = _anthropic_kwargs("claude-fable-5-1", {
            "reasoning_effort": "high", "max_tokens": 1000, "thinking_type": thinking_type,
        })
        assert kwargs["thinking"] == ADAPTIVE
        assert kwargs["effort"] == "high"
        assert kwargs["temperature"] is None
        assert kwargs["max_tokens"] == 1000 + 9092

    def test_stored_enabled_overrides_an_adaptive_looking_name(self):
        kwargs = _anthropic_kwargs("claude-opus-5-custom", {
            "reasoning_effort": "low", "max_tokens": 1000, "thinking_type": "enabled",
        })
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert kwargs["temperature"] == 1
        assert "effort" not in kwargs
        assert kwargs["max_tokens"] == 1000 + 2048

    def test_fable_row_without_the_field_keeps_todays_budget_shape(self):
        kwargs = _anthropic_kwargs("claude-fable-5-1", {"reasoning_effort": "high", "max_tokens": 1000})
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 9092}

    def test_tuple_model_without_the_field_keeps_todays_adaptive_shape(self):
        kwargs = _anthropic_kwargs("eu.anthropic.claude-opus-4-7", {"reasoning_effort": "medium", "max_tokens": -1,
                                                                     "max_output_tokens": 32000})
        assert kwargs["thinking"] == ADAPTIVE
        assert kwargs["effort"] == "medium"
        assert kwargs["max_tokens"] == 32000

    def test_supported_efforts_and_default_effort_are_ignored_by_the_client(self):
        kwargs = _anthropic_kwargs("claude-sonnet-4-6", {
            "reasoning_effort": "high", "max_tokens": 1000, "thinking_type": "adaptive",
            "supported_efforts": ["low", "medium", "high", "max"], "default_effort": "high",
        })
        assert kwargs["thinking"] == ADAPTIVE
        assert kwargs["effort"] == "high"
        assert "supported_efforts" not in kwargs and "default_effort" not in kwargs

    def test_no_effort_means_no_thinking_regardless_of_the_field(self):
        kwargs = _anthropic_kwargs("claude-fable-5-1", {"max_tokens": 1000, "thinking_type": "always_on"})
        assert "thinking" not in kwargs and "effort" not in kwargs
