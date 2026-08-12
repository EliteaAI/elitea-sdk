"""Regression tests for issue #6235.

EliteAClient.get_llm's adaptive-thinking branch (sonnet-5, opus-4-7+, opus-5)
did not pad `max_tokens` by the effort-based thinking budget the way the
legacy "enabled" thinking branch does. With a low/default `max_tokens`,
Anthropic could spend the whole budget on adaptive thinking and stop with
stop_reason="max_tokens" before emitting any text block, producing a
genuinely empty completion ("LLM returned an empty response").
"""

from unittest.mock import patch

import pytest

from elitea_sdk.runtime.clients.client import EliteAClient


def _make_client():
    client = EliteAClient.__new__(EliteAClient)
    client.base_url = "http://proxy"
    client.allm_path = "/anthropic"
    client.llm_path = "/openai"
    client.auth_token = "tok"
    client.project_id = "1"
    return client


class TestAdaptiveThinkingMaxTokensPadding:
    """Adaptive-thinking-only models must pad max_tokens by the effort budget."""

    @pytest.mark.parametrize(
        "effort,budget",
        [("low", 2048), ("medium", 4096), ("high", 9092)],
    )
    def test_adaptive_model_pads_max_tokens_by_effort_budget(self, effort, budget):
        client = _make_client()
        with patch("elitea_sdk.runtime.clients.client.ChatAnthropic") as mock_anthropic:
            client.get_llm(
                "eu.anthropic.claude-sonnet-5",
                {"reasoning_effort": effort, "max_tokens": 1000},
            )
        kwargs = mock_anthropic.call_args.kwargs
        assert kwargs["max_tokens"] == 1000 + budget
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kwargs["effort"] == effort

    def test_adaptive_model_pads_default_auto_max_tokens(self):
        """max_tokens: -1 (auto) is defaulted to 4000, then padded by the budget."""
        client = _make_client()
        with patch("elitea_sdk.runtime.clients.client.ChatAnthropic") as mock_anthropic:
            client.get_llm(
                "eu.anthropic.claude-sonnet-5",
                {"reasoning_effort": "medium", "max_tokens": -1},
            )
        kwargs = mock_anthropic.call_args.kwargs
        assert kwargs["max_tokens"] == 4000 + 4096

    def test_opus_4_7_adaptive_model_also_padded(self):
        client = _make_client()
        with patch("elitea_sdk.runtime.clients.client.ChatAnthropic") as mock_anthropic:
            client.get_llm(
                "claude-opus-4-7",
                {"reasoning_effort": "high", "max_tokens": 2000},
            )
        kwargs = mock_anthropic.call_args.kwargs
        assert kwargs["max_tokens"] == 2000 + 9092
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}

    def test_adaptive_model_without_reasoning_effort_unpadded(self):
        """No reasoning_effort requested: max_tokens must stay untouched."""
        client = _make_client()
        with patch("elitea_sdk.runtime.clients.client.ChatAnthropic") as mock_anthropic:
            client.get_llm("eu.anthropic.claude-sonnet-5", {"max_tokens": 1000})
        kwargs = mock_anthropic.call_args.kwargs
        assert kwargs["max_tokens"] == 1000
        assert "thinking" not in kwargs

    def test_legacy_enabled_thinking_model_still_padded(self):
        """Non-adaptive-only models keep the pre-existing 'enabled' thinking behavior."""
        client = _make_client()
        with patch("elitea_sdk.runtime.clients.client.ChatAnthropic") as mock_anthropic:
            client.get_llm(
                "claude-3-5-sonnet-20241022",
                {"reasoning_effort": "medium", "max_tokens": 1000},
            )
        kwargs = mock_anthropic.call_args.kwargs
        assert kwargs["max_tokens"] == 1000 + 4096
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
        assert kwargs["temperature"] == 1
