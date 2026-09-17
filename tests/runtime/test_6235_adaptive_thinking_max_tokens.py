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

    def test_adaptive_model_default_uses_model_max_without_padding(self):
        """Default uses the configured model maximum, which already includes thinking."""
        client = _make_client()
        with patch("elitea_sdk.runtime.clients.client.ChatAnthropic") as mock_anthropic:
            client.get_llm(
                "eu.anthropic.claude-sonnet-5",
                {
                    "reasoning_effort": "medium",
                    "max_tokens": -1,
                    "max_output_tokens": 64000,
                },
            )
        kwargs = mock_anthropic.call_args.kwargs
        assert kwargs["max_tokens"] == 64000

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


class TestAutoTotalOutputAllowance:
    def test_measured_chat_transport_overrides_worker_preference_only_for_auto(self, monkeypatch):
        import sys
        from types import SimpleNamespace
        worker = SimpleNamespace(descriptor=SimpleNamespace(config={'use_responses_api_for': ['terra']}))
        monkeypatch.setitem(sys.modules, 'tools', SimpleNamespace(this=SimpleNamespace(for_module=lambda _: worker)))
        client = _make_client()
        with patch('elitea_sdk.runtime.clients.client.ChatOpenAI') as native:
            client.get_llm('global.openai.gpt-5.6-terra', {'max_tokens': 32000,
                'routing_transport': 'chat_completions', 'routing_pin': 'signed-fixture',
                'routing_invocation_id': 'a'*64})
            assert native.call_args.kwargs['use_responses_api'] is False
            assert native.call_args.kwargs['max_tokens'] == 32000
            client.get_llm('global.openai.gpt-5.6-terra', {'max_tokens': 32000})
            assert native.call_args.kwargs['use_responses_api'] is True

    @pytest.mark.parametrize('name', ['eu.anthropic.claude-opus-5', 'eu.anthropic.claude-opus-4-8', 'eu.anthropic.claude-opus-4-7'])
    def test_native_default_contract_does_not_invent_effort_or_pad_output(self, name):
        client = _make_client()
        with patch('elitea_sdk.runtime.clients.client.ChatAnthropic') as native:
            client.get_llm(name, {'max_tokens': 32000, 'reasoning_effort': None,
                'routing_transport': 'anthropic_messages', 'routing_total_output_cap': True,
                'routing_pin': 'signed-fixture', 'routing_invocation_id': 'a'*64})
        assert native.call_args.kwargs['max_tokens'] == 32000
        assert 'thinking' not in native.call_args.kwargs
        assert 'effort' not in native.call_args.kwargs

    def test_native_calibration_contract_rejects_compatible_reinterpretation(self):
        client = _make_client()
        with patch('elitea_sdk.runtime.clients.client.ChatOpenAI') as native:
            with pytest.raises(ValueError, match='measured contract'):
                client.get_llm('eu.anthropic.claude-opus-5', {'max_tokens': 32000,
                    'openai_compatible': True, 'routing_transport': 'anthropic_messages',
                    'routing_pin': 'signed-fixture', 'routing_invocation_id': 'a'*64})
        native.assert_not_called()

    @pytest.mark.parametrize('name', ['global.anthropic.claude-sonnet-5', 'eu.anthropic.claude-opus-5', 'claude-3-5-sonnet-20241022'])
    def test_auto_total_cap_not_padded_twice_and_pin_constructs_native_headers(self, name):
        client = _make_client()
        with patch('elitea_sdk.runtime.clients.client.ChatAnthropic') as native:
            client.get_llm(name, {'reasoning_effort': 'medium', 'max_tokens': 8000,
                'routing_total_output_cap': True, 'routing_pin': 'signed-fixture', 'routing_invocation_id': 'a'*64})
        assert native.call_args.kwargs['max_tokens'] == 8000
        assert native.call_args.kwargs['default_headers']['X-Elitea-Routing-Pin'] == 'signed-fixture'

    def test_manual_model_has_no_auto_headers(self):
        client = _make_client()
        with patch('elitea_sdk.runtime.clients.client.ChatOpenAI') as native:
            client.get_llm('gpt-5.4', {'max_tokens': 8000})
        assert 'X-Elitea-Routing-Pin' not in native.call_args.kwargs.get('default_headers', {})


class TestAutoFactoryAndStructuredOwnership:
    def test_saved_auto_agent_constructs_deferred_client(self):
        from elitea_sdk.runtime.clients.routing import AutoChatModel
        client = _make_client()
        for name in ['_inject_project_context', '_inject_summarization', '_inject_context_editing',
                     '_inject_sensitive_tool_guard', '_inject_tool_exception_handler']:
            setattr(client, name, lambda *a, **k: None)
        with patch('elitea_sdk.runtime.clients.client.LangChainAssistant') as assistant:
            client.application(1, 1, runtime='nonrunnable', version_details={
                'agent_type': 'openai', 'llm_settings': {'selection': {'mode': 'auto'}}})
        assert isinstance(assistant.call_args.args[2], AutoChatModel)

    def test_auto_pipeline_and_conflicting_model_rejected_at_factory(self):
        client = _make_client()
        with pytest.raises(ValueError, match='Pipeline'):
            client.get_llm(None, {'selection': {'mode': 'auto'}, 'routing_surface': 'pipeline'})
        with pytest.raises(ValueError):
            client.get_llm('explicit', {'selection': {'mode': 'auto'}, 'routing_surface': 'agent'})

    def test_structured_schema_uses_actual_native_provider_adapter(self):
        from elitea_sdk.runtime.clients.routing import AutoChatModel
        from elitea_sdk.runtime.tools.llm import LLMNode
        from pydantic import BaseModel, JsonValue
        from unittest.mock import Mock
        class Result(BaseModel):
            values: list[JsonValue]
        native = Mock()
        with patch.object(LLMNode, '_is_anthropic_client', return_value=True), patch.object(LLMNode, '_is_anthropic_thinking_client', return_value=True):
            AutoChatModel._structured_native(native, Result, {'method': 'function_calling', 'include_raw': True})
        assert native.with_structured_output.call_args.kwargs['method'] == 'json_schema'
        assert native.with_structured_output.call_args.kwargs['include_raw'] is True
        assert native.with_structured_output.call_args.args[0]['properties']['values']['items'] != {}

    def test_llm_node_binding_channel_survives_before_model_compaction(self):
        from elitea_sdk.runtime.clients.routing import AutoChatModel, routing_scope_id
        from elitea_sdk.runtime.tools.llm import LLMNode
        from langchain_core.messages import AIMessage, HumanMessage
        from types import SimpleNamespace
        from unittest.mock import Mock
        auto = AutoChatModel(owner=SimpleNamespace(), settings={'selection': {'mode': 'auto'}})
        node = LLMNode(client=auto, name='ordinary_agent')
        config = {'configurable': {'thread_id': 'child', 'elitea_routing_run_id': 'server-run'}}
        scope = routing_scope_id(config)
        binding = {'scope_id': scope, 'pin': 'signed-fixture'}
        state = {'messages': [HumanMessage(content='task')], '_auto_routing': {scope: binding}}
        captured = {}
        def internal(self, incoming, actual_config, updates):
            captured.update(actual_config['configurable'])
            actual_config['configurable']['elitea_routing_sink']['binding'] = binding
            return {'messages': [AIMessage(content='complete')]}
        with patch.object(LLMNode, '_invoke_llm_internal', internal):
            result = node.invoke(state, config)
        assert captured['elitea_routing_checkpoint'] == binding
        assert result['_auto_routing'][scope] == binding
        assert 'elitea_routing_sink' not in config['configurable']


class TestAutoEmbeddedChildBoundary:
    def test_null_child_under_auto_raises_typed_error_before_graph_creation(self):
        from elitea_sdk.runtime.exceptions import AutoRoutingChildModelRequired
        client = _make_client()
        auto = client.get_llm(None, {'selection': {'mode': 'auto'}})
        with patch('elitea_sdk.runtime.clients.client.LangChainAssistant') as assistant:
            with pytest.raises(AutoRoutingChildModelRequired, match='no effective model after default resolution'):
                client.application(1, 1, llm=auto, runtime='nonrunnable', version_details={
                    'agent_type': 'agent', 'llm_settings': None})
        assistant.assert_not_called()

    def test_null_child_under_fixed_parent_retains_existing_client(self):
        client = _make_client()
        for name in ['_inject_project_context', '_inject_summarization', '_inject_context_editing',
                     '_inject_sensitive_tool_guard', '_inject_tool_exception_handler']:
            setattr(client, name, lambda *a, **k: None)
        with patch('elitea_sdk.runtime.clients.client.ChatOpenAI') as native, patch(
                'elitea_sdk.runtime.clients.client.LangChainAssistant') as assistant:
            client.application(1, 1, llm=native, runtime='nonrunnable', version_details={
                'agent_type': 'agent', 'llm_settings': None})
        assert assistant.call_args.args[2] is native
