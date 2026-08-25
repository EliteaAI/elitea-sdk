"""#6167: exception_handling_enabled is gone. TEHM is now unconditional on every
call path that builds an agent, and the low-tier LLM it may need is resolved lazily
so construction itself never pays for it."""

import inspect
from unittest.mock import MagicMock, patch

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from elitea_sdk.runtime.clients import client as client_module
from elitea_sdk.runtime.middleware.tool_exception_handler import (
    ToolExceptionHandlerMiddleware,
)

VERSION_DETAILS = {
    'agent_type': 'agent',
    'instructions': 'test',
    'tools': [],
    'variables': [],
    'meta': {},
    'llm_settings': {
        'model_name': 'fake-model',
        'max_tokens': 1000,
        'temperature': 0,
        'reasoning_effort': None,
    },
}


def _make_client():
    """EliteAClient without __init__ — same idiom as test_nested_application_hitl.py."""
    client = client_module.EliteAClient.__new__(client_module.EliteAClient)
    client.get_llm = lambda *args, **kwargs: object()
    client._inject_summarization = lambda *args, **kwargs: None
    client._inject_context_editing = lambda *args, **kwargs: None
    client._inject_sensitive_tool_guard = lambda *args, **kwargs: None
    return client


class _CapturingAssistant:
    """Stand-in for LangChainAssistant: records kwargs, no real agent gets built."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def runnable(self):
        return self


def _capture_application_middleware(client, **overrides):
    with patch.object(client_module, 'LangChainAssistant', _CapturingAssistant):
        assistant = client.application(
            application_id=1,
            application_version_id=2,
            version_details=VERSION_DETAILS,
            runtime='nonrunnable',
            **overrides,
        )
    return assistant.kwargs['middleware']


class TestNoFlagLeftToPass:
    """The parameter is gone, not just defaulted — passing it must raise TypeError."""

    def test_application_signature_has_no_flag(self):
        params = inspect.signature(client_module.EliteAClient.application).parameters
        assert 'exception_handling_enabled' not in params

    def test_predict_agent_signature_has_no_flag(self):
        params = inspect.signature(client_module.EliteAClient.predict_agent).parameters
        assert 'exception_handling_enabled' not in params


class TestUnconditionalInjection:
    """Both entry points must produce exactly one TEHM, unconditionally."""

    def test_application_injects_exactly_one_tehm(self):
        middleware = _capture_application_middleware(_make_client())

        assert sum(isinstance(mw, ToolExceptionHandlerMiddleware) for mw in middleware) == 1

    def test_predict_agent_injects_exactly_one_tehm(self):
        client = _make_client()
        captured = {}

        class Capturing(_CapturingAssistant):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

        with patch.object(client_module, 'LangChainAssistant', Capturing):
            client.predict_agent(llm=object())

        middleware = captured['middleware']
        assert sum(isinstance(mw, ToolExceptionHandlerMiddleware) for mw in middleware) == 1


class TestCallerSuppliedTehmIsRespected:
    """Unconditional injection must not double up on a TEHM the caller already built —
    that's exactly the class docstring's own documented usage pattern, and assistant.py
    rejects more than one instance."""

    def test_application_keeps_the_callers_instance_only(self):
        caller_tehm = ToolExceptionHandlerMiddleware(strategies=[MagicMock()])

        middleware = _capture_application_middleware(_make_client(), middleware=[caller_tehm])

        tehms = [mw for mw in middleware if isinstance(mw, ToolExceptionHandlerMiddleware)]
        assert tehms == [caller_tehm]


def _make_failing_tool():
    """A StructuredTool whose call raises ValueError -> ToolErrorClass.INPUT, which
    enriches (unlike INFRASTRUCTURE), so it drives the LLM factory."""

    class Args(BaseModel):
        value: str = Field(description="value")

    def failing_func(value: str) -> str:
        raise ValueError("bad value")

    return StructuredTool.from_function(
        func=failing_func,
        name="failing_tool",
        description="A tool that always fails with an input error",
        args_schema=Args,
    )


class _StubLLM:
    """Deterministic .invoke(...).content stand-in, same shape as the production LLM."""

    def __init__(self, content="STUB REWRITE"):
        self._content = content

    def invoke(self, messages, *args, **kwargs):
        return type("_Resp", (), {"content": self._content})()


@patch("elitea_sdk.runtime.utils.tool_code_extractor.extract_tool_code", return_value=None)
@patch("elitea_sdk.runtime.utils.tool_code_loader.load_tool_code", return_value=None)
@patch("elitea_sdk.runtime.middleware.faq_fetcher.get_toolkit_faq", return_value=None)
class TestLowTierLlmIsLazy:
    """The whole point of the factory: construction must not pay for get_low_tier_llm's
    two uncached HTTP round-trips, and once paid it must not be paid again."""

    def test_not_called_during_application_construction(self, _faq, _load_code, _extract_code):
        client = _make_client()
        client.get_low_tier_llm = MagicMock(return_value=_StubLLM())

        _capture_application_middleware(client)

        client.get_low_tier_llm.assert_not_called()

    def test_called_once_on_first_error_and_not_again_on_second(self, _faq, _load_code, _extract_code):
        client = _make_client()
        client.get_low_tier_llm = MagicMock(return_value=_StubLLM())

        middleware_list = _capture_application_middleware(client)
        tehm = next(mw for mw in middleware_list if isinstance(mw, ToolExceptionHandlerMiddleware))
        wrapped = tehm.wrap_tool(_make_failing_tool())

        wrapped.run({"value": "x"})
        assert client.get_low_tier_llm.call_count == 1

        wrapped.run({"value": "y"})
        assert client.get_low_tier_llm.call_count == 1
