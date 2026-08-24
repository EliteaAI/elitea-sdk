from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field
from pydantic_core import PydanticUndefined

from elitea_sdk.runtime.middleware.strategies import (
    CircuitBreakerStrategy,
    ExceptionContext,
    LoggingStrategy,
    TransformErrorStrategy,
)
from elitea_sdk.runtime.middleware.tool_exception_handler import (
    ToolExceptionHandlerMiddleware,
)
from elitea_sdk.runtime.tool_outcome import ToolErrorClass, ToolResultStatus
from elitea_sdk.runtime.tools.function import FunctionTool


def _make_tool(func=None, required_str=True):
    """Create a StructuredTool with a required str field for testing."""

    class Args(BaseModel):
        issue_number: str = Field(description="The issue number")
        title: str = Field(description="Title", default="default")

    def dummy_func(issue_number: str, title: str = "default") -> str:
        return f"ok: {issue_number}"

    return StructuredTool.from_function(
        func=func or dummy_func,
        name="update_issue",
        description="Update an issue",
        args_schema=Args,
    )


def _make_middleware(threshold=5):
    """Create middleware with default strategies (no LLM)."""
    return ToolExceptionHandlerMiddleware(
        strategies=[
            TransformErrorStrategy(llm=None),
            CircuitBreakerStrategy(threshold=threshold),
            LoggingStrategy(),
        ]
    )


class TestValidationErrorHandling:
    """Tests for pydantic ValidationError handling in wrap_tool."""

    def test_wrap_tool_handles_validation_error(self):
        """ValidationError returns an error string instead of crashing."""
        middleware = _make_middleware()
        tool = _make_tool()
        wrapped = middleware.wrap_tool(tool)

        # Invoke with PydanticUndefined — triggers ValidationError
        result = wrapped.run({"issue_number": PydanticUndefined})

        assert isinstance(result, str)
        assert "failed" in result.lower() or "error" in result.lower()

    def test_validation_error_routes_through_strategies(self):
        """ValidationError flows through all strategies, not just LangChain's default."""
        mock_strategy = MagicMock()
        mock_strategy.handle_exception.return_value = ExceptionContext(
            tool=MagicMock(),
            error=ValueError("test"),
            args=(),
            kwargs={},
            error_message="custom strategy message",
        )

        middleware = ToolExceptionHandlerMiddleware(strategies=[mock_strategy])
        tool = _make_tool()
        wrapped = middleware.wrap_tool(tool)

        result = wrapped.run({"issue_number": PydanticUndefined})

        # Strategy was called
        mock_strategy.handle_exception.assert_called_once()

        # The ExceptionContext passed to strategy has a ValidationError
        ctx = mock_strategy.handle_exception.call_args[0][0]
        assert "ValidationError" in type(ctx.error).__name__ or "validation" in str(ctx.error).lower()

        # Returned message comes from strategy, not LangChain default
        assert result == "custom strategy message"

    def test_validation_error_tracks_in_circuit_breaker(self):
        """Repeated ValidationErrors trigger circuit breaker ToolException."""
        middleware = _make_middleware(threshold=2)
        tool = _make_tool()
        wrapped = middleware.wrap_tool(tool)

        # First call — handled gracefully
        result = wrapped.run({"issue_number": PydanticUndefined})
        assert isinstance(result, str)

        # Second call — circuit breaker fires
        with pytest.raises(ToolException, match="temporarily disabled"):
            wrapped.run({"issue_number": PydanticUndefined})

    def test_validation_error_logged_by_logging_strategy(self):
        """ValidationError is recorded in LoggingStrategy error counts."""
        logging_strategy = LoggingStrategy()
        middleware = ToolExceptionHandlerMiddleware(
            strategies=[
                TransformErrorStrategy(llm=None),
                logging_strategy,
            ]
        )
        tool = _make_tool()
        wrapped = middleware.wrap_tool(tool)

        wrapped.run({"issue_number": PydanticUndefined})

        summary = logging_strategy.get_error_summary()
        assert summary.get("update_issue", 0) == 1

    def test_tool_error_callback_fires_on_failure(self):
        """LoggingStrategy invokes the 'tool_error' callback with the failure payload."""
        tool_error_callback = MagicMock()
        logging_strategy = LoggingStrategy(callbacks={'tool_error': tool_error_callback})
        middleware = ToolExceptionHandlerMiddleware(
            strategies=[
                TransformErrorStrategy(llm=None),
                logging_strategy,
            ]
        )
        tool = _make_tool()
        wrapped = middleware.wrap_tool(tool)

        wrapped.run({"issue_number": PydanticUndefined})

        tool_error_callback.assert_called_once()
        payload = tool_error_callback.call_args[0][0]
        assert payload['tool_name'] == 'update_issue'
        assert payload['total_errors'] == 1
        assert 'error_type' in payload
        assert 'error' in payload

    def test_valid_input_still_works(self):
        """Normal valid input is not affected by the fix."""
        middleware = _make_middleware()
        tool = _make_tool()
        wrapped = middleware.wrap_tool(tool)

        result = wrapped.run({"issue_number": "42"})
        assert result == "ok: 42"

    def test_runtime_exception_still_handled(self):
        """Runtime exceptions inside the tool still go through strategies."""

        def failing_func(issue_number: str, title: str = "default") -> str:
            raise RuntimeError("connection refused")

        middleware = _make_middleware()
        tool = _make_tool(func=failing_func)
        wrapped = middleware.wrap_tool(tool)

        result = wrapped.run({"issue_number": "42"})
        assert isinstance(result, str)
        assert "failed" in result.lower() or "error" in result.lower()


def _failing_func(issue_number: str, title: str = "default") -> str:
    raise RuntimeError("connection refused")


class _StubLLM:
    """Minimal stand-in for a chat model.

    TransformErrorStrategy._generate_llm_error only calls .invoke(messages) and then
    normalize_message_content(response.content), so a plain object with a str .content
    is enough — and it makes the enriched path deterministic, which is what lets us
    byte-assert it.
    """

    def __init__(self, content="STUB REWRITE"):
        self._content = content

    def invoke(self, messages, *args, **kwargs):
        return type("_Resp", (), {"content": self._content})()


class TestErrorProseIsByteStable:
    """Pins the EXACT string a failing tool returns, on both prose paths.

    Epic #6164 moves tool errors onto a typed ToolOutcome envelope. Sub-item #6169 is
    strictly additive: it must construct the envelope without changing a single byte of
    what any consumer already receives. These tests are the baseline for that claim —
    they are written against the unmodified middleware and must stay green through
    every later commit. Asserting on literal expected strings (rather than
    re-deriving them from the same f-strings the production code uses) is deliberate:
    a re-derivation would silently follow the code it is supposed to be guarding.
    """

    def test_template_path_message_is_byte_stable(self):
        """llm=None -> TransformErrorStrategy._generate_template_error, verbatim."""
        middleware = _make_middleware()
        wrapped = middleware.wrap_tool(_make_tool(func=_failing_func))

        result = wrapped.run({"issue_number": "42"})

        assert result == (
            "Tool 'update_issue' failed.\n"
            "\n"
            "Error: connection refused\n"
            "\n"
            "Please check the input parameters and try again, "
            "or use an alternative approach."
        )

    @patch("elitea_sdk.runtime.utils.tool_code_extractor.extract_tool_code", return_value=None)
    @patch("elitea_sdk.runtime.utils.tool_code_loader.load_tool_code", return_value=None)
    @patch("elitea_sdk.runtime.middleware.faq_fetcher.get_toolkit_faq", return_value=None)
    def test_enriched_path_message_is_byte_stable(self, _faq, _load_code, _extract_code):
        """A stub LLM makes the enriched path deterministic, so it is byte-assertable too.

        The returned message is not the raw model output: handle_exception appends a
        fixed support-contact suffix. Nothing else asserted that suffix until now.
        The three patches only keep the test hermetic — get_toolkit_faq does a real
        5s-timeout urlopen against GitHub docs. None of them can affect the result,
        since they feed the prompt and the stub ignores it.
        """
        middleware = ToolExceptionHandlerMiddleware(
            strategies=[TransformErrorStrategy(llm=_StubLLM("STUB REWRITE")), LoggingStrategy()]
        )
        wrapped = middleware.wrap_tool(_make_tool(func=_failing_func))

        result = wrapped.run({"issue_number": "42"})

        assert result == (
            "STUB REWRITE\n"
            "\n"
            "*IMPORTANT*: if fixing logic is clear - you can re-try tool execution "
            "according to fix.\n"
            "If you continue experiencing issues, please [contact support]"
            "(https://elitea.ai/docs/support/contact-support/)"
        )


class TestHandleToolErrorSurvivesTheRebuild:
    """wrap_tool rebuilds each tool as a fresh StructuredTool, which silently dropped
    handle_tool_error and gave every wrapped tool LangChain's default of False.

    Near-inert today — only ToolExceptions that *escape* the middleware are affected, and
    the one escape path (the circuit breaker) is not in either production strategy list.
    It matters for intent: the MCP proxies set False deliberately, because
    McpAuthorizationRequired is a ToolException and must reach the agent, and #6167 flips
    github/tool.py from True to False. Both readings need this field to be honoured.
    """

    def test_true_is_preserved(self):
        tool = _make_tool()
        tool.handle_tool_error = True

        assert _make_middleware().wrap_tool(tool).handle_tool_error is True

    def test_unset_stays_false(self):
        wrapped = _make_middleware().wrap_tool(_make_tool())

        assert wrapped.handle_tool_error is False


class _ProbeStrategy:
    """Records the metadata the middleware had already written when it was called.

    Deliberately not a subclass of ExceptionHandlerStrategy — the middleware duck-types,
    and a plain object keeps the probe to the three methods that actually get called.
    """

    def __init__(self):
        self.seen = None

    def handle_exception(self, context):
        self.seen = dict(context.metadata)
        context.error_message = "probed"
        return context

    def on_success(self, tool_name):
        pass

    def reset(self):
        pass


class TestClassificationHappensBeforeStrategies:
    """The ordering sub-item 3 (#6167) depends on, and which is otherwise invisible.

    #6167 makes the LLM prose rewrite opt-out and wants to skip it for errors that gain
    nothing from being reworded (an infrastructure timeout). That gate lives inside a
    strategy, so error_class has to be in context.metadata *before* the strategy loop —
    but the message can only be finalised *after* it. Assert the first half here; the
    byte-identity tests above cover the second.
    """

    def test_runtime_error_is_classified_before_the_loop(self):
        probe = _ProbeStrategy()
        middleware = ToolExceptionHandlerMiddleware(strategies=[probe])
        wrapped = middleware.wrap_tool(_make_tool(func=_failing_func))

        assert wrapped.run({"issue_number": "42"}) == "probed"
        assert probe.seen["error_class"] is ToolErrorClass.INFRASTRUCTURE
        assert probe.seen["retriable"] is True
        assert probe.seen["exception_type"] == "RuntimeError"

    def test_validation_error_is_classified_before_the_loop(self):
        """The ValidationError route is a second finalisation point, so it classifies too."""
        probe = _ProbeStrategy()
        middleware = ToolExceptionHandlerMiddleware(strategies=[probe])
        wrapped = middleware.wrap_tool(_make_tool())

        assert wrapped.run({"issue_number": PydanticUndefined}) == "probed"
        assert probe.seen["error_class"] is ToolErrorClass.INPUT
        assert probe.seen["retriable"] is False


class TestOutcomeEnvelope:
    """#6169 constructs the envelope but publishes it nowhere yet (sub-items 6-8 do).

    Until then _finalize_outcome is the only place to observe it, so assert on it
    directly rather than leaving the new fields untested for a release.
    """

    def test_envelope_carries_the_classified_facts(self):
        middleware = _make_middleware()
        tool = _make_tool()
        tool.metadata = {"toolkit_type": "github", "toolkit_id": 7}
        context = ExceptionContext(
            tool=tool, error=ConnectionError("connection refused"), args=(), kwargs={},
            error_message="Tool 'update_issue' failed.",
        )
        middleware._classify_into(context)

        outcome = middleware._finalize_outcome(context)

        assert outcome.status is ToolResultStatus.ERROR
        assert outcome.message == "Tool 'update_issue' failed."
        assert outcome.tool_name == "update_issue"
        assert outcome.error_class is ToolErrorClass.INFRASTRUCTURE
        assert outcome.retriable is True
        assert outcome.exception_type == "ConnectionError"
        assert outcome.toolkit_type == "github"

    def test_message_falls_back_to_the_raw_error_string(self):
        """Pass-through of the pre-envelope `context.error_message or str(e)`."""
        middleware = _make_middleware()
        context = ExceptionContext(
            tool=_make_tool(), error=RuntimeError("raw boom"), args=(), kwargs={},
        )
        middleware._classify_into(context)

        assert middleware._finalize_outcome(context).message == "raw boom"

    def test_non_str_metadata_cannot_break_the_error_path(self):
        """Tool metadata is free-form; a bad value must not turn an error into a crash."""
        middleware = _make_middleware()
        tool = _make_tool()
        tool.metadata = {"toolkit_type": {"unexpected": "shape"}}
        context = ExceptionContext(
            tool=tool, error=RuntimeError("boom"), args=(), kwargs={},
        )
        middleware._classify_into(context)

        assert middleware._finalize_outcome(context).toolkit_type is None


class TestPipelineFailureShapes:
    """Pins BOTH shapes a failing pipeline node produces today.

    They differ in a way that is easy to miss: with the middleware active the tool
    returns an error *string* (a successful return, as far as FunctionTool knows), so
    the prose lands in the declared output_variables. Without the middleware the raise
    escapes into FunctionTool.invoke's `except Exception`, which writes only `messages`
    and no output_variables at all. Sub-item 7 projects the envelope onto pipeline
    state, so both shapes need to be pinned first — a later change to either one should
    trip a test rather than a user.
    """

    @staticmethod
    def _node(tool):
        return FunctionTool(
            name="update_issue_node",
            tool=tool,
            input_mapping={"issue_number": {"type": "fixed", "value": "42"}},
            input_variables=[],
            output_variables=["issue"],
        )

    def test_wrapped_tool_puts_error_prose_in_output_variables(self):
        middleware = _make_middleware()
        node = self._node(middleware.wrap_tool(_make_tool(func=_failing_func)))

        with patch("elitea_sdk.runtime.tools.function.dispatch_custom_event"):
            result = node.invoke({})

        assert set(result) == {"issue"}
        assert result["issue"] == (
            "Tool 'update_issue' failed.\n"
            "\n"
            "Error: connection refused\n"
            "\n"
            "Please check the input parameters and try again, "
            "or use an alternative approach."
        )

    def test_unwrapped_tool_writes_only_messages(self):
        node = self._node(_make_tool(func=_failing_func))

        with patch("elitea_sdk.runtime.tools.function.dispatch_custom_event"):
            result = node.invoke({})

        # The declared output variable is absent entirely — not empty, absent.
        assert set(result) == {"messages"}
        assert "connection refused" in result["messages"][0]["content"]
