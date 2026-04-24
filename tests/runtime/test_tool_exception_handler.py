from unittest.mock import MagicMock

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
