"""Budget rejections must propagate, never become message content.

There is no recovery from an exhausted budget, so swallowing the error into an
AIMessage/ToolMessage tells the user the run succeeded and — on nested paths — feeds
a policy rejection back into a model as if it were data. These tests pin the detection
contract and the re-raise behaviour, and guard that every other error class is
untouched.
"""

import pytest
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

from elitea_sdk.runtime.exceptions import (
    BUDGET_SCOPES,
    DEFAULT_BUDGET_SCOPE,
    BudgetExceededError,
    budget_exceeded_from,
)
from elitea_sdk.runtime.middleware.strategies import (
    CircuitBreakerStrategy,
    LoggingStrategy,
    TransformErrorStrategy,
)
from elitea_sdk.runtime.middleware.tool_exception_handler import (
    ToolExceptionHandlerMiddleware,
)
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired

PROJECT_SCOPE, MEMBER_SCOPE = BUDGET_SCOPES


class FakeProviderError(Exception):
    """Stands in for openai/anthropic BadRequestError, which expose .body."""

    def __init__(self, body=None, message="Error code: 400"):
        super().__init__(message)
        self.body = body


def _budget_body(scope=PROJECT_SCOPE, wrapped=False):
    detail = {
        "message": "The budget for shared models has been reached.",
        "type": "budget_exceeded",
        "code": scope,
    }
    return {"error": detail} if wrapped else detail


class TestDetection:
    def test_openai_shape(self):
        # The OpenAI client strips the "error" wrapper before storing body
        error = budget_exceeded_from(FakeProviderError(_budget_body()))

        assert isinstance(error, BudgetExceededError)
        assert error.scope == PROJECT_SCOPE

    def test_anthropic_shape(self):
        # The Anthropic client keeps the wrapper, so both shapes must be read
        error = budget_exceeded_from(FakeProviderError(_budget_body(MEMBER_SCOPE, wrapped=True)))

        assert isinstance(error, BudgetExceededError)
        assert error.scope == MEMBER_SCOPE

    def test_member_scope_is_preserved(self):
        # Scope selects the message and usage link the user is shown
        error = budget_exceeded_from(FakeProviderError(_budget_body(MEMBER_SCOPE)))

        assert error.scope == MEMBER_SCOPE

    def test_unknown_scope_falls_back(self):
        error = budget_exceeded_from(FakeProviderError(_budget_body("something_new")))

        assert error.scope == DEFAULT_BUDGET_SCOPE

    def test_missing_scope_falls_back(self):
        error = budget_exceeded_from(FakeProviderError({"type": "budget_exceeded"}))

        assert error.scope == DEFAULT_BUDGET_SCOPE

    def test_user_facing_message_is_carried(self):
        error = budget_exceeded_from(FakeProviderError(_budget_body()))

        assert "budget for shared models has been reached" in str(error)

    def test_message_fallback_when_body_is_lost(self):
        # Some paths wrap the provider error and keep only its text
        raw = "Error code: 400 - {'type': 'budget_exceeded', 'code': 'member_budget_exceeded'}"
        error = budget_exceeded_from(Exception(raw))

        assert isinstance(error, BudgetExceededError)
        assert error.scope == MEMBER_SCOPE

    def test_already_typed_passes_through(self):
        original = BudgetExceededError("x", MEMBER_SCOPE)

        assert budget_exceeded_from(original) is original

    @pytest.mark.parametrize(
        "error",
        [
            FakeProviderError({"error": {"type": "invalid_request_error", "message": "bad model"}}),
            FakeProviderError("a string body"),
            FakeProviderError(None),
            Exception("connection refused"),
            ToolException("tool blew up"),
        ],
        ids=["other_400", "non_dict_body", "no_body", "unrelated", "tool_exception"],
    )
    def test_unrelated_errors_are_not_budget_errors(self, error):
        # The regression that matters: a plain 400 is far too broad to treat as budget
        assert budget_exceeded_from(error) is None


class TestMiddlewareReRaise:
    """The middleware's TransformErrorStrategy spends an LLM call rewriting errors,
    so a budget rejection must bypass the strategies entirely."""

    def _wrapped(self, raising_func):
        class Args(BaseModel):
            value: str = Field(description="anything", default="x")

        middleware = ToolExceptionHandlerMiddleware(
            strategies=[
                TransformErrorStrategy(llm=None),
                CircuitBreakerStrategy(threshold=5),
                LoggingStrategy(),
            ]
        )
        tool = StructuredTool.from_function(
            func=raising_func, name="budget_tool", description="t", args_schema=Args,
        )
        return middleware.wrap_tool(tool)

    def test_budget_error_propagates(self):
        def boom(value: str = "x") -> str:
            raise FakeProviderError(_budget_body())

        with pytest.raises(BudgetExceededError) as caught:
            self._wrapped(boom).run({"value": "x"})

        assert caught.value.scope == PROJECT_SCOPE

    def test_mcp_authorization_still_propagates(self):
        # Pre-existing clause, previously untested; the budget clause must not shadow it
        def boom(value: str = "x") -> str:
            raise McpAuthorizationRequired("login required", server_url="https://mcp.example")

        with pytest.raises(McpAuthorizationRequired):
            self._wrapped(boom).run({"value": "x"})

    def test_ordinary_error_is_still_swallowed(self):
        # Everything that is not a budget rejection keeps its existing handling
        def boom(value: str = "x") -> str:
            raise RuntimeError("connection refused")

        result = self._wrapped(boom).run({"value": "x"})

        assert isinstance(result, str)
        assert not isinstance(result, BudgetExceededError)


class TestToolkitTestReRaise:
    """test_toolkit_tool returns errors as a dict, which loses the exception type.

    That is the toolkit-testing and indexing path. Returning the budget rejection left the
    caller with nothing but a stringified provider payload, so the UI showed the raw
    "Error code: 400 - {'error': ...}" instead of explaining the block. Only budget
    rejections raise; every other failure keeps the dict contract its callers rely on.
    """

    def _client(self):
        from elitea_sdk.runtime.clients.client import EliteAClient

        client = EliteAClient.__new__(EliteAClient)
        client.get_llm = lambda *a, **kw: object()
        return client

    def _run(self, monkeypatch, raising_func):
        """Drive test_toolkit_tool with toolkit instantiation stubbed to one fake tool."""
        class Args(BaseModel):
            value: str = Field(description="anything", default="x")

        tool = StructuredTool.from_function(
            func=raising_func, name="index_data", description="t", args_schema=Args,
        )
        #
        from elitea_sdk.runtime.utils import toolkit_utils

        monkeypatch.setattr(
            toolkit_utils, "instantiate_toolkit_with_client",
            lambda *a, **kw: [tool], raising=False,
        )
        #
        return self._client().test_toolkit_tool(
            toolkit_config={"toolkit_name": "artifact", "settings": {}},
            tool_name="index_data",
            tool_params={"value": "x"},
        )

    def test_budget_error_propagates(self, monkeypatch):
        def boom(value: str = "x") -> str:
            raise FakeProviderError(_budget_body())

        with pytest.raises(BudgetExceededError) as caught:
            self._run(monkeypatch, boom)

        assert caught.value.scope == PROJECT_SCOPE

    def test_member_scope_is_preserved(self, monkeypatch):
        def boom(value: str = "x") -> str:
            raise FakeProviderError(_budget_body(MEMBER_SCOPE))

        with pytest.raises(BudgetExceededError) as caught:
            self._run(monkeypatch, boom)

        assert caught.value.scope == MEMBER_SCOPE

    def test_ordinary_error_still_returns_a_dict(self, monkeypatch):
        # The contract every other caller depends on: failures are returned, not raised
        def boom(value: str = "x") -> str:
            raise RuntimeError("connection refused")

        result = self._run(monkeypatch, boom)

        assert result["success"] is False
        assert "connection refused" in result["error"]

    def test_budget_payload_does_not_reach_the_returned_error_string(self, monkeypatch):
        # The reported symptom: the raw provider dict shown to the user verbatim
        def boom(value: str = "x") -> str:
            raise FakeProviderError(_budget_body())

        with pytest.raises(BudgetExceededError):
            self._run(monkeypatch, boom)
