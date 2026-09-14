"""Regression for #6625: LangChain-derived middlewares must still reach before_model.

`transform_messages_for_model` is an Elitea-only hook declared on BaseMiddleware. Middlewares
that subclass LangChain's AgentMiddleware (SummarizationMiddleware, ContextEditingMiddleware)
do not have it, and calling it unconditionally raised AttributeError inside run_before_model's
try/except -- which silently skipped the summarization/context-editing work in before_model.
"""

from langchain_core.messages import HumanMessage

from elitea_sdk.runtime.middleware.base import MiddlewareManager


class _LangChainStyleMiddleware:
    """Only what AgentMiddleware offers: before_model, no Elitea transform hook."""

    def __init__(self):
        self.before_model_calls = 0

    def before_model(self, state, config):
        self.before_model_calls += 1
        return None


def test_before_model_runs_for_a_middleware_without_the_transform_hook():
    middleware = _LangChainStyleMiddleware()
    assert not hasattr(middleware, 'transform_messages_for_model')

    state, checkpoint_operations = MiddlewareManager().add(middleware).run_before_model(
        {"messages": [HumanMessage(content="Hello")]}, {},
    )

    assert middleware.before_model_calls == 1
    assert checkpoint_operations == []
    assert [m.content for m in state["messages"]] == ["Hello"]
