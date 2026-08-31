"""Project Context progressive-disclosure contracts for issue #5326."""

from hashlib import sha256

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from elitea_sdk.runtime.clients.client import EliteAClient
from elitea_sdk.runtime.middleware.base import MiddlewareManager
from elitea_sdk.runtime.middleware.project_context import ProjectContextMiddleware


def _middleware(**overrides):
    payload = {
        "content": "Use dry, concise humor and avoid puns.",
        "activation_description": "Use for requests to generate, rewrite, or evaluate jokes.",
        "revision": "revision-7",
    }
    payload.update(overrides)
    return ProjectContextMiddleware(payload)


def test_loader_schema_carries_activation_but_not_full_context():
    middleware = _middleware()
    tool = middleware.get_tools()[0]
    system_prompt = middleware.get_system_prompt()

    assert tool.name == "read_project_context"
    assert "generate, rewrite, or evaluate jokes" in tool.description
    assert "revision-7" in tool.description
    assert "dry, concise humor" not in tool.description
    assert "MUST call" in system_prompt
    assert "read_project_context" in system_prompt
    assert "generate, rewrite, or evaluate jokes" not in system_prompt
    assert "dry, concise humor" not in system_prompt


def test_loader_returns_full_context_and_revision(caplog):
    caplog.set_level("INFO")
    result = _middleware().get_tools()[0].invoke({})

    assert "Project Context revision: revision-7" in result
    assert "Use dry, concise humor and avoid puns." in result
    assert "Project Context loaded on demand: revision=revision-7" in caplog.text
    assert "context_chars=38" in caplog.text


def test_revision_is_derived_when_core_does_not_supply_one():
    content = "Context with a stable digest"
    middleware = _middleware(content=content, revision=None)

    assert middleware.revision == sha256(content.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    "payload",
    [
        {"activation_description": "Use for jokes."},
        {"content": "Joke rules"},
        {"content": " ", "activation_description": "Use for jokes."},
    ],
)
def test_incomplete_context_is_rejected(payload):
    with pytest.raises(ValueError):
        ProjectContextMiddleware(payload)


def test_middleware_manager_exposes_loader_as_a_middleware_tool():
    manager = MiddlewareManager().add(_middleware())

    assert [tool.name for tool in manager.get_all_tools()] == ["read_project_context"]


def test_client_injects_one_project_context_middleware():
    middleware = []
    payload = {
        "content": "Joke rules",
        "activation_description": "Use for joke requests.",
        "revision": "revision-8",
    }

    EliteAClient._inject_project_context(middleware, payload, "conversation-1")
    EliteAClient._inject_project_context(middleware, payload, "conversation-1")

    assert len(middleware) == 1
    assert isinstance(middleware[0], ProjectContextMiddleware)
    assert middleware[0].conversation_id == "conversation-1"


def test_full_context_is_visible_only_for_immediate_tool_response_step():
    middleware = _middleware()
    tool_result = ToolMessage(
        content=middleware.get_tools()[0].invoke({}),
        name="read_project_context",
        tool_call_id="call-1",
    )
    messages = [
        HumanMessage(content="Tell me a joke"),
        AIMessage(content="", tool_calls=[{
            "name": "read_project_context",
            "args": {},
            "id": "call-1",
            "type": "tool_call",
        }]),
        tool_result,
    ]

    assert middleware.transform_messages_for_model(messages, {}) is None

    later_messages = [*messages, AIMessage(content="A dry joke"), HumanMessage(content="Hello")]
    transformed = middleware.transform_messages_for_model(later_messages, {})

    redacted_result = transformed[2]
    assert redacted_result.tool_call_id == "call-1"
    assert "dry, concise humor" not in redacted_result.content
    assert "Previous Project Context revision: revision-7" in redacted_result.content
    assert "call read_project_context" in redacted_result.content


def test_transient_redaction_is_not_returned_as_checkpoint_operation():
    middleware = _middleware()
    manager = MiddlewareManager().add(middleware)
    historical_result = ToolMessage(
        content=middleware.get_tools()[0].invoke({}),
        name="read_project_context",
        tool_call_id="call-2",
    )

    state, checkpoint_operations = manager.run_before_model(
        {"messages": [historical_result, HumanMessage(content="Unrelated question")]},
        {},
    )

    assert "dry, concise humor" not in state["messages"][0].content
    assert checkpoint_operations == []
    assert "dry, concise humor" in historical_result.content
