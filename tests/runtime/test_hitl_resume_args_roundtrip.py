"""Test that HITL resume works correctly when tool args undergo JSON round-trip.

This test targets the root cause of issue #4333: tool args (dicts with non-trivial
values) get serialized into the LangGraph checkpoint and deserialized on resume.
The round-trip can reorder dict keys, breaking strict Python dict equality.

The test mimics reality:
- First invocation: LLM calls several tools, one is sensitive → interrupt
- Resume: a NEW graph is created with the same thread_id (as happens in production)
- After resume: agent must NOT re-invoke prior tools from scratch

Each "pass of control to the user" (HITL) creates a new graph with the same thread_id.
"""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.middleware.sensitive_tool_guard import SensitiveToolGuardMiddleware
from elitea_sdk.runtime.toolkits.security import configure_sensitive_tools, reset_sensitive_tools


def setup_function() -> None:
    reset_sensitive_tools()


def teardown_function() -> None:
    reset_sensitive_tools()


class _DummyRuntime:
    def get_mcp_toolkits(self):
        return []


# ─── LLM Stubs ────────────────────────────────────────────────────────────────

# Key insight: the tool args here have multiple keys whose dict ordering
# may change after JSON checkpoint round-trip (e.g., Python dicts before 3.7
# didn't guarantee order, and JSON parsers may reorder).

JIRA_CREATE_ARGS = {
    "project_key": "PROJ",
    "summary": "Implement feature X",
    "issue_type": "Task",
    "priority": "High",
    "description": "Full description of the feature to implement",
}

SEARCH_ARGS = {
    "query": "feature X implementation",
    "project": "PROJ",
    "max_results": 10,
}


class _MultiToolWithArgsLLM:
    """LLM that calls search_issues (non-sensitive) then create_issue (sensitive).

    The LLM issues one batch of tool calls:
    1. search_issues(query="feature X implementation", project="PROJ", max_results=10)
    2. create_issue(project_key="PROJ", summary="Implement feature X", ...)

    create_issue is sensitive, so it triggers HITL interrupt.
    On resume, the LLM should see the search_issues result + create_issue result
    and produce a final answer WITHOUT re-calling search_issues.
    """

    def __init__(self):
        self.invocations: list[list] = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _MultiToolWithArgsLLMBound(self, tools, kwargs)

    def invoke(self, messages, config=None):
        return _MultiToolWithArgsLLMBound(self, [], {}).invoke(messages, config=config)


class _MultiToolWithArgsLLMBound:
    def __init__(self, root: _MultiToolWithArgsLLM, tools, kwargs):
        self.root = root
        self.tools = list(tools)
        self.kwargs = dict(kwargs)

    def invoke(self, messages, config=None):
        self.root.invocations.append(list(messages))
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        tool_contents = [str(m.content) for m in tool_messages]

        # After both tool results are present → final answer
        if "search-result:3 issues found" in tool_contents and "create-result:PROJ-123" in tool_contents:
            return AIMessage(content="Done! Created PROJ-123 based on search results.")

        # First call: emit batch with complex args
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "search_issues", "args": SEARCH_ARGS, "id": "call-search-001"},
                {"name": "create_issue", "args": JIRA_CREATE_ARGS, "id": "call-create-001"},
            ],
        )


class _ThinkingMultiToolLLM:
    """Anthropic thinking model variant — same scenario but with thinking blocks."""

    def __init__(self):
        self.invocations: list[list] = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _ThinkingMultiToolLLMBound(self, tools, kwargs)

    def invoke(self, messages, config=None):
        return _ThinkingMultiToolLLMBound(self, [], {}).invoke(messages, config=config)


class _ThinkingMultiToolLLMBound:
    def __init__(self, root: _ThinkingMultiToolLLM, tools, kwargs):
        self.root = root
        self.tools = list(tools)
        self.kwargs = dict(kwargs)

    def invoke(self, messages, config=None):
        self.root.invocations.append(list(messages))
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        tool_contents = [str(m.content) for m in tool_messages]

        if "search-result:3 issues found" in tool_contents and "create-result:PROJ-123" in tool_contents:
            return AIMessage(
                content=[
                    {"type": "thinking", "thinking": "Both tools completed successfully.", "signature": "sig_final"},
                    {"type": "text", "text": "Done! Created PROJ-123."},
                ]
            )

        # First call: thinking + tool_use (Anthropic-style)
        return AIMessage(
            content=[
                {
                    "type": "thinking",
                    "thinking": "I need to search for existing issues first, then create a new one.",
                    "signature": "sig_planning_001",
                },
                {"type": "text", "text": ""},
            ],
            tool_calls=[
                {"name": "search_issues", "args": SEARCH_ARGS, "id": "call-search-001"},
                {"name": "create_issue", "args": JIRA_CREATE_ARGS, "id": "call-create-001"},
            ],
        )


# ─── Test Fixtures ─────────────────────────────────────────────────────────────

def _build_runnable(memory, llm):
    """Build assistant graph with sensitive create_issue tool."""
    tools = [
        StructuredTool.from_function(
            func=lambda query="", project="", max_results=10: "search-result:3 issues found",
            name="search_issues",
            description="Search for issues in a project",
            metadata={"toolkit_type": "jira", "toolkit_name": "jira", "tool_name": "search_issues"},
        ),
        StructuredTool.from_function(
            func=lambda project_key="", summary="", issue_type="", priority="", description="": "create-result:PROJ-123",
            name="create_issue",
            description="Create a new issue",
            metadata={"toolkit_type": "jira", "toolkit_name": "jira", "tool_name": "create_issue"},
        ),
    ]
    assistant = Assistant(
        elitea=_DummyRuntime(),
        data={"instructions": "use tools as needed", "tools": [], "meta": {}},
        client=llm,
        tools=tools,
        memory=memory,
        app_type="predict",
        middleware=[SensitiveToolGuardMiddleware()],
    )
    return assistant.runnable()


# ─── Tests ─────────────────────────────────────────────────────────────────────

def test_hitl_resume_with_complex_args_does_not_reinvoke_prior_tools():
    """Core regression test for #4333: tool with complex args must resume
    correctly after HITL approval without re-invoking prior tools.

    Simulates the real production flow:
    1. First request → LLM calls search_issues + create_issue in one batch
    2. create_issue is sensitive → HITL interrupt returned to UI
    3. UI approves → NEW graph created with same thread_id → resume
    4. Agent should NOT re-invoke search_issues; should see its prior result
    """
    configure_sensitive_tools({"jira": ["create_issue"]})

    memory = MemorySaver()
    thread_config = {"configurable": {"thread_id": "args-roundtrip-test"}}

    # ═══ Phase 1: Initial request → interrupt on create_issue ═══
    first_llm = _MultiToolWithArgsLLM()
    first_runnable = _build_runnable(memory, first_llm)

    first_result = first_runnable.invoke(
        {"messages": [HumanMessage(content="Search for feature X issues and create a task for it")]},
        config=thread_config,
    )

    assert first_result["execution_finished"] is False
    assert first_result["hitl_interrupt"]["tool_name"] == "create_issue"
    # Verify the interrupt payload contains the complex args
    assert first_result["hitl_interrupt"]["tool_args"]["project_key"] == "PROJ"

    # ═══ Phase 2: Resume with NEW graph (mimics real app behavior) ═══
    resume_llm = _MultiToolWithArgsLLM()
    resumed_runnable = _build_runnable(memory, resume_llm)

    resume_result = resumed_runnable.invoke(
        {"hitl_resume": True, "hitl_action": "approve", "hitl_value": ""},
        config=thread_config,
    )

    # ═══ Assertions ═══
    assert resume_result["execution_finished"] is True
    assert "PROJ-123" in resume_result["output"]

    # The KEY assertion: the LLM should be called exactly ONCE on resume.
    # If args comparison fails, _build_resume_completion returns None,
    # causing a new LLM call that re-invokes all tools from scratch.
    assert len(resume_llm.invocations) == 1, (
        f"Expected 1 LLM invocation on resume (tool result → final answer), "
        f"got {len(resume_llm.invocations)}. This indicates the agent "
        f"re-invoked tools from scratch after HITL resume."
    )

    # Verify the resume LLM call received both tool results
    post_tool_invocation = resume_llm.invocations[0]
    tool_msgs = [m for m in post_tool_invocation if isinstance(m, ToolMessage)]
    tool_contents = [str(m.content) for m in tool_msgs]
    assert "search-result:3 issues found" in tool_contents, (
        "search_issues result must be visible to LLM on resume (from pending_messages)"
    )
    assert "create-result:PROJ-123" in tool_contents, (
        "create_issue result must be visible to LLM on resume (tool just executed)"
    )

    # Verify tool_call_id is present (must have a valid ID for tool_use ↔ tool_result pairing)
    create_tool_msg = next(m for m in tool_msgs if "create-result" in str(m.content))
    assert create_tool_msg.tool_call_id, (
        "create_issue ToolMessage must have a tool_call_id for LLM provider pairing"
    )
    # The AIMessage in the LLM call must have a matching tool_call
    ai_msgs_with_tc = [m for m in post_tool_invocation if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)]
    assert ai_msgs_with_tc, "Expected AIMessage with tool_calls in resumed LLM call"
    all_tc_ids = {
        tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
        for m in ai_msgs_with_tc
        for tc in (m.tool_calls or [])
    }
    assert create_tool_msg.tool_call_id in all_tc_ids, (
        f"ToolMessage tool_call_id={create_tool_msg.tool_call_id} must match "
        f"one of the AIMessage tool_call ids: {all_tc_ids}"
    )


def test_hitl_resume_anthropic_thinking_with_complex_args():
    """Same as above but with Anthropic Extended Thinking model.

    This is the EXACT scenario from issue #4333: Anthropic thinking model +
    complex tool args + HITL resume = agent loses all prior tool call results.
    """
    configure_sensitive_tools({"jira": ["create_issue"]})

    memory = MemorySaver()
    thread_config = {"configurable": {"thread_id": "thinking-args-roundtrip-test"}}

    # ═══ Phase 1: Initial request → interrupt ═══
    first_llm = _ThinkingMultiToolLLM()
    first_runnable = _build_runnable(memory, first_llm)

    first_result = first_runnable.invoke(
        {"messages": [HumanMessage(content="Search for feature X issues and create a task")]},
        config=thread_config,
    )

    assert first_result["execution_finished"] is False
    assert first_result["hitl_interrupt"]["tool_name"] == "create_issue"

    # ═══ Phase 2: Resume with NEW graph ═══
    resume_llm = _ThinkingMultiToolLLM()
    resumed_runnable = _build_runnable(memory, resume_llm)

    resume_result = resumed_runnable.invoke(
        {"hitl_resume": True, "hitl_action": "approve", "hitl_value": ""},
        config=thread_config,
    )

    # ═══ Assertions ═══
    assert resume_result["execution_finished"] is True
    assert "PROJ-123" in resume_result["output"]

    # Single LLM call on resume (not re-planning)
    assert len(resume_llm.invocations) == 1, (
        f"Expected 1 LLM invocation on resume, got {len(resume_llm.invocations)}. "
        f"Anthropic thinking model lost context and re-invoked tools."
    )

    # Verify thinking blocks survived in the AIMessage preceding tool results
    post_tool_invocation = resume_llm.invocations[0]
    ai_msgs = [m for m in post_tool_invocation if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)]
    assert ai_msgs, "Expected AIMessage with tool_calls in the resumed LLM call"

    last_ai = ai_msgs[-1]
    assert isinstance(last_ai.content, list), (
        "Anthropic AIMessage content must remain a list after HITL resume"
    )
    thinking_blocks = [b for b in last_ai.content if isinstance(b, dict) and b.get("type") == "thinking"]
    assert thinking_blocks, (
        "Thinking blocks were stripped during HITL resume — "
        "Anthropic reasoning continuity broken"
    )
    assert thinking_blocks[0]["signature"] == "sig_planning_001"

    # Tool results present
    tool_msgs = [m for m in post_tool_invocation if isinstance(m, ToolMessage)]
    tool_contents = [str(m.content) for m in tool_msgs]
    assert "search-result:3 issues found" in tool_contents
    assert "create-result:PROJ-123" in tool_contents


def test_hitl_resume_with_reordered_args_in_checkpoint():
    """Directly test the scenario where checkpoint stores args with different
    key ordering than what the LLM originally emitted.

    This simulates what happens when:
    - LLM emits: {"project_key": "PROJ", "summary": "Test", "priority": "High"}
    - Checkpoint stores (after JSON round-trip): {"priority": "High", "project_key": "PROJ", "summary": "Test"}
    - On resume, _build_resume_completion must still match these as equal
    """
    configure_sensitive_tools({"jira": ["create_issue"]})

    memory = MemorySaver()
    thread_config = {"configurable": {"thread_id": "reordered-args-test"}}

    # Phase 1: normal tool call with specific arg ordering
    first_llm = _MultiToolWithArgsLLM()
    first_runnable = _build_runnable(memory, first_llm)

    first_result = first_runnable.invoke(
        {"messages": [HumanMessage(content="Create a Jira task")]},
        config=thread_config,
    )
    assert first_result["execution_finished"] is False

    # Phase 2: Resume — the checkpoint may have reordered the args internally
    # This test validates that our normalized comparison handles it
    resume_llm = _MultiToolWithArgsLLM()
    resumed_runnable = _build_runnable(memory, resume_llm)

    resume_result = resumed_runnable.invoke(
        {"hitl_resume": True, "hitl_action": "approve", "hitl_value": ""},
        config=thread_config,
    )

    assert resume_result["execution_finished"] is True
    # Must complete in a single LLM call (not re-invoke)
    assert len(resume_llm.invocations) == 1
