"""Real-graph HITL resume test for issue #4333.

Uses Assistant.runnable() which internally calls create_graph() to build a
LangGraphAgentRunnable with an LLM node bound to tools. This tests the EXACT
production code path end-to-end:

  Assistant.runnable() → create_graph() → LangGraphAgentRunnable
    → LLMNode (with tool calling) → SensitiveToolGuard interrupt
    → NEW Assistant.runnable() built with same MemorySaver → HITL resume

Each "pass of control to the user" (HITL) builds a fresh runnable from a
new Assistant instance with the same shared MemorySaver, matching real
application behavior where every HTTP request creates a fresh graph.

Scenarios tested:
1. Standard model: LLM calls tools with complex nested args → sensitive tool
   interrupt → resume with new graph → LLM sees all prior results → final
   answer (NO re-invocation of prior tools)
2. Anthropic thinking model: same as above but with thinking/text content
   blocks → thinking blocks survive the HITL round-trip
3. Multiple sensitive tools: two sensitive tools interrupted sequentially →
   both approved → pipeline completes
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


# ─── Complex tool arguments (the core of #4333) ──────────────────────────────

JIRA_CREATE_ARGS = {
    "project_key": "PROJ",
    "summary": "Implement feature X",
    "issue_type": "Task",
    "priority": "High",
    "description": "Full description with special chars: <>&\"'",
    "labels": ["backend", "urgent"],
    "custom_fields": {
        "story_points": 5,
        "sprint_id": 42,
        "acceptance_criteria": "Must pass all tests",
    },
}

SEARCH_ARGS = {
    "query": "feature X implementation",
    "project": "PROJ",
    "max_results": 10,
    "filters": {"status": ["Open", "In Progress"], "assignee": None},
}


# ─── LLM Stubs ───────────────────────────────────────────────────────────────


class _ComplexArgsLLM:
    """LLM that issues a batch of tool calls with complex nested args.

    First call: search_issues + create_issue (sensitive → HITL)
    After resume (both tool results present): final answer
    """

    def __init__(self):
        self.invocations: list[list] = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _ComplexArgsLLMBound(self, tools, kwargs)

    def invoke(self, messages, config=None):
        return _ComplexArgsLLMBound(self, [], {}).invoke(messages, config=config)


class _ComplexArgsLLMBound:
    def __init__(self, root: _ComplexArgsLLM, tools, kwargs):
        self.root = root
        self.tools = list(tools)
        self.kwargs = dict(kwargs)

    def invoke(self, messages, config=None):
        self.root.invocations.append(list(messages))
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        tool_contents = [str(m.content) for m in tool_messages]

        # Both tool results present → final answer
        if "search-result:3 issues found" in tool_contents and "create-result:PROJ-123" in tool_contents:
            return AIMessage(content="Done! Created PROJ-123 based on search results.")

        # First call: emit batch with complex nested args
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "search_issues", "args": SEARCH_ARGS, "id": "call-search-001"},
                {"name": "create_issue", "args": JIRA_CREATE_ARGS, "id": "call-create-001"},
            ],
        )


class _ThinkingComplexArgsLLM:
    """Anthropic thinking model variant: same tool calls but with thinking blocks."""

    def __init__(self):
        self.invocations: list[list] = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _ThinkingComplexArgsLLMBound(self, tools, kwargs)

    def invoke(self, messages, config=None):
        return _ThinkingComplexArgsLLMBound(self, [], {}).invoke(messages, config=config)


class _ThinkingComplexArgsLLMBound:
    def __init__(self, root: _ThinkingComplexArgsLLM, tools, kwargs):
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
                    {"type": "thinking", "thinking": "Both tools completed.", "signature": "sig_final"},
                    {"type": "text", "text": "Done! Created PROJ-123."},
                ]
            )

        # First call with thinking blocks (Anthropic Extended Thinking)
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


# ─── Helper to build a fresh runnable (new Assistant + new graph each time) ──


class _ToolExecutionCounter:
    """Side-effect counter shared across runnables built within a single test.

    Mirrors the real-system behavior where each HTTP request creates a fresh
    graph + tool instances, but real toolkits hit the same external system,
    so we can tell whether a non-sensitive tool was actually re-executed
    on the resume path. Closes the gap flagged in PR review (issue #4333).
    """

    def __init__(self):
        self.search_issues = 0
        self.create_issue = 0


def _make_tools(counter: "_ToolExecutionCounter | None" = None):
    """Create the tools that the LLM will call.

    When ``counter`` is provided, each tool invocation increments the matching
    counter so the test can assert exactly-once execution semantics across
    HITL round-trips.
    """

    def _search_issues(query="", project="", max_results=10, filters=None):
        if counter is not None:
            counter.search_issues += 1
        return "search-result:3 issues found"

    def _create_issue(
        project_key="",
        summary="",
        issue_type="",
        priority="",
        description="",
        labels=None,
        custom_fields=None,
    ):
        if counter is not None:
            counter.create_issue += 1
        return "create-result:PROJ-123"

    return [
        StructuredTool.from_function(
            func=_search_issues,
            name="search_issues",
            description="Search for issues in a project",
            metadata={"toolkit_type": "jira", "toolkit_name": "jira", "tool_name": "search_issues"},
        ),
        StructuredTool.from_function(
            func=_create_issue,
            name="create_issue",
            description="Create a new issue",
            metadata={"toolkit_type": "jira", "toolkit_name": "jira", "tool_name": "create_issue"},
        ),
    ]


def _build_runnable(memory: MemorySaver, llm, counter: "_ToolExecutionCounter | None" = None):
    """Build a fresh Assistant → runnable (creates a new graph each time).

    This matches production: each HTTP request creates a new Assistant and
    calls .runnable() which internally calls create_graph().
    """
    assistant = Assistant(
        elitea=_DummyRuntime(),
        data={"instructions": "Use tools as needed to complete the task.", "tools": [], "meta": {}},
        client=llm,
        tools=_make_tools(counter),
        memory=memory,
        app_type="predict",
        middleware=[SensitiveToolGuardMiddleware()],
    )
    return assistant.runnable()


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestHITLResumeRealGraphComplexArgs:
    """End-to-end tests using Assistant.runnable() → create_graph() internally.

    Each phase creates a NEW runnable (new Assistant + new compiled graph),
    sharing only the MemorySaver checkpoint store and thread_id — exactly
    like production.
    """

    def test_standard_model_resume_does_not_reinvoke_tools(self):
        """Standard model (no thinking blocks): HITL resume with complex args.

        Regression test for #4333 core scenario:
        - LLM calls search_issues({complex nested args}) + create_issue({complex args})
        - create_issue is sensitive → HITL interrupt
        - User approves → NEW graph created → resume
        - Agent must NOT re-invoke search_issues; LLM sees all prior results
        """
        configure_sensitive_tools({"jira": ["create_issue"]})
        memory = MemorySaver()
        thread_cfg = {"configurable": {"thread_id": "real-graph-standard-args"}}
        counter = _ToolExecutionCounter()

        # ═══ Phase 1: Initial request → HITL interrupt on create_issue ═══
        llm1 = _ComplexArgsLLM()
        runnable1 = _build_runnable(memory, llm1, counter)

        r1 = runnable1.invoke(
            {"messages": [HumanMessage(content="Search for feature X and create a task")]},
            config={**thread_cfg},
        )

        assert r1["execution_finished"] is False, "Should pause for HITL"
        assert r1.get("hitl_interrupt"), "Should have HITL interrupt payload"
        assert r1["hitl_interrupt"]["tool_name"] == "create_issue"
        # Verify args preserved in interrupt payload
        assert r1["hitl_interrupt"]["tool_args"]["project_key"] == "PROJ"

        # Pre-resume: search_issues ran once; create_issue was blocked before run
        assert counter.search_issues == 1, (
            f"search_issues should have run exactly once before HITL, "
            f"got {counter.search_issues}"
        )
        assert counter.create_issue == 0, (
            "create_issue must NOT execute before HITL approval, "
            f"got {counter.create_issue}"
        )

        # ═══ Phase 2: Resume with a BRAND NEW runnable (new graph, same memory) ═══
        llm2 = _ComplexArgsLLM()
        runnable2 = _build_runnable(memory, llm2, counter)

        r2 = runnable2.invoke(
            {"hitl_resume": True, "hitl_action": "approve", "hitl_value": ""},
            config={**thread_cfg},
        )

        # ═══ Assertions ═══
        assert r2.get("execution_finished") is True, (
            f"Pipeline should complete after HITL resume. Got: {r2}"
        )
        assert "PROJ-123" in r2.get("output", ""), (
            f"Final output should mention PROJ-123. Got: {r2.get('output')}"
        )

        # KEY ASSERTION: LLM should be called exactly ONCE on resume
        # (to produce the final answer after seeing both tool results).
        # If the bug is present, it would be called multiple times as it
        # re-invokes tools from scratch.
        assert len(llm2.invocations) == 1, (
            f"Expected 1 LLM invocation on resume (tool results → final answer), "
            f"got {len(llm2.invocations)}. Agent re-invoked tools from scratch!"
        )

        # CRITICAL: side-effect assertions — non-sensitive sibling must NOT re-run
        assert counter.search_issues == 1, (
            f"search_issues was re-executed on HITL resume "
            f"(count={counter.search_issues}). Issue #4333 regression: "
            f"non-sensitive sibling tools must not re-execute."
        )
        assert counter.create_issue == 1, (
            f"create_issue should run exactly once (only on resume), "
            f"got {counter.create_issue}"
        )

        # Verify the resume LLM call received BOTH tool results
        post_resume_msgs = llm2.invocations[0]
        tool_msgs = [m for m in post_resume_msgs if isinstance(m, ToolMessage)]
        tool_contents = [str(m.content) for m in tool_msgs]
        assert "search-result:3 issues found" in tool_contents, (
            "search_issues result must survive HITL round-trip (from pending_messages)"
        )
        assert "create-result:PROJ-123" in tool_contents, (
            "create_issue result must be present (tool executed on resume)"
        )

    def test_anthropic_thinking_model_resume_preserves_thinking_blocks(self):
        """Anthropic thinking model: HITL resume preserves thinking content blocks.

        This is the EXACT bug scenario from #4333:
        - Anthropic Extended Thinking model emits thinking + text + tool_use
        - Tool is sensitive → HITL interrupt
        - On resume, the thinking blocks must survive in the AIMessage
        - The LLM must NOT be re-invoked from scratch
        """
        configure_sensitive_tools({"jira": ["create_issue"]})
        memory = MemorySaver()
        thread_cfg = {"configurable": {"thread_id": "real-graph-thinking-args"}}
        counter = _ToolExecutionCounter()

        # ═══ Phase 1: Initial request → HITL ═══
        llm1 = _ThinkingComplexArgsLLM()
        runnable1 = _build_runnable(memory, llm1, counter)

        r1 = runnable1.invoke(
            {"messages": [HumanMessage(content="Search and create task")]},
            config={**thread_cfg},
        )

        assert r1["execution_finished"] is False
        assert r1["hitl_interrupt"]["tool_name"] == "create_issue"
        assert counter.search_issues == 1
        assert counter.create_issue == 0

        # ═══ Phase 2: Resume with NEW runnable (new graph) ═══
        llm2 = _ThinkingComplexArgsLLM()
        runnable2 = _build_runnable(memory, llm2, counter)

        r2 = runnable2.invoke(
            {"hitl_resume": True, "hitl_action": "approve", "hitl_value": ""},
            config={**thread_cfg},
        )

        # ═══ Assertions ═══
        assert r2.get("execution_finished") is True, (
            f"Pipeline should complete. Got: {r2}"
        )
        assert "PROJ-123" in r2.get("output", "")

        # Single LLM call on resume
        assert len(llm2.invocations) == 1, (
            f"Expected 1 LLM invocation, got {len(llm2.invocations)}. "
            f"Anthropic thinking model lost context and re-invoked tools."
        )

        # CRITICAL: side-effect assertions for thinking-model HITL resume
        assert counter.search_issues == 1, (
            f"search_issues was re-executed on Anthropic thinking-model HITL "
            f"resume (count={counter.search_issues}). Issue #4333 regression."
        )
        assert counter.create_issue == 1, (
            f"create_issue should run exactly once (only on resume), "
            f"got {counter.create_issue}"
        )

        # Verify thinking blocks survived in the AIMessage
        post_resume_msgs = llm2.invocations[0]
        ai_msgs = [m for m in post_resume_msgs
                   if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)]
        assert ai_msgs, "Expected AIMessage with tool_calls in resumed LLM call"

        last_ai = ai_msgs[-1]
        # The content should be a list with thinking blocks preserved
        if isinstance(last_ai.content, list):
            thinking_blocks = [
                b for b in last_ai.content
                if isinstance(b, dict) and b.get("type") == "thinking"
            ]
            assert thinking_blocks, (
                "Thinking blocks were stripped during HITL resume — "
                "Anthropic reasoning continuity broken"
            )
            assert thinking_blocks[0]["signature"] == "sig_planning_001"

        # Tool results present
        tool_msgs = [m for m in post_resume_msgs if isinstance(m, ToolMessage)]
        tool_contents = [str(m.content) for m in tool_msgs]
        assert "search-result:3 issues found" in tool_contents
        assert "create-result:PROJ-123" in tool_contents

    def test_multiple_sensitive_tools_sequential_approval(self):
        """Multiple sensitive tool calls approved one at a time.

        Tests the scenario where LLM calls two sensitive tools in one batch:
        1. First batch: search_issues + create_issue → first tool pauses
        2. Resume (approve) → continue → second tool may pause
        3. Resume (approve) → final answer

        This validates that the checkpoint correctly preserves the entire
        conversation history across multiple HITL interrupts.
        """
        # Both tools are sensitive
        configure_sensitive_tools({"jira": ["create_issue", "search_issues"]})
        memory = MemorySaver()
        thread_cfg = {"configurable": {"thread_id": "real-graph-multi-sensitive"}}
        counter = _ToolExecutionCounter()

        # ═══ Phase 1: Initial request → first sensitive tool interrupted ═══
        llm1 = _ComplexArgsLLM()
        runnable1 = _build_runnable(memory, llm1, counter)

        r1 = runnable1.invoke(
            {"messages": [HumanMessage(content="Do the things")]},
            config={**thread_cfg},
        )

        assert r1["execution_finished"] is False
        hitl1 = r1.get("hitl_interrupt")
        assert hitl1, "Should have first HITL interrupt"
        first_blocked_tool = hitl1["tool_name"]
        assert first_blocked_tool in ("search_issues", "create_issue"), (
            f"Unexpected blocked tool: {first_blocked_tool}"
        )

        # ═══ Phase 2: Approve first tool → expect second tool interrupt ═══
        llm2 = _ComplexArgsLLM()
        runnable2 = _build_runnable(memory, llm2, counter)

        r2 = runnable2.invoke(
            {"hitl_resume": True, "hitl_action": "approve", "hitl_value": ""},
            config={**thread_cfg},
        )

        if r2.get("execution_finished") is False and r2.get("hitl_interrupt"):
            # Second sensitive tool was blocked — approve it too
            second_blocked = r2["hitl_interrupt"]["tool_name"]
            assert second_blocked in ("search_issues", "create_issue")

            # ═══ Phase 3: Approve second tool → pipeline completes ═══
            llm3 = _ComplexArgsLLM()
            runnable3 = _build_runnable(memory, llm3, counter)

            r3 = runnable3.invoke(
                {"hitl_resume": True, "hitl_action": "approve", "hitl_value": ""},
                config={**thread_cfg},
            )

            assert r3.get("execution_finished") is True, (
                f"Pipeline should complete after both approvals. Got: {r3}"
            )
            assert "PROJ-123" in r3.get("output", "")
        else:
            # Both tools ran after first approval (only one gets blocked at a time
            # because the guard processes sequentially)
            assert r2.get("execution_finished") is True, (
                f"Expected pipeline to complete. Got: {r2}"
            )
            assert "PROJ-123" in r2.get("output", "")

        # CRITICAL: each tool must execute exactly once across the full
        # multi-interrupt round-trip. If an HITL approve causes the LLM to
        # restart from scratch, these counters will exceed 1.
        assert counter.search_issues == 1, (
            f"search_issues executed {counter.search_issues} times across the "
            f"multi-sensitive HITL round-trip; expected exactly 1."
        )
        assert counter.create_issue == 1, (
            f"create_issue executed {counter.create_issue} times across the "
            f"multi-sensitive HITL round-trip; expected exactly 1."
        )
