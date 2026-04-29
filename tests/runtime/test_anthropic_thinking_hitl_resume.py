"""Regression tests for Anthropic Extended Thinking + sensitive-tool HITL resume.

Anthropic reasoning models return assistant messages whose ``content`` is a list
of blocks beginning with a signed ``thinking`` block. Anthropic's stateless API
contract requires those thinking blocks to be present on the assistant message
preceding any subsequent ``tool_result`` so reasoning continuity is maintained.

Before the fix in this file's companion changes, the HITL resume path replaced
the original tool-calling AIMessage with a synthetic ``AIMessage(content='',
tool_calls=[…])`` — stripping the thinking block — which caused Anthropic
reasoning models to lose the original task plan and start over after resume.
This regression suite locks in the fix:

* ``LLMNode._build_resume_completion`` reuses the original AIMessage (with
  thinking blocks intact) when ``hitl_ctx['original_ai_message']`` is provided
  and a tool_call matches the resumed action.
* ``LangGraphAgentRunnable._extract_original_ai_message`` captures that message
  from ``_pending_messages`` at resume time so the LLM node receives it.

GPT and Anthropic-without-reasoning paths are not affected because they don't
emit thinking blocks; the helper falls back to the existing synthetic AIMessage
in those cases.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.base import message_to_dict
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.langchain.langraph_agent import LangGraphAgentRunnable
from elitea_sdk.runtime.middleware.sensitive_tool_guard import SensitiveToolGuardMiddleware
from elitea_sdk.runtime.toolkits.security import configure_sensitive_tools, reset_sensitive_tools
from elitea_sdk.runtime.tools.llm import LLMNode


THINKING_TEXT = "I should call the danger tool because the user explicitly asked for it."
THINKING_SIGNATURE = "sig_abc123"
ORIGINAL_TOOL_CALL_ID = "toolu_01ABC"


def _make_anthropic_thinking_ai_message(
    tool_name: str = "danger",
    tool_args: dict | None = None,
    tool_call_id: str = ORIGINAL_TOOL_CALL_ID,
) -> AIMessage:
    """Build an AIMessage shaped like a real Anthropic Extended Thinking response."""
    tool_args = tool_args if tool_args is not None else {"target": "demo"}
    return AIMessage(
        content=[
            {
                "type": "thinking",
                "thinking": THINKING_TEXT,
                "signature": THINKING_SIGNATURE,
            },
            {
                "type": "tool_use",
                "id": tool_call_id,
                "name": tool_name,
                "input": tool_args,
            },
        ],
        tool_calls=[
            {"name": tool_name, "args": tool_args, "id": tool_call_id},
        ],
    )


def _make_anthropic_text_tool_ai_message(
    tool_name: str = "danger",
    tool_args: dict | None = None,
    tool_call_id: str = ORIGINAL_TOOL_CALL_ID,
) -> AIMessage:
    """Build a later Anthropic tool-call turn with text + tool_use blocks."""
    tool_args = tool_args if tool_args is not None else {"target": "demo"}
    return AIMessage(
        content=[
            {
                "type": "text",
                "text": "I created the branch. Now I will update the PR.",
            },
            {
                "type": "tool_use",
                "id": tool_call_id,
                "name": tool_name,
                "input": tool_args,
            },
        ],
        tool_calls=[
            {"name": tool_name, "args": tool_args, "id": tool_call_id},
        ],
    )


# ───────────────────────────────────────────────────────────────────────────────
# Unit: LangGraphAgentRunnable._extract_original_ai_message
# ───────────────────────────────────────────────────────────────────────────────


def test_extract_original_ai_message_returns_dict_for_matching_tool_call():
    ai = _make_anthropic_thinking_ai_message(
        tool_name="danger", tool_args={"target": "demo"}
    )
    pending = [
        message_to_dict(HumanMessage(content="do it")),
        message_to_dict(ai),
    ]

    result = LangGraphAgentRunnable._extract_original_ai_message(
        pending_msgs_dicts=pending,
        tool_name="danger",
        tool_args={"target": "demo"},
    )

    assert result is not None
    assert result["type"] == "ai"
    assert isinstance(result["data"]["content"], list)
    assert any(
        isinstance(b, dict) and b.get("type") == "thinking"
        for b in result["data"]["content"]
    )


def test_extract_original_ai_message_returns_none_for_mismatched_args():
    ai = _make_anthropic_thinking_ai_message(
        tool_name="danger", tool_args={"target": "demo"}
    )
    pending = [message_to_dict(ai)]

    result = LangGraphAgentRunnable._extract_original_ai_message(
        pending_msgs_dicts=pending,
        tool_name="danger",
        tool_args={"target": "other"},
    )

    assert result is None


def test_extract_original_ai_message_returns_none_when_no_ai_in_pending():
    pending = [message_to_dict(HumanMessage(content="hello"))]

    result = LangGraphAgentRunnable._extract_original_ai_message(
        pending_msgs_dicts=pending, tool_name="danger", tool_args={}
    )

    assert result is None


# ───────────────────────────────────────────────────────────────────────────────
# Unit: LLMNode._build_resume_completion
# ───────────────────────────────────────────────────────────────────────────────


def test_build_resume_completion_reuses_original_when_thinking_present():
    original = _make_anthropic_thinking_ai_message()
    hitl_ctx = {
        "tool_name": "danger",
        "tool_args": {"target": "demo"},
        "tool_call_id": "uuid_synth_should_be_overwritten",
        "original_ai_message": message_to_dict(original),
    }

    completion = LLMNode._build_resume_completion(hitl_ctx, messages=[])

    assert completion is not None
    # Thinking blocks must be preserved in the reused completion's content.
    assert isinstance(completion.content, list)
    thinking_blocks = [
        b for b in completion.content
        if isinstance(b, dict) and b.get("type") == "thinking"
    ]
    assert thinking_blocks, "Expected thinking block to survive resume"
    assert thinking_blocks[0]["thinking"] == THINKING_TEXT
    assert thinking_blocks[0]["signature"] == THINKING_SIGNATURE
    # Tool call id from the original message must replace the synthetic uuid so
    # the downstream ToolMessage matches Anthropic's stateless tool_use → tool_result pairing.
    assert hitl_ctx["tool_call_id"] == ORIGINAL_TOOL_CALL_ID
    assert any(
        tc.get("id") == ORIGINAL_TOOL_CALL_ID
        for tc in completion.tool_calls
    )


def test_build_resume_completion_reuses_original_text_only_structured_content():
    """Later Anthropic tool-call turns may contain text + tool_use blocks but
    no thinking block. Those structured content blocks still must survive
    HITL resume or the model loses continuity."""
    plain_ai = _make_anthropic_text_tool_ai_message()
    hitl_ctx = {
        "tool_name": "danger",
        "tool_args": {"target": "demo"},
        "tool_call_id": "synth",
        "original_ai_message": message_to_dict(plain_ai),
    }

    completion = LLMNode._build_resume_completion(hitl_ctx, messages=[])

    assert completion is not None
    assert isinstance(completion.content, list)
    assert completion.content[0]["type"] == "text"
    assert completion.content[0]["text"] == "I created the branch. Now I will update the PR."
    assert hitl_ctx["tool_call_id"] == ORIGINAL_TOOL_CALL_ID


def test_build_resume_completion_reuses_empty_content_ai_message():
    """Non-thinking models emit ``content=''`` when calling tools. The original
    AIMessage must still be reused (not replaced by a synthetic) so that:

    * its canonical tool_call ids survive the HITL round-trip, and
    * downstream sibling-skip logic in ``__perform_tool_calling`` matches
      the LLM's next-iteration tool_calls against the existing ToolMessages.

    See issue #4333.
    """
    plain_ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "danger", "args": {"target": "demo"}, "id": "call_1"},
            {"name": "danger", "args": {"target": "demo"}, "id": "call_2"},
        ],
    )
    hitl_ctx = {
        "tool_name": "danger",
        "tool_args": {"target": "demo"},
        "tool_call_id": "synth",
        "original_ai_message": message_to_dict(plain_ai),
    }

    completion = LLMNode._build_resume_completion(hitl_ctx, messages=[])

    assert isinstance(completion, AIMessage)
    # First matching tool_call's id is propagated into hitl_ctx
    assert hitl_ctx["tool_call_id"] == "call_1"
    # All original tool_calls survive (sibling skip in __perform_tool_calling
    # protects against double-execution)
    tc_ids = {tc.get("id") for tc in (completion.tool_calls or [])}
    assert tc_ids == {"call_1", "call_2"}


def test_build_resume_completion_returns_none_when_no_original_in_ctx():
    hitl_ctx = {
        "tool_name": "danger",
        "tool_args": {},
        "tool_call_id": "synth",
    }

    assert LLMNode._build_resume_completion(hitl_ctx, messages=[]) is None


def test_build_resume_completion_skips_reuse_when_already_in_messages():
    """Multi-tool sibling case: _trim_pending_messages keeps the AIMessage in
    the restored ``messages`` list. When the AIMessage is already present,
    _build_resume_completion returns the existing message so it is NOT
    duplicated. ``__perform_tool_calling`` then dedups via
    ``_append_completion_dedup`` and skips already-completed siblings via
    ``_tool_call_already_completed`` — no separate sentinel flag required.
    """
    original = _make_anthropic_thinking_ai_message()
    hitl_ctx = {
        "tool_name": "danger",
        "tool_args": {"target": "demo"},
        "tool_call_id": "synth",
        "original_ai_message": message_to_dict(original),
    }

    completion = LLMNode._build_resume_completion(
        hitl_ctx, messages=[original]  # same id already present
    )

    # Returns the existing AIMessage (deduplicated downstream)
    assert completion is not None
    assert isinstance(completion, AIMessage)
    # tool_call_id is updated to the original's id (so HITL reconciliation lines up)
    assert hitl_ctx['tool_call_id'] == 'toolu_01ABC'


# ───────────────────────────────────────────────────────────────────────────────
# Integration: full HITL resume cycle preserves thinking blocks
# ───────────────────────────────────────────────────────────────────────────────


class _DummyEliteARuntime:
    def get_mcp_toolkits(self):
        return []


class _ThinkingResumeLLM:
    """LLM stub that mimics Anthropic Extended Thinking responses.

    First invocation returns a thinking-prefixed AIMessage that calls the
    sensitive ``danger`` tool. After resume, the second invocation (which
    happens inside ``__perform_tool_calling`` after the tool runs) must
    receive the original thinking block in the prior assistant message.
    """

    def __init__(self):
        self.invocations: list[list] = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _ThinkingResumeLLMBound(self, tools, kwargs)

    def invoke(self, messages, config=None):
        return _ThinkingResumeLLMBound(self, [], {}).invoke(messages, config=config)


class _ThinkingResumeLLMBound:
    def __init__(self, root: _ThinkingResumeLLM, tools, kwargs):
        self.root = root
        self.tools = list(tools)
        self.kwargs = dict(kwargs)

    def invoke(self, messages, config=None):
        self.root.invocations.append(list(messages))
        # First call: model emits thinking + tool_use for the sensitive tool.
        if not any(isinstance(m, ToolMessage) for m in messages):
            return _make_anthropic_thinking_ai_message()
        # After tool result: model returns final answer (with another thinking
        # block, but no tool calls, so the loop terminates).
        return AIMessage(
            content=[
                {"type": "thinking", "thinking": "Tool succeeded.", "signature": "sig2"},
                {"type": "text", "text": "Done."},
            ]
        )


class _TextResumeLLM:
    """LLM stub for later Anthropic tool-call turns with text + tool_use."""

    def __init__(self):
        self.invocations: list[list] = []
        self.temperature = 0
        self.max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _TextResumeLLMBound(self, tools, kwargs)

    def invoke(self, messages, config=None):
        return _TextResumeLLMBound(self, [], {}).invoke(messages, config=config)


class _TextResumeLLMBound:
    def __init__(self, root: _TextResumeLLM, tools, kwargs):
        self.root = root
        self.tools = list(tools)
        self.kwargs = dict(kwargs)

    def invoke(self, messages, config=None):
        self.root.invocations.append(list(messages))
        if not any(isinstance(m, ToolMessage) for m in messages):
            return _make_anthropic_text_tool_ai_message()
        return AIMessage(
            content=[
                {"type": "text", "text": "PR updated."},
            ]
        )


def _build_runnable(memory: MemorySaver, llm: _ThinkingResumeLLM):
    danger_tool = StructuredTool.from_function(
        func=lambda target: f"danger-ok:{target}",
        name="danger",
        description="A sensitive tool.",
        metadata={"toolkit_type": "dummy", "toolkit_name": "dummy", "tool_name": "danger"},
    )
    assistant = Assistant(
        elitea=_DummyEliteARuntime(),
        data={"instructions": "use tools", "tools": [], "meta": {}},
        client=llm,
        tools=[danger_tool],
        memory=memory,
        app_type="predict",
        middleware=[SensitiveToolGuardMiddleware()],
    )
    return assistant.runnable()


def test_anthropic_thinking_blocks_survive_hitl_resume():
    """End-to-end: pause on sensitive tool, approve, verify next LLM call
    sees the original thinking block on the prior assistant message."""
    reset_sensitive_tools()
    try:
        configure_sensitive_tools({"dummy": ["danger"]})

        memory = MemorySaver()
        thread_config = {"configurable": {"thread_id": "anthropic-thinking-resume"}}

        first_llm = _ThinkingResumeLLM()
        first_runnable = _build_runnable(memory, first_llm)
        first_result = first_runnable.invoke(
            {"messages": [HumanMessage(content="run the danger tool on demo")]},
            config=thread_config,
        )
        assert first_result["execution_finished"] is False
        assert first_result["hitl_interrupt"]["tool_name"] == "danger"

        # Resume with approve.
        resume_llm = _ThinkingResumeLLM()
        resumed_runnable = _build_runnable(memory, resume_llm)
        resume_result = resumed_runnable.invoke(
            {"hitl_resume": True, "hitl_action": "approve", "hitl_value": ""},
            config=thread_config,
        )
        assert resume_result["execution_finished"] is True

        # The post-tool LLM call must see the original AIMessage with thinking
        # blocks and matching tool_call id, NOT a stripped synthetic AIMessage.
        assert resume_llm.invocations, "Expected at least one LLM invocation on resume"
        post_tool_invocation = resume_llm.invocations[0]

        ai_msgs_with_tool_calls = [
            m for m in post_tool_invocation
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        ]
        assert ai_msgs_with_tool_calls, (
            "Expected the resumed LLM call to receive the AIMessage that "
            "originally invoked the sensitive tool"
        )
        last_tool_calling_ai = ai_msgs_with_tool_calls[-1]

        # ── PRIMARY ASSERTION ──
        # Thinking block must be present in the assistant message preceding
        # the ToolMessage. Without the fix this is empty-string content.
        assert isinstance(last_tool_calling_ai.content, list), (
            "Anthropic-shaped AIMessage content must remain a list across "
            "HITL resume; if it became a string, thinking blocks were stripped."
        )
        thinking_blocks = [
            b for b in last_tool_calling_ai.content
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        assert thinking_blocks, (
            "Anthropic Extended Thinking reasoning continuity broken: "
            "the assistant message preceding tool_result no longer carries "
            "its signed `thinking` block after HITL resume."
        )
        assert thinking_blocks[0]["thinking"] == THINKING_TEXT
        assert thinking_blocks[0]["signature"] == THINKING_SIGNATURE

        # The ToolMessage must reference the ORIGINAL tool_call id (not a new
        # uuid) so Anthropic can pair tool_use ↔ tool_result.
        tool_msgs = [m for m in post_tool_invocation if isinstance(m, ToolMessage)]
        assert tool_msgs, "Expected ToolMessage from approved tool execution"
        assert tool_msgs[-1].tool_call_id == ORIGINAL_TOOL_CALL_ID
        assert any(
            tc.get("id") == ORIGINAL_TOOL_CALL_ID
            for tc in last_tool_calling_ai.tool_calls
        )
    finally:
        reset_sensitive_tools()


def test_anthropic_text_blocks_survive_hitl_resume():
    """Later Anthropic tool-calling turns can contain text + tool_use blocks
    without thinking. Resume must still reuse that original AIMessage instead
    of the empty synthetic fallback."""
    reset_sensitive_tools()
    try:
        configure_sensitive_tools({"dummy": ["danger"]})

        memory = MemorySaver()
        thread_config = {"configurable": {"thread_id": "anthropic-text-resume"}}

        first_llm = _TextResumeLLM()
        first_runnable = _build_runnable(memory, first_llm)
        first_result = first_runnable.invoke(
            {"messages": [HumanMessage(content="update the PR after creating the branch")]},
            config=thread_config,
        )
        assert first_result["execution_finished"] is False
        assert first_result["hitl_interrupt"]["tool_name"] == "danger"

        resume_llm = _TextResumeLLM()
        resumed_runnable = _build_runnable(memory, resume_llm)
        resume_result = resumed_runnable.invoke(
            {"hitl_resume": True, "hitl_action": "approve", "hitl_value": ""},
            config=thread_config,
        )
        assert resume_result["execution_finished"] is True

        assert resume_llm.invocations, "Expected at least one LLM invocation on resume"
        post_tool_invocation = resume_llm.invocations[0]

        ai_msgs_with_tool_calls = [
            m for m in post_tool_invocation
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        ]
        assert ai_msgs_with_tool_calls
        last_tool_calling_ai = ai_msgs_with_tool_calls[-1]

        assert isinstance(last_tool_calling_ai.content, list)
        assert last_tool_calling_ai.content[0]["type"] == "text"
        assert last_tool_calling_ai.content[0]["text"] == "I created the branch. Now I will update the PR."

        tool_msgs = [m for m in post_tool_invocation if isinstance(m, ToolMessage)]
        assert tool_msgs
        assert tool_msgs[-1].tool_call_id == ORIGINAL_TOOL_CALL_ID
        assert any(
            tc.get("id") == ORIGINAL_TOOL_CALL_ID
            for tc in last_tool_calling_ai.tool_calls
        )
    finally:
        reset_sensitive_tools()
