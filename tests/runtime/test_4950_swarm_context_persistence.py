import uuid
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.assistant import (
    Assistant,
    _extract_subagent_output,
    _extract_task_for_agent,
)
from elitea_sdk.runtime.tools.application import Application


class DummyEliteaRuntime:
    def get_mcp_toolkits(self):
        return []


def _make_application_tool(name: str, output: str = "agent output") -> Application:
    subapp = MagicMock()
    subapp.invoke.return_value = {"output": output}
    return Application(
        name=name,
        description=f"Agent {name}",
        application=subapp,
        return_type="str",
        client=None,
        is_subgraph=True,
    )


def _build_swarm_assistant(agent_tools, memory=None) -> Assistant:
    return Assistant(
        elitea=DummyEliteaRuntime(),
        data={
            "instructions": "You are the main agent",
            "tools": [],
            "meta": {"internal_tools": ["swarm"]},
            "internal_tools": ["swarm"],
        },
        client=MagicMock(),
        tools=agent_tools,
        memory=memory or MemorySaver(),
        app_type="agent",
    )


class TestExtractSubagentOutput:

    def test_pipeline_state_dict_extracts_last_assistant_content(self):
        # The exact shape that was producing the str(dict) noise on the live env:
        # the sub-agent returns its pipeline state with no "output" key, so the
        # naive fallback dumped the whole dict (thread_id, execution_finished,
        # context_info, hitl_decisions and all) into the next sub-agent's task.
        result = {
            "messages": [
                {"role": "assistant", "content": '[{"id": 1, "title": "issue"}]'},
            ],
            "thread_id": None,
            "execution_finished": True,
            "context_info": {"message_count": 0, "token_count": 0, "summarized": False},
            "hitl_decisions": [],
        }

        out = _extract_subagent_output(result)

        assert out == '[{"id": 1, "title": "issue"}]'
        assert "thread_id" not in out
        assert "execution_finished" not in out
        assert "context_info" not in out

    def test_explicit_output_key_wins(self):
        result = {"output": "clean answer", "messages": [{"role": "assistant", "content": "ignored"}]}
        assert _extract_subagent_output(result) == "clean answer"

    def test_message_objects_are_handled(self):
        result = {"messages": [HumanMessage(content="ignored"), AIMessage(content="real answer")]}
        assert _extract_subagent_output(result) == "real answer"

    def test_anthropic_list_content(self):
        result = {
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "block answer"}]},
            ],
        }
        assert _extract_subagent_output(result) == "block answer"

    def test_string_result_passes_through(self):
        assert _extract_subagent_output("plain") == "plain"

    def test_skips_user_messages_when_walking_tail(self):
        result = {
            "messages": [
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "should be ignored"},
            ],
        }
        assert _extract_subagent_output(result) == "first"


class TestInvokeApplicationTaskExtraction:

    def test_structured_brief_with_task_user_and_prior_sections(self):
        messages = [
            HumanMessage(content="Get and fix all security issues"),
            AIMessage(
                content="I'll retrieve the issues first.",
                tool_calls=[{"name": "transfer_to_issuesprovider", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="Successfully transferred to issuesprovider", tool_call_id="tc1"),
            AIMessage(
                content=[{"type": "text", "text": "Here are the issues: [issue1, issue2]"}],
                tool_calls=[{"name": "transfer_to_main_agent", "args": {}, "id": "tc2"}],
            ),
            ToolMessage(content="Successfully transferred to main_agent", tool_call_id="tc2"),
            AIMessage(
                content="I found 2 security issues. Fix both CVEs: issue1 and issue2.",
                tool_calls=[{"name": "transfer_to_securityresolver", "args": {}, "id": "tc3"}],
            ),
            ToolMessage(content="Successfully transferred to SecurityResolver", tool_call_id="tc3"),
        ]

        task = _extract_task_for_agent(messages, "securityresolver")

        assert "## Your Task" in task
        assert "Fix both CVEs: issue1 and issue2" in task
        assert "## Original User Request" in task
        assert "Get and fix all security issues" in task
        assert "## Prior Agent Outputs" in task
        assert "Here are the issues: [issue1, issue2]" in task
        # Orchestrator's intermediate transition dropped
        assert "I'll retrieve the issues first" not in task
        assert "Successfully transferred" not in task

        # Section ordering: task first, user request, then references
        task_idx = task.index("## Your Task")
        req_idx = task.index("## Original User Request")
        ref_idx = task.index("## Prior Agent Outputs")
        assert task_idx < req_idx < ref_idx

    def test_aggregates_prior_agent_outputs_for_chained_handoffs(self):
        # Issue #4950 production case: the orchestrator's handoff message uses
        # anaphora ("hand off all 4 issues") referring to data emitted by prior
        # sub-agents. The next sub-agent must see those outputs, not just the
        # transition narrative.
        messages = [
            HumanMessage(content="Fix all open issues"),
            AIMessage(
                content="Retrieving issues first.",
                tool_calls=[{"name": "transfer_to_issuesprovider", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="Successfully transferred to issuesprovider", tool_call_id="tc1"),
            AIMessage(
                content="Issues: #11, #14, #20, #21. #11 and #14 are security.",
                tool_calls=[{"name": "transfer_to_main_agent", "args": {}, "id": "tc2"}],
            ),
            ToolMessage(content="Successfully transferred to main_agent", tool_call_id="tc2"),
            AIMessage(
                content="I've got 4 issues, 2 security.",
                tool_calls=[{"name": "transfer_to_securityissuesresolver", "args": {}, "id": "tc3"}],
            ),
            ToolMessage(content="Successfully transferred to SecurityIssuesResolver", tool_call_id="tc3"),
            AIMessage(
                content="Security fixes for #11 and #14 applied: patch X and Y.",
                tool_calls=[{"name": "transfer_to_main_agent", "args": {}, "id": "tc4"}],
            ),
            ToolMessage(content="Successfully transferred to main_agent", tool_call_id="tc4"),
            AIMessage(
                content="Now hand off all 4 issues to GuidedDeveloper.",
                tool_calls=[{"name": "transfer_to_guideddeveloper", "args": {}, "id": "tc5"}],
            ),
            ToolMessage(content="Successfully transferred to GuidedDeveloper", tool_call_id="tc5"),
        ]

        task = _extract_task_for_agent(messages, "GuidedDeveloper")

        assert "## Your Task" in task
        assert "hand off all 4 issues" in task
        assert "## Original User Request" in task
        assert "Fix all open issues" in task
        assert "## Prior Agent Outputs" in task
        assert "Issues: #11, #14, #20, #21" in task
        assert "Security fixes for #11 and #14 applied: patch X and Y." in task
        # Orchestrator transitions dropped — they restate context already in the
        # sub-agent results that follow them
        assert "Retrieving issues first" not in task
        assert "I've got 4 issues, 2 security" not in task
        assert "Successfully transferred" not in task

    def test_falls_back_to_human_message_when_no_assigning_aimessage(self):
        messages = [HumanMessage(content="Initial user request")]

        assert _extract_task_for_agent(messages, "some_agent") == "Initial user request"

    def test_handles_list_content_in_assigning_aimessage(self):
        messages = [
            HumanMessage(content="Do the work"),
            AIMessage(
                content=[
                    {"type": "text", "text": "Please process these records for the data agent."},
                    {"type": "tool_use", "id": "tc1", "name": "transfer_to_dataagent", "input": {}},
                ],
                tool_calls=[{"name": "transfer_to_dataagent", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="Successfully transferred to dataagent", tool_call_id="tc1"),
        ]

        task = _extract_task_for_agent(messages, "dataagent")

        assert "process these records" in task
        assert "[{" not in task

    def test_session_boundary_anchors_at_latest_human_message(self):
        # Earlier turns (already-completed swarm runs) must not bleed into the
        # current turn's transcript. The session anchor is the latest
        # HumanMessage at-or-before the assigning AIMessage.
        messages = [
            HumanMessage(content="OLD turn user message"),
            AIMessage(content="OLD turn assistant reply"),
            HumanMessage(content="NEW turn task"),
            AIMessage(
                content="Routing.",
                tool_calls=[{"name": "transfer_to_worker", "args": {}, "id": "tc1"}],
            ),
        ]

        task = _extract_task_for_agent(messages, "Worker")

        assert "NEW turn task" in task
        assert "OLD turn" not in task

    def test_empty_messages_returns_empty_string(self):
        assert _extract_task_for_agent([], "any_agent") == ""

    def test_agent_name_normalization_matches_langgraph_swarm(self):
        # langgraph_swarm.create_handoff_tool lowercases agent_name and collapses
        # whitespace; the helper must mirror that or never match the tool_call
        # and silently fall back to the HumanMessage — the symptom on #4950.
        messages = [
            HumanMessage(content="user request"),
            AIMessage(
                content="Delegating the security work to the resolver.",
                tool_calls=[{"name": "transfer_to_securityresolver", "args": {}, "id": "tc1"}],
            ),
        ]

        task_camel = _extract_task_for_agent(messages, "SecurityResolver")
        assert "Delegating the security work" in task_camel
        assert "user request" in task_camel

        messages_ws = [
            HumanMessage(content="ws request"),
            AIMessage(
                content="Hand off to data team.",
                tool_calls=[{"name": "transfer_to_data_agent", "args": {}, "id": "tc1"}],
            ),
        ]
        task_ws = _extract_task_for_agent(messages_ws, "Data Agent")
        assert "Hand off to data team" in task_ws
        assert "ws request" in task_ws


class TestSwarmResultAdapterContract:

    def test_invoke_returns_pylon_contract(self):
        agent_tool = _make_application_tool("subagent")
        assistant = _build_swarm_assistant([agent_tool])

        fake_compiled = MagicMock()
        fake_compiled.invoke.return_value = {
            "messages": [HumanMessage(content="hello"), AIMessage(content="world")],
        }

        with patch("langgraph_swarm.create_swarm") as mock_create_swarm:
            mock_swarm = MagicMock()
            mock_swarm.compile.return_value = fake_compiled
            mock_create_swarm.return_value = mock_swarm

            adapter = assistant._create_swarm_agent(
                all_tools=[agent_tool],
                agent_tools=[agent_tool],
            )

        result = adapter.invoke(
            {"messages": [HumanMessage(content="hello")]},
            {"configurable": {"thread_id": str(uuid.uuid4())}},
        )

        assert result == {"output": "world", "thread_id": None, "execution_finished": True}


class TestSwarmPersistentCheckpointer:

    def test_persistent_memory_used_as_checkpointer(self):
        persistent_memory = MemorySaver()
        agent_tool = _make_application_tool("subagent")
        assistant = _build_swarm_assistant([agent_tool], memory=persistent_memory)

        captured_checkpointer = {}

        def fake_compile(**kwargs):
            captured_checkpointer["value"] = kwargs.get("checkpointer")
            mock = MagicMock()
            mock.invoke.return_value = {"messages": [AIMessage(content="done")]}
            return mock

        with patch("langgraph_swarm.create_swarm") as mock_create_swarm:
            mock_swarm = MagicMock()
            mock_swarm.compile.side_effect = fake_compile
            mock_create_swarm.return_value = mock_swarm

            assistant._create_swarm_agent(
                all_tools=[agent_tool],
                agent_tools=[agent_tool],
            )

        assert captured_checkpointer.get("value") is persistent_memory

    def test_fresh_memorysaver_used_when_memory_is_none(self):
        agent_tool = _make_application_tool("subagent")
        assistant = _build_swarm_assistant([agent_tool], memory=None)
        assistant.memory = None

        captured_checkpointer = {}

        def fake_compile(**kwargs):
            captured_checkpointer["value"] = kwargs.get("checkpointer")
            mock = MagicMock()
            mock.invoke.return_value = {"messages": [AIMessage(content="done")]}
            return mock

        with patch("langgraph_swarm.create_swarm") as mock_create_swarm:
            mock_swarm = MagicMock()
            mock_swarm.compile.side_effect = fake_compile
            mock_create_swarm.return_value = mock_swarm

            assistant._create_swarm_agent(
                all_tools=[agent_tool],
                agent_tools=[agent_tool],
            )

        assert captured_checkpointer.get("value") is not None


class TestSwarmStaleCheckpointClearing:

    def test_stale_messages_cleared_before_fresh_turn(self):
        from langchain_core.messages import RemoveMessage

        thread_id = str(uuid.uuid4())
        memory = MemorySaver()

        agent_tool = _make_application_tool("subagent")
        assistant = _build_swarm_assistant([agent_tool], memory=memory)

        update_state_calls = []

        fake_compiled = MagicMock()
        fake_compiled.invoke.return_value = {"messages": [AIMessage(content="fresh result")]}

        # Simulate a prior completed checkpoint
        prior_msg = AIMessage(content="old result", id="msg-old-1")
        fake_state = MagicMock()
        fake_state.next = []  # completed — at END
        fake_state.values = {"messages": [prior_msg]}
        fake_compiled.get_state.return_value = fake_state

        def fake_update_state(cfg, update):
            update_state_calls.append(update)

        fake_compiled.update_state = fake_update_state

        with patch("langgraph_swarm.create_swarm") as mock_create_swarm:
            mock_swarm = MagicMock()
            mock_swarm.compile.return_value = fake_compiled
            mock_create_swarm.return_value = mock_swarm

            adapter = assistant._create_swarm_agent(
                all_tools=[agent_tool],
                agent_tools=[agent_tool],
            )

        config = {"configurable": {"thread_id": thread_id}}
        adapter.invoke({"messages": [HumanMessage(content="new request")]}, config)

        remove_calls = [
            call for call in update_state_calls
            if isinstance(call, dict)
            and "messages" in call
            and any(isinstance(m, RemoveMessage) for m in call["messages"])
        ]
        assert len(remove_calls) > 0


# ---------------------------------------------------------------------------
# Multi-turn integration test: a fresh Assistant per turn sharing the same
# MemorySaver and thread_id, mirroring how pylon invokes the swarm from the UI.
# pylon's prepare_invoke_input sends chat_history + new HumanMessage each turn
# (see centry/pylon_indexer/.../utils/agent_execution_common.py:856), so each
# turn the swarm receives the full conversation as input.
# ---------------------------------------------------------------------------


class _RecordingApp:
    """Stand-in for a sub-agent's compiled graph: records every invoke."""

    def __init__(self, output: str):
        self.output = output
        self.calls = []

    def invoke(self, payload, config=None):
        self.calls.append(dict(payload) if isinstance(payload, dict) else payload)
        return {"output": self.output}


class _ScriptedOrchestratorLLM:
    """Fake orchestrator LLM that:

    - Emits ``transfer_to_provider`` for any HumanMessage containing 'generate'
    - Emits ``transfer_to_fixer`` for any HumanMessage containing 'fix' (with
      anaphoric handoff text — the exact production failure mode)
    - Emits a final response (no tool_calls) once a sub-agent has handed back
      in the current turn.
    """

    temperature = 0
    max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {"temperature": 0, "max_tokens": 1000}

    def bind_tools(self, tools, **_kwargs):
        return _ScriptedBound(list(tools))

    def invoke(self, messages, config=None):
        return _ScriptedBound([]).invoke(messages, config=config)


class _ScriptedBound:
    def __init__(self, tools):
        self.tools = tools

    def invoke(self, messages, config=None, **_kwargs):
        latest_human_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                latest_human_idx = i
                break
        if latest_human_idx is None:
            return AIMessage(content="empty")

        # If a sub-agent has already handed back in this turn, emit final.
        for msg in messages[latest_human_idx + 1:]:
            if isinstance(msg, AIMessage):
                tc_names = [
                    tc.get("name") for tc in (getattr(msg, "tool_calls", None) or [])
                ]
                if "transfer_to_main_agent" in tc_names:
                    return AIMessage(content="Done.")

        human = messages[latest_human_idx]
        text = human.content if isinstance(human.content, str) else str(human.content)
        lower = text.lower()

        if "generate" in lower:
            target = "transfer_to_provider"
            handoff_text = "Routing to provider to generate items."
        elif "fix" in lower:
            target = "transfer_to_fixer"
            # Deliberate anaphora — the production failure mode the helper must compensate for.
            handoff_text = "Now routing the items to the fixer."
        else:
            return AIMessage(content="unknown request")

        return AIMessage(
            content=handoff_text,
            tool_calls=[{
                "name": target,
                "args": {},
                "id": f"tc-{len(messages)}",
                "type": "tool_call",
            }],
        )


class TestSwarmMultiTurnIntegration:
    """Each turn = a fresh Assistant + same MemorySaver + same thread_id.
    Mirrors pylon's UI invocation: full chat_history + new HumanMessage per turn.
    """

    def _build_runnable(self, llm, memory, provider_app, fixer_app):
        # Fresh Application instances per turn — _create_swarm_agent mutates
        # tool.is_subgraph = False during pipeline detection (assistant.py:1180),
        # so reusing the same instance across turns silently demotes the pipeline
        # peer to an LLM peer. Real pylon rebuilds tools per request; we mirror that.
        provider_tool = Application(
            name="Provider",
            description="Generates items",
            application=provider_app,
            return_type="str",
            client=None,
            is_subgraph=True,
        )
        fixer_tool = Application(
            name="Fixer",
            description="Fixes items of a given kind",
            application=fixer_app,
            return_type="str",
            client=None,
            is_subgraph=True,
        )
        return Assistant(
            elitea=DummyEliteaRuntime(),
            data={
                "instructions": "Coordinate sub-agents",
                "tools": [],
                "meta": {"internal_tools": ["swarm"]},
                "internal_tools": ["swarm"],
            },
            client=llm,
            tools=[provider_tool, fixer_tool],
            memory=memory,
            app_type="agent",
        ).runnable()

    def test_subagent_in_turn_2_sees_prior_turn_output(self):
        memory = MemorySaver()
        thread_id = str(uuid.uuid4())

        provider_app = _RecordingApp(output='[{"id": 1, "kind": "X"}, {"id": 2, "kind": "Y"}]')
        fixer_app = _RecordingApp(output="fixed")
        llm = _ScriptedOrchestratorLLM()

        config = {"configurable": {"thread_id": thread_id}}

        # ---- Turn 1: "generate items" ----
        runnable_t1 = self._build_runnable(llm, memory, provider_app, fixer_app)
        h1 = HumanMessage(content="please generate items")
        runnable_t1.invoke({"messages": [h1]}, config=config)

        assert len(provider_app.calls) == 1, (
            "Provider should have been invoked exactly once in turn 1"
        )
        assert len(fixer_app.calls) == 0, "Fixer must not run in turn 1"

        # Capture the conversation as pylon would: full message history.
        post_t1_state = runnable_t1._graph.get_state(config)
        history = list(post_t1_state.values.get("messages", []))
        assert any(isinstance(m, HumanMessage) for m in history)

        # ---- Turn 2: "fix only the X-kind items" — fresh Assistant + fresh tools ----
        runnable_t2 = self._build_runnable(llm, memory, provider_app, fixer_app)
        h2 = HumanMessage(content="fix only the X-kind items")
        # Mirror pylon's prepare_invoke_input: chat_history + new user_message
        runnable_t2.invoke({"messages": history + [h2]}, config=config)

        assert len(fixer_app.calls) == 1, (
            "Fixer should have been invoked exactly once in turn 2"
        )
        # _create_swarm_agent mutates is_subgraph=False on pipeline peers
        # (assistant.py:1180), so formulate_query wraps the task as
        # {"input": [HumanMessage(content=task)], ...}. Pull the content from
        # the first HumanMessage to get the actual task string.
        payload = fixer_app.calls[0]
        wrapped_input = payload.get("input")
        assert isinstance(wrapped_input, list) and wrapped_input, (
            f"Expected formulate_query-wrapped input, got: {payload!r}"
        )
        fixer_task = wrapped_input[0].content

        # The production gap: even though the orchestrator's turn-2 handoff uses
        # anaphora ("the items") without inlining the data, the fixer's task
        # must surface turn 1's provider output as reference.
        assert '"kind": "X"' in fixer_task or '"id": 1' in fixer_task, (
            f"Fixer task in turn 2 does not include turn 1's provider output. "
            f"Multi-turn fallback failed.\nTask:\n{fixer_task}"
        )
        # The structured-brief shape is preserved
        assert "## Your Task" in fixer_task
        assert "## Original User Request" in fixer_task
        assert "fix only the X-kind items" in fixer_task
