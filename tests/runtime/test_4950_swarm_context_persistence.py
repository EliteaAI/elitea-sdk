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

    def test_brief_contains_task_user_request_and_one_reference(self):
        # The peer's task is the orchestrator's handoff text + the user's
        # most recent request + a single reference (the most recent prior
        # sub-agent output) as a fallback safety net for orchestrator
        # anaphora. The reference is bounded to ONE payload — not the full
        # dump that bloated earlier shapes.
        messages = [
            HumanMessage(content="Get and fix all security issues"),
            AIMessage(
                content="I'll retrieve the issues first.",
                tool_calls=[{"name": "transfer_to_issuesprovider", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="Successfully transferred to issuesprovider", tool_call_id="tc1"),
            AIMessage(
                content="Issues retrieved: [{id:1,kind:security}, {id:2,kind:bug}]",
                tool_calls=[{"name": "transfer_to_main_agent", "args": {}, "id": "tc2"}],
            ),
            ToolMessage(content="Successfully transferred to main_agent", tool_call_id="tc2"),
            AIMessage(
                content="Resolve the security issue.",
                tool_calls=[{"name": "transfer_to_securityresolver", "args": {}, "id": "tc3"}],
            ),
            ToolMessage(content="Successfully transferred to SecurityResolver", tool_call_id="tc3"),
        ]

        task = _extract_task_for_agent(messages, "securityresolver")

        assert "## Your Task" in task
        assert "Resolve the security issue." in task
        assert "## User's Most Recent Request" in task
        assert "Get and fix all security issues" in task
        # Safety-net reference — exactly the most recent prior sub-agent output
        assert "## Last Available Result" in task
        assert "Issues retrieved: [{id:1,kind:security}, {id:2,kind:bug}]" in task
        assert "Successfully transferred" not in task

        # Section ordering
        assert (
            task.index("## Your Task")
            < task.index("## User's Most Recent Request")
            < task.index("## Last Available Result")
        )

    def test_only_most_recent_subagent_output_is_included_not_all(self):
        # The previous design dumped ALL prior sub-agent outputs (noisy).
        # The reference section must contain ONE payload only — the most
        # recent — even when multiple sub-agent results exist.
        messages = [
            HumanMessage(content="user"),
            AIMessage(
                content="route1",
                tool_calls=[{"name": "transfer_to_a", "args": {}, "id": "x1"}],
            ),
            AIMessage(
                content="OLDEST sub-agent output",
                tool_calls=[{"name": "transfer_to_main_agent", "args": {}, "id": "x2"}],
            ),
            AIMessage(
                content="route2",
                tool_calls=[{"name": "transfer_to_b", "args": {}, "id": "x3"}],
            ),
            AIMessage(
                content="NEWEST sub-agent output",
                tool_calls=[{"name": "transfer_to_main_agent", "args": {}, "id": "x4"}],
            ),
            AIMessage(
                content="anaphora",
                tool_calls=[{"name": "transfer_to_target", "args": {}, "id": "x5"}],
            ),
        ]

        task = _extract_task_for_agent(messages, "target")

        assert "NEWEST sub-agent output" in task
        assert "OLDEST sub-agent output" not in task

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

        # AgentResponse.to_dict() exposes the full standardized contract,
        # including the messages list. Assert the load-bearing fields without
        # over-specifying keys we don't own.
        assert result["output"] == "world"
        assert result["thread_id"] is None
        assert result["execution_finished"] is True
        assert isinstance(result["messages"], list) and len(result["messages"]) == 2


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


class TestSwarmContinuationInputRewrite:
    """On a follow-up turn (prior checkpoint exists) the adapter must pass
    only the new HumanMessage to graph.invoke. pylon's chat_history dicts
    don't carry sub-agent raw outputs; the langgraph checkpoint does.
    """

    def _make_adapter(self, memory, prior_messages, captured):
        agent_tool = _make_application_tool("subagent")
        assistant = _build_swarm_assistant([agent_tool], memory=memory)

        fake_compiled = MagicMock()
        fake_compiled.invoke.side_effect = lambda inp, cfg, **kw: (
            captured.update({"input": inp, "config": cfg})
            or {"messages": [AIMessage(content="ok")]}
        )

        if prior_messages:
            fake_state = MagicMock()
            fake_state.values = {"messages": prior_messages}
            fake_compiled.get_state.return_value = fake_state
        else:
            empty_state = MagicMock()
            empty_state.values = {"messages": []}
            fake_compiled.get_state.return_value = empty_state

        with patch("langgraph_swarm.create_swarm") as mock_create_swarm:
            mock_swarm = MagicMock()
            mock_swarm.compile.return_value = fake_compiled
            mock_create_swarm.return_value = mock_swarm
            return assistant._create_swarm_agent(
                all_tools=[agent_tool],
                agent_tools=[agent_tool],
            )

    def test_continuation_passes_only_new_human_message(self):
        captured = {}
        # Prior state has the full turn-1 stream including a raw sub-agent output
        prior = [
            HumanMessage(content="turn 1 request", id="m1"),
            AIMessage(
                content="route",
                id="m2",
                tool_calls=[{"name": "transfer_to_subagent", "args": {}, "id": "tc"}],
            ),
            AIMessage(content="raw subagent payload", id="m3"),
            AIMessage(content="turn 1 final", id="m4"),
        ]
        adapter = self._make_adapter(MemorySaver(), prior, captured)

        # pylon-style input: chat_history dicts + new HumanMessage
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        pylon_input = {
            "messages": [
                {"role": "user", "content": "turn 1 request"},
                {"role": "assistant", "content": "turn 1 final"},
                HumanMessage(content="follow-up request"),
            ]
        }
        adapter.invoke(pylon_input, config)

        forwarded = captured["input"]["messages"]
        assert len(forwarded) == 1, (
            f"Expected only the new HumanMessage to be forwarded, got {forwarded!r}"
        )
        assert isinstance(forwarded[0], HumanMessage)
        assert forwarded[0].content == "follow-up request"

    def test_fresh_thread_passes_input_unchanged(self):
        captured = {}
        adapter = self._make_adapter(MemorySaver(), [], captured)

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        adapter.invoke({"messages": [HumanMessage(content="first turn")]}, config)

        forwarded = captured["input"]["messages"]
        assert len(forwarded) == 1
        assert forwarded[0].content == "first turn"


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
    """Fake orchestrator LLM that simulates the PRODUCTION FAILURE MODE:

    Emits anaphoric handoff messages ("Now routing the items to the fixer.")
    instead of inlining the data. This exercises the helper's safety-net
    fallback (the ## Last Available Result section). The orchestrator's
    real prompt asks it to inline, but compliance varies — the SDK must
    deliver something actionable to the peer regardless.
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
            # Anaphoric — does NOT inline the data. The helper's safety net
            # must surface the prior sub-agent output for the peer to act on.
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

        # ---- Turn 2: "fix only the X-kind items" — fresh Assistant + fresh tools ----
        # Pylon sends chat_history-as-dicts + the new HumanMessage. The dicts
        # don't contain raw sub-agent outputs; SwarmResultAdapter must rely on
        # the persistent checkpoint and forward only the new HumanMessage.
        runnable_t2 = self._build_runnable(llm, memory, provider_app, fixer_app)
        h2 = HumanMessage(content="fix only the X-kind items")
        pylon_chat_history_dicts = [
            {"role": "user", "content": "please generate items"},
            {"role": "assistant", "content": "Done."},
        ]
        runnable_t2.invoke({"messages": pylon_chat_history_dicts + [h2]}, config=config)

        # The orchestrator must NOT re-call provider: it has visibility into
        # turn-1's checkpoint state and knows items are already generated.
        assert len(provider_app.calls) == 1, (
            f"Provider was re-called in turn 2 — orchestrator lost prior context. "
            f"calls={len(provider_app.calls)}"
        )

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

        # The orchestrator emitted anaphora ("Now routing the items..."), so
        # Your Task does NOT contain the items data. The safety-net section
        # ## Last Available Result must surface turn 1's provider payload so
        # the fixer can still act.
        assert '"kind": "X"' in fixer_task or '"id": 1' in fixer_task, (
            f"Fixer task lacks turn 1's provider data. The safety-net "
            f"section is missing or empty.\nTask:\n{fixer_task}"
        )
        # Brief shape: three sections, with the reference bounded to ONE payload
        assert "## Your Task" in fixer_task
        assert "## User's Most Recent Request" in fixer_task
        assert "fix only the X-kind items" in fixer_task
        assert "## Last Available Result" in fixer_task
        # Old verbose label must not be used
        assert "## Prior Agent Outputs" not in fixer_task
