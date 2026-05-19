import uuid
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.assistant import Assistant, _extract_task_for_agent
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


class TestInvokeApplicationTaskExtraction:

    def test_aggregates_session_transcript_through_assigning_aimessage(self):
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

        assert "User: Get and fix all security issues" in task
        assert "Here are the issues: [issue1, issue2]" in task
        assert "Fix both CVEs: issue1 and issue2" in task
        assert "Successfully transferred" not in task

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

        assert "Fix all open issues" in task
        assert "Issues: #11, #14, #20, #21" in task
        assert "Security fixes for #11 and #14 applied: patch X and Y." in task
        assert "hand off all 4 issues" in task
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
        assert "User: user request" in task_camel

        messages_ws = [
            HumanMessage(content="ws request"),
            AIMessage(
                content="Hand off to data team.",
                tool_calls=[{"name": "transfer_to_data_agent", "args": {}, "id": "tc1"}],
            ),
        ]
        task_ws = _extract_task_for_agent(messages_ws, "Data Agent")
        assert "Hand off to data team" in task_ws
        assert "User: ws request" in task_ws


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
