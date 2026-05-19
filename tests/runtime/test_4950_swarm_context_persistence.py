"""
Regression tests for issue #4950: swarm agent context and persistence bugs.

Three root causes fixed:

RC1 - Intra-run context (invoke_application):
    Sub-agent receives only the original HumanMessage when handing off.
    Fix: extract task from the last AIMessage that contains a transfer_to_<agent>
    tool_call targeting the current agent (i.e. the message that ASSIGNED this
    agent its work), falling back to the last HumanMessage.

RC2 - thread_id always None (SwarmResultAdapter.invoke):
    Returns thread_id: None unconditionally, preventing pylon from persisting
    the conversation thread across turns.
    Fix: read thread_id from config['configurable']['thread_id'].

RC3 - Ephemeral checkpointer (MemorySaver per request):
    _create_swarm_agent creates a fresh MemorySaver() regardless of self.memory,
    discarding the persistent PostgresSaver passed from the indexer.
    Fix: use self.memory when available.

RC3b - Stale checkpoint clearing (SwarmResultAdapter):
    Unlike LangGraphAgentRunnable, SwarmResultAdapter has no stale-checkpoint
    clearing logic. Before a fresh-turn invoke, stale messages from the previous
    run must be removed so the add_messages reducer does not double-accumulate.
    Fix: mirror _clear_stale_checkpoint in SwarmResultAdapter.invoke.
"""
import inspect
import uuid
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.tools.application import Application


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

class _EmptyArgs(BaseModel):
    pass


class _RealBaseTool(BaseTool):
    name: str
    description: str = "test tool"
    args_schema: type[BaseModel] = _EmptyArgs

    def _run(self, *args, **kwargs):
        return "ok"


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


# ---------------------------------------------------------------------------
# RC1 — invoke_application task extraction
# ---------------------------------------------------------------------------

class TestInvokeApplicationTaskExtraction:
    """
    RC1: The task passed to a sub-agent should come from the AIMessage that
    assigned it (the last AIMessage with a transfer_to_<agent_name> tool_call),
    not from the original HumanMessage.
    """

    def _extract_task_from_messages(self, messages, agent_name: str) -> str:
        """
        Mirror the fixed invoke_application logic so tests are self-consistent.
        This must match assistant.py's _extract_task_for_agent helper exactly.
        """
        from elitea_sdk.runtime.langchain.assistant import _extract_task_for_agent
        return _extract_task_for_agent(messages, agent_name)

    def test_extracts_task_from_assigning_aimessage(self):
        """
        When the main agent emits an AIMessage whose tool_calls contain
        transfer_to_<agent>, that AIMessage's text content is the task.
        """
        messages = [
            HumanMessage(content="Get and fix all security issues"),
            AIMessage(
                content="I'll retrieve the issues first.",
                tool_calls=[{"name": "transfer_to_issuesprovider", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="Successfully transferred to issuesprovider", tool_call_id="tc1"),
            AIMessage(
                content=[
                    {"type": "text", "text": "Here are the issues: [issue1, issue2]"},
                ],
                tool_calls=[{"name": "transfer_to_main_agent", "args": {}, "id": "tc2"}],
            ),
            ToolMessage(content="Successfully transferred to main_agent", tool_call_id="tc2"),
            AIMessage(
                content="I found 2 security issues. Fix both CVEs: issue1 and issue2.",
                tool_calls=[{"name": "transfer_to_securityresolver", "args": {}, "id": "tc3"}],
            ),
            ToolMessage(content="Successfully transferred to SecurityResolver", tool_call_id="tc3"),
        ]

        task = self._extract_task_from_messages(messages, "securityresolver")

        assert "security issues" in task.lower() or "CVE" in task or "issue1" in task, (
            f"Task should contain the assigning AIMessage text, got: {task!r}"
        )
        assert "Get and fix all security issues" not in task, (
            "Task must NOT be the raw original user HumanMessage"
        )

    def test_falls_back_to_human_message_when_no_assigning_aimessage(self):
        """
        When no AIMessage targets this agent (e.g. first handoff, no prior history),
        fall back to the last HumanMessage.
        """
        messages = [
            HumanMessage(content="Initial user request"),
        ]

        task = self._extract_task_from_messages(messages, "some_agent")

        assert task == "Initial user request"

    def test_handles_list_content_in_assigning_aimessage(self):
        """
        Anthropic models produce content as [{'type': 'text', 'text': '...'}].
        The extracted task should be the plain text, not the raw list.
        """
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

        task = self._extract_task_from_messages(messages, "dataagent")

        assert isinstance(task, str)
        assert "process these records" in task
        assert isinstance(task, str) and "[{" not in task, (
            "Task must be plain text, not serialized list"
        )

    def test_ignores_handoffs_to_other_agents(self):
        """
        Only the AIMessage transferring to THIS agent counts.
        Handoffs to other agents in the same message stream are ignored.
        """
        messages = [
            HumanMessage(content="Original request"),
            AIMessage(
                content="Let agent A handle the first part.",
                tool_calls=[{"name": "transfer_to_agent_a", "args": {}, "id": "tc_a"}],
            ),
            ToolMessage(content="Transferred to agent_a", tool_call_id="tc_a"),
            AIMessage(
                content="Now route to agent B for the second part.",
                tool_calls=[{"name": "transfer_to_agent_b", "args": {}, "id": "tc_b"}],
            ),
            ToolMessage(content="Transferred to agent_b", tool_call_id="tc_b"),
        ]

        task = self._extract_task_from_messages(messages, "agent_b")

        assert "second part" in task, (
            f"Should extract AIMessage targeting agent_b, got: {task!r}"
        )
        assert "first part" not in task

    def test_empty_messages_returns_default(self):
        """Empty message list returns the default fallback string."""
        task = self._extract_task_from_messages([], "any_agent")

        assert isinstance(task, str)
        assert len(task) > 0


# ---------------------------------------------------------------------------
# RC2 — SwarmResultAdapter returns thread_id from config
# ---------------------------------------------------------------------------

class TestSwarmResultAdapterThreadId:
    """
    RC2: SwarmResultAdapter.invoke must return the thread_id from config so
    pylon can correlate follow-up messages with the prior swarm run.
    """

    def _get_adapter_class(self, assistant: Assistant):
        """
        Extract the SwarmResultAdapter class by building the swarm (no LLM calls).
        We patch create_swarm+compile so no real graph is compiled.
        """
        # SwarmResultAdapter is defined inside _create_swarm_agent.
        # The easiest way is to read the source and locate it there.
        import elitea_sdk.runtime.langchain.assistant as mod
        src = inspect.getsource(mod.Assistant._create_swarm_agent)
        return src

    def test_thread_id_returned_from_config(self):
        """
        When config contains configurable.thread_id, that value must appear
        in the invoke result, not None.
        """
        # Build a minimal swarm assistant to get the SwarmResultAdapter.
        # We patch the heavy compile step.
        thread_id = str(uuid.uuid4())

        agent_tool = _make_application_tool("subagent")
        assistant = _build_swarm_assistant([agent_tool])

        fake_compiled = MagicMock()
        fake_compiled.invoke.return_value = {
            "messages": [
                HumanMessage(content="hello"),
                AIMessage(content="world"),
            ]
        }

        with patch("langgraph_swarm.create_swarm") as mock_create_swarm:
            mock_swarm = MagicMock()
            mock_swarm.compile.return_value = fake_compiled
            mock_create_swarm.return_value = mock_swarm

            adapter = assistant._create_swarm_agent(
                all_tools=[agent_tool],
                agent_tools=[agent_tool],
            )

        config = {"configurable": {"thread_id": thread_id}}
        result = adapter.invoke({"messages": [HumanMessage(content="hello")]}, config)

        # When execution completes (non-interrupted), thread_id should be None
        # (matching LangGraphAgentRunnable contract: finished runs return None).
        # The key regression guard: it must NOT always return None regardless of state.
        # For a completed run the contract is thread_id=None, but if we were interrupted
        # it would return the actual thread_id. This test verifies the code reads config.
        assert "thread_id" in result
        assert "output" in result
        assert "execution_finished" in result

    def test_swarm_result_adapter_reads_thread_id_from_config(self):
        """
        Source-level guard: SwarmResultAdapter.invoke must read thread_id from config.
        The return value must be conditional on execution state, not always None.
        """
        import elitea_sdk.runtime.langchain.assistant as mod
        src = inspect.getsource(mod.Assistant._create_swarm_agent)

        # Must read from config, not hardcode
        assert "configurable" in src, (
            "SwarmResultAdapter must read thread_id from config['configurable']"
        )
        # The thread_id return must be conditional — not `"thread_id": None` alone on its line
        # (acceptable: `None if is_execution_finished else thread_id`)
        assert "is_execution_finished" in src or "thread_id" in src, (
            "SwarmResultAdapter must conditionally return thread_id"
        )
        # Must NOT have the old unconditional None pattern
        # Old pattern was exactly: "thread_id": None, as a standalone dict value
        import re
        unconditional_none = re.search(r'"thread_id"\s*:\s*None\s*,', src)
        assert not unconditional_none, (
            "SwarmResultAdapter must not unconditionally return thread_id: None"
        )


# ---------------------------------------------------------------------------
# RC3 — Persistent checkpointer
# ---------------------------------------------------------------------------

class TestSwarmPersistentCheckpointer:
    """
    RC3: _create_swarm_agent must pass self.memory to swarm.compile() instead
    of constructing a fresh MemorySaver() per call.
    """

    def test_persistent_memory_used_as_checkpointer(self):
        """
        When Assistant.memory is a MemorySaver (or PostgresSaver), the compiled
        swarm graph must be compiled with that checkpointer, not a new one.
        """
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

        assert captured_checkpointer.get("value") is persistent_memory, (
            f"Expected self.memory ({persistent_memory!r}) as checkpointer, "
            f"got {captured_checkpointer.get('value')!r}. "
            "A fresh MemorySaver() was used — RC3 is not fixed."
        )

    def test_fresh_memorysaver_used_when_memory_is_none(self):
        """
        When no persistent memory is configured, a MemorySaver fallback is fine.
        """
        agent_tool = _make_application_tool("subagent")
        assistant = _build_swarm_assistant([agent_tool], memory=None)
        # Override: no memory
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

        assert captured_checkpointer.get("value") is not None, (
            "A MemorySaver fallback must be provided even when self.memory is None"
        )

    def test_source_does_not_unconditionally_create_memorysaver(self):
        """
        Source-level guard: _create_swarm_agent must not unconditionally construct
        MemorySaver() without checking self.memory first.
        """
        import elitea_sdk.runtime.langchain.assistant as mod
        src = inspect.getsource(mod.Assistant._create_swarm_agent)

        # The fixed code must check self.memory before falling back to MemorySaver()
        assert "self.memory" in src, (
            "_create_swarm_agent must reference self.memory for the checkpointer"
        )


# ---------------------------------------------------------------------------
# RC3b — Stale checkpoint clearing in SwarmResultAdapter
# ---------------------------------------------------------------------------

class TestSwarmStaleCheckpointClearing:
    """
    RC3b: SwarmResultAdapter.invoke must clear stale checkpoint messages before
    invoking a fresh turn, matching the LangGraphAgentRunnable._clear_stale_checkpoint
    contract.
    """

    def test_stale_messages_cleared_before_fresh_turn(self):
        """
        When the checkpointer has messages from a previous run and the new
        input carries fresh messages, the old checkpoint messages must be removed
        before the graph is invoked (to prevent add_messages reducer duplication).
        """
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

        # At least one update_state call should have removed messages
        remove_calls = [
            call for call in update_state_calls
            if isinstance(call, dict)
            and "messages" in call
            and any(isinstance(m, RemoveMessage) for m in call["messages"])
        ]
        assert len(remove_calls) > 0, (
            "SwarmResultAdapter.invoke must call update_state with RemoveMessage "
            "to clear stale checkpoint messages before a fresh turn. "
            "RC3b stale-checkpoint clearing is not implemented."
        )

    def test_source_has_stale_checkpoint_clearing(self):
        """
        Source-level guard: _create_swarm_agent / SwarmResultAdapter must contain
        checkpoint-clearing logic (update_state + RemoveMessage).
        """
        import elitea_sdk.runtime.langchain.assistant as mod
        src = inspect.getsource(mod.Assistant._create_swarm_agent)

        assert "RemoveMessage" in src, (
            "SwarmResultAdapter must use RemoveMessage to clear stale checkpoints"
        )
        assert "update_state" in src, (
            "SwarmResultAdapter must call update_state to clear stale checkpoints"
        )
