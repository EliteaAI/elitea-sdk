"""
Tests for #5046: standalone Application child config handling.

Changes verified:
  1. application.py _run: __pregel_task_id is STRIPPED in standalone mode so the
     child runs as a root pregel. The child's LangGraphAgentRunnable.invoke()
     handles interrupts internally and returns hitl_interrupt in its response
     dict (dict-bridge path).
  2. application.py _run: is_subgraph is forced to False when calling
     client.application() so we always get a LangGraphAgentRunnable (not a raw
     CompiledStateGraph compiled with checkpointer=True, which cannot be invoked
     as a root graph).
  3. application.py _run: child HITL interrupts returned in response dict
     are bubbled to the parent via interrupt(), and resume values are routed
     back to the child.
"""
from unittest.mock import MagicMock, patch

from langgraph.errors import GraphInterrupt

from elitea_sdk.runtime.tools.application import Application


# ---------------------------------------------------------------------------
# Primary fix: __pregel_task_id is STRIPPED for standalone children
# ---------------------------------------------------------------------------

class TestPregelTaskIdStrip:
    """__pregel_task_id must be STRIPPED in standalone mode (child runs as root).

    Standalone mode: child is rebuilt as a root LangGraphAgentRunnable. Stripping
    __pregel_task_id ensures is_nested=False → the child's pregel handles
    interrupts internally (stores them, terminates loop, returns result dict)
    rather than re-raising GraphInterrupt.

    Structural subgraph mode (client=None): child was compiled with
    checkpointer=True and runs embedded in the parent graph's execution tree.
    The full config (including __pregel_task_id) must be forwarded so LangGraph
    can wire the shared checkpointer.
    """

    def _make_standalone_tool(self, name="ChildPipeline"):
        """Application tool in standalone mode: has client + args_runnable."""
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"output": "result"}
        mock_client = MagicMock()
        mock_client.application.return_value = mock_app
        tool = Application(
            name=name,
            description="Child",
            application=mock_app,
            return_type="str",
            client=mock_client,
            is_subgraph=True,
            args_runnable={
                "application_id": 1,
                "application_version_id": 1,
                "is_subgraph": True,
            },
        )
        return tool, mock_app

    def _make_subgraph_tool(self, name="ChildPipeline"):
        """Application tool in structural-subgraph mode: client=None, pre-built app."""
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"output": "result"}
        tool = Application(
            name=name,
            description="Child",
            application=mock_app,
            return_type="str",
            client=None,
            is_subgraph=True,
            args_runnable={},
        )
        return tool, mock_app

    def test_standalone_pregel_task_id_stripped(self):
        """In standalone mode, __pregel_task_id must NOT reach the child so the
        child runs as root (handles interrupts internally, returns result dict)."""
        tool, mock_app = self._make_standalone_tool()

        parent_config = {
            "configurable": {
                "thread_id": "swarm-thread-abc",
                "__pregel_task_id": "task-xyz-123",
                "__pregel_task_ids": ["task-xyz-123"],
                "some_user_key": "keep_me",
            },
            "metadata": {},
        }

        captured = {}

        def record_invoke(payload, config=None):
            captured["config"] = config
            return {"output": "result"}

        mock_app.invoke.side_effect = record_invoke

        tool._run(task="do the work", config=parent_config)

        child_cfg = captured.get("config", {}).get("configurable", {})
        assert "__pregel_task_id" not in child_cfg, (
            f"__pregel_task_id must be stripped for standalone child; got: {child_cfg}"
        )

    def test_subgraph_pregel_task_id_preserved(self):
        """In structural-subgraph mode (client=None), __pregel_task_id must be
        forwarded so LangGraph can wire the shared checkpointer correctly."""
        tool, mock_app = self._make_subgraph_tool()

        parent_config = {
            "configurable": {
                "thread_id": "parent-thread",
                "__pregel_task_id": "task-abc",
                "__pregel_task_ids": ["task-abc"],
            },
            "metadata": {},
        }

        captured = {}

        def record_invoke(payload, config=None):
            captured["config"] = config
            return {"output": "result"}

        mock_app.invoke.side_effect = record_invoke

        tool._run(task="subgraph task", config=parent_config)

        child_cfg = captured.get("config", {}).get("configurable", {})
        assert "__pregel_task_id" in child_cfg, (
            f"__pregel_task_id must be forwarded for structural subgraph; got: {child_cfg}"
        )
        assert child_cfg["__pregel_task_id"] == "task-abc"

    def test_standalone_pregel_task_id_absent_no_keyerror(self):
        """When parent config has no __pregel_task_id, standalone _run must not raise."""
        tool, mock_app = self._make_standalone_tool()
        plain_config = {
            "configurable": {"thread_id": "thread-1"},
            "metadata": {},
        }
        result = tool._run(task="plain task", config=plain_config)
        assert result is not None

    def test_standalone_other_configurable_keys_preserved(self):
        """Non-pregel keys in parent configurable must pass through to the child."""
        tool, mock_app = self._make_standalone_tool("Child")

        parent_config = {
            "configurable": {
                "thread_id": "t1",
                "__pregel_task_id": "should-be-stripped",
                "my_custom_key": "keep-this",
            },
            "metadata": {},
        }

        captured = {}

        def record_invoke(payload, config=None):
            captured["config"] = config
            return {"output": "result"}

        mock_app.invoke.side_effect = record_invoke

        tool._run(task="work", config=parent_config)

        child_cfg = captured.get("config", {}).get("configurable", {})
        assert "my_custom_key" in child_cfg
        assert child_cfg["my_custom_key"] == "keep-this"
        assert "__pregel_task_id" not in child_cfg


# ---------------------------------------------------------------------------
# Primary fix: is_subgraph forced to False when calling client.application()
# ---------------------------------------------------------------------------

class TestIsSubgraphForcedFalse:

    def test_client_application_called_with_is_subgraph_false(self):
        """Even when the Application tool was registered with is_subgraph=True,
        client.application() must be called with is_subgraph=False so we get a
        LangGraphAgentRunnable (not a raw CompiledStateGraph with checkpointer=True).
        """
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"output": "ok"}
        mock_client = MagicMock()
        mock_client.application.return_value = mock_app

        app_tool = Application(
            name="PipelineTool",
            description="Pipeline",
            application=mock_app,
            return_type="str",
            client=mock_client,
            is_subgraph=True,
            args_runnable={
                "application_id": 99,
                "application_version_id": 7,
                "is_subgraph": True,
            },
        )

        config = {
            "configurable": {"thread_id": "t1"},
            "metadata": {},
        }

        app_tool._run(task="run", config=config)

        call_kwargs = mock_client.application.call_args
        assert call_kwargs is not None, "client.application() was not called"
        passed_is_subgraph = call_kwargs.kwargs.get("is_subgraph")
        assert passed_is_subgraph is False, (
            f"client.application() must be called with is_subgraph=False, "
            f"got is_subgraph={passed_is_subgraph!r}"
        )

    def test_args_runnable_is_not_mutated(self):
        """The original args_runnable dict on the tool must not be mutated
        by _run (we use a copy). Verified by checking value before and after.
        """
        mock_app = MagicMock()
        mock_app.invoke.return_value = {"output": "ok"}
        mock_client = MagicMock()
        mock_client.application.return_value = mock_app

        app_tool = Application(
            name="PipelineTool",
            description="Pipeline",
            application=mock_app,
            return_type="str",
            client=mock_client,
            is_subgraph=True,
            args_runnable={
                "application_id": 5,
                "application_version_id": 2,
                "is_subgraph": True,
            },
        )

        config = {
            "configurable": {"thread_id": "t1"},
            "metadata": {},
        }

        app_tool._run(task="run", config=config)

        assert app_tool.args_runnable["is_subgraph"] is True, (
            "_run must not mutate self.args_runnable; original is_subgraph must remain True"
        )


# ---------------------------------------------------------------------------
# HITL bubble-up via dict-bridge: child returns hitl_interrupt in response
# ---------------------------------------------------------------------------

class TestHitlBubbleUp:
    """When a child agent/pipeline returns hitl_interrupt in its response dict,
    Application._run must call interrupt() to bubble it to the parent graph.
    On resume, the user's decision is routed back to the child."""

    def _make_tool(self):
        mock_app = MagicMock()
        mock_client = MagicMock()
        mock_client.application.return_value = mock_app
        tool = Application(
            name="ChildAgent",
            description="Child",
            application=mock_app,
            return_type="str",
            client=mock_client,
            is_subgraph=True,
            args_runnable={
                "application_id": 1,
                "application_version_id": 1,
                "is_subgraph": True,
            },
        )
        return tool, mock_app, mock_client

    def test_child_interrupt_raises_graph_interrupt(self):
        """When child returns hitl_interrupt, interrupt() is called which
        raises GraphInterrupt in the parent graph context."""
        tool, mock_app, _ = self._make_tool()

        hitl_payload = {
            "type": "hitl",
            "tool_name": "jira_create_issue",
            "message": "Approve creating JIRA ticket?",
            "guardrail_type": "sensitive_tool",
        }
        mock_app.invoke.return_value = {
            "output": "Awaiting human review...",
            "execution_finished": False,
            "hitl_interrupt": hitl_payload,
        }

        config = {"configurable": {"thread_id": "t1"}, "metadata": {}}

        with patch("elitea_sdk.runtime.tools.application.interrupt") as mock_interrupt:
            mock_interrupt.side_effect = GraphInterrupt([{"value": hitl_payload}])
            try:
                tool._run(task="do work", config=config)
                assert False, "Should have raised GraphInterrupt"
            except GraphInterrupt:
                pass
            called_payload = mock_interrupt.call_args[0][0]
            assert called_payload["type"] == "hitl"
            assert called_payload["tool_name"] == "jira_create_issue"
            assert called_payload["_parent_tool_name"] == "ChildAgent"
            assert called_payload["_parent_tool_args"] == {"task": "do work"}

    def test_resume_routes_to_child(self):
        """On resume, interrupt() returns the user's decision which is
        forwarded to the child as an HITL resume invocation."""
        tool, mock_app, _ = self._make_tool()

        hitl_payload = {
            "type": "hitl",
            "tool_name": "jira_create_issue",
            "message": "Approve?",
            "guardrail_type": "sensitive_tool",
        }
        mock_app.invoke.side_effect = [
            {
                "output": "Awaiting human review...",
                "execution_finished": False,
                "hitl_interrupt": hitl_payload,
            },
            {"output": "Ticket created", "execution_finished": True},
        ]

        config = {"configurable": {"thread_id": "t1"}, "metadata": {}}
        resume_value = {"action": "approve", "value": ""}

        with patch("elitea_sdk.runtime.tools.application.interrupt") as mock_interrupt:
            mock_interrupt.return_value = resume_value
            result = tool._run(task="do work", config=config)

        called_payload = mock_interrupt.call_args[0][0]
        assert called_payload["tool_name"] == "jira_create_issue"
        assert called_payload["_parent_tool_name"] == "ChildAgent"
        assert called_payload["_parent_tool_args"] == {"task": "do work"}

        assert mock_app.invoke.call_count == 2
        resume_call_args = mock_app.invoke.call_args_list[1]
        resume_input = resume_call_args[0][0]
        assert resume_input["hitl_resume"] is True
        assert resume_input["hitl_action"] == "approve"
        assert resume_input["hitl_value"] == ""

        assert result["output"] == "Ticket created"
        assert result["execution_finished"] is True

    def test_resume_with_reject_action(self):
        """Reject action is correctly routed to child."""
        tool, mock_app, _ = self._make_tool()

        hitl_payload = {"type": "hitl", "tool_name": "dangerous_tool", "message": "Allow?"}
        mock_app.invoke.side_effect = [
            {"output": "Awaiting...", "execution_finished": False, "hitl_interrupt": hitl_payload},
            {"output": "Operation cancelled", "execution_finished": True},
        ]

        config = {"configurable": {"thread_id": "t1"}, "metadata": {}}

        with patch("elitea_sdk.runtime.tools.application.interrupt") as mock_interrupt:
            mock_interrupt.return_value = {"action": "reject", "value": "not allowed"}
            result = tool._run(task="do work", config=config)

        resume_input = mock_app.invoke.call_args_list[1][0][0]
        assert resume_input["hitl_action"] == "reject"
        assert resume_input["hitl_value"] == "not allowed"
        assert result["output"] == "Operation cancelled"

    def test_no_interrupt_passes_through(self):
        """Normal responses without hitl_interrupt pass through unchanged."""
        tool, mock_app, _ = self._make_tool()

        mock_app.invoke.return_value = {"output": "All done", "execution_finished": True}
        config = {"configurable": {"thread_id": "t1"}, "metadata": {}}

        with patch("elitea_sdk.runtime.tools.application.interrupt") as mock_interrupt:
            result = tool._run(task="do work", config=config)
            mock_interrupt.assert_not_called()

        assert result["output"] == "All done"
        assert result["execution_finished"] is True

    def test_empty_hitl_interrupt_not_triggered(self):
        """Empty/falsy hitl_interrupt does not trigger bubble-up."""
        tool, mock_app, _ = self._make_tool()

        mock_app.invoke.return_value = {
            "output": "Done",
            "execution_finished": True,
            "hitl_interrupt": None,
        }
        config = {"configurable": {"thread_id": "t1"}, "metadata": {}}

        with patch("elitea_sdk.runtime.tools.application.interrupt") as mock_interrupt:
            result = tool._run(task="work", config=config)
            mock_interrupt.assert_not_called()

        assert result["output"] == "Done"

    def test_non_dict_resume_value_defaults_to_approve(self):
        """If interrupt() returns a non-dict (langgraph drift), _run coerces
        to empty dict so child still gets hitl_action='approve'."""
        tool, mock_app, _ = self._make_tool()

        hitl_payload = {"type": "hitl", "tool_name": "x", "guardrail_type": "sensitive_tool"}
        mock_app.invoke.side_effect = [
            {"output": "Awaiting...", "execution_finished": False, "hitl_interrupt": hitl_payload},
            {"output": "done", "execution_finished": True},
        ]

        config = {"configurable": {"thread_id": "t1"}, "metadata": {}}

        with patch("elitea_sdk.runtime.tools.application.interrupt") as mock_interrupt:
            mock_interrupt.return_value = "unexpected-string-not-dict"
            tool._run(task="do work", config=config)

        resume_input = mock_app.invoke.call_args_list[1][0][0]
        assert resume_input["hitl_resume"] is True
        assert resume_input["hitl_action"] == "approve"
        assert resume_input["hitl_value"] == ""
