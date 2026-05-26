"""
Tests for the fix in #5046: __pregel_task_id leaks from SWARM context into
child pipeline config, causing LangGraph to treat the pipeline as nested
(is_nested=True) and not suppress GraphInterrupt.

Two changes are verified:
  1. application.py _run: __pregel_task_id / __pregel_task_ids are stripped
     from parent_configurable before building nested_config.
  2. application.py _run: is_subgraph is forced to False when calling
     client.application() so we always get a LangGraphAgentRunnable (not a raw
     CompiledStateGraph compiled with checkpointer=True, which cannot be invoked
     as a root graph).
"""
from unittest.mock import MagicMock

from elitea_sdk.runtime.tools.application import Application


# ---------------------------------------------------------------------------
# Primary fix: __pregel_task_id is NOT forwarded to child configurable
# ---------------------------------------------------------------------------

class TestPregelTaskIdStrip:
    """__pregel_task_id must be stripped in standalone-tool mode (client+args_runnable set)
    and preserved in structural-subgraph mode (client=None, pre-built application).

    Standalone mode: child is rebuilt as a root LangGraphAgentRunnable. Forwarding
    __pregel_task_id sets is_nested=True → GraphInterrupt not suppressed (#5046).

    Structural subgraph mode: child was compiled with checkpointer=True and runs
    embedded in the parent graph's execution tree. The full config (including
    __pregel_task_id) must be forwarded so LangGraph can wire the shared checkpointer.
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
            client=None,   # No client — pre-built application
            is_subgraph=True,
            args_runnable={},
        )
        return tool, mock_app

    def test_standalone_pregel_task_id_stripped(self):
        """In standalone mode, __pregel_task_id must NOT reach the child's config."""
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
            f"__pregel_task_id must NOT be forwarded to standalone child; got: {child_cfg}"
        )
        assert "__pregel_task_ids" not in child_cfg, (
            f"__pregel_task_ids must NOT be forwarded to standalone child; got: {child_cfg}"
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
                "__pregel_task_id": "should-go",
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
                "is_subgraph": True,       # registered as True
            },
        )

        config = {
            "configurable": {"thread_id": "t1"},
            "metadata": {},
        }

        app_tool._run(task="run", config=config)

        # The call to client.application() must have is_subgraph=False
        call_kwargs = mock_client.application.call_args
        assert call_kwargs is not None, "client.application() was not called"
        # is_subgraph can be positional or keyword — check both
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

        # Original must still have True
        assert app_tool.args_runnable["is_subgraph"] is True, (
            "_run must not mutate self.args_runnable; original is_subgraph must remain True"
        )


