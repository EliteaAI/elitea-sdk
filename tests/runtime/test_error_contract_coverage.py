"""#6170: the error contract must be reachable from every tool construction path.

Three separate holes are covered here:
  * MiddlewareManager.wrap_tool applied every middleware EXCEPT the exception handler,
    so the one call site that used it (the pipeline Code node) got no error handling.
  * The wrapper rebuilt the tool as a StructuredTool from its sync callable, which
    dropped the async path, the metadata-forwarding patched invoke and response_format.
  * Nothing failed when a node type constructed a tool and forgot to wrap it.
"""

import asyncio
import re
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, Field

from elitea_sdk.runtime.clients import client as client_module
from elitea_sdk.runtime.exceptions import BudgetExceededError
from elitea_sdk.runtime.langchain import langraph_agent
from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.middleware.base import MiddlewareManager
from elitea_sdk.runtime.middleware.strategies import LoggingStrategy
from elitea_sdk.runtime.middleware.tool_exception_handler import (
    ToolExceptionHandlerMiddleware,
)
from elitea_sdk.runtime.tool_outcome import ToolErrorClass
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired
from elitea_sdk.tools import _patch_tool_invoke

# getattr, not attribute access: the file must still collect against a build without
# the marker so the pre-fix runs report failures instead of a collection error.
MARKER = getattr(ToolExceptionHandlerMiddleware, 'WRAPPED_MARKER', '_elitea_error_wrapped')


class _Args(BaseModel):
    value: str = Field(default="x", description="value")


def _tehm(**kwargs) -> ToolExceptionHandlerMiddleware:
    """LoggingStrategy only: it never rewrites the message, so assertions can match the
    raw error text instead of an LLM-generated rewrite."""
    return ToolExceptionHandlerMiddleware(strategies=[LoggingStrategy()], **kwargs)


def _sync_failing_tool(name="sync_tool", error=None, metadata=None) -> StructuredTool:
    def call(value: str = "x") -> str:
        raise error or RuntimeError("sync boom")

    return StructuredTool.from_function(
        func=call, name=name, description="d", args_schema=_Args, metadata=metadata,
    )


def _async_failing_tool(name="async_tool", **tool_kwargs) -> StructuredTool:
    def call(value: str = "x") -> str:
        return "sync path"

    async def acall(value: str = "x") -> str:
        raise RuntimeError("async boom")

    return StructuredTool(
        name=name, description="d", args_schema=_Args, func=call, coroutine=acall,
        **tool_kwargs,
    )


def _coroutine_only_tool(name="coroutine_only") -> StructuredTool:
    """The shape langchain_mcp_adapters produces: coroutine set, func absent."""

    async def acall(value: str = "x") -> str:
        raise RuntimeError("async boom")

    return StructuredTool(name=name, description="d", args_schema=_Args, coroutine=acall)


class _AsyncOnlyTool(BaseTool):
    """A BaseTool subclass with a real _arun; _run is the mandatory abstract stub every
    async-only tool has to declare, and it raises rather than executing anything."""

    name: str = "async_only"
    description: str = "d"
    args_schema: type = _Args

    def _run(self, value: str = "x") -> str:
        raise NotImplementedError("sync not supported")

    async def _arun(self, value: str = "x") -> str:
        raise RuntimeError("arun boom")


class TestManagerAppliesExceptionHandling:
    """The acceptance criterion: wrap_tool must mean what its name says."""

    def test_manager_wrap_tool_applies_exception_handling(self):
        manager = MiddlewareManager()
        manager.add_wrap_only(_tehm())

        wrapped = manager.wrap_tool(_sync_failing_tool())

        assert wrapped.invoke({"value": "x"}) == "sync boom"

    def test_manager_wrap_tool_still_applies_it_when_the_guard_is_skipped(self):
        """The Code node passes skip_sensitive_guard=True; that must not skip errors too."""
        manager = MiddlewareManager()
        manager.add_wrap_only(_tehm())

        wrapped = manager.wrap_tool(_sync_failing_tool(), skip_sensitive_guard=True)

        assert wrapped.invoke({"value": "x"}) == "sync boom"

    def test_exception_handler_is_outermost(self):
        """Order today: every other middleware wraps first, the handler wraps last."""
        order = []

        class _Recording:
            def wrap_tool(self, tool):
                order.append('other')
                return tool

        manager = MiddlewareManager()
        manager.add_wrap_only(_tehm())
        manager.add(_Recording())

        manager.wrap_tool(_sync_failing_tool())

        assert order == ['other']

    def test_registration_does_not_change_the_middleware_prompt(self):
        """assistant.py lets a non-empty middleware prompt displace PLAN_ADDON, so the
        handler's own prompt section must stay out of the aggregated prompt."""
        manager = MiddlewareManager()
        manager.add_wrap_only(_tehm())

        assert manager.get_combined_prompt() == ""
        assert manager.get_all_tools() == []


class TestAssistantWiring:

    @staticmethod
    def _assistant(tools):
        return Assistant(
            elitea=MagicMock(get_mcp_toolkits=lambda: []),
            data={"instructions": "i", "tools": [], "meta": {}},
            client=MagicMock(),
            tools=tools,
            memory=MemorySaver(),
            app_type="agent",
            middleware=[_tehm()],
        )

    def test_tools_are_wrapped_through_the_manager(self):
        assistant = self._assistant([_sync_failing_tool()])

        assert getattr(assistant.tools[0], MARKER, False) is True

    def test_manager_can_wrap_tools_built_after_construction(self):
        """What the Code node and the lazy meta-tools rely on."""
        assistant = self._assistant([_sync_failing_tool()])

        late = assistant.middleware_manager.wrap_tool(_sync_failing_tool("late"))

        assert late.invoke({"value": "x"}) == "sync boom"

    def test_middleware_prompt_stays_empty(self):
        assistant = self._assistant([_sync_failing_tool()])

        assert assistant._middleware_prompt == ""


class TestAsyncCoverage:

    def test_coroutine_is_preserved_and_its_errors_are_handled(self):
        wrapped = _tehm().wrap_tool(_async_failing_tool())

        assert wrapped.coroutine is not None
        assert asyncio.run(wrapped.ainvoke({"value": "x"})) == "async boom"

    def test_async_path_is_not_downgraded_to_the_sync_one(self):
        """The rebuilt-StructuredTool wrapper lost `coroutine`, so ainvoke silently ran
        the sync implementation in an executor and returned its result instead."""
        wrapped = _tehm().wrap_tool(_async_failing_tool())

        assert asyncio.run(wrapped.ainvoke({"value": "x"})) != "sync path"

    def test_basetool_arun_errors_are_handled(self):
        wrapped = _tehm().wrap_tool(_AsyncOnlyTool())

        assert asyncio.run(wrapped.ainvoke({"value": "x"})) == "arun boom"

    def test_sync_only_tool_is_still_covered_on_the_async_entry_point(self):
        """No coroutine to patch: the inherited _arun runs the patched sync func."""
        wrapped = _tehm().wrap_tool(_sync_failing_tool())

        assert wrapped.coroutine is None
        assert asyncio.run(wrapped.ainvoke({"value": "x"})) == "sync boom"

    def test_coroutine_only_tool_keeps_working(self):
        """Wrapping used to fabricate a sync func from BaseTool._run's stub, so ainvoke
        returned 'StructuredTool does not support sync invocation.' as a *success*."""
        wrapped = _tehm().wrap_tool(_coroutine_only_tool())

        assert wrapped.func is None
        assert asyncio.run(wrapped.ainvoke({"value": "x"})) == "async boom"

    def test_a_tool_with_no_wrappable_callable_is_returned_unwrapped(self, caplog):
        # Neither func nor coroutine: there is nothing to patch, and staying silent about
        # it is the "silent bail" #6170 asks to remove.
        tool = StructuredTool.model_construct(
            name="opaque", description="d", args_schema=_Args, func=None, coroutine=None)
        with caplog.at_level("ERROR"):
            result = _tehm().wrap_tool(tool)

        assert result is tool
        assert "bypass the tool error contract" in caplog.text


class TestPreservedContracts:
    """Everything #6170 lists under "must not break"."""

    def test_patched_invoke_survives_wrapping(self):
        tool = _sync_failing_tool(metadata={"toolkit_name": "tk", "toolkit_type": "github"})
        _patch_tool_invoke(tool)

        wrapped = _tehm().wrap_tool(tool)

        assert getattr(wrapped, '_invoke_metadata_patched', False) is True
        assert 'invoke' in wrapped.__dict__
        # Re-bound to the copy, or invoke() would route back around the error handling
        assert wrapped.__dict__['invoke'].__self__ is wrapped

    def test_patched_invoke_still_reaches_the_error_handling(self):
        tool = _sync_failing_tool(metadata={"toolkit_name": "tk", "toolkit_type": "github"})
        _patch_tool_invoke(tool)

        wrapped = _tehm().wrap_tool(tool)

        assert wrapped.invoke({"value": "x"}) == "sync boom"

    def test_metadata_is_preserved(self):
        metadata = {"toolkit_name": "tk", "toolkit_type": "github"}

        wrapped = _tehm().wrap_tool(_sync_failing_tool(metadata=metadata))

        assert wrapped.metadata == metadata

    def test_response_format_is_preserved_and_errors_are_two_tuples(self):
        """content_and_artifact is what langchain_mcp_adapters sets; BaseTool.run raises
        over a bare string, so the handled message has to come back as a two-tuple."""
        tool = _async_failing_tool(response_format="content_and_artifact")

        wrapped = _tehm().wrap_tool(tool)

        assert wrapped.response_format == "content_and_artifact"
        assert asyncio.run(wrapped.ainvoke({"value": "x"})) == "async boom"

    def test_tool_class_is_preserved(self):
        wrapped = _tehm().wrap_tool(_AsyncOnlyTool())

        assert type(wrapped) is _AsyncOnlyTool

    def test_original_tool_is_reachable(self):
        tool = _sync_failing_tool()

        wrapped = _tehm().wrap_tool(tool)

        assert wrapped._original_tool is tool

    def test_same_tool_is_not_wrapped_twice(self):
        middleware, tool = _tehm(), _sync_failing_tool()

        assert middleware.wrap_tool(tool) is middleware.wrap_tool(tool)

    def test_an_already_wrapped_tool_is_not_wrapped_again(self):
        """A second handler instance (a nested application builds its own) must not stack."""
        wrapped = _tehm().wrap_tool(_sync_failing_tool())

        assert _tehm().wrap_tool(wrapped) is wrapped

    def test_same_named_tools_from_different_toolkits_stay_distinct(self):
        middleware = _tehm()
        first = _sync_failing_tool("index_data", metadata={"toolkit_name": "github"})
        second = _sync_failing_tool("index_data", metadata={"toolkit_name": "confluence"})

        assert middleware.wrap_tool(first) is not middleware.wrap_tool(second)

    def test_excluded_tools_are_not_wrapped(self):
        tool = _sync_failing_tool()

        assert _tehm(excluded_tools=["sync_tool"]).wrap_tool(tool) is tool

    @pytest.mark.parametrize("error", [
        McpAuthorizationRequired("auth", server_url="https://s", tool_name="t"),
        GraphInterrupt(()),
        BudgetExceededError("budget exhausted"),
    ])
    def test_pass_throughs_traverse_the_wrapper_unaltered(self, error):
        wrapped = _tehm().wrap_tool(_sync_failing_tool(error=error))

        with pytest.raises(type(error)):
            wrapped.invoke({"value": "x"})

    @pytest.mark.parametrize("error", [
        McpAuthorizationRequired("auth", server_url="https://s", tool_name="t"),
        GraphInterrupt(()),
        BudgetExceededError("budget exhausted"),
    ])
    def test_pass_throughs_traverse_the_async_wrapper_too(self, error):
        async def acall(value: str = "x") -> str:
            raise error

        tool = StructuredTool(name="t", description="d", args_schema=_Args, coroutine=acall)
        wrapped = _tehm().wrap_tool(tool)

        with pytest.raises(type(error)):
            asyncio.run(wrapped.ainvoke({"value": "x"}))

    def test_not_implemented_error_is_not_turned_into_prose(self):
        """The agent loop branches on it to fall back from ainvoke to invoke."""
        wrapped = _tehm().wrap_tool(_sync_failing_tool(error=NotImplementedError("no sync")))

        with pytest.raises(NotImplementedError):
            wrapped.invoke({"value": "x"})

    def test_handle_tool_error_is_preserved(self):
        tool = _sync_failing_tool()
        tool.handle_tool_error = "handled"

        assert _tehm().wrap_tool(tool).handle_tool_error == "handled"

    def test_validation_errors_are_routed_through_the_strategies(self):
        wrapped = _tehm().wrap_tool(_sync_failing_tool())

        # value must be a str; the ValidationError fires before the callable is reached
        assert "value" in str(wrapped.invoke({"value": {"not": "a string"}}))

    def test_strategy_raised_tool_exception_is_not_swallowed(self):
        class _Raising(LoggingStrategy):
            def handle_exception(self, context):
                raise ToolException("circuit open")

        middleware = ToolExceptionHandlerMiddleware(strategies=[_Raising()])
        wrapped = middleware.wrap_tool(_sync_failing_tool())

        with pytest.raises(ToolException):
            wrapped.invoke({"value": "x"})

    def test_success_still_returns_the_tool_result(self):
        def call(value: str = "x") -> str:
            return f"ok:{value}"

        tool = StructuredTool.from_function(
            func=call, name="fine", description="d", args_schema=_Args)

        assert _tehm().wrap_tool(tool).invoke({"value": "y"}) == "ok:y"


class TestConstructionPathCoverage:

    def test_code_node_sandbox_tool_is_wrapped(self):
        manager = MiddlewareManager()
        manager.add_wrap_only(_tehm())
        sandbox = _sync_failing_tool("pyodide_sandbox")
        schema = """
name: p
entry_point: code_node
nodes:
  - id: code_node
    type: code
    code:
      type: fixed
      value: "return 1"
    transition: END
"""

        with patch("elitea_sdk.runtime.tools.sandbox.create_sandbox_tool",
                   return_value=sandbox):
            graph = langraph_agent.create_graph(
                client=None, yaml_schema=schema, tools=[],
                memory=MemorySaver(), middleware_manager=manager,
            )

        node_tool = graph.builder.nodes["code_node"].runnable.tool
        assert getattr(node_tool, MARKER, False) is True
        assert node_tool.invoke({"value": "x"}) == "sync boom"

    def test_lazy_meta_tools_are_wrapped(self):
        from elitea_sdk.runtime.tools.lazy_tools import ToolRegistry
        from elitea_sdk.runtime.tools.llm import LLMNode

        manager = MiddlewareManager()
        manager.add_wrap_only(_tehm())
        node = LLMNode.model_construct(
            name="llm",
            tool_registry=ToolRegistry.from_tools([_sync_failing_tool()]),
            middleware_manager=manager,
            _meta_tools=None,
        )
        node._meta_tools = None

        meta_tools = node._get_meta_tools()

        assert meta_tools, "expected meta-tools to be created"
        assert all(getattr(t, MARKER, False) is True for t in meta_tools)

    def test_only_the_code_node_constructs_a_tool_locally(self):
        """Guard: a node branch that builds its own tool must also wrap it. Add the
        wrap_tool call (not an entry here) when a new node type constructs a tool."""
        import inspect

        source = inspect.getsource(langraph_agent.create_graph)
        branches = re.split(r"\n            (?:el)?if node_type", source)
        constructs = ("create_sandbox_tool(", "StructuredTool(", "StructuredTool.from_function(",
                      "instantiate_toolkit", "get_tools(")

        unwrapped = [
            branch.splitlines()[0]
            for branch in branches
            if any(pattern in branch for pattern in constructs) and "wrap_tool" not in branch
        ]

        assert unwrapped == []

    def test_node_types_are_a_known_set(self):
        """Fails when a node type is added, so its tool coverage gets reviewed."""
        import inspect

        source = inspect.getsource(langraph_agent.create_graph)
        found = set(re.findall(r"node_type == '([a-z_]+)'", source))
        found |= {t for group in re.findall(r"node_type in \[([^\]]+)\]", source)
                  for t in re.findall(r"'([a-z_]+)'", group)}

        assert found == {
            'function', 'toolkit', 'mcp', 'tool', 'loop', 'loop_from_tool', 'indexer',
            'subgraph', 'pipeline', 'agent', 'code', 'llm', 'router', 'decision',
            'state_modifier', 'printer', 'hitl', 'custom',
        }


class TestToolTestPanelClassification:
    """Decision for #6170: the test panel classifies but does not rewrite - the raw
    provider error is the whole point of that panel."""

    @staticmethod
    def _run_failing_tool(error):
        client = client_module.EliteAClient.__new__(client_module.EliteAClient)
        client.get_llm = lambda *a, **kw: object()
        client._validate_toolkit_config = lambda config: config

        with patch("elitea_sdk.runtime.utils.toolkit_utils.instantiate_toolkit_with_client",
                   return_value=[_sync_failing_tool("failing_tool", error=error)]):
            return client.test_toolkit_tool(
                toolkit_config={"type": "github", "toolkit_name": "tk", "settings": {}},
                tool_name="failing_tool",
                tool_params={"value": "x"},
            )

    def test_typed_fields_are_added(self):
        result = self._run_failing_tool(TimeoutError("connection timed out"))

        assert result["success"] is False
        assert result["error_class"] == ToolErrorClass.INFRASTRUCTURE.value
        assert result["retriable"] is True
        assert result["exception_type"] == "TimeoutError"

    def test_raw_provider_error_is_preserved(self):
        result = self._run_failing_tool(ValueError("repo not found"))

        assert "repo not found" in result["error"]
        assert result["error_class"] == ToolErrorClass.INPUT.value
        assert result["retriable"] is False

    def test_unclassifiable_error_reports_no_class(self):
        result = self._run_failing_tool(Exception("something opaque"))

        assert result["error_class"] is None
        assert result["retriable"] is False
