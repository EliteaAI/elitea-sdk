import json

import pytest
import yaml
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.langchain.langraph_agent import create_graph
from elitea_sdk.runtime.tools.application import Application
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired


def _assert_tool_messages_have_owning_calls(messages):
    """Enforce the provider-neutral assistant-call -> tool-result contract."""
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            continue
        owner = index - 1
        while owner >= 0 and isinstance(messages[owner], ToolMessage):
            owner -= 1
        assert owner >= 0
        assert isinstance(messages[owner], AIMessage)
        assert message.tool_call_id in {
            call.get("id") for call in messages[owner].tool_calls
        }


class _Runtime:
    def get_mcp_toolkits(self):
        return []


class _CustomEventCapture(BaseCallbackHandler):
    def __init__(self):
        self.events = []

    def on_custom_event(self, name, data, **kwargs):
        self.events.append((name, data, kwargs))


class _AuthToolCounter:
    def __init__(self):
        self.calls = 0

    def raise_auth(self, query=""):
        self.calls += 1
        raise McpAuthorizationRequired(
            "SharePoint authorization is required",
            server_url="https://tenant.sharepoint.com/sites/research",
            resource_metadata={
                "resource_name": "SharePoint",
                "authorization_servers": ["https://login.microsoftonline.com/common"],
            },
            tool_name="search",
            toolkit_name="SharePoint",
            toolkit_type="sharepoint",
        )


class _ChildLLM:
    temperature = 0
    max_tokens = 1000

    def __init__(self):
        self.invocations = []

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _BoundChildLLM(self, tools)

    def invoke(self, messages, config=None):
        return _BoundChildLLM(self, []).invoke(messages, config=config)


class _BoundChildLLM:
    def __init__(self, root, tools):
        self.root = root
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        self.root.invocations.append(list(messages))
        _assert_tool_messages_have_owning_calls(messages)
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            try:
                payload = json.loads(str(message.content))
            except (TypeError, ValueError):
                continue
            if payload.get("type") == "mcp_auth_decision":
                return AIMessage(content=f"child-finished:{payload['status']}")

        return AIMessage(
            content="",
            tool_calls=[{
                "name": "sharepoint_search",
                "args": {"query": "quarterly results"},
                "id": "call-sharepoint-auth",
                "type": "tool_call",
            }],
        )


class _ParentLLM:
    temperature = 0
    max_tokens = 1000

    def __init__(self):
        self.invocations = []

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return _BoundParentLLM(self, tools)

    def invoke(self, messages, config=None):
        return _BoundParentLLM(self, []).invoke(messages, config=config)


class _BoundParentLLM:
    def __init__(self, root, tools):
        self.root = root
        self.tools = list(tools)

    def invoke(self, messages, config=None):
        self.root.invocations.append(list(messages))
        tool_contents = [str(message.content) for message in messages if isinstance(message, ToolMessage)]
        if "child-finished:authorized" in tool_contents:
            return AIMessage(content="parent-finished")
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "ResearchAgent",
                "args": {"task": "Search SharePoint for quarterly results"},
                "id": "call-research-agent",
                "type": "tool_call",
            }],
        )


def _assistant(llm, tools, memory):
    return Assistant(
        elitea=_Runtime(),
        data={"instructions": "Use tools", "tools": [], "meta": {}},
        client=llm,
        tools=tools,
        memory=memory,
        app_type="predict",
    ).runnable()


def test_direct_agent_skip_restores_tool_call_before_structured_result():
    counter = _AuthToolCounter()
    auth_tool = StructuredTool.from_function(
        func=counter.raise_auth,
        name="sharepoint_search",
        description="Search SharePoint",
        metadata={
            "tool_name": "search",
            "toolkit_name": "SharePoint",
            "toolkit_type": "sharepoint",
        },
    )
    memory = MemorySaver()
    capture = _CustomEventCapture()
    config = {
        "configurable": {"thread_id": "issue-6072-direct-skip"},
        "callbacks": [capture],
    }

    paused = _assistant(_ChildLLM(), [auth_tool], memory).invoke(
        {"messages": [HumanMessage(content="Search SharePoint")]},
        config=config,
    )
    assert paused["execution_finished"] is False
    assert counter.calls == 1

    resumed_llm = _ChildLLM()
    resumed = _assistant(resumed_llm, [auth_tool], memory).invoke(
        {"mcp_auth_resume": True, "mcp_auth_action": "skip"},
        config=config,
    )

    assert resumed["execution_finished"] is True
    assert resumed["output"] == "child-finished:declined"
    assert counter.calls == 1
    assert len(resumed_llm.invocations) == 1

    tool_result = next(
        message
        for message in resumed_llm.invocations[0]
        if isinstance(message, ToolMessage)
    )
    payload = json.loads(str(tool_result.content))
    assert payload["type"] == "mcp_auth_decision"
    assert payload["status"] == "declined"
    assert payload["next_step"] == (
        "Do not request this toolkit again during the current run. Execute every "
        "remaining step that does not require it, including all other required "
        "tool calls. Do not finish early solely because authorization was skipped, "
        "and do not repeat completed work."
    )
    assert payload["denial_reason"] == "User skipped authorization for this run."
    auth_events = [
        data for name, data, _kwargs in capture.events
        if name == "mcp_auth_decision"
    ]
    assert len(auth_events) == 1
    assert auth_events[0]["tool_name"] == "sharepoint_search"
    assert auth_events[0]["tool_call_id"] == "call-sharepoint-auth"
    assert json.loads(auth_events[0]["tool_output"])["status"] == "declined"


class _ContinueAfterAuthLLM(_ChildLLM):
    def bind_tools(self, tools, **kwargs):
        return _ContinueAfterAuthBound(self, tools)

    def invoke(self, messages, config=None):
        return _ContinueAfterAuthBound(self, []).invoke(messages, config=config)


class _ContinueAfterAuthBound(_BoundChildLLM):
    def invoke(self, messages, config=None):
        self.root.invocations.append(list(messages))
        _assert_tool_messages_have_owning_calls(messages)
        results = [str(message.content) for message in messages if isinstance(message, ToolMessage)]
        if "follow-up-complete" in results:
            return AIMessage(content="child-finished-after-follow-up")
        for result in results:
            try:
                payload = json.loads(result)
            except (TypeError, ValueError):
                continue
            if payload.get("type") == "mcp_auth_decision":
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "follow_up_tool",
                        "args": {},
                        "id": "call-follow-up",
                        "type": "tool_call",
                    }],
                )
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "sharepoint_search",
                "args": {"query": "quarterly results"},
                "id": "call-sharepoint-auth",
                "type": "tool_call",
            }],
        )


class _PipelineAfterAuthLLM:
    temperature = 0
    max_tokens = 1000

    def __init__(self):
        self.invocations = []
        self.responses = [
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "mcp_authorize_sharepoint",
                    "args": {"arguments": {"operation": "get_lists"}},
                    "id": "call-sharepoint-auth",
                    "type": "tool_call",
                }],
            ),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_lists",
                    "args": {},
                    "id": "call-get-lists",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="lists-complete"),
            AIMessage(content="summary-complete"),
        ]

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        _ = tools, kwargs
        return self

    def invoke(self, messages, config=None):
        _ = config
        self.invocations.append(list(messages))
        return self.responses.pop(0)


class _NestedPipelineParentLLM:
    temperature = 0
    max_tokens = 1000

    def __init__(self):
        self.invocations = []

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        _ = tools, kwargs
        return self

    def invoke(self, messages, config=None):
        _ = config
        self.invocations.append(list(messages))
        if any(
            isinstance(message, ToolMessage)
            and "summary-complete" in str(message.content)
            for message in messages
        ):
            return AIMessage(content="parent-finished")
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "NestedPipeline",
                "args": {"task": "list sites"},
                "id": "call-nested-pipeline",
                "type": "tool_call",
            }],
        )


class _ParallelNestedPipelineParentLLM(_NestedPipelineParentLLM):
    def invoke(self, messages, config=None):
        _ = config
        self.invocations.append(list(messages))
        tool_messages = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        if tool_messages:
            return AIMessage(content="parent-finished")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "NestedPipeline",
                    "args": {"task": "input-a"},
                    "id": "call-pipeline-a",
                    "type": "tool_call",
                },
                {
                    "name": "NestedPipeline",
                    "args": {"task": "input-b"},
                    "id": "call-pipeline-b",
                    "type": "tool_call",
                },
            ],
        )


class _ParallelPipelineChildLLM:
    temperature = 0
    max_tokens = 1000

    def __init__(self):
        self.invocations = []

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        _ = tools, kwargs
        return self

    def invoke(self, messages, config=None):
        configurable = (config or {}).get("configurable", {})
        thread_id = configurable.get("thread_id", "")
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        node_name = checkpoint_ns.rsplit("|", 1)[-1].split(":", 1)[0]
        tag = "a" if thread_id.endswith("call-pipeline-a") else "b"
        self.invocations.append((thread_id, node_name, list(messages)))

        if node_name == "LLM2":
            return AIMessage(content=f"summary-{tag}")

        tool_messages = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        if any(
            message.tool_call_id == f"call-get-lists-{tag}"
            for message in tool_messages
        ):
            return AIMessage(content=f"lists-complete-{tag}")
        if any("mcp_auth_decision" in str(message.content) for message in tool_messages):
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_lists",
                    "args": {"request": tag},
                    "id": f"call-get-lists-{tag}",
                    "type": "tool_call",
                }],
            )
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "mcp_authorize_sharepoint",
                "args": {"arguments": {"operation": "get_lists"}},
                "id": f"call-sharepoint-auth-{tag}",
                "type": "tool_call",
            }],
        )


class _RebuildingApplicationClient:
    def __init__(self, runnable):
        self.runnable = runnable

    def application(self, *args, **kwargs):
        _ = args, kwargs
        return self.runnable


def _llm_pipeline_schema():
    return yaml.safe_dump({
        "name": "llm-toolkit-auth",
        "state": {
            "input": {"type": "str"},
            "sharepoint": {"type": "str"},
            "summary": {"type": "str"},
            "messages": {"type": "list"},
        },
        "nodes": [{
            "id": "LLM1",
            "type": "llm",
            "input_mapping": {
                "system": {"type": "fixed", "value": "Use SharePoint tools."},
                "task": {"type": "variable", "value": "input"},
            },
            "input": ["input"],
            "output": ["sharepoint"],
            "tool_names": {"sharepoint": ["get_lists"]},
            "transition": "LLM2",
        }, {
            "id": "LLM2",
            "type": "llm",
            "input_mapping": {
                "system": {"type": "fixed", "value": "Summarize the result."},
                "task": {"type": "variable", "value": "sharepoint"},
            },
            "input": ["sharepoint"],
            "output": ["summary"],
            "transition": "END",
        }],
        "entry_point": "LLM1",
    })


def test_pipeline_does_not_resurface_resolved_mcp_auth_after_downstream_node():
    memory = MemorySaver()
    client = _PipelineAfterAuthLLM()
    config = {"configurable": {"thread_id": "issue-6451-llm-pipeline"}}

    def require_auth(arguments=None):
        _ = arguments
        raise McpAuthorizationRequired(
            "SharePoint authorization is required",
            server_url="https://tenant.sharepoint.com/sites/pipeline",
            tool_name="get_lists",
            toolkit_name="sharepoint",
            toolkit_type="sharepoint",
        )

    auth_proxy = StructuredTool.from_function(
        func=require_auth,
        name="mcp_authorize_sharepoint",
        description="SharePoint authorization gateway",
        metadata={
            "tool_name": "mcp_authorize_sharepoint",
            "toolkit_name": "sharepoint",
            "toolkit_type": "sharepoint",
        },
    )
    paused_graph = create_graph(
        client=client,
        yaml_schema=_llm_pipeline_schema(),
        tools=[auth_proxy],
        memory=memory,
    )
    paused = paused_graph.invoke({"input": "list sites"}, config=config)

    assert paused["execution_finished"] is False
    original_interrupt_id = paused["hitl_interrupt"]["interrupt_id"]

    list_calls = []
    real_tool = StructuredTool.from_function(
        func=lambda: list_calls.append(True) or "list-a",
        name="get_lists",
        description="List SharePoint lists",
        metadata={
            "tool_name": "get_lists",
            "toolkit_name": "sharepoint",
            "toolkit_type": "sharepoint",
        },
    )
    resumed_graph = create_graph(
        client=client,
        yaml_schema=_llm_pipeline_schema(),
        tools=[auth_proxy, real_tool],
        memory=memory,
    )
    resumed = resumed_graph.invoke(
        {
            "mcp_auth_resume": True,
            "mcp_auth_action": "authorize",
            "authorization_request_id": original_interrupt_id,
        },
        config=config,
    )

    assert list_calls == [True]
    assert resumed["execution_finished"] is True
    assert "hitl_interrupt" not in resumed
    assert resumed["summary"] == "summary-complete"


def test_nested_pipeline_auth_resumes_parent_application_without_replanning():
    child_memory = MemorySaver()
    child_client = _PipelineAfterAuthLLM()

    def require_auth(arguments=None):
        _ = arguments
        raise McpAuthorizationRequired(
            "SharePoint authorization is required",
            server_url="https://tenant.sharepoint.com/sites/pipeline",
            tool_name="get_lists",
            toolkit_name="sharepoint",
            toolkit_type="sharepoint",
        )

    auth_proxy = StructuredTool.from_function(
        func=require_auth,
        name="mcp_authorize_sharepoint",
        description="SharePoint authorization gateway",
        metadata={
            "tool_name": "mcp_authorize_sharepoint",
            "toolkit_name": "sharepoint",
            "toolkit_type": "sharepoint",
        },
    )
    list_calls = []
    real_tool = StructuredTool.from_function(
        func=lambda: list_calls.append(True) or "list-a",
        name="get_lists",
        description="List SharePoint lists",
        metadata={
            "tool_name": "get_lists",
            "toolkit_name": "sharepoint",
            "toolkit_type": "sharepoint",
        },
    )
    child_graph = create_graph(
        client=child_client,
        yaml_schema=_llm_pipeline_schema(),
        tools=[auth_proxy, real_tool],
        memory=child_memory,
    )
    child_tool = Application(
        name="Nested Pipeline",
        description="Delegated pipeline",
        application=child_graph,
        return_type="str",
        client=_RebuildingApplicationClient(child_graph),
        args_runnable={
            "application_id": 25,
            "application_version_id": 1,
            "version_details": {},
        },
        metadata={
            "original_name": "Nested Pipeline",
            "agent_type": "pipeline",
        },
    )

    parent_memory = MemorySaver()
    config = {"configurable": {"thread_id": "issue-6451-nested-pipeline"}}
    first_parent = _NestedPipelineParentLLM()
    paused = _assistant(first_parent, [child_tool], parent_memory).invoke(
        {"messages": [HumanMessage(content="Run the delegated pipeline")]},
        config=config,
    )

    assert paused["execution_finished"] is False
    assert paused["hitl_interrupt"]["parent_agent_call_id"] == (
        "call-nested-pipeline"
    )

    resumed_parent = _NestedPipelineParentLLM()
    resumed = _assistant(resumed_parent, [child_tool], parent_memory).invoke(
        {"mcp_auth_resume": True, "mcp_auth_action": "authorize"},
        config=config,
    )

    assert resumed["execution_finished"] is True
    assert resumed["output"] == "parent-finished"
    assert list_calls == [True]
    # Only the post-tool finalization is allowed. A second invocation before
    # the Application result means the orchestrator replanned a new child call.
    assert len(resumed_parent.invocations) == 1


def test_parallel_same_nested_pipeline_resumes_isolated_calls_without_replanning():
    child_memory = MemorySaver()
    child_client = _ParallelPipelineChildLLM()

    def require_auth(arguments=None):
        _ = arguments
        raise McpAuthorizationRequired(
            "SharePoint authorization is required",
            server_url="https://tenant.sharepoint.com/sites/pipeline",
            tool_name="get_lists",
            toolkit_name="sharepoint",
            toolkit_type="sharepoint",
        )

    auth_proxy = StructuredTool.from_function(
        func=require_auth,
        name="mcp_authorize_sharepoint",
        description="SharePoint authorization gateway",
        metadata={
            "tool_name": "mcp_authorize_sharepoint",
            "toolkit_name": "sharepoint",
            "toolkit_type": "sharepoint",
        },
    )
    list_calls = []

    def get_lists(request: str):
        list_calls.append(request)
        return f"lists-{request}"

    real_tool = StructuredTool.from_function(
        func=get_lists,
        name="get_lists",
        description="List SharePoint lists for one request",
        metadata={
            "tool_name": "get_lists",
            "toolkit_name": "sharepoint",
            "toolkit_type": "sharepoint",
        },
    )
    child_graph = create_graph(
        client=child_client,
        yaml_schema=_llm_pipeline_schema(),
        tools=[auth_proxy, real_tool],
        memory=child_memory,
    )
    child_tool = Application(
        name="Nested Pipeline",
        description="Delegated pipeline",
        application=child_graph,
        return_type="str",
        client=_RebuildingApplicationClient(child_graph),
        args_runnable={
            "application_id": 25,
            "application_version_id": 1,
            "version_details": {},
        },
        metadata={
            "original_name": "Nested Pipeline",
            "agent_type": "pipeline",
        },
    )

    parent_memory = MemorySaver()
    config = {"configurable": {"thread_id": "issue-6451-parallel-pipelines"}}
    initial_parent = _ParallelNestedPipelineParentLLM()
    paused = _assistant(initial_parent, [child_tool], parent_memory).invoke(
        {"messages": [HumanMessage(content="Run both pipeline inputs")]},
        config=config,
    )

    assert paused["execution_finished"] is False
    interrupts = paused["hitl_interrupts"]
    by_call = {interrupt["tool_call_id"]: interrupt for interrupt in interrupts}
    assert set(by_call) == {"call-pipeline-a", "call-pipeline-b"}
    assert by_call["call-pipeline-a"]["child_thread_id"].endswith(
        ":call-pipeline-a"
    )
    assert by_call["call-pipeline-b"]["child_thread_id"].endswith(
        ":call-pipeline-b"
    )
    assert (
        by_call["call-pipeline-a"]["child_thread_id"]
        != by_call["call-pipeline-b"]["child_thread_id"]
    )

    resumed_parent = _ParallelNestedPipelineParentLLM()
    resumed = _assistant(resumed_parent, [child_tool], parent_memory).invoke(
        {
            "hitl_decisions": [
                {"tool_call_id": "call-pipeline-a", "action": "authorize"},
                {"tool_call_id": "call-pipeline-b", "action": "authorize"},
            ],
        },
        config=config,
    )

    assert resumed["execution_finished"] is True
    assert resumed["output"] == "parent-finished"
    assert sorted(list_calls) == ["a", "b"]
    assert len(resumed_parent.invocations) == 1
    final_tool_results = {
        str(message.content)
        for message in resumed_parent.invocations[0]
        if isinstance(message, ToolMessage)
    }
    assert final_tool_results == {"summary-a", "summary-b"}


def test_direct_agent_auth_decision_continues_same_leaf_tool_loop():
    counter = _AuthToolCounter()
    auth_tool = StructuredTool.from_function(
        func=counter.raise_auth,
        name="sharepoint_search",
        description="Search SharePoint",
    )
    follow_up_calls = []
    follow_up_tool = StructuredTool.from_function(
        func=lambda: follow_up_calls.append(True) or "follow-up-complete",
        name="follow_up_tool",
        description="Perform remaining child work",
    )
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "issue-6072-continue-loop"}}

    paused = _assistant(
        _ContinueAfterAuthLLM(), [auth_tool, follow_up_tool], memory,
    ).invoke(
        {"messages": [HumanMessage(content="Authorize, then continue work")]},
        config=config,
    )
    assert paused["execution_finished"] is False

    resumed = _assistant(
        _ContinueAfterAuthLLM(), [auth_tool, follow_up_tool], memory,
    ).invoke(
        {"mcp_auth_resume": True, "mcp_auth_action": "skip"},
        config=config,
    )

    assert resumed["execution_finished"] is True
    assert resumed["output"] == "child-finished-after-follow-up"
    assert follow_up_calls == [True]
    assert counter.calls == 1


def test_nested_mcp_auth_uses_durable_interrupt_and_resumes_same_child_call():
    counter = _AuthToolCounter()
    auth_tool = StructuredTool.from_function(
        func=counter.raise_auth,
        name="sharepoint_search",
        description="Search SharePoint",
        metadata={
            "tool_name": "search",
            "toolkit_name": "SharePoint",
            "toolkit_type": "sharepoint",
        },
    )
    child_memory = MemorySaver()
    child_llm = _ChildLLM()
    child_runnable = _assistant(child_llm, [auth_tool], child_memory)
    child_tool = Application(
        name="Research Agent",
        description="Delegated research agent",
        application=child_runnable,
        return_type="str",
        client=None,
        metadata={"original_name": "Research Agent", "agent_type": "agent"},
    )

    parent_memory = MemorySaver()
    thread_config = {"configurable": {"thread_id": "issue-6072-parent"}}
    first_parent_llm = _ParentLLM()
    first_runnable = _assistant(first_parent_llm, [child_tool], parent_memory)

    paused = first_runnable.invoke(
        {"messages": [HumanMessage(content="Delegate the SharePoint search")]},
        config=thread_config,
    )

    assert paused["execution_finished"] is False
    interrupt = paused["hitl_interrupt"]
    assert interrupt["guardrail_type"] == "mcp_auth"
    assert interrupt["tool_call_id"] == "call-sharepoint-auth"
    assert interrupt["parent_agent_call_id"] == "call-research-agent"
    assert interrupt["parent_agent_path"] == [{
        "name": "Research Agent",
        "call_id": "call-research-agent",
    }]
    assert interrupt["thread_id"] == "issue-6072-parent:ResearchAgent"
    assert counter.calls == 1

    second_parent_llm = _ParentLLM()
    resumed = _assistant(second_parent_llm, [child_tool], parent_memory).invoke(
        {"mcp_auth_resume": True, "mcp_auth_action": "authorize"},
        config=thread_config,
    )

    assert resumed["execution_finished"] is True
    assert resumed["output"] == "parent-finished"
    # LangGraph re-executes the interrupted node.  The authorization boundary
    # is therefore probed again, but the parent LLM is not re-planned and the
    # child/tool call identities above remain unchanged.
    assert counter.calls == 2
    assert len(second_parent_llm.invocations) == 1


def test_nested_mcp_auth_rejects_stale_interrupt_identity():
    counter = _AuthToolCounter()
    auth_tool = StructuredTool.from_function(
        func=counter.raise_auth,
        name="sharepoint_search",
        description="Search SharePoint",
        metadata={
            "tool_name": "search",
            "toolkit_name": "SharePoint",
            "toolkit_type": "sharepoint",
        },
    )
    child_memory = MemorySaver()
    child_runnable = _assistant(_ChildLLM(), [auth_tool], child_memory)
    child_tool = Application(
        name="Research Agent",
        description="Delegated research agent",
        application=child_runnable,
        return_type="str",
        client=None,
        metadata={"original_name": "Research Agent", "agent_type": "agent"},
    )
    parent_memory = MemorySaver()
    config = {"configurable": {"thread_id": "issue-6072-stale-id"}}
    paused = _assistant(_ParentLLM(), [child_tool], parent_memory).invoke(
        {"messages": [HumanMessage(content="Delegate the SharePoint search")]},
        config=config,
    )

    stale = _assistant(_ParentLLM(), [child_tool], parent_memory).invoke(
        {
            "hitl_resume": True,
            "hitl_decisions": [{
                "interrupt_id": "mcp_auth_stale",
                "action": "skip",
                "value": "",
            }],
        },
        config=config,
    )

    assert stale["execution_finished"] is False
    assert stale["hitl_interrupt"]["interrupt_id"] == (
        paused["hitl_interrupt"]["interrupt_id"]
    )
    assert counter.calls == 1


def _direct_toolkit_schema():
    return yaml.safe_dump({
        "name": "direct-toolkit-auth",
        "state": {
            "messages": {"type": "list"},
            "toolkit_result": {"type": "str"},
        },
        "nodes": [{
            "id": "SharePointNode",
            "type": "toolkit",
            "toolkit_name": "SharePoint",
            "tool": "get_lists",
            "output": ["toolkit_result"],
            "transition": "END",
        }],
        "entry_point": "SharePointNode",
    })


def _direct_toolkit_schema_with_downstream_node():
    return yaml.safe_dump({
        "name": "direct-toolkit-auth-with-downstream",
        "state": {
            "messages": {"type": "list"},
            "toolkit_result": {"type": "str"},
            "downstream_result": {"type": "str"},
        },
        "nodes": [{
            "id": "SharePointNode",
            "type": "toolkit",
            "toolkit_name": "SharePoint",
            "tool": "get_lists",
            "output": ["toolkit_result"],
            "transition": "DownstreamNode",
        }, {
            "id": "DownstreamNode",
            "type": "llm",
            "input_mapping": {
                "system": {"type": "fixed", "value": "Summarize the result."},
                "task": {"type": "variable", "value": "toolkit_result"},
            },
            "input": ["toolkit_result"],
            "output": ["downstream_result"],
            "transition": "END",
        }],
        "entry_point": "SharePointNode",
    })


@pytest.mark.parametrize("action", ["authorize", "skip"])
def test_direct_pipeline_toolkit_auth_interrupt_resumes_exact_function_node(action):
    memory = MemorySaver()
    config = {"configurable": {"thread_id": f"issue-6072-pipeline-{action}"}}

    def require_auth(arguments=None):
        _ = arguments
        raise McpAuthorizationRequired(
            "SharePoint authorization is required",
            server_url="https://tenant.sharepoint.com/sites/pipeline",
            tool_name="get_lists",
            toolkit_name="SharePoint",
            toolkit_type="sharepoint",
        )

    proxy = StructuredTool.from_function(
        func=require_auth,
        name="mcp_authorize_SharePoint",
        description="SharePoint authorization gateway",
        metadata={
            "tool_name": "mcp_authorize_SharePoint",
            "toolkit_name": "SharePoint",
            "toolkit_type": "sharepoint",
        },
    )
    paused_graph = create_graph(
        client=None,
        yaml_schema=_direct_toolkit_schema(),
        tools=[proxy],
        memory=memory,
    )
    paused = paused_graph.invoke(
        {"messages": [HumanMessage(content="run the pipeline")]},
        config=config,
    )

    assert paused["execution_finished"] is False
    assert paused["hitl_interrupt"]["guardrail_type"] == "mcp_auth"
    assert paused["hitl_interrupt"]["node_name"] == "SharePointNode"
    assert paused["hitl_interrupt"]["tool_name"] == "get_lists"
    assert paused["hitl_interrupt"]["tool_call_id"] == "pipeline:SharePointNode"

    if action == "authorize":
        real_tool = StructuredTool.from_function(
            func=lambda messages=None: "authorized-lists",
            name="get_lists",
            description="List SharePoint lists",
            metadata={
                "tool_name": "get_lists",
                "toolkit_name": "SharePoint",
                "toolkit_type": "sharepoint",
            },
        )
        resumed_graph = create_graph(
            client=None,
            yaml_schema=_direct_toolkit_schema(),
            tools=[real_tool],
            memory=memory,
        )
    else:
        resumed_graph = paused_graph

    resumed = resumed_graph.invoke(
        {"mcp_auth_resume": True, "mcp_auth_action": action},
        config=config,
    )

    assert resumed["execution_finished"] is True
    if action == "authorize":
        assert resumed["toolkit_result"] == "authorized-lists"
    else:
        assert resumed["toolkit_result"] is None
        assert "authorization for **SharePoint**" in resumed["output"]


def test_direct_pipeline_does_not_resurface_resolved_auth_after_downstream_node():
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "issue-6451-direct-pipeline"}}
    client = _PipelineAfterAuthLLM()
    client.responses = [AIMessage(content="summary-complete")]

    def require_auth(arguments=None):
        _ = arguments
        raise McpAuthorizationRequired(
            "SharePoint authorization is required",
            server_url="https://tenant.sharepoint.com/sites/pipeline",
            tool_name="get_lists",
            toolkit_name="SharePoint",
            toolkit_type="sharepoint",
        )

    proxy = StructuredTool.from_function(
        func=require_auth,
        name="mcp_authorize_SharePoint",
        description="SharePoint authorization gateway",
        metadata={
            "tool_name": "mcp_authorize_SharePoint",
            "toolkit_name": "SharePoint",
            "toolkit_type": "sharepoint",
        },
    )
    paused_graph = create_graph(
        client=client,
        yaml_schema=_direct_toolkit_schema_with_downstream_node(),
        tools=[proxy],
        memory=memory,
    )
    paused = paused_graph.invoke(
        {"messages": [HumanMessage(content="run the pipeline")]},
        config=config,
    )

    assert paused["execution_finished"] is False
    original_interrupt_id = paused["hitl_interrupt"]["interrupt_id"]

    real_tool = StructuredTool.from_function(
        func=lambda: "authorized-lists",
        name="get_lists",
        description="List SharePoint lists",
        metadata={
            "tool_name": "get_lists",
            "toolkit_name": "SharePoint",
            "toolkit_type": "sharepoint",
        },
    )
    resumed_graph = create_graph(
        client=client,
        yaml_schema=_direct_toolkit_schema_with_downstream_node(),
        tools=[real_tool],
        memory=memory,
    )
    resumed = resumed_graph.invoke(
        {
            "mcp_auth_resume": True,
            "mcp_auth_action": "authorize",
            "authorization_request_id": original_interrupt_id,
        },
        config=config,
    )

    assert resumed["execution_finished"] is True
    assert "hitl_interrupt" not in resumed
    assert resumed["toolkit_result"] == "authorized-lists"
    assert resumed["downstream_result"] == "summary-complete"


def test_direct_pipeline_authorize_fails_closed_if_rebuild_has_no_real_tool():
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "issue-6072-pipeline-stale"}}

    def require_auth(arguments=None):
        _ = arguments
        raise McpAuthorizationRequired(
            "SharePoint authorization is required",
            server_url="https://tenant.sharepoint.com/sites/pipeline",
            tool_name="get_lists",
            toolkit_name="SharePoint",
            toolkit_type="sharepoint",
        )

    proxy = StructuredTool.from_function(
        func=require_auth,
        name="mcp_authorize_SharePoint",
        description="SharePoint authorization gateway",
        metadata={
            "tool_name": "mcp_authorize_SharePoint",
            "toolkit_name": "SharePoint",
            "toolkit_type": "sharepoint",
        },
    )
    graph = create_graph(
        client=None,
        yaml_schema=_direct_toolkit_schema(),
        tools=[proxy],
        memory=memory,
    )
    graph.invoke(
        {"messages": [HumanMessage(content="run the pipeline")]},
        config=config,
    )

    resumed = graph.invoke(
        {"mcp_auth_resume": True, "mcp_auth_action": "authorize"},
        config=config,
    )

    assert resumed["execution_finished"] is True
    assert resumed["toolkit_result"] is None
    assert "authorization for **SharePoint**" in resumed["output"]
    assert "completed" in resumed["output"]
    assert "was skipped" not in resumed["output"]
