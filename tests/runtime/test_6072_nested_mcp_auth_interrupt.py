import json

import pytest
import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.langchain.langraph_agent import create_graph
from elitea_sdk.runtime.tools.application import Application
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired


class _Runtime:
    def get_mcp_toolkits(self):
        return []


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
