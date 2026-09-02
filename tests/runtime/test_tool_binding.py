import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from elitea_sdk.runtime.langchain.langraph_agent import create_graph
from elitea_sdk.runtime.tools.lazy_tools import ToolRegistry
from elitea_sdk.runtime.tools.llm import LLMNode
from elitea_sdk.runtime.tools.tool_binding import (
    build_tool_binding_plan,
    select_tools_for_binding,
)


def _tool(toolkit_name, toolkit_type, tool_name, result):
    return StructuredTool.from_function(
        func=lambda: result,
        name=tool_name,
        description=f"Run {tool_name}",
        metadata={
            "toolkit_id": 1 if toolkit_name == "attachments" else 2,
            "toolkit_name": toolkit_name,
            "toolkit_type": toolkit_type,
            "tool_name": tool_name,
        },
    )


def test_binding_plan_qualifies_every_collision_and_routes_exact_tools():
    attachments = _tool("attachments", "artifact", "list_indexes", "bucket indexes")
    configurations = _tool("configurations", "github", "list_indexes", "repo indexes")

    plan = build_tool_binding_plan([attachments, configurations])

    assert [tool.name for tool in plan.provider_tools] == [
        "attachments__list_indexes",
        "configurations__list_indexes",
    ]
    assert plan.resolve("attachments__list_indexes") is attachments
    assert plan.resolve("configurations__list_indexes") is configurations
    assert plan.provider_tools[0].invoke({}) == "bucket indexes"
    assert plan.provider_tools[1].invoke({}) == "repo indexes"
    assert attachments.name == configurations.name == "list_indexes"
    assert "Toolkit: attachments (artifact)" in plan.provider_tools[0].description
    assert "Toolkit: configurations (github)" in plan.provider_tools[1].description


def test_binding_plan_preserves_unique_names_and_is_order_independent():
    attachments = _tool("attachments", "artifact", "list_indexes", "bucket indexes")
    configurations = _tool("configurations", "github", "list_indexes", "repo indexes")
    unique = _tool("configurations", "github", "get_branches", "branches")

    forward = build_tool_binding_plan([attachments, configurations, unique])
    reverse = build_tool_binding_plan([unique, configurations, attachments])

    assert unique in forward.provider_tools
    assert {tool.name for tool in forward.provider_tools} == {
        "attachments__list_indexes",
        "configurations__list_indexes",
        "get_branches",
    }
    assert {tool.name for tool in reverse.provider_tools} == {
        "attachments__list_indexes",
        "configurations__list_indexes",
        "get_branches",
    }


def test_binding_plan_preserves_unique_existing_runtime_qualification():
    sharepoint_search = _tool("sharepoint", "mcp", "sharepoint_search", "results")
    sharepoint_search.metadata["tool_name"] = "search"

    plan = build_tool_binding_plan([sharepoint_search])

    assert plan.provider_tools == [sharepoint_search]
    assert plan.provider_tools[0].name == "sharepoint_search"
    assert plan.resolve("sharepoint_search") is sharepoint_search


def test_sanitized_toolkit_alias_collision_hashes_every_member_deterministically():
    first = _tool("same.name", "artifact", "list_indexes", "first")
    second = _tool("same name", "github", "list_indexes", "second")

    forward_names = {
        tool.name for tool in build_tool_binding_plan([first, second]).provider_tools
    }
    reverse_names = {
        tool.name for tool in build_tool_binding_plan([second, first]).provider_tools
    }

    assert forward_names == reverse_names
    assert len(forward_names) == 2
    assert all(name.startswith("same_name__list_indexes__") for name in forward_names)


def test_collision_alias_does_not_shadow_an_existing_runtime_name():
    attachments = _tool("attachments", "artifact", "list_indexes", "bucket indexes")
    configurations = _tool("configurations", "github", "list_indexes", "repo indexes")
    existing = _tool(
        "custom",
        "mcp",
        "attachments__list_indexes",
        "existing result",
    )

    plan = build_tool_binding_plan([attachments, configurations, existing])

    assert len({tool.name for tool in plan.provider_tools}) == 3
    assert plan.resolve("attachments__list_indexes") is existing
    assert any(
        tool.name.startswith("attachments__list_indexes__")
        for tool in plan.provider_tools
    )


def test_toolkit_scoped_selection_does_not_substitute_another_toolkit():
    attachments = _tool("attachments", "artifact", "list_indexes", "bucket indexes")
    configurations = _tool("configurations", "github", "list_indexes", "repo indexes")
    branches = _tool("configurations", "github", "get_branches", "branches")

    selected, missing = select_tools_for_binding(
        [attachments, configurations, branches],
        {"configurations": ["list_indexes"]},
    )

    assert selected == [configurations]
    assert missing == []


def test_legacy_name_list_keeps_all_matching_toolkit_operations():
    attachments = _tool("attachments", "artifact", "list_indexes", "bucket indexes")
    configurations = _tool("configurations", "github", "list_indexes", "repo indexes")

    selected, missing = select_tools_for_binding(
        [attachments, configurations],
        ["list_indexes"],
    )

    assert selected == [attachments, configurations]
    assert missing == []


def test_pipeline_llm_node_preserves_qualified_tool_selection():
    attachments = _tool("attachments", "artifact", "list_indexes", "bucket indexes")
    configurations = _tool("configurations", "github", "list_indexes", "repo indexes")
    schema = yaml.safe_dump(
        {
            "name": "pipeline",
            "state": {"messages": {"type": "list"}},
            "nodes": [
                {
                    "id": "LLM1",
                    "type": "llm",
                    "tool_names": {
                        "attachments": ["list_indexes"],
                        "configurations": ["list_indexes"],
                    },
                    "transition": "END",
                },
            ],
            "entry_point": "LLM1",
        },
        default_flow_style=False,
    )

    graph = create_graph(
        client=None,
        yaml_schema=schema,
        tools=[attachments, configurations],
        memory=MemorySaver(),
    )
    llm_node = graph.get_graph().nodes["LLM1"].data
    filtered = llm_node.get_filtered_tools()
    plan = build_tool_binding_plan(filtered)

    assert filtered == [attachments, configurations]
    assert [tool.name for tool in plan.provider_tools] == [
        "attachments__list_indexes",
        "configurations__list_indexes",
    ]
    assert llm_node._resolve_tool_to_execute("attachments__list_indexes", {}) is attachments
    assert llm_node._resolve_tool_to_execute("configurations__list_indexes", {}) is configurations


def test_pipeline_llm_selective_tool_does_not_fall_back_to_same_name():
    attachments = _tool("attachments", "artifact", "list_indexes", "bucket indexes")
    configurations = _tool("configurations", "github", "list_indexes", "repo indexes")
    schema = yaml.safe_dump(
        {
            "name": "pipeline",
            "state": {"messages": {"type": "list"}},
            "nodes": [
                {
                    "id": "LLM1",
                    "type": "llm",
                    "tool_names": {"configurations": ["list_indexes"]},
                    "transition": "END",
                },
            ],
            "entry_point": "LLM1",
        },
        default_flow_style=False,
    )

    graph = create_graph(
        client=None,
        yaml_schema=schema,
        tools=[attachments, configurations],
        memory=MemorySaver(),
    )
    llm_node = graph.get_graph().nodes["LLM1"].data

    assert llm_node.get_filtered_tools() == [configurations]


class _CollisionCallingClient:
    def __init__(self):
        self.bound_tool_names = []
        self.invocations = 0

    def bind_tools(self, tools, **_kwargs):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages, config=None):
        self.invocations += 1
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="done")
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "attachments__list_indexes", "args": {}, "id": "bucket-call"},
                {"name": "configurations__list_indexes", "args": {}, "id": "repo-call"},
            ],
        )


def test_llm_node_binds_and_executes_both_qualified_collisions():
    attachments = _tool("attachments", "artifact", "list_indexes", "bucket indexes")
    configurations = _tool("configurations", "github", "list_indexes", "repo indexes")
    client = _CollisionCallingClient()
    node = LLMNode(
        client=client,
        available_tools=[attachments, configurations],
        lazy_tools_mode=False,
        input_mapping={},
        output_variables=["messages"],
    )

    result = node.invoke({"messages": [HumanMessage(content="list indexes")]})

    assert client.bound_tool_names == [
        "attachments__list_indexes",
        "configurations__list_indexes",
    ]
    assert client.invocations == 2
    assert [
        message.content for message in result["messages"] if isinstance(message, ToolMessage)
    ] == ["bucket indexes", "repo indexes"]


def test_smart_tool_selection_preserves_colliding_toolkit_tools():
    attachments = _tool("attachments", "artifact", "list_indexes", "bucket indexes")
    configurations = _tool("configurations", "github", "list_indexes", "repo indexes")
    selected_tools = [attachments, configurations]
    client = _CollisionCallingClient()
    node = LLMNode(
        client=client,
        available_tools=selected_tools,
        tool_registry=ToolRegistry.from_tools(selected_tools),
        lazy_tools_mode=True,
        input_mapping={},
        output_variables=["messages"],
    )

    result = node.invoke(
        {"messages": [HumanMessage(content="list indexes")]},
        config={"configurable": {"selected_tools": selected_tools}},
    )

    assert client.bound_tool_names == [
        "attachments__list_indexes",
        "configurations__list_indexes",
    ]
    assert [
        message.content for message in result["messages"] if isinstance(message, ToolMessage)
    ] == ["bucket indexes", "repo indexes"]


def test_provider_aliases_delegate_through_standard_tool_node():
    attachments = _tool("attachments", "artifact", "list_indexes", "bucket indexes")
    configurations = _tool("configurations", "github", "list_indexes", "repo indexes")
    plan = build_tool_binding_plan([attachments, configurations])
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode(plan.provider_tools))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()

    result = graph.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "attachments__list_indexes",
                            "args": {},
                            "id": "bucket-call",
                        },
                        {
                            "name": "configurations__list_indexes",
                            "args": {},
                            "id": "repo-call",
                        },
                    ],
                )
            ]
        }
    )

    assert [message.content for message in result["messages"][-2:]] == [
        "bucket indexes",
        "repo indexes",
    ]
