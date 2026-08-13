"""Regression coverage for MCP auth routing in pipelines and conversation chat."""

import json
from types import SimpleNamespace

import pytest
import yaml
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

import elitea_sdk.tools as elitea_tools_mod
from elitea_sdk.runtime.langchain.langraph_agent import create_graph
from elitea_sdk.runtime.toolkits import tools as runtime_tools
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired, McpContext


SERVER_NAME = "Epam Sharepoint Delegated"
TOOLKIT_NAME = "EpamSharepointDelegated"
SERVER_URL = "https://sharepoint-mcp.example.test/mcp"
SHAREPOINT_URL = "https://tenant.sharepoint.com/sites/demo"


def _auth_error() -> McpAuthorizationRequired:
    return McpAuthorizationRequired(
        message="Authorization required",
        server_url=SERVER_URL,
    )


def _sharepoint_tool_config(
    toolkit_name: str = "sharepoint",
    site_url: str = SHAREPOINT_URL,
) -> dict:
    return {
        "type": "sharepoint",
        "toolkit_name": toolkit_name,
        "settings": {
            "sharepoint_configuration": {
                "site_url": site_url,
                "oauth_discovery_endpoint": "https://login.example.test",
            },
        },
    }


def _patch_sharepoint_auth_loader(monkeypatch) -> None:
    def load_sharepoint(tool):
        site_url = tool["settings"]["sharepoint_configuration"]["site_url"]
        raise McpAuthorizationRequired(
            message="SharePoint authorization required",
            server_url=site_url,
            resource_metadata={"resource_name": "SharePoint"},
        )

    sharepoint_entry = dict(elitea_tools_mod.AVAILABLE_TOOLS.get("sharepoint") or {})
    sharepoint_entry["get_tools"] = load_sharepoint
    monkeypatch.setitem(elitea_tools_mod.AVAILABLE_TOOLS, "sharepoint", sharepoint_entry)


def test_pipeline_mcp_config_defers_auth_to_checkpointed_node(monkeypatch):
    """A direct pipeline node receives a gateway that raises only when invoked."""

    def get_toolkit(**_kwargs):
        return SimpleNamespace(get_tools=lambda: (_ for _ in ()).throw(_auth_error()))

    monkeypatch.setattr(runtime_tools.McpConfigToolkit, "get_toolkit", get_toolkit)

    toolkit_config = {
        "type": "mcp_config",
        "settings": {"server_name": SERVER_NAME},
    }
    context = McpContext(pipeline_node_toolkit_names={SERVER_NAME})

    tools = runtime_tools.get_tools([toolkit_config], mcp_context=context)
    proxy = next(tool for tool in tools if tool.name.startswith("mcp_authorize_"))

    with pytest.raises(McpAuthorizationRequired):
        proxy.invoke({})


def test_auth_control_resolves_participant_name_through_dynamic_mcp_type(monkeypatch):
    """Conversation toolkit display names must resolve to the configured MCP URL."""

    discovered_urls = []

    monkeypatch.setattr(
        runtime_tools,
        "load_mcp_servers_config",
        lambda: {SERVER_NAME: {"url": SERVER_URL}},
    )

    from elitea_sdk.runtime.utils import mcp_tools_discovery

    def discover_mcp_tools(*, url, **_kwargs):
        discovered_urls.append(url)
        return []

    monkeypatch.setattr(mcp_tools_discovery, "discover_mcp_tools", discover_mcp_tools)

    toolkit_config = {
        "type": f"mcp_{SERVER_NAME}",
        "toolkit_name": TOOLKIT_NAME,
        "settings": {},
    }
    auth_control = runtime_tools._make_mcp_auth_control_tool([toolkit_config])[0]

    result = json.loads(
        auth_control.func(
            action="authorize",
            server_url=TOOLKIT_NAME,
            tool_name="get_lists",
        )
    )

    assert discovered_urls == [SERVER_URL]
    assert result["server_url"] == SERVER_URL


def test_auth_control_uses_delegated_toolkit_auth_url(monkeypatch):
    """A built-in SharePoint alias must re-raise auth with its configured URL."""

    _patch_sharepoint_auth_loader(monkeypatch)
    tools = runtime_tools.get_tools([_sharepoint_tool_config()])
    auth_control = next(tool for tool in tools if tool.name == "mcp_auth_control")

    with pytest.raises(McpAuthorizationRequired) as exc_info:
        auth_control.func(action="authorize", server_url="sharepoint")

    assert exc_info.value.server_url == SHAREPOINT_URL
    assert (exc_info.value.resource_metadata or {}).get("resource_name") == "SharePoint"


def test_direct_sharepoint_toolkit_node_defers_auth_until_execution(monkeypatch):
    """A no-LLM Toolkit node must compile a runtime authorization gateway."""

    _patch_sharepoint_auth_loader(monkeypatch)
    context = McpContext(pipeline_node_toolkit_names={"sharepoint"})

    tools = runtime_tools.get_tools([_sharepoint_tool_config()], mcp_context=context)
    proxy = next(tool for tool in tools if tool.name.startswith("mcp_authorize_"))

    with pytest.raises(McpAuthorizationRequired) as exc_info:
        proxy.invoke({})

    assert exc_info.value.server_url == SHAREPOINT_URL


def test_declined_direct_sharepoint_toolkit_marks_clean_pipeline_stop(monkeypatch):
    """After Skip, a direct Toolkit node must terminate gently without ToolException."""

    _patch_sharepoint_auth_loader(monkeypatch)
    skipped_toolkits = set()
    context = McpContext(
        user_declined_servers=[{"server_url": SHAREPOINT_URL}],
        pipeline_node_toolkit_names={"sharepoint"},
        skipped_pipeline_toolkit_names=skipped_toolkits,
    )

    tools = runtime_tools.get_tools([_sharepoint_tool_config()], mcp_context=context)

    assert skipped_toolkits == {"sharepoint"}
    assert any(tool.name.startswith("mcp_authorize_") for tool in tools)

    schema = yaml.safe_dump(
        {
            "name": "sharepoint-toolkit-skip",
            "state": {
                "messages": {"type": "list"},
                "sharepoint_result": {"type": "str"},
            },
            "nodes": [
                {
                    "id": "Toolkit1",
                    "type": "toolkit",
                    "toolkit_name": "sharepoint",
                    "tool": "get_lists",
                    "output": ["sharepoint_result"],
                    "transition": "END",
                },
            ],
            "entry_point": "Toolkit1",
        },
        default_flow_style=False,
    )
    graph = create_graph(
        client=None,
        yaml_schema=schema,
        tools=tools,
        memory=MemorySaver(),
        skipped_pipeline_toolkit_names=skipped_toolkits,
    )
    result = graph.invoke({"messages": [HumanMessage(content="go")]})

    assert result["execution_finished"] is True
    assert "authentication was skipped" in result["output"]
    assert result["sharepoint_result"] is None


def test_multiple_direct_toolkits_are_all_available_as_runtime_auth_gateways(monkeypatch):
    """Every direct Toolkit can reach its own checkpointed auth boundary."""

    _patch_sharepoint_auth_loader(monkeypatch)
    first_url = "https://tenant.sharepoint.com/sites/first"
    second_url = "https://tenant.sharepoint.com/sites/second"
    tool_configs = [
        _sharepoint_tool_config("sharepoint-first", first_url),
        _sharepoint_tool_config("sharepoint-second", second_url),
    ]
    pipeline_toolkits = {"sharepoint-first", "sharepoint-second"}

    tools = runtime_tools.get_tools(
        tool_configs,
        mcp_context=McpContext(pipeline_node_toolkit_names=pipeline_toolkits),
    )
    proxies = [tool for tool in tools if tool.name.startswith("mcp_authorize_")]
    assert len(proxies) == 2

    with pytest.raises(McpAuthorizationRequired) as first_request:
        proxies[0].invoke({})
    with pytest.raises(McpAuthorizationRequired) as second_request:
        proxies[1].invoke({})
    assert first_request.value.server_url == first_url
    assert second_request.value.server_url == second_url


def test_multiple_agent_toolkits_keep_distinct_auth_urls(monkeypatch):
    """Deferred Agent gateways must retain each delegated toolkit's own URL."""

    _patch_sharepoint_auth_loader(monkeypatch)
    first_url = "https://tenant.sharepoint.com/sites/first"
    second_url = "https://tenant.sharepoint.com/sites/second"
    tools = runtime_tools.get_tools(
        [
            _sharepoint_tool_config("sharepoint-first", first_url),
            _sharepoint_tool_config("sharepoint-second", second_url),
        ]
    )
    auth_control = next(tool for tool in tools if tool.name == "mcp_auth_control")

    with pytest.raises(McpAuthorizationRequired) as first_request:
        auth_control.func(action="authorize", server_url="sharepoint-first")
    with pytest.raises(McpAuthorizationRequired) as second_request:
        auth_control.func(action="authorize", server_url="sharepoint-second")

    assert first_request.value.server_url == first_url
    assert second_request.value.server_url == second_url


def test_pipeline_llm_binds_auth_tools_for_selected_unavailable_toolkit():
    """Pipeline LLM nodes must retain smart-auth tools when real tools cannot load."""

    proxy = StructuredTool.from_function(
        func=lambda: "authorization required",
        name="mcp_authorize_sharepoint",
        description="SharePoint authorization gateway",
        metadata={"toolkit_name": "sharepoint", "toolkit_type": "sharepoint"},
    )
    auth_control = StructuredTool.from_function(
        func=lambda: "authorization required",
        name="mcp_auth_control",
        description="Authorization control",
    )
    normal_tool = StructuredTool.from_function(
        func=lambda: "ok",
        name="search_issues",
        description="Search issues",
        metadata={"toolkit_name": "github", "toolkit_type": "github"},
    )
    schema = yaml.safe_dump(
        {
            "name": "pipeline",
            "state": {"messages": {"type": "list"}},
            "nodes": [
                {
                    "id": "LLM1",
                    "type": "llm",
                    "tool_names": {
                        "sharepoint": ["list_files"],
                        "github": ["search_issues"],
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
        tools=[proxy, auth_control, normal_tool],
        memory=MemorySaver(),
    )
    llm_node = graph.get_graph().nodes["LLM1"].data

    assert {tool.name for tool in llm_node.get_filtered_tools()} == {
        "mcp_authorize_sharepoint",
        "mcp_auth_control",
        "search_issues",
    }


def test_declined_mcp_for_llm_node_returns_declined_proxy(monkeypatch):
    """LLM-driven MCP use remains available as a non-raising declined proxy."""

    monkeypatch.setattr(runtime_tools, "_mcp_tools", lambda *_args, **_kwargs: [])

    toolkit_config = {
        "type": "mcp_config",
        "settings": {"server_name": SERVER_NAME},
    }
    skipped_pipeline_toolkits = set()
    context = McpContext(
        ignored_servers=[SERVER_NAME],
        pipeline_node_toolkit_names=set(),
        skipped_pipeline_toolkit_names=skipped_pipeline_toolkits,
    )

    tools = runtime_tools.get_tools([toolkit_config], mcp_context=context)
    proxy = next(tool for tool in tools if tool.name.startswith("mcp_authorize_"))
    result = json.loads(proxy.func())

    assert result["status"] == "declined"
    assert skipped_pipeline_toolkits == set()


def test_declined_mcp_for_toolkit_node_stops_pipeline_cleanly():
    """A direct Toolkit node must stop gently without executing downstream state."""

    schema = yaml.safe_dump(
        {
            "name": "mcp-auth-skip",
            "state": {
                "messages": {"type": "list"},
                "sharepoint_result": {"type": "str"},
            },
            "nodes": [
                {
                    "id": "Toolkit",
                    "type": "toolkit",
                    "toolkit_name": SERVER_NAME,
                    "tool": "get_lists",
                    "output": ["sharepoint_result"],
                    "transition": "END",
                },
            ],
            "entry_point": "Toolkit",
        },
        default_flow_style=False,
    )

    graph = create_graph(
        client=None,
        yaml_schema=schema,
        tools=[],
        memory=MemorySaver(),
        skipped_pipeline_toolkit_names={SERVER_NAME},
    )
    result = graph.invoke({"messages": [HumanMessage(content="go")]})

    assert result["execution_finished"] is True
    assert "authentication was skipped" in result["output"]
    assert result["sharepoint_result"] is None
