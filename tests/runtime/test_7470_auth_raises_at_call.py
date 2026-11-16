"""Every logged-out toolkit asks for authorization at the call that needs it (#7470)."""
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import MemorySaver

import elitea_sdk.tools as elitea_tools_mod
from elitea_sdk.runtime.langchain.assistant import Assistant
from elitea_sdk.runtime.middleware.tool_exception_handler import pending_authorization
from elitea_sdk.runtime.toolkits import tools as runtime_tools
from elitea_sdk.runtime.toolkits.tools import _build_deferred_mcp_auth_tools
from elitea_sdk.runtime.tools.tool_binding import build_tool_binding_plan
from elitea_sdk.runtime.tools.application import Application
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired, McpContext

ATLASSIAN_URL = "https://mcp.atlassian.com/v1/sse"
ATLASSIAN_TOOLKIT_ID = 77


@pytest.fixture(autouse=True)
def _disable_unrelated_remote_tool_inventory(monkeypatch):
    monkeypatch.setattr(runtime_tools, "_mcp_tools", lambda *_args, **_kwargs: [])


@pytest.fixture
def logged_out_toolkits(monkeypatch):
    def load_sharepoint(tool):
        raise McpAuthorizationRequired(
            "SharePoint authorization required",
            server_url=tool["settings"]["sharepoint_configuration"]["site_url"],
            resource_metadata={"resource_name": "SharePoint"},
        )

    def load_atlassian(**_kwargs):
        raise McpAuthorizationRequired("Authorization required", server_url=ATLASSIAN_URL)

    sharepoint_entry = dict(elitea_tools_mod.AVAILABLE_TOOLS.get("sharepoint") or {})
    sharepoint_entry["get_tools"] = load_sharepoint
    monkeypatch.setitem(elitea_tools_mod.AVAILABLE_TOOLS, "sharepoint", sharepoint_entry)
    monkeypatch.setattr(runtime_tools.McpToolkit, "get_toolkit", load_atlassian)


def _sharepoint(name, site_url):
    return {
        "type": "sharepoint",
        "toolkit_name": name,
        "settings": {"sharepoint_configuration": {"site_url": site_url}},
    }


def _atlassian():
    return {"id": ATLASSIAN_TOOLKIT_ID, "type": "mcp", "toolkit_name": "Atlassian", "settings": {"url": ATLASSIAN_URL}}


def _ticket_toolkits():
    return [
        _sharepoint("Sharepoint", "https://tenant.sharepoint.com/sites/files"),
        _atlassian(),
        _sharepoint("Teams", "https://tenant.sharepoint.com/sites/teams"),
    ]


class _OneTurnLLM:
    temperature = 0
    max_tokens = 1000

    def __init__(self, tool_names):
        self.tool_names = tool_names
        self.invocations = []

    @property
    def _get_model_default_parameters(self):
        return {"temperature": 0, "max_tokens": 1000}

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        self.invocations.append(list(messages))
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="done")
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": {}, "id": f"call-{index}", "type": "tool_call"}
            for index, name in enumerate(self.tool_names)
        ])


def _gateways(tools):
    return [tool for tool in tools if tool.name.startswith("mcp_authorize_")]


def _agent(llm, tools, memory):
    return Assistant(
        elitea=MagicMock(get_mcp_toolkits=lambda: []),
        data={"instructions": "Use tools", "tools": [], "meta": {}},
        client=llm, tools=tools, memory=memory, app_type="predict",
    ).runnable()


def _run_to_first_card(tools, thread_id):
    memory = MemorySaver()
    llm = _OneTurnLLM([tool.name for tool in _gateways(tools)])
    config = {"configurable": {"thread_id": thread_id}}
    result = _agent(llm, tools, memory).invoke({"messages": [HumanMessage(content="run them all")]}, config=config)
    return llm, memory, config, result


def test_every_logged_out_toolkit_in_one_turn_gets_its_own_card(logged_out_toolkits):
    tools = runtime_tools.get_tools(_ticket_toolkits())
    llm, memory, config, result = _run_to_first_card(tools, "issue-7470-ticket")

    cards = []
    while not result["execution_finished"]:
        cards.append(result["hitl_interrupt"]["toolkit_name"])
        result = _agent(llm, tools, memory).invoke({"mcp_auth_resume": True, "mcp_auth_action": "authorize"}, config=config)

    assert sorted(cards) == ["Atlassian", "Sharepoint", "Teams"]
    decisions = [json.loads(m.content)["status"] for m in llm.invocations[-1] if isinstance(m, ToolMessage)]
    assert decisions == ["authorized", "authorized", "authorized"]
    assert len(llm.invocations) == 2


def test_mcp_card_carries_the_mcp_toolkits_own_identity(logged_out_toolkits):
    tools = runtime_tools.get_tools([_sharepoint("Sharepoint", "https://tenant.sharepoint.com/sites/files"), _atlassian()])
    gateway = next(tool for tool in _gateways(tools) if tool.metadata["toolkit_name"] == "Atlassian")

    with pytest.raises(McpAuthorizationRequired) as raised:
        gateway.func()

    assert raised.value.server_url == ATLASSIAN_URL
    assert raised.value.toolkit_id == ATLASSIAN_TOOLKIT_ID


def test_skip_on_the_mcp_card_declines_its_gateway_on_the_next_run(logged_out_toolkits):
    tools = runtime_tools.get_tools([_atlassian()])
    _, _, _, result = _run_to_first_card(tools, "issue-7470-skip")
    declined = [{"server_url": result["hitl_interrupt"]["server_url"]}]

    next_run = runtime_tools.get_tools([_atlassian()], mcp_context=McpContext(user_declined_servers=declined))

    assert json.loads(_gateways(next_run)[0].func())["status"] == "declined"


def test_no_tool_lets_the_model_start_authorization_itself(logged_out_toolkits):
    names = {tool.name for tool in runtime_tools.get_tools(_ticket_toolkits())}

    assert names.isdisjoint({"mcp_auth_control", "request_mcp_authorization"})


class _SwarmLLM:
    def __init__(self, gateway_name):
        self.gateway_name = gateway_name
        self.invocations = []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None, **kwargs):
        self.invocations.append(list(messages))
        if any(isinstance(message, ToolMessage) and "issue #" in str(message.content) for message in messages):
            return AIMessage(content="done")
        if sum(isinstance(message, HumanMessage) for message in messages) == 1:
            calls = [("create_issue", "a"), (self.gateway_name, "b")]
        else:
            calls = [("create_issue", "retry")]
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": {}, "id": call_id, "type": "tool_call"} for name, call_id in calls
        ])


def _swarm_with_issue_tracker(llm, memory):
    created = []

    def create_issue() -> str:
        """Create a GitHub issue."""
        created.append(True)
        return f"issue #{len(created)}"

    child = Application(
        name="Child", description="Child agent", return_type="str", client=None, is_subgraph=True,
        application=MagicMock(invoke=MagicMock(return_value={"output": "child"})),
    )
    gateway = _gateways(runtime_tools.get_tools([_atlassian()]))[0]
    llm.gateway_name = gateway.name
    tools = [child, StructuredTool.from_function(func=create_issue, name="create_issue")]
    return created, lambda authorized=False: Assistant(
        elitea=MagicMock(get_mcp_toolkits=lambda: []),
        data={"instructions": "main", "tools": [], "meta": {"internal_tools": ["swarm"]}, "internal_tools": ["swarm"]},
        client=llm, tools=tools + ([] if authorized else [gateway]), memory=memory, app_type="agent",
    ).runnable()


def test_swarm_continue_resolves_the_deferred_call_instead_of_leaving_it_pending(logged_out_toolkits):
    llm = _SwarmLLM(gateway_name=None)
    _, swarm = _swarm_with_issue_tracker(llm, MemorySaver())
    config = {"configurable": {"thread_id": "issue-7470-swarm-resolve"}}

    with pytest.raises(McpAuthorizationRequired):
        swarm().invoke({"messages": [HumanMessage(content="file it and link Jira")]}, config)
    swarm(authorized=True).invoke({"messages": [HumanMessage(content="file it and link Jira")]}, config)

    model_input = llm.invocations[-1]
    gateway_results = [json.loads(m.content) for m in model_input if isinstance(m, ToolMessage) and m.tool_call_id == "b"]
    assert [result["status"] for result in gateway_results] == ["authorization_resolved"]
    assert "Do not repeat" in gateway_results[0]["next_step"]
    assert [m.text for m in model_input if isinstance(m, HumanMessage)] == ["file it and link Jira"]


def test_swarm_new_request_sent_while_paused_still_reaches_the_model(logged_out_toolkits):
    llm = _SwarmLLM(gateway_name=None)
    _, swarm = _swarm_with_issue_tracker(llm, MemorySaver())
    config = {"configurable": {"thread_id": "issue-7470-swarm-new-request"}}

    with pytest.raises(McpAuthorizationRequired):
        swarm().invoke({"messages": [HumanMessage(content="file it and link Jira")]}, config)
    swarm().invoke({"messages": [HumanMessage(content="also add a label")]}, config)

    assert [m.text for m in llm.invocations[-1] if isinstance(m, HumanMessage)] == [
        "file it and link Jira", "also add a label",
    ]


def test_swarm_batch_keeps_sibling_results_when_authorization_is_deferred(logged_out_toolkits):
    llm = _SwarmLLM(gateway_name=None)
    created, swarm = _swarm_with_issue_tracker(llm, MemorySaver())
    config = {"configurable": {"thread_id": "issue-7470-swarm"}}

    with pytest.raises(McpAuthorizationRequired) as raised:
        swarm().invoke({"messages": [HumanMessage(content="file it and link Jira")]}, config)
    saved = [m.tool_call_id for m in swarm()._graph.get_state(config).values["messages"] if isinstance(m, ToolMessage)]
    swarm(authorized=True).invoke({"messages": [HumanMessage(content="file it and link Jira")]}, config)

    assert raised.value.toolkit_id == ATLASSIAN_TOOLKIT_ID
    assert saved == ["a", "b"]
    assert len(created) == 1


class _HandoffLLM:
    def __init__(self, gateway_name):
        self.gateway_name = gateway_name

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None, **kwargs):
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="done")
        return AIMessage(content="", tool_calls=[
            {"name": "transfer_to_child", "args": {"task": "do it"}, "id": "h", "type": "tool_call"},
            {"name": self.gateway_name, "args": {}, "id": "g", "type": "tool_call"},
        ])


def test_swarm_handoff_batch_stops_at_the_card_before_the_peer_runs(logged_out_toolkits):
    child_app = MagicMock(invoke=MagicMock(return_value={"output": "child"}))
    child = Application(name="Child", description="Child agent", return_type="str", client=None,
                        is_subgraph=True, application=child_app)
    gateway = _gateways(runtime_tools.get_tools([_atlassian()]))[0]
    swarm = Assistant(
        elitea=MagicMock(get_mcp_toolkits=lambda: []),
        data={"instructions": "main", "tools": [], "meta": {"internal_tools": ["swarm"]}, "internal_tools": ["swarm"]},
        client=_HandoffLLM(gateway.name), tools=[child, gateway], memory=MemorySaver(), app_type="agent",
    ).runnable()

    with pytest.raises(McpAuthorizationRequired):
        swarm.invoke({"messages": [HumanMessage(content="hand off and link Jira")]},
                     {"configurable": {"thread_id": "issue-7470-handoff"}})

    assert child_app.invoke.call_count == 0


def test_swarm_batch_with_two_pending_calls_raises_the_first(logged_out_toolkits):
    def pending(call_id, toolkit_name):
        error = McpAuthorizationRequired("Authorization required", server_url=ATLASSIAN_URL, toolkit_name=toolkit_name)
        return ToolMessage(content="", tool_call_id=call_id, artifact={"mcp_auth_pending": error.to_dict()})

    batch = [HumanMessage(content="go"), AIMessage(content=""), pending("a", "Atlassian"), pending("b", "Jira")]

    assert pending_authorization(batch).toolkit_name == "Atlassian"


def _child_agent():
    return Application(
        name="Child", description="Child agent", return_type="str", client=None, is_subgraph=True,
        application=MagicMock(invoke=MagicMock(return_value={"output": "child"})),
    )


class _TwoGatewaySwarmLLM:
    def __init__(self, gateway_names):
        self.gateway_names = gateway_names
        self.invocations = []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None, **kwargs):
        self.invocations.append(list(messages))
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="done")
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": {}, "id": name, "type": "tool_call"} for name in self.gateway_names
        ])


def test_swarm_toolkit_left_undecided_gets_its_own_card_on_continue(logged_out_toolkits):
    gateways = {t.metadata["toolkit_name"]: t for t in _gateways(
        runtime_tools.get_tools([_sharepoint("Sharepoint", "https://tenant.sharepoint.com/sites/files"), _atlassian()]))}
    llm = _TwoGatewaySwarmLLM([gateways["Sharepoint"].name, gateways["Atlassian"].name])
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "issue-7470-swarm-two-toolkits"}}

    child = Application(
        name="Child", description="Child agent", return_type="str", client=None, is_subgraph=True,
        application=MagicMock(invoke=MagicMock(return_value={"output": "child"})),
    )

    def swarm(*offered):
        return Assistant(
            elitea=MagicMock(get_mcp_toolkits=lambda: []),
            data={"instructions": "main", "tools": [], "meta": {"internal_tools": ["swarm"]}, "internal_tools": ["swarm"]},
            client=llm, tools=[child, *offered], memory=memory, app_type="agent",
        ).runnable()

    with pytest.raises(McpAuthorizationRequired) as first:
        swarm(gateways["Sharepoint"], gateways["Atlassian"]).invoke({"messages": [HumanMessage(content="read and link")]}, config)
    with pytest.raises(McpAuthorizationRequired) as second:
        swarm(gateways["Atlassian"]).invoke({"messages": [HumanMessage(content="read and link")]}, config)

    assert [first.value.toolkit_name, second.value.toolkit_name] == ["Sharepoint", "Atlassian"]
    assert len(llm.invocations) == 1


def test_swarm_continue_gives_each_toolkit_its_own_decision(logged_out_toolkits):
    toolkits = [_sharepoint("Sharepoint", "https://tenant.sharepoint.com/sites/files"), _atlassian()]
    gateways = {t.metadata["toolkit_name"]: t for t in _gateways(runtime_tools.get_tools(toolkits))}
    sharepoint_url = "https://tenant.sharepoint.com/sites/files"
    skipped = {t.metadata["toolkit_name"]: t for t in _gateways(runtime_tools.get_tools(
        toolkits, mcp_context=McpContext(user_declined_servers=[{"server_url": sharepoint_url}])))}["Sharepoint"]
    llm = _TwoGatewaySwarmLLM([gateways["Sharepoint"].name, gateways["Atlassian"].name])
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "issue-7470-swarm-two-decisions"}}

    def swarm(*offered):
        return Assistant(
            elitea=MagicMock(get_mcp_toolkits=lambda: []),
            data={"instructions": "main", "tools": [], "meta": {"internal_tools": ["swarm"]}, "internal_tools": ["swarm"]},
            client=llm, tools=[_child_agent(), *offered], memory=memory, app_type="agent",
        ).runnable()

    with pytest.raises(McpAuthorizationRequired):
        swarm(gateways["Sharepoint"], gateways["Atlassian"]).invoke({"messages": [HumanMessage(content="read and link")]}, config)
    swarm(skipped).invoke({"messages": [HumanMessage(content="read and link")]}, config)

    statuses = {m.tool_call_id: json.loads(m.content)["status"] for m in llm.invocations[-1] if isinstance(m, ToolMessage)}
    assert statuses == {gateways["Sharepoint"].name: "declined", gateways["Atlassian"].name: "authorization_resolved"}


class _ExpiredTokenTool(BaseTool):
    """A loaded toolkit tool whose delegated token expired after load."""

    name: str = "sharepoint_get_files_list"
    description: str = "List SharePoint files"
    metadata: dict = {"toolkit_name": "Sharepoint", "toolkit_type": "sharepoint"}
    runs: int = 0

    def _run(self, **kwargs):
        self.runs += 1
        raise McpAuthorizationRequired("Authorization required", server_url="https://tenant.sharepoint.com/sites/files",
                                       toolkit_name="Sharepoint", toolkit_type="sharepoint")


def test_swarm_real_tool_needing_authorization_does_not_break_later_messages(logged_out_toolkits):
    expired = _ExpiredTokenTool()
    llm = _TwoGatewaySwarmLLM([expired.name])
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "issue-7470-swarm-expired-token"}}

    def swarm():
        return Assistant(
            elitea=MagicMock(get_mcp_toolkits=lambda: []),
            data={"instructions": "main", "tools": [], "meta": {"internal_tools": ["swarm"]}, "internal_tools": ["swarm"]},
            client=llm, tools=[_child_agent(), expired], memory=memory, app_type="agent",
        ).runnable()

    with pytest.raises(McpAuthorizationRequired):
        swarm().invoke({"messages": [HumanMessage(content="list files")]}, config)
    swarm().invoke({"messages": [HumanMessage(content="list files")]}, config)
    swarm().invoke({"messages": [HumanMessage(content="something else")]}, config)

    assert expired.runs == 1
    statuses = [json.loads(m.content)["status"] for m in llm.invocations[-1] if isinstance(m, ToolMessage)]
    assert statuses == ["authorization_resolved"]


def test_swarm_gateways_renamed_for_the_model_keep_their_own_decision(logged_out_toolkits):
    def gateway(toolkit_name, url):
        error = McpAuthorizationRequired("Authorization required", server_url=url, toolkit_name=toolkit_name, toolkit_type="mcp")
        return _build_deferred_mcp_auth_tools({"type": "mcp", "toolkit_name": toolkit_name, "settings": {"url": url}}, error)[0]

    first, second = gateway("Jira EPAM", "https://jira-a.example/mcp"), gateway("Jira.EPAM", "https://jira-b.example/mcp")
    child = _child_agent()
    provider_names = [t.name for t in build_tool_binding_plan([first, second]).provider_tools]
    llm = _TwoGatewaySwarmLLM(provider_names)
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "issue-7470-swarm-renamed-gateways"}}

    def swarm(*offered):
        return Assistant(
            elitea=MagicMock(get_mcp_toolkits=lambda: []),
            data={"instructions": "main", "tools": [], "meta": {"internal_tools": ["swarm"]}, "internal_tools": ["swarm"]},
            client=llm, tools=[child, *offered], memory=memory, app_type="agent",
        ).runnable()

    with pytest.raises(McpAuthorizationRequired):
        swarm(first, second).invoke({"messages": [HumanMessage(content="read both")]}, config)
    with pytest.raises(McpAuthorizationRequired) as still_undecided:
        swarm(second).invoke({"messages": [HumanMessage(content="read both")]}, config)

    assert first.name == second.name
    assert still_undecided.value.server_url == "https://jira-b.example/mcp"
