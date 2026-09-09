"""Regression coverage for EL-6546: toolkit_id and provided_settings forwarding.

Contracts tested:
- McpAuthorizationRequired.to_dict() preserves masked provided_settings and toolkit_id.
- LLMNode._build_mcp_auth_interrupt() forwards both fields.
- FunctionNode._build_mcp_auth_interrupt() forwards both fields.

Credential masking belongs to the code that constructs ``provided_settings``;
these serialization layers deliberately preserve the prepared mapping unchanged.
"""
import yaml

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from elitea_sdk.runtime.langchain.langraph_agent import create_graph
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired


_SERVER_URL = "https://api.githubcopilot.com/mcp/"
_TOOLKIT_ID = 314
_MASKED_SETTINGS = {
    "mcp_client_id": "client-abc",
    "mcp_client_secret": "****cret",
}


def _make_exc(toolkit_id=_TOOLKIT_ID, provided_settings=None):
    exc = McpAuthorizationRequired(
        "Authorization required",
        server_url=_SERVER_URL,
        tool_name="list_repos",
        toolkit_name="GitHubCopilot",
        toolkit_type="mcp",
    )
    exc.toolkit_id = toolkit_id
    if provided_settings is not None:
        exc.provided_settings = provided_settings
    return exc


# ---------------------------------------------------------------------------
# McpAuthorizationRequired.to_dict()
# ---------------------------------------------------------------------------

def test_to_dict_includes_toolkit_id():
    d = _make_exc().to_dict()
    assert d["toolkit_id"] == _TOOLKIT_ID


def test_to_dict_includes_masked_provided_settings():
    exc = _make_exc(provided_settings=_MASKED_SETTINGS)
    d = exc.to_dict()
    assert d["provided_settings"] == _MASKED_SETTINGS


def test_to_dict_omits_provided_settings_when_absent():
    d = _make_exc(provided_settings=None).to_dict()
    assert "provided_settings" not in d


# ---------------------------------------------------------------------------
# LLMNode._build_mcp_auth_interrupt() — exercised via a full agent graph
# ---------------------------------------------------------------------------

class _SingleCallLLM:
    temperature = 0
    max_tokens = 1000

    @property
    def _get_model_default_parameters(self):
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, config=None):
        from langchain_core.messages import AIMessage
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "copilot_list_repos",
                "args": {},
                "id": "call-copilot-auth",
                "type": "tool_call",
            }],
        )


class _Runtime:
    def get_mcp_toolkits(self):
        return []


def _make_auth_tool(toolkit_id, provided_settings):
    def raise_auth():
        exc = McpAuthorizationRequired(
            "Authorization required",
            server_url=_SERVER_URL,
            tool_name="list_repos",
            toolkit_name="GitHubCopilot",
            toolkit_type="mcp",
        )
        exc.toolkit_id = toolkit_id
        exc.provided_settings = provided_settings
        raise exc

    return StructuredTool.from_function(
        func=raise_auth,
        name="copilot_list_repos",
        description="List GitHub Copilot repos",
        metadata={
            "tool_name": "list_repos",
            "toolkit_name": "GitHubCopilot",
            "toolkit_type": "mcp",
        },
    )


def _run_agent_to_interrupt(toolkit_id=_TOOLKIT_ID, provided_settings=None):
    from elitea_sdk.runtime.langchain.assistant import Assistant

    auth_tool = _make_auth_tool(toolkit_id, provided_settings or _MASKED_SETTINGS)
    memory = MemorySaver()
    runnable = Assistant(
        elitea=_Runtime(),
        data={"instructions": "Use tools", "tools": [], "meta": {}},
        client=_SingleCallLLM(),
        tools=[auth_tool],
        memory=memory,
        app_type="predict",
    ).runnable()
    return runnable.invoke(
        {"messages": [HumanMessage(content="List repos")]},
        config={"configurable": {"thread_id": "el-6546-llm"}},
    )


def test_llm_interrupt_builder_forwards_toolkit_id():
    result = _run_agent_to_interrupt(toolkit_id=_TOOLKIT_ID)
    assert result["execution_finished"] is False
    interrupt = result["hitl_interrupt"]
    assert interrupt["guardrail_type"] == "mcp_auth"
    assert interrupt["toolkit_id"] == _TOOLKIT_ID


def test_llm_interrupt_builder_forwards_provided_settings():
    result = _run_agent_to_interrupt(provided_settings=_MASKED_SETTINGS)
    interrupt = result["hitl_interrupt"]
    assert interrupt["provided_settings"] == _MASKED_SETTINGS


def test_llm_interrupt_omits_toolkit_id_when_absent():
    """When no toolkit_id is set on the exception, the key must be absent."""
    auth_tool = _make_auth_tool(toolkit_id=None, provided_settings=None)

    def raise_auth_no_id():
        exc = McpAuthorizationRequired(
            "Authorization required",
            server_url=_SERVER_URL,
            tool_name="list_repos",
            toolkit_name="GitHubCopilot",
            toolkit_type="mcp",
        )
        raise exc

    bare_tool = StructuredTool.from_function(
        func=raise_auth_no_id,
        name="copilot_list_repos",
        description="List GitHub Copilot repos",
        metadata={
            "tool_name": "list_repos",
            "toolkit_name": "GitHubCopilot",
            "toolkit_type": "mcp",
        },
    )
    from elitea_sdk.runtime.langchain.assistant import Assistant
    memory = MemorySaver()
    runnable = Assistant(
        elitea=_Runtime(),
        data={"instructions": "Use tools", "tools": [], "meta": {}},
        client=_SingleCallLLM(),
        tools=[bare_tool],
        memory=memory,
        app_type="predict",
    ).runnable()
    result = runnable.invoke(
        {"messages": [HumanMessage(content="List repos")]},
        config={"configurable": {"thread_id": "el-6546-llm-no-id"}},
    )
    assert result["execution_finished"] is False
    interrupt = result["hitl_interrupt"]
    assert "toolkit_id" not in interrupt or interrupt.get("toolkit_id") is None


# ---------------------------------------------------------------------------
# FunctionNode._build_mcp_auth_interrupt() — exercised via a pipeline graph
# ---------------------------------------------------------------------------

def _pipeline_schema():
    return yaml.safe_dump({
        "name": "el-6546-pipeline",
        "state": {
            "messages": {"type": "list"},
            "result": {"type": "str"},
        },
        "nodes": [{
            "id": "CopilotNode",
            "type": "toolkit",
            "toolkit_name": "GitHubCopilot",
            "tool": "list_repos",
            "output": ["result"],
            "transition": "END",
        }],
        "entry_point": "CopilotNode",
    })


def _run_pipeline_to_interrupt(toolkit_id=_TOOLKIT_ID, provided_settings=None):
    def raise_auth(arguments=None):
        exc = McpAuthorizationRequired(
            "Authorization required",
            server_url=_SERVER_URL,
            tool_name="list_repos",
            toolkit_name="GitHubCopilot",
            toolkit_type="mcp",
        )
        exc.toolkit_id = toolkit_id
        if provided_settings is not None:
            exc.provided_settings = provided_settings
        raise exc

    proxy = StructuredTool.from_function(
        func=raise_auth,
        name="mcp_authorize_GitHubCopilot",
        description="GitHubCopilot authorization gateway",
        metadata={
            "tool_name": "mcp_authorize_GitHubCopilot",
            "toolkit_name": "GitHubCopilot",
            "toolkit_type": "mcp",
        },
    )
    memory = MemorySaver()
    graph = create_graph(
        client=None,
        yaml_schema=_pipeline_schema(),
        tools=[proxy],
        memory=memory,
    )
    return graph.invoke(
        {"messages": [HumanMessage(content="run")]},
        config={"configurable": {"thread_id": "el-6546-fn"}},
    )


def test_function_interrupt_builder_forwards_toolkit_id():
    result = _run_pipeline_to_interrupt(toolkit_id=_TOOLKIT_ID)
    assert result["execution_finished"] is False
    interrupt = result["hitl_interrupt"]
    assert interrupt["guardrail_type"] == "mcp_auth"
    assert interrupt["toolkit_id"] == _TOOLKIT_ID


def test_function_interrupt_builder_forwards_provided_settings():
    result = _run_pipeline_to_interrupt(provided_settings=_MASKED_SETTINGS)
    interrupt = result["hitl_interrupt"]
    assert interrupt["provided_settings"] == _MASKED_SETTINGS


def test_function_interrupt_omits_toolkit_id_when_absent():
    def raise_auth_no_id(arguments=None):
        exc = McpAuthorizationRequired(
            "Authorization required",
            server_url=_SERVER_URL,
            tool_name="list_repos",
            toolkit_name="GitHubCopilot",
            toolkit_type="mcp",
        )
        raise exc

    proxy = StructuredTool.from_function(
        func=raise_auth_no_id,
        name="mcp_authorize_GitHubCopilot",
        description="GitHubCopilot authorization gateway",
        metadata={
            "tool_name": "mcp_authorize_GitHubCopilot",
            "toolkit_name": "GitHubCopilot",
            "toolkit_type": "mcp",
        },
    )
    memory = MemorySaver()
    graph = create_graph(
        client=None,
        yaml_schema=_pipeline_schema(),
        tools=[proxy],
        memory=memory,
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="run")]},
        config={"configurable": {"thread_id": "el-6546-fn-no-id"}},
    )
    assert result["execution_finished"] is False
    interrupt = result["hitl_interrupt"]
    assert "toolkit_id" not in interrupt or interrupt.get("toolkit_id") is None
