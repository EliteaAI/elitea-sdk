"""Code Mode tools and agent wiring (#7579).

Dispatcher, stub and report tests use fake BaseTools; execute_code runs against a small Python script that
speaks the bridge protocol (as in test_7578_sandbox_bridge.py) and runs the program's calls in plain Python,
so the whole path works without Deno.
"""
import asyncio
import json
import sys
import textwrap
from typing import List, Optional

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import StructuredTool, ToolException
from langgraph.errors import GraphInterrupt
from pydantic import BaseModel, Field

from elitea_sdk.runtime.exceptions import BudgetExceededError, InputGuardrailBlocked
from elitea_sdk.runtime.langchain.sandbox_types import BridgeExecutionResult
from elitea_sdk.runtime.toolkits import security
from elitea_sdk.runtime.tools import code_mode
from elitea_sdk.runtime.tools.code_mode import (
    BridgeDispatcher, CodeModeStubs, ExecuteCodeTool, GetToolApiTool, build_run_report,
    code_mode_eligible_tools, create_code_mode_tools, generate_code_mode_index, read_code_mode_limits,
)
from elitea_sdk.runtime.tools.lazy_tools import ToolRegistry
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired


class IssuesArgs(BaseModel):
    repo: str = Field(description="owner/name")
    state: str = Field("open", description="Issue state", json_schema_extra={"enum": ["open", "closed"]})
    per_page: Optional[int] = None


class WriteArgs(BaseModel):
    filename: str
    filedata: str


def make_tool(name, toolkit, toolkit_type, func, schema=None, description="does things"):
    return StructuredTool.from_function(
        func=func, name=name, description=description, args_schema=schema,
        metadata={"toolkit_name": toolkit, "toolkit_type": toolkit_type, "tool_name": name},
    )


def github_tools(toolkit="gh-main"):
    return [
        make_tool("get_issues", toolkit, "github",
                  lambda repo, state="open", per_page=None: [{"number": 1, "state": state, "repo": repo}],
                  IssuesArgs, "List issues of a repository."),
        make_tool("create_file", toolkit, "github", lambda filename, filedata: "created", WriteArgs),
    ]


def dispatcher_for(tools, limits=None, guard=None, config=None):
    stubs = CodeModeStubs(ToolRegistry.from_tools(tools))
    return BridgeDispatcher(stubs, config, {**read_code_mode_limits(), **(limits or {})}, guard)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def clean_security():
    security.reset_blocklist()
    security.reset_sensitive_tools()
    yield
    security.reset_blocklist()
    security.reset_sensitive_tools()


# --- stubs ---------------------------------------------------------------------------------------------------

def test_stub_signature_types_defaults_and_enum():
    stub = CodeModeStubs(ToolRegistry.from_tools(github_tools())).all_stubs()
    assert "class gh_main:" in stub
    # Required args come first, defaults are kept, optional nulls are typed.
    assert "async def get_issues(*, repo: str, state: str = 'open', per_page: int | None = None) -> Any:" in stub
    assert "One of ['open', 'closed']" in stub
    assert 'return await _call("gh_main.get_issues"' in stub
    compile(code_mode.PREAMBLE + stub, "<stubs>", "exec")


def test_two_instances_of_one_toolkit_type_stay_separate():
    stubs = CodeModeStubs(ToolRegistry.from_tools(github_tools("gh-main") + github_tools("gh private")))
    assert set(stubs.class_names.values()) == {"gh_main", "gh_private"}
    assert stubs.bridge_names["gh_private.get_issues"] == ("gh private", "get_issues")


def test_indexer_tools_are_left_out():
    index_tool = make_tool("index_data", "gh-main", "github", lambda: "x")
    names = [t.name for t in code_mode_eligible_tools(github_tools() + [index_tool])]
    assert names == ["get_issues", "create_file"]


def test_inline_stubs_for_small_agents_get_tool_api_for_large(monkeypatch):
    small = create_code_mode_tools(ToolRegistry.from_tools(github_tools()))
    assert [t.name for t in small] == ["execute_code"]
    assert "async def get_issues" in small[0].description

    monkeypatch.setenv("CODE_MODE_INLINE_STUBS_MAX", "1")
    large = create_code_mode_tools(ToolRegistry.from_tools(github_tools()))
    assert [t.name for t in large] == ["execute_code", "get_tool_api"]
    assert "async def get_issues" not in large[0].description
    assert "get_tool_api" in generate_code_mode_index(ToolRegistry.from_tools(github_tools()))


def test_get_tool_api_returns_selected_stubs_and_names_unknown():
    api = GetToolApiTool(stubs=CodeModeStubs(ToolRegistry.from_tools(github_tools())))
    out = api.invoke({"toolkit": "gh_main", "tools": ["get_issues", "nope"]})
    assert "async def get_issues" in out and "async def create_file" not in out
    assert "Not found in gh_main: nope" in out
    assert "not found" in api.invoke({"toolkit": "jira"})


def test_index_lists_toolkits_and_tool_names_only():
    index = generate_code_mode_index(ToolRegistry.from_tools(github_tools()))
    assert "gh_main" in index and "get_issues, create_file" in index
    assert "async def" not in index


# --- dispatcher ----------------------------------------------------------------------------------------------

def test_dispatch_round_trip_and_null_args_stripped():
    d = dispatcher_for(github_tools())
    value = run(d("gh_main.get_issues", json.dumps({"repo": "a/b", "per_page": None})))
    assert json.loads(value) == [{"number": 1, "state": "open", "repo": "a/b"}]
    assert d.call_log[0]["ok"] and d.call_log[0]["bytes"] > 0


def test_allowlist_rejects_unknown_tool_and_lists_valid_names():
    d = dispatcher_for(github_tools())
    with pytest.raises(RuntimeError, match="Unknown tool 'jira.search'.*gh_main.get_issues"):
        run(d("jira.search", "{}"))
    assert d.call_log[0]["error_class"] == "tool_error"


def test_blocklist_applies():
    security.configure_blocklist(blocked_tools={"github": ["create_file"]})
    d = dispatcher_for(github_tools())
    with pytest.raises(RuntimeError, match="blocked by security policy"):
        run(d("gh_main.create_file", json.dumps({"filename": "a", "filedata": "b"})))


def test_argument_validation_against_args_schema():
    d = dispatcher_for(github_tools())
    with pytest.raises(RuntimeError, match="Invalid arguments for gh_main.get_issues: repo: Field required"):
        run(d("gh_main.get_issues", "{}"))


def test_oversized_result_is_refused_before_the_sandbox():
    big = make_tool("dump", "art", "artifact", lambda: "x" * 5000)
    d = dispatcher_for([big], limits={"max_result_bytes": 1000})
    with pytest.raises(RuntimeError, match=r"art.dump result was (over )?5 KB, limit 1 KB.*fewer items"):
        run(d("art.dump", "{}"))
    assert d.call_log[0]["error_class"] == "result_too_large"


def test_sensitive_tool_needing_approval_is_refused():
    from elitea_sdk.runtime.middleware.sensitive_tool_guard import SensitiveToolGuardMiddleware
    security.configure_sensitive_tools({"github": ["create_file"]})
    d = dispatcher_for(github_tools(), guard=SensitiveToolGuardMiddleware())
    with pytest.raises(RuntimeError, match="needs human approval"):
        run(d("gh_main.create_file", json.dumps({"filename": "a", "filedata": "b"})))
    assert d.call_log[0]["error_class"] == "approval_required"


def test_sensitive_tool_auto_approved_by_the_guard_runs():
    from elitea_sdk.runtime.middleware.sensitive_tool_guard import SensitiveToolGuardMiddleware
    security.configure_sensitive_tools({"github": ["create_file"]})
    d = dispatcher_for(github_tools(), guard=SensitiveToolGuardMiddleware(auto_approve=True))
    assert json.loads(run(d("gh_main.create_file", json.dumps({"filename": "a", "filedata": "b"})))) == "created"


def test_tool_error_becomes_a_catchable_error():
    def boom(**_):
        raise ToolException("rate limited")
    d = dispatcher_for([make_tool("get", "gh", "github", boom)])
    with pytest.raises(RuntimeError, match="gh.get failed: rate limited"):
        run(d("gh.get", "{}"))
    assert d.abort is None


@pytest.mark.parametrize("exc", [
    GraphInterrupt(()),
    McpAuthorizationRequired("login", server_url="https://mcp"),
    InputGuardrailBlocked(categories=["secret"]),
    BudgetExceededError("budget gone"),
])
def test_control_flow_errors_abort_the_run(exc):
    def raiser(**_):
        raise exc
    d = dispatcher_for([make_tool("get", "gh", "github", raiser)])
    with pytest.raises(RuntimeError, match="stopping"):
        run(d("gh.get", "{}"))
    assert d.abort is exc
    # Later calls are refused so the program can't keep working after the stop.
    with pytest.raises(RuntimeError, match="stopping"):
        run(d("gh.get", "{}"))


def test_bridged_calls_run_under_the_given_callbacks():
    seen = []

    class Recorder(BaseCallbackHandler):
        def on_tool_start(self, serialized, input_str, **kwargs):
            seen.append(serialized["name"])

    d = dispatcher_for(github_tools(), config={"callbacks": [Recorder()]})
    run(d("gh_main.get_issues", json.dumps({"repo": "a/b"})))
    assert seen == ["get_issues"]


# --- run report (error shape) --------------------------------------------------------------------------------

def result(status="error", error_type=None, error=None, stdout="", value=None):
    return BridgeExecutionResult(status=status, execution_time=1.0, result=value, stdout=stdout,
                                 error=error, error_type=error_type)


CALLS = [{"tool": "gh.create_file", "ok": True, "ms": 300, "bytes": 10},
         {"tool": "gh.get", "ok": False, "ms": 5, "bytes": 0, "error_class": "tool_error", "error": "x"}]


def test_success_report_caps_output():
    report = build_run_report(result("success", stdout="y" * 100), CALLS, 40)
    assert report["status"] == "success"
    assert "output truncated" in report["output"]
    assert report["calls"][0].startswith("gh.create_file ok")


@pytest.mark.parametrize("error_type", ["timeout", "call_limit", "sandbox_crashed", "out_of_memory"])
def test_error_shape_for_each_class(error_type):
    report = build_run_report(result(error_type=error_type, error="boom", stdout="partial"), CALLS, 1000)
    assert report["error_class"] == error_type
    assert report["hint"] == code_mode._HINTS[error_type]
    assert report["output_before_failure"] == "partial"
    assert report["calls_completed"][0].startswith("gh.create_file ok")
    assert "Do not repeat the writes" in report["calls_completed"][-1]


def test_python_memory_error_is_out_of_memory():
    report = build_run_report(result(error_type="python_exception", error="Traceback...\nMemoryError"), [], 1000)
    assert report["error_class"] == "out_of_memory"


def test_uncaught_refusal_is_reported_as_result_too_large():
    refused = [{"tool": "a.dump", "ok": False, "ms": 1, "bytes": 5000, "error_class": "result_too_large",
                "error": "a.dump result was 5 KB, limit 1 KB."}]
    report = build_run_report(result(error_type="python_exception", error="ToolError: ..."), refused, 1000)
    assert report["error_class"] == "result_too_large"
    assert "5 KB" in report["message"]
    assert report["calls_completed"] == []


# --- execute_code end to end against a fake sandbox ----------------------------------------------------------

# Runs the program in plain CPython with a fake elitea_bridge module that talks the real line protocol.
FAKE_SANDBOX = textwrap.dedent(r'''
    import asyncio, contextlib, io, json, sys, types, traceback

    def send(message):
        sys.__stdout__.write(json.dumps(message) + "\n"); sys.__stdout__.flush()

    class Out(io.TextIOBase):
        def write(self, text):
            send({"type": "stdout", "text": text}); return len(text)

    counter = [0]
    async def call(name, args):
        counter[0] += 1
        send({"type": "call", "id": counter[0], "name": name, "args": args})
        reply = json.loads(sys.stdin.readline())
        if not reply["ok"]:
            raise Exception(reply["error"])
        return reply["value"]

    sys.modules["elitea_bridge"] = types.SimpleNamespace(call=call)
    code = json.loads(sys.stdin.readline())["code"]
    async def main():
        scope = {}
        with contextlib.redirect_stdout(Out()):
            exec(compile("async def __program():\n" + "\n".join("    " + l for l in code.splitlines()),
                         "<program>", "exec"), scope)
            await scope["__program"]()
    try:
        asyncio.run(main())
        send({"type": "final", "success": True, "result": None, "error": None, "error_type": None})
    except BaseException:
        error = traceback.format_exc()
        kind = "out_of_memory" if "MemoryError" in error else "python_exception"
        send({"type": "final", "success": False, "result": None, "error": error, "error_type": kind})
''')


@pytest.fixture
def fake_sandbox(tmp_path, monkeypatch):
    from elitea_sdk.runtime.langchain.pyodide_sandbox import PyodideSandbox
    from elitea_sdk.runtime.tools import sandbox as sandbox_module
    script = tmp_path / "fake_sandbox.py"
    script.write_text(FAKE_SANDBOX)

    def build():
        box = PyodideSandbox(pkg_name="unused.js", skip_deno_check=True)
        box._build_command = lambda **_kwargs: [sys.executable, str(script)]
        return box

    monkeypatch.setattr(sandbox_module, "build_offline_pyodide_sandbox", build)
    monkeypatch.setattr(sandbox_module, "enforce_sandbox_admission", lambda limits: None)


def execute(tools, code, **limits):
    tool = ExecuteCodeTool(stubs=CodeModeStubs(ToolRegistry.from_tools(tools)),
                           limits={**read_code_mode_limits(), **limits})
    return json.loads(tool.invoke({"code": code}))


def test_execute_code_runs_a_multi_call_program(fake_sandbox):
    writes = []
    tools = github_tools() + [make_tool("append", "art", "artifact",
                                        lambda filename, filedata: writes.append(filedata) or "ok", WriteArgs)]
    report = execute(tools, textwrap.dedent('''
        issues = await gh_main.get_issues(repo="a/b", state="closed")
        for i in issues:
            await art.append(filename="r.md", filedata=f"#{i['number']} {i['state']}")
        print(len(issues))
    '''))
    assert report["status"] == "success", report
    assert report["output"].strip() == "1"
    assert writes == ["#1 closed"]
    assert len(report["calls"]) == 2


def test_execute_code_reports_refused_result_then_rewrite_succeeds(fake_sandbox):
    big = make_tool("list_all", "art", "artifact", lambda limit=1000: ["x" * 100] * limit)
    failed = execute([big], "items = await art.list_all()\nprint(len(items))", max_result_bytes=10_000)
    assert failed["status"] == "error"
    assert failed["error_class"] == "result_too_large"
    assert "fewer items" in failed["hint"]

    # The rewritten program asks for less and succeeds.
    ok = execute([big], "items = await art.list_all(limit=10)\nprint(len(items))", max_result_bytes=10_000)
    assert ok["status"] == "success" and ok["output"].strip() == "10"


def test_execute_code_python_exception_keeps_output_and_completed_calls(fake_sandbox):
    report = execute(github_tools(), textwrap.dedent('''
        await gh_main.create_file(filename="a", filedata="b")
        print("wrote")
        1 / 0
    '''))
    assert report["error_class"] == "python_exception"
    assert "ZeroDivisionError" in report["message"]
    assert report["output_before_failure"].strip() == "wrote"
    assert report["calls_completed"][0].startswith("gh_main.create_file ok")


def test_execute_code_call_limit(fake_sandbox):
    report = execute(github_tools(), "for _ in range(5):\n    await gh_main.get_issues(repo='a/b')",
                     max_tool_calls=2)
    assert report["error_class"] == "call_limit"


def test_execute_code_propagates_interrupt(fake_sandbox):
    def ask(**_):
        raise GraphInterrupt(())
    with pytest.raises(GraphInterrupt):
        execute([make_tool("ask", "gh", "github", ask)], "await gh.ask()")


# --- wiring --------------------------------------------------------------------------------------------------

def llm_node(tools, **kwargs):
    from elitea_sdk.runtime.tools.llm import LLMNode
    return LLMNode(client=None, input_mapping={}, name="agent", available_tools=tools, **kwargs)


def test_llm_node_code_mode_binds_code_mode_tools_plus_always_bind():
    planner = make_tool("update_plan", "planner", "internal", lambda: "ok")
    node = llm_node(github_tools(), code_mode=True, lazy_tools_mode=False,
                    tool_registry=ToolRegistry.from_tools(github_tools()), always_bind_tools=[planner])
    assert [t.name for t in node.get_filtered_tools()] == ["execute_code", "update_plan"]
    assert "## Code Mode" in node.get_tool_index()


def test_llm_node_without_code_mode_is_unchanged():
    node = llm_node(github_tools(), code_mode=False, lazy_tools_mode=False)
    assert [t.name for t in node.get_filtered_tools()] == ["get_issues", "create_file"]


def test_create_graph_without_deno_ignores_code_mode(monkeypatch):
    from elitea_sdk.runtime.langchain import langraph_agent
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert langraph_agent._build_code_mode_registry(github_tools(), set()) == (False, None)


def test_create_graph_registry_excludes_always_bind_and_indexer_tools(monkeypatch):
    from elitea_sdk.runtime.langchain import langraph_agent
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/deno")
    tools = github_tools() + [make_tool("index_data", "gh-main", "github", lambda: "x")]
    enabled, registry = langraph_agent._build_code_mode_registry(tools, {"create_file"})
    assert enabled
    assert list(registry.get_toolkit_tools("gh-main")) == ["get_issues"]


# --- fewer turns: no imports, shapes, signatures (#7579 follow-up) -------------------------------------------

def test_toolkit_imports_are_stripped_other_imports_kept():
    stubs = CodeModeStubs(ToolRegistry.from_tools(github_tools()))
    code = "from gh_main import gh_main\nimport gh_main as g\nimport json\nx = 1"
    assert stubs.strip_toolkit_imports(code).split("\n") == ["", "", "import json", "x = 1"]


def test_value_shape_describes_structure():
    value = {"total": 2, "rows": [{"name": "a", "size": 1}], "truncated": False}
    assert code_mode.value_shape(value) == "{total:int, rows:[{name:str, size:int}] x1, truncated:bool}"
    assert code_mode.value_shape('[{"n": 1}]') == "str(JSON [{n:int}] x1)"
    assert code_mode.value_shape("plain") == "str(5 chars)"


def test_calls_report_includes_result_shape():
    d = dispatcher_for(github_tools())
    run(d("gh_main.get_issues", json.dumps({"repo": "a/b"})))
    line = code_mode._format_calls(d.call_log)[0]
    assert line.endswith("-> [{number:int, state:str, repo:str}] x1")


def test_description_has_example_and_signatures_when_stubs_not_inline(monkeypatch):
    monkeypatch.setenv("CODE_MODE_INLINE_STUBS_MAX", "1")
    execute = create_code_mode_tools(ToolRegistry.from_tools(github_tools()))[0]
    assert "result = await gh_main.get_issues(...)" in execute.description
    assert "Do not import them" in execute.description
    assert "await gh_main.get_issues(repo: str, state: str=..., per_page: int | None=...)" in execute.description
    assert "await gh_main.create_file(filename: str, filedata: str)" in execute.description


def test_execute_code_survives_a_toolkit_import(fake_sandbox):
    report = execute(github_tools(), "from gh_main import gh_main\nprint(len(await gh_main.get_issues(repo='a/b')))")
    assert report["status"] == "success", report
    assert report["output"].strip() == "1"


def test_code_mode_graph_skips_dynamic_toolkit_selection(monkeypatch):
    """Live bug: the runnable's dynamic selection pre-selected toolkit tools, so they were bound instead of execute_code."""
    import yaml
    from langgraph.checkpoint.memory import MemorySaver
    from elitea_sdk.runtime.langchain.langraph_agent import create_graph
    from elitea_sdk.runtime.langchain.constants import DEAULT_AGENT_NAME
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/deno")
    schema = yaml.safe_dump({
        "name": DEAULT_AGENT_NAME, "state": {"messages": {"type": "list"}},
        "nodes": [{"id": "agent", "type": "llm", "transition": "END"}], "entry_point": "agent",
    })
    graph = create_graph(client=None, yaml_schema=schema, tools=github_tools(), memory=MemorySaver(), code_mode=True)
    assert graph.tool_registry is None
    node = graph.get_graph().nodes["agent"].data
    assert [t.name for t in node.get_filtered_tools()] == ["execute_code"]


def _internal(name, func):
    tool = StructuredTool.from_function(func=func, name=name, description=name)
    tool.metadata = {"toolkit_type": "internal", "toolkit_name": name}
    return tool


def test_ask_user_and_sandbox_tools_stay_bound_directly(monkeypatch):
    import yaml
    from langgraph.checkpoint.memory import MemorySaver
    from elitea_sdk.runtime.langchain.langraph_agent import create_graph
    from elitea_sdk.runtime.langchain.constants import DEAULT_AGENT_NAME
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/deno")
    ask = _internal("ask_user", lambda question: "answer")
    sandbox = _internal("pyodide", lambda code: "ran")
    schema = yaml.safe_dump({
        "name": DEAULT_AGENT_NAME, "state": {"messages": {"type": "list"}},
        "nodes": [{"id": "agent", "type": "llm", "transition": "END"}], "entry_point": "agent",
    })
    graph = create_graph(client=None, yaml_schema=schema, tools=github_tools() + [ask, sandbox],
                         memory=MemorySaver(), code_mode=True)
    node = graph.get_graph().nodes["agent"].data
    assert [t.name for t in node.get_filtered_tools()] == ["execute_code", "ask_user", "pyodide"]
    assert "ask_user" not in node.tool_registry.get_toolkit_names()
    assert "ask_user" not in node.get_filtered_tools()[0].description


def test_swarm_with_code_mode_warns_and_builds_swarm(caplog):
    from elitea_sdk.runtime.langchain.assistant import Assistant
    assistant = object.__new__(Assistant)
    child = make_tool("child_agent", "child", "application", lambda task: "done")
    assistant.tools = github_tools() + [child]
    assistant.swarm_mode, assistant.code_mode = True, True
    assistant._is_agent_tool = lambda t: t is child
    assistant._create_swarm_agent = lambda tools, agents: "swarm-graph"
    with caplog.at_level("WARNING"):
        assert assistant.getLangGraphReactAgent() == "swarm-graph"
    assert "[CodeMode] Ignored: swarm mode" in caplog.text


# --- predictable results: returned errors, JSON text, declared return types ---------------------------------

@pytest.mark.parametrize("result", [
    "Failed to get issue: 410 This issue was deleted",
    "Error: repository not found",
    "Unable to read file",
    {"error": "rate limited"},
    {"status": "error", "message": "boom"},
])
def test_returned_error_becomes_tool_error(result):
    d = dispatcher_for([make_tool("get", "gh", "github", lambda: result)])
    with pytest.raises(RuntimeError, match="gh.get failed"):
        run(d("gh.get", "{}"))
    assert d.call_log[0]["error_class"] == "tool_error"


@pytest.mark.parametrize("result", ["Errors found: 0", {"number": 1, "error": None}, "No issues found."])
def test_normal_results_are_not_errors(result):
    assert code_mode.returned_error(result) is None


def test_json_text_reaches_the_program_as_data():
    d = dispatcher_for([make_tool("get", "gh", "github", lambda: '[{"number": 7}]')])
    assert json.loads(run(d("gh.get", "{}"))) == [{"number": 7}]
    assert code_mode.parse_json_text("not json [") == "not json ["


class _Wrapper(BaseModel):
    def get_available_tools(self):
        return [{"name": "get_issue", "ref": self.get_issue}, {"name": "list_prs", "ref": self.list_prs}]

    def get_issue(self, issue_number: int) -> "Union[Dict[str, Any], str]":
        return {}

    def list_prs(self) -> str:
        return "[]"


def test_stub_return_type_comes_from_the_wrapper_annotation():
    from typing import Any, Dict, Union  # noqa: F401 - resolved by get_type_hints
    from elitea_sdk.runtime.tools.code_mode import return_hint
    issue, prs = (make_tool(n, "gh", "github", lambda **_: {}) for n in ("get_issue", "list_prs"))
    for t in (issue, prs):
        object.__setattr__(t, "api_wrapper", _Wrapper())
    globals().update(Union=Union, Dict=Dict, Any=Any)
    assert return_hint(issue) == "dict | str"
    # A bare `-> str` is not trusted: wrappers often return lists under that annotation.
    assert return_hint(prs) is None
    stub = CodeModeStubs(ToolRegistry.from_tools([issue, prs])).all_stubs()
    assert "async def get_issue(*, issue_number: Any) -> dict | str:" in stub or "-> dict | str:" in stub
    assert "Returns: shape not declared" in stub


# --- review fixes ---------------------------------------------------------------------------------------------

def test_backslashes_in_descriptions_keep_stubs_valid():
    class PathArgs(BaseModel):
        path: str = Field(description=r"Like C:\Users\x or \N{bad}, ends with \ ")
    tool = make_tool("read", "fs", "fs", lambda path: path, PathArgs, description=r"Reads C:\Users\x \u12")
    compile(code_mode.PREAMBLE + CodeModeStubs(ToolRegistry.from_tools([tool])).all_stubs(), "<stubs>", "exec")


def test_list_valued_schema_type():
    assert code_mode._hint({"type": ["string", "null"]}) == "str | None"
    assert code_mode._hint({"type": []}) == "Any"


def test_colliding_tool_names_get_distinct_methods():
    tools = [make_tool("get-x", "tk", "t", lambda: "dash"), make_tool("get_x", "tk", "t", lambda: "under"),
             make_tool("ping", "ToolError", "t", lambda: "pong")]
    stubs = CodeModeStubs(ToolRegistry.from_tools(tools))
    assert stubs.bridge_names["tk.get_x"] != stubs.bridge_names["tk.get_x_"]
    assert stubs.class_names["ToolError"] == "ToolError_"
    compile(code_mode.PREAMBLE + stubs.all_stubs(), "<stubs>", "exec")


@pytest.mark.parametrize("text", ["Failed login attempts: 12", "Exception handling guide", "Error budget is 1%"])
def test_text_that_merely_starts_like_an_error_is_data(text):
    assert code_mode.returned_error(text) is None


def test_no_guard_means_sensitive_tool_runs_as_it_would_directly():
    security.configure_sensitive_tools({"github": ["create_file"]})
    d = dispatcher_for(github_tools())
    assert json.loads(run(d("gh_main.create_file", json.dumps({"filename": "a", "filedata": "b"})))) == "created"


def test_stubs_are_built_once_per_registry():
    registry = ToolRegistry.from_tools(github_tools())
    assert code_mode.stubs_for(registry) is code_mode.stubs_for(registry)


def test_sync_invoke_keeps_bridged_calls_under_the_callbacks(fake_sandbox):
    seen = []

    class Recorder(BaseCallbackHandler):
        def on_tool_start(self, serialized, input_str, **kwargs):
            seen.append(serialized["name"])

    tool = ExecuteCodeTool(stubs=CodeModeStubs(ToolRegistry.from_tools(github_tools())))
    tool.invoke({"code": "await gh_main.get_issues(repo='a/b')"}, config={"callbacks": [Recorder()]})
    assert seen == ["execute_code", "get_issues"]


def test_lazy_mode_survives_when_code_mode_cannot_start(monkeypatch):
    import yaml
    from langgraph.checkpoint.memory import MemorySaver
    from elitea_sdk.runtime.langchain.langraph_agent import create_graph
    from elitea_sdk.runtime.langchain.constants import DEAULT_AGENT_NAME
    monkeypatch.setattr("shutil.which", lambda _name: None)
    tools = [make_tool(f"t{i}", f"tk{i % 3}", "github", lambda: "x") for i in range(25)]
    schema = yaml.safe_dump({
        "name": DEAULT_AGENT_NAME, "state": {"messages": {"type": "list"}},
        "nodes": [{"id": "agent", "type": "llm", "transition": "END"}], "entry_point": "agent",
    })
    graph = create_graph(client=None, yaml_schema=schema, tools=tools, memory=MemorySaver(),
                         code_mode=True, lazy_tools_mode=True)
    node = graph.get_graph().nodes["agent"].data
    assert node.lazy_tools_mode and not node.code_mode


# --- review round 2 -------------------------------------------------------------------------------------------

def test_huge_result_is_refused_without_serializing_it(monkeypatch):
    calls = []
    real_dumps = json.dumps
    monkeypatch.setattr(code_mode.json, "dumps", lambda *a, **k: calls.append(1) or real_dumps(*a, **k))
    d = dispatcher_for([make_tool("dump", "art", "artifact", lambda: [{"text": "x" * 1000}] * 50)],
                       limits={"max_result_bytes": 10_000})
    with pytest.raises(RuntimeError, match="result was over"):
        run(d("art.dump", "{}"))
    assert calls == []


def test_small_result_shape_is_still_reported():
    d = dispatcher_for(github_tools())
    run(d("gh_main.get_issues", json.dumps({"repo": "a/b"})))
    assert d.call_log[0]["shape"] == "[{number:int, state:str, repo:str}] x1"


def test_timeout_never_exceeds_the_platform_sandbox_timeout(fake_sandbox, monkeypatch):
    from elitea_sdk.runtime.langchain.pyodide_sandbox import PyodideSandbox
    seen = {}
    real = PyodideSandbox.execute_with_bridge

    async def spy(self, code, dispatcher, **kwargs):
        seen["timeout"] = kwargs["timeout_seconds"]
        return await real(self, code, dispatcher, **kwargs)

    monkeypatch.setattr(PyodideSandbox, "execute_with_bridge", spy)
    monkeypatch.setenv("SANDBOX_TIMEOUT_SECONDS", "30")
    execute(github_tools(), "print(1)", timeout_seconds=120)
    assert seen["timeout"] == 30


def test_calls_cancelled_by_a_timeout_are_reported_as_indeterminate(fake_sandbox):
    import time as _time
    slow = make_tool("write_slow", "art", "artifact", lambda: _time.sleep(5) or "ok")
    report = execute([slow], "await art.write_slow()", timeout_seconds=2)
    assert report["error_class"] == "timeout"
    assert report["calls_indeterminate"][0].startswith("art.write_slow failed (indeterminate)")
    assert "may or may not have taken effect" in report["calls_indeterminate"][-1]
    assert report["calls_completed"] == []
