"""Code Mode bridged calls get the full tool result, not the LLM-context cut (#7721).

The middleware-wrapped tool truncates results to TOOL_RESULT_MAX_CHARS for the LLM. A program run by
execute_code only sends its printed output to the LLM, so bridged calls must see the whole result,
while direct calls keep today's truncation.
"""
import asyncio
import json

import pytest
from langchain_core.tools import StructuredTool

from elitea_sdk.runtime.middleware.strategies import LoggingStrategy
from elitea_sdk.runtime.middleware.tool_exception_handler import ToolExceptionHandlerMiddleware
from elitea_sdk.runtime.tool_outcome import ToolResultStatus, outcome_sink
from elitea_sdk.runtime.tool_result_bounds import tool_results_unbounded, unbounded_tool_results
from elitea_sdk.runtime.tools import code_mode
from elitea_sdk.runtime.tools.code_mode import (
    BridgeDispatcher, CodeModeStubs, create_code_mode_tools, read_code_mode_limits,
)
from elitea_sdk.runtime.tools.lazy_tools import ToolRegistry
from elitea_sdk.runtime.utils.trace_limits import (
    TOOL_RESULT_MARKER_KEY, TOOL_RESULT_MAX_CHARS, configure_tool_result_limits,
)

LIMIT = TOOL_RESULT_MAX_CHARS
N_ISSUES = 700


@pytest.fixture(autouse=True)
def default_limits():
    configure_tool_result_limits(enabled=True, limit=LIMIT, per_toolkit={})
    yield
    configure_tool_result_limits(enabled=True, limit=LIMIT, per_toolkit={})


def _issues():
    # ~700 issues with long titles and nested lists: well over the 200k LLM cap, well under Code Mode's 1 MB.
    return [{"number": i, "title": "t" * 300, "created_at": "2026-11-01",
             "labels": ["bug", "p1"], "assignees": ["alice", "bob"]} for i in range(N_ISSUES)]


def _wrap(tool):
    return ToolExceptionHandlerMiddleware(strategies=[LoggingStrategy()]).wrap_tool(tool)


def _meta(name):
    return {"toolkit_name": "gh", "toolkit_type": "github", "tool_name": name}


def sync_tool():
    return _wrap(StructuredTool.from_function(func=_issues, name="get_issues", description="x",
                                              metadata=_meta("get_issues")))


def async_tool():
    async def _aissues():
        return _issues()
    return _wrap(StructuredTool.from_function(coroutine=_aissues, name="get_issues", description="x",
                                              metadata=_meta("get_issues")))


def dispatcher_for(tools, limits=None):
    stubs = CodeModeStubs(ToolRegistry.from_tools(tools))
    return BridgeDispatcher(stubs, None, {**read_code_mode_limits(), **(limits or {})}, None)


def _assert_full(items):
    assert len(items) == N_ISSUES
    assert all(isinstance(x, dict) for x in items)
    assert not any(TOOL_RESULT_MARKER_KEY in x for x in items)
    assert all(x["labels"] == ["bug", "p1"] and x["assignees"] == ["alice", "bob"] for x in items)


def _assert_truncated(result):
    assert len(result) < N_ISSUES
    assert any(not isinstance(x, dict) or TOOL_RESULT_MARKER_KEY in x for x in result)


@pytest.mark.parametrize("make", [sync_tool, async_tool], ids=["sync", "async"])
def test_bridged_call_returns_full_result_and_records_no_truncation(make):
    d = dispatcher_for([make()])
    with outcome_sink() as sink:
        value = asyncio.run(d("gh.get_issues", "{}"))
    _assert_full(json.loads(value))
    assert d.call_log[0]["ok"] is True
    assert not [o for o in sink if o.status == ToolResultStatus.TRUNCATED]


def test_direct_call_is_still_truncated_and_recorded():
    with outcome_sink() as sink:
        result = sync_tool().invoke({})
    _assert_truncated(result)
    assert [o for o in sink if o.status == ToolResultStatus.TRUNCATED]


def test_flag_is_scoped_to_the_bridged_call_even_after_an_error():
    tool = sync_tool()
    d = dispatcher_for([tool])
    asyncio.run(d("gh.get_issues", "{}"))
    assert tool_results_unbounded() is False
    _assert_truncated(tool.invoke({}))

    with pytest.raises(ValueError):
        with unbounded_tool_results():
            raise ValueError("boom")
    assert tool_results_unbounded() is False


def test_result_over_code_mode_limit_is_still_refused():
    d = dispatcher_for([sync_tool()], limits={"max_result_bytes": 100_000})
    with pytest.raises(RuntimeError, match=r"gh.get_issues result was (over )?.* limit 98 KB.*fewer items"):
        asyncio.run(d("gh.get_issues", "{}"))
    assert d.call_log[0]["error_class"] == "result_too_large"


def test_huge_nested_result_is_refused_without_serializing(monkeypatch):
    # Nested ints slip past a string-only bound; the estimate must refuse before json.dumps copies them.
    huge = [[list(range(1000)) for _ in range(100)] for _ in range(100)]
    tool = _wrap(StructuredTool.from_function(func=lambda: huge, name="dump", description="x",
                                              metadata={"toolkit_name": "art", "toolkit_type": "artifact"}))
    d = dispatcher_for([tool])

    def _no_dumps(*a, **k):
        raise AssertionError("json.dumps must not run on a refused result")
    monkeypatch.setattr(code_mode.json, "dumps", _no_dumps)
    with pytest.raises(RuntimeError, match=r"art.dump result was over .* limit 1.0 MB"):
        asyncio.run(d("art.dump", "{}"))
    assert d.call_log[0]["error_class"] == "result_too_large"


def test_small_int_result_near_the_limit_is_not_refused_by_the_estimate():
    # Small ints are over-estimated (~3x); a result that truly fits must still pass.
    ints = list(range(10)) * 50_000
    tool = _wrap(StructuredTool.from_function(func=lambda: ints, name="dump", description="x",
                                              metadata={"toolkit_name": "art", "toolkit_type": "artifact"}))
    real = len(json.dumps(ints))
    d = dispatcher_for([tool], limits={"max_result_bytes": real + 10})
    assert json.loads(asyncio.run(d("art.dump", "{}"))) == ints


def test_description_has_the_new_program_rules():
    desc = create_code_mode_tools(ToolRegistry.from_tools([sync_tool()]))[0].description
    assert "Each program runs in a fresh process" in desc
    assert "Never print a whole result" in desc
    assert "Do not import them" in desc
