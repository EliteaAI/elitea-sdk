"""
Tests for bounded tool-result logging (issue #5679).

A tool (especially an MCP toolkit) can return a very large payload — e.g. a
data-fetch tool returning a full dataset. The SDK previously logged the entire
raw result at INFO on every tool call, which expanded a single statement into
thousands of log lines and drowned the runtime pod's log pipeline. These tests
lock in the bounded-summary behaviour shared by all tool-node types (FunctionTool,
ToolNode, AgentNode): INFO carries only identity + size + a capped preview; the
full body is only ever emitted at DEBUG.
"""
import logging

from elitea_sdk.runtime.langchain.utils import (
    tool_result_summary,
    log_tool_result,
    TOOL_RESULT_PREVIEW_CHARS,
)


def test_summary_truncates_large_result():
    big = "x" * 50_000
    line = tool_result_summary("node_a", "staffing_tool", 42, big)
    assert "node_a" in line
    assert "staffing_tool" in line
    assert "toolkit=42" in line
    assert "50000 chars total" in line
    assert "truncated" in line
    # the full body never appears — line stays small regardless of payload size
    assert len(line) < TOOL_RESULT_PREVIEW_CHARS + 300


def test_summary_small_result_not_marked_truncated():
    line = tool_result_summary("node_b", "tiny_tool", None, "ok")
    assert "truncated" not in line
    assert "toolkit=" not in line  # no id -> no fragment
    assert "tiny_tool" in line


def test_summary_handles_non_string_result():
    payload = {"result": list(range(1000))}
    line = tool_result_summary("node_c", "dict_tool", 7, payload)
    assert "dict" in line  # type name rendered
    assert "toolkit=7" in line


def test_summary_custom_label():
    # ToolNode logs LLM-extracted params through the same helper with a params label
    line = tool_result_summary("node_p", "structured_tool", 3, {"a": 1}, label="tool params")
    assert "tool params:" in line


def test_info_line_is_bounded_and_debug_has_full_body(caplog):
    big = "y" * 20_000
    node_logger = logging.getLogger("elitea_sdk.runtime.tools.function")
    with caplog.at_level(logging.DEBUG, logger="elitea_sdk.runtime.tools.function"):
        log_tool_result(node_logger, "node_d", "big_tool", 1, big)

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]

    assert len(info_records) == 1
    assert len(info_records[0].getMessage()) < TOOL_RESULT_PREVIEW_CHARS + 300
    assert "truncated" in info_records[0].getMessage()

    assert len(debug_records) == 1
    assert "y" * 1000 in debug_records[0].getMessage()


def test_no_debug_body_when_debug_disabled(caplog):
    node_logger = logging.getLogger("elitea_sdk.runtime.tools.agent")
    with caplog.at_level(logging.INFO, logger="elitea_sdk.runtime.tools.agent"):
        log_tool_result(node_logger, "node_e", "some_tool", None, "z" * 5000)
    assert all(r.levelno != logging.DEBUG for r in caplog.records)
