# Copyright (c) 2026 EPAM Systems
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Regression tests for issue #5253:
  Smart Tool Selection collapses all same-type toolkits to the FIRST one,
  silently omitting tools from every other toolkit of that type.

The fix must ensure:
  - generate_index() advertises the UNION of tools across ALL toolkits of a type.
  - ListToolkitsTool._run() does the same.
  - Same-type toolkits with IDENTICAL tool sets produce no duplication.
"""

import json
import pytest
from typing import Optional
from unittest.mock import MagicMock

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from elitea_sdk.runtime.tools.lazy_tools import ToolRegistry, ListToolkitsTool


# ---------------------------------------------------------------------------
# Minimal stub tool — carries metadata but does nothing else.
# ---------------------------------------------------------------------------

class _StubTool(BaseTool):
    """Minimal stub for ToolRegistry tests — no real execution needed."""

    name: str
    description: str = "stub"
    metadata: dict = {}

    def _run(self, *args, **kwargs):
        return "stub"


def _make_tool(tool_name: str, toolkit_name: str, toolkit_type: str) -> _StubTool:
    return _StubTool(
        name=tool_name,
        metadata={"toolkit_name": toolkit_name, "toolkit_type": toolkit_type},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mcp_registry():
    """
    Registry with 3 mcp-type toolkits mirroring the Elitea MCP setup:
      - "Elitea Applications"  : 9 tools  (snake_case)
      - "Elitea Chat"          : 11 tools (8 conversation + 3 participant, mixed casing)
      - "Elitea Toolkits"      : 3 tools  (includes get_elitea_core_tools)

    Total: 23 distinct tools.
    """
    apps_tools = [
        _make_tool(f"get_elitea_core_application_{i}", "Elitea Applications", "mcp")
        for i in range(9)
    ]
    chat_tools = [
        _make_tool(f"getEliteaCoreConversation_{i}", "Elitea Chat", "mcp")
        for i in range(8)
    ] + [
        _make_tool(f"postEliteaCoreParticipants_{i}", "Elitea Chat", "mcp")
        for i in range(3)
    ]
    toolkit_tools = [
        _make_tool("get_elitea_core_tools", "Elitea Toolkits", "mcp"),
        _make_tool("list_elitea_core_toolkits", "Elitea Toolkits", "mcp"),
        _make_tool("invoke_elitea_core_tool", "Elitea Toolkits", "mcp"),
    ]
    all_tools = apps_tools + chat_tools + toolkit_tools
    return ToolRegistry.from_tools(all_tools)


# Convenient constant for how many total tools are in the mcp_registry fixture.
_MCP_TOOL_COUNT = 23  # 9 apps + 11 chat (8 conversation + 3 participants) + 3 toolkits


@pytest.fixture
def identical_tool_registry():
    """
    Two same-type toolkits with IDENTICAL tool sets (multi-repo GitHub pattern).
    The union must equal the original set — no duplication.
    """
    tools = [
        _make_tool("search_code", "sdk", "github"),
        _make_tool("create_issue", "sdk", "github"),
        _make_tool("search_code", "core", "github"),
        _make_tool("create_issue", "core", "github"),
    ]
    return ToolRegistry.from_tools(tools)


# ---------------------------------------------------------------------------
# Tests: generate_index()
# ---------------------------------------------------------------------------

class TestGenerateIndexMultipleMcpToolkits:
    """generate_index() must advertise ALL tools across same-type toolkits."""

    def test_all_26_tools_appear_in_index(self, mcp_registry):
        """Regression: before fix, only 9 tools were listed; after fix all 26 must appear."""
        index = mcp_registry.generate_index()
        # Spot-check one tool from each toolkit
        assert "get_elitea_core_application_0" in index, "Applications tool missing from index"
        assert "getEliteaCoreConversation_0" in index, "Chat conversation tool missing from index"
        assert "postEliteaCoreParticipants_0" in index, "Chat participants tool missing from index"
        assert "get_elitea_core_tools" in index, "Toolkits tool missing from index"

    def test_all_tool_names_present(self, mcp_registry):
        """Every one of the 26 distinct tool names must appear in the index."""
        index = mcp_registry.generate_index()
        for i in range(9):
            assert f"get_elitea_core_application_{i}" in index, f"Missing application tool {i}"
        for i in range(8):
            assert f"getEliteaCoreConversation_{i}" in index, f"Missing chat conversation tool {i}"
        for i in range(3):
            assert f"postEliteaCoreParticipants_{i}" in index, f"Missing chat participant tool {i}"
        assert "get_elitea_core_tools" in index
        assert "list_elitea_core_toolkits" in index
        assert "invoke_elitea_core_tool" in index

    def test_tool_count_in_index_reflects_all_toolkits(self, mcp_registry):
        """The total tool count shown in the index must cover all toolkits (23 in the fixture)."""
        index = mcp_registry.generate_index()
        # The index ends with a "Total: N toolkits, M tools" line
        assert f"{_MCP_TOOL_COUNT} tools" in index, (
            f"Expected '{_MCP_TOOL_COUNT} tools' in index summary line"
        )

    def test_casing_does_not_affect_inclusion(self, mcp_registry):
        """snake_case and camelCase tool names must both appear — casing is irrelevant to indexing."""
        index = mcp_registry.generate_index()
        assert "get_elitea_core_application_0" in index   # snake_case
        assert "getEliteaCoreConversation_0" in index     # camelCase


class TestGenerateIndexIdenticalToolkits:
    """Same-type toolkits with identical tool sets must not duplicate tools in the index."""

    def test_no_duplication_for_identical_same_type_toolkits(self, identical_tool_registry):
        """Union of identical sets = the original set; no duplication in the Tools line."""
        index = identical_tool_registry.generate_index()
        # Find the "  Tools (N): ..." line and verify no duplicate names there.
        tools_line = next(
            (line for line in index.splitlines() if line.strip().startswith("Tools (")),
            None,
        )
        assert tools_line is not None, "Could not find Tools line in index"
        tool_names_str = tools_line.split(":", 1)[1].strip()
        tool_names_listed = [t.strip() for t in tool_names_str.split(",") if t.strip()]
        assert len(tool_names_listed) == len(set(tool_names_listed)), (
            f"Duplicate tool names in Tools line: {tool_names_listed}"
        )
        assert len(tool_names_listed) == 2, (
            f"Expected 2 distinct tools (create_issue, search_code), got {tool_names_listed}"
        )

    def test_both_toolkit_names_listed_for_identical_type(self, identical_tool_registry):
        """Both toolkit names must still appear even when tools are identical."""
        index = identical_tool_registry.generate_index()
        assert '"sdk"' in index or "sdk" in index
        assert '"core"' in index or "core" in index


# ---------------------------------------------------------------------------
# Tests: ListToolkitsTool._run()
# ---------------------------------------------------------------------------

class TestListToolkitsToolMultipleMcpToolkits:
    """ListToolkitsTool must report the union of tools for same-type toolkits."""

    def _run_list_toolkits(self, registry: ToolRegistry) -> list:
        tool = ListToolkitsTool(registry=registry)
        result_json = tool._run()
        return json.loads(result_json)

    def test_all_tools_in_list_toolkits_output(self, mcp_registry):
        """Regression: before fix, only 9 tools were listed; after fix all 23 must appear."""
        entries = self._run_list_toolkits(mcp_registry)
        mcp_entry = next(e for e in entries if e["toolkit_type"] == "mcp")
        tools_listed = mcp_entry["tools"]
        assert len(tools_listed) == _MCP_TOOL_COUNT, (
            f"Expected {_MCP_TOOL_COUNT} tools for mcp type, got {len(tools_listed)}: {tools_listed}"
        )

    def test_applications_tools_in_list_toolkits(self, mcp_registry):
        entries = self._run_list_toolkits(mcp_registry)
        mcp_entry = next(e for e in entries if e["toolkit_type"] == "mcp")
        tools = mcp_entry["tools"]
        for i in range(9):
            assert f"get_elitea_core_application_{i}" in tools

    def test_chat_tools_in_list_toolkits(self, mcp_registry):
        entries = self._run_list_toolkits(mcp_registry)
        mcp_entry = next(e for e in entries if e["toolkit_type"] == "mcp")
        tools = mcp_entry["tools"]
        for i in range(8):
            assert f"getEliteaCoreConversation_{i}" in tools
        for i in range(3):
            assert f"postEliteaCoreParticipants_{i}" in tools

    def test_toolkit_tools_in_list_toolkits(self, mcp_registry):
        entries = self._run_list_toolkits(mcp_registry)
        mcp_entry = next(e for e in entries if e["toolkit_type"] == "mcp")
        tools = mcp_entry["tools"]
        assert "get_elitea_core_tools" in tools
        assert "list_elitea_core_toolkits" in tools
        assert "invoke_elitea_core_tool" in tools

    def test_tool_count_field_equals_total(self, mcp_registry):
        entries = self._run_list_toolkits(mcp_registry)
        mcp_entry = next(e for e in entries if e["toolkit_type"] == "mcp")
        assert mcp_entry["tool_count"] == _MCP_TOOL_COUNT

    def test_all_three_toolkit_names_present(self, mcp_registry):
        """All 3 toolkit names must appear in the 'toolkits' array."""
        entries = self._run_list_toolkits(mcp_registry)
        mcp_entry = next(e for e in entries if e["toolkit_type"] == "mcp")
        toolkit_names = [t["name"] for t in mcp_entry["toolkits"]]
        assert "Elitea Applications" in toolkit_names
        assert "Elitea Chat" in toolkit_names
        assert "Elitea Toolkits" in toolkit_names


class TestListToolkitsToolIdenticalToolkits:
    """ListToolkitsTool must not duplicate tools for same-type identical toolkits."""

    def _run_list_toolkits(self, registry: ToolRegistry) -> list:
        tool = ListToolkitsTool(registry=registry)
        result_json = tool._run()
        return json.loads(result_json)

    def test_no_duplication_for_identical_same_type_toolkits(self, identical_tool_registry):
        entries = self._run_list_toolkits(identical_tool_registry)
        github_entry = next(e for e in entries if e["toolkit_type"] == "github")
        tools = github_entry["tools"]
        assert len(tools) == len(set(tools)), "Duplicate tool names in list_toolkits output"
        assert len(tools) == 2  # search_code + create_issue
