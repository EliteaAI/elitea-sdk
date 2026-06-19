"""Tests for blocked tools security enforcement across all code paths."""
import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.tools import BaseTool, StructuredTool

from elitea_sdk.runtime.toolkits.security import (
    configure_blocklist,
    is_tool_blocked,
    is_toolkit_blocked,
    get_blocked_tools_for_toolkit,
    normalize_tool_name,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_blocklist():
    """Reset global blocklist state before each test."""
    configure_blocklist(blocked_toolkits=[], blocked_tools={})
    yield
    configure_blocklist(blocked_toolkits=[], blocked_tools={})


def _make_mock_tool(name: str, toolkit_type: str = "", toolkit_name: str = "") -> MagicMock:
    """Create a mock BaseTool with metadata."""
    mock = MagicMock(spec=BaseTool)
    mock.name = name
    mock.description = f"mock {name}"
    mock.metadata = {}
    if toolkit_type:
        mock.metadata["toolkit_type"] = toolkit_type
    if toolkit_name:
        mock.metadata["toolkit_name"] = toolkit_name
    return mock


# ── Core blocklist checks ────────────────────────────────────────────────

class TestBlocklistConfiguration:
    def test_configure_blocked_tools(self):
        configure_blocklist(blocked_tools={"github": ["create_issue", "delete_repo"]})
        assert is_tool_blocked("github", "create_issue")
        assert is_tool_blocked("github", "delete_repo")
        assert not is_tool_blocked("github", "get_issue")

    def test_configure_blocked_toolkits(self):
        configure_blocklist(blocked_toolkits=["shell"])
        assert is_toolkit_blocked("shell")
        assert not is_toolkit_blocked("github")

    def test_blocked_toolkit_blocks_all_tools(self):
        configure_blocklist(blocked_toolkits=["shell"])
        assert is_tool_blocked("shell", "execute_command")
        assert is_tool_blocked("shell", "any_other_tool")

    def test_case_insensitive(self):
        configure_blocklist(blocked_tools={"GitHub": ["Create_Issue"]})
        assert is_tool_blocked("github", "create_issue")

    def test_tool_name_alias_normalization(self):
        configure_blocklist(blocked_tools={"github": ["create_issue"]})
        # Prefixed variants should also be blocked
        assert is_tool_blocked("github", "github___create_issue")
        assert is_tool_blocked("github", "github:create_issue")

    def test_empty_blocklist(self):
        configure_blocklist(blocked_toolkits=[], blocked_tools={})
        assert not is_tool_blocked("github", "create_issue")
        assert not is_toolkit_blocked("github")


# ── Separator/format-insensitive matching (issue #5199) ──────────────────

class TestCanonicalMatching:
    def test_blocked_tool_matches_naming_style_variants(self):
        # Configured in one style; invoked in many. All collapse to "createfile".
        configure_blocklist(blocked_tools={"GitHub": ["Create-File"]})
        for invoked in ("create_file", "CreateFile", "create-file", "Create File", "createfile"):
            assert is_tool_blocked("github", invoked), invoked

    def test_blocked_tool_matches_prefixed_and_styled(self):
        configure_blocklist(blocked_tools={"github": ["create_file"]})
        # Routing prefixes (___ / :) plus a casing change still match.
        assert is_tool_blocked("github", "github___CreateFile")
        assert is_tool_blocked("GITHUB", "github:create-file")

    def test_blocked_toolkit_matches_naming_style_variants(self):
        configure_blocklist(blocked_toolkits=["Data_Analysis"])
        for invoked in ("data_analysis", "data-analysis", "DataAnalysis", "Data Analysis"):
            assert is_toolkit_blocked(invoked), invoked

    def test_matching_is_toolkit_scoped(self):
        # Same logical tool blocked under github must NOT be blocked elsewhere
        # (protects e.g. an Artifacts tool sharing a common verb). AC9.
        configure_blocklist(blocked_tools={"github": ["create_file"]})
        assert is_tool_blocked("github", "CreateFile")
        assert not is_tool_blocked("artifacts", "CreateFile")
        assert not is_tool_blocked("filesystem", "create_file")

    def test_get_blocked_tools_for_toolkit_styled_lookup(self):
        configure_blocklist(blocked_tools={"GitHub": ["Create-File", "Delete_Repo"]})
        blocked = get_blocked_tools_for_toolkit("GITHUB")
        assert set(blocked) == {"createfile", "deleterepo"}

    def test_separator_only_entries_are_dropped(self):
        # Keys/values that canonicalize to "" (only separators) must not be
        # stored as empty entries — otherwise an empty toolkit key or tool name
        # would match nothing meaningful and confuse debugging.
        configure_blocklist(
            blocked_toolkits=["---", "  ", "shell"],
            blocked_tools={"***": ["create_file"], "github": ["---", "delete_repo"]},
        )
        assert get_blocked_tools_for_toolkit("github") == ["deleterepo"]
        assert get_blocked_tools_for_toolkit("***") == []
        # The empty-canonical toolkit must not be treated as blocked.
        assert not is_toolkit_blocked("")
        assert is_toolkit_blocked("shell")


# ── _filter_blocked_tools (tools/__init__.py) ────────────────────────────

class TestFilterBlockedTools:
    def test_filters_blocked_tool_from_toolkit(self):
        from elitea_sdk.tools import _filter_blocked_tools
        configure_blocklist(blocked_tools={"github": ["create_issue"]})
        tools = [
            _make_mock_tool("create_issue"),
            _make_mock_tool("get_issue"),
        ]
        result = _filter_blocked_tools(tools, "github")
        assert len(result) == 1
        assert result[0].name == "get_issue"

    def test_passes_through_when_no_blocklist(self):
        from elitea_sdk.tools import _filter_blocked_tools
        tools = [_make_mock_tool("create_issue"), _make_mock_tool("get_issue")]
        result = _filter_blocked_tools(tools, "github")
        assert len(result) == 2

    def test_handles_prefixed_tool_names(self):
        from elitea_sdk.tools import _filter_blocked_tools
        configure_blocklist(blocked_tools={"github": ["create_issue"]})
        tools = [
            _make_mock_tool("github___create_issue"),
            _make_mock_tool("get_issue"),
        ]
        result = _filter_blocked_tools(tools, "github")
        assert len(result) == 1
        assert result[0].name == "get_issue"


# ── _final_blocked_tools_filter (runtime/toolkits/tools.py) ──────────────

class TestFinalBlockedToolsFilter:
    def test_filters_by_metadata_toolkit_type(self):
        from elitea_sdk.runtime.toolkits.tools import _final_blocked_tools_filter
        configure_blocklist(blocked_tools={"github": ["create_issue"]})
        tools = [
            _make_mock_tool("create_issue", toolkit_type="github"),
            _make_mock_tool("get_issue", toolkit_type="github"),
        ]
        result = _final_blocked_tools_filter(tools)
        assert len(result) == 1
        assert result[0].name == "get_issue"

    def test_passes_non_basetool_objects(self):
        from elitea_sdk.runtime.toolkits.tools import _final_blocked_tools_filter
        configure_blocklist(blocked_tools={"github": ["create_issue"]})
        tools = ["not_a_tool", 42]
        result = _final_blocked_tools_filter(tools)
        assert len(result) == 2

    def test_passes_tools_without_metadata(self):
        from elitea_sdk.runtime.toolkits.tools import _final_blocked_tools_filter
        configure_blocklist(blocked_tools={"github": ["create_issue"]})
        tool = _make_mock_tool("create_issue")
        tool.metadata = {}  # no toolkit_type
        result = _final_blocked_tools_filter([tool])
        # Without toolkit_type we can't match → tool passes through
        assert len(result) == 1


# ── InvokeToolTool blocklist gate (lazy_tools.py) ────────────────────────

class TestInvokeToolBlockedGate:
    def test_invoke_tool_rejects_blocked_tool(self):
        from elitea_sdk.runtime.tools.lazy_tools import InvokeToolTool, ToolRegistry

        configure_blocklist(blocked_tools={"github": ["create_issue"]})

        mock_tool = _make_mock_tool("create_issue", toolkit_type="github", toolkit_name="gh")
        registry = ToolRegistry()
        registry._toolkits["gh"] = {"create_issue": mock_tool}
        registry._tool_to_toolkit["create_issue"] = "gh"
        registry._toolkit_types["gh"] = "github"

        invoke = InvokeToolTool(registry=registry)
        result = invoke._run(toolkit="gh", tool="create_issue", arguments={})

        assert "blocked" in result.lower()
        assert "security policy" in result.lower()
        # The actual tool must NOT have been invoked
        mock_tool.invoke.assert_not_called()

    def test_invoke_tool_allows_non_blocked_tool(self):
        from elitea_sdk.runtime.tools.lazy_tools import InvokeToolTool, ToolRegistry

        configure_blocklist(blocked_tools={"github": ["delete_repo"]})

        mock_tool = _make_mock_tool("get_issue", toolkit_type="github", toolkit_name="gh")
        mock_tool.invoke.return_value = '{"number": 42}'
        registry = ToolRegistry()
        registry._toolkits["gh"] = {"get_issue": mock_tool}
        registry._tool_to_toolkit["get_issue"] = "gh"
        registry._toolkit_types["gh"] = "github"

        invoke = InvokeToolTool(registry=registry)
        result = invoke._run(toolkit="gh", tool="get_issue", arguments={})

        assert "42" in result
        mock_tool.invoke.assert_called_once()

    def test_invoke_tool_rejects_blocked_toolkit(self):
        from elitea_sdk.runtime.tools.lazy_tools import InvokeToolTool, ToolRegistry

        configure_blocklist(blocked_toolkits=["shell"])

        mock_tool = _make_mock_tool("execute", toolkit_type="shell", toolkit_name="sh")
        registry = ToolRegistry()
        registry._toolkits["sh"] = {"execute": mock_tool}
        registry._tool_to_toolkit["execute"] = "sh"
        registry._toolkit_types["sh"] = "shell"

        invoke = InvokeToolTool(registry=registry)
        result = invoke._run(toolkit="sh", tool="execute", arguments={})

        assert "blocked" in result.lower()
        mock_tool.invoke.assert_not_called()
