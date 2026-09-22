"""Regression test for https://github.com/EliteaAI/elitea_issues/issues/6274

MCP tool schemas may contain property names (e.g. "fname[]") that violate
Anthropic's tool schema property-name pattern (^[a-zA-Z0-9_.-]{1,64}$),
causing all downstream API calls to fail with a 400 error. Property names
must be sanitized when building the args_schema, while the
original name is still used when the tool is actually invoked against the
MCP server.
"""
import pytest

from elitea_sdk.runtime.tools.mcp_input_schema import (
    build_mcp_args_schema,
    sanitize_property_name,
)
from elitea_sdk.runtime.tools.mcp_server_tool import McpServerTool


def test_sanitize_property_name_strips_invalid_chars():
    assert sanitize_property_name("fname[]") == "fname"
    assert sanitize_property_name("valid_name.1") == "valid_name.1"
    assert sanitize_property_name("a[b]c{d}e") == "abcde"


def test_sanitize_property_name_fallback_when_empty():
    assert sanitize_property_name("[]") == "field"


def test_sanitize_property_name_truncates_to_64_chars():
    long_name = "a" * 100
    # Already valid (only allowed chars), so it should be returned unchanged
    # since it matches the pattern only up to 64 chars requirement check.
    result = sanitize_property_name(long_name + "[]")
    assert len(result) <= 64
    assert set(result) <= set("a")


def test_args_schema_sanitizes_invalid_property_names():
    schema = {
        "type": "object",
        "properties": {
            "fname[]": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files to delete",
            }
        },
        "required": ["fname[]"],
    }
    fields = build_mcp_args_schema(schema)

    json_schema = fields["args_schema"]
    assert "fname" in json_schema["properties"]
    assert "fname[]" not in json_schema["properties"]
    import re

    for prop_name in json_schema["properties"]:
        assert re.match(r"^[a-zA-Z0-9_.-]{1,64}$", prop_name), prop_name

    # Mapping from sanitized -> original name is retained for the tool call.
    assert json_schema["required"] == ["fname"]
    assert fields["property_name_map"] == {"fname": "fname[]"}


def test_args_schema_leaves_valid_property_names_untouched():
    schema = {
        "type": "object",
        "properties": {
            "valid_name": {"type": "string", "description": "A valid name"}
        },
        "required": [],
    }
    fields = build_mcp_args_schema(schema)
    assert fields["args_schema"] == schema
    assert fields["property_name_map"] == {}


class _FakeClient:
    def __init__(self):
        self.last_call_data = None

    def mcp_tool_call(self, call_data):
        self.last_call_data = call_data
        return "ok"


def test_run_translates_sanitized_names_back_to_original():
    schema = {
        "type": "object",
        "properties": {
            "fname[]": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files to delete",
            }
        },
        "required": ["fname[]"],
    }

    client = _FakeClient()
    tool = McpServerTool(
        name="delete_artifacts_artifacts",
        description="Delete artifacts",
        **build_mcp_args_schema(schema),
        client=client,
        server="test-server",
    )

    # The LLM/agent will call the tool using the sanitized field name.
    result = tool._run(fname=["file1.txt", "file2.txt"])

    assert result == "ok"
    assert client.last_call_data["params"]["name"] == "delete_artifacts_artifacts"
    # The MCP server must receive the ORIGINAL property name.
    assert client.last_call_data["params"]["arguments"] == {
        "fname[]": ["file1.txt", "file2.txt"]
    }


def test_run_without_sanitized_properties_is_unaffected():
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    client = _FakeClient()
    tool = McpServerTool(
        name="search",
        description="Search",
        **build_mcp_args_schema(schema),
        client=client,
        server="test-server",
    )

    tool._run(query="hello")
    assert client.last_call_data["params"]["arguments"] == {"query": "hello"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
