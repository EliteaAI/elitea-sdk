"""Regression test for https://github.com/EliteaAI/elitea_issues/issues/6274

MCP tool schemas may contain property names (e.g. "fname[]") that violate
Anthropic's tool schema property-name pattern (^[a-zA-Z0-9_.-]{1,64}$),
causing all downstream API calls to fail with a 400 error. Property names
must be sanitized when building the pydantic args_schema, while the
original name is still used when the tool is actually invoked against the
MCP server.
"""
import pytest

from elitea_sdk.runtime.tools.mcp_server_tool import (
    McpServerTool,
    sanitize_property_name,
)


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


def test_create_pydantic_model_sanitizes_invalid_property_names():
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
    model = McpServerTool.create_pydantic_model_from_schema(schema, "ArgsSchema")

    # The sanitized field name is used in the pydantic model / JSON schema.
    assert "fname" in model.model_fields
    assert "fname[]" not in model.model_fields

    json_schema = model.model_json_schema()
    assert "fname" in json_schema["properties"]
    assert "fname[]" not in json_schema["properties"]
    import re

    for prop_name in json_schema["properties"]:
        assert re.match(r"^[a-zA-Z0-9_.-]{1,64}$", prop_name), prop_name

    # Mapping from sanitized -> original name is retained for the tool call.
    assert model.__property_name_map__ == {"fname": "fname[]"}


def test_create_pydantic_model_leaves_valid_property_names_untouched():
    schema = {
        "type": "object",
        "properties": {
            "valid_name": {"type": "string", "description": "A valid name"}
        },
        "required": [],
    }
    model = McpServerTool.create_pydantic_model_from_schema(schema, "ArgsSchema")
    assert "valid_name" in model.model_fields
    assert model.__property_name_map__ == {}


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
    args_schema = McpServerTool.create_pydantic_model_from_schema(schema, "ArgsSchema")

    client = _FakeClient()
    tool = McpServerTool(
        name="delete_artifacts_artifacts",
        description="Delete artifacts",
        args_schema=args_schema,
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
    args_schema = McpServerTool.create_pydantic_model_from_schema(schema, "ArgsSchema")

    client = _FakeClient()
    tool = McpServerTool(
        name="search",
        description="Search",
        args_schema=args_schema,
        client=client,
        server="test-server",
    )

    tool._run(query="hello")
    assert client.last_call_data["params"]["arguments"] == {"query": "hello"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
