from types import SimpleNamespace

import pytest

from elitea_sdk.runtime.models.mcp_models import McpConnectionConfig
from elitea_sdk.runtime.toolkits.mcp import McpToolkit


LONG_DESCRIPTION = (
    "Searches accounts using the supplied filters. "
    "Use the narrowest filters available before calling this tool. "
    + "Detailed matching guidance for account searches. " * 30
    + "Returns the complete matching account records."
)


def test_mcp_schema_keeps_full_descriptions_by_default():
    schema = McpToolkit.toolkit_config_schema().model_json_schema()

    setting = schema["properties"]["max_tool_description_length"]

    assert setting["default"] == 0
    assert setting["minimum"] == 0


def test_direct_mcp_tool_keeps_full_description_by_default():
    connection = McpConnectionConfig(url="https://mcp.example.test/tools")

    tool = McpToolkit._create_tool_from_dict(
        tool_dict={
            "name": "search_accounts",
            "description": LONG_DESCRIPTION,
            "inputSchema": {"type": "object"},
        },
        toolkit_name="compliance",
        toolkit_type="mcp",
        connection_config=connection,
        timeout=60,
        client=None,
    )

    assert tool is not None
    assert tool.description == (
        f"{LONG_DESCRIPTION}\n"
        "Toolkit: compliance (https://mcp.example.test/tools)"
    )
    assert len(tool.description) > 1000


def test_get_toolkit_routes_configured_description_cap(monkeypatch):
    captured = {}

    def fake_create_tools_from_server(cls, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        McpToolkit,
        "_create_tools_from_server",
        classmethod(fake_create_tools_from_server),
    )

    McpToolkit.get_toolkit(
        url="https://mcp.example.test/tools",
        toolkit_name="compliance",
        max_tool_description_length="240",
    )

    assert captured["max_tool_description_length"] == 240


@pytest.mark.parametrize("factory", ["metadata", "static"])
def test_configured_cap_uses_readable_boundary_and_preserves_toolkit_context(
    factory,
    caplog,
):
    max_length = 240

    with caplog.at_level("WARNING"):
        if factory == "metadata":
            tool = McpToolkit._create_tool_from_metadata(
                tool_metadata=SimpleNamespace(
                    name="search_accounts",
                    description=LONG_DESCRIPTION,
                    input_schema={"type": "object"},
                    server="compliance",
                ),
                toolkit_name="compliance",
                toolkit_type="mcp",
                timeout=60,
                client=object(),
                max_tool_description_length=max_length,
            )
        else:
            tool = McpToolkit._create_single_tool(
                toolkit_name="compliance",
                toolkit_type="mcp",
                available_tool={
                    "name": "search_accounts",
                    "description": LONG_DESCRIPTION,
                    "inputSchema": {"type": "object"},
                },
                timeout=60,
                client=object(),
                max_tool_description_length=max_length,
            )

    assert tool is not None
    assert len(tool.description) <= max_length
    assert tool.description.endswith("\nToolkit: compliance")
    assert "...\nToolkit: compliance" in tool.description
    assert "description shortened" in caplog.text
    assert LONG_DESCRIPTION not in caplog.text
