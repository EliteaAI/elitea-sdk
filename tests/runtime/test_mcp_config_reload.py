"""MCP configuration cache reload contract."""

import pytest

from elitea_sdk.runtime.toolkits import mcp_config


@pytest.fixture(autouse=True)
def clear_mcp_config_cache(monkeypatch):
    monkeypatch.setattr(mcp_config, "_server_configs", None)
    monkeypatch.setattr(mcp_config, "_mcp_config_schema_cache", {})


def test_refresh_replaces_cached_definitions():
    mcp_config.refresh_mcp_server_configs({
        "old": {"type": "http", "url": "https://old.example.test/mcp"},
    })

    mcp_config.refresh_mcp_server_configs({
        "new": {"type": "http", "url": "https://new.example.test/mcp"},
    })

    assert mcp_config.get_mcp_server_config("old") is None
    assert mcp_config.get_mcp_server_config("new")["url"] == "https://new.example.test/mcp"


def test_refresh_copies_runtime_owned_configuration():
    definitions = {
        "remote": {"type": "http", "url": "https://mcp.example.test/api"},
    }

    mcp_config.refresh_mcp_server_configs(definitions)
    definitions["remote"]["url"] = "https://mutated.example.test/api"

    assert mcp_config.get_mcp_server_config("remote")["url"] == "https://mcp.example.test/api"


def test_refresh_without_value_loads_current_configuration(monkeypatch):
    monkeypatch.setattr(
        mcp_config,
        "load_mcp_servers_config",
        lambda: {"loaded": {"type": "stdio", "command": "uvx"}},
    )

    mcp_config.refresh_mcp_server_configs()

    assert mcp_config.get_all_mcp_server_configs() == {
        "loaded": {"type": "stdio", "command": "uvx"},
    }


def test_refresh_rejects_non_mapping_configuration():
    with pytest.raises(TypeError, match="must be an object"):
        mcp_config.refresh_mcp_server_configs([])


def test_generated_toolkit_schemas_follow_added_and_removed_servers():
    mcp_config.refresh_mcp_server_configs({
        "First": {
            "type": "http",
            "url": "https://first.example.test/mcp",
            "headers": {"Authorization": "Bearer {token}"},
        },
    })
    assert [model.__name__ for model in mcp_config.get_mcp_config_toolkit_schemas()] == [
        "mcp_First",
    ]

    mcp_config.refresh_mcp_server_configs({
        "Second": {
            "type": "http",
            "url": "https://second.example.test/mcp",
            "headers": {"Authorization": "Bearer {token}"},
        },
    })
    assert [model.__name__ for model in mcp_config.get_mcp_config_toolkit_schemas()] == [
        "mcp_Second",
    ]


def test_generated_schema_cache_rediscovers_only_changed_servers(monkeypatch):
    discovered_urls = []

    def discover(url, _headers):
        discovered_urls.append(url)
        return []

    monkeypatch.setattr(mcp_config, "_discover_tools_for_http_server", discover)
    first = {
        "One": {"type": "http", "url": "https://one.example.test/mcp"},
        "Two": {"type": "http", "url": "https://two.example.test/mcp"},
    }
    mcp_config.refresh_mcp_server_configs(first)

    mcp_config.get_mcp_config_toolkit_schemas()
    mcp_config.get_mcp_config_toolkit_schemas()

    assert discovered_urls == [
        "https://one.example.test/mcp",
        "https://two.example.test/mcp",
    ]

    mcp_config.refresh_mcp_server_configs({
        "One": first["One"],
        "Two": {"type": "http", "url": "https://two-new.example.test/mcp"},
        "Three": {"type": "http", "url": "https://three.example.test/mcp"},
    })
    schemas = mcp_config.get_mcp_config_toolkit_schemas()

    assert [model.__name__ for model in schemas] == ["mcp_One", "mcp_Two", "mcp_Three"]
    assert discovered_urls == [
        "https://one.example.test/mcp",
        "https://two.example.test/mcp",
        "https://two-new.example.test/mcp",
        "https://three.example.test/mcp",
    ]
