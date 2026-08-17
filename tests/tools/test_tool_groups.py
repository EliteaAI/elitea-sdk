import pytest

from elitea_sdk.tools import AVAILABLE_TOOLKITS
from elitea_sdk.tools.github import EliteAGitHubToolkit
from elitea_sdk.tools.utils.tool_groups import GROUPS, tool_group, with_tool_groups


class FakeProducer:
    @tool_group('read')
    def get_thing(self):
        pass

    def frobnicate_widget(self):
        pass

    @tool_group('delete')
    def drop_thing(self):
        pass

    @with_tool_groups
    def get_available_tools(self):
        return [
            {"name": "get_thing", "ref": self.get_thing},
            {"name": "frobnicate_widget", "ref": self.frobnicate_widget},
            {"name": "drop_thing", "ref": self.drop_thing, "group": "read"},
            {"name": "no_ref_tool"},
        ]


def test_stamping_reads_group_from_the_ref():
    tools = {t["name"]: t for t in FakeProducer().get_available_tools()}
    assert tools["get_thing"]["group"] == "read"


def test_stamping_never_guesses_and_never_overwrites():
    tools = {t["name"]: t for t in FakeProducer().get_available_tools()}
    assert "group" not in tools["frobnicate_widget"]
    assert "group" not in tools["no_ref_tool"]
    assert tools["drop_thing"]["group"] == "read"


def test_invalid_group_name_fails_at_decoration():
    with pytest.raises(ValueError, match="not a valid tool group"):
        tool_group('reed')


def test_marker_travels_with_borrowed_methods():
    class Owner:
        @tool_group('write')
        def sync_thing(self):
            pass

    class Borrower:
        sync_thing = Owner.sync_thing

    assert Borrower().sync_thing._tool_group == "write"


def get_selected_tools_schema():
    schema = EliteAGitHubToolkit.toolkit_config_schema().model_json_schema()
    return schema["properties"]["selected_tools"]


def test_github_schema_emits_tool_groups():
    selected_tools = get_selected_tools_schema()
    groups = selected_tools["tool_groups"]
    assert set(groups) <= set(selected_tools["args_schemas"])
    assert set(groups.values()) <= set(GROUPS)


def test_github_composed_sources_all_arrive_stamped():
    groups = get_selected_tools_schema()["tool_groups"]
    assert groups["search_index"] == "read"
    assert groups["index_data"] == "write"
    assert groups["grep_file"] == "read"
    assert groups["read_multiple_files"] == "read"
    assert groups["delete_branch"] == "delete"
    assert groups["list_project_issues"] == "read"
    assert groups["generic_github_api_call"] == "execute"
    assert groups["get_me"] == "read"
    assert groups["apply_git_patch_from_file"] == "write"


def test_every_github_tool_is_classified():
    selected_tools = get_selected_tools_schema()
    unclassified = set(selected_tools["args_schemas"]) - set(selected_tools["tool_groups"])
    assert not unclassified, (
        f"Unclassified github tools: {sorted(unclassified)}. Every tool must carry @tool_group "
        "on the method that implements it — including overrides, which do not inherit the "
        "parent method's marker."
    )


def test_every_toolkit_schema_builds():
    built = 0
    for toolkit in AVAILABLE_TOOLKITS.values():
        if not hasattr(toolkit, "toolkit_config_schema"):
            continue
        schema = toolkit.toolkit_config_schema().model_json_schema()
        assert isinstance(schema, dict)
        built += 1
    assert built, "static toolkit registry is empty — toolkit imports are broken"


def test_every_toolkit_tool_is_classified():
    unclassified = {}
    empty = []
    checked = 0
    for name, toolkit in sorted(AVAILABLE_TOOLKITS.items()):
        if not hasattr(toolkit, "toolkit_config_schema"):
            continue
        schema = toolkit.toolkit_config_schema().model_json_schema()
        selected_tools = (schema.get("properties") or {}).get("selected_tools") or {}
        tool_names = set(selected_tools.get("args_schemas") or {}) or set(
            (selected_tools.get("items") or {}).get("enum") or []
        )
        if not tool_names:
            empty.append(name)
            continue
        checked += 1
        groups = selected_tools.get("tool_groups") or {}
        assert set(groups.values()) <= set(GROUPS), f"{name}: invalid group values {set(groups.values()) - set(GROUPS)}"
        missing = tool_names - set(groups)
        if missing:
            unclassified[name] = sorted(missing)
    assert not unclassified, (
        f"Unclassified tools per toolkit: {unclassified}. Every tool must carry @tool_group "
        "on the method that implements it (overrides do not inherit the marker), and the "
        "toolkit's schema builder must emit tool_groups next to args_schemas."
    )
    from elitea_sdk.tools import FAILED_IMPORTS

    assert checked > 30, (
        f"expected the full static registry, checked only {checked}; registry size {len(AVAILABLE_TOOLKITS)}; "
        f"toolkits with no enumerable tools: {empty}; failed imports: {dict(FAILED_IMPORTS)}"
    )
