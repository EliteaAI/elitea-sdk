import pytest

from elitea_sdk.tools.github import EliteAGitHubToolkit
from elitea_sdk.tools.utils.tool_groups import GROUPS, resolve_declared_groups, with_tool_groups


class FakeProducer:
    class ToolGroups:
        read = ["get_thing"]
        execute = ["mystery_dispatch"]

    @with_tool_groups
    def get_available_tools(self):
        return [
            {"name": "get_thing"},
            {"name": "mystery_dispatch"},
            {"name": "frobnicate_widget"},
            {"name": "delete_thing", "group": "read"},
        ]


def test_stamping_applies_declared_groups():
    tools = {t["name"]: t for t in FakeProducer().get_available_tools()}
    assert tools["get_thing"]["group"] == "read"
    assert tools["mystery_dispatch"]["group"] == "execute"


def test_stamping_never_guesses_and_never_overwrites():
    tools = {t["name"]: t for t in FakeProducer().get_available_tools()}
    assert "group" not in tools["frobnicate_widget"]
    assert tools["delete_thing"]["group"] == "read"


def test_misspelled_group_attribute_is_rejected():
    class Producer:
        class ToolGroups:
            reed = ["get_thing"]

    with pytest.raises(ValueError, match="reed is not a valid group"):
        resolve_declared_groups(Producer)


def test_bare_string_declaration_is_rejected():
    class Producer:
        class ToolGroups:
            read = "get_thing"

    with pytest.raises(ValueError, match="must be a collection of tool names"):
        resolve_declared_groups(Producer)


def test_non_string_entry_is_rejected():
    class Producer:
        class ToolGroups:
            read = [len]

    with pytest.raises(ValueError, match="entries must be tool-name strings"):
        resolve_declared_groups(Producer)


def test_same_tool_in_two_groups_is_rejected():
    class Producer:
        class ToolGroups:
            read = ["ambiguous_tool"]
            write = ["ambiguous_tool"]

    with pytest.raises(ValueError, match="'ambiguous_tool' in both"):
        resolve_declared_groups(Producer)


def test_set_declarations_are_accepted():
    class Producer:
        class ToolGroups:
            read = {"get_thing"}

    assert resolve_declared_groups(Producer)["get_thing"] == "read"


def test_every_toolkit_schema_builds():
    from elitea_sdk.tools import AVAILABLE_TOOLKITS

    built = 0
    for toolkit in AVAILABLE_TOOLKITS.values():
        if not hasattr(toolkit, "toolkit_config_schema"):
            continue
        schema = toolkit.toolkit_config_schema().model_json_schema()
        assert isinstance(schema, dict)
        built += 1
    assert built > 40, f"expected the full static toolkit registry, built only {built}"


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
    assert groups["delete_branch"] == "delete"
    assert groups["list_project_issues"] == "read"
    assert groups["generic_github_api_call"] == "execute"


def test_unlisted_tools_stay_unresolved():
    selected_tools = get_selected_tools_schema()
    unclassified = set(selected_tools["args_schemas"]) - set(selected_tools["tool_groups"])
    assert unclassified == {"get_me", "apply_git_patch_from_file"}, (
        "Every github tool must be classified in the ToolGroups declaration of the class "
        "that declares it (GitHubClient / GraphQLClientWrapper / the indexer base). "
        "get_me and apply_git_patch_from_file are unlisted on purpose while prototyping."
    )
