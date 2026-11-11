"""Each ADO toolkit treats its configured project as a default a tool call may override.

Calls go through the toolkit's tools (`tool.invoke`) so the argument schema, the
placeholder handling and the wrapper's `run` are all on the path.
"""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from azure.devops.v7_0.git.git_client import GitClient
from langchain_core.tools import ToolException

from elitea_sdk.tools.ado.repos import AzureDevOpsReposToolkit
from elitea_sdk.tools.ado.repos import repos_wrapper as repos_module
from elitea_sdk.tools.ado.test_plan import AzureDevOpsPlansToolkit
from elitea_sdk.tools.ado.test_plan import test_plan_wrapper as test_plan_module
from elitea_sdk.tools.ado.wiki import AzureDevOpsWikiToolkit
from elitea_sdk.tools.ado.wiki import ado_wrapper as wiki_module
from elitea_sdk.tools.ado.work_item import AzureDevOpsWorkItemsToolkit
from elitea_sdk.tools.ado.work_item import ado_wrapper as work_item_module

ADO_CONFIGURATION = {"organization_url": "https://dev.azure.com/org", "token": "fake-token"}


def _tools(toolkit_class, clients, project="Digital", **settings):
    """The toolkit's tools by name, built with the ADO client factories in `clients` mocked."""
    with ExitStack() as stack:
        for module, factory in clients:
            stack.enter_context(patch.object(module, factory))
        toolkit = toolkit_class.get_toolkit(
            ado_configuration=ADO_CONFIGURATION, project=project, toolkit_name="ado", llm=None, **settings)
    return {tool.name: tool for tool in toolkit.get_tools()}


def _index_schema(wrapper):
    # Without a vector store the toolkit drops its index tools, so read the wrapper's own list
    return next(t["args_schema"] for t in wrapper.get_available_tools() if t["name"] == "index_data")


WORK_ITEM_CLIENTS = [(work_item_module, "Connection")]
WIKI_CLIENTS = [(wiki_module, "Connection")]
# A test plan toolkit also builds a work item wrapper of its own
TEST_PLAN_CLIENTS = [(test_plan_module, "Connection"), (work_item_module, "Connection")]


def _work_item_tools(project="Digital"):
    tools = _tools(AzureDevOpsWorkItemsToolkit, WORK_ITEM_CLIENTS, project)
    client = tools["get_work_item"].api_wrapper._client
    client.get_work_item.return_value = SimpleNamespace(id=1, fields={"System.Title": "t"}, relations=None)
    return tools, client


class TestWorkItems:

    def test_a_call_targets_the_project_it_names(self):
        tools, client = _work_item_tools()

        tools["get_work_item"].invoke({"id": 1, "project": "Other"})

        assert client.get_work_item.call_args.kwargs["project"] == "Other"

    @pytest.mark.parametrize("project", [None, "", "null", "None"])
    def test_a_call_without_a_project_uses_the_configured_one(self, project):
        tools, client = _work_item_tools()

        tools["get_work_item"].invoke({"id": 1, "project": project})

        assert client.get_work_item.call_args.kwargs["project"] == "Digital"

    def test_the_override_does_not_leak_into_the_next_call(self):
        tools, client = _work_item_tools()

        tools["get_work_item"].invoke({"id": 1, "project": "Other"})
        tools["get_work_item"].invoke({"id": 1})

        assert client.get_work_item.call_args.kwargs["project"] == "Digital"

    def test_a_toolkit_without_a_project_asks_for_one(self):
        tools, _ = _work_item_tools(project=None)

        with pytest.raises(ToolException, match="No Azure DevOps project specified"):
            tools["get_work_item"].api_wrapper.run("get_work_item", id=1)

    def test_a_toolkit_without_a_project_works_once_the_call_names_one(self):
        tools, client = _work_item_tools(project=None)

        tools["get_work_item"].invoke({"id": 1, "project": "Other"})

        assert client.get_work_item.call_args.kwargs["project"] == "Other"

    def test_the_project_argument_comes_last(self):
        tools, _ = _work_item_tools()

        for name, tool in tools.items():
            if name in {"list_indexes", "search_index", "stepback_search_index",
                        "stepback_summary_index", "remove_index"}:
                assert "project" not in tool.args_schema.model_fields, name
            else:
                assert list(tool.args_schema.model_fields)[-1] == "project", name

    def test_type_fields_are_cached_per_project(self):
        tools, _ = _work_item_tools()
        wrapper = tools["get_work_item"].api_wrapper
        wrapper._get_work_item_type_fields = MagicMock(side_effect=lambda work_item_type: {"project": wrapper._project()})
        wrapper._format_work_item_type_fields_for_display = lambda work_item_type, fields: fields["project"]

        first = wrapper.run("get_work_item_type_fields", work_item_type="Bug")
        other = wrapper.run("get_work_item_type_fields", work_item_type="Bug", project="Other")

        assert (first, other) == ("Digital", "Other")

    def test_index_documents_record_their_project_and_fetch_in_it_on_every_worker(self):
        tools, client = _work_item_tools()
        wrapper = tools["get_work_item"].api_wrapper
        client.query_by_wiql.return_value = SimpleNamespace(
            work_items=[SimpleNamespace(id=i) for i in range(1, 5)], work_item_relations=None)
        client.get_work_item.side_effect = lambda id, project, expand: SimpleNamespace(
            id=id, fields={"System.Title": "t"}, relations=None)

        with wrapper._project_scope("Other"):
            loader = wrapper._base_loader(wiql="SELECT [System.Id] FROM workitems", workers=4, project="Other")
        documents = list(loader)

        assert [d.metadata["project"] for d in documents] == ["Other"] * 4
        assert {call.kwargs["project"] for call in client.get_work_item.call_args_list} == {"Other"}

    def test_the_index_tool_records_the_project(self):
        tools, _ = _work_item_tools()

        assert "project" in _index_schema(tools["get_work_item"].api_wrapper).model_fields


PROJECT_IDS = {"digital-guid": "Digital"}


def _repos_tools(project="Digital", repository_id=("alpha",)):
    client = MagicMock(spec=GitClient)
    client.get_repository.side_effect = lambda repository_id, project=None: SimpleNamespace(
        name=repository_id, id=f"{repository_id}-id", default_branch="refs/heads/main",
        project=SimpleNamespace(name=PROJECT_IDS.get(project, project)))
    with patch.object(repos_module, "GitClient", return_value=client):
        tools = _tools(AzureDevOpsReposToolkit, [], project, repository_id=list(repository_id))
    client.get_branches.return_value = [SimpleNamespace(name="main")]
    return tools, client


class TestRepos:

    def test_the_configured_repository_is_the_default_in_the_configured_project(self):
        tools, client = _repos_tools()

        tools["list_branches_in_repo"].invoke({})

        assert client.get_branches.call_args.kwargs == {"repository_id": "alpha", "project": "Digital"}

    def test_another_project_needs_a_repository_of_its_own(self):
        tools, _ = _repos_tools()

        with pytest.raises(ToolException, match="do not belong to project 'Other'"):
            tools["list_branches_in_repo"].api_wrapper.run("list_branches_in_repo", project="Other")

    def test_the_allowlist_covers_only_the_configured_project(self):
        tools, client = _repos_tools()
        wrapper = tools["list_branches_in_repo"].api_wrapper

        with pytest.raises(ToolException, match="not available in this toolkit"):
            wrapper.run("list_branches_in_repo", repository_id="beta")
        wrapper.run("list_branches_in_repo", project="Other", repository_id="beta")

        assert client.get_branches.call_args.kwargs == {"repository_id": "beta", "project": "Other"}

    @pytest.mark.parametrize("project", ["digital", "DIGITAL", "digital-guid"])
    def test_the_allowlist_holds_however_the_call_names_the_configured_project(self, project):
        tools, client = _repos_tools()
        wrapper = tools["list_branches_in_repo"].api_wrapper

        with pytest.raises(ToolException, match="not available in this toolkit"):
            wrapper.run("list_branches_in_repo", project=project, repository_id="beta")
        wrapper.run("list_branches_in_repo", project=project, repository_id="alpha")

        assert client.get_branches.call_args.kwargs["repository_id"] == "alpha"

    def test_repositories_without_a_project_are_rejected(self):
        with pytest.raises(ToolException, match="only be configured together with a project"):
            _repos_tools(project=None)

    def test_a_call_in_another_project_restores_the_default_repository(self):
        tools, client = _repos_tools()
        wrapper = tools["list_branches_in_repo"].api_wrapper

        wrapper.run("list_branches_in_repo", project="Other", repository_id="beta")
        wrapper.run("list_branches_in_repo")

        assert client.get_branches.call_args.kwargs == {"repository_id": "alpha", "project": "Digital"}
        assert wrapper.repository_id == "alpha"

    def test_branch_state_is_kept_per_project(self):
        tools, _ = _repos_tools()
        wrapper = tools["list_branches_in_repo"].api_wrapper

        wrapper.run("set_active_branch", project="Other", repository_id="alpha", branch_name="main")

        assert ("Other", "alpha") in wrapper._repo_state
        assert ("Digital", "alpha") in wrapper._repo_state

    def test_file_cache_keys_name_the_project(self):
        tools, _ = _repos_tools()
        wrapper = tools["list_branches_in_repo"].api_wrapper

        with wrapper._project_scope("Other"):
            assert wrapper._file_cache_key("/a.py", None)[0] == "Other"

    def test_list_repositories_needs_no_configured_repository(self):
        tools, client = _repos_tools(project=None, repository_id=())
        client.get_repositories.return_value = []

        tools["list_repositories"].invoke({"project": "Other"})

        assert client.get_repositories.call_args.kwargs == {"project": "Other"}

    def test_index_data_keeps_both_targets_as_index_parameters(self):
        tools, _ = _repos_tools()
        fields = _index_schema(tools["list_branches_in_repo"].api_wrapper).model_fields

        assert {"project", "repository_id"} <= set(fields)


class TestWiki:

    def test_a_call_targets_the_project_it_names(self):
        tools = _tools(AzureDevOpsWikiToolkit, WIKI_CLIENTS)
        client = tools["get_wiki"].api_wrapper._client

        tools["get_wiki"].invoke({"wiki_identified": "w", "project": "Other"})
        tools["get_wiki"].invoke({"wiki_identified": "w"})

        projects = [call.kwargs["project"] for call in client.get_wiki.call_args_list]
        assert projects == ["Other", "Digital"]

    def test_the_wiki_repository_wrapper_is_cached_per_project(self):
        tools = _tools(AzureDevOpsWikiToolkit, WIKI_CLIENTS)
        wrapper = tools["get_wiki"].api_wrapper
        wrapper._client.get_wiki.return_value = SimpleNamespace(repository_id="repo")

        with patch.object(wiki_module, "ReposApiWrapper", side_effect=lambda **kwargs: kwargs["project"]):
            with wrapper._project_scope("Other"):
                other = wrapper._get_repos_wrapper("w")
            default = wrapper._get_repos_wrapper("w")

        assert (other, default) == ("Other", "Digital")


class TestTestPlans:

    def test_toolkits_do_not_share_a_work_item_wrapper(self):
        first = _tools(AzureDevOpsPlansToolkit, TEST_PLAN_CLIENTS, "A")["get_test_plan"].api_wrapper
        second = _tools(AzureDevOpsPlansToolkit, TEST_PLAN_CLIENTS, "B")["get_test_plan"].api_wrapper

        assert first._work_item_wrapper is not second._work_item_wrapper
        assert first._client is not second._client

    def test_a_call_targets_the_project_it_names(self):
        tools = _tools(AzureDevOpsPlansToolkit, TEST_PLAN_CLIENTS)
        client = tools["get_test_plan"].api_wrapper._client

        tools["get_test_plan"].invoke({"plan_id": 3, "project": "Other"})

        assert client.get_test_plan_by_id.call_args.args == ("Other", 3)

    def test_delegated_work_item_calls_follow_the_call_project(self):
        tools = _tools(AzureDevOpsPlansToolkit, TEST_PLAN_CLIENTS)
        wrapper = tools["get_test_case"].api_wrapper
        wrapper._client.get_test_case.return_value = [
            SimpleNamespace(as_dict=lambda: {"work_item": {"id": 9}})]
        work_items = wrapper._work_item_wrapper._client
        work_items.get_work_item.return_value = SimpleNamespace(id=9, fields={}, relations=None)

        tools["get_test_case"].invoke({"plan_id": 1, "suite_id": 2, "test_case_id": "9", "project": "Other"})

        assert wrapper._client.get_test_case.call_args.args[0] == "Other"
        assert work_items.get_work_item.call_args.kwargs["project"] == "Other"

    def test_a_toolkit_without_a_project_can_still_be_built(self):
        tools = _tools(AzureDevOpsPlansToolkit, TEST_PLAN_CLIENTS, project=None)

        with pytest.raises(ToolException, match="No Azure DevOps project specified"):
            tools["get_test_plan"].api_wrapper.run("get_test_plan", plan_id=1)
