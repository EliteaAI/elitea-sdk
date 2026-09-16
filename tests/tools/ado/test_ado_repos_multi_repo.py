from unittest.mock import MagicMock, patch

import pytest
from azure.devops.v7_0.git.git_client import GitClient
from langchain_core.tools import ToolException

from elitea_sdk.tools.ado.repos.repos_wrapper import ReposApiWrapper


def _named_mock(name):
    """MagicMock whose `.name` is the string (name= is reserved by the constructor)."""
    mock = MagicMock()
    mock.name = name
    return mock


def _repository(name):
    repository = _named_mock(name)
    repository.id = f"{name}-id"
    repository.default_branch = "refs/heads/main"
    return repository


def _make_git_client(branches_by_repo=None):
    """Mock GitClient where every repository exists and exposes the given branches."""
    branches_by_repo = branches_by_repo or {}
    client = MagicMock(spec=GitClient)
    client.get_repository.side_effect = lambda repository_id, project=None: _repository(repository_id)
    client.get_repositories.return_value = [_repository("alpha"), _repository("beta")]

    def get_branch(repository_id, name, project):
        if name in branches_by_repo.get(repository_id, {"main"}):
            return MagicMock()
        raise Exception(f"branch '{name}' not found in '{repository_id}'")

    client.get_branch.side_effect = get_branch
    client.get_branches.side_effect = lambda repository_id, project=None: [
        _named_mock(branch) for branch in branches_by_repo.get(repository_id, {"main"})
    ]
    return client


def _base_kwargs(**overrides):
    kwargs = dict(
        llm=MagicMock(),
        organization_url="https://dev.azure.com/org",
        project="Digital",
        token="fake-token",
        base_branch="main",
        active_branch="main",
    )
    kwargs.update(overrides)
    return kwargs


def _schema_of(wrapper, tool_name):
    for tool in wrapper.get_available_tools():
        if tool["name"] == tool_name:
            return tool["args_schema"]
    raise AssertionError(f"tool '{tool_name}' is not available")


@patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient")
def test_single_repository_as_string_is_kept_working(mock_git_client_cls):
    mock_git_client_cls.return_value = _make_git_client()

    wrapper = ReposApiWrapper(**_base_kwargs(repository_id="alpha"))

    assert wrapper.repositories == ["alpha"]
    assert wrapper.repository_id == "alpha"
    assert wrapper.searchable_repository_name == "alpha"


@patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient")
def test_single_repository_keeps_the_per_call_argument_optional(mock_git_client_cls):
    mock_git_client_cls.return_value = _make_git_client()

    wrapper = ReposApiWrapper(**_base_kwargs(repository_id=["alpha"]))
    schema = _schema_of(wrapper, "read_file").model_json_schema()

    assert "repository_id" not in schema.get("required", [])
    assert wrapper.run("list_branches_in_repo").startswith("Found")


@patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient")
def test_multiple_repositories_require_and_constrain_the_argument(mock_git_client_cls):
    mock_git_client_cls.return_value = _make_git_client()

    wrapper = ReposApiWrapper(**_base_kwargs(repository_id=["alpha", "beta"]))
    schema = _schema_of(wrapper, "read_file").model_json_schema()

    assert "repository_id" in schema["required"]
    assert schema["properties"]["repository_id"]["enum"] == ["alpha", "beta"]
    # Nothing is bound until a call names its repository
    assert wrapper.repository_id is None

    with pytest.raises(ToolException, match="configured with multiple repositories"):
        wrapper.run("list_branches_in_repo")

    with pytest.raises(ToolException, match="not available in this toolkit"):
        wrapper.run("list_branches_in_repo", repository_id="gamma")

    wrapper.run("list_branches_in_repo", repository_id="beta")
    assert wrapper._repo_state["beta"]["searchable_name"] == "beta"


@patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient")
def test_multiple_repositories_are_validated_lazily_per_repository(mock_git_client_cls):
    client = _make_git_client(branches_by_repo={"alpha": {"main"}, "beta": {"release"}})
    mock_git_client_cls.return_value = client

    wrapper = ReposApiWrapper(
        **_base_kwargs(repository_id=["alpha", "beta"], base_branch="main", active_branch="release")
    )
    assert client.get_repository.call_count == 0

    wrapper.run("list_branches_in_repo", repository_id="beta")
    # 'main' is missing in beta, so its base branch falls back to the active one
    assert wrapper._repo_state["beta"]["base_branch"] == "release"
    assert wrapper._repo_state["beta"]["active_branch"] == "release"

    # ... while alpha resolves the other way around, from its own branches
    wrapper.run("list_branches_in_repo", repository_id="alpha")
    assert wrapper._repo_state["alpha"]["base_branch"] == "main"
    assert wrapper._repo_state["alpha"]["active_branch"] == "main"


@patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient")
def test_no_configured_repository_accepts_any_repository_of_the_project(mock_git_client_cls, monkeypatch):
    monkeypatch.delenv("ADO_REPOSITORY_ID", raising=False)
    client = _make_git_client()
    mock_git_client_cls.return_value = client

    wrapper = ReposApiWrapper(**_base_kwargs())
    assert wrapper.repositories == []
    assert client.get_repository.call_count == 0

    schema = _schema_of(wrapper, "read_file").model_json_schema()
    assert "repository_id" in schema["required"]
    assert schema["properties"]["repository_id"]["type"] == "string"

    with pytest.raises(ToolException, match="not configured with any repository"):
        wrapper.run("list_branches_in_repo")

    wrapper.run("list_branches_in_repo", repository_id="anything")
    assert wrapper._repo_state["anything"]["searchable_name"] == "anything"


@patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient")
def test_list_repositories_is_repository_agnostic_and_respects_the_allowlist(mock_git_client_cls):
    mock_git_client_cls.return_value = _make_git_client()

    wrapper = ReposApiWrapper(**_base_kwargs(repository_id=["alpha", "beta"]))
    assert "repository_id" not in _schema_of(wrapper, "list_repositories").model_fields

    assert '"name": "alpha"' in wrapper.run("list_repositories")

    scoped = ReposApiWrapper(**_base_kwargs(repository_id=["beta"]))
    payload = scoped.run("list_repositories")
    assert "beta" in payload and "alpha" not in payload


@patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient")
def test_binding_does_not_leak_between_calls(mock_git_client_cls):
    mock_git_client_cls.return_value = _make_git_client()

    wrapper = ReposApiWrapper(**_base_kwargs(repository_id=["alpha", "beta"]))

    wrapper.run("list_branches_in_repo", repository_id="alpha")
    assert wrapper.repository_id is None

    with pytest.raises(ToolException):
        wrapper.run("list_branches_in_repo", repository_id="gamma")
    assert wrapper.repository_id is None


@patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient")
def test_active_branch_switch_survives_rebinding(mock_git_client_cls):
    mock_git_client_cls.return_value = _make_git_client(
        branches_by_repo={"alpha": {"main", "feature"}, "beta": {"main"}}
    )

    wrapper = ReposApiWrapper(**_base_kwargs(repository_id=["alpha", "beta"]))
    wrapper.run("set_active_branch", branch_name="feature", repository_id="alpha")

    assert wrapper._repo_state["alpha"]["active_branch"] == "feature"
    assert wrapper._repo_state.get("beta") is None


@patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient")
def test_index_data_takes_the_repository_argument(mock_git_client_cls):
    mock_git_client_cls.return_value = _make_git_client()

    wrapper = ReposApiWrapper(**_base_kwargs(repository_id=["alpha", "beta"]))
    schema = _schema_of(wrapper, "index_data").model_json_schema()

    assert "repository_id" in schema["required"]
    assert schema["properties"]["repository_id"]["enum"] == ["alpha", "beta"]
    # Searching stays repository agnostic: one index per repository is the contract
    assert "repository_id" not in _schema_of(wrapper, "search_index").model_fields


def test_toolkit_config_schema_accepts_a_legacy_string_repository():
    from elitea_sdk.tools.ado.repos import AzureDevOpsReposToolkit

    config_model = AzureDevOpsReposToolkit.toolkit_config_schema()
    json_schema = config_model.model_json_schema()

    assert json_schema["properties"]["repository_id"]["type"] == "array"
    assert "repository_id" not in json_schema["required"]
    assert config_model(project="p", repository_id="alpha").repository_id == ["alpha"]
    assert config_model(project="p").repository_id == []
