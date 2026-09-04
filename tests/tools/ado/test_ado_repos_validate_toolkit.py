from unittest.mock import MagicMock, patch

import pytest
from azure.devops.v7_0.git.git_client import GitClient
from langchain_core.tools import ToolException

from elitea_sdk.tools.ado.repos.repos_wrapper import ReposApiWrapper


def _make_git_client(existing_branches):
    """Build a mock GitClient whose get_branch succeeds only for existing_branches."""
    client = MagicMock(spec=GitClient)
    repository = MagicMock()
    repository.name = "PAL_AprimoSaaS"
    client.get_repository.return_value = repository

    def get_branch(repository_id, name, project):
        if name in existing_branches:
            return MagicMock()
        raise Exception(f"branch '{name}' not found")

    client.get_branch.side_effect = get_branch
    return client


def _base_kwargs(**overrides):
    kwargs = dict(
        llm=MagicMock(),
        organization_url="https://dev.azure.com/org",
        project="1.0 DC Digital",
        token="fake-token",
        repository_id="PAL_AprimoSaaS",
        base_branch="main",
        active_branch="dev",
    )
    kwargs.update(overrides)
    return kwargs


@patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient")
def test_missing_active_branch_falls_back_to_base_branch(mock_git_client_cls):
    mock_git_client_cls.return_value = _make_git_client(existing_branches={"main"})

    wrapper = ReposApiWrapper(**_base_kwargs(base_branch="main", active_branch="dev"))

    assert wrapper.base_branch == "main"
    assert wrapper.active_branch == "main"


@patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient")
def test_missing_base_branch_falls_back_to_active_branch(mock_git_client_cls):
    mock_git_client_cls.return_value = _make_git_client(existing_branches={"dev"})

    wrapper = ReposApiWrapper(**_base_kwargs(base_branch="main", active_branch="dev"))

    assert wrapper.active_branch == "dev"
    assert wrapper.base_branch == "dev"


@patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient")
def test_both_branches_missing_raises_tool_exception(mock_git_client_cls):
    mock_git_client_cls.return_value = _make_git_client(existing_branches=set())

    with pytest.raises(ToolException, match="Neither the base branch 'main' nor the active branch 'dev'"):
        ReposApiWrapper(**_base_kwargs(base_branch="main", active_branch="dev"))


@patch("elitea_sdk.tools.ado.repos.repos_wrapper.GitClient")
def test_both_branches_present_no_fallback(mock_git_client_cls):
    mock_git_client_cls.return_value = _make_git_client(existing_branches={"main", "dev"})

    wrapper = ReposApiWrapper(**_base_kwargs(base_branch="main", active_branch="dev"))

    assert wrapper.base_branch == "main"
    assert wrapper.active_branch == "dev"
