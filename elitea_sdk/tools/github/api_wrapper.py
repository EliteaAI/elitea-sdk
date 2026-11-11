import logging
from typing import Any, Dict, Optional

from pydantic import model_validator, Field, SecretStr

from .github_client import GitHubClient
from .graphql_client_wrapper import GraphQLClientWrapper
from .schemas import (
    GitHubAuthConfig,
    GitHubRepoConfig
)
from ..code_indexer_toolkit import CodeIndexerToolkit
from ..utils.available_tools_decorator import extend_with_parent_available_tools
from ..utils import normalize_base_url
from ..utils.tool_groups import with_tool_groups

logger = logging.getLogger(__name__)

# Import prompts for tools

# Create schema models for the new indexing functionality


class EliteAGitHubAPIWrapper(CodeIndexerToolkit):
    """
    Wrapper for GitHub API that integrates both REST and GraphQL functionality.
    """
    # Authentication config
    github_access_token: Optional[SecretStr] = None
    github_username: Optional[SecretStr] = None
    github_password: Optional[SecretStr] = None
    github_app_id: Optional[str] = None
    github_app_private_key: Optional[str] = None
    github_base_url: Optional[str] = None

    # Repository config
    github_repository: Optional[str] = None
    active_branch: Optional[str] = None
    github_base_branch: Optional[str] = None

    # EliteA instance
    elitea: Optional[Any] = None
    
    # Client instances - renamed without leading underscores and marked as exclude=True
    github_client_instance: Optional[GitHubClient] = Field(default=None, exclude=True)
    graphql_client_instance: Optional[GraphQLClientWrapper] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Initialize GitHub clients based on the provided values.

        Args:
            values (Dict): Configuration values for GitHub API wrapper

        Returns:
            Dict: Updated values dictionary
        """
        from langchain_core.utils import get_from_dict_or_env

        # Get all authentication values
        github_access_token = get_from_dict_or_env(values, ["access_token", "github_access_token"], "GITHUB_ACCESS_TOKEN", default='')
        github_username = get_from_dict_or_env(values, ["username", "github_username"], "GITHUB_USERNAME", default='')
        github_password = get_from_dict_or_env(values, ["password", "github_password"], "GITHUB_PASSWORD", default='')
        github_app_id = get_from_dict_or_env(values, ["app_id", "github_app_id"], "GITHUB_APP_ID", default='')
        github_app_private_key = get_from_dict_or_env(values, ["app_private_key", "github_app_private_key"], "GITHUB_APP_PRIVATE_KEY", default='')
        github_base_url = get_from_dict_or_env(values, ["base_url", "github_base_url"], "GITHUB_BASE_URL", default='https://api.github.com')
        github_base_url = normalize_base_url(github_base_url) or 'https://api.github.com'

        auth_config = GitHubAuthConfig(
            github_access_token=github_access_token,
            github_username=github_username,
            github_password=github_password,
            github_app_id=github_app_id,  # This will be None if not provided - GitHubAuthConfig should allow this
            github_app_private_key=github_app_private_key,
            github_base_url=github_base_url
        )

        # The repository is only a default: tools and index runs can name another one per call.
        github_repository = GitHubClient._normalize_repo_name(
            get_from_dict_or_env(values, "github_repository", "GITHUB_REPOSITORY", default=""))
        github_repository = GitHubClient.clean_repository_name(github_repository) if github_repository else None

        repo_config = GitHubRepoConfig(
            github_repository=github_repository,
            active_branch=get_from_dict_or_env(values, "active_branch", "ACTIVE_BRANCH", default='main'),  # Change from 'ai' to 'main'
            github_base_branch=get_from_dict_or_env(values, "github_base_branch", "GITHUB_BASE_BRANCH", default="main")
        )

        # Initialize GitHub client with keyword arguments
        github_client = GitHubClient(auth_config=auth_config, repo_config=repo_config, elitea=values.get("elitea"))
        # Initialize GraphQL client with keyword argument
        graphql_client = GraphQLClientWrapper(github_graphql_instance=github_client.github_api._Github__requester)
        # Set client attributes on the class (renamed from _github_client to github_client_instance)
        values["github_client_instance"] = github_client
        values["graphql_client_instance"] = graphql_client

        # Update values
        values["github_repository"] = github_repository
        values["active_branch"] = repo_config.active_branch
        values["github_base_branch"] = repo_config.github_base_branch

        # Ensure LLM is available in values if needed
        if "llm" not in values:
            values["llm"] = None

        return super().validate_toolkit(values)

    # Expose GitHub REST client methods directly via property
    @property
    def github_client(self) -> GitHubClient:
        """Access to GitHub REST client methods"""
        return self.github_client_instance

    # Expose GraphQL client methods directly via property
    @property
    def graphql_client(self) -> GraphQLClientWrapper:
        """Access to GitHub GraphQL client methods"""
        return self.graphql_client_instance

    @with_tool_groups
    @extend_with_parent_available_tools
    def get_available_tools(self):
        # this is horrible, I need to think on something better
        if not self.github_client_instance:
            github_tools = GitHubClient.model_construct().get_available_tools()
        else:
            github_tools = self.github_client_instance.get_available_tools()
        if not self.graphql_client_instance:
            graphql_tools = GraphQLClientWrapper.model_construct().get_available_tools()
        else:
            graphql_tools = self.graphql_client_instance.get_available_tools()
        
        return github_tools + graphql_tools

    def _index_tool_params(self):
        return {
            **super()._index_tool_params(),
            "repo_name": (Optional[str], Field(
                default=None,
                description="Repository to index in format 'owner/repo'. Leave empty to use the toolkit's "
                            "repository; required only when the toolkit has no repository configured.")),
        }

    def _index_target(self, index_kwargs: dict) -> Dict[str, str]:
        if not self.github_client_instance:
            raise ValueError("GitHub client not initialized")
        # Resolved before any listing, so an index without a repository fails with the actionable message.
        return {"repo_name": self.github_client_instance._resolve_repo_name(index_kwargs.get("repo_name"))}

    def _index_target_metadata(self, target: Dict[str, str]) -> Dict[str, str]:
        return {"repository": target["repo_name"]} if target.get("repo_name") else {}

    def _index_branch(self, branch: Optional[str], target: Dict[str, str]) -> Optional[str]:
        # The toolkit's branches belong to its own repository; another one is read on its default branch.
        repo_name = target.get("repo_name")
        if branch or not self.github_client_instance or not self.github_client_instance._foreign_repo_key(repo_name):
            return super()._index_branch(branch, target)
        return self.github_client_instance._active_branch_for(repo_name)

    def _branch_for(self, branch: Optional[str], repo_name: Optional[str]) -> str:
        return branch or self.github_client_instance._active_branch_for(repo_name)

    def _get_files(self, path: str = "", branch: str = None, repo_name: Optional[str] = None):
        """Get list of files from GitHub repository."""
        if not self.github_client_instance:
            raise ValueError("GitHub client not initialized")
        
        # Deliberately not the client's _get_files, which stringifies failures for agents:
        # the indexer needs the exception so the provider's type and status survive.
        return list(self.github_client_instance._get_files_with_identity(
            path, self._branch_for(branch, repo_name), repo_name=repo_name))

    def _get_files_with_identity(self, path: str = "", branch: str = None, repo_name: Optional[str] = None):
        if not self.github_client_instance:
            raise ValueError("GitHub client not initialized")

        return self.github_client_instance._get_files_with_identity(
            path, self._branch_for(branch, repo_name), repo_name=repo_name)

    def _file_commit_hash(self, file_path: str, branch: str, repo_name: Optional[str] = None):
        """Get the commit hash of a file in the GitHub repository."""
        if not self.github_client_instance:
            raise ValueError("GitHub client not initialized")

        # Use the GitHub client's method to get commit hash
        return self.github_client_instance._file_commit_hash(
            file_path, self._branch_for(branch, repo_name), repo_name=repo_name)

    def _read_file(self, file_path: str, branch: str, repo_name: Optional[str] = None):
        """Read file content from GitHub repository."""
        if not self.github_client_instance:
            raise ValueError("GitHub client not initialized")
        
        # Use the GitHub client's method to read file
        return self.github_client_instance._read_file(file_path, branch, repo_name=repo_name)

    def run(self, name: str, *args: Any, **kwargs: Any):
        for tool in self.get_available_tools():
            if tool["name"] == name:
                # Handle potential dictionary input for args when only one dict is passed
                if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                     kwargs = args[0]
                     args = () # Clear args
                try:
                    return tool["ref"](*args, **kwargs)
                except TypeError as e:
                     # Attempt to call with kwargs only if args fail and kwargs exist
                     if kwargs and not args:
                         try:
                             return tool["ref"](**kwargs)
                         except TypeError:
                             raise ValueError(f"Argument mismatch for tool '{name}'. Error: {e}") from e
                     else:
                         raise ValueError(f"Argument mismatch for tool '{name}'. Error: {e}") from e
        else:
            raise ValueError(f"Unknown tool name: {name}")
