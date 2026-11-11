"""The configured repository is only a default: every call may name another one.

Resolution is runtime repo_name, then the configured repository, then an actionable
error. A call on another repository must leave the wrapper's own state alone.
"""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.github.api_wrapper import EliteAGitHubAPIWrapper
from elitea_sdk.tools.github.github_client import NO_REPOSITORY_MESSAGE, GitHubClient


class FakeFile:
    def __init__(self, content, sha='sha-1'):
        self.decoded_content = content.encode('utf-8')
        self.sha = sha


class FakeRepo:
    def __init__(self, full_name, default_branch='main', branches=('main',)):
        self.full_name = full_name
        self.default_branch = default_branch
        self.branches = set(branches)
        self.files = {}
        self.update_calls = []

    def get_branch(self, name):
        if name not in self.branches:
            raise Exception(f'{name} missing')
        return SimpleNamespace(commit=SimpleNamespace(sha=f'{self.full_name}-{name}-sha'))

    def create_git_ref(self, ref, sha):
        self.branches.add(ref.removeprefix('refs/heads/'))

    def get_contents(self, path, ref=None):
        if (path, ref) not in self.files:
            raise Exception(f'{path} missing on {ref}')
        return self.files[(path, ref)]

    def update_file(self, path, message, content, sha, branch):
        self.update_calls.append({'path': path, 'content': content, 'branch': branch})
        self.files[(path, branch)] = FakeFile(content, sha)


class FakeSearchResults(list):
    totalCount = 0


class FakeGitHubApi:
    def __init__(self, *repos):
        self.repos = {repo.full_name: repo for repo in repos}
        self.requested = []
        self.search_issues = MagicMock(return_value=SimpleNamespace(totalCount=0))
        self.search_code = MagicMock(return_value=FakeSearchResults())

    def get_repo(self, name):
        self.requested.append(name)
        return self.repos[name]

    def get_user(self, login=None):
        return SimpleNamespace(raw_data={'login': login or 'me'})


def _client(api, repository='owner/repo'):
    return GitHubClient.model_construct(
        github_repository=repository,
        active_branch='main',
        github_base_branch='main',
        github_api=api,
        elitea=None,
    )


@pytest.fixture
def repos():
    return FakeRepo('owner/repo'), FakeRepo('other/repo', default_branch='trunk', branches=('trunk',))


class TestResolution:
    def test_configured_repository_is_the_fallback(self, repos):
        assert _client(FakeGitHubApi(*repos))._resolve_repo_name(None) == 'owner/repo'

    def test_runtime_repo_name_wins(self, repos):
        api = FakeGitHubApi(*repos)
        client = _client(api)
        assert client._resolve_repo_name('other/repo') == 'other/repo'
        assert client._get_repo('other/repo') is repos[1]
        assert api.requested == ['other/repo']

    def test_another_repository_is_fetched_once_per_client(self, repos):
        api = FakeGitHubApi(*repos)
        client = _client(api)
        for _ in range(3):
            client._get_repo('other/repo')
        client._get_repo('Other/Repo')
        assert api.requested == ['other/repo']
        assert client.github_repository == 'owner/repo'

    def test_no_repository_anywhere_is_actionable(self, repos):
        client = _client(FakeGitHubApi(*repos), repository=None)
        with pytest.raises(ToolException) as error:
            client._resolve_repo_name(None)
        assert str(error.value) == NO_REPOSITORY_MESSAGE
        with pytest.raises(ToolException):
            client._get_repo(None)

    @pytest.mark.parametrize('placeholder', ['', '  ', 'null', 'None', "''", '"  "', 'undefined'])
    def test_pipeline_placeholders_mean_the_configured_repository(self, repos, placeholder):
        client = _client(FakeGitHubApi(*repos))
        assert client._resolve_repo_name(placeholder) == 'owner/repo'
        assert client._get_repo(placeholder) is repos[0]

    def test_naming_the_configured_repository_is_not_foreign(self, repos):
        client = _client(FakeGitHubApi(*repos))
        assert client._foreign_repo_key('OWNER/repo') is None


class TestBranchStateIsPerRepository:
    def test_set_active_branch_on_another_repository_leaves_the_configured_one(self, repos):
        repos[1].branches.add('feature')
        client = _client(FakeGitHubApi(*repos))

        result = client.set_active_branch('feature', repo_name='other/repo')

        assert 'other/repo' in result
        assert client.active_branch == 'main'
        assert client._active_branch_for('other/repo') == 'feature'
        assert client._active_branch_for(None) == 'main'

    def test_create_branch_on_another_repository_branches_from_its_default(self, repos):
        client = _client(FakeGitHubApi(*repos))

        client.create_branch('bot', repo_name='other/repo')

        assert 'bot' in repos[1].branches and 'bot' not in repos[0].branches
        assert client.active_branch == 'main'
        assert client._active_branch_for('other/repo') == 'bot'

    def test_unrecorded_foreign_repository_uses_its_default_branch(self, repos):
        client = _client(FakeGitHubApi(*repos))
        assert client._active_branch_for('other/repo') == 'trunk'
        assert client._base_branch_for('other/repo') == 'trunk'

    def test_wrapper_fields_survive_foreign_calls(self, repos):
        repos[1].branches.add('feature')
        client = _client(FakeGitHubApi(*repos))

        client.set_active_branch('feature', repo_name='other/repo')
        client.create_branch('bot', repo_name='other/repo')

        assert (client.github_repository, client.active_branch, client.github_base_branch) == \
            ('owner/repo', 'main', 'main')

    def test_concurrent_calls_on_different_repositories_do_not_cross(self):
        names = [f'org/repo{i}' for i in range(8)]
        client = _client(FakeGitHubApi(*(FakeRepo(name) for name in names)))
        barrier = threading.Barrier(len(names))

        def record(name):
            barrier.wait()
            for round_ in range(50):
                client._record_active_branch(f'{name}-{round_}', name)

        threads = [threading.Thread(target=record, args=(name,)) for name in names]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert {name: client._active_branch_for(name) for name in names} == \
            {name: f'{name}-49' for name in names}
        assert client.active_branch == 'main'


class TestWritesTargetTheNamedRepository:
    def test_update_file_writes_to_the_named_repository(self, repos):
        repos[1].files[('a.txt', 'feature')] = FakeFile('Hello Earth!')
        client = _client(FakeGitHubApi(*repos))

        result = client.update_file(
            'a.txt\nOLD <<<<\nHello Earth!\n>>>> OLD\nNEW <<<<\nHello Mars!\n>>>> NEW',
            branch='feature', repo_name='other/repo')

        assert 'Hello Mars!' in repos[1].update_calls[0]['content'], result
        assert repos[0].update_calls == []

    def test_protected_check_uses_the_named_repositorys_default_branch(self, repos):
        client = _client(FakeGitHubApi(*repos))

        result = client.update_file('a.txt\nOLD <<<<\nx\n>>>> OLD\nNEW <<<<\ny\n>>>> NEW',
                                    branch='trunk', repo_name='other/repo')

        assert 'protected' in result
        assert repos[1].update_calls == []


class TestAccountLevelOperations:
    def test_search_issues_without_a_repository_runs_unscoped(self, repos):
        api = FakeGitHubApi(*repos)
        _client(api, repository=None).search_issues('org:acme is:open')
        api.search_issues.assert_called_once_with('org:acme is:open')

    def test_search_issues_scopes_to_the_named_repository(self, repos):
        api = FakeGitHubApi(*repos)
        _client(api).search_issues('is:open', repo_name='other/repo')
        api.search_issues.assert_called_once_with('repo:other/repo is:open')

    def test_generic_account_call_needs_no_repository(self, repos):
        client = _client(FakeGitHubApi(*repos), repository=None)
        assert client.generic_github_api_call('get_user', {'login': 'octo'}) == {'login': 'octo'}

    def test_search_code_without_any_scope_is_actionable(self, repos):
        client = _client(FakeGitHubApi(*repos), repository=None)
        with pytest.raises(ToolException, match="repo_name='owner/repo'"):
            client.search_code('def main')

    def test_search_code_scopes_to_the_named_repository(self, repos):
        api = FakeGitHubApi(*repos)
        _client(api).search_code('def main', repo_name='other/repo')
        assert api.search_code.call_args.kwargs['query'] == 'def main repo:other/repo'

    def test_search_code_keeps_an_explicit_query_scope(self, repos):
        api = FakeGitHubApi(*repos)
        _client(api, repository=None).search_code('def main org:acme')
        assert api.search_code.call_args.kwargs['query'] == 'def main org:acme'


class TestSchemas:
    def test_toolkit_repository_is_optional(self):
        from elitea_sdk.tools.github import EliteAGitHubToolkit

        schema = EliteAGitHubToolkit.toolkit_config_schema().model_json_schema()
        assert 'repository' not in schema.get('required', [])

    def test_every_tool_schema_agrees_with_its_method_on_repo_name(self):
        import inspect

        for tool in _client(FakeGitHubApi()).get_available_tools():
            parameters = inspect.signature(tool['ref']).parameters
            # The shared file operations forward repo_name through **target.
            takes_repo = 'repo_name' in parameters or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
            offers_repo = 'repo_name' in tool['args_schema'].model_fields
            assert takes_repo == offers_repo, tool['name']


class TestIndexTarget:
    def _wrapper(self, client):
        return EliteAGitHubAPIWrapper.model_construct(github_client_instance=client, active_branch='main')

    def test_index_params_offer_repo_name(self):
        assert 'repo_name' in self._wrapper(None)._index_tool_params()

    def test_index_target_resolves_runtime_then_configured(self, repos):
        wrapper = self._wrapper(_client(FakeGitHubApi(*repos)))
        assert wrapper._index_target({'repo_name': 'other/repo'}) == {'repo_name': 'other/repo'}
        assert wrapper._index_target({}) == {'repo_name': 'owner/repo'}

    def test_index_without_any_repository_fails_before_listing(self, repos):
        wrapper = self._wrapper(_client(FakeGitHubApi(*repos), repository=None))
        with pytest.raises(ToolException, match=NO_REPOSITORY_MESSAGE[:30]):
            wrapper._index_target({})

    def test_loader_reads_the_target_and_records_it(self, repos, monkeypatch):
        client = _client(FakeGitHubApi(*repos))
        seen = []
        monkeypatch.setattr(GitHubClient, '_get_files_with_identity',
                            lambda self, path, ref, repo_name=None: seen.append(('list', repo_name)) or {'a.py': ''})
        monkeypatch.setattr(GitHubClient, '_read_file',
                            lambda self, path, branch, repo_name=None, **kw: seen.append(('read', repo_name)) or 'x = 1')
        wrapper = self._wrapper(client)
        monkeypatch.setattr(EliteAGitHubAPIWrapper, '_log_tool_event', lambda self, *a, **kw: None)

        documents = list(wrapper.loader(chunked=False, target={'repo_name': 'other/repo'}))

        assert seen == [('list', 'other/repo'), ('read', 'other/repo')]
        assert documents[0].metadata['repository'] == 'other/repo'

    def test_index_on_another_repository_reads_its_default_branch(self, repos, monkeypatch):
        client = _client(FakeGitHubApi(*repos))
        seen = []
        monkeypatch.setattr(GitHubClient, '_get_files_with_identity',
                            lambda self, path, ref, repo_name=None: seen.append(ref) or {'a.py': ''})
        monkeypatch.setattr(GitHubClient, '_read_file',
                            lambda self, path, branch, repo_name=None, **kw: seen.append(branch) or 'x = 1')
        monkeypatch.setattr(EliteAGitHubAPIWrapper, '_log_tool_event', lambda self, *a, **kw: None)

        list(self._wrapper(client).loader(chunked=False, target={'repo_name': 'other/repo'}))
        list(self._wrapper(client).loader(chunked=False, target={'repo_name': 'owner/repo'}))

        assert seen == ['trunk', 'trunk', 'main', 'main']


class TestBuildWithoutRepository:
    """Validated construction, unlike model_construct above, which skips the field types."""

    @pytest.mark.parametrize('repository', [None, '', 'null'])
    def test_wrapper_builds_without_a_repository(self, repository, monkeypatch):
        # GitHub Actions sets GITHUB_REPOSITORY, which the wrapper reads as a fallback
        monkeypatch.delenv('GITHUB_REPOSITORY', raising=False)
        values = {'access_token': 'x'}
        if repository is not None:
            values['github_repository'] = repository
        wrapper = EliteAGitHubAPIWrapper(**values)
        assert wrapper.github_repository is None
        assert wrapper.github_client_instance.github_repository is None

    def test_toolkit_builds_without_a_repository(self):
        from elitea_sdk.tools.github import EliteAGitHubToolkit

        toolkit = EliteAGitHubToolkit.get_toolkit(
            selected_tools=['get_issues'], github_configuration={'access_token': 'x'})
        assert [tool.name for tool in toolkit.get_tools()]
