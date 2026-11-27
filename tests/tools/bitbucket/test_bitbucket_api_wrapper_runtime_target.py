"""The configured project/repository are only defaults: every call may name another one.

Resolution is the runtime pair, then the configured one, then an actionable error. A
call on another repository must leave the wrapper's own state alone, and two toolkits
in one process must never share a client.
"""
import inspect
import threading
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from elitea_sdk.tools.bitbucket import api_wrapper as bb_module
from elitea_sdk.tools.bitbucket.api_wrapper import (
    FOREIGN_CLIENT_CACHE_SIZE,
    NO_REPOSITORY_MESSAGE,
    BitbucketAPIWrapper,
)


class FakeClient:
    def __init__(self, name, default_branch='main', branches=('main',), files=None):
        self.name = name
        self._default_branch = default_branch
        self.branches = list(branches)
        self.files = dict(files or {})
        self.created = []
        self.writes = []

    def default_branch(self):
        return self._default_branch

    def list_branches(self):
        return list(self.branches)

    def create_branch(self, branch_name, source):
        self.created.append((branch_name, source))
        self.branches.append(branch_name)

    def get_files_list(self, file_path, branch, recursive):
        return [path for path, ref in self.files if ref == branch]

    def get_file(self, file_path, branch):
        if (file_path, branch) not in self.files:
            raise ToolException(f'{file_path} missing on {branch}')
        return self.files[(file_path, branch)]

    def _write_file(self, file_path, content, branch, commit_message=None):
        self.writes.append((file_path, branch))
        self.files[(file_path, branch)] = content

    def get_pull_requests(self):
        return [self.name]


def _wrapper(configured=None, project='PRJ', repository='repo', foreign=()):
    """A wrapper whose client builder hands out the given fakes per (project, repository)."""
    wrapper = BitbucketAPIWrapper.model_construct(
        url='https://bitbucket.example.com', username='u', password='p',
        project=project, repository=repository, branch='main', cloud=False)
    wrapper._bitbucket = configured if configured is not None else FakeClient(f'{project}/{repository}')
    wrapper._active_branch = 'main'
    builds = []
    clients = {key: client for key, client in foreign}

    def build(project_, repository_):
        builds.append((project_, repository_))
        return clients[(project_, repository_)]

    wrapper.__dict__['_build_client'] = build
    return wrapper, builds


@pytest.fixture
def other():
    return FakeClient('OTHER/svc', default_branch='trunk', branches=('trunk',))


class TestResolution:
    def test_configured_pair_is_the_fallback(self):
        wrapper, _ = _wrapper()
        assert wrapper._resolve_target() == ('PRJ', 'repo')
        assert wrapper._client_for() is wrapper._bitbucket

    def test_runtime_pair_wins(self, other):
        wrapper, builds = _wrapper(foreign=[(('OTHER', 'svc'), other)])
        assert wrapper._resolve_target('OTHER', 'svc') == ('OTHER', 'svc')
        assert wrapper._client_for('OTHER', 'svc') is other
        assert builds == [('OTHER', 'svc')]

    def test_a_partial_pair_takes_the_rest_from_the_configuration(self, other):
        wrapper, _ = _wrapper(foreign=[(('PRJ', 'svc'), other)])
        assert wrapper._resolve_target(repository='svc') == ('PRJ', 'svc')
        assert wrapper._client_for(repository='svc') is other

    def test_another_repository_is_built_once_per_wrapper(self, other):
        wrapper, builds = _wrapper(foreign=[(('OTHER', 'svc'), other)])
        for _ in range(3):
            wrapper._client_for('OTHER', 'svc')
        wrapper._client_for('other', 'SVC')
        assert builds == [('OTHER', 'svc')]

    def test_naming_the_configured_pair_is_not_foreign(self):
        wrapper, builds = _wrapper()
        assert wrapper._client_for('prj', 'REPO') is wrapper._bitbucket
        assert builds == []

    @pytest.mark.parametrize('placeholder', ['', '  ', 'null', 'None', "''", 'undefined'])
    def test_pipeline_placeholders_mean_the_configured_pair(self, placeholder):
        wrapper, _ = _wrapper()
        assert wrapper._resolve_target(placeholder, placeholder) == ('PRJ', 'repo')

    def test_no_repository_anywhere_is_actionable(self):
        wrapper, _ = _wrapper(project=None, repository=None)
        wrapper._bitbucket = None
        with pytest.raises(ToolException) as error:
            wrapper._resolve_target()
        assert str(error.value) == NO_REPOSITORY_MESSAGE
        with pytest.raises(ToolException, match='No repository specified'):
            wrapper.get_pull_requests()

    def test_half_a_pair_with_nothing_configured_is_actionable(self):
        wrapper, _ = _wrapper(project=None, repository=None)
        with pytest.raises(ToolException, match='No repository specified'):
            wrapper._client_for(repository='svc')

    @pytest.mark.parametrize('project, repository', [
        ('PRJ', '../../../admin/groups'),
        ('..', 'repo'),
        ('PRJ', 'repo/../../users'),
        ('PRJ', 'repo?limit=1'),
    ])
    def test_a_target_that_is_not_a_path_segment_is_rejected(self, project, repository):
        wrapper, builds = _wrapper()
        with pytest.raises(ToolException, match='Invalid'):
            wrapper._client_for(project, repository)
        assert builds == []

    @pytest.mark.parametrize('project', ['{3f2b8c1e-0d4a-4b6e-9a1f-1c2d3e4f5a6b}', 'my-team', 'PRJ_1'])
    def test_workspace_uuids_and_keys_are_accepted(self, project):
        client = FakeClient('x')
        wrapper, _ = _wrapper(foreign=[((project, 'my.repo'), client)])
        assert wrapper._client_for(project, 'my.repo') is client

    def test_cache_is_bounded(self):
        foreign = [((f'P{i}', 'r'), FakeClient(f'P{i}/r')) for i in range(FOREIGN_CLIENT_CACHE_SIZE + 5)]
        wrapper, _ = _wrapper(foreign=foreign)
        for (project, repository), _client in foreign:
            wrapper._client_for(project, repository)
        assert len(wrapper._foreign_clients) == FOREIGN_CLIENT_CACHE_SIZE
        assert ('p0', 'r') not in wrapper._foreign_clients


class TestToolkitsDoNotShareState:
    def test_two_wrappers_keep_their_own_client_and_branch(self):
        with patch.object(bb_module, 'BitbucketServerApi', side_effect=lambda **kw: MagicMock(name=kw['repository'])):
            first = BitbucketAPIWrapper(url='https://b.example.com', username='u', password='p',
                                        project='A', repository='one', branch='dev', llm=None)
            second = BitbucketAPIWrapper(url='https://b.example.com', username='u', password='p',
                                         project='B', repository='two', branch='main', llm=None)

        assert first._bitbucket is not second._bitbucket
        assert (first._active_branch, second._active_branch) == ('dev', 'main')
        assert 'BitbucketAPIWrapper' not in str(type(BitbucketAPIWrapper.__dict__.get('_bitbucket')))

    @pytest.mark.parametrize('values', [{}, {'project': '', 'repository': 'null'}])
    def test_wrapper_builds_without_a_repository(self, values):
        with patch.object(bb_module, 'BitbucketServerApi') as server_api, \
                patch.object(bb_module, 'BitbucketCloudApi') as cloud_api:
            wrapper = BitbucketAPIWrapper(url='https://bitbucket.org', username='u', password='p',
                                          cloud=True, llm=None, **values)

        assert (wrapper.project, wrapper.repository, wrapper._bitbucket) == (None, None, None)
        server_api.assert_not_called()
        cloud_api.assert_not_called()

    def test_secret_password_reaches_the_client_as_plain_text(self):
        with patch.object(bb_module, 'BitbucketServerApi') as server_api:
            BitbucketAPIWrapper(url='https://b.example.com', username='u', password='p',
                                project='A', repository='one', llm=None)
        assert server_api.call_args.kwargs['password'] == 'p'


class TestBranchStateIsPerRepository:
    def test_set_active_branch_on_another_repository_leaves_the_configured_one(self, other):
        other.branches.append('feature')
        wrapper, _ = _wrapper(foreign=[(('OTHER', 'svc'), other)])

        result = wrapper.set_active_branch('feature', project='OTHER', repository='svc')

        assert 'OTHER/svc' in result
        assert wrapper._active_branch == 'main'
        assert wrapper._active_branch_for('OTHER', 'svc') == 'feature'

    def test_unrecorded_foreign_repository_uses_its_default_branch(self, other):
        wrapper, _ = _wrapper(foreign=[(('OTHER', 'svc'), other)])
        assert wrapper._active_branch_for('OTHER', 'svc') == 'trunk'

    def test_unreadable_default_branch_is_actionable(self):
        broken = FakeClient('OTHER/svc')
        broken.default_branch = MagicMock(side_effect=RuntimeError('404 Not Found'))
        wrapper, _ = _wrapper(foreign=[(('OTHER', 'svc'), broken)])
        with pytest.raises(ToolException, match='default branch of `OTHER/svc`'):
            wrapper.list_files(project='OTHER', repository='svc')

    def test_repository_without_a_default_branch_is_actionable_and_not_cached(self):
        empty = FakeClient('OTHER/svc', default_branch=None)
        wrapper, _ = _wrapper(foreign=[(('OTHER', 'svc'), empty)])
        for _ in range(2):
            with pytest.raises(ToolException, match='no default branch'):
                wrapper._active_branch_for('OTHER', 'svc')
        assert ('other', 'svc') not in (getattr(wrapper, '_foreign_branches', None) or {})

    def test_create_branch_on_another_repository_branches_from_its_default(self, other):
        wrapper, _ = _wrapper(foreign=[(('OTHER', 'svc'), other)])

        wrapper.create_branch('bot', project='OTHER', repository='svc')

        assert other.created == [('bot', 'trunk')]
        assert wrapper._bitbucket.created == []
        assert wrapper._active_branch == 'main'
        assert wrapper._active_branch_for('OTHER', 'svc') == 'bot'

    def test_delete_branch_guards_the_named_repositorys_active_branch(self, other):
        wrapper, _ = _wrapper(foreign=[(('OTHER', 'svc'), other)])
        with pytest.raises(ToolException, match='active'):
            wrapper.delete_branch('trunk', project='OTHER', repository='svc')

    def test_wrapper_fields_survive_foreign_calls(self, other):
        other.branches.append('feature')
        wrapper, _ = _wrapper(foreign=[(('OTHER', 'svc'), other)])

        wrapper.set_active_branch('feature', project='OTHER', repository='svc')
        wrapper.create_branch('bot', project='OTHER', repository='svc')

        assert (wrapper.project, wrapper.repository, wrapper._active_branch) == ('PRJ', 'repo', 'main')

    def test_concurrent_calls_on_different_repositories_do_not_cross(self):
        names = [f'repo{i}' for i in range(8)]
        wrapper, _ = _wrapper(foreign=[(('PRJ', name), FakeClient(name)) for name in names])
        barrier = threading.Barrier(len(names))

        def record(name):
            barrier.wait()
            for round_ in range(50):
                wrapper._record_active_branch(f'{name}-{round_}', 'PRJ', name)

        threads = [threading.Thread(target=record, args=(name,)) for name in names]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert {name: wrapper._active_branch_for('PRJ', name) for name in names} == \
            {name: f'{name}-49' for name in names}
        assert wrapper._active_branch == 'main'


class TestCallsTargetTheNamedRepository:
    def test_read_and_list_use_the_named_repository_on_its_default_branch(self, other):
        other.files[('a.py', 'trunk')] = 'x = 1'
        wrapper, _ = _wrapper(foreign=[(('OTHER', 'svc'), other)])

        assert wrapper.read_file('a.py', branch=None, project='OTHER', repository='svc') == 'x = 1'
        assert 'a.py' in str(wrapper.list_files(project='OTHER', repository='svc'))

    def test_read_multiple_files_forwards_the_target(self, other):
        other.files[('a.py', 'trunk')] = 'x = 1'
        wrapper, _ = _wrapper(foreign=[(('OTHER', 'svc'), other)])

        result = wrapper.read_multiple_files(['a.py'], branch='trunk', project='OTHER', repository='svc')

        assert 'x = 1' in str(result)

    def test_update_file_writes_to_the_named_repository(self, other):
        other.files[('a.txt', 'feature')] = 'Hello Earth!'
        wrapper, _ = _wrapper(foreign=[(('OTHER', 'svc'), other)])

        wrapper.update_file('a.txt', 'OLD <<<<\nHello Earth!\n>>>> OLD\nNEW <<<<\nHello Mars!\n>>>> NEW',
                            branch='feature', project='OTHER', repository='svc')

        assert other.writes == [('a.txt', 'feature')]
        assert wrapper._bitbucket.writes == []

    def test_pull_requests_of_the_named_repository(self, other):
        wrapper, _ = _wrapper(foreign=[(('OTHER', 'svc'), other)])
        assert wrapper.get_pull_requests(project='OTHER', repository='svc') == ['OTHER/svc']
        assert wrapper.get_pull_requests() == ['PRJ/repo']


class TestSchemas:
    def test_every_tool_schema_agrees_with_its_method_on_the_target(self):
        wrapper, _ = _wrapper()
        # Inherited index tools read the target from their index params instead.
        index_tools = {'index_data', 'search_index', 'stepback_search_index',
                       'stepback_summary_index', 'list_collections', 'remove_index'}
        for tool in wrapper.get_available_tools():
            if tool['name'] in index_tools:
                continue
            parameters = inspect.signature(tool['ref']).parameters
            fields = tool['args_schema'].model_fields
            for name in ('project', 'repository'):
                takes = name in parameters or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
                assert takes == (name in fields), (tool['name'], name)

    def test_toolkit_project_and_repository_are_optional(self):
        from elitea_sdk.tools.bitbucket import EliteABitbucketToolkit

        schema = EliteABitbucketToolkit.toolkit_config_schema().model_json_schema()
        assert {'project', 'repository'}.isdisjoint(schema.get('required', []))

    @pytest.mark.parametrize('cloud, target, expected', [
        (True, {}, 'https://api.bitbucket.org/2.0/user'),
        (False, {}, 'https://b.example.com/rest/api/1.0/projects'),
        (True, {'project': 'ws', 'repository': 'r'}, 'https://api.bitbucket.org/2.0/repositories/ws/r'),
        (False, {'project': 'P', 'repository': 'r'}, 'https://b.example.com/rest/api/1.0/projects/P/repos/r'),
    ])
    def test_check_connection_without_a_repository_checks_the_account(self, cloud, target, expected):
        from elitea_sdk.tools.bitbucket import EliteABitbucketToolkit

        schema = EliteABitbucketToolkit.toolkit_config_schema()
        url = 'https://api.bitbucket.org' if cloud else 'https://b.example.com'
        settings = schema.model_construct(
            cloud=cloud, bitbucket_configuration={'url': url, 'username': 'u', 'password': 'p'},
            project=target.get('project'), repository=target.get('repository'))
        response = MagicMock(status_code=200)
        with patch('elitea_sdk.tools.bitbucket.requests.get', return_value=response) as get:
            schema.check_connection(settings)
        assert get.call_args.args[0] == expected

    def test_cloud_is_detected_from_the_configuration_url(self):
        from elitea_sdk.tools.bitbucket import EliteABitbucketToolkit

        with patch.object(bb_module, 'BitbucketCloudApi') as cloud_api, \
                patch.object(bb_module, 'BitbucketServerApi') as server_api:
            EliteABitbucketToolkit.get_toolkit(
                selected_tools=['list_files'], project='ws', repository='r', cloud=None, llm=None,
                bitbucket_configuration={'url': 'https://bitbucket.org', 'username': 'u', 'password': 'p'})
        cloud_api.assert_called_once()
        server_api.assert_not_called()

    def test_toolkit_builds_without_a_repository(self):
        from elitea_sdk.tools.bitbucket import EliteABitbucketToolkit

        toolkit = EliteABitbucketToolkit.get_toolkit(
            selected_tools=['list_files'], cloud=False, llm=None,
            bitbucket_configuration={'url': 'https://b.example.com', 'username': 'u', 'password': 'p'})
        tools = toolkit.get_tools()
        assert [tool.name for tool in tools] == ['list_files']
        assert 'repo:' not in tools[0].description


class TestCloudClient:
    def test_unknown_workspace_is_actionable(self):
        from elitea_sdk.tools.bitbucket import cloud_api_wrapper
        cloud = MagicMock()
        cloud.workspaces.get.side_effect = RuntimeError('404 Client Error')
        with patch.object(cloud_api_wrapper, 'Cloud', return_value=cloud):
            with pytest.raises(ToolException, match="Unable to connect to the repository 'svc'"):
                cloud_api_wrapper.BitbucketCloudApi(
                    url='https://api.bitbucket.org', workspace='typo', repository='svc',
                    username='u', password='p')


class TestIndexTarget:
    def test_index_params_offer_the_target(self):
        params = _wrapper()[0]._index_tool_params()
        assert {'project', 'repository'} <= set(params)

    def test_index_target_resolves_runtime_then_configured(self):
        wrapper, _ = _wrapper()
        assert wrapper._index_target({'project': 'OTHER', 'repository': 'svc'}) == \
            {'project': 'OTHER', 'repository': 'svc'}
        assert wrapper._index_target({}) == {'project': 'PRJ', 'repository': 'repo'}

    def test_index_without_any_repository_fails_before_listing(self):
        wrapper, _ = _wrapper(project=None, repository=None)
        with pytest.raises(ToolException, match='No repository specified'):
            wrapper._index_target({})

    def test_loader_reads_the_target_and_records_it(self, other, monkeypatch):
        other.files[('a.py', 'trunk')] = 'x = 1'
        wrapper, _ = _wrapper(foreign=[(('OTHER', 'svc'), other)])
        monkeypatch.setattr(BitbucketAPIWrapper, '_log_tool_event', lambda self, *a, **kw: None)

        documents = list(wrapper.loader(chunked=False, target={'project': 'OTHER', 'repository': 'svc'}))

        assert [doc.metadata['repository'] for doc in documents] == ['OTHER/svc']

    def test_index_on_the_configured_repository_keeps_its_branch(self):
        configured = FakeClient('PRJ/repo', files={('a.py', 'main'): 'x = 1'})
        wrapper, builds = _wrapper(configured=configured)
        wrapper.__dict__['_log_tool_event'] = lambda *a, **kw: None

        documents = list(wrapper.loader(chunked=False, target={'project': 'PRJ', 'repository': 'repo'}))

        assert [doc.metadata['repository'] for doc in documents] == ['PRJ/repo']
        assert builds == []
