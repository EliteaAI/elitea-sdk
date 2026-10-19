"""
Issue #6810: a configured base URL with a trailing slash must behave like the
same URL without one.

Toolkits join the base URL with paths that start with '/', so an unnormalized
'https://host/' produced 'https://host//rest/...' (or a client that rejected it).
"""
from unittest.mock import MagicMock, patch

import pytest

from elitea_sdk.tools.utils import normalize_base_url, normalize_host


class TestNormalizeBaseUrl:
    @pytest.mark.parametrize('raw, expected', [
        ('https://jiraeu.epam.com/', 'https://jiraeu.epam.com'),
        ('https://jiraeu.epam.com//', 'https://jiraeu.epam.com'),
        ('  https://jiraeu.epam.com/  ', 'https://jiraeu.epam.com'),
        ('https://github.example.com/api/v3/', 'https://github.example.com/api/v3'),
        ('https://jiraeu.epam.com', 'https://jiraeu.epam.com'),
        ('', ''),
    ])
    def test_strips_trailing_slashes(self, raw, expected):
        assert normalize_base_url(raw) == expected

    @pytest.mark.parametrize('raw', [None, 123])
    def test_non_strings_pass_through(self, raw):
        assert normalize_base_url(raw) == raw

    @pytest.mark.parametrize('raw, expected', [
        ('rally1.rallydev.com', 'rally1.rallydev.com'),
        ('https://rally1.rallydev.com/', 'rally1.rallydev.com'),
        ('HTTP://rally1.rallydev.com', 'rally1.rallydev.com'),
        (' rally1.rallydev.com/ ', 'rally1.rallydev.com'),
    ])
    def test_normalize_host(self, raw, expected):
        assert normalize_host(raw) == expected


# --- High priority -----------------------------------------------------------

class TestJira:
    def test_wrapper_and_client_get_normalized_base_url(self):
        from elitea_sdk.tools.jira import api_wrapper as jira_module

        registry = MagicMock()
        registry.get_client.side_effect = lambda **kw: kw['factory']()
        with patch.object(jira_module, 'JiraClient') as client_cls, \
                patch('elitea_sdk.tools.client_registry.get_client_registry', return_value=registry):
            wrapper = jira_module.JiraApiWrapper(
                base_url='https://jiraeu.epam.com/', username='user', api_key='key',
                cloud=False, api_version='2', llm=None,
            )

        assert wrapper.base_url == 'https://jiraeu.epam.com'
        assert client_cls.call_args.kwargs['url'] == 'https://jiraeu.epam.com'
        assert registry.get_client.call_args.kwargs['base_url'] == 'https://jiraeu.epam.com'

    def test_check_connection_hits_single_slash_endpoint(self):
        from elitea_sdk.configurations.jira import JiraConfiguration

        response = MagicMock(status_code=200)
        response.json.return_value = {'name': 'user'}
        with patch('requests.get', return_value=response) as get:
            JiraConfiguration.check_connection({
                'base_url': 'https://jiraeu.epam.com/', 'username': 'user', 'api_key': 'key',
                'hosting': 'Server', 'api_version': '2',
            })

        assert get.call_args.args[0] == 'https://jiraeu.epam.com/rest/api/2/myself'


class TestGitHub:
    def _build(self, base_url):
        from elitea_sdk.tools.github import api_wrapper as gh_module

        with patch.object(gh_module, 'GitHubClient') as client_cls, \
                patch.object(gh_module, 'GraphQLClientWrapper'), \
                patch.object(gh_module.CodeIndexerToolkit, "validate_toolkit",
                             side_effect=lambda values: values, create=True):
            client_cls.clean_repository_name.side_effect = lambda name: name
            try:
                gh_module.EliteAGitHubAPIWrapper(
                    base_url=base_url, access_token='t', github_repository='org/repo',
                )
            except Exception:
                # Only the auth config handed to GitHubClient matters here
                pass
        return client_cls.call_args.kwargs['auth_config'].github_base_url

    def test_enterprise_base_url_trailing_slash_is_stripped(self):
        assert self._build('https://github.example.com/api/v3/') == 'https://github.example.com/api/v3'

    def test_unchanged_without_trailing_slash(self):
        assert self._build('https://github.example.com/api/v3') == 'https://github.example.com/api/v3'

    def test_check_connection_hits_single_slash_endpoint(self):
        from elitea_sdk.configurations.github import GithubConfiguration

        with patch('requests.get', return_value=MagicMock(status_code=200)) as get:
            assert GithubConfiguration.check_connection({
                'base_url': 'https://github.example.com/api/v3/', 'access_token': 't',
            }) is None

        assert get.call_args.args[0] == 'https://github.example.com/api/v3/user'


class TestSalesforce:
    def test_oauth_and_rest_paths_use_single_slash(self):
        from elitea_sdk.tools.salesforce.api_wrapper import SalesforceApiWrapper

        wrapper = SalesforceApiWrapper(
            base_url='https://company.my.salesforce.com/', client_id='id', client_secret='secret',
        )
        assert wrapper.base_url == 'https://company.my.salesforce.com'

        token_response = MagicMock()
        token_response.json.return_value = {'access_token': 'a'}
        with patch('requests.post', return_value=token_response) as post:
            wrapper.authenticate()
        assert post.call_args.args[0] == 'https://company.my.salesforce.com/services/oauth2/token'


class TestSonar:
    @pytest.mark.parametrize('url', ['https://sonar.example.com/', 'https://sonar.example.com'])
    @pytest.mark.parametrize('relative_url', ['/api/issues/search', 'api/issues/search'])
    def test_get_sonar_data_builds_single_slash_url(self, url, relative_url):
        from elitea_sdk.tools.code.sonar.api_wrapper import SonarApiWrapper

        wrapper = SonarApiWrapper(url=url, sonar_token='t', sonar_project_name='proj')
        assert wrapper.url == 'https://sonar.example.com'

        session = MagicMock()
        with patch.object(SonarApiWrapper, '_client', session):
            wrapper.get_sonar_data(relative_url)
        assert session.get.call_args.kwargs['url'] == 'https://sonar.example.com/api/issues/search'


# --- Follow-up scope ---------------------------------------------------------

class TestAzureDevOps:
    @pytest.mark.parametrize('module_path, class_name', [
        ('elitea_sdk.tools.ado.work_item.ado_wrapper', 'AzureDevOpsApiWrapper'),
        ('elitea_sdk.tools.ado.wiki.ado_wrapper', 'AzureDevOpsApiWrapper'),
        ('elitea_sdk.tools.ado.test_plan.test_plan_wrapper', 'TestPlanApiWrapper'),
    ])
    def test_connection_gets_normalized_org_url(self, module_path, class_name):
        import importlib
        module = importlib.import_module(module_path)
        wrapper_cls = getattr(module, class_name)

        with patch.object(module, 'Connection') as connection_cls:
            try:
                wrapper_cls(organization_url='https://dev.azure.com/my-org/', project='p', token='t')
            except Exception:
                # Downstream init on mocked clients is out of scope; only the URL matters
                pass

        assert connection_cls.call_args_list
        for call in connection_cls.call_args_list:
            assert call.kwargs['base_url'] == 'https://dev.azure.com/my-org'

    def test_repos_git_client_gets_normalized_org_url(self):
        from elitea_sdk.tools.ado.repos import repos_wrapper

        with patch.object(repos_wrapper, 'GitClient') as git_client_cls:
            try:
                repos_wrapper.ReposApiWrapper(
                    organization_url='https://dev.azure.com/my-org/', project='p', token='t',
                )
            except Exception:
                pass

        assert git_client_cls.call_args.kwargs['base_url'] == 'https://dev.azure.com/my-org'


class TestBitbucket:
    @pytest.mark.parametrize('cloud, api_attr', [(False, 'BitbucketServerApi'), (True, 'BitbucketCloudApi')])
    def test_client_gets_normalized_url(self, cloud, api_attr):
        from elitea_sdk.tools.bitbucket import api_wrapper as bb_module

        with patch.object(bb_module, 'BitbucketServerApi') as server_api, \
                patch.object(bb_module, 'BitbucketCloudApi') as cloud_api:
            try:
                bb_module.BitbucketAPIWrapper(
                    url='https://bitbucket.example.com/', username='u', password='p',
                    project='PRJ', repository='repo', cloud=cloud,
                )
            except Exception:
                pass

        api = cloud_api if cloud else server_api
        assert api.call_args.kwargs['url'] == 'https://bitbucket.example.com'


class TestTestRail:
    def test_client_gets_normalized_url(self):
        from elitea_sdk.tools.testrail import api_wrapper as tr_module

        # The validator re-imports TestRailAPI locally, so patch it at the source
        with patch('testrail_api.TestRailAPI') as api_cls:
            try:
                tr_module.TestrailAPIWrapper(url='https://testrail.example.com/', email='e', password='p')
            except Exception:
                pass

        assert api_cls.call_args.args[0] == 'https://testrail.example.com'


class TestServiceNow:
    def test_wrapper_stores_normalized_base_url(self):
        from elitea_sdk.tools.servicenow import api_wrapper as sn_module

        with patch.object(sn_module, 'ServiceNowClient') as client_cls:
            wrapper = sn_module.ServiceNowAPIWrapper(
                base_url='https://company.service-now.com/', username='u', password='p',
            )

        assert wrapper.base_url == 'https://company.service-now.com'
        assert client_cls.call_args.kwargs['instance'] == 'https://company.service-now.com'


class TestRally:
    @pytest.mark.parametrize('server', [
        'rally1.rallydev.com', 'https://rally1.rallydev.com/', 'rally1.rallydev.com/',
    ])
    def test_pyral_gets_bare_host(self, server):
        from elitea_sdk.tools.rally import api_wrapper as rally_module

        with patch.object(rally_module, 'Rally') as rally_cls:
            wrapper = rally_module.RallyApiWrapper(server=server, api_key='k')

        assert rally_cls.call_args.kwargs['server'] == 'rally1.rallydev.com'
        assert wrapper.server == 'rally1.rallydev.com'
