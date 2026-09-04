"""Read tools hand back native, JSON-serializable data (#6532).

One test per converted toolkit: the payload must survive json.dumps, since the
toolkit test panel classifies a result as JSON only when it is a native
collection (or a string json.loads accepts).
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from elitea_sdk.tools.ado.repos.repos_wrapper import ReposApiWrapper
from elitea_sdk.tools.bitbucket.api_wrapper import BitbucketAPIWrapper
from elitea_sdk.tools.confluence.api_wrapper import ConfluenceAPIWrapper
from elitea_sdk.tools.jira.api_wrapper import JiraApiWrapper
from elitea_sdk.tools.qtest.api_wrapper import QtestApiWrapper
from elitea_sdk.tools.rally.api_wrapper import RallyApiWrapper
from elitea_sdk.tools.testrail.api_wrapper import TestrailAPIWrapper
from elitea_sdk.tools.xray.api_wrapper import XrayApiWrapper
from elitea_sdk.tools.zephyr.api_wrapper import ZephyrV1ApiWrapper
from elitea_sdk.tools.zephyr_scale.api_wrapper import ZephyrScaleApiWrapper


def _bare(wrapper_class):
    return wrapper_class.model_construct()


def _assert_json_ready(result):
    assert isinstance(result, (list, dict)), f"expected native data, got {type(result).__name__}"
    assert json.loads(json.dumps(result, default=str)) is not None


class TestAdoRepos:
    def _wrapper(self, pull_requests, monkeypatch):
        client = MagicMock()
        client.get_pull_requests.return_value = pull_requests
        monkeypatch.setattr(
            ReposApiWrapper, 'parse_pull_requests',
            lambda self, prs: [{'title': 'Fix login', 'id': 7}],
        )
        return ReposApiWrapper.model_construct(ado_client_instance=client, repository_id='r', project='p')

    def test_list_open_pull_requests_returns_native_list(self, monkeypatch):
        wrapper = self._wrapper(['pr-object'], monkeypatch)

        result = wrapper.list_open_pull_requests()

        _assert_json_ready(result)
        assert result == [{'title': 'Fix login', 'id': 7}]

    def test_list_open_pull_requests_returns_empty_list(self, monkeypatch):
        wrapper = self._wrapper([], monkeypatch)

        assert wrapper.list_open_pull_requests() == []


class TestBitbucket:
    def _wrapper(self, paths):
        wrapper = _bare(BitbucketAPIWrapper)
        wrapper.__dict__['_bitbucket'] = SimpleNamespace(
            get_files_list=lambda file_path, branch, recursive: paths
        )
        wrapper.__dict__['_active_branch'] = 'main'
        return wrapper

    def test_list_files_returns_a_flat_list(self):
        """The old body ran ast.literal_eval over what _get_files returned.

        Once _get_files stopped stringifying, literal_eval raised on the native list
        and a bare `except` wrapped it again, yielding [['a.py', 'b.py']].
        """
        result = self._wrapper(['a.py', 'dir/b.py']).list_files()

        _assert_json_ready(result)
        assert result == ['a.py', 'dir/b.py']
        assert not any(isinstance(entry, list) for entry in result)

    def test_list_files_handles_no_files(self):
        assert self._wrapper([]).list_files() == []


class TestJira:
    def test_list_projects_returns_native_list(self):
        wrapper = _bare(JiraApiWrapper)
        projects = [{'key': 'PROJ', 'name': 'Project'}]
        wrapper._get_client = lambda: SimpleNamespace(projects=lambda: projects)
        wrapper._parse_projects = lambda raw: projects

        result = wrapper.list_projects()

        _assert_json_ready(result)
        assert result == projects

    def test_get_remote_links_returns_native_list(self):
        wrapper = _bare(JiraApiWrapper)
        links = [{'id': 1, 'object': {'url': 'https://example'}}]
        wrapper._get_client = lambda: SimpleNamespace(get_issue_remotelinks=lambda key: links)

        result = wrapper.get_remote_links('PROJ-1')

        _assert_json_ready(result)
        assert result == links


class TestQtest:
    def test_search_by_dql_reports_total_and_shown(self, monkeypatch):
        wrapper = QtestApiWrapper.model_construct(no_of_tests_shown_in_dql_search=2)
        found = [{'id': index} for index in range(5)]
        monkeypatch.setattr(
            QtestApiWrapper, '_QtestApiWrapper__perform_search_by_dql',
            lambda self, *args, **kwargs: found, raising=False,
        )

        result = wrapper.search_by_dql(dql="Id = 'TC-1'")

        _assert_json_ready(result)
        assert result['total'] == 5
        assert result['shown'] == 2
        assert result['items'] == found[:2]

    def test_get_modules_returns_native_list(self, monkeypatch):
        wrapper = _bare(QtestApiWrapper)
        module = SimpleNamespace(
            to_dict=lambda: {'id': 1, 'name': 'Module', 'pid': 'MD-1', 'children': None}
        )
        module_api = SimpleNamespace(get_sub_modules_of=lambda project_id, **kwargs: [module])
        monkeypatch.setattr(
            QtestApiWrapper, '_QtestApiWrapper__instantiate_module_api_instance',
            lambda self: module_api, raising=False,
        )
        monkeypatch.setattr(QtestApiWrapper, 'qtest_project_id', 1, raising=False)

        result = wrapper.get_modules()

        _assert_json_ready(result)
        assert result[0]['name'] == 'Module'


class TestTestrail:
    def test_json_output_format_is_parseable(self):
        wrapper = _bare(TestrailAPIWrapper)
        data = [{'id': 1, 'title': 'Case'}]

        result = wrapper._to_markup(data, 'json')

        assert json.loads(result) == data
        assert not result.startswith('Extracted data:')

    def test_csv_and_markdown_formats_still_return_text(self):
        wrapper = _bare(TestrailAPIWrapper)
        data = [{'id': 1, 'title': 'Case'}]

        assert 'id,title' in wrapper._to_markup(data, 'csv')
        assert '|' in wrapper._to_markup(data, 'markdown')

    def test_get_case_returns_native_dict(self):
        wrapper = _bare(TestrailAPIWrapper)
        case = {'id': 5, 'title': 'Login works'}
        wrapper._client = SimpleNamespace(cases=SimpleNamespace(get_case=lambda testcase_id: case))

        result = wrapper.get_case('5')

        _assert_json_ready(result)
        assert result == case


class TestZephyr:
    def test_get_test_case_steps_returns_native_list(self):
        wrapper = _bare(ZephyrV1ApiWrapper)
        steps = [{'step': 'open', 'data': '', 'result': 'ok'}]
        wrapper._client = SimpleNamespace(
            get_test_case_steps=lambda issue_id, project_id: SimpleNamespace(json=lambda: {})
        )
        wrapper._parse_test_steps = lambda payload: steps

        result = wrapper.get_test_case_steps(1, 2)

        _assert_json_ready(result)
        assert result == steps


class TestZephyrScale:
    def test_get_tests_returns_native_list(self, monkeypatch):
        wrapper = _bare(ZephyrScaleApiWrapper)
        cases = [{'id': 11, 'key': 'TC-1', 'name': 'Login', 'priority': {'id': 3}}]
        monkeypatch.setattr(
            ZephyrScaleApiWrapper, '_api',
            SimpleNamespace(test_cases=SimpleNamespace(get_test_cases=lambda **kwargs: cases)),
            raising=False,
        )

        result = wrapper.get_tests(project_key='PROJ')

        _assert_json_ready(result)
        assert result == [{
            'id': 11,
            'key': 'TC-1',
            'name': 'Login',
            'project_id': None,
            'precondition': None,
            'priority_id': 3,
            'status_id': None,
            'owner_account_id': None,
        }]


    def test_get_links_works_at_all(self):
        """On main this raised KeyError for every input.

        The body read `kwargs['return_only_links']`, a kwarg its args_schema never
        declared, so `kwargs` was always empty; the surrounding `except Exception`
        reported it as a generic "Unable to get links" ToolException, hiding the cause.
        """
        wrapper = _bare(ZephyrScaleApiWrapper)
        links = {'issues': [{'id': 9, 'issueId': 1001}], 'webLinks': []}
        wrapper.__dict__['_api'] = SimpleNamespace(
            test_cases=SimpleNamespace(get_links=lambda key: links)
        )

        result = wrapper.get_links('TC-1')

        _assert_json_ready(result)
        assert result == links

    def test_get_test_steps_works_at_all(self):
        """On main this raised for every invocation, from two directions.

        The body read `kwargs['return_list']`, which the registered ZephyrGetTestCase
        schema never supplies, so a model call always raised; and the internal indexer
        caller passed return_list=True, which was then forwarded into the API client
        as an unexpected kwarg, so indexing silently lost every test step.
        """
        wrapper = _bare(ZephyrScaleApiWrapper)
        steps = [{'inline': {'description': 'Click *login*'}}]
        seen = {}

        def get_test_steps(key, **kwargs):
            seen.update(kwargs)
            return steps

        wrapper.__dict__['_api'] = SimpleNamespace(
            test_cases=SimpleNamespace(get_test_steps=get_test_steps)
        )

        result = wrapper.get_test_steps('TC-1')

        _assert_json_ready(result)
        assert result == steps
        assert seen == {}

    def test_get_test_script_returns_native_script(self):
        wrapper = _bare(ZephyrScaleApiWrapper)
        script = {'id': 5, 'type': 'plain', 'text': 'do the thing'}
        wrapper.__dict__['_api'] = SimpleNamespace(
            test_cases=SimpleNamespace(get_test_script=lambda key: script)
        )

        result = wrapper.get_test_script('TC-1')

        _assert_json_ready(result)
        assert result == script


    def test_create_test_case_returns_an_envelope(self):
        wrapper = _bare(ZephyrScaleApiWrapper)
        created = {'id': 77, 'test_case_key': 'TC-9'}
        wrapper.__dict__['_api'] = SimpleNamespace(
            test_cases=SimpleNamespace(
                create_test_case=lambda **kwargs: created,
                post_test_steps=lambda *args: {'status': 'ok'},
            )
        )

        result = wrapper.create_test_case('PROJ', 'Login works', {}, steps=[{'description': 'open'}])

        _assert_json_ready(result)
        assert 'was created' in result['message']
        assert result['test_case'] == created
        assert result['steps']['steps'] == {'status': 'ok'}


class TestXray:
    def test_get_tests_returns_native_list(self, monkeypatch):
        wrapper = _bare(XrayApiWrapper)
        results = [{'issueId': '1', 'preconditions': {'total': 0}}]
        monkeypatch.setattr(XrayApiWrapper, 'limit', 2, raising=False)
        monkeypatch.setattr(
            XrayApiWrapper, '_client',
            SimpleNamespace(
                execute=lambda query, variables=None: {
                    'data': {'getTests': {'results': results, 'total': 1}}
                }
            ),
            raising=False,
        )

        result = wrapper.get_tests(jql='project = X')

        _assert_json_ready(result)
        assert len(result) == 1

    def test_create_test_returns_message_with_payload(self, monkeypatch):
        wrapper = _bare(XrayApiWrapper)
        monkeypatch.setattr(
            XrayApiWrapper, '_client',
            SimpleNamespace(execute=lambda query: {'data': {'createTest': {'key': 'X-1'}}}),
            raising=False,
        )

        result = wrapper.create_test('mutation {}')

        _assert_json_ready(result)
        assert 'Created test case' in result['message']
        assert result['test'] == {'data': {'createTest': {'key': 'X-1'}}}


class TestRally:
    def test_get_entities_returns_native_list(self, monkeypatch):
        wrapper = _bare(RallyApiWrapper)
        rows = [{'Name': 'Story 1'}, {'Name': 'Story 2'}]
        monkeypatch.setattr(
            RallyApiWrapper, '_client',
            SimpleNamespace(
                get=lambda *args, **kwargs: SimpleNamespace(content={'QueryResult': {'Results': rows}})
            ),
            raising=False,
        )

        result = wrapper.get_entities(limit=1)

        _assert_json_ready(result)
        assert result == rows[:1]

    def test_get_types_returns_names(self, monkeypatch):
        wrapper = _bare(RallyApiWrapper)
        monkeypatch.setattr(
            RallyApiWrapper, '_client',
            SimpleNamespace(
                get=lambda *args, **kwargs: SimpleNamespace(
                    content={'QueryResult': {'Results': [{'ElementName': 'Defect'}]}}
                )
            ),
            raising=False,
        )

        assert wrapper.get_types() == ['Defect']


class TestConfluence:
    def test_get_page_tree_returns_native_pages(self, monkeypatch):
        wrapper = _bare(ConfluenceAPIWrapper)
        pages = [{'id': '1', 'title': 'Root', 'parent_id': None}]
        monkeypatch.setattr(ConfluenceAPIWrapper, 'get_all_descendants', lambda self, page_id: pages)

        result = wrapper.get_page_tree('1')

        _assert_json_ready(result)
        assert result == pages
