"""The configured QTest project is only a default: every call may name another one.

Resolution is the call's project_id, then the configured qtest_project_id, then an
actionable error. A call on another project leaves the wrapper alone, and the per-project
caches never serve one project's data for another.
"""
import inspect
import threading
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.tools import ToolException
from pydantic import SecretStr

from elitea_sdk.tools.qtest import api_wrapper as qtest_module
from elitea_sdk.tools.qtest.api_wrapper import NO_PROJECT_MESSAGE, QtestApiWrapper


def _wrapper(configured=7):
    wrapper = QtestApiWrapper.model_construct(
        base_url='https://qtest.example.com', qtest_project_id=configured,
        qtest_api_token=SecretStr('token'), no_of_items_per_page=100, page=1,
        no_of_tests_shown_in_dql_search=10, llm=None, elitea=None)
    wrapper._client = None
    wrapper._init_indexing_stats()
    return wrapper


class RecordingTestCaseApi:
    def __init__(self, calls):
        self.calls = calls

    def delete_test_case(self, project_id, qtest_id):
        self.calls.append(project_id)


@pytest.fixture
def deletes(monkeypatch):
    calls = []
    monkeypatch.setattr(QtestApiWrapper, '_QtestApiWrapper__instantiate_test_api_instance',
                        lambda self: RecordingTestCaseApi(calls))
    return calls


@pytest.fixture
def modules(monkeypatch):
    """ModuleApi that names each module after the project it was listed from."""
    calls = []

    class ModuleApi:
        def __init__(self, client):
            pass

        def get_sub_modules_of(self, project_id, **kwargs):
            calls.append(project_id)
            return [SimpleNamespace(id=project_id * 10, pid=f'MD-{project_id}',
                                    name=f'Suite of {project_id}', children=None)]

    monkeypatch.setattr(qtest_module.swagger_client, 'ModuleApi', ModuleApi)
    return calls


class TestResolution:
    def test_configured_project_is_the_fallback(self):
        assert _wrapper()._resolve_project_id() == 7

    def test_runtime_project_wins(self):
        assert _wrapper()._resolve_project_id(42) == 42

    def test_numeric_strings_are_accepted(self):
        assert _wrapper()._resolve_project_id(' "42" ') == 42

    @pytest.mark.parametrize('placeholder', ['', '  ', 'null', 'None', "''", 'undefined', 'n/a'])
    def test_pipeline_placeholders_mean_the_configured_project(self, placeholder):
        assert _wrapper()._resolve_project_id(placeholder) == 7

    def test_no_project_anywhere_is_actionable(self, deletes):
        wrapper = _wrapper(configured=None)
        with pytest.raises(ToolException) as error:
            wrapper._resolve_project_id()
        assert str(error.value) == NO_PROJECT_MESSAGE
        with pytest.raises(ToolException, match='No QTest project specified'):
            wrapper.delete_test_case(qtest_id=1)
        assert deletes == []

    def test_a_non_numeric_project_is_rejected(self):
        with pytest.raises(ToolException, match='must be a positive number'):
            _wrapper()._resolve_project_id('PRJ')

    def test_the_wrapper_never_reads_the_elitea_project_id(self):
        """`project_id` in toolkit values is the Elitea project, not a QTest one."""
        wrapper = QtestApiWrapper(base_url='https://qtest.example.com', qtest_api_token='t',
                                  project_id=999, llm=None)
        assert wrapper.qtest_project_id is None


class TestCallsTargetTheNamedProject:
    def test_a_call_uses_its_own_project_then_the_configured_one(self, deletes):
        wrapper = _wrapper()

        wrapper.delete_test_case(qtest_id=1, project_id=42)
        wrapper.delete_test_case(qtest_id=1)

        assert deletes == [42, 7]
        assert wrapper.qtest_project_id == 7

    def test_dispatch_through_run_forwards_the_project(self, deletes):
        _wrapper().run('delete_test_case', qtest_id=1, project_id=42)
        assert deletes == [42]

    def test_the_result_names_the_project_used(self, deletes):
        assert 'project with id - 42' in _wrapper().delete_test_case(qtest_id=1, project_id=42)

    def test_a_nested_tool_keeps_the_outer_project(self, monkeypatch):
        seen = []
        monkeypatch.setattr(QtestApiWrapper, '_QtestApiWrapper__perform_search_by_dql',
                            lambda self, *args, **kwargs: seen.append(self._project_id()) or [])

        _wrapper().find_test_case_by_id(test_id='TC-1', project_id=42)

        assert seen == [42]

    def test_a_project_scope_ends_with_its_call(self, deletes):
        wrapper = _wrapper()
        wrapper.delete_test_case(qtest_id=1, project_id=42)
        assert wrapper._project_id() == 7

    def test_scopes_are_per_wrapper(self):
        first, second = _wrapper(configured=1), _wrapper(configured=2)
        with first._project_scope(42):
            assert (first._project_id(), second._project_id()) == (42, 2)

    def test_concurrent_calls_on_different_projects_do_not_cross(self, monkeypatch):
        wrapper = _wrapper()
        projects = list(range(100, 108))
        barrier = threading.Barrier(len(projects))
        crossed = []

        class SlowApi:
            def delete_test_case(self, project_id, qtest_id):
                barrier.wait()
                if wrapper._project_id() != project_id:
                    crossed.append(project_id)

        monkeypatch.setattr(QtestApiWrapper, '_QtestApiWrapper__instantiate_test_api_instance',
                            lambda self: SlowApi())
        results = {}

        def call(project):
            results[project] = wrapper.delete_test_case(qtest_id=1, project_id=project)

        threads = [threading.Thread(target=call, args=(project,)) for project in projects]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert crossed == []
        assert all(f'project with id - {project} ' in results[project] for project in projects)
        assert wrapper._project_id() == 7


class TestCachesArePerProject:
    def test_modules_are_listed_once_per_project(self, modules):
        wrapper = _wrapper()

        with wrapper._project_scope(42):
            assert wrapper._parse_modules()[0]['module_name'] == 'Suite of 42'
            wrapper._parse_modules()
        assert wrapper._parse_modules()[0]['module_name'] == 'Suite of 7'

        assert modules == [42, 7]

    def test_field_definitions_are_cached_per_project(self, monkeypatch):
        wrapper = _wrapper()
        loads = []
        monkeypatch.setattr(QtestApiWrapper, '_QtestApiWrapper__get_project_field_definitions',
                            lambda self: loads.append(self._project_id()) or {'Status': {'project': self._project_id()}})

        wrapper.get_all_test_cases_fields_for_project(project_id=42)
        wrapper.get_all_test_cases_fields_for_project(project_id=42)
        wrapper.get_all_test_cases_fields_for_project()
        wrapper.get_all_test_cases_fields_for_project(project_id=42, force_refresh=True)

        assert loads == [42, 7, 42]
        assert set(wrapper._field_definitions_cache) == {42, 7}


class TestSchemas:
    def test_every_tool_offers_an_optional_project_matching_its_method(self):
        wrapper = _wrapper()
        index_tools = {'index_data', 'search_index', 'stepback_search_index',
                       'stepback_summary_index', 'list_indexes', 'remove_index'}
        tools = [tool for tool in wrapper.get_available_tools() if tool['name'] not in index_tools]

        assert len(tools) == 19
        for tool in tools:
            field = tool['args_schema'].model_fields.get('project_id')
            assert field is not None and not field.is_required(), tool['name']
            assert 'project_id' in inspect.signature(tool['ref']).parameters, tool['name']
            assert tool.get('group'), tool['name']

    def test_project_is_the_last_field_of_every_tool(self):
        """A plain-string tool input binds to the first field, which must stay the tool's own."""
        for tool in _wrapper().get_available_tools():
            fields = list(tool['args_schema'].model_fields)
            if 'project_id' in fields:
                assert fields[-1] == 'project_id', tool['name']

    @pytest.mark.parametrize('sent, expected', [
        ('', 7), ('null', 7), ('None', 7), (None, 7), ('"42"', 42), ('42', 42), (42, 42)])
    def test_pipeline_placeholders_pass_tool_validation(self, deletes, sent, expected):
        from elitea_sdk.tools.qtest import QtestToolkit

        toolkit = QtestToolkit.get_toolkit(
            selected_tools=['delete_test_case'], llm=None, qtest_project_id=7,
            qtest_configuration={'base_url': 'https://qtest.example.com', 'qtest_api_token': 't'})
        tool = toolkit.get_tools()[0]
        tool.api_wrapper._client = None

        tool.invoke({'qtest_id': 1, 'project_id': sent})

        assert deletes == [expected]

    @pytest.mark.parametrize('rejected', [0, -3, True, 4.7])
    def test_non_positive_or_fractional_projects_are_rejected(self, rejected):
        with pytest.raises(ToolException, match='must be a positive number'):
            _wrapper()._resolve_project_id(rejected)

    def test_toolkit_project_is_optional(self):
        from elitea_sdk.tools.qtest import QtestToolkit

        schema = QtestToolkit.toolkit_config_schema().model_json_schema()
        assert 'qtest_project_id' not in schema.get('required', [])

    def test_toolkit_builds_without_a_project(self):
        from elitea_sdk.tools.qtest import QtestToolkit

        toolkit = QtestToolkit.get_toolkit(
            selected_tools=['get_modules'], llm=None,
            qtest_configuration={'base_url': 'https://qtest.example.com', 'qtest_api_token': 't'})
        tools = toolkit.get_tools()

        assert [tool.name for tool in tools] == ['get_modules']
        assert 'Project id:' not in tools[0].description

    def test_configured_project_stays_in_the_description(self):
        from elitea_sdk.tools.qtest import QtestToolkit

        toolkit = QtestToolkit.get_toolkit(
            selected_tools=['get_modules'], llm=None, qtest_project_id=7,
            qtest_configuration={'base_url': 'https://qtest.example.com', 'qtest_api_token': 't'})

        assert 'Project id: 7' in toolkit.get_tools()[0].description


class TestIndexTarget:
    @pytest.fixture
    def listing(self, monkeypatch):
        calls = []

        class Api:
            def get_test_cases(self, project_id, **kwargs):
                calls.append(project_id)
                if kwargs['page'] > 1:
                    return []
                return [{'id': project_id, 'pid': f'TC-{project_id}', 'name': 'Login',
                         'parent_id': project_id * 10}]

        monkeypatch.setattr(QtestApiWrapper, '_QtestApiWrapper__instantiate_test_api_instance',
                            lambda self: Api())
        return calls

    def test_index_params_offer_the_project(self):
        assert 'qtest_project_id' in _wrapper()._index_tool_params()

    def test_loader_reads_the_named_project_and_records_it(self, listing, modules):
        wrapper = _wrapper()

        documents = list(wrapper._base_loader(qtest_project_id=42))

        assert listing == [42]
        assert [doc.metadata['project_id'] for doc in documents] == [42]
        assert documents[0].metadata['module_name'] == 'MD-42 Suite of 42'
        assert wrapper._project_id() == 7

    def test_loader_defaults_to_the_configured_project(self, listing, modules):
        documents = list(_wrapper()._base_loader())
        assert [doc.metadata['project_id'] for doc in documents] == [7]

    def test_index_without_any_project_fails_before_listing(self, listing):
        with pytest.raises(ToolException, match='No QTest project specified'):
            next(_wrapper(configured=None)._base_loader())
        assert listing == []

    def test_the_project_does_not_leak_into_the_consumer_between_documents(self, listing, modules):
        wrapper = _wrapper()
        for _document in wrapper._base_loader(qtest_project_id=42):
            assert wrapper._project_id() == 7

    def test_extend_data_formats_in_the_documents_project(self, modules):
        wrapper = _wrapper()
        document = Document(page_content='', metadata={
            'id': 'TC-1', 'name': 'Login', 'project_id': 42,
            '_raw_item': {'pid': 'TC-1', 'name': 'Login', 'parent_id': 420}})

        [extended] = list(wrapper._extend_data(iter([document])))

        raw = next(value for value in extended.metadata.values() if isinstance(value, bytes))
        assert 'MD-42 Suite of 42' in raw.decode()
        assert modules == [42]
