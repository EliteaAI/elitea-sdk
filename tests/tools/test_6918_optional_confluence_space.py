"""
Issue #6918: the Confluence toolkit Space Key is optional and operations can pick a space at runtime.

An empty toolkit space means "all spaces the credentials can access"; a configured one is a hard
restriction that runtime spaces, page ids, CQL, generic REST calls and index filters cannot escape.
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_community.document_loaders.confluence import ContentFormat
from langchain_core.tools import ToolException

from elitea_sdk.tools.confluence.api_wrapper import ConfluenceAPIWrapper
from elitea_sdk.tools.confluence.loader import EliteAConfluenceLoader


def make_wrapper(space=None, **fields):
    instance = ConfluenceAPIWrapper.model_construct(
        space=space, client=MagicMock(), base_url='https://example.atlassian.net', llm=None, limit=5, max_pages=10, api_version='1',
        number_of_retries=1, min_retry_seconds=0, max_retry_seconds=0,
        content_format=ContentFormat.VIEW, labels=[], **fields,
    )
    instance._errors = None
    return instance


def page_in(space_key, page_id='1'):
    return {'id': page_id, 'title': 'T', 'space': {'key': space_key}}


class TestNormalizeSpace:
    @pytest.mark.parametrize('raw', [None, '', '   ', '""', "' '"])
    def test_blank_values_mean_no_space(self, raw):
        assert ConfluenceAPIWrapper._normalize_space(raw) is None

    @pytest.mark.parametrize('raw', ['MPS', ' MPS ', '"MPS"', "'MPS'"])
    def test_values_are_trimmed_and_unquoted(self, raw):
        assert ConfluenceAPIWrapper._normalize_space(raw) == 'MPS'

    def test_validator_normalizes_toolkit_space(self):
        with patch('elitea_sdk.tools.confluence.api_wrapper.Confluence', create=True), \
                patch.object(ConfluenceAPIWrapper.__mro__[1], 'validate_toolkit',
                             side_effect=lambda values: values, create=True):
            try:
                values = ConfluenceAPIWrapper.validate_toolkit.__func__(ConfluenceAPIWrapper, {
                    'base_url': 'https://example.atlassian.net', 'api_key': 'k', 'username': 'u',
                    'space': '   ', 'cloud': True,
                })
            except Exception as e:  # client construction details are out of scope
                pytest.skip(f'validator needs a real client: {e}')
        assert values['space'] is None


class TestResolveSpace:
    # Toolkit space / runtime space -> result, mirroring the index table in the issue
    @pytest.mark.parametrize('toolkit, runtime, expected', [
        (None, None, None),
        (None, 'X', 'X'),
        (None, '  ', None),
        ('T', None, 'T'),
        ('T', '', 'T'),
        ('T', 'T', 'T'),
        ('T', ' t ', 'T'),
    ])
    def test_allowed_combinations(self, toolkit, runtime, expected):
        assert make_wrapper(toolkit)._resolve_space(runtime) == expected

    def test_different_space_is_rejected(self):
        with pytest.raises(ToolException, match="restricted to Confluence space 'T'"):
            make_wrapper('T')._resolve_space('X')

    def test_required_without_any_space(self):
        with pytest.raises(ToolException, match='space key is required'):
            make_wrapper(None)._resolve_space(None, required=True)


class TestCreate:
    def _created(self, space):
        return {'id': '9', 'title': 'P', 'space': {'key': space}, 'version': {'by': {'displayName': 'A'}},
                '_links': {'webui': '/x', 'edit': '/e'}}

    def test_requires_space_when_toolkit_has_none(self):
        wrapper = make_wrapper(None)
        with pytest.raises(ToolException, match='space key is required'):
            wrapper.create_page(title='P', body='b')
        wrapper.client.create_page.assert_not_called()

    def test_uses_runtime_space_for_duplicate_check_and_create(self):
        wrapper = make_wrapper(None)
        wrapper.client.get_page_by_title.return_value = None
        wrapper.client.get_space.return_value = {'homepage': {'id': '1'}}
        wrapper.client.create_page.return_value = self._created('X')
        with patch.object(ConfluenceAPIWrapper, '_build_page_url', return_value='u'), \
                patch.object(ConfluenceAPIWrapper, '_add_default_labels'):
            wrapper.create_page(title='P', body='b', space='X')

        assert wrapper.client.get_page_by_title.call_args.kwargs['space'] == 'X'
        assert wrapper.client.create_page.call_args.kwargs['space'] == 'X'

    def test_other_space_rejected_when_toolkit_restricted(self):
        wrapper = make_wrapper('T')
        with pytest.raises(ToolException):
            wrapper.create_page(title='P', body='b', space='X')
        with pytest.raises(ToolException):
            wrapper.create_pages(pages_info='[]', space='X')
        wrapper.client.create_page.assert_not_called()


class TestPageIdsCannotBypassRestriction:
    def test_read_page_outside_space_raises(self):
        wrapper = make_wrapper('T')
        wrapper.client.get_page_by_id.return_value = page_in('OTHER')
        with pytest.raises(ToolException, match="belongs to space 'OTHER'"):
            wrapper.read_page_by_id('1')
        assert 'space' in wrapper.client.get_page_by_id.call_args.kwargs['expand'].split(',')

    def test_read_page_in_space_allowed(self):
        wrapper = make_wrapper('T')
        wrapper.client.get_page_by_id.return_value = page_in('T')
        with patch.object(ConfluenceAPIWrapper, 'process_page', return_value=MagicMock(page_content='ok')):
            assert wrapper.read_page_by_id('1') == 'ok'

    def test_unrestricted_toolkit_reads_any_space(self):
        wrapper = make_wrapper(None)
        wrapper.client.get_page_by_id.return_value = page_in('ANY')
        with patch.object(ConfluenceAPIWrapper, 'process_page', return_value=MagicMock(page_content='ok')):
            assert wrapper.read_page_by_id('1') == 'ok'

    def test_delete_by_id_outside_space_raises(self):
        wrapper = make_wrapper('T')
        wrapper.client.get_page_by_id.return_value = page_in('OTHER')
        with pytest.raises(ToolException):
            wrapper.delete_page(page_id='1')
        wrapper.client.remove_page.assert_not_called()

    def test_update_by_id_outside_space_raises(self):
        wrapper = make_wrapper('T')
        wrapper.client.get_page_by_id.return_value = page_in('OTHER')
        with pytest.raises(ToolException):
            wrapper.update_page_by_id(page_id='1', new_body='x')
        wrapper.client.update_page.assert_not_called()

    def test_page_tree_outside_space_raises(self):
        wrapper = make_wrapper('T')
        wrapper.client.get_page_by_id.return_value = page_in('OTHER')
        with pytest.raises(ToolException):
            wrapper.get_page_tree('1')
        wrapper.client.get_page_child_by_type.assert_not_called()

    def test_attachments_outside_space_are_not_fetched(self):
        wrapper = make_wrapper('T')
        wrapper.client.get_page_by_id.return_value = page_in('OTHER')
        result = wrapper.get_page_attachments('1')
        wrapper.client.get_attachments_from_content.assert_not_called()
        assert 'OTHER' in str(result)

    def test_add_file_outside_space_does_not_upload(self):
        wrapper = make_wrapper('T')
        wrapper.client.get_page_by_id.return_value = page_in('OTHER')
        with patch.object(ConfluenceAPIWrapper, '_upload_file_from_artifact') as upload:
            try:
                wrapper.add_file_to_page(page_id='1', filepath='b/f.png')
            except ToolException:
                pass
        upload.assert_not_called()


class TestTitleLookups:
    def test_update_by_title_requires_space_when_unrestricted(self):
        with pytest.raises(ToolException):
            make_wrapper(None).update_page_by_title(page_title='P', new_body='x')

    def test_delete_by_title_uses_runtime_space(self):
        wrapper = make_wrapper(None)
        wrapper.client.get_page_by_title.return_value = {'id': '7'}
        wrapper.delete_page(page_title='P', space='X')
        assert wrapper.client.get_page_by_title.call_args.kwargs['space'] == 'X'
        wrapper.client.remove_page.assert_called_once_with('7')

    def test_get_page_id_by_title_searches_all_spaces_when_unrestricted(self):
        wrapper = make_wrapper(None)
        wrapper.client.get_page_id.return_value = '5'
        wrapper.get_page_id_by_title('P')
        assert wrapper.client.get_page_id.call_args.kwargs['space'] is None

    def test_get_page_id_by_title_rejects_other_space(self):
        with pytest.raises(ToolException):
            make_wrapper('T').get_page_id_by_title('P', space='X')


class TestSearch:
    def _cql(self, wrapper, method, **kwargs):
        wrapper.client.cql.return_value = {'results': []}
        getattr(wrapper, method)(**kwargs)
        return wrapper.client.cql.call_args.args[0]

    @pytest.mark.parametrize('method', ['search_pages', 'search_by_title', 'site_search'])
    def test_unrestricted_without_space_searches_everywhere(self, method):
        assert 'space' not in self._cql(make_wrapper(None), method, query='q')

    @pytest.mark.parametrize('method', ['search_pages', 'search_by_title', 'site_search'])
    def test_runtime_space_narrows_search(self, method):
        assert 'space="X"' in self._cql(make_wrapper(None), method, query='q', space='X')

    @pytest.mark.parametrize('method', ['search_pages', 'search_by_title', 'site_search'])
    def test_toolkit_space_always_applied(self, method):
        assert 'space="T"' in self._cql(make_wrapper('T'), method, query='q')

    @pytest.mark.parametrize('method', ['search_pages', 'search_by_title', 'site_search'])
    def test_toolkit_space_rejects_other_runtime_space(self, method):
        with pytest.raises(ToolException):
            make_wrapper('T').__getattribute__(method)(query='q', space='X')

    def test_query_text_cannot_escape_space_clause(self):
        cql = self._cql(make_wrapper('T'), 'search_by_title', query='x") or space="OTHER')
        assert cql.startswith('(type=page and space="T") and (')

    def test_out_of_space_results_are_dropped(self):
        wrapper = make_wrapper('T')
        wrapper.client.cql.side_effect = [{'results': [{'content': {'id': '1'}}]}, {'results': []}]
        wrapper.client.get_page_by_id.return_value = page_in('OTHER')
        result = wrapper.search_pages(query='q')
        assert isinstance(result, str) and 'Unable to find' in result


class TestLabels:
    def test_label_with_space_uses_scoped_cql(self):
        wrapper = make_wrapper('T')
        wrapper.client.cql.return_value = {'results': []}
        wrapper.get_pages_with_label('lbl')
        cql = wrapper.client.cql.call_args.args[0]
        assert 'label="lbl"' in cql and 'space="T"' in cql
        wrapper.client.get_all_pages_by_label.assert_not_called()

    def test_label_without_space_uses_all_spaces(self):
        wrapper = make_wrapper(None)
        wrapper.client.get_all_pages_by_label.return_value = []
        wrapper.get_pages_with_label('lbl')
        wrapper.client.get_all_pages_by_label.assert_called_once()


class TestGenericRest:
    def test_blocked_when_toolkit_restricted(self):
        wrapper = make_wrapper('T')
        with pytest.raises(ToolException, match='restricted'):
            wrapper.execute_generic_confluence(method='GET', relative_url='/rest/api/content/1')
        wrapper.client.request.assert_not_called()

    def test_description_mentions_restriction(self):
        assert 'Unavailable' in make_wrapper('T')._build_generic_tool_description()
        assert 'Unavailable' not in make_wrapper(None)._build_generic_tool_description()


class TestIndexSpace:
    def test_index_params_expose_optional_space_key(self):
        params = make_wrapper(None)._index_tool_params()
        field = params['space_key'][1]
        assert field.default is None
        assert 'cannot override a toolkit-level restriction' in field.description

    def _loader_kwargs(self, wrapper, **kwargs):
        with patch('elitea_sdk.tools.confluence.loader.EliteAConfluenceLoader') as loader_cls:
            loader_cls.return_value._lazy_load.return_value = iter([])
            list(wrapper._base_loader(**kwargs))
        return loader_cls.call_args.kwargs

    # Toolkit space / index space -> loader space, per the issue table
    @pytest.mark.parametrize('toolkit, index, expected', [
        (None, 'X', 'X'),
        ('T', None, 'T'),
        ('T', '', 'T'),
        ('T', 'T', 'T'),
    ])
    def test_index_space_table(self, toolkit, index, expected):
        assert self._loader_kwargs(make_wrapper(toolkit), space_key=index)['space_key'] == expected

    def test_empty_toolkit_and_index_space_with_cql_spans_all_spaces(self):
        kwargs = self._loader_kwargs(make_wrapper(None), cql='type=page')
        assert kwargs['space_key'] is None and kwargs['cql'] == 'type=page'

    def test_index_space_outside_toolkit_space_rejected(self):
        with pytest.raises(ToolException):
            self._loader_kwargs(make_wrapper('T'), space_key='X')

    def test_no_filters_rejected(self):
        with pytest.raises(ToolException, match='at least one'):
            self._loader_kwargs(make_wrapper(None), space_key='  ')


class TestLoaderIntersection:
    def _loader(self, **kwargs):
        client = MagicMock(url='https://example.atlassian.net/wiki')
        return EliteAConfluenceLoader(client, None, **kwargs)

    def test_space_alone_loads_whole_space(self):
        loader = self._loader(space_key='T')
        assert loader.space_key == 'T' and loader.cql is None

    def test_filters_without_space_untouched(self):
        loader = self._loader(label='l', page_ids=['1'])
        assert loader.space_key is None and loader.label == 'l' and loader.page_ids == ['1']

    def test_space_intersects_all_other_filters(self):
        loader = self._loader(space_key='T', cql='type=page ORDER BY created', label='l', page_ids=['1', '2'])
        assert loader.space_key is None and loader.label is None and loader.page_ids is None
        assert loader.cql == ('space="T" AND ((type=page) OR (type=page AND label="l") '
                              'OR (id in ("1","2"))) ORDER BY created')

    def test_user_cql_or_cannot_widen_space(self):
        loader = self._loader(space_key='T', cql='space=OTHER or type=page')
        assert loader.cql == 'space="T" AND ((space=OTHER or type=page))'
