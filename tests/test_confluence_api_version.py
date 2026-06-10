"""Tests for the Confluence API-version handling overhaul.

Covers:
- ``_resolve_confluence_api_version`` — Confluence-specific 1/2 resolver
  (separate from Jira's 2/3 resolver in ``_resolve_api_version``).
- ``ConfluenceAPIWrapper._build_generic_tool_description`` — the description
  the LLM sees for ``execute_generic_confluence`` must reflect the active
  api_version so generated paths target the correct base.
- ``ConfluenceAPIWrapper._create_page_v2`` — payload shape, endpoint,
  representation normalization, and space-id resolution.
- ``ConfluenceAPIWrapper._adapt_v2_page_response`` — v2 responses must be
  reshaped to look like v1 so the existing ``create_page`` callers keep
  working.
"""

import pytest
from unittest.mock import MagicMock

from elitea_sdk.configurations.utils import (
    _resolve_api_version,
    _resolve_confluence_api_version,
)
from elitea_sdk.tools.confluence.api_wrapper import ConfluenceAPIWrapper


# ---------------------------------------------------------------------------
# _resolve_confluence_api_version
# ---------------------------------------------------------------------------

class TestResolveConfluenceApiVersion:
    """Confluence has v1 (legacy /rest/api) and v2 (/api/v2, Cloud only)."""

    @pytest.mark.parametrize(
        ('api_version', 'cloud', 'base_url', 'expected'),
        [
            # Auto resolution
            ('auto', True, None, '2'),
            ('auto', False, 'https://confluence.local', '1'),
            ('auto', None, 'https://example.atlassian.net', '2'),
            ('auto', None, 'https://confluence.example.com', '1'),
            (None, True, None, '2'),
            (None, False, None, '1'),
            ('', True, None, '2'),
            # Explicit valid versions pass through
            ('1', True, None, '1'),
            ('1', False, None, '1'),
            ('2', True, None, '2'),
            ('2', False, None, '2'),
        ],
    )
    def test_resolution(self, api_version, cloud, base_url, expected):
        assert _resolve_confluence_api_version(api_version, cloud, base_url) == expected

    def test_unknown_version_falls_back_to_auto(self):
        # '3' is not a valid Confluence version — must auto-resolve, not pass through.
        assert _resolve_confluence_api_version('3', True, None) == '2'
        assert _resolve_confluence_api_version('3', False, None) == '1'

    def test_jira_resolver_unchanged(self):
        """Regression guard: Jira's 2/3 resolver must not be affected."""
        assert _resolve_api_version('auto', True, None) == '3'
        assert _resolve_api_version('auto', False, None) == '2'
        assert _resolve_api_version('2', True, None) == '2'
        assert _resolve_api_version('3', False, None) == '3'


# ---------------------------------------------------------------------------
# _build_generic_tool_description
# ---------------------------------------------------------------------------

def _make_wrapper(api_version: str) -> ConfluenceAPIWrapper:
    w = ConfluenceAPIWrapper.model_construct(
        base_url='https://example.atlassian.net',
        space='AT',
        cloud=True,
        limit=25,
        max_pages=50,
        api_version=api_version,
    )
    w.client = MagicMock()
    return w


class TestGenericToolDescription:
    def test_v1_description_points_at_rest_api(self):
        desc = _make_wrapper('1')._build_generic_tool_description()
        assert 'v1' in desc
        assert '/rest/api' in desc
        assert '/wiki/api/v2' not in desc
        assert '/rest/api/content' in desc  # example path
        assert 'space.key' in desc  # v1 body shape hint

    def test_v2_description_points_at_v2_base(self):
        desc = _make_wrapper('2')._build_generic_tool_description()
        assert 'v2' in desc
        # The underlying Atlassian client auto-prepends `/wiki`; relative_url
        # must NOT include it, otherwise requests double up to /wiki/wiki/...
        assert '/wiki/api/v2' not in desc
        assert '/api/v2' in desc
        assert '/api/v2/pages' in desc  # example path
        assert 'spaceId' in desc  # v2 body shape hint

    def test_description_includes_base_docstring(self):
        desc = _make_wrapper('1')._build_generic_tool_description()
        # The original execute_generic_confluence docstring should be preserved
        assert 'Generic Confluence Tool' in desc


# ---------------------------------------------------------------------------
# _normalize_v2_representation
# ---------------------------------------------------------------------------

class TestNormalizeV2Representation:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            (None, 'storage'),
            ('', 'storage'),
            ('storage', 'storage'),
            ('STORAGE', 'storage'),
            ('wiki', 'wiki'),
            ('atlas_doc_format', 'atlas_doc_format'),
            ('adf', 'atlas_doc_format'),  # alias
            ('view', 'storage'),  # read-only format → fall back
            ('export_view', 'storage'),
            ('editor', 'storage'),
            ('unknown', 'storage'),
        ],
    )
    def test_mapping(self, value, expected):
        assert ConfluenceAPIWrapper._normalize_v2_representation(value) == expected


# ---------------------------------------------------------------------------
# _create_page_v2
# ---------------------------------------------------------------------------

class TestCreatePageV2:
    def test_posts_v2_endpoint_with_correct_payload(self):
        w = _make_wrapper('2')
        # Space-id lookup via v2
        w.client.get.return_value = {'results': [{'id': '987654'}]}
        # Page-create response (minimal v2 shape)
        w.client.post.return_value = {
            'id': '111',
            'title': 'New Page',
            'status': 'current',
            'spaceId': '987654',
            'version': {'number': 1, 'authorId': 'user-abc'},
            'body': {'storage': {'value': '<p>hi</p>'}},
            '_links': {'webui': '/spaces/AT/pages/111', 'editui': '/pages/edit-v2.action?pageId=111'},
        }

        result = w._create_page_v2(
            space='AT',
            title='New Page',
            body='<p>hi</p>',
            parent_id='42',
            representation='storage',
            status='current',
        )

        # Endpoint
        post_args, post_kwargs = w.client.post.call_args
        assert post_args[0] == 'api/v2/pages'

        # Payload shape — v2 uses spaceId/parentId and body:{representation,value}
        payload = post_kwargs.get('data') or (post_args[1] if len(post_args) > 1 else None)
        assert payload == {
            'spaceId': '987654',
            'status': 'current',
            'title': 'New Page',
            'parentId': '42',
            'body': {'representation': 'storage', 'value': '<p>hi</p>'},
        }

        # Response was adapted to v1-shape so callers keep working
        assert result['id'] == '111'
        assert result['space']['key'] == 'AT'
        assert result['_links']['webui'].endswith('/spaces/AT/pages/111')
        assert result['version']['number'] == 1

    def test_omits_parent_id_when_none(self):
        w = _make_wrapper('2')
        w.client.get.return_value = {'results': [{'id': '987654'}]}
        w.client.post.return_value = {'id': '1', 'spaceId': '987654', '_links': {}}

        w._create_page_v2(
            space='AT', title='t', body='b', parent_id=None,
            representation='storage', status='current',
        )

        _, post_kwargs = w.client.post.call_args
        payload = post_kwargs.get('data')
        assert 'parentId' not in payload

    def test_normalizes_representation_in_payload(self):
        w = _make_wrapper('2')
        w.client.get.return_value = {'results': [{'id': '987654'}]}
        w.client.post.return_value = {'id': '1', 'spaceId': '987654', '_links': {}}

        w._create_page_v2(
            space='AT', title='t', body='b', parent_id=None,
            representation='adf',  # alias
            status='current',
        )

        _, post_kwargs = w.client.post.call_args
        assert post_kwargs['data']['body']['representation'] == 'atlas_doc_format'

    def test_falls_back_to_v1_space_lookup_when_v2_fails(self):
        w = _make_wrapper('2')
        # v2 lookup raises, v1 get_space succeeds
        w.client.get.side_effect = RuntimeError('v2 lookup failed')
        w.client.get_space = MagicMock(return_value={'id': '777', 'key': 'AT'})
        w.client.post.return_value = {'id': '1', 'spaceId': '777', '_links': {}}

        w._create_page_v2(
            space='AT', title='t', body='b', parent_id=None,
            representation='storage', status='current',
        )

        _, post_kwargs = w.client.post.call_args
        assert post_kwargs['data']['spaceId'] == '777'

    def test_raises_when_space_id_unresolvable(self):
        from langchain_core.tools import ToolException
        w = _make_wrapper('2')
        w.client.get.return_value = {'results': []}  # v2 returns nothing
        w.client.get_space = MagicMock(return_value={})  # v1 also empty

        with pytest.raises(ToolException, match='spaceId'):
            w._create_page_v2(
                space='MISSING', title='t', body='b', parent_id=None,
                representation='storage', status='current',
            )

    def test_create_page_routes_v2_for_api_version_2(self):
        """create_page must dispatch to the v2 path when api_version='2'."""
        w = _make_wrapper('2')
        # Page-existence pre-check + space-homepage lookup
        w.client.get_page_by_title.return_value = None
        w.client.get_space.return_value = {'homepage': {'id': '1'}}
        # v2 space-id lookup + page POST
        w.client.get.return_value = {'results': [{'id': '987654'}]}
        w.client.post.return_value = {
            'id': '111', 'title': 't', 'status': 'current',
            'spaceId': '987654',
            'version': {'number': 1, 'authorId': 'user-x'},
            '_links': {'webui': '/spaces/AT/pages/111', 'editui': '/pages/edit-v2.action?pageId=111'},
        }
        w._add_default_labels = MagicMock()

        w.create_page(title='t', body='b', representation='storage', space='AT')

        # v2 endpoint, NOT v1's "rest/api/content/"
        assert w.client.post.call_args[0][0] == 'api/v2/pages'
        # Upstream lib's v1 create_page must NOT be called when api_version='2'
        w.client.create_page.assert_not_called()

    def test_create_page_uses_lib_create_page_for_api_version_1(self):
        """create_page must delegate to the upstream lib's v1 client.create_page."""
        w = _make_wrapper('1')
        w.client.get_page_by_title.return_value = None
        w.client.get_space.return_value = {'homepage': {'id': '1'}}
        w.client.create_page.return_value = {
            'id': '1', 'title': 't',
            'space': {'key': 'AT'},
            'version': {'by': {'displayName': 'Alice'}},
            '_links': {'webui': '/wiki/spaces/AT/pages/1', 'edit': '/wiki/pages/edit?pageId=1'},
        }
        w._add_default_labels = MagicMock()

        w.create_page(title='t', body='b', representation='storage', space='AT')

        # Delegated to the lib's v1 create_page — wrapper must not POST directly
        w.client.create_page.assert_called_once()
        call_kwargs = w.client.create_page.call_args.kwargs
        assert call_kwargs['space'] == 'AT'
        assert call_kwargs['title'] == 't'
        assert call_kwargs['representation'] == 'storage'
        # And must NOT have constructed a v2 POST
        assert all(c.args[0] != 'api/v2/pages' for c in w.client.post.call_args_list)


# ---------------------------------------------------------------------------
# _adapt_v2_page_response
# ---------------------------------------------------------------------------

class TestAdaptV2PageResponse:
    def test_reshapes_to_v1(self):
        v2 = {
            'id': '321',
            'title': 'My Page',
            'status': 'current',
            'spaceId': '987654',
            'version': {'number': 4, 'authorId': 'user-xyz'},
            'body': {'storage': {'value': '<p>x</p>'}},
            '_links': {
                'webui': '/spaces/AT/pages/321',
                'editui': '/pages/edit-v2.action?pageId=321',
                'base': 'https://example.atlassian.net/wiki',
            },
        }
        out = ConfluenceAPIWrapper._adapt_v2_page_response(
            ConfluenceAPIWrapper.model_construct(),
            v2,
            space_key='AT',
            representation='storage',
        )

        assert out['id'] == '321'
        assert out['title'] == 'My Page'
        assert out['space'] == {'key': 'AT', 'id': '987654'}
        assert out['version']['number'] == 4
        # author is best-effort: v2 only gives authorId, not displayName
        assert out['version']['by']['displayName'] == 'user-xyz'
        assert out['_links']['webui'] == '/spaces/AT/pages/321'
        assert out['_links']['edit'] == '/pages/edit-v2.action?pageId=321'
        assert out['_links']['base'] == 'https://example.atlassian.net/wiki'
        assert out['_v2_raw'] is v2  # original kept for debugging

    def test_synthesizes_links_when_missing(self):
        v2 = {'id': '500', 'title': 't', 'status': 'current', 'spaceId': '1'}
        out = ConfluenceAPIWrapper._adapt_v2_page_response(
            ConfluenceAPIWrapper.model_construct(),
            v2, space_key='AT', representation='storage',
        )
        # Synthesized fallbacks let downstream link building work
        assert '500' in out['_links']['webui']
        assert 'AT' in out['_links']['webui']
        assert out['_links']['edit']

    def test_passthrough_for_non_dict(self):
        out = ConfluenceAPIWrapper._adapt_v2_page_response(
            ConfluenceAPIWrapper.model_construct(),
            'not-a-dict', space_key='AT', representation='storage',
        )
        assert out == 'not-a-dict'
