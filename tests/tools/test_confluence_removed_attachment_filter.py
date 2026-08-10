"""Coverage for excluding editor-removed attachments from get_page_attachments.

Deleting a file from the page in the Confluence editor removes its body
reference but keeps the attachment (status stays 'current'), so the attachment
listing alone can't tell removed files apart. The wrapper cross-references the
atlas_doc_format body: media nodes carry the attachment's immutable fileId.
"""

import json

import pytest
from unittest.mock import MagicMock

from elitea_sdk.tools.confluence.api_wrapper import ConfluenceAPIWrapper

EMBEDDED_FILE_ID = 'a7c79fb8-e574-4061-9d38-c5b0e2bc0c1f'
REMOVED_FILE_ID = 'f191dc06-c1ea-467c-90ef-5b6601f02c9f'


def attachment(att_id, title, file_id):
    entry = {
        'id': att_id,
        'title': title,
        'metadata': {'mediaType': 'text/plain'},
        'extensions': {'fileSize': 10},
        '_links': {'download': f'/download/attachments/1000/{title}'},
    }
    if file_id:
        entry['extensions']['fileId'] = file_id
    return entry


def adf_page(*media_nodes):
    document = {'type': 'doc', 'content': [
        {'type': 'paragraph', 'content': [{'type': 'text', 'text': 'body'}]},
        *media_nodes,
    ], 'version': 1}
    return {'body': {'atlas_doc_format': {'value': json.dumps(document)}}}


def media_inline(file_id):
    return {'type': 'paragraph', 'content': [
        {'type': 'mediaInline', 'attrs': {'id': file_id, 'collection': 'contentId-1000'}}]}


def media_single(file_id):
    return {'type': 'mediaSingle', 'attrs': {'layout': 'center'}, 'content': [
        {'type': 'media', 'attrs': {'id': file_id, 'collection': 'contentId-1000'}}]}


def macro_extension(extension_key, **macro_params):
    attrs = {'extensionType': 'com.atlassian.confluence.macro.core', 'extensionKey': extension_key}
    if macro_params:
        attrs['parameters'] = {
            'macroParams': {name: {'value': value} for name, value in macro_params.items()},
            'macroMetadata': {'schemaVersion': {'value': '1'}},
        }
    return {'type': 'extension', 'attrs': attrs}


@pytest.fixture
def wrapper():
    instance = ConfluenceAPIWrapper.model_construct()
    instance.base_url = 'https://example.atlassian.net'
    instance.client = MagicMock()
    instance.client.url = 'https://example.atlassian.net/wiki'
    instance.client.history.return_value = {}
    instance.client.get_comments_for_attachment.return_value = {'results': []}
    response = MagicMock(status_code=200)
    response.content = b'attachment body'
    instance.client.request.return_value = response
    return instance


def listed_names(result):
    return [entry['metadata']['name'] for entry in result] if isinstance(result, list) else result


def test_removed_attachment_is_excluded(wrapper):
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_a', 'kept.txt', EMBEDDED_FILE_ID),
        attachment('att_b', 'removed.txt', REMOVED_FILE_ID),
    ]}
    wrapper.client.get.return_value = adf_page(media_inline(EMBEDDED_FILE_ID))

    assert listed_names(wrapper.get_page_attachments('1000')) == ['kept.txt']
    wrapper.client.get.assert_called_once_with(
        'api/v2/pages/1000', params={'body-format': 'atlas_doc_format'})


def test_media_single_reference_counts_as_visible(wrapper):
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_a', 'image.png', EMBEDDED_FILE_ID),
    ]}
    wrapper.client.get.return_value = adf_page(media_single(EMBEDDED_FILE_ID))

    assert listed_names(wrapper.get_page_attachments('1000')) == ['image.png']


def test_all_attachments_removed_reports_distinct_message(wrapper):
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_b', 'removed.txt', REMOVED_FILE_ID),
    ]}
    wrapper.client.get.return_value = adf_page(media_inline(EMBEDDED_FILE_ID))

    assert wrapper.get_page_attachments('1000') == (
        'No attachments are visible on page 1000: none of the 1 attachment(s) '
        'returned by Confluence are referenced in the page body.')


def test_truncated_listing_is_flagged_in_the_message(wrapper):
    """The claim must stay bounded to the fetched slice when more pages exist."""
    wrapper.client.get_attachments_from_content.return_value = {
        'results': [attachment('att_b', 'removed.txt', REMOVED_FILE_ID)],
        '_links': {'next': '/rest/api/content/1000/child/attachment?start=50&limit=50'},
    }
    wrapper.client.get.return_value = adf_page(media_inline(EMBEDDED_FILE_ID))

    result = wrapper.get_page_attachments('1000')

    assert 'listing was truncated' in result


def test_blogpost_body_is_inspected_via_fallback(wrapper):
    """api/v2/pages 404s for blogpost ids — the blogposts endpoint takes the same body-format."""
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_a', 'kept.txt', EMBEDDED_FILE_ID),
        attachment('att_b', 'removed.txt', REMOVED_FILE_ID),
    ]}
    wrapper.client.get.side_effect = [
        RuntimeError('404: not a page'),
        adf_page(media_inline(EMBEDDED_FILE_ID)),
    ]

    assert listed_names(wrapper.get_page_attachments('2000')) == ['kept.txt']
    endpoints = [call.args[0] for call in wrapper.client.get.call_args_list]
    assert endpoints == ['api/v2/pages/2000', 'api/v2/blogposts/2000']


def test_macro_named_file_counts_as_visible(wrapper):
    """viewpdf/multimedia survive as extension nodes naming the file only in macroParams."""
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_a', 'doc.pdf', EMBEDDED_FILE_ID),
        attachment('att_b', 'removed.txt', REMOVED_FILE_ID),
    ]}
    wrapper.client.get.return_value = adf_page(macro_extension('viewpdf', name='doc.pdf'))

    assert listed_names(wrapper.get_page_attachments('1000')) == ['doc.pdf']


@pytest.mark.parametrize('listing_macro', ['attachments', 'gallery', 'space-attachments', 'recently-updated'])
def test_attachment_listing_macro_keeps_everything(wrapper, listing_macro):
    """Listing macros render files without naming them — nothing to match, so don't filter.

    The fixture emits no `parameters` key at all, the shape a parameterless
    macro can serialize with — the bailout must not depend on macroParams.
    """
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_a', 'kept.txt', EMBEDDED_FILE_ID),
        attachment('att_b', 'unreferenced.txt', REMOVED_FILE_ID),
    ]}
    wrapper.client.get.return_value = adf_page(macro_extension(listing_macro))

    assert listed_names(wrapper.get_page_attachments('1000')) == ['kept.txt', 'unreferenced.txt']


def test_unrelated_macro_parameter_does_not_rescue(wrapper):
    """Only filename-shaped parameters count — a jql string equal to a title must not un-filter it."""
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_b', 'removed.txt', REMOVED_FILE_ID),
    ]}
    wrapper.client.get.return_value = adf_page(macro_extension('jira', jqlQuery='removed.txt'))

    assert 'No attachments are visible' in wrapper.get_page_attachments('1000')


def test_non_dict_body_document_keeps_everything(wrapper):
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_b', 'unreferenced.txt', REMOVED_FILE_ID),
    ]}
    wrapper.client.get.return_value = {'body': {'atlas_doc_format': {'value': 'null'}}}

    assert listed_names(wrapper.get_page_attachments('1000')) == ['unreferenced.txt']


def test_server_dc_never_calls_v2_page_endpoint(wrapper):
    wrapper.cloud = False
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_b', 'unreferenced.txt', REMOVED_FILE_ID),
    ]}

    assert listed_names(wrapper.get_page_attachments('1000')) == ['unreferenced.txt']
    wrapper.client.get.assert_not_called()


def test_unresolved_cloud_still_inspects_the_body(wrapper):
    """cloud=None means unknown — attempt inspection; a DC 404 fails open anyway."""
    wrapper.cloud = None
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_a', 'kept.txt', EMBEDDED_FILE_ID),
        attachment('att_b', 'removed.txt', REMOVED_FILE_ID),
    ]}
    wrapper.client.get.return_value = adf_page(media_inline(EMBEDDED_FILE_ID))

    assert listed_names(wrapper.get_page_attachments('1000')) == ['kept.txt']


def test_empty_body_keeps_everything(wrapper):
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_b', 'unreferenced.txt', REMOVED_FILE_ID),
    ]}
    wrapper.client.get.return_value = {'body': {'atlas_doc_format': {
        'value': json.dumps({'type': 'doc', 'content': [], 'version': 1})}}}

    assert listed_names(wrapper.get_page_attachments('1000')) == ['unreferenced.txt']


def test_null_extensions_field_is_kept():
    entry = attachment('att_a', 'no-extensions.txt', None)
    entry['extensions'] = None

    assert ConfluenceAPIWrapper._visible_on_page(entry, ({EMBEDDED_FILE_ID}, set())) is True


def test_uninspectable_body_keeps_everything(wrapper):
    """Server/DC has no atlas_doc_format endpoint — never drop attachments blindly."""
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_a', 'kept.txt', EMBEDDED_FILE_ID),
        attachment('att_b', 'removed.txt', REMOVED_FILE_ID),
    ]}
    wrapper.client.get.side_effect = RuntimeError('404 not found')

    assert listed_names(wrapper.get_page_attachments('1000')) == ['kept.txt', 'removed.txt']


def test_attachment_without_file_id_is_kept(wrapper):
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_a', 'no-file-id.txt', None),
    ]}
    wrapper.client.get.return_value = adf_page()

    assert listed_names(wrapper.get_page_attachments('1000')) == ['no-file-id.txt']


def test_extension_and_visibility_filters_compose(wrapper):
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_a', 'kept.txt', EMBEDDED_FILE_ID),
        attachment('att_b', 'kept.png', EMBEDDED_FILE_ID),
        attachment('att_c', 'removed.txt', REMOVED_FILE_ID),
    ]}
    wrapper.client.get.return_value = adf_page(
        media_inline(EMBEDDED_FILE_ID), media_single(EMBEDDED_FILE_ID))

    result = wrapper.get_page_attachments('1000', allowed_extensions=['txt'])

    assert listed_names(result) == ['kept.txt']
