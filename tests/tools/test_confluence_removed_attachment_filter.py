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


def test_all_attachments_removed_reports_none(wrapper):
    wrapper.client.get_attachments_from_content.return_value = {'results': [
        attachment('att_b', 'removed.txt', REMOVED_FILE_ID),
    ]}
    wrapper.client.get.return_value = adf_page()

    assert wrapper.get_page_attachments('1000') == 'No attachments found for page ID 1000.'


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
