"""End-to-end verification of visible-attachment filtering — hits live Confluence Cloud.

Opt-in: runs only when ``CONFLUENCE_BASE_URL``, ``CONFLUENCE_USERNAME``,
``CONFLUENCE_API_KEY`` and ``CONFLUENCE_SPACE`` are set (environment or a
``.env`` file loaded via python-dotenv).

.env keys read by this suite
----------------------------
  CONFLUENCE_BASE_URL   e.g. https://yoursite.atlassian.net
  CONFLUENCE_USERNAME   Atlassian account email
  CONFLUENCE_API_KEY    API token from id.atlassian.com/manage-profile/security/api-tokens
  CONFLUENCE_SPACE      Space key to create throwaway pages in, e.g. TEST

Semantics pinned here
---------------------
``get_page_attachments`` returns only attachments *visible on the page*, i.e.
referenced from the page body (matched by immutable fileId against the
atlas_doc_format media nodes):

* embedded file → listed
* file whose reference was removed in the editor (the attachment itself stays
  ``current``) → NOT listed
* file uploaded but never embedded → NOT listed (deliberate consequence)
* properly deleted file (``DELETE /rest/api/content/{attId}``) → NOT listed
  (Confluence already drops it from the attachment listing itself)

Every page is created under a unique title and removed in ``finally``.

"""

from __future__ import annotations

import os
import time
import uuid

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv()
    for candidate in ('.elitea/.env', '.alita/.env'):
        if os.path.exists(candidate):
            load_dotenv(candidate, override=False)
except ImportError:
    pass

from elitea_sdk.tools.confluence.api_wrapper import ConfluenceAPIWrapper

REQUIRED_ENV = ('CONFLUENCE_BASE_URL', 'CONFLUENCE_USERNAME', 'CONFLUENCE_API_KEY', 'CONFLUENCE_SPACE')

pytestmark = pytest.mark.skipif(
    not all(os.environ.get(name) for name in REQUIRED_ENV),
    reason=f"{', '.join(REQUIRED_ENV)} must be set to run Confluence e2e tests",
)

CONSISTENCY_DELAY_SECONDS = 3


@pytest.fixture
def wrapper():
    return ConfluenceAPIWrapper(
        base_url=os.environ['CONFLUENCE_BASE_URL'],
        username=os.environ['CONFLUENCE_USERNAME'],
        api_key=os.environ['CONFLUENCE_API_KEY'],
        space=os.environ['CONFLUENCE_SPACE'],
        cloud=True,
        llm=None,
    )


@pytest.fixture
def page(wrapper):
    title = f"elitea-sdk-e2e-visible-{uuid.uuid4().hex[:8]}"
    created = wrapper.client.create_page(
        space=os.environ['CONFLUENCE_SPACE'], title=title, body='<p>visible-attachments e2e</p>')
    page_id = created['id']
    try:
        yield page_id
    finally:
        wrapper.client.remove_page(page_id)


def attach_file(wrapper, page_id, filename):
    wrapper.client.attach_content(
        b'elitea e2e attachment body', name=filename, content_type='text/plain', page_id=page_id)


def embed_file(wrapper, page_id, filename):
    page = wrapper.client.get_page_by_id(page_id, expand='body.storage,version')
    body = page['body']['storage']['value'] + (
        f'<ac:link><ri:attachment ri:filename="{filename}" /></ac:link>')
    wrapper.client.update_page(page_id, page['title'], body, representation='storage')
    time.sleep(CONSISTENCY_DELAY_SECONDS)


def listed_names(wrapper, page_id):
    result = wrapper.get_page_attachments(page_id)
    return [entry['metadata']['name'] for entry in result] if isinstance(result, list) else []


def test_embedded_attachment_is_listed(wrapper, page):
    filename = f"embedded-{uuid.uuid4().hex[:8]}.txt"
    attach_file(wrapper, page, filename)
    embed_file(wrapper, page, filename)

    assert filename in listed_names(wrapper, page)


def test_unembedded_attachment_is_not_listed(wrapper, page):
    filename = f"orphan-{uuid.uuid4().hex[:8]}.txt"
    attach_file(wrapper, page, filename)
    time.sleep(CONSISTENCY_DELAY_SECONDS)

    assert filename not in listed_names(wrapper, page)


def test_attachment_removed_from_body_is_not_listed(wrapper, page):
    filename = f"removed-{uuid.uuid4().hex[:8]}.txt"
    attach_file(wrapper, page, filename)
    embed_file(wrapper, page, filename)
    assert filename in listed_names(wrapper, page)

    fresh = wrapper.client.get_page_by_id(page, expand='version')
    wrapper.client.update_page(page, fresh['title'], '<p>reference removed</p>', representation='storage')
    time.sleep(CONSISTENCY_DELAY_SECONDS)

    assert filename not in listed_names(wrapper, page)
    still_attached = wrapper.client.get_attachments_from_content(page)['results']
    assert filename in [a['title'] for a in still_attached]


def test_properly_deleted_attachment_is_not_listed(wrapper, page):
    filename = f"deleted-{uuid.uuid4().hex[:8]}.txt"
    attach_file(wrapper, page, filename)
    embed_file(wrapper, page, filename)
    attachments = wrapper.client.get_attachments_from_content(page)['results']
    attachment_id = next(a['id'] for a in attachments if a['title'] == filename)

    wrapper.client.delete(f"rest/api/content/{attachment_id}")
    time.sleep(CONSISTENCY_DELAY_SECONDS)

    assert filename not in listed_names(wrapper, page)
