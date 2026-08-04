"""Unit coverage for ConfluenceAPIWrapper._dependents_diverged.

The stored dependent set only ever holds attachments that survived the
skip_extensions/include_extensions filters, so the current set has to be
filtered the same way — otherwise every page carrying a filtered attachment
looks changed on every run and gets re-downloaded, re-chunked and re-embedded.
"""

import pytest
from langchain_core.documents import Document

from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.confluence.api_wrapper import ConfluenceAPIWrapper

DEPS = IndexerKeywords.DEPENDENT_DOCS.value


@pytest.fixture
def wrapper():
    instance = ConfluenceAPIWrapper.model_construct()
    instance._index_include_attachments = True
    instance._skip_extensions = []
    instance._include_extensions = []
    return instance


def attachment(att_id, title):
    return {'id': att_id, 'title': title}


def page(*attachments):
    return Document(page_content='', metadata={
        'id': '1000',
        '_attachments_data': list(attachments),
    })


def indexed(*dependent_ids):
    return {DEPS: list(dependent_ids)}


def test_unchanged_attachment_set_is_not_diverged(wrapper):
    document = page(attachment('att_a', 'a.txt'), attachment('att_b', 'b.txt'))

    assert wrapper._dependents_diverged(document, indexed('att_a', 'att_b')) is False


def test_added_and_removed_attachments_are_diverged(wrapper):
    document = page(attachment('att_b', 'b.txt'), attachment('att_c', 'c.txt'))

    assert wrapper._dependents_diverged(document, indexed('att_a', 'att_b')) is True


def test_skip_filtered_attachment_is_not_diverged(wrapper):
    wrapper._skip_extensions = ['*.md']
    document = page(attachment('att_a', 'a.txt'), attachment('att_notes', 'notes.md'))

    assert wrapper._dependents_diverged(document, indexed('att_a')) is False


def test_include_filtered_attachment_is_not_diverged(wrapper):
    wrapper._include_extensions = ['*.txt']
    document = page(attachment('att_a', 'a.txt'), attachment('att_shot', 'shot.png'))

    assert wrapper._dependents_diverged(document, indexed('att_a')) is False


def test_change_behind_a_filtered_attachment_is_still_detected(wrapper):
    wrapper._skip_extensions = ['*.md']
    document = page(attachment('att_b', 'b.txt'), attachment('att_notes', 'notes.md'))

    assert wrapper._dependents_diverged(document, indexed('att_a')) is True


def test_no_divergence_check_without_include_attachments(wrapper):
    wrapper._index_include_attachments = False
    document = page(attachment('att_b', 'b.txt'))

    assert wrapper._dependents_diverged(document, indexed('att_a')) is False


def test_missing_prefetch_does_not_force_a_reindex(wrapper):
    document = Document(page_content='', metadata={'id': '1000'})

    assert wrapper._dependents_diverged(document, indexed('att_a')) is False
