"""Unit coverage for XrayApiWrapper._dependents_diverged.

`_attachments_data` is only present when attachments are enabled and the test
has at least one, so the check has to bail out rather than read an absent key
as "no attachments" and report every test as changed.
"""

import pytest
from langchain_core.documents import Document

from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.xray.api_wrapper import XrayApiWrapper

DEPS = IndexerKeywords.DEPENDENT_DOCS.value


@pytest.fixture
def wrapper():
    instance = XrayApiWrapper.model_construct()
    instance._include_attachments = True
    instance._skipped_attachment_extensions = set()
    return instance


def test_unchanged_attachment_set_is_not_diverged(wrapper):
    document = Document(page_content='', metadata={
        'id': 'T-1',
        '_attachments_data': [{'id': 'a', 'filename': 'a.txt'}],
    })

    assert wrapper._dependents_diverged(document, {DEPS: ['attach_a']}) is False


def test_added_attachment_is_diverged(wrapper):
    document = Document(page_content='', metadata={
        'id': 'T-1',
        '_attachments_data': [{'id': 'a', 'filename': 'a.txt'},
                              {'id': 'b', 'filename': 'b.txt'}],
    })

    assert wrapper._dependents_diverged(document, {DEPS: ['attach_a']}) is True


def test_attachments_disabled_does_not_force_a_reindex(wrapper):
    wrapper._include_attachments = False
    document = Document(page_content='', metadata={'id': 'T-1'})

    assert wrapper._dependents_diverged(document, {DEPS: ['attach_a']}) is False


def test_test_without_attachments_does_not_force_a_reindex(wrapper):
    document = Document(page_content='', metadata={'id': 'T-1'})

    assert wrapper._dependents_diverged(document, {DEPS: ['attach_a']}) is False


def test_skipped_extension_is_not_counted(wrapper):
    wrapper._skipped_attachment_extensions = {'.png'}
    document = Document(page_content='', metadata={
        'id': 'T-1',
        '_attachments_data': [{'id': 'a', 'filename': 'a.txt'},
                              {'id': 'shot', 'filename': 'shot.png'}],
    })

    assert wrapper._dependents_diverged(document, {DEPS: ['attach_a']}) is False
