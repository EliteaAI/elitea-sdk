"""Unit coverage for NonCodeIndexerToolkit.remove_ids_fn.

Deleting a document's dependents must not depend on the parent's stored
dependent_docs string: older builds wrote it unusably (no ids at all, or
comma-joined while the adapter splits on ';') and it can name dependents that
never produced a row.
"""

import pytest

from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.non_code_indexer_toolkit import NonCodeIndexerToolkit

PARENT = IndexerKeywords.PARENT.value
DEPS = IndexerKeywords.DEPENDENT_DOCS.value


@pytest.fixture
def toolkit():
    return NonCodeIndexerToolkit.model_construct()


def entry(db_id, all_chunks=None, parent=-1, deps=None):
    return {
        'metadata': {},
        'id': db_id,
        'all_chunks': all_chunks or [db_id],
        DEPS: deps or [],
        PARENT: parent,
    }


def test_removes_parent_and_dependent_chunks(toolkit):
    idx_data = {
        'page-1': entry('db-page', ['db-page', 'db-page-2'], deps=['att-a']),
        'att-a': entry('db-att-a', ['db-att-a', 'db-att-a-2'], parent='page-1'),
    }

    assert set(toolkit.remove_ids_fn(idx_data, 'page-1')) == {
        'db-page', 'db-page-2', 'db-att-a', 'db-att-a-2'
    }


def test_dependent_named_but_never_indexed_is_not_an_error(toolkit):
    """An attachment whose content is unsupported, unparseable or empty is
    registered on the parent but produces no row of its own."""
    idx_data = {
        'page-1': entry('db-page', deps=['att-a', 'att-ghost']),
        'att-a': entry('db-att-a', parent='page-1'),
    }

    assert set(toolkit.remove_ids_fn(idx_data, 'page-1')) == {'db-page', 'db-att-a'}


def test_legacy_index_without_dependent_docs(toolkit):
    """Builds before attachment ids existed stored no dependent_docs; the
    dependents are still reachable through the parent stamp they carry."""
    idx_data = {
        'page-1': entry('db-page', deps=[]),
        'db-att-a': entry('db-att-a', parent='page-1'),
        'db-att-b': entry('db-att-b', parent='page-1'),
    }

    assert set(toolkit.remove_ids_fn(idx_data, 'page-1')) == {
        'db-page', 'db-att-a', 'db-att-b'
    }


def test_legacy_index_with_comma_joined_dependent_docs(toolkit):
    """The adapter splits dependent_docs on ';', so a comma-joined value reads
    back as one token matching nothing."""
    idx_data = {
        'page-1': entry('db-page', deps=['att-a,att-b']),
        'att-a': entry('db-att-a', parent='page-1'),
        'att-b': entry('db-att-b', parent='page-1'),
    }

    assert set(toolkit.remove_ids_fn(idx_data, 'page-1')) == {
        'db-page', 'db-att-a', 'db-att-b'
    }


def test_leaves_other_parents_alone(toolkit):
    idx_data = {
        'page-1': entry('db-page-1'),
        'page-2': entry('db-page-2'),
        'att-a': entry('db-att-a', parent='page-1'),
        'att-b': entry('db-att-b', parent='page-2'),
    }

    assert set(toolkit.remove_ids_fn(idx_data, 'page-1')) == {'db-page-1', 'db-att-a'}


def test_numeric_parent_ids_match_string_keys(toolkit):
    idx_data = {
        '4242': entry('db-page'),
        'att-a': entry('db-att-a', parent=4242),
    }

    assert set(toolkit.remove_ids_fn(idx_data, '4242')) == {'db-page', 'db-att-a'}


def test_reverse_index_is_rebuilt_for_a_new_idx_data(toolkit):
    first = {
        'page-1': entry('db-page-1'),
        'att-a': entry('db-att-a', parent='page-1'),
    }
    assert set(toolkit.remove_ids_fn(first, 'page-1')) == {'db-page-1', 'db-att-a'}

    second = {
        'page-1': entry('db-page-1'),
        'att-z': entry('db-att-z', parent='page-1'),
    }
    assert set(toolkit.remove_ids_fn(second, 'page-1')) == {'db-page-1', 'db-att-z'}


def test_reverse_index_is_released_when_the_dedup_pass_ends(toolkit, monkeypatch):
    """The toolkit instance outlives the call, so the pass must not leave the
    whole collection's metadata pinned behind it."""
    idx_data = {
        'page-1': entry('db-page-1'),
        'att-a': entry('db-att-a', parent='page-1'),
    }
    monkeypatch.setattr(toolkit, '_get_indexed_data', lambda index_name: idx_data)
    monkeypatch.setattr(toolkit, '_ensure_vectorstore_initialized', lambda: None)
    monkeypatch.setattr(toolkit, '_log_tool_event', lambda *a, **kw: None)

    list(toolkit._reduce_duplicates(iter([]), 'idx'))

    assert toolkit._dependents_index == (None, None)
