"""
Unit tests for ChromaAdapter cleanup behavior used by remove_index.

Tests verify:
1. Named cleanup deletes only docs with metadata collection == "foo"
2. Missing indexes return 0 deleted records so callers can raise ToolException
3. Full cleanup propagates backend get/delete failures rather than returning success
"""

import pytest
from unittest.mock import MagicMock

from elitea_sdk.tools.vector_adapters.VectorStoreAdapter import ChromaAdapter
from elitea_sdk.runtime.utils.utils import IndexerKeywords


class TestChromaAdapterCleanCollection:
    """Tests for ChromaAdapter.clean_collection() filtering behavior."""

    @pytest.fixture
    def adapter(self):
        return ChromaAdapter()

    @pytest.fixture
    def mock_vectorstore_wrapper(self):
        """Create a mock vectorstore wrapper with Chroma-like behavior."""
        wrapper = MagicMock()
        wrapper.vectorstore = MagicMock()
        return wrapper

    def test_clean_collection_filters_by_index_name(self, adapter, mock_vectorstore_wrapper):
        """remove_index(index_name="foo") should only delete docs with collection=="foo"."""
        mock_vectorstore_wrapper.vectorstore.get.return_value = {
            'ids': ['id1', 'id2'],
            'metadatas': [
                {'collection': 'foo', 'content': 'doc1'},
                {'collection': 'foo', 'content': 'doc2'}
            ]
        }

        deleted_count = adapter.clean_collection(
            mock_vectorstore_wrapper,
            index_name="foo",
            including_index_meta=True
        )

        mock_vectorstore_wrapper.vectorstore.get.assert_called_once_with(
            where={"collection": "foo"},
            include=['metadatas']
        )
        mock_vectorstore_wrapper.vectorstore.delete.assert_called_once_with(ids=['id1', 'id2'])
        assert deleted_count == 2

    def test_clean_collection_excludes_index_meta_when_flag_false(self, adapter, mock_vectorstore_wrapper):
        """When including_index_meta=False, should not delete INDEX_META_TYPE records."""
        mock_vectorstore_wrapper.vectorstore.get.return_value = {
            'ids': ['id1', 'id2', 'id3'],
            'metadatas': [
                {'collection': 'foo', 'content': 'doc1'},
                {'collection': 'foo', 'type': IndexerKeywords.INDEX_META_TYPE.value},
                {'collection': 'foo', 'content': 'doc3'}
            ]
        }

        deleted_count = adapter.clean_collection(
            mock_vectorstore_wrapper,
            index_name="foo",
            including_index_meta=False
        )

        mock_vectorstore_wrapper.vectorstore.delete.assert_called_once_with(ids=['id1', 'id3'])
        assert deleted_count == 2

    def test_clean_collection_returns_zero_for_missing_index(self, adapter, mock_vectorstore_wrapper):
        """remove_index(index_name="missing") should return 0 when no matching docs exist."""
        mock_vectorstore_wrapper.vectorstore.get.return_value = {
            'ids': [],
            'metadatas': []
        }

        deleted_count = adapter.clean_collection(
            mock_vectorstore_wrapper,
            index_name="missing",
            including_index_meta=True
        )

        mock_vectorstore_wrapper.vectorstore.get.assert_called_once_with(
            where={"collection": "missing"},
            include=['metadatas']
        )
        mock_vectorstore_wrapper.vectorstore.delete.assert_not_called()
        assert deleted_count == 0

    def test_clean_collection_does_not_delete_other_indexes(self, adapter, mock_vectorstore_wrapper):
        """Deleting index "foo" should not affect index "bar" in the same physical collection."""
        mock_vectorstore_wrapper.vectorstore.get.return_value = {
            'ids': ['foo_id1'],
            'metadatas': [{'collection': 'foo'}]
        }

        adapter.clean_collection(mock_vectorstore_wrapper, index_name="foo", including_index_meta=True)

        mock_vectorstore_wrapper.vectorstore.get.assert_called_with(
            where={"collection": "foo"},
            include=['metadatas']
        )
        mock_vectorstore_wrapper.vectorstore.delete.assert_called_once_with(ids=['foo_id1'])

    def test_clean_collection_full_cleanup_deletes_all(self, adapter, mock_vectorstore_wrapper):
        """remove_index() with no index_name should delete all records."""
        mock_vectorstore_wrapper.vectorstore.get.return_value = {
            'ids': ['id1', 'id2', 'id3'],
        }

        deleted_count = adapter.clean_collection(
            mock_vectorstore_wrapper,
            index_name="",
            including_index_meta=True
        )

        mock_vectorstore_wrapper.vectorstore.get.assert_called_once_with(include=[])
        mock_vectorstore_wrapper.vectorstore.delete.assert_called_once_with(ids=['id1', 'id2', 'id3'])
        assert deleted_count == 3


class TestChromaAdapterErrorPropagation:
    """Tests for ChromaAdapter error propagation behavior."""

    @pytest.fixture
    def adapter(self):
        return ChromaAdapter()

    @pytest.fixture
    def mock_vectorstore_wrapper(self):
        wrapper = MagicMock()
        wrapper.vectorstore = MagicMock()
        return wrapper

    def test_clean_collection_propagates_get_error(self, adapter, mock_vectorstore_wrapper):
        """Backend .get() failures should propagate, not be swallowed."""
        mock_vectorstore_wrapper.vectorstore.get.side_effect = Exception("Chroma connection failed")

        with pytest.raises(Exception, match="Chroma connection failed"):
            adapter.clean_collection(mock_vectorstore_wrapper, index_name="foo", including_index_meta=True)

    def test_clean_collection_propagates_delete_error(self, adapter, mock_vectorstore_wrapper):
        """Backend .delete() failures should propagate, not be swallowed."""
        mock_vectorstore_wrapper.vectorstore.get.return_value = {
            'ids': ['id1'],
            'metadatas': [{'collection': 'foo'}]
        }
        mock_vectorstore_wrapper.vectorstore.delete.side_effect = Exception("Chroma delete failed")

        with pytest.raises(Exception, match="Chroma delete failed"):
            adapter.clean_collection(mock_vectorstore_wrapper, index_name="foo", including_index_meta=True)

    def test_full_cleanup_propagates_get_error(self, adapter, mock_vectorstore_wrapper):
        """Full cleanup (no index_name) should propagate .get() failures."""
        mock_vectorstore_wrapper.vectorstore.get.side_effect = Exception("Chroma unavailable")

        with pytest.raises(Exception, match="Chroma unavailable"):
            adapter.clean_collection(mock_vectorstore_wrapper, index_name="", including_index_meta=True)


class TestChromaAdapterMissingIndexBehavior:
    """
    Tests for ChromaAdapter behavior when index doesn't exist.

    Verifies that clean_collection returns 0 for missing indexes,
    enabling callers (remove_index tool) to detect and raise ToolException.
    """

    @pytest.fixture
    def mock_chroma_adapter(self):
        """Create a ChromaAdapter with mocked vectorstore."""
        adapter = ChromaAdapter()
        return adapter

    def test_missing_index_returns_zero_deleted_count(self, mock_chroma_adapter):
        """
        When index "missing" doesn't exist, clean_collection should return 0.
        This enables the remove_index tool to detect and raise ToolException.
        """
        wrapper = MagicMock()
        wrapper.vectorstore.get.return_value = {'ids': [], 'metadatas': []}

        deleted_count = mock_chroma_adapter.clean_collection(
            wrapper,
            index_name="missing",
            including_index_meta=True
        )

        assert deleted_count == 0


class TestChromaAdapterGetIndexedIds:
    """Tests for ChromaAdapter.get_indexed_ids() filtering behavior."""

    @pytest.fixture
    def adapter(self):
        return ChromaAdapter()

    @pytest.fixture
    def mock_vectorstore_wrapper(self):
        wrapper = MagicMock()
        wrapper.vectorstore = MagicMock()
        return wrapper

    def test_get_indexed_ids_filters_by_index_name(self, adapter, mock_vectorstore_wrapper):
        """get_indexed_ids with index_name should filter by collection metadata."""
        mock_vectorstore_wrapper.vectorstore.get.return_value = {
            'ids': ['id1', 'id2']
        }

        ids = adapter.get_indexed_ids(mock_vectorstore_wrapper, index_name="foo")

        mock_vectorstore_wrapper.vectorstore.get.assert_called_once_with(
            where={"collection": "foo"},
            include=[]
        )
        assert ids == ['id1', 'id2']

    def test_get_indexed_ids_no_filter_when_no_index_name(self, adapter, mock_vectorstore_wrapper):
        """get_indexed_ids without index_name should return all IDs."""
        mock_vectorstore_wrapper.vectorstore.get.return_value = {
            'ids': ['id1', 'id2', 'id3']
        }

        ids = adapter.get_indexed_ids(mock_vectorstore_wrapper, index_name="")

        mock_vectorstore_wrapper.vectorstore.get.assert_called_once_with(include=[])
        assert ids == ['id1', 'id2', 'id3']
