from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import ToolException

from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit, IndexingStats


class NonCodeIndexerToolkit(BaseIndexerToolkit):
    """
    Base class for non-code indexer toolkits (Jira, Confluence, SharePoint, etc.).
    Provides document/attachment tracking capabilities.
    """

    def _get_indexed_data(self, index_name: str):
        self._ensure_vectorstore_initialized()
        if not self.vector_adapter:
            raise ToolException("Vector adapter is not initialized. "
                             "Check your configuration: embedding_model and vectorstore_type.")
        return self.vector_adapter.get_indexed_data(self, index_name)

    def key_fn(self, document: Document):
        return document.metadata.get('id')

    def compare_fn(self, document: Document, idx_data):
        return (document.metadata.get('updated_on')
                and idx_data['metadata'].get('updated_on')
                and document.metadata.get('updated_on') == idx_data['metadata'].get('updated_on'))

    def remove_ids_fn(self, idx_data, key: str):
        return (idx_data[key]['all_chunks'] +
                [idx_data[dep_id]['id'] for dep_id in idx_data[key][IndexerKeywords.DEPENDENT_DOCS.value]] +
                [chunk_db_id for dep_id in idx_data[key][IndexerKeywords.DEPENDENT_DOCS.value] for chunk_db_id in
                 idx_data[dep_id]['all_chunks']])

    def _init_indexing_stats(self) -> IndexingStats:
        """Initialize or reset indexing stats for this indexing run."""
        self._indexing_stats = IndexingStats()
        return self._indexing_stats

    def get_indexing_stats(self) -> Optional[IndexingStats]:
        """Get the indexing statistics from the last indexing run."""
        return getattr(self, '_indexing_stats', None)

    def get_indexing_stats_summary(self) -> str:
        """Get a human-readable summary of skipped items."""
        stats = self.get_indexing_stats()
        return stats.get_summary() if stats else ""

    def _track_skipped_document(self, doc_id: str, reason: str = "error"):
        """Track a skipped document during indexing."""
        if not hasattr(self, '_indexing_stats'):
            self._init_indexing_stats()
        self._indexing_stats.documents_skipped_error.append(doc_id)

    def _track_runtime_skipped(self, item_name: str, reason: str = "extension"):
        """Track a runtime skipped item during indexing (e.g., attachments, artifacts)."""
        if not hasattr(self, '_indexing_stats'):
            self._init_indexing_stats()
        if reason == "extension":
            self._indexing_stats.runtime_skipped_extension.append(item_name)
        else:
            self._indexing_stats.runtime_skipped_error.append(item_name)

    # Backward compatibility alias
    def _track_skipped_attachment(self, attachment_name: str, reason: str = "extension"):
        """Deprecated: Use _track_runtime_skipped instead."""
        self._track_runtime_skipped(attachment_name, reason)

    def _track_skipped_file_unsupported(self, file_name: str):
        """Track a file skipped due to unsupported extension."""
        if not hasattr(self, '_indexing_stats'):
            self._init_indexing_stats()
        self._indexing_stats.files_unsupported_extension.append(file_name)

    def _track_skipped_file_read_error(self, file_name: str):
        """Track a file skipped due to read error."""
        if not hasattr(self, '_indexing_stats'):
            self._init_indexing_stats()
        self._indexing_stats.files_skipped_read_error.append(file_name)

    def _track_skipped_file_empty(self, file_name: str):
        """Track a file skipped due to empty content."""
        if not hasattr(self, '_indexing_stats'):
            self._init_indexing_stats()
        self._indexing_stats.files_skipped_empty.append(file_name)

    def _track_processed_item(self):
        """Increment the count of processed items."""
        if not hasattr(self, '_indexing_stats'):
            self._init_indexing_stats()
        self._indexing_stats.items_processed += 1
