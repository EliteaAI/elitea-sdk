import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import ToolException

from elitea_sdk.runtime.utils.utils import IndexerKeywords
from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit, IndexingStats

logger = logging.getLogger(__name__)


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
        same_updated_on = (
            document.metadata.get('updated_on')
            and idx_data['metadata'].get('updated_on')
            and document.metadata.get('updated_on') == idx_data['metadata'].get('updated_on')
        )
        if not same_updated_on:
            return False
        # Same updated_on alone would let _reduce_duplicates skip the parent,
        # which also skips _process_document — so attachments/children fetched
        # there would never be reprocessed when their set diverges from the
        # indexed copy (e.g., toggling include_attachments True on a rerun,
        # or a source-side add/remove that didn't bump the parent's
        # updated_on). Subclasses that emit dependents opt in via
        # _dependents_diverged.
        try:
            if self._dependents_diverged(document, idx_data):
                return False
        except Exception as e:
            logger.warning(
                f"Failed to check dependent divergence for "
                f"{document.metadata.get('id')}: {e}. Treating as changed."
            )
            return False
        return True

    def _dependents_diverged(self, document: Document, idx_data) -> bool:
        """Return True to force reprocessing of a doc whose updated_on is
        unchanged but whose dependent set (attachments, images, comments,
        etc.) has diverged from the stored copy. Default False (opt out —
        dedup uses updated_on alone).

        Override in subclasses that emit dependent documents. Each
        subclass owns its own diff strategy: strict set-equality,
        prefix-filtered subset (when multiple dep types share
        dependent_docs), presence-only check, etc. Read the stored set
        via idx_data[IndexerKeywords.DEPENDENT_DOCS.value] (already
        list-parsed by the vector adapter)."""
        return False

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
        if reason == "filtered":
            self._indexing_stats.documents_skipped_filtered.add(doc_id)
        else:
            self._indexing_stats.documents_skipped_error.add(doc_id)

    def _track_runtime_skipped(self, item_name: str, reason: str = "extension"):
        """Track a runtime skipped item during indexing (e.g., attachments, artifacts)."""
        if not hasattr(self, '_indexing_stats'):
            self._init_indexing_stats()
        if reason == "extension":
            self._indexing_stats.runtime_skipped_extension.add(item_name)
        else:
            self._indexing_stats.runtime_skipped_error.add(item_name)

    # Backward compatibility alias
    def _track_skipped_attachment(self, attachment_name: str, reason: str = "extension"):
        """Deprecated: Use _track_runtime_skipped instead."""
        self._track_runtime_skipped(attachment_name, reason)

    def _track_skipped_file_unsupported(self, file_name: str):
        """Track a file skipped due to unsupported extension."""
        if not hasattr(self, '_indexing_stats'):
            self._init_indexing_stats()
        self._indexing_stats.files_unsupported_extension.add(file_name)

    def _track_skipped_file_read_error(self, file_name: str):
        """Track a file skipped due to read error."""
        if not hasattr(self, '_indexing_stats'):
            self._init_indexing_stats()
        self._indexing_stats.files_skipped_read_error.add(file_name)

    def _track_skipped_file_empty(self, file_name: str):
        """Track a file skipped due to empty content."""
        if not hasattr(self, '_indexing_stats'):
            self._init_indexing_stats()
        self._indexing_stats.files_skipped_empty.add(file_name)

    def _track_processed_item(self):
        """Increment the count of processed items."""
        if not hasattr(self, '_indexing_stats'):
            self._init_indexing_stats()
        self._indexing_stats.items_processed += 1

    def _track_document_unchanged(self, doc_identifier: str):
        """Track a document matched by incremental dedup — same updated_on as the
        indexed copy, so we skipped re-indexing it. Not a failure; counted separately
        from documents_skipped_* so the report can distinguish 'nothing to do' from
        'something went wrong'."""
        if not hasattr(self, '_indexing_stats'):
            self._init_indexing_stats()
        self._indexing_stats.documents_already_indexed.add(doc_identifier)

    def _track_dependent_item_skipped(self, item_name: str):
        """
        Track a dependent/child item that failed within a successful parent document.

        Use this for sub-items like individual images in a Figma file or attachments
        in a Confluence page where the parent document was still indexed successfully.
        These are tracked separately from top-level skipped items.
        """
        if not hasattr(self, '_indexing_stats'):
            self._init_indexing_stats()
        self._indexing_stats.dependent_items_skipped.add(item_name)
