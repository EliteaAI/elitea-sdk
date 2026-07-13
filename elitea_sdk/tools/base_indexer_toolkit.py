import copy
import json
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, List, Dict, Generator, Set

from langchain_core.callbacks import dispatch_custom_event
from langchain_core.documents import Document
from langchain_core.tools import ToolException
from pydantic import create_model, Field, SecretStr

from .utils.content_parser import file_extension_by_chunker, process_document_by_type
from .vector_adapters.VectorStoreAdapter import VectorStoreAdapterFactory
from ..runtime.langchain.document_loaders.constants import loaders_allowed_to_override
from ..runtime.tools.vectorstore_base import VectorStoreWrapperBase
from ..runtime.utils.utils import IndexerKeywords

logger = logging.getLogger(__name__)


@dataclass
class IndexingStats:
    """
    Tracks statistics during indexing process.
    Used by both CodeIndexerToolkit and NonCodeIndexerToolkit.

    Terminology:
    - total_fetched: All items initially fetched/considered from source
    - items_processed: Items successfully processed and yielded (after filtering)
    - total_skipped: Top-level items that were filtered out or failed entirely
    - dependent_items_skipped: Child/sub-items that failed within a successful parent document
      (e.g., individual images in a Figma file, attachments in a Confluence page)

    Invariant: total_fetched = items_processed + total_skipped
    Note: dependent_items_skipped is NOT included in total_skipped since parent docs succeeded
    """
    # Common counters
    items_processed: int = 0
    total_fetched: int = 0  # All items from source before any filtering

    # For code toolkits (files) - Use sets to deduplicate entries
    files_skipped_whitelist: Set[str] = field(default_factory=set)
    files_skipped_blacklist: Set[str] = field(default_factory=set)
    files_skipped_read_error: Set[str] = field(default_factory=set)
    files_skipped_empty: Set[str] = field(default_factory=set)
    files_unsupported_extension: Set[str] = field(default_factory=set)

    # For non-code toolkits (documents/runtime)
    documents_skipped_error: Set[str] = field(default_factory=set)
    documents_skipped_filtered: Set[str] = field(default_factory=set)
    runtime_skipped_extension: Set[str] = field(default_factory=set)
    runtime_skipped_error: Set[str] = field(default_factory=set)

    # Documents already indexed with the same updated_on hash — matched by
    # incremental dedup (clean_index=False) and intentionally not re-indexed.
    # Not a failure: parent is unchanged, so we skip it to save work.
    documents_already_indexed: Set[str] = field(default_factory=set)

    # Dependent/child items that failed within successful parent documents
    # These are tracked separately since the parent document was still indexed
    dependent_items_skipped: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict:
        """Convert stats to dictionary for reporting."""
        # Calculate counts for each category
        files_skipped_count = (
            len(self.files_skipped_whitelist) +
            len(self.files_skipped_blacklist) +
            len(self.files_skipped_read_error) +
            len(self.files_skipped_empty) +
            len(self.files_unsupported_extension)
        )
        documents_skipped_count = (
            len(self.documents_skipped_error) +
            len(self.documents_skipped_filtered)
        )
        runtime_skipped_count = (
            len(self.runtime_skipped_extension) +
            len(self.runtime_skipped_error)
        )
        # total_skipped only includes top-level items, not dependent items
        total_skipped = files_skipped_count + documents_skipped_count + runtime_skipped_count
        dependent_items_count = len(self.dependent_items_skipped)

        return {
            "items_processed": self.items_processed,
            "total_fetched": self.total_fetched,
            "total_skipped": total_skipped,
            "files_skipped": {
                "count": files_skipped_count,
                "whitelist_filtered": sorted(self.files_skipped_whitelist),
                "whitelist_filtered_count": len(self.files_skipped_whitelist),
                "blacklist_filtered": sorted(self.files_skipped_blacklist),
                "blacklist_filtered_count": len(self.files_skipped_blacklist),
                "read_error": sorted(self.files_skipped_read_error),
                "read_error_count": len(self.files_skipped_read_error),
                "empty_content": sorted(self.files_skipped_empty),
                "empty_content_count": len(self.files_skipped_empty),
                "unsupported_extension": sorted(self.files_unsupported_extension),
                "unsupported_extension_count": len(self.files_unsupported_extension),
            },
            "documents_skipped": {
                "count": documents_skipped_count,
                "error": sorted(self.documents_skipped_error),
                "error_count": len(self.documents_skipped_error),
                "filtered": sorted(self.documents_skipped_filtered),
                "filtered_count": len(self.documents_skipped_filtered),
            },
            "runtime_skipped": {
                "count": runtime_skipped_count,
                "extension_filtered": sorted(self.runtime_skipped_extension),
                "extension_filtered_count": len(self.runtime_skipped_extension),
                "error": sorted(self.runtime_skipped_error),
                "error_count": len(self.runtime_skipped_error),
            },
            "dependent_items_skipped": {
                "count": dependent_items_count,
                "items": sorted(self.dependent_items_skipped),
            },
            "documents_already_indexed": {
                "count": len(self.documents_already_indexed),
                "items": sorted(self.documents_already_indexed),
            },
        }

    def get_summary(self) -> str:
        """Generate human-readable summary of skipped items."""
        lines = []

        # Count file-related skips (top-level)
        file_skips = (len(self.files_skipped_whitelist) +
                     len(self.files_skipped_blacklist) +
                     len(self.files_skipped_read_error) +
                     len(self.files_skipped_empty) +
                     len(self.files_unsupported_extension))

        # Count document/runtime-related skips (top-level)
        doc_skips = (len(self.documents_skipped_error) +
                    len(self.documents_skipped_filtered) +
                    len(self.runtime_skipped_extension) +
                    len(self.runtime_skipped_error))

        total_skipped = file_skips + doc_skips
        dependent_skipped = len(self.dependent_items_skipped)

        if total_skipped == 0 and dependent_skipped == 0:
            return ""

        if total_skipped > 0:
            lines.append(f"\nSkipped items ({total_skipped} total):")

        # File-related skips (for code toolkits)
        if self.files_skipped_whitelist:
            sorted_whitelist = sorted(self.files_skipped_whitelist)
            lines.append(f"  - Files not in whitelist ({len(sorted_whitelist)}): {', '.join(sorted_whitelist[:5])}")
            if len(sorted_whitelist) > 5:
                lines.append(f"    ... and {len(sorted_whitelist) - 5} more")

        if self.files_skipped_blacklist:
            sorted_blacklist = sorted(self.files_skipped_blacklist)
            lines.append(f"  - Files blacklisted ({len(sorted_blacklist)}): {', '.join(sorted_blacklist[:5])}")
            if len(sorted_blacklist) > 5:
                lines.append(f"    ... and {len(sorted_blacklist) - 5} more")

        if self.files_skipped_read_error:
            sorted_read_error = sorted(self.files_skipped_read_error)
            lines.append(f"  - Files with read errors ({len(sorted_read_error)}): {', '.join(sorted_read_error[:5])}")
            if len(sorted_read_error) > 5:
                lines.append(f"    ... and {len(sorted_read_error) - 5} more")

        if self.files_skipped_empty:
            sorted_empty = sorted(self.files_skipped_empty)
            lines.append(f"  - Files with empty content ({len(sorted_empty)}): {', '.join(sorted_empty[:5])}")
            if len(sorted_empty) > 5:
                lines.append(f"    ... and {len(sorted_empty) - 5} more")

        if self.files_unsupported_extension:
            sorted_unsupported = sorted(self.files_unsupported_extension)
            lines.append(f"  - Files with unsupported extension ({len(sorted_unsupported)}): {', '.join(sorted_unsupported[:5])}")
            if len(sorted_unsupported) > 5:
                lines.append(f"    ... and {len(sorted_unsupported) - 5} more")

        # Document/attachment-related skips (for non-code toolkits)
        if self.documents_skipped_error:
            sorted_doc_error = sorted(self.documents_skipped_error)
            lines.append(f"  - Documents with errors ({len(sorted_doc_error)}): {', '.join(sorted_doc_error[:5])}")
            if len(sorted_doc_error) > 5:
                lines.append(f"    ... and {len(sorted_doc_error) - 5} more")

        if self.documents_skipped_filtered:
            sorted_doc_filtered = sorted(self.documents_skipped_filtered)
            lines.append(f"  - Documents filtered out ({len(sorted_doc_filtered)}): {', '.join(sorted_doc_filtered[:5])}")
            if len(sorted_doc_filtered) > 5:
                lines.append(f"    ... and {len(sorted_doc_filtered) - 5} more")

        if self.runtime_skipped_extension:
            sorted_runtime_ext = sorted(self.runtime_skipped_extension)
            lines.append(f"  - Runtime skipped (extension) ({len(sorted_runtime_ext)}): {', '.join(sorted_runtime_ext[:5])}")
            if len(sorted_runtime_ext) > 5:
                lines.append(f"    ... and {len(sorted_runtime_ext) - 5} more")

        if self.runtime_skipped_error:
            sorted_runtime_err = sorted(self.runtime_skipped_error)
            lines.append(f"  - Runtime skipped (errors) ({len(sorted_runtime_err)}): {', '.join(sorted_runtime_err[:5])}")
            if len(sorted_runtime_err) > 5:
                lines.append(f"    ... and {len(sorted_runtime_err) - 5} more")

        # Dependent items (sub-items within successful parent documents)
        if self.dependent_items_skipped:
            sorted_dependent = sorted(self.dependent_items_skipped)
            lines.append(f"\nSkipped sub-items ({len(sorted_dependent)} total, parent docs still indexed):")
            lines.append(f"  - Failed sub-items: {', '.join(sorted_dependent[:5])}")
            if len(sorted_dependent) > 5:
                lines.append(f"    ... and {len(sorted_dependent) - 5} more")

        # Documents that matched by incremental dedup — not skipped in the failure
        # sense, so tracked in their own section to keep counts unambiguous.
        if self.documents_already_indexed:
            sorted_unchanged = sorted(self.documents_already_indexed)
            lines.append(f"\nAlready indexed, unchanged ({len(sorted_unchanged)}): "
                         f"{', '.join(sorted_unchanged[:5])}")
            if len(sorted_unchanged) > 5:
                lines.append(f"    ... and {len(sorted_unchanged) - 5} more")

        return "\n".join(lines)

DEFAULT_CUT_OFF = 0.1
INDEX_META_UPDATE_INTERVAL = 600.0

class IndexTools(str, Enum):
    """Enum for index-related tool names."""
    INDEX_DATA = "index_data"
    SEARCH_INDEX = "search_index"
    STEPBACK_SEARCH_INDEX = "stepback_search_index"
    STEPBACK_SUMMARY_INDEX = "stepback_summary_index"
    REMOVE_INDEX = "remove_index"
    LIST_COLLECTIONS = "list_collections"

RemoveIndexParams = create_model(
    "RemoveIndexParams",
    index_name=(Optional[str], Field(description="Optional index name (max 7 characters)", default="", max_length=7)),
)

BaseSearchParams = create_model(
    "BaseSearchParams",
    query=(str, Field(description="Query text to search in the index")),
    index_name=(Optional[str], Field(
        description="Optional index name (max 7 characters). Leave empty to search across all datasets",
        default="", max_length=7)),
    filter=(Optional[dict | str], Field(
        description="Filter to apply to the search results. Can be a dictionary or a JSON string.",
        default={},
        examples=["{\"key\": \"value\"}", "{\"status\": \"active\"}"]
    )),
    cut_off=(Optional[float], Field(description="Cut-off score for search results", default=DEFAULT_CUT_OFF, ge=0, le=1)),
    search_top=(Optional[int], Field(description="Number of top results to return", default=10, gt=0)),
    full_text_search=(Optional[Dict[str, Any]], Field(
        description="Full text search parameters. Can be a dictionary with search options.",
        default=None
    )),
    extended_search=(Optional[List[str]], Field(
        description="List of additional fields to include in the search results.",
        default=None
    )),
    reranker=(Optional[dict], Field(
        description="Reranker configuration. Can be a dictionary with reranking parameters.",
        default={}
    )),
    reranking_config=(Optional[Dict[str, Dict[str, Any]]], Field(
        description="Reranking configuration. Can be a dictionary with reranking settings.",
        default=None
    )),
    output_fields=(Optional[List[str]], Field(
        description="Fields to include in output. Supports: 'page_content', 'score', 'metadata' (all metadata), "
                    "or 'metadata.<field>' for specific metadata fields (e.g., 'metadata.source'). "
                    "If None or empty, returns all fields.",
        default=None,
        examples=[["metadata", "score"], ["page_content", "metadata.source"], ["metadata.id", "metadata.source"]]
    )),
)

BaseStepbackSearchParams = create_model(
    "BaseStepbackSearchParams",
    query=(str, Field(description="Query text to search in the index")),
    index_name=(Optional[str], Field(description="Optional index name (max 7 characters)", default="", max_length=7)),
    messages=(Optional[List], Field(description="Chat messages for stepback search context", default=[])),
    filter=(Optional[dict | str], Field(
        description="Filter to apply to the search results. Can be a dictionary or a JSON string.",
        default={},
        examples=["{\"key\": \"value\"}", "{\"status\": \"active\"}"]
    )),
    cut_off=(Optional[float], Field(description="Cut-off score for search results", default=DEFAULT_CUT_OFF, ge=0, le=1)),
    search_top=(Optional[int], Field(description="Number of top results to return", default=10, gt=0)),
    full_text_search=(Optional[Dict[str, Any]], Field(
        description="Full text search parameters. Can be a dictionary with search options.",
        default=None
    )),
    extended_search=(Optional[List[str]], Field(
        description="List of additional fields to include in the search results.",
        default=None
    )),
    reranker=(Optional[dict], Field(
            description="Reranker configuration. Can be a dictionary with reranking parameters.",
            default={}
        )),
    reranking_config=(Optional[Dict[str, Dict[str, Any]]], Field(
            description="Reranking configuration. Can be a dictionary with reranking settings.",
            default=None
        )),
)


class BaseIndexerToolkit(VectorStoreWrapperBase):
    """Base class for tool API wrappers that support vector store functionality."""

    doctype: str = "document"

    connection_string: Optional[SecretStr] = None
    collection_name: Optional[str] = None
    elitea: Any = None # Elitea client, if available

    def __init__(self, **kwargs):
        conn = kwargs.get('connection_string', None)
        connection_string = conn.get_secret_value() if isinstance(conn, SecretStr) else conn
        collection_name = kwargs.get('collection_schema')
        
        if 'vectorstore_type' not in kwargs:
            kwargs['vectorstore_type'] = 'PGVector'
        vectorstore_type = kwargs.get('vectorstore_type')
        if connection_string:
            # Initialize vectorstore params only if connection string is provided
            kwargs['vectorstore_params'] = VectorStoreAdapterFactory.create_adapter(vectorstore_type).get_vectorstore_params(collection_name, connection_string)
        super().__init__(**kwargs)

    def _index_tool_params(self, **kwargs) -> dict[str, tuple[type, Field]]:
        """
        Returns a list of fields for index_data args schema.
        NOTE: override this method in subclasses to provide specific parameters for certain toolkit.
        """
        return {}

    def _has_collections(self) -> bool:
        """
        Safely check if there are any indexed collections for this toolkit.

        Returns:
            bool: True if collections exist, False otherwise.
        """
        try:
            self._ensure_vectorstore_initialized()
            collections = self.vector_adapter.list_collections(self)
            return bool(collections and len(collections) > 0)
        except Exception as e:
            logger.debug(f"Could not check collections (vectorstore may not be configured): {e}")
            return False

    def _remove_metadata_keys(self) -> List[str]:
        """ Returns a list of metadata keys to be removed from documents before indexing.
        Override this method in subclasses to provide specific keys to remove."""
        return [IndexerKeywords.CONTENT_IN_BYTES.value, IndexerKeywords.CONTENT_FILE_NAME.value]

    def _base_loader(self, **kwargs) -> Generator[Document, None, None]:
        """ Loads documents from a source, processes them,
        and returns a list of Document objects with base metadata: id and created_on."""
        yield from ()

    def _process_document(self, base_document: Document) -> Generator[Document, None, None]:
        """ Process an existing base document to extract relevant metadata for full document preparation.
        Used for late processing of documents after we ensure that the document has to be indexed to avoid
        time-consuming operations for documents which might be useless.

        Args:
            document (Document): The base document to process.

        Returns:
            Document: The processed document with metadata."""
        yield from ()

    def index_data(self, **kwargs):
        index_name = kwargs.get("index_name")
        # Reject empty/None index_name early — otherwise it gets formatted into the
        # index_meta document's page_content as the literal string "index_meta_None",
        # poisoning the collection with unattached meta rows.
        if not index_name or not str(index_name).strip():
            raise ToolException(
                "index_data requires a non-empty 'index_name' (the collection suffix)."
            )
        clean_index = kwargs.get("clean_index")
        chunking_tool = kwargs.get("chunking_tool")
        chunking_config = kwargs.get("chunking_config")

        # Store the interval in a private dict to avoid Pydantic field errors
        if not hasattr(self, "_index_meta_config"):
            self._index_meta_config: Dict[str, Any] = {}

        self._index_meta_config["update_interval"] = kwargs.get(
            "meta_update_interval",
            INDEX_META_UPDATE_INTERVAL,
        )

        result = {"count": 0, "failed_count": 0, "docs_count": 0}
        #
        try:
            if clean_index:
                self._clean_index(index_name)
            #
            self.index_meta_init(index_name, kwargs)
            self._emit_index_event(index_name)
            #
            self._log_tool_event(f"Indexing data into collection with suffix '{index_name}'. It can take some time...")
            self._log_tool_event(f"Loading the documents to index...{kwargs}")
            documents = self._base_loader(**kwargs)
            documents = list(documents) # consume/exhaust generator to count items
            documents_count = len(documents)
            documents = (doc for doc in documents)
            self._log_tool_event(f"Base documents were pre-loaded. "
                                 f"Search for possible document duplicates and remove them from the indexing list...")
            documents = self._reduce_duplicates(documents, index_name)
            self._log_tool_event(f"Duplicates were removed. "
                                 f"Processing documents to collect dependencies and prepare them for indexing...")
            self._save_index_generator(documents, documents_count, chunking_tool, chunking_config, index_name=index_name, result=result)
            #
            chunks_count = result["count"]
            failed_chunks_count = result.get("failed_count", 0)
            succeeded_chunks_count = chunks_count - failed_chunks_count
            docs_count = result.get("docs_count", 0)
            errors = result.get("errors", [])
            issues_detail = ("\nIssues: " + "; ".join(errors)) if errors else ""

            # Get skipped files summary and data if available
            skipped_summary = ""
            skipped_data = None
            if hasattr(self, 'get_indexing_stats_summary') and callable(self.get_indexing_stats_summary):
                skipped_summary = self.get_indexing_stats_summary()
            if hasattr(self, 'get_indexing_stats') and callable(self.get_indexing_stats):
                stats = self.get_indexing_stats()
                if stats:
                    skipped_data = stats.to_dict()

            # For code indexer: chunking happens inside _base_loader via universal_chunker,
            # so docs_count counts chunks instead of files. Use items_processed from stats
            # which tracks actual file count.
            # Detection: if docs_count equals chunks and items_processed differs, it's code indexer.
            # Subtract already-indexed (unchanged, dedup-matched) docs from items_processed so
            # that a non-code indexer with all-unchanged docs (docs_count=0, chunks=0,
            # items_processed=N) doesn't get its docs_count inflated back to N.
            unchanged_count = (
                skipped_data.get("documents_already_indexed", {}).get("count", 0)
                if skipped_data else 0
            )
            effective_processed = (
                (skipped_data.get("items_processed", 0) - unchanged_count)
                if skipped_data else 0
            )
            if (skipped_data and effective_processed > 0
                    and docs_count == succeeded_chunks_count
                    and docs_count != effective_processed):
                docs_count = effective_processed

            unchanged_detail = (
                f" {unchanged_count} document(s) already indexed (unchanged)."
                if unchanged_count > 0 else ""
            )

            # Use docs_count for user-facing messages (number of documents)
            # Use succeeded_chunks_count for internal tracking (number of chunks in vector store)
            if failed_chunks_count > 0 and succeeded_chunks_count > 0:
                final_state = IndexerKeywords.INDEX_META_PARTLY_OK.value
                status = "partly_indexed"
                message = (f"Successfully indexed {docs_count} documents ({succeeded_chunks_count} chunks)."
                           f"{unchanged_detail} "
                           f"Failed to index {failed_chunks_count} chunks.{issues_detail}{skipped_summary}")
            elif failed_chunks_count > 0 >= succeeded_chunks_count:
                final_state = IndexerKeywords.INDEX_META_FAILED.value
                status = "error"
                message = f"Failed to index documents ({failed_chunks_count} chunks failed).{issues_detail}{skipped_summary}"
            elif docs_count > 0:
                final_state = IndexerKeywords.INDEX_META_COMPLETED.value
                status = "ok"
                message = (f"Successfully indexed {docs_count} documents ({succeeded_chunks_count} chunks)."
                           f"{unchanged_detail}{skipped_summary}")
            elif unchanged_count > 0:
                final_state = IndexerKeywords.INDEX_META_COMPLETED.value
                status = "ok"
                message = (f"No new documents to index; {unchanged_count} document(s) already indexed "
                           f"(unchanged).{skipped_summary}")
            else:
                final_state = IndexerKeywords.INDEX_META_COMPLETED.value
                status = "ok"
                message = f"No new documents to index.{skipped_summary}"

            # Final update should always be forced (pass chunks count for indexed_chunks field).
            # Include unchanged docs in the indexed count so the UI reflects total items
            # currently in the vector store, not just newly indexed ones.
            indexed_total = docs_count + unchanged_count
            self.index_meta_update(index_name, final_state, succeeded_chunks_count, update_force=True,
                                   error=message if status != "ok" else None, skipped=skipped_data,
                                   docs_count=indexed_total)
            self._emit_index_event(index_name)
            #
            return {"status": status, "message": message}
        except Exception as e:
            # Do maximum effort at least send custom event for supposed changed status
            msg = str(e)
            try:
                # Error update should also be forced and include the error message
                self.index_meta_update(index_name, IndexerKeywords.INDEX_META_FAILED.value, result["count"], update_force=True, error=msg)
            except Exception as ie:
                logger.error(f"Failed to update index meta status to FAILED for index '{index_name}': {ie}")
                msg = f"{msg}; additionally failed to update index meta status to FAILED: {ie}"
            self._emit_index_event(index_name, error=msg)
            raise e

    def _save_index_generator(self, base_documents: Generator[Document, None, None], base_total: int, chunking_tool, chunking_config, result, index_name: Optional[str] = None):
        self._ensure_vectorstore_initialized()
        self._log_tool_event(f"Base documents are ready for indexing. {base_total} base documents in total to index.")
        from ..runtime.langchain.interfaces.llm_processor import add_documents
        #
        pg_vector_add_docs_chunk: list = []

        def _flush_chunk(chunk: list):
            """Flush a chunk of documents to the vectorstore, tracking failures in result."""
            if not chunk:
                return
            try:
                add_documents(vectorstore=self.vectorstore, documents=chunk)
                self._log_tool_event(f"{len(chunk)} documents have been indexed. Continuing...")
            except Exception as exc:
                from traceback import format_exc
                err = format_exc()
                logger.error(f"Failed to add {len(chunk)} documents to vectorstore: {err}")
                result["failed_count"] = result.get("failed_count", 0) + len(chunk)
                error_msg = str(exc)
                if error_msg not in result.setdefault("errors", []):
                    result["errors"].append(error_msg)

        def _run_pipeline(base_doc: Document) -> List[Document]:
            """Full serial pipeline for one base doc → materialized chunk list.
            Safe to invoke from a worker thread: the nested-pool guards inside
            ``_collect_dependencies`` and ``_apply_loaders_chunkers`` force the
            serial branch when called off the main thread, so we never nest
            executors. All side effects (skip trackers, metadata mutation on
            the doc's own dict) are thread-safe: set.add is GIL-safe and each
            worker touches only its own doc."""
            docs_gen = self._extend_data((base_doc for _ in range(1)))
            docs_gen = self._collect_dependencies(docs_gen)
            docs_gen = self._apply_loaders_chunkers(docs_gen, chunking_tool, chunking_config)
            docs_gen = self._clean_metadata(docs_gen)
            return list(docs_gen)

        def _consume_pipeline_output(base_doc: Document, base_doc_counter: int, chunks: List[Document]) -> None:
            """Main-thread-only. Applies index_name/collection metadata, buffers
            chunks, and updates per-base-doc counters + skip trackers. Kept
            single-writer to preserve original ordering of side effects."""
            _doc_name = self._extract_doc_name(base_doc.metadata)
            logger.debug(f"Indexing base document #{base_doc_counter}: {base_doc} with {len(chunks)} dependent chunks")

            dependent_docs_counter = 0
            for doc in chunks:
                if not doc.page_content:
                    # To avoid case when all documents have empty content
                    # See llm_processor.add_documents which exclude metadata of docs with empty content
                    continue
                if 'id' not in doc.metadata or 'updated_on' not in doc.metadata:
                    logger.warning(f"Document is missing required metadata field 'id' or 'updated_on': {doc.metadata}")
                if index_name:
                    if not doc.metadata.get('collection'):
                        doc.metadata['collection'] = index_name
                    else:
                        doc.metadata['collection'] += f";{index_name}"
                pg_vector_add_docs_chunk.append(doc)
                dependent_docs_counter += 1
                if len(pg_vector_add_docs_chunk) >= self.max_docs_per_add:
                    _flush_chunk(pg_vector_add_docs_chunk)
                    pg_vector_add_docs_chunk.clear()

            msg = f"Indexed document #{base_doc_counter} '{_doc_name}' out of {base_total} (with {dependent_docs_counter} chunks)."
            logger.debug(msg)
            self._log_tool_event(msg)
            result["count"] += dependent_docs_counter
            if dependent_docs_counter > 0:
                result["docs_count"] += 1
            else:
                # Base doc yielded zero chunks (empty content, chunker returned nothing,
                # or every chunk got filtered as empty/parse-error). Track as skipped so
                # `total_fetched = items_processed + total_skipped` still holds. Skip if
                # the doc was already tracked deeper to avoid double-counting.
                if not self._is_base_doc_tracked_as_skipped(base_doc):
                    if hasattr(self, '_track_skipped_document'):
                        self._track_skipped_document(_doc_name, reason="error")
            try:
                self.index_meta_update(index_name, IndexerKeywords.INDEX_META_IN_PROGRESS.value, result["count"], update_force=False)
            except Exception as exc:  # best-effort, do not break indexing
                logger.warning(f"Failed to update index meta during indexing process for index '{index_name}': {exc}")

        workers = getattr(self, "_index_workers", 1) or 1

        if workers <= 1:
            # Serial path — identical semantics to the pre-Phase-6 loop.
            base_doc_counter = 0
            for base_doc in base_documents:
                base_doc_counter += 1
                _doc_name = self._extract_doc_name(base_doc.metadata)
                self._log_tool_event(f"Processing document #{base_doc_counter}: '{_doc_name}'.")
                self._log_tool_event(
                    f"Dependent documents for '{_doc_name}' were processed. "
                    f"Applying chunking tool '{chunking_tool if chunking_tool else 'default'}' if specified and preparing documents for indexing..."
                )
                chunks = _run_pipeline(base_doc)
                _consume_pipeline_output(base_doc, base_doc_counter, chunks)
            if pg_vector_add_docs_chunk:
                _flush_chunk(pg_vector_add_docs_chunk)
            return

        # Parallel path — outer per-base-doc executor with unordered dispatch.
        # This is where cross-doc parallelism lives: previously the outer for-loop
        # was serial and only fed 1 doc into the inner Phase 4/5 executors, so
        # they only ever saw 1 in-flight task. Now the full pipeline runs per
        # worker (with inner parallelism disabled via the nested-pool guards),
        # so N base docs are truly processed concurrently.
        executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="indexer-base",
        )
        try:
            in_flight: Dict[Any, Any] = {}  # future -> (base_doc, counter)
            counter = 0
            doc_iter = iter(base_documents)

            def _submit_next() -> bool:
                nonlocal counter
                try:
                    base_doc = next(doc_iter)
                except StopIteration:
                    return False
                counter += 1
                _doc_name = self._extract_doc_name(base_doc.metadata)
                self._log_tool_event(f"Processing document #{counter}: '{_doc_name}'.")
                self._log_tool_event(
                    f"Dependent documents for '{_doc_name}' were processed. "
                    f"Applying chunking tool '{chunking_tool if chunking_tool else 'default'}' if specified and preparing documents for indexing..."
                )
                in_flight[executor.submit(_run_pipeline, base_doc)] = (base_doc, counter)
                return True

            for _ in range(workers):
                if not _submit_next():
                    break

            while in_flight:
                done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    base_doc, base_doc_counter = in_flight.pop(future)
                    try:
                        chunks = future.result()
                    except Exception as exc:
                        from traceback import format_exc
                        logger.error(f"Pipeline failed for base doc #{base_doc_counter}: {format_exc()}")
                        result.setdefault("errors", []).append(str(exc))
                        chunks = []
                    _consume_pipeline_output(base_doc, base_doc_counter, chunks)
                    _submit_next()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if pg_vector_add_docs_chunk:
            _flush_chunk(pg_vector_add_docs_chunk)

    def _apply_loaders_chunkers(self, documents: Generator[Document, None, None], chunking_tool: str=None, chunking_config=None) -> Generator[Document, None, None]:
        from ..tools.chunkers import __all__ as chunkers

        if chunking_config is None:
            chunking_config = {}
        chunking_config['embedding'] = self.embeddings
        chunking_config['llm'] = self.llm

        def _filter_parsing_errors(docs_generator, source_name: str):
            """Filter out documents with parsing errors or empty content and track them as skipped."""
            for doc in docs_generator:
                if doc.page_content and doc.page_content.startswith("Unsupported extension for file"):
                    # Track as skipped due to unsupported extension
                    if hasattr(self, '_track_skipped_file_unsupported'):
                        self._track_skipped_file_unsupported(source_name)
                    # Skip this document - don't add to vector store
                    continue
                if doc.page_content and doc.page_content.startswith("Error during content parsing for file"):
                    # Track as skipped due to parsing error
                    if hasattr(self, '_track_runtime_skipped'):
                        self._track_runtime_skipped(source_name, reason="error")
                    elif hasattr(self, '_track_skipped_document'):
                        self._track_skipped_document(source_name, reason="error")
                    # Skip this document - don't add to vector store
                    continue
                # Check for empty content (e.g., OCR returned nothing for image)
                if not doc.page_content or not doc.page_content.strip():
                    # Track as skipped due to empty content
                    if hasattr(self, '_track_skipped_file_empty'):
                        self._track_skipped_file_empty(source_name)
                    elif hasattr(self, '_track_runtime_skipped'):
                        self._track_runtime_skipped(source_name, reason="error")
                    # Skip this document - don't add to vector store with empty content
                    continue
                yield doc

        def _chunk_one(document):
            """Per-doc chunking. Returns a list of chunk Documents (materialized).
            Safe to run on a worker thread — mutates only the doc's own metadata
            and calls _track_* (set.add — GIL-safe). Uses a per-call shallow
            copy of chunking_config to avoid cross-worker mutation."""
            local_config = dict(chunking_config)
            if content_type := document.metadata.get(IndexerKeywords.CONTENT_FILE_NAME.value, None):
                # apply parsing based on content type and chunk if chunker was applied to parent doc
                content = document.metadata.pop(IndexerKeywords.CONTENT_IN_BYTES.value, None)
                return list(_filter_parsing_errors(
                    process_document_by_type(
                        document=document,
                        content=content,
                        extension_source=content_type, llm=self.llm, chunking_config=local_config),
                    source_name=content_type
                ))
            if chunking_tool and (content_in_bytes := document.metadata.pop(IndexerKeywords.CONTENT_IN_BYTES.value, None)) is not None:
                if not content_in_bytes:
                    # Content bytes are empty (e.g. an empty ADO wiki page).
                    # Track so the number shows up in indexing stats instead of
                    # silently disappearing between "fetched" and "indexed",
                    # and drop the document — yielding it downstream would end up
                    # in the vector store with no content.
                    source_name = document.metadata.get('id') or document.metadata.get('name') or document.metadata.get('path') or 'unknown'
                    if hasattr(self, '_track_skipped_file_empty'):
                        self._track_skipped_file_empty(str(source_name))
                    return []
                # apply parsing based on content type resolved from chunking_tool
                content_type = file_extension_by_chunker(chunking_tool)
                source_name = document.metadata.get('id') or document.metadata.get('name') or content_type
                return list(_filter_parsing_errors(
                    process_document_by_type(
                        document=document,
                        content=content_in_bytes,
                        extension_source=content_type, llm=self.llm, chunking_config=local_config),
                    source_name=source_name
                ))
            if chunking_tool:
                # apply default chunker from toolkit config. No parsing.
                chunker = chunkers.get(chunking_tool)
                return list(chunker(file_content_generator=iter([document]), config=local_config))
            # return as is if neither chunker nor content type are specified
            return [document]

        workers = getattr(self, "_index_workers", 1) or 1
        # Nested-pool guard: when called from an outer indexer worker thread
        # (Phase 6 base-doc executor), fall back to serial to avoid explosive
        # thread fan-out. The outer pool already provides cross-doc parallelism.
        if workers <= 1 or threading.current_thread() is not threading.main_thread():
            for document in documents:
                yield from _chunk_one(document)
            return

        # Unordered fan-out mirrors _collect_dependencies_parallel (Phase 4).
        # Workers do the heavy per-doc chunker LLM parse; yielding as-completed
        # keeps the pool saturated when one doc (e.g. large PDF, multi-sheet
        # Excel) is much slower than the others. Doc order is not required
        # downstream — add_documents writes chunks independently.
        executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="indexer-chunk",
        )
        try:
            in_flight: set = set()
            doc_iter = iter(documents)
            for _ in range(workers):
                try:
                    in_flight.add(executor.submit(_chunk_one, next(doc_iter)))
                except StopIteration:
                    break
            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    chunks = future.result()
                    try:
                        in_flight.add(executor.submit(_chunk_one, next(doc_iter)))
                    except StopIteration:
                        pass
                    yield from chunks
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    
    def _extend_data(self, documents: Generator[Document, None, None]):
        yield from documents

    @staticmethod
    def _extract_doc_name(metadata: dict) -> str:
        meta_lower = {k.lower(): v for k, v in metadata.items()}
        return (
            meta_lower.get('name') or
            meta_lower.get('file_path') or
            meta_lower.get('path') or
            'unknown'
        )

    def _is_base_doc_tracked_as_skipped(self, base_doc: Document) -> bool:
        """Return True if any identifier of ``base_doc`` already appears in a skip set.

        Prevents double-counting when ``_filter_parsing_errors`` or the empty-content-bytes
        path already recorded the document, and the outer per-base-doc fallback in
        ``_save_index_generator`` would otherwise add it again to a different category.
        """
        stats = getattr(self, '_indexing_stats', None)
        if stats is None:
            return False
        meta = base_doc.metadata or {}
        candidates = {
            str(meta.get('id') or ''),
            str(meta.get('name') or ''),
            str(meta.get('path') or ''),
            str(meta.get('file_path') or ''),
            self._extract_doc_name(meta),
        }
        candidates.discard('')
        candidates.discard('unknown')
        if not candidates:
            return False
        for skip_set in (
            stats.files_skipped_whitelist,
            stats.files_skipped_blacklist,
            stats.files_skipped_read_error,
            stats.files_skipped_empty,
            stats.files_unsupported_extension,
            stats.documents_skipped_error,
            stats.documents_skipped_filtered,
            stats.runtime_skipped_extension,
            stats.runtime_skipped_error,
        ):
            if candidates & skip_set:
                return True
        return False

    def _collect_dependencies(self, documents: Generator[Document, None, None]):
        # Parallelism opt-in: subclasses (e.g. AzureDevOpsApiWrapper) set
        # self._index_workers from tool params. Absent → 1 → identical to
        # the original serial path. When >1, we fan out _process_document calls
        # (per-page image LLM + attachment downloads) across a bounded worker
        # pool while keeping all yields and metadata mutation on the main
        # thread so downstream _apply_loaders_chunkers / add_documents see the
        # same deterministic order as the serial version.
        workers = getattr(self, "_index_workers", 1) or 1

        # Nested-pool guard: when called from an outer indexer worker thread
        # (Phase 6 base-doc executor), fall back to serial. The outer pool
        # already provides cross-doc parallelism; nesting would multiply threads.
        if workers <= 1 or threading.current_thread() is not threading.main_thread():
            yield from self._collect_dependencies_serial(documents)
            return

        yield from self._collect_dependencies_parallel(documents, workers)

    def _collect_dependencies_serial(self, documents: Generator[Document, None, None]):
        for document in documents:
            yield from self._emit_document_with_deps(
                document,
                list(self._process_document(document)),
            )

    def _collect_dependencies_parallel(
        self,
        documents: Generator[Document, None, None],
        workers: int,
    ):
        def _work(doc):
            # Materialize the per-doc generator on the worker so image LLM /
            # attachment I/O overlap. All state mutation stays on main thread.
            return doc, list(self._process_document(doc))

        executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="indexer-doc",
        )
        try:
            # Unordered dispatch: results flow through as they complete, so a
            # single slow document (large PDF vision parse, big attachment)
            # cannot leave the other workers idle. Doc order is not required —
            # add_documents writes chunks independently and _reduce_duplicates
            # already ran upstream.
            in_flight: set = set()
            doc_iter = iter(documents)
            for _ in range(workers):
                try:
                    in_flight.add(executor.submit(_work, next(doc_iter)))
                except StopIteration:
                    break
            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    document, deps = future.result()
                    try:
                        in_flight.add(executor.submit(_work, next(doc_iter)))
                    except StopIteration:
                        pass
                    yield from self._emit_document_with_deps(document, deps)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _emit_document_with_deps(self, document: Document, deps: List[Document]):
        """Yield dependents (with parent-refs set) then the parent with merged
        dependent_docs metadata. Shared body of serial + parallel paths — keeps
        the yield contract identical."""
        # Build a case-insensitive lookup for important metadata keys
        meta = document.metadata
        meta_lower = {k.lower(): v for k, v in meta.items()}

        doc_id = (
                meta_lower.get('id') or
                meta_lower.get('path') or
                meta_lower.get('name') or
                meta_lower.get('file_path') or
                meta_lower.get('source') or
                'unknown'
        )
        doc_display = {
            k: meta_lower.get(k)
            for k in ('id', 'name', 'path', 'link', 'created', 'updated_on')
            if meta_lower.get(k) is not None
        }
        logger.debug(f"_collect_dependencies: processing document — {doc_display}")

        doc_name = self._extract_doc_name(meta)
        self._log_tool_event(message=f"Collecting the dependencies for document "
                                     f"'{doc_name}' (ID: '{doc_id}') to collect dependencies if any...")

        parent_id = document.metadata.get('id', None)
        parent_updated_on = document.metadata.get('updated_on')
        collected_dep_ids = []

        for dep in deps:
            dep_id = dep.metadata.get('id', '')
            if dep_id:
                collected_dep_ids.append(dep_id)
            dep.metadata[IndexerKeywords.PARENT.value] = parent_id
            if parent_updated_on is not None:
                dep.metadata.setdefault('updated_on', parent_updated_on)
            yield dep

        existing_deps = document.metadata.get(IndexerKeywords.DEPENDENT_DOCS.value, '')
        collected_deps_str = ','.join(collected_dep_ids)

        if existing_deps and collected_deps_str:
            document.metadata[IndexerKeywords.DEPENDENT_DOCS.value] = f"{existing_deps},{collected_deps_str}"
        elif collected_deps_str:
            document.metadata[IndexerKeywords.DEPENDENT_DOCS.value] = collected_deps_str
        # else: keep existing_deps as-is (or empty)

        yield document

    def _clean_metadata(self, documents: Generator[Document, None, None]):
        for document in documents:
            remove_keys = self._remove_metadata_keys()
            for key in remove_keys:
                document.metadata.pop(key, None)
            yield document

    def _reduce_duplicates(
            self,
            documents: Generator[Any, None, None],
            index_name: str,
            log_msg: str = "Verification of documents to index started"
    ) -> Generator[Document, None, None]:
        """Generic duplicate reduction logic for documents."""
        self._ensure_vectorstore_initialized()
        self._log_tool_event(log_msg, tool_name="index_documents")
        indexed_data = self._get_indexed_data(index_name)
        indexed_keys = set(indexed_data.keys())
        if not indexed_keys:
            self._log_tool_event("Vectorstore is empty, indexing all incoming documents", tool_name="index_documents")
            yield from documents
            return

        docs_to_remove = set()

        for document in documents:
            key = self.key_fn(document)
            key = key if isinstance(key, str) else str(key)
            if key in indexed_keys and index_name == indexed_data[key]['metadata'].get('collection'):
                if self.compare_fn(document, indexed_data[key]):
                    if hasattr(self, '_track_document_unchanged'):
                        self._track_document_unchanged(
                            document.metadata.get('path')
                            or document.metadata.get('name')
                            or key
                        )
                    continue
                yield document
                docs_to_remove.update(self.remove_ids_fn(indexed_data, key))
            else:
                yield document

        if docs_to_remove:
            self._log_tool_event(
                f"Removing {len(docs_to_remove)} documents from vectorstore that are already indexed with different updated_on.",
                tool_name="index_documents"
            )
            self.vectorstore.delete(ids=list(docs_to_remove))
    
    def _get_indexed_data(self, index_name: str):
        raise NotImplementedError("Subclasses must implement this method")

    def key_fn(self, document: Document):
        raise NotImplementedError("Subclasses must implement this method")

    def compare_fn(self, document: Document, idx):
        raise NotImplementedError("Subclasses must implement this method")
    
    def remove_ids_fn(self, idx_data, key: str):
        raise NotImplementedError("Subclasses must implement this method")

    def remove_index(self, index_name: str = ""):
        """Cleans the indexed data in the collection."""
        deleted_count = super()._clean_collection(index_name=index_name, including_index_meta=True)

        if index_name and deleted_count == 0:
            raise ToolException(f"Index '{index_name}' not found. Available collections: {self.list_collections()}")

        self._emit_index_data_removed_event(index_name)
        return (f"Collection '{index_name}' has been removed from the vector store.\n"
                f"Available collections: {self.list_collections()}") if index_name \
            else "All collections have been removed from the vector store." 

    def _build_collection_filter(self, filter: dict | str, index_name: str = "") -> dict:
        """Builds a filter for the collection based on the provided suffix."""

        filter = filter if isinstance(filter, dict) else json.loads(filter)
        if index_name:
            filter.update({"collection": {
                "$eq": index_name.strip()
            }})

        if filter:
            # Exclude index meta documents from search results
            filter = {
                "$and": [
                    filter,
                    {"$or": [
                        {"type": {"$exists": False}},
                        {"type": {"$ne": IndexerKeywords.INDEX_META_TYPE.value}}
                    ]},
                ]
            }
        else:
            filter = {"$or": [
                {"type": {"$exists": False}},
                {"type": {"$ne": IndexerKeywords.INDEX_META_TYPE.value}}
            ]}
        return filter

    def search_index(self,
                     query: str,
                     index_name: str = "",
                     filter: dict | str = {}, cut_off: float = DEFAULT_CUT_OFF,
                     search_top: int = 10, reranker: dict = {},
                     full_text_search: Optional[Dict[str, Any]] = None,
                     reranking_config: Optional[Dict[str, Dict[str, Any]]] = None,
                     extended_search: Optional[List[str]] = None,
                     output_fields: Optional[List[str]] = None,
                     **kwargs):
        """Searches indexed documents in the vector store.

        Args:
            query: Search query string.
            index_name: Collection/index name to search in.
            filter: Filter criteria for search.
            cut_off: Minimum similarity score threshold.
            search_top: Maximum number of results to return.
            reranker: Legacy reranking configuration.
            full_text_search: Full-text search configuration.
            reranking_config: Advanced reranking configuration.
            extended_search: Extended search chunk types.
            output_fields: Fields to include in output. Supports:
                - "page_content": document content
                - "score": similarity score
                - "metadata": all metadata fields
                - "metadata.<field>": specific metadata field (e.g., "metadata.source")
                If None or empty list, returns all fields (backward compatible).
                If all specified fields are invalid, returns all fields.

        Returns:
            List of documents with requested fields, or error message string.
        """
        available_collections = super().list_collections()
        if index_name and index_name not in available_collections:
            return f"Collection '{index_name}' not found. Available collections: {available_collections}"

        filter = self._build_collection_filter(filter, index_name)
        found_docs = super().search_documents(
            query,
            doctype=self.doctype,
            filter=filter,
            cut_off=cut_off,
            search_top=search_top,
            reranker=reranker,
            full_text_search=full_text_search,
            reranking_config=reranking_config,
            extended_search=extended_search
        )

        # Apply field filtering if specified (non-empty list)
        if output_fields and isinstance(found_docs, list):
            found_docs = self._filter_result_fields(found_docs, output_fields)

        return found_docs if found_docs else f"No documents found by query '{query}' and filter '{filter}'"

    def _filter_result_fields(
        self,
        docs: List[Dict[str, Any]],
        include_fields: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter document results to include only specified fields.

        Args:
            docs: List of document dicts with page_content, metadata, score.
            include_fields: Fields to include. Supports:
                - "page_content", "score", "metadata" (top-level)
                - "metadata.<field>" (specific metadata field, split by first dot only)

        Returns:
            Filtered list of documents. If no valid fields found, returns original docs.
        """
        if not include_fields or not docs:
            return docs

        # Parse field specifications
        top_level_fields = set()  # page_content, score, metadata
        metadata_fields = set()   # specific metadata fields (without "metadata." prefix)
        include_full_metadata = False

        for field in include_fields:
            if field == "metadata":
                include_full_metadata = True
            elif field.startswith("metadata."):
                # Split by first dot only: "metadata.config.timeout" -> "config.timeout"
                metadata_field = field.split(".", 1)[1] if len(field) > 9 else None
                if metadata_field:
                    metadata_fields.add(metadata_field)
            elif field in ("page_content", "score"):
                top_level_fields.add(field)

        # If no valid fields specified, return original docs (backward compatible)
        if not top_level_fields and not include_full_metadata and not metadata_fields:
            return docs

        filtered_docs = []
        any_field_found = False

        for doc in docs:
            filtered = {}

            # Include top-level fields
            if "page_content" in top_level_fields and "page_content" in doc:
                filtered["page_content"] = doc["page_content"]
                any_field_found = True

            if "score" in top_level_fields and "score" in doc:
                filtered["score"] = doc["score"]
                any_field_found = True

            # Handle metadata
            if include_full_metadata:
                if "metadata" in doc:
                    filtered["metadata"] = doc["metadata"]
                    any_field_found = True
            elif metadata_fields:
                if "metadata" in doc and doc["metadata"]:
                    filtered_metadata = {}
                    for mfield in metadata_fields:
                        if mfield in doc["metadata"]:
                            filtered_metadata[mfield] = doc["metadata"][mfield]
                            any_field_found = True

                    if filtered_metadata:
                        filtered["metadata"] = filtered_metadata

            filtered_docs.append(filtered)

        # If no valid fields were actually found in any doc, return original docs
        if not any_field_found:
            return docs

        return filtered_docs

    def stepback_search_index(self,
                     query: str,
                     messages: List[Dict[str, Any]] = [],
                     index_name: str = "",
                     filter: dict | str = {}, cut_off: float = DEFAULT_CUT_OFF,
                     search_top: int = 10, reranker: dict = {},
                     full_text_search: Optional[Dict[str, Any]] = None,
                     reranking_config: Optional[Dict[str, Dict[str, Any]]] = None,
                     extended_search: Optional[List[str]] = None,
                     **kwargs):
        """ Searches indexed documents in the vector store."""
        filter = self._build_collection_filter(filter, index_name)
        found_docs = super().stepback_search(
            query,
            messages,
            self.doctype,
            filter=filter,
            cut_off=cut_off,
            search_top=search_top,
            full_text_search=full_text_search,
            reranking_config=reranking_config,
            extended_search=extended_search
        )
        return f"Found {len(found_docs)} documents matching the query\n{json.dumps(found_docs, indent=4)}" if found_docs else "No documents found matching the query."

    def stepback_summary_index(self,
                     query: str,
                     messages: List[Dict[str, Any]] = [],
                     index_name: str = "",
                     filter: dict | str = {}, cut_off: float = DEFAULT_CUT_OFF,
                     search_top: int = 10, reranker: dict = {},
                     full_text_search: Optional[Dict[str, Any]] = None,
                     reranking_config: Optional[Dict[str, Dict[str, Any]]] = None,
                     extended_search: Optional[List[str]] = None,
                     **kwargs):
        """ Generates a summary of indexed documents using stepback technique."""

        filter = self._build_collection_filter(filter, index_name)
        return super().stepback_summary(
            query, 
            messages, 
            self.doctype, 
            filter=filter, 
            cut_off=cut_off, 
            search_top=search_top,
            full_text_search=full_text_search, 
            reranking_config=reranking_config, 
            extended_search=extended_search
        )
    
    def index_meta_init(self, index_name: str, index_configuration: dict[str, Any]):
        from ..runtime.langchain.interfaces.llm_processor import add_documents
        self._ensure_vectorstore_initialized()
        index_meta = super().get_index_meta(index_name)
        if not index_meta:
            self._log_tool_event(
                f"There is no existing index_meta for collection '{index_name}'. Initializing it.",
                tool_name="index_data"
            )
            created_on = time.time()
            metadata = {
                "collection": index_name,
                "type": IndexerKeywords.INDEX_META_TYPE.value,
                "indexed": 0,
                "updated": 0,
                "state": IndexerKeywords.INDEX_META_IN_PROGRESS.value,
                "index_configuration": index_configuration,
                "created_on": created_on,
                "updated_on": created_on,
                "task_id": None,
                "conversation_id": None,
                "toolkit_id": self.toolkit_id,
                # Initialize error field to keep track of the latest failure reason if any
                "error": None,
            }
            metadata["history"] = json.dumps([metadata])
            index_meta_doc = Document(page_content=f"{IndexerKeywords.INDEX_META_TYPE.value}_{index_name}", metadata=metadata)
            add_documents(vectorstore=self.vectorstore, documents=[index_meta_doc])
        else:
            # Reindex: the collection already has an index_meta row. Reset it to
            # in_progress with a fresh created_on so the start event (emitted right after)
            # carries state=in_progress instead of the previous run's terminal state.
            # Without this the platform's reconcile-on-stop registry never populates for a
            # reindex (it only registers on in_progress), so a stopped reindex stays stuck.
            now = time.time()
            metadata = copy.deepcopy(index_meta.get("metadata", {}))
            metadata["state"] = IndexerKeywords.INDEX_META_IN_PROGRESS.value
            metadata["created_on"] = now
            metadata["updated_on"] = now
            metadata["error"] = None
            # Reset run linkage like the fresh-init branch: a previous (e.g. completed)
            # run may have left task_id stamped, and the platform's reconcile guard skips
            # rows whose task_id doesn't match the stopping task — leaving a stopped
            # reindex stuck. Clearing it lets this run's task_id be (re)stamped/matched.
            metadata["task_id"] = None
            metadata["conversation_id"] = None
            # Append a new run entry to history (like fresh-init records the run) instead of
            # leaving history[-1] pointing at the previous run's terminal state; also makes
            # is_reindex (len(history) > 1) correct for this run.
            history_raw = metadata.pop("history", "[]")
            try:
                history = json.loads(history_raw) if history_raw and history_raw.strip() else []
                if not isinstance(history, list):
                    history = []
            except (json.JSONDecodeError, TypeError):
                history = []
            history.append(dict(metadata))  # history key already popped -> no nesting
            metadata["history"] = json.dumps(history)
            index_meta_doc = Document(
                page_content=index_meta.get("content", f"{IndexerKeywords.INDEX_META_TYPE.value}_{index_name}"),
                metadata=metadata,
            )
            add_documents(vectorstore=self.vectorstore, documents=[index_meta_doc], ids=[index_meta.get("id")])

    def index_meta_update(self, index_name: str, state: str, result: int, update_force: bool = True, interval: Optional[float] = None, error: Optional[str] = None, skipped: Optional[Dict] = None, docs_count: Optional[int] = None):
        """Update `index_meta` document with optional time-based throttling.

        Args:
            index_name: Index name to update meta for.
            state: New state value for the `index_meta` record.
            result: Number of processed documents to store in the `updated` field.
            update_force: If `True`, perform the update unconditionally, ignoring throttling.
                          If `False`, perform the update only when the effective time interval has passed.
            interval: Optional custom interval (in seconds) for this call when `update_force` is `False`.
                      If `None`, falls back to the value stored in `self._index_meta_config["update_interval"]`
                      if present, otherwise uses `INDEX_META_UPDATE_INTERVAL`.
            error: Optional error message to record when the state represents a failed index.
            skipped: Optional dictionary containing skipped items data from indexing stats.
        """
        self._ensure_vectorstore_initialized()
        if not hasattr(self, "_index_meta_last_update_time"):
            self._index_meta_last_update_time: Dict[str, float] = {}

        if not update_force:
            # Resolve effective interval:
            # 1\) explicit arg
            # 2\) value from `_index_meta_config`
            # 3\) default constant
            cfg_interval = None
            if hasattr(self, "_index_meta_config"):
                cfg_interval = self._index_meta_config.get("update_interval")

            eff_interval = (
                interval
                if interval is not None
                else (cfg_interval if cfg_interval is not None else INDEX_META_UPDATE_INTERVAL)
            )

            last_time = self._index_meta_last_update_time.get(index_name)
            now = time.time()
            if last_time is not None and (now - last_time) < eff_interval:
                return
            self._index_meta_last_update_time[index_name] = now
        else:
            # For forced updates, always refresh last update time
            self._index_meta_last_update_time[index_name] = time.time()

        index_meta_raw = super().get_index_meta(index_name)
        from ..runtime.langchain.interfaces.llm_processor import add_documents
        #
        if index_meta_raw:
            metadata = copy.deepcopy(index_meta_raw.get("metadata", {}))
            # indexed_chunks = number of chunks stored in vector store
            metadata["indexed_chunks"] = self.get_indexed_count(index_name)
            metadata["updated"] = result
            metadata["state"] = state
            metadata["updated_on"] = time.time()
            # Attach error if provided, else clear on success
            if error is not None:
                metadata["error"] = error
            elif state == IndexerKeywords.INDEX_META_COMPLETED.value:
                # Clear previous error on successful completion
                metadata["error"] = None
            # Attach skipped items data if provided
            if skipped is not None:
                metadata["skipped"] = json.dumps(skipped)
                items_processed = skipped.get("items_processed", 0)
                total_skipped = skipped.get("total_skipped", 0)
                total_fetched = skipped.get("total_fetched", 0)

                # Consistent formula for both code and non-code indexers:
                # - total = all items initially fetched/considered from source
                # - indexed = items successfully processed (total - skipped)
                #
                # If total_fetched is set, use it. Otherwise fall back to heuristic:
                # - If total_skipped > items_processed, they're disjoint (code indexer)
                # - Otherwise items_processed includes all fetched items (non-code indexer)
                if total_fetched > 0:
                    # Explicit total_fetched provided
                    metadata["total"] = total_fetched
                    metadata["indexed"] = docs_count if docs_count is not None else total_fetched - total_skipped
                elif total_skipped > items_processed:
                    # Code indexer: items_processed and total_skipped are disjoint
                    metadata["total"] = items_processed + total_skipped
                    metadata["indexed"] = docs_count if docs_count is not None else items_processed
                else:
                    # Non-code indexer: items_processed is all fetched, skipped is subset
                    metadata["total"] = items_processed
                    metadata["indexed"] = docs_count if docs_count is not None else items_processed - total_skipped
            else:
                # Fallback: if no skipped data, use chunks count for backward compatibility
                metadata["indexed"] = metadata["indexed_chunks"]
            #
            history_raw = metadata.pop("history", "[]")
            try:
                history = json.loads(history_raw) if history_raw.strip() else []
                # replace the last history item with updated metadata
                if history and isinstance(history, list):
                    history[-1] = metadata
                else:
                    history = [metadata]
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to load index history: {history_raw}. Create new with only current item.")
                history = [metadata]
            #
            metadata["history"] = json.dumps(history)
            index_meta_doc = Document(page_content=index_meta_raw.get("content", ""), metadata=metadata)
            add_documents(vectorstore=self.vectorstore, documents=[index_meta_doc], ids=[index_meta_raw.get("id")])

    def _emit_index_event(self, index_name: str, error: Optional[str] = None):
        """
        Emit custom event for index data operation.
        
        Args:
            index_name: The name of the index
            error: Error message if the operation failed, None otherwise
        """
        index_meta = super().get_index_meta(index_name)
        
        if not index_meta:
            logger.warning(
                f"No index_meta found for index '{index_name}'. "
                "Cannot emit index event."
            )
            return
        
        metadata = index_meta.get("metadata", {})
        
        # Determine if this is a reindex operation
        history_raw = metadata.get("history", "[]")
        try:
            history = json.loads(history_raw) if history_raw.strip() else []
            is_reindex = len(history) > 1
        except (json.JSONDecodeError, TypeError):
            is_reindex = False
        
        # Build event message
        event_data = {
            "id": index_meta.get("id"),
            "index_name": index_name,
            "state": "failed" if error is not None else metadata.get("state"),
            "error": error,
            "reindex": is_reindex,
            "indexed": metadata.get("indexed", 0),
            "updated": metadata.get("updated", 0),
            "toolkit_id": metadata.get("toolkit_id"),
            "created_at": metadata.get("created_on"),
            "updated_on": metadata.get("updated_on"),
        }
        
        # Emit the event — skip silently when no ambient LangChain run context
        # exists (e.g. standalone script mode), matching the guard in
        # BaseToolApiWrapper._log_tool_event.
        if self._runnable_config is None and not self._has_ambient_runnable_context():
            return
        try:
            dispatch_custom_event("index_data_status", event_data)
            logger.debug(
                f"Emitted index_data_status event for index "
                f"'{index_name}': {event_data}"
            )
        except Exception as e:
            logger.warning(f"Failed to emit index_data_status event: {e}")

    def _emit_index_data_removed_event(self, index_name: str):
        """
        Emit custom event for index data removing.

        Args:
            index_name: The name of the index
            toolkit_id: The toolkit identifier
        """
        # Build event message
        event_data = {
            "index_name": index_name,
            "toolkit_id": self.toolkit_id,
            "project_id": self.elitea.project_id,
        }
        # Emit the event — skip silently when no ambient LangChain run context
        # exists (e.g. standalone script mode).
        if self._runnable_config is None and not self._has_ambient_runnable_context():
            return
        try:
            dispatch_custom_event("index_data_removed", event_data)
            logger.debug(
                f"Emitted index_data_removed event for index "
                f"'{index_name}': {event_data}"
            )
        except Exception as e:
            logger.warning(f"Failed to emit index_data_removed event: {e}")

    def get_available_tools(self, filter_by_collections: bool = False):
        """
        Returns the standardized vector search tools.

        Args:
            filter_by_collections: When True, only return indexer tools if the toolkit has indexed collections.
                                   This reduces token usage for agents with lazy_tools_mode enabled.
                                   Default is False (always return all indexer tools for UI display).

        When collections exist (or filter_by_collections=False), the following tools are available:
        - index_data: Load data to index
        - list_collections: List available collections
        - search_index: Search indexed documents
        - stepback_search_index: Search with stepback technique
        - stepback_summary_index: Generate summary using stepback
        - remove_index: Remove indexed data

        This method constructs the argument schemas for each tool, merging base parameters with any extra parameters
        defined in the subclass. It also handles the special case for chunking tools and their configuration.

        Returns:
            list: List of tool dictionaries with name, ref, description, and args_schema.
        """
        # Only filter by collections when explicitly requested (lazy_tools_mode optimization)
        # By default, always show all tools for UI display
        if filter_by_collections:
            has_collections = self._has_collections()
            if not has_collections:
                logger.debug(f"Toolkit has no collections and filter_by_collections=True, skipping all indexer tools")
                return []
            logger.debug(f"Toolkit has collections, adding all indexer tools")
        else:
            logger.debug(f"filter_by_collections=False, returning all indexer tools")

        index_params = {
            "index_name": (
                str,
                Field(description="Index name (max 7 characters)", min_length=1, max_length=7)
            ),
            "clean_index": (
                Optional[bool],
                Field(default=False, description="Optional flag to enforce clean existing index before indexing new data")
            ),
            "progress_step": (
                Optional[int],
                Field(default=10, ge=0, le=100, description="Optional step size for progress reporting during indexing")
            ),
        }
        chunking_config = (
            Optional[dict],
            Field(description="Chunking tool configuration", default=loaders_allowed_to_override)
        )

        index_extra_params = self._index_tool_params() or {}
        chunking_tool = index_extra_params.pop("chunking_tool", None)
        if chunking_tool:
            index_params = {
                **index_params,
                "chunking_tool": chunking_tool,
            }
        index_params["chunking_config"] = chunking_config
        index_args_schema = create_model("IndexData", **index_params, **index_extra_params)

        tools = [
            {
                "name": IndexTools.INDEX_DATA.value,
                "mode": IndexTools.INDEX_DATA.value,
                "ref": self.index_data,
                "description": "Loads data to index.",
                "args_schema": index_args_schema,
            },
            {
                "name": IndexTools.LIST_COLLECTIONS.value,
                "mode": IndexTools.LIST_COLLECTIONS.value,
                "ref": self.list_collections,
                "description": self.list_collections.__doc__,
                "args_schema": create_model("ListCollectionsParams")
            },
            {
                "name": IndexTools.SEARCH_INDEX.value,
                "mode": IndexTools.SEARCH_INDEX.value,
                "ref": self.search_index,
                "description": self.search_index.__doc__,
                "args_schema": BaseSearchParams
            },
            {
                "name": IndexTools.STEPBACK_SEARCH_INDEX.value,
                "mode": IndexTools.STEPBACK_SEARCH_INDEX.value,
                "ref": self.stepback_search_index,
                "description": self.stepback_search_index.__doc__,
                "args_schema": BaseStepbackSearchParams
            },
            {
                "name": IndexTools.STEPBACK_SUMMARY_INDEX.value,
                "mode": IndexTools.STEPBACK_SUMMARY_INDEX.value,
                "ref": self.stepback_summary_index,
                "description": self.stepback_summary_index.__doc__,
                "args_schema": BaseStepbackSearchParams
            },
            {
                "name": IndexTools.REMOVE_INDEX.value,
                "mode": IndexTools.REMOVE_INDEX.value,
                "ref": self.remove_index,
                "description": self.remove_index.__doc__,
                "args_schema": RemoveIndexParams
            },
        ]

        return tools
