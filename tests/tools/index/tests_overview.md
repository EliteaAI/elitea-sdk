# BaseIndexerToolkit Integration Tests Overview

## Package Structure

```
tests/tools/index/
├── test_indexer_db_integration.py   # Integration tests for pgvector persistence
├── tests_overview.md                # This file
└── indexer_db_integration_requirements.txt  # Test requirements spec
```

## Test Coverage Matrix

| Test Class | Test Method | Validates |
|------------|-------------|-----------|
| **TestCodeIndexerDBPersistence** | | Code content indexing (GitHub-like) |
| | `test_index_code_creates_db_records` | Records created, count matches, collection name correct |
| | `test_code_embeddings_are_valid` | Non-null embeddings, correct dimension (1536), valid floats |
| | `test_code_metadata_populated` | source, language, repository, commit_hash preserved |
| | `test_remove_code_index_cleanup` | remove_index deletes all records |
| **TestNonCodeIndexerDBPersistence** | | Document indexing (SharePoint-like) |
| | `test_index_documents_creates_db_records` | Records created, count matches, collection name correct |
| | `test_document_embeddings_are_valid` | Non-null embeddings, correct dimension, valid floats |
| | `test_document_metadata_populated` | source, title, file_type, author, created_date preserved |
| | `test_remove_document_index_cleanup` | remove_index deletes all records |
| **TestIndexerDBEdgeCases** | | Edge cases and boundary conditions |
| | `test_empty_index_name` | Indexing with empty index_name works |
| | `test_clean_index_flag` | clean_index=True removes old records before new index |
| **TestIndexerMetadataTracking** | | index_meta lifecycle management |
| | `test_index_meta_state_transitions` | state: in_progress → completed |
| | `test_index_meta_error_handling` | error null on success, populated on failure |
| | `test_index_meta_count_accuracy` | indexed count matches actual DB row count |
| **TestCollectionManagement** | | Collection operations and isolation |
| | `test_list_collections_accuracy` | list_collections returns only indexed collections |
| | `test_multi_collection_isolation` | No cross-contamination between collections |
| | `test_remove_index_zero_results` | remove_index + search returns 0 results |
| **TestDeduplicationAndUpdates** | | Deduplication logic |
| | `test_reindex_unchanged_no_duplicates` | Re-indexing unchanged file: no duplicate rows |
| | `test_reindex_modified_replaces_old` | Re-indexing modified file: replaces old, count stable |
| **TestPartialBatchFailure** | | Error resilience |
| | `test_partial_embedder_failure` | Partial failure: some docs stored, no crash |
| | `test_failure_does_not_leave_inconsistent_state` | Complete failure: no content rows, error tracked |
| **TestWritePathChunkCount** | | Chunk count validation |
| | `test_chunk_count_matches_indexed` | DB rows == reported count, all have content + embedding |
| **TestRetrievalCorrectness** | | search_index behavior |
| | `test_search_returns_relevant_chunks` | Returns semantically relevant results |
| | `test_search_top_parameter` | search_top limits result count |
| | `test_cut_off_parameter` | cut_off filters low-similarity results |
| | `test_filter_by_metadata_key` | Metadata filter limits to matching docs |
| | `test_empty_index_returns_no_documents` | Empty index returns graceful message |

## Key Behaviours Under Test

### Write Path Correctness
- Chunks land in DB with non-null `page_content`, `embedding`, and `metadata`
- Chunk count matches expected output from indexer
- Document source metadata preserved end-to-end

### DB State and Lifecycle
- `index_meta` tracks correct state transitions: `in_progress` → `completed`
- `indexed` count in `index_meta` equals actual DB row count
- `error` field null on success; populated correctly on failure

### Deduplication and Updates
- Re-indexing unchanged file does not create duplicate DB rows
- Re-indexing modified file replaces old version; total row count stable
- `clean_index=True` removes all prior chunks before writing new batch

### Collection Management
- `list_collections()` returns only indexed collections
- `remove_index()` deletes all chunks; subsequent search returns 0 results
- Multi-collection queries are fully isolated (no cross-contamination)

### Retrieval Correctness
- `search_index` returns semantically relevant chunks
- `cut_off` and `search_top` parameters respected
- Metadata filter limits results to matching documents only
- Empty index returns "No documents found" gracefully

### Error Resilience
- Partial batch failure: `failed_count > 0`, remaining docs stored, no uncaught exception
- `index_meta.error` populated on failure; pipeline does not leave DB inconsistent

## Functions Under Test

| Function | Location | Description |
|----------|----------|-------------|
| `index_data()` | `BaseIndexerToolkit` | Main indexing pipeline |
| `remove_index()` | `BaseIndexerToolkit` | Delete chunks from collection |
| `search_index()` | `BaseIndexerToolkit` | Vector similarity search |
| `list_collections()` | `VectorAdapter` | List indexed collections |
| `_ensure_vectorstore_initialized()` | `BaseIndexerToolkit` | Initialize pgvector connection |

## What is NOT Yet Tested

- `stepback_search_index()` — LLM query rewrite + retrieval
- `stepback_summary_index()` — LLM summarization of retrieved context
- RAG quality metrics (ContextualPrecision, ContextualRecall, Faithfulness, AnswerRelevancy)
- Concurrent indexing operations
- Large document batches (>1000 docs)
- Custom chunker configurations
- Connection pool exhaustion scenarios
- Network partition / timeout handling

## Running Tests

```bash
# All integration tests
pytest tests/tools/index/test_indexer_db_integration.py -v -m integration

# Specific test class
pytest tests/tools/index/test_indexer_db_integration.py::TestCodeIndexerDBPersistence -v

# Quick smoke tests (basic persistence only)
pytest tests/tools/index/test_indexer_db_integration.py::TestCodeIndexerDBPersistence \
       tests/tools/index/test_indexer_db_integration.py::TestNonCodeIndexerDBPersistence -v
```

## Prerequisites

- Docker or compatible container runtime (Docker Desktop, Colima, Podman)
- Environment variables: `ELITEA_DEPLOYMENT_URL`, `ELITEA_PROJECT_ID`, `ELITEA_TOKEN`
- Tests use testcontainers to auto-spin pgvector instances
