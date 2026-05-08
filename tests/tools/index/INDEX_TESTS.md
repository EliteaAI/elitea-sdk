# Index Tests

Two test suites for indexer functionality:

| Suite | File | Purpose |
|-------|------|---------|
| **DB Integration** | `test_indexer_db_integration.py` | DB persistence, deduplication, lifecycle |
| **RAG Quality** | `test_indexer_rag_quality.py` | Retrieval quality with DeepEval metrics |

---

## Local Execution

### Prerequisites

- Docker running (testcontainers auto-starts pgvector)
- EliteA credentials in environment or `.env` file
- Python venv with SDK installed

### Environment Variables

```bash
ELITEA_DEPLOYMENT_URL=https://...
ELITEA_PROJECT_ID=...
ELITEA_TOKEN=...
DEFAULT_LLM_MODEL=gpt-5.5-mini  # For DeepEval metrics
```

### Run DB Integration Tests

```bash
# From elitea-sdk directory
source .venv/bin/activate
set -a && source .env && set +a

# Run all DB integration tests
pytest tests/tools/index/test_indexer_db_integration.py -v

# Run specific test class
pytest tests/tools/index/test_indexer_db_integration.py::TestDeduplicationAndUpdates -v
```

### Run RAG Quality Tests

```bash
# FIRST: Generate the SQL dump (one-time, or when corpus changes)
python tests/tools/index/generate_rag_embeddings.py

# Verify dump exists
python tests/tools/index/generate_rag_embeddings.py --verify

# Run all RAG quality tests
pytest tests/tools/index/test_indexer_rag_quality.py -v
```

### Run Both Suites

```bash
pytest tests/tools/index/test_indexer_db_integration.py \
       tests/tools/index/test_indexer_rag_quality.py -v
```

---

## Docker-Compose (Skip Testcontainers)

```bash
# Start pgvector service
docker-compose up -d pgvector-db

# Set connection string (skips testcontainers)
export PGVECTOR_CONNECTION_STRING="postgresql://postgres:yourpassword@localhost:5435/postgres"

# Run tests
pytest tests/tools/index/ -v
```

---

## CI/CD

**Workflow:** `.github/workflows/execute-indexing-tests-on-demand.yml`

| Option | Description |
|--------|-------------|
| `test_suites=indexer_db` | DB integration tests only |
| `test_suites=indexer_rag` | RAG quality tests only |

Uses pre-built container `ghcr.io/eliteaai/elitea-sdk:pyodide` with pgvector service.

---

## Test Coverage

### Suite A: DB Integration

#### Write Path Correctness
- Chunks land in DB with non-null `page_content`, `embedding`, and `metadata`
- Chunk count matches expected output from the paired chunker
- Document source metadata preserved end-to-end

#### DB State and Lifecycle
- `index_meta` tracks state transitions: `in_progress` → `completed`
- `indexed` count equals actual DB row count
- `error` field null on success; populated on failure

#### Deduplication and Updates
- Re-indexing unchanged file does not create duplicates
- Re-indexing modified file replaces old version
- `clean_index=True` removes all prior chunks before writing

#### Collection Management
- `list_collections()` returns only indexed collections
- `remove_index()` deletes all chunks; search returns 0 results
- Multi-collection queries are fully isolated

#### Error Resilience
- Partial batch failure: `failed_count > 0`, remaining docs stored
- `index_meta.error` populated on failure; no inconsistent DB state

---

### Suite B: RAG Quality

#### Pre-computed Embeddings

| Item | Path |
|------|------|
| SQL dump | `fixtures/rag_corpus_embeddings.sql` |
| Generator | `generate_rag_embeddings.py` |
| Corpus | 10 documents, 1536-dim (`text-embedding-ada-002`) |

> Tests **FAIL** if dump is missing (no silent skip, no API fallback)

#### TestRetrievalCorrectness (4 tests)

| Test | Validates |
|------|-----------|
| `test_search_top_parameter` | `search_top` limits result count |
| `test_cut_off_filters_by_score` | All returned scores ≥ `cut_off` |
| `test_filter_by_metadata_key` | Metadata filter works correctly |
| `test_nonexistent_index_returns_error` | Graceful error for missing index |

#### TestRAGQualitySearchIndex (5 tests)

| Test | Validates |
|------|-----------|
| `test_search_ranking_and_cutoff_easy` | Count ≤ `search_top`, scores ≥ `cut_off`, ranking order, source match |
| `test_search_ranking_and_cutoff_medium` | Same for semantic query |
| `test_contextual_precision` | DeepEval ContextualPrecisionMetric ≥ 0.7 |
| `test_contextual_recall` | DeepEval ContextualRecallMetric ≥ 0.7 |
| `test_adversarial_query_low_relevancy` | Unrelated queries return few/low-score results |

#### TestRAGQualityStepback (4 tests)

| Test | Validates |
|------|-----------|
| `test_stepback_search_ranking_and_cutoff` | Stepback search: count, scores, ranking, source |
| `test_stepback_search_ranking_and_cutoff_hard` | Multi-doc query returns multiple relevant docs |
| `test_stepback_summary_faithfulness` | DeepEval FaithfulnessMetric ≥ 0.7 |
| `test_stepback_summary_answer_relevancy` | DeepEval AnswerRelevancyMetric ≥ 0.7 |

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cut_off` | 0.75 | Minimum similarity score for results |
| `search_top` | 5 | Maximum results to return |

All tests use realistic parameters to validate actual search behavior.
