"""
Unit tests for search_index output_fields parameter.

Tests the _filter_result_fields helper method that filters search results
to include only specified fields. These are pure unit tests - no DB or
external services required.

Run:
    pytest tests/tools/index/test_search_output_fields.py -v
"""

import pytest
from elitea_sdk.tools.base_indexer_toolkit import BaseIndexerToolkit


# Sample documents mimicking search_index output structure
SAMPLE_DOCS = [
    {
        "page_content": "def authenticate(user, password): return validate(user)",
        "metadata": {
            "source": "auth.py",
            "category": "code",
            "language": "python",
            "config.timeout": 30,  # Dotted key (not nested)
        },
        "score": 0.89,
    },
    {
        "page_content": "class User:\n    def __init__(self, name): self.name = name",
        "metadata": {
            "source": "user.py",
            "category": "code",
            "language": "python",
        },
        "score": 0.75,
    },
    {
        "page_content": "SELECT * FROM users WHERE active = true",
        "metadata": {
            "source": "queries.sql",
            "category": "database",
            "language": "sql",
        },
        "score": 0.62,
    },
]


class MockIndexerToolkit(BaseIndexerToolkit):
    """Minimal mock to access _filter_result_fields without full initialization."""

    def __init__(self):
        # Skip parent __init__ - we only need the helper method
        pass

    def _get_tools_info(self):
        return []


@pytest.fixture
def toolkit():
    """Provide a mock toolkit instance for testing _filter_result_fields."""
    return MockIndexerToolkit()


@pytest.fixture
def sample_docs():
    """Provide a fresh copy of sample docs for each test."""
    import copy
    return copy.deepcopy(SAMPLE_DOCS)


class TestFilterResultFieldsUnit:
    """Unit tests for _filter_result_fields helper method."""

    # -------------------------------------------------------------------------
    # Backward compatibility: None / empty / invalid → return original
    # -------------------------------------------------------------------------

    def test_none_returns_original(self, toolkit, sample_docs):
        """output_fields=None returns unchanged docs."""
        result = toolkit._filter_result_fields(sample_docs, None)
        assert result == sample_docs

    def test_empty_list_returns_original(self, toolkit, sample_docs):
        """output_fields=[] returns unchanged docs (backward compatible)."""
        result = toolkit._filter_result_fields(sample_docs, [])
        assert result == sample_docs

    def test_all_invalid_fields_returns_original(self, toolkit, sample_docs):
        """All invalid field names returns unchanged docs (graceful fallback)."""
        result = toolkit._filter_result_fields(sample_docs, ["invalid", "fake", "nonexistent"])
        assert result == sample_docs

    def test_empty_docs_returns_empty(self, toolkit):
        """Empty docs list returns empty list."""
        result = toolkit._filter_result_fields([], ["page_content"])
        assert result == []

    # -------------------------------------------------------------------------
    # Top-level fields: page_content, score
    # -------------------------------------------------------------------------

    def test_page_content_only(self, toolkit, sample_docs):
        """output_fields=["page_content"] returns only page_content."""
        result = toolkit._filter_result_fields(sample_docs, ["page_content"])

        assert len(result) == 3
        for doc in result:
            assert "page_content" in doc
            assert "score" not in doc
            assert "metadata" not in doc

        assert result[0]["page_content"] == sample_docs[0]["page_content"]

    def test_score_only(self, toolkit, sample_docs):
        """output_fields=["score"] returns only score."""
        result = toolkit._filter_result_fields(sample_docs, ["score"])

        assert len(result) == 3
        for doc in result:
            assert "score" in doc
            assert "page_content" not in doc
            assert "metadata" not in doc

        assert result[0]["score"] == 0.89
        assert result[1]["score"] == 0.75

    def test_page_content_and_score(self, toolkit, sample_docs):
        """output_fields=["page_content", "score"] returns both."""
        result = toolkit._filter_result_fields(sample_docs, ["page_content", "score"])

        assert len(result) == 3
        for doc in result:
            assert "page_content" in doc
            assert "score" in doc
            assert "metadata" not in doc

    # -------------------------------------------------------------------------
    # Full metadata
    # -------------------------------------------------------------------------

    def test_full_metadata(self, toolkit, sample_docs):
        """output_fields=["metadata"] returns full metadata object."""
        result = toolkit._filter_result_fields(sample_docs, ["metadata"])

        assert len(result) == 3
        for i, doc in enumerate(result):
            assert "metadata" in doc
            assert "page_content" not in doc
            assert "score" not in doc
            assert doc["metadata"] == sample_docs[i]["metadata"]

    def test_metadata_and_score(self, toolkit, sample_docs):
        """output_fields=["metadata", "score"] returns both, no page_content."""
        result = toolkit._filter_result_fields(sample_docs, ["metadata", "score"])

        assert len(result) == 3
        for doc in result:
            assert "metadata" in doc
            assert "score" in doc
            assert "page_content" not in doc

    # -------------------------------------------------------------------------
    # Specific metadata fields
    # -------------------------------------------------------------------------

    def test_specific_metadata_field(self, toolkit, sample_docs):
        """output_fields=["metadata.source"] returns only that metadata field."""
        result = toolkit._filter_result_fields(sample_docs, ["metadata.source"])

        assert len(result) == 3
        for doc in result:
            assert "metadata" in doc
            assert doc["metadata"] == {"source": sample_docs[result.index(doc)]["metadata"]["source"]}
            assert "page_content" not in doc
            assert "score" not in doc

        assert result[0]["metadata"]["source"] == "auth.py"
        assert result[1]["metadata"]["source"] == "user.py"

    def test_multiple_metadata_fields(self, toolkit, sample_docs):
        """output_fields with multiple metadata fields returns only those."""
        result = toolkit._filter_result_fields(
            sample_docs, ["metadata.source", "metadata.category"]
        )

        assert len(result) == 3
        for i, doc in enumerate(result):
            assert "metadata" in doc
            assert set(doc["metadata"].keys()) == {"source", "category"}
            assert doc["metadata"]["source"] == sample_docs[i]["metadata"]["source"]
            assert doc["metadata"]["category"] == sample_docs[i]["metadata"]["category"]

    def test_metadata_fields_with_score(self, toolkit, sample_docs):
        """Combine specific metadata fields with score."""
        result = toolkit._filter_result_fields(
            sample_docs, ["metadata.source", "metadata.language", "score"]
        )

        assert len(result) == 3
        for i, doc in enumerate(result):
            assert "metadata" in doc
            assert "score" in doc
            assert "page_content" not in doc
            assert "source" in doc["metadata"]
            assert "language" in doc["metadata"]
            assert "category" not in doc["metadata"]

    # -------------------------------------------------------------------------
    # Dotted metadata keys (not nested - split by first dot only)
    # -------------------------------------------------------------------------

    def test_dotted_metadata_key(self, toolkit, sample_docs):
        """metadata.config.timeout extracts key 'config.timeout' (not nested)."""
        result = toolkit._filter_result_fields(sample_docs, ["metadata.config.timeout"])

        # Only first doc has config.timeout
        assert result[0]["metadata"] == {"config.timeout": 30}
        # Other docs don't have this key - metadata should be absent or empty
        assert "metadata" not in result[1] or result[1].get("metadata") == {}
        assert "metadata" not in result[2] or result[2].get("metadata") == {}

    def test_dotted_key_with_regular_field(self, toolkit, sample_docs):
        """Combine dotted key with regular metadata field."""
        result = toolkit._filter_result_fields(
            sample_docs, ["metadata.config.timeout", "metadata.source"]
        )

        # First doc has both
        assert result[0]["metadata"]["source"] == "auth.py"
        assert result[0]["metadata"]["config.timeout"] == 30

        # Other docs have only source
        assert result[1]["metadata"] == {"source": "user.py"}
        assert result[2]["metadata"] == {"source": "queries.sql"}

    # -------------------------------------------------------------------------
    # Edge cases and mixed scenarios
    # -------------------------------------------------------------------------

    def test_mixed_valid_invalid_fields(self, toolkit, sample_docs):
        """Mix of valid and invalid fields returns only valid ones."""
        result = toolkit._filter_result_fields(
            sample_docs, ["invalid", "score", "also_invalid"]
        )

        assert len(result) == 3
        for doc in result:
            assert doc == {"score": doc["score"]}

    def test_metadata_field_not_present_in_some_docs(self, toolkit, sample_docs):
        """Missing metadata field in some docs handled gracefully."""
        # config.timeout only exists in first doc
        result = toolkit._filter_result_fields(
            sample_docs, ["metadata.config.timeout", "score"]
        )

        assert result[0] == {"metadata": {"config.timeout": 30}, "score": 0.89}
        assert result[1] == {"score": 0.75}  # No metadata key, but has score
        assert result[2] == {"score": 0.62}

    def test_full_metadata_overrides_specific(self, toolkit, sample_docs):
        """When both 'metadata' and 'metadata.X' specified, full metadata wins."""
        result = toolkit._filter_result_fields(
            sample_docs, ["metadata", "metadata.source"]
        )

        # Should return full metadata (metadata wins)
        for i, doc in enumerate(result):
            assert doc["metadata"] == sample_docs[i]["metadata"]

    def test_nonexistent_metadata_field_only(self, toolkit, sample_docs):
        """Only nonexistent metadata field + valid score still returns score."""
        result = toolkit._filter_result_fields(
            sample_docs, ["metadata.nonexistent", "score"]
        )

        # Should return score, but no metadata (field doesn't exist)
        for i, doc in enumerate(result):
            assert "score" in doc
            assert "metadata" not in doc

    def test_preserves_document_order(self, toolkit, sample_docs):
        """Filtered results preserve original document order."""
        result = toolkit._filter_result_fields(sample_docs, ["score"])

        scores = [doc["score"] for doc in result]
        assert scores == [0.89, 0.75, 0.62]

    def test_does_not_modify_original(self, toolkit, sample_docs):
        """Filtering does not modify the original docs list."""
        import copy
        original_copy = copy.deepcopy(sample_docs)

        toolkit._filter_result_fields(sample_docs, ["metadata.source"])

        assert sample_docs == original_copy
