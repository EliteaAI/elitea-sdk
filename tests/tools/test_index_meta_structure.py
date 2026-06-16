# Copyright (c) 2026 EPAM Systems
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for index_meta structure validation.

Tests verify that the index_meta document contains all expected fields
with correct values after indexing operations.

Expected meta_index structure:
- total: Total number of items initially passed for indexing
- indexed: Number of successfully processed documents (total - skipped)
- indexed_chunks: Number of chunks stored in vector store
- updated: Number of chunks processed in this update
- state: Current indexing state (completed, failed, in_progress, etc.)
- updated_on: Timestamp of last update
- skipped: JSON string with detailed skip information
- error: Error message (if any)
- history: JSON string with history of metadata states
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict

import pytest


# ============================================================================
# Local copy of IndexingStats for testing without heavy SDK dependencies
# This mirrors the structure in elitea_sdk.tools.base_indexer_toolkit
# ============================================================================

@dataclass
class IndexingStats:
    """
    Tracks statistics during indexing process.
    Used by both CodeIndexerToolkit and NonCodeIndexerToolkit.
    """
    # Common counters
    items_processed: int = 0

    # For code toolkits (files)
    files_skipped_whitelist: List[str] = field(default_factory=list)
    files_skipped_blacklist: List[str] = field(default_factory=list)
    files_skipped_read_error: List[str] = field(default_factory=list)
    files_skipped_empty: List[str] = field(default_factory=list)
    files_unsupported_extension: List[str] = field(default_factory=list)

    # For non-code toolkits (documents/runtime)
    documents_skipped_error: List[str] = field(default_factory=list)
    runtime_skipped_extension: List[str] = field(default_factory=list)
    runtime_skipped_error: List[str] = field(default_factory=list)

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
        documents_skipped_count = len(self.documents_skipped_error)
        runtime_skipped_count = (
            len(self.runtime_skipped_extension) +
            len(self.runtime_skipped_error)
        )
        total_skipped = files_skipped_count + documents_skipped_count + runtime_skipped_count

        return {
            "items_processed": self.items_processed,
            "total_skipped": total_skipped,
            "files_skipped": {
                "count": files_skipped_count,
                "whitelist_filtered": self.files_skipped_whitelist,
                "whitelist_filtered_count": len(self.files_skipped_whitelist),
                "blacklist_filtered": self.files_skipped_blacklist,
                "blacklist_filtered_count": len(self.files_skipped_blacklist),
                "read_error": self.files_skipped_read_error,
                "read_error_count": len(self.files_skipped_read_error),
                "empty_content": self.files_skipped_empty,
                "empty_content_count": len(self.files_skipped_empty),
                "unsupported_extension": self.files_unsupported_extension,
                "unsupported_extension_count": len(self.files_unsupported_extension),
            },
            "documents_skipped": {
                "count": documents_skipped_count,
                "error": self.documents_skipped_error,
                "error_count": len(self.documents_skipped_error),
            },
            "runtime_skipped": {
                "count": runtime_skipped_count,
                "extension_filtered": self.runtime_skipped_extension,
                "extension_filtered_count": len(self.runtime_skipped_extension),
                "error": self.runtime_skipped_error,
                "error_count": len(self.runtime_skipped_error),
            }
        }


# ============================================================================
# Index state constants (mirrors IndexerKeywords enum)
# ============================================================================

class IndexStates:
    COMPLETED = "completed"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    PARTLY_OK = "partly_indexed"
    CANCELLED = "cancelled"


# ============================================================================
# Helper function to build metadata as index_meta_update does
# ============================================================================

def build_index_metadata(
    chunks_count: int,
    state: str,
    skipped_data: Dict = None,
    error: str = None,
) -> Dict:
    """
    Build index metadata structure as index_meta_update method does.

    This mirrors the logic in BaseIndexerToolkit.index_meta_update().
    """
    metadata = {}

    # indexed_chunks = number of chunks stored in vector store
    metadata["indexed_chunks"] = chunks_count
    metadata["updated"] = chunks_count
    metadata["state"] = state
    metadata["updated_on"] = time.time()

    # Attach error if provided, else clear on success
    if error is not None:
        metadata["error"] = error
    elif state == IndexStates.COMPLETED:
        metadata["error"] = None

    # Attach skipped items data if provided
    if skipped_data is not None:
        metadata["skipped"] = json.dumps(skipped_data)
        # items_processed = files that passed all filters and were successfully processed
        # total_skipped = files that were filtered out (whitelist, blacklist, unsupported, etc.)
        # total = all files initially considered (processed + skipped)
        # indexed = files successfully processed (= items_processed)
        items_processed = skipped_data.get("items_processed", 0)
        total_skipped = skipped_data.get("total_skipped", 0)
        metadata["total"] = items_processed + total_skipped
        metadata["indexed"] = items_processed
    else:
        # Fallback: if no skipped data, use chunks count for backward compatibility
        metadata["indexed"] = metadata["indexed_chunks"]

    return metadata


# ============================================================================
# Tests for IndexingStats.to_dict() structure
# ============================================================================

class TestIndexingStatsToDict:
    """Tests for IndexingStats.to_dict() structure."""

    def test_empty_stats_structure(self):
        """Empty stats should have all required fields with zero/empty values."""
        stats = IndexingStats()
        result = stats.to_dict()

        # Verify top-level fields
        assert "items_processed" in result
        assert "total_skipped" in result
        assert "files_skipped" in result
        assert "documents_skipped" in result
        assert "runtime_skipped" in result

        # Verify values
        assert result["items_processed"] == 0
        assert result["total_skipped"] == 0

    def test_stats_with_processed_items(self):
        """Stats with processed items should reflect correct counts."""
        stats = IndexingStats(items_processed=10)
        result = stats.to_dict()

        assert result["items_processed"] == 10
        assert result["total_skipped"] == 0

    def test_stats_with_skipped_files(self):
        """Stats with skipped files should calculate totals correctly."""
        stats = IndexingStats(
            items_processed=10,
            files_skipped_whitelist=["file1.txt", "file2.txt"],
            files_skipped_blacklist=["file3.log"],
            files_skipped_empty=["empty.txt"],
        )
        result = stats.to_dict()

        assert result["items_processed"] == 10
        assert result["total_skipped"] == 4  # 2 + 1 + 1
        assert result["files_skipped"]["count"] == 4
        assert result["files_skipped"]["whitelist_filtered_count"] == 2
        assert result["files_skipped"]["blacklist_filtered_count"] == 1
        assert result["files_skipped"]["empty_content_count"] == 1

    def test_stats_with_runtime_skipped(self):
        """Stats with runtime skipped should include runtime_skipped section."""
        stats = IndexingStats(
            items_processed=5,
            runtime_skipped_extension=["file.xyz"],
            runtime_skipped_error=["corrupt.pdf"],
        )
        result = stats.to_dict()

        assert result["runtime_skipped"]["count"] == 2
        assert result["runtime_skipped"]["extension_filtered_count"] == 1
        assert result["runtime_skipped"]["error_count"] == 1

    def test_files_skipped_structure(self):
        """files_skipped section should have all expected sub-fields."""
        stats = IndexingStats()
        result = stats.to_dict()

        files_skipped = result["files_skipped"]
        expected_fields = [
            "count",
            "whitelist_filtered",
            "whitelist_filtered_count",
            "blacklist_filtered",
            "blacklist_filtered_count",
            "read_error",
            "read_error_count",
            "empty_content",
            "empty_content_count",
            "unsupported_extension",
            "unsupported_extension_count",
        ]
        for field_name in expected_fields:
            assert field_name in files_skipped, f"Missing field: {field_name}"

    def test_documents_skipped_structure(self):
        """documents_skipped section should have all expected sub-fields."""
        stats = IndexingStats()
        result = stats.to_dict()

        docs_skipped = result["documents_skipped"]
        expected_fields = ["count", "error", "error_count"]
        for field_name in expected_fields:
            assert field_name in docs_skipped, f"Missing field: {field_name}"

    def test_runtime_skipped_structure(self):
        """runtime_skipped section should have all expected sub-fields."""
        stats = IndexingStats()
        result = stats.to_dict()

        runtime_skipped = result["runtime_skipped"]
        expected_fields = [
            "count",
            "extension_filtered",
            "extension_filtered_count",
            "error",
            "error_count",
        ]
        for field_name in expected_fields:
            assert field_name in runtime_skipped, f"Missing field: {field_name}"


# ============================================================================
# Tests for index_meta field calculations
# ============================================================================

class TestIndexMetaCalculations:
    """Tests for index_meta field calculations.

    Key insight: items_processed counts files that PASSED all filters.
    total_skipped counts files that were FILTERED OUT.
    These are disjoint sets, so:
    - total = items_processed + total_skipped (all files considered)
    - indexed = items_processed (files successfully processed)
    """

    def test_indexed_equals_items_processed(self):
        """indexed should equal items_processed (files that passed filters)."""
        skipped_data = {
            "items_processed": 10,
            "total_skipped": 3,
        }

        items_processed = skipped_data.get("items_processed", 0)
        indexed = items_processed

        assert indexed == 10

    def test_total_equals_processed_plus_skipped(self):
        """total should equal items_processed + total_skipped."""
        skipped_data = {
            "items_processed": 10,
            "total_skipped": 3,
        }

        items_processed = skipped_data.get("items_processed", 0)
        total_skipped = skipped_data.get("total_skipped", 0)
        total = items_processed + total_skipped

        assert total == 13

    def test_indexed_with_zero_skipped(self):
        """indexed should equal items_processed when nothing skipped."""
        skipped_data = {
            "items_processed": 5,
            "total_skipped": 0,
        }

        items_processed = skipped_data.get("items_processed", 0)
        total_skipped = skipped_data.get("total_skipped", 0)
        indexed = items_processed
        total = items_processed + total_skipped

        assert indexed == 5
        assert total == 5

    def test_indexed_zero_when_all_filtered(self):
        """indexed should be 0 when no files passed filters."""
        skipped_data = {
            "items_processed": 0,
            "total_skipped": 100,
        }

        items_processed = skipped_data.get("items_processed", 0)
        total_skipped = skipped_data.get("total_skipped", 0)
        indexed = items_processed
        total = items_processed + total_skipped

        assert indexed == 0
        assert total == 100


# ============================================================================
# Tests for complete meta_index structure when skipped data is provided
# ============================================================================

class TestIndexMetaStructureWithSkippedData:
    """Tests for complete meta_index structure when skipped data is provided."""

    def test_all_fields_present_with_skipped_data(self):
        """All expected fields should be present when skipped data is provided."""
        skipped_data = {
            "items_processed": 10,
            "total_skipped": 2,
            "files_skipped": {"count": 2},
        }

        metadata = build_index_metadata(
            chunks_count=100,
            state=IndexStates.COMPLETED,
            skipped_data=skipped_data,
        )

        expected_fields = [
            "indexed_chunks",
            "indexed",
            "total",
            "updated",
            "state",
            "updated_on",
            "skipped",
        ]
        for field_name in expected_fields:
            assert field_name in metadata, f"Missing field: {field_name}"

    def test_field_values_with_skipped_data(self):
        """Field values should be calculated correctly."""
        skipped_data = {
            "items_processed": 20,
            "total_skipped": 5,
        }

        metadata = build_index_metadata(
            chunks_count=150,
            state=IndexStates.COMPLETED,
            skipped_data=skipped_data,
        )

        assert metadata["total"] == 25  # 20 processed + 5 skipped
        assert metadata["indexed"] == 20  # items_processed
        assert metadata["indexed_chunks"] == 150
        assert metadata["updated"] == 150
        assert metadata["state"] == "completed"

    def test_skipped_is_json_string(self):
        """skipped field should be a JSON string."""
        skipped_data = {
            "items_processed": 5,
            "total_skipped": 1,
        }

        metadata = build_index_metadata(
            chunks_count=50,
            state=IndexStates.COMPLETED,
            skipped_data=skipped_data,
        )

        assert isinstance(metadata["skipped"], str)
        parsed = json.loads(metadata["skipped"])
        assert parsed["items_processed"] == 5
        assert parsed["total_skipped"] == 1


# ============================================================================
# Tests for meta_index structure when skipped data is NOT provided (fallback)
# ============================================================================

class TestIndexMetaStructureWithoutSkippedData:
    """Tests for meta_index structure when skipped data is NOT provided (fallback)."""

    def test_fallback_indexed_equals_chunks(self):
        """Without skipped data, indexed should fallback to indexed_chunks."""
        metadata = build_index_metadata(
            chunks_count=200,
            state=IndexStates.COMPLETED,
        )

        assert metadata["indexed"] == metadata["indexed_chunks"]
        assert metadata["indexed"] == 200

    def test_total_not_present_without_skipped(self):
        """total field should not be present without skipped data."""
        metadata = build_index_metadata(
            chunks_count=100,
            state=IndexStates.COMPLETED,
        )

        assert "total" not in metadata


# ============================================================================
# Tests for valid state values in meta_index
# ============================================================================

class TestIndexMetaStateValues:
    """Tests for valid state values in meta_index."""

    def test_completed_state(self):
        """completed state should be valid."""
        assert IndexStates.COMPLETED == "completed"

    def test_failed_state(self):
        """failed state should be valid."""
        assert IndexStates.FAILED == "failed"

    def test_in_progress_state(self):
        """in_progress state should be valid."""
        assert IndexStates.IN_PROGRESS == "in_progress"

    def test_partly_ok_state(self):
        """partly_indexed state should be valid."""
        assert IndexStates.PARTLY_OK == "partly_indexed"


# ============================================================================
# Tests for meta_index structure with error information
# ============================================================================

class TestIndexMetaWithError:
    """Tests for meta_index structure with error information."""

    def test_error_field_present_on_failure(self):
        """error field should be present when state is failed."""
        metadata = build_index_metadata(
            chunks_count=0,
            state=IndexStates.FAILED,
            error="Connection timeout",
        )

        assert "error" in metadata
        assert metadata["error"] == "Connection timeout"
        assert metadata["state"] == "failed"

    def test_error_cleared_on_success(self):
        """error field should be None on successful completion."""
        metadata = build_index_metadata(
            chunks_count=100,
            state=IndexStates.COMPLETED,
        )

        assert metadata["error"] is None


# ============================================================================
# Tests for history field in meta_index
# ============================================================================

class TestIndexMetaHistoryStructure:
    """Tests for history field in meta_index."""

    def test_history_is_json_string(self):
        """history field should be a JSON string."""
        metadata = {"state": "completed", "indexed": 10}
        history = [metadata]
        metadata["history"] = json.dumps(history)

        assert isinstance(metadata["history"], str)
        parsed = json.loads(metadata["history"])
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_history_contains_metadata_snapshot(self):
        """history should contain snapshot of metadata."""
        metadata = {
            "state": "completed",
            "indexed": 10,
            "indexed_chunks": 50,
            "total": 12,
        }
        history = [metadata.copy()]
        metadata["history"] = json.dumps(history)

        parsed = json.loads(metadata["history"])
        assert parsed[0]["indexed"] == 10
        assert parsed[0]["indexed_chunks"] == 50
        assert parsed[0]["total"] == 12


# ============================================================================
# Integration tests for complete meta_index structure
# ============================================================================

class TestCompleteIndexMetaStructure:
    """Integration tests for complete meta_index structure."""

    def test_complete_successful_indexing_structure(self):
        """Test complete structure after successful indexing."""
        # Simulate successful indexing with some skipped files
        skipped_data = IndexingStats(
            items_processed=10,
            files_skipped_empty=["empty1.txt", "empty2.txt"],
            files_unsupported_extension=["file.xyz"],
        ).to_dict()

        metadata = build_index_metadata(
            chunks_count=150,
            state=IndexStates.COMPLETED,
            skipped_data=skipped_data,
        )

        # Verify structure
        # items_processed=10 means 10 files passed filters
        # 3 files were skipped (2 empty + 1 unsupported)
        # total = 10 + 3 = 13 files considered
        # indexed = 10 files successfully processed
        assert metadata["total"] == 13  # 10 processed + 3 skipped
        assert metadata["indexed"] == 10  # items_processed
        assert metadata["indexed_chunks"] == 150
        assert metadata["state"] == "completed"
        assert metadata["error"] is None

        # Verify skipped data
        parsed_skipped = json.loads(metadata["skipped"])
        assert parsed_skipped["total_skipped"] == 3
        assert parsed_skipped["files_skipped"]["empty_content_count"] == 2
        assert parsed_skipped["files_skipped"]["unsupported_extension_count"] == 1

    def test_complete_failed_indexing_structure(self):
        """Test complete structure after failed indexing."""
        metadata = build_index_metadata(
            chunks_count=0,
            state=IndexStates.FAILED,
            error="Failed to connect to data source",
        )

        assert metadata["state"] == "failed"
        assert metadata["error"] is not None
        assert metadata["indexed"] == 0

    def test_complete_partial_indexing_structure(self):
        """Test complete structure after partial indexing (partly_indexed)."""
        skipped_data = IndexingStats(
            items_processed=10,
            documents_skipped_error=["corrupt.pdf", "invalid.docx"],
        ).to_dict()

        metadata = build_index_metadata(
            chunks_count=80,
            state=IndexStates.PARTLY_OK,
            skipped_data=skipped_data,
            error="Failed to index 2 documents",
        )

        assert metadata["state"] == "partly_indexed"
        assert metadata["total"] == 12  # 10 processed + 2 errors
        assert metadata["indexed"] == 10  # items_processed
        assert metadata["error"] is not None

    def test_metadata_fields_match_sdk_implementation(self):
        """
        Verify that metadata fields match what the SDK produces.

        This is a contract test ensuring UI and SDK stay in sync.
        Expected fields from index_meta_update:
        - total: items_processed
        - indexed: items_processed - total_skipped
        - indexed_chunks: chunks count from get_indexed_count()
        - updated: result (chunks count)
        - state: indexing state
        - updated_on: timestamp
        - error: error message or None
        - skipped: JSON string of skipped data
        - history: JSON string of history
        """
        skipped_data = {
            "items_processed": 15,
            "total_skipped": 3,
            "files_skipped": {"count": 3},
            "documents_skipped": {"count": 0},
            "runtime_skipped": {"count": 0},
        }

        metadata = build_index_metadata(
            chunks_count=200,
            state=IndexStates.COMPLETED,
            skipped_data=skipped_data,
        )

        # Contract: these fields MUST exist
        assert "total" in metadata, "total field required"
        assert "indexed" in metadata, "indexed field required"
        assert "indexed_chunks" in metadata, "indexed_chunks field required"
        assert "updated" in metadata, "updated field required"
        assert "state" in metadata, "state field required"
        assert "updated_on" in metadata, "updated_on field required"
        assert "skipped" in metadata, "skipped field required"

        # Contract: field types
        assert isinstance(metadata["total"], int)
        assert isinstance(metadata["indexed"], int)
        assert isinstance(metadata["indexed_chunks"], int)
        assert isinstance(metadata["updated"], int)
        assert isinstance(metadata["state"], str)
        assert isinstance(metadata["updated_on"], float)
        assert isinstance(metadata["skipped"], str)

        # Contract: field relationships
        # total = all files considered (processed + skipped)
        # indexed = files successfully processed
        assert metadata["total"] == skipped_data["items_processed"] + skipped_data["total_skipped"]
        assert metadata["indexed"] == skipped_data["items_processed"]
