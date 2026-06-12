"""
Unit tests for chunking strategies.

Tests cover:
- BisectionChunkingStrategy: bulk → binary split fallback
"""

import pytest
from typing import Dict, List

from elitea_sdk.tools.figma.chunking_strategy import (
    BisectionChunkingStrategy,
    ChunkResult,
)


class VolumeError(Exception):
    """Simulates a volume/timeout error from API."""
    pass


class NetworkError(Exception):
    """Simulates a network error (connection reset, response ended prematurely)."""
    pass


class AuthError(Exception):
    """Simulates a non-retriable auth error."""
    pass


def is_retriable(e: Exception) -> bool:
    """Check if error is retriable (volume or network related)."""
    return isinstance(e, (VolumeError, NetworkError))


class TestBisectionChunkingStrategy:
    """Tests for BisectionChunkingStrategy."""

    def test_bulk_success(self):
        """All items succeed in bulk request - same as sequential."""
        strategy = BisectionChunkingStrategy(min_chunk_size=1)
        items = ["A", "B", "C", "D"]
        call_count = [0]

        def fetch_fn(ids: List[str]) -> Dict[str, str]:
            call_count[0] += 1
            return {id_: f"result_{id_}" for id_ in ids}

        result = strategy.execute(items, fetch_fn, is_retriable)

        assert len(result.successful) == 4
        assert result.failed == []
        assert call_count[0] == 1  # Only one bulk call

    def test_single_problematic_item_isolation(self):
        """
        Single problematic item is isolated efficiently.

        For 8 items with 1 bad item (C), bisection should:
        1. Try [A-H] → fail
        2. Try [A-D] → fail, Try [E-H] → success
        3. Try [A-B] → success, Try [C-D] → fail
        4. Try [C] → fail (isolated), Try [D] → success

        Total: ~7 calls instead of 8 for sequential
        """
        strategy = BisectionChunkingStrategy(min_chunk_size=1)
        items = ["A", "B", "C", "D", "E", "F", "G", "H"]
        problematic = {"C"}
        call_log = []

        def fetch_fn(ids: List[str]) -> Dict[str, str]:
            call_log.append(ids)
            if any(id_ in problematic for id_ in ids):
                raise VolumeError("Contains problematic item")
            return {id_: f"result_{id_}" for id_ in ids}

        result = strategy.execute(items, fetch_fn, is_retriable)

        # Only C should fail
        assert set(result.failed) == {"C"}
        # All others should succeed
        assert len(result.successful) == 7
        assert all(id_ in result.successful for id_ in items if id_ != "C")
        # Should be fewer calls than sequential (which would be 8)
        assert len(call_log) < 8

    def test_first_item_problematic(self):
        """First item is problematic - should still isolate correctly."""
        strategy = BisectionChunkingStrategy(min_chunk_size=1)
        items = ["A", "B", "C", "D"]
        problematic = {"A"}

        def fetch_fn(ids: List[str]) -> Dict[str, str]:
            if any(id_ in problematic for id_ in ids):
                raise VolumeError("Contains problematic item")
            return {id_: f"result_{id_}" for id_ in ids}

        result = strategy.execute(items, fetch_fn, is_retriable)

        assert result.failed == ["A"]
        assert len(result.successful) == 3
        assert "B" in result.successful
        assert "C" in result.successful
        assert "D" in result.successful

    def test_last_item_problematic(self):
        """Last item is problematic - should still isolate correctly."""
        strategy = BisectionChunkingStrategy(min_chunk_size=1)
        items = ["A", "B", "C", "D"]
        problematic = {"D"}

        def fetch_fn(ids: List[str]) -> Dict[str, str]:
            if any(id_ in problematic for id_ in ids):
                raise VolumeError("Contains problematic item")
            return {id_: f"result_{id_}" for id_ in ids}

        result = strategy.execute(items, fetch_fn, is_retriable)

        assert result.failed == ["D"]
        assert len(result.successful) == 3

    def test_multiple_scattered_failures(self):
        """Multiple problematic items scattered across the list."""
        strategy = BisectionChunkingStrategy(min_chunk_size=1)
        items = ["A", "B", "C", "D", "E", "F", "G", "H"]
        problematic = {"B", "E", "G"}

        def fetch_fn(ids: List[str]) -> Dict[str, str]:
            if any(id_ in problematic for id_ in ids):
                raise VolumeError("Contains problematic item")
            return {id_: f"result_{id_}" for id_ in ids}

        result = strategy.execute(items, fetch_fn, is_retriable)

        assert set(result.failed) == problematic
        assert len(result.successful) == 5
        assert all(id_ in result.successful for id_ in items if id_ not in problematic)

    def test_all_items_fail(self):
        """All items fail individually - no infinite loop."""
        strategy = BisectionChunkingStrategy(min_chunk_size=1)
        items = ["A", "B", "C", "D"]

        def fetch_fn(ids: List[str]) -> Dict[str, str]:
            raise VolumeError("Everything fails")

        result = strategy.execute(items, fetch_fn, is_retriable)

        assert set(result.failed) == set(items)
        assert result.successful == {}

    def test_non_retriable_error_propagates(self):
        """Non-retriable errors should propagate up."""
        strategy = BisectionChunkingStrategy(min_chunk_size=1)
        items = ["A", "B", "C"]

        def fetch_fn(ids: List[str]) -> Dict[str, str]:
            raise AuthError("Unauthorized")

        with pytest.raises(AuthError):
            strategy.execute(items, fetch_fn, is_retriable)

    def test_empty_items(self):
        """Empty item list returns empty result."""
        strategy = BisectionChunkingStrategy(min_chunk_size=1)

        def fetch_fn(ids: List[str]) -> Dict[str, str]:
            return {}

        result = strategy.execute([], fetch_fn, is_retriable)

        assert result.successful == {}
        assert result.failed == []

    def test_single_item_success(self):
        """Single item that succeeds."""
        strategy = BisectionChunkingStrategy(min_chunk_size=1)
        items = ["A"]

        def fetch_fn(ids: List[str]) -> Dict[str, str]:
            return {"A": "result_A"}

        result = strategy.execute(items, fetch_fn, is_retriable)

        assert result.successful == {"A": "result_A"}
        assert result.failed == []

    def test_single_item_failure(self):
        """Single item that fails."""
        strategy = BisectionChunkingStrategy(min_chunk_size=1)
        items = ["A"]

        def fetch_fn(ids: List[str]) -> Dict[str, str]:
            raise VolumeError("Item A fails")

        result = strategy.execute(items, fetch_fn, is_retriable)

        assert result.successful == {}
        assert result.failed == ["A"]

    def test_call_count_efficiency(self):
        """
        Verify bisection is more efficient than sequential for single failure.

        For N items with 1 problematic:
        - Sequential: N calls (1 bulk + N individual)
        - Bisection: ~2*log2(N) calls
        """
        strategy = BisectionChunkingStrategy(min_chunk_size=1)
        # 16 items with 1 problematic
        items = [f"item_{i}" for i in range(16)]
        problematic = {"item_7"}
        call_count = [0]

        def fetch_fn(ids: List[str]) -> Dict[str, str]:
            call_count[0] += 1
            if any(id_ in problematic for id_ in ids):
                raise VolumeError("Contains problematic item")
            return {id_: f"result_{id_}" for id_ in ids}

        result = strategy.execute(items, fetch_fn, is_retriable)

        assert result.failed == ["item_7"]
        assert len(result.successful) == 15
        # Bisection should be much fewer than 16 calls
        # Worst case for bisection: 2*log2(16) + overhead = ~10-12 calls
        assert call_count[0] < 16, f"Expected < 16 calls, got {call_count[0]}"

    def test_network_error_is_retriable(self):
        """Network errors (response ended prematurely) should be treated as retriable."""
        strategy = BisectionChunkingStrategy(min_chunk_size=1)
        items = ["A", "B", "C", "D"]
        problematic = {"B"}

        def fetch_fn(ids: List[str]) -> Dict[str, str]:
            if any(id_ in problematic for id_ in ids):
                # Simulate network error like "Response ended prematurely"
                raise NetworkError("Response ended prematurely")
            return {id_: f"result_{id_}" for id_ in ids}

        result = strategy.execute(items, fetch_fn, is_retriable)

        # Only B should fail - network error was treated as retriable
        assert result.failed == ["B"]
        assert len(result.successful) == 3


class TestChunkResult:
    """Tests for ChunkResult dataclass."""

    def test_default_values(self):
        """Default values are empty collections."""
        result = ChunkResult()
        assert result.successful == {}
        assert result.failed == []
        assert result.skipped == []

    def test_with_values(self):
        """Values are stored correctly."""
        result = ChunkResult(
            successful={"A": "result_A"},
            failed=["B"],
            skipped=["C"],
        )
        assert result.successful == {"A": "result_A"}
        assert result.failed == ["B"]
        assert result.skipped == ["C"]
