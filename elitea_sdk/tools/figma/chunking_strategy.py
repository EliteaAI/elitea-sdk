"""
Chunking strategy for handling large API requests with fallback.

This module provides a bisection strategy for fetching data in chunks when
bulk API requests fail due to size/timeout errors.

Usage:
    strategy = BisectionChunkingStrategy()
    result = strategy.execute(items, fetch_fn, is_retriable_error)
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, List, Tuple, TypeVar

# Type variables for generic strategy
T = TypeVar('T')  # Item type (page_id, node_id, etc.)
R = TypeVar('R')  # Result type (page content, image URL, etc.)


@dataclass
class ChunkResult(Generic[T, R]):
    """Result of processing items with a chunking strategy."""
    successful: Dict[T, R] = field(default_factory=dict)  # item_id → result
    failed: List[T] = field(default_factory=list)         # item_ids that failed
    skipped: List[T] = field(default_factory=list)        # item_ids skipped due to non-retriable errors
    api_calls: int = 0                                     # total API calls made
    splits: int = 0                                        # number of splits performed


class BisectionChunkingStrategy(Generic[T, R]):
    """
    Bisection Fallback Strategy (Divide and Conquer).

    Efficiently handles large API requests by binary splitting on failure.

    Behavior:
    1. Try all items at once (bulk request)
    2. On retriable error, split into 2 halves
    3. Recursively apply to each half
    4. Stop splitting when reaching min_chunk_size (default: 1)

    Advantages:
    - Fewer API calls than sequential for large datasets
    - Isolates problematic items efficiently
    - O(log N) calls in best case vs O(N) for sequential

    Single-Item Isolation Guarantee:
    The algorithm ensures only individual problematic items are marked as failed:
    - On any failure, we split and try BOTH halves
    - Successful items from the "good" half are always preserved
    - We continue splitting the "bad" half until reaching single items
    - Only items that fail at min_chunk_size=1 are marked as failed

    Example trace for [A,B,C,D,E,F,G,H] where C is problematic:
        Try [A-H] → FAIL → split
        Try [A-D] → FAIL → split  |  Try [E-H] → SUCCESS {E,F,G,H}
        Try [A-B] → SUCCESS {A,B} |  Try [C-D] → FAIL → split
                                  |  Try [C] → FAIL (min size) → failed=[C]
                                  |  Try [D] → SUCCESS {D}
        Final: successful={A,B,D,E,F,G,H}, failed=[C]

    Logging: DEBUG for splits and fetches, caller logs INFO summary.
    """

    def __init__(self, min_chunk_size: int = 1):
        """
        Initialize bisection strategy.

        Args:
            min_chunk_size: Stop splitting when chunk reaches this size.
                           Default: 1 (single item) for maximum isolation.
        """
        self.min_chunk_size = min_chunk_size

    def execute(
        self,
        items: List[T],
        fetch_fn: Callable[[List[T]], Dict[T, R]],
        is_retriable_error: Callable[[Exception], bool],
    ) -> ChunkResult[T, R]:
        """
        Execute bisection strategy with recursive binary splitting.

        Args:
            items: List of items to process (e.g., page IDs, node IDs)
            fetch_fn: Function that fetches results for a list of items.
                      Should raise exception on failure, return dict on success.
                      Signature: (List[T]) -> Dict[T, R]
            is_retriable_error: Function to check if an error is retriable
                               (e.g., volume/timeout errors vs auth errors).
                               Signature: (Exception) -> bool

        Returns:
            ChunkResult with successful, failed, and skipped items
        """
        if not items:
            return ChunkResult()

        # Track statistics during execution
        stats = {"api_calls": 0, "splits": 0}

        successful, failed, skipped = self._bisect_fetch(
            items=items,
            fetch_fn=fetch_fn,
            is_retriable_error=is_retriable_error,
            depth=0,
            stats=stats,
        )
        return ChunkResult(
            successful=successful,
            failed=failed,
            skipped=skipped,
            api_calls=stats["api_calls"],
            splits=stats["splits"],
        )

    def _bisect_fetch(
        self,
        items: List[T],
        fetch_fn: Callable[[List[T]], Dict[T, R]],
        is_retriable_error: Callable[[Exception], bool],
        depth: int = 0,
        stats: Dict[str, int] = None,
    ) -> Tuple[Dict[T, R], List[T], List[T]]:
        """
        Recursive bisection fetch (Divide and Conquer).

        Args:
            items: Items to fetch
            fetch_fn: Function to fetch items
            is_retriable_error: Function to check if error is retriable
            depth: Current recursion depth (for logging)
            stats: Mutable dict to track api_calls and splits

        Returns:
            Tuple of (successful_dict, failed_list, skipped_list)
        """
        if stats is None:
            stats = {"api_calls": 0, "splits": 0}

        if not items:
            return {}, [], []

        # Try fetching current chunk
        try:
            stats["api_calls"] += 1
            result = fetch_fn(items)
            logging.debug(f"Bisection: fetched {len(items)} items at depth {depth}")
            return result, [], []
        except Exception as e:
            if not is_retriable_error(e):
                logging.warning(f"Bisection: non-retriable error at depth {depth}: {e}")
                raise  # Non-retriable error (auth, invalid request), propagate up

            # Retriable error (volume/timeout) - check if we can split further
            if len(items) <= self.min_chunk_size:
                logging.debug(f"Bisection: item {items[0]} failed at minimum chunk size (isolated)")
                return {}, list(items), []

            # Split in half and recurse
            stats["splits"] += 1
            mid = len(items) // 2
            left_items = items[:mid]
            right_items = items[mid:]

            logging.debug(
                f"Bisection: splitting {len(items)} items into "
                f"{len(left_items)} + {len(right_items)} at depth {depth}"
            )

            # Process BOTH halves independently
            left_success, left_failed, left_skipped = self._bisect_fetch(
                left_items, fetch_fn, is_retriable_error, depth + 1, stats
            )
            right_success, right_failed, right_skipped = self._bisect_fetch(
                right_items, fetch_fn, is_retriable_error, depth + 1, stats
            )

            # Merge results from both halves
            merged_success = {**left_success, **right_success}
            merged_failed = left_failed + right_failed
            merged_skipped = left_skipped + right_skipped

            return merged_success, merged_failed, merged_skipped
