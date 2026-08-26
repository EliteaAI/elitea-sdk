"""
Tests for the load-phase index_meta heartbeat in `BaseIndexerToolkit`.

The document-loading phase used to write no `updated_on` refresh at all: the
loader generator was drained before the first chunk-loop heartbeat, so a live
run with a slow source read as stale (default threshold 2h), and the platform's
abandoned-run reclaim could not tell it apart from a dead one. A timer thread
now spans the whole load phase — including work a loader does before its first
yield, and a single hung request — while keeping the row's writer count at one.
"""
import threading
import time
from unittest.mock import MagicMock

import pytest

from elitea_sdk.tools.base_indexer_toolkit import (
    BaseIndexerToolkit,
    COMPLETED_INDEX_STATES,
)
from elitea_sdk.runtime.utils.utils import IndexerKeywords


def _wrapper():
    # Bypass full Pydantic validation — we only need the methods bound.
    wrapper = BaseIndexerToolkit.model_construct()
    object.__setattr__(wrapper, "index_meta_update", MagicMock())
    return wrapper


class TestIndexMetaHeartbeatTimer:

    def test_ticks_during_work_that_yields_nothing(self):
        # Most loaders bulk-fetch the whole source before their first yield; the
        # timer must cover that phase, which any per-document tick cannot.
        wrapper = _wrapper()
        with wrapper._index_meta_heartbeat("idx", interval=0.02):
            time.sleep(0.2)
        assert wrapper.index_meta_update.call_count >= 1
        for call in wrapper.index_meta_update.call_args_list:
            assert call.args == ("idx", IndexerKeywords.INDEX_META_IN_PROGRESS.value, 0)
            assert call.kwargs["update_force"] is False
            assert call.kwargs["refresh_counts"] is False

    def test_stops_ticking_after_exit(self):
        wrapper = _wrapper()
        with wrapper._index_meta_heartbeat("idx", interval=0.02):
            time.sleep(0.05)
        settled = wrapper.index_meta_update.call_count
        time.sleep(0.1)
        assert wrapper.index_meta_update.call_count == settled

    def test_thread_is_joined_when_the_loader_raises(self):
        wrapper = _wrapper()
        with pytest.raises(RuntimeError, match="source down"):
            with wrapper._index_meta_heartbeat("idx", interval=0.02):
                raise RuntimeError("source down")
        assert not any(
            thread.name.startswith("index-heartbeat-") and thread.is_alive()
            for thread in threading.enumerate()
        )

    def test_tick_failure_does_not_break_loading(self):
        wrapper = _wrapper()
        wrapper.index_meta_update.side_effect = RuntimeError("db gone")
        with wrapper._index_meta_heartbeat("idx", interval=0.02):
            time.sleep(0.08)
        assert wrapper.index_meta_update.call_count >= 1


class TestInterruptedStateContract:

    def test_interrupted_is_not_a_completed_state(self):
        # The platform-side reclaim writes 'interrupted' for abandoned runs; it
        # must never count as a searchable/completed generation here.
        assert "interrupted" not in COMPLETED_INDEX_STATES
        assert COMPLETED_INDEX_STATES == {
            IndexerKeywords.INDEX_META_COMPLETED.value,
            IndexerKeywords.INDEX_META_PARTLY_OK.value,
            IndexerKeywords.INDEX_META_SCHEDULED_REINDEX.value,
        }
