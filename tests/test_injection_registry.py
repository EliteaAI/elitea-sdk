"""Unit tests for the mid-turn user input injection registry (Phase 0)."""

import pytest

from elitea_sdk.runtime import _injection_registry as reg


@pytest.fixture(autouse=True)
def _clean_registry():
    # Ensure each test starts and ends with a clean, isolated thread namespace.
    tid = "t-test"
    reg.unregister(tid)
    yield
    reg.unregister(tid)


def test_push_and_drain_in_order():
    tid = "t-test"
    reg.register(tid)
    assert reg.push(tid, "first")
    assert reg.push(tid, "second")
    assert reg.drain(tid) == ["first", "second"]
    # Second drain is empty — pending was cleared.
    assert reg.drain(tid) == []


def test_dedup_by_injection_id():
    tid = "t-test"
    reg.register(tid)
    assert reg.push(tid, "hello", injection_id="abc")
    # Same id is dropped even with different text.
    assert reg.push(tid, "hello again", injection_id="abc") is False
    assert reg.drain(tid) == ["hello"]


def test_empty_text_dropped():
    tid = "t-test"
    reg.register(tid)
    assert reg.push(tid, "") is False
    assert reg.drain(tid) == []


def test_unregister_clears_state():
    tid = "t-test"
    reg.register(tid)
    reg.push(tid, "x")
    assert reg.is_active(tid)
    reg.unregister(tid)
    assert not reg.is_active(tid)
    assert reg.drain(tid) == []


def test_drain_unknown_thread_is_empty():
    assert reg.drain("nonexistent") == []


def test_is_active_reflects_registration():
    tid = "t-test"
    assert not reg.is_active(tid)
    reg.register(tid)
    assert reg.is_active(tid)


def test_drain_items_returns_ids_for_acking():
    tid = "t-test"
    reg.register(tid)
    reg.push(tid, "one", injection_id="i1")
    reg.push(tid, "two", injection_id="i2")
    assert reg.drain_items(tid) == [("i1", "one"), ("i2", "two")]


def test_consumed_records_only_marked_ids():
    """The turn-end report must reflect what the loop folded in, not what arrived."""
    tid = "t-test"
    reg.register(tid)
    reg.push(tid, "landed", injection_id="i1")
    reg.push(tid, "missed", injection_id="i2")
    reg.drain_items(tid)
    # Only i1 was actually folded into the message list.
    reg.mark_consumed(tid, "i1")
    assert reg.consumed(tid) == ["i1"]


def test_consumed_cleared_on_unregister():
    tid = "t-test"
    reg.register(tid)
    reg.mark_consumed(tid, "i1")
    reg.unregister(tid)
    assert reg.consumed(tid) == []
