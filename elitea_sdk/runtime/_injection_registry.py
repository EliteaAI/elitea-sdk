"""In-process registry for mid-turn user input injections (Phase 0 POC).

Thread-safe, keyed by thread_id. A running agent turn drains pending
injections at each tool-call loop boundary and folds them into the next
LLM invocation. Deduplicated per (thread_id, injection_id).
"""

import threading
import uuid as _uuid
from typing import Dict, List, Optional, Tuple

_lock = threading.Lock()
# thread_id -> list of (injection_id, text) not yet drained
_pending: Dict[str, List[Tuple[str, str]]] = {}
# thread_id -> set of injection_ids already seen (dedup across pushes/drains)
_seen: Dict[str, set] = {}
# thread_ids with an active turn registered to receive injections
_active: set = set()


def register(thread_id: str) -> None:
    """Mark a thread_id as accepting injections for an active turn."""
    if not thread_id:
        return
    with _lock:
        _active.add(thread_id)
        _pending.setdefault(thread_id, [])
        _seen.setdefault(thread_id, set())


def unregister(thread_id: str) -> None:
    """Clear all injection state for a finished turn."""
    if not thread_id:
        return
    with _lock:
        _active.discard(thread_id)
        _pending.pop(thread_id, None)
        _seen.pop(thread_id, None)


def is_active(thread_id: str) -> bool:
    """Return True if a turn is currently registered for thread_id."""
    with _lock:
        return thread_id in _active


def push(thread_id: str, text: str, injection_id: Optional[str] = None) -> bool:
    """Queue an injection for thread_id. Returns False on dedup/empty drop."""
    if not thread_id or not text:
        return False
    injection_id = injection_id or str(_uuid.uuid4())
    with _lock:
        seen = _seen.setdefault(thread_id, set())
        if injection_id in seen:
            return False
        seen.add(injection_id)
        _pending.setdefault(thread_id, []).append((injection_id, text))
        return True


def drain(thread_id: str) -> List[str]:
    """Return and clear all pending injection texts for thread_id, in order."""
    if not thread_id:
        return []
    with _lock:
        items = _pending.get(thread_id)
        if not items:
            return []
        texts = [text for _id, text in items]
        _pending[thread_id] = []
        return texts
