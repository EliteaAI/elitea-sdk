"""Shared bounds for trace-step fields transported to persistence."""

import json
from collections.abc import Mapping
from typing import Any, Optional


# Matches the platform's single bounded read limit (Epic #5431 / TS-5 #5729).
TRACE_STEP_FIELD_MAX_CHARS = 200_000


def cap_trace_text(value: Any, limit: int = TRACE_STEP_FIELD_MAX_CHARS) -> str | None:
    """Return a string no longer than ``limit`` with an explicit truncation marker."""
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    marker = f'\n…[trace field truncated: {len(text)} chars; limit={limit}]'
    return text[:max(limit - len(marker), 0)] + marker[:limit]


def cap_trace_json(value: Any, limit: int = TRACE_STEP_FIELD_MAX_CHARS) -> Any:
    """Bound a JSON-compatible field while preserving its original shape when small.

    Oversized structured inputs become a small explicit envelope instead of an
    invalid partial JSON document.
    """
    if value is None:
        return None
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    if len(serialized) <= limit:
        return json.loads(serialized)

    envelope = {
        '_trace_truncated': True,
        'original_characters': len(serialized),
        'limit': limit,
        'preview': '',
    }
    overhead = len(json.dumps(envelope, ensure_ascii=False))
    preview_limit = max(limit - overhead - 8, 0)
    envelope['preview'] = serialized[:preview_limit]
    # JSON escaping can expand the preview. Shrink deterministically until the
    # serialized envelope itself respects the contract.
    while len(json.dumps(envelope, ensure_ascii=False)) > limit and envelope['preview']:
        overflow = len(json.dumps(envelope, ensure_ascii=False)) - limit
        envelope['preview'] = envelope['preview'][:-max(overflow, 1)]
    return envelope


# --- Tool-result bounds (#6140) ---------------------------------------------
# Same 200K number as the bounded-read cap, so a read tool that already capped
# itself sits AT the limit and passes through here untouched.
TOOL_RESULT_MAX_CHARS = TRACE_STEP_FIELD_MAX_CHARS

# Reserved key carrying the truncation note on a structured result.
TOOL_RESULT_MARKER_KEY = '_elitea_truncated'
# Idempotence sentinel for text results: present => already bounded, leave alone.
TOOL_RESULT_TEXT_SENTINEL = '[tool result truncated:'

_DO_NOT_TRUST = (
    'This is a PARTIAL result - do not treat it as complete. '
    'Narrow the request, page through the data, or tell the user the output was too large.'
)

_bounding_enabled: bool = True
_bounding_limit: int = TOOL_RESULT_MAX_CHARS
_bounding_per_toolkit: dict = {}


def configure_tool_result_limits(enabled=True, limit=None, per_toolkit=None) -> None:
    """Push the platform's tool-result bounds into this process.

    Applied unconditionally (like the toolkit blocklist) so that REMOVING a
    per-toolkit override takes effect, not only adding one.
    """
    global _bounding_enabled, _bounding_limit, _bounding_per_toolkit  # pylint: disable=W0603
    overrides = {}
    # Built fully before any global is assigned: a malformed per_toolkit value must
    # not leave half the bounds updated and the old overrides still live.
    if isinstance(per_toolkit, Mapping):
        for key, value in per_toolkit.items():
            resolved = _positive_int(value, 0)
            if key and resolved:
                overrides[str(key)] = resolved
    elif per_toolkit:
        raise TypeError(f'per_toolkit must be a mapping, got {type(per_toolkit).__name__}')
    _bounding_enabled = bool(enabled)
    _bounding_limit = _positive_int(limit, TOOL_RESULT_MAX_CHARS)
    _bounding_per_toolkit = overrides


def _positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def tool_result_bounding_enabled() -> bool:
    return _bounding_enabled


def resolve_tool_result_limit(toolkit_type: Any = None) -> int:
    """Per-toolkit override, else the global limit. Independent of the on/off flag."""
    if toolkit_type:
        override = _bounding_per_toolkit.get(str(toolkit_type))
        if override:
            return override
    return _bounding_limit


def estimate_chars(value: Any, ceiling: Optional[int] = None) -> int:
    """Approximate serialized length WITHOUT building a serialized copy.

    ``ceiling`` short-circuits as soon as it is passed, so the cost is
    O(min(size, ceiling)) rather than O(size): a 200MB result is recognised as
    oversized after walking a few hundred KB and is never fully serialized.
    Iterative because a recursive walk can blow the stack on deep tool output.
    """
    total = 0
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            total += len(item)
        elif isinstance(item, (bytes, bytearray)):
            total += len(item)
        elif isinstance(item, bool) or item is None:
            total += 5
        elif isinstance(item, (int, float)):
            total += 8
        elif isinstance(item, dict):
            total += 2 + 2 * len(item)
            for key, sub in item.items():
                total += len(key) if isinstance(key, str) else 8
                stack.append(sub)
        elif isinstance(item, (list, tuple, set)):
            total += 2 + len(item)
            stack.extend(item)
        else:
            # Arbitrary object: str() could itself be the expensive thing we are
            # trying to avoid (a DataFrame repr), so probe cheaply and accept an
            # underestimate - such a value is not truncatable anyway.
            try:
                total += len(item)  # type: ignore[arg-type]
            except TypeError:
                total += 16
        if ceiling is not None and total > ceiling:
            return total
    return total


_B64_ALPHABET = frozenset(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=-_'
)
_B64_MIN_CHARS = 1024
# Below this a leaf cannot meaningfully reduce the payload, and cutting it only
# destroys a structural value ('status': 'ok') that downstream nodes index into.
_MIN_TRIM_CHARS = 512


def looks_like_encoded_blob(value: Any) -> bool:
    """Binary carried as text: cutting it yields a corrupt file with no error."""
    if isinstance(value, (bytes, bytearray)):
        return True
    if not isinstance(value, str) or len(value) < _B64_MIN_CHARS:
        return False
    if value[:64].strip().startswith('data:'):
        return True
    sample = value[:512]
    if not all(char in _B64_ALPHABET for char in sample):
        return False
    # Encoded binary is high-entropy: mixed case and digits appear almost at once.
    # Without this, a long low-entropy string (one repeated word, a hex dump) would
    # be dropped whole as if it were an image.
    return (
        any(char.isupper() for char in sample)
        and any(char.islower() for char in sample)
        and any(char.isdigit() for char in sample)
    )


def _text_marker(original: int, limit: int, tool_name: Any) -> str:
    name = f" for tool '{tool_name}'" if tool_name else ''
    return (
        f"\n\n...{TOOL_RESULT_TEXT_SENTINEL} {original} chars exceeded the "
        f"{limit}-char limit{name}.] {_DO_NOT_TRUST}"
    )


def cap_tool_result_text(
    value: str,
    limit: int = TOOL_RESULT_MAX_CHARS,
    tool_name: Any = None,
    original: Optional[int] = None,
) -> str:
    """Cut a text result and append a self-contained, model-readable note."""
    size = original if original is not None else len(value)
    marker = _text_marker(size, limit, tool_name)
    keep = max(limit - len(marker), 0)
    return value[:keep] + marker


def _structured_marker(original: int, limit: int, tool_name: Any) -> dict:
    return {
        'truncated': True,
        'original_characters': original,
        'limit': limit,
        'tool_name': str(tool_name) if tool_name else None,
        'note': _DO_NOT_TRUST,
    }


def _collect_text_leaves(container: Any) -> list:
    """(parent, key, size) for every text/bytes leaf, so the biggest can be cut first."""
    leaves = []
    stack = [container]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            pairs = item.items()
        elif isinstance(item, list):
            pairs = enumerate(item)
        else:
            continue
        for key, sub in list(pairs):
            if isinstance(sub, (str, bytes, bytearray)):
                leaves.append((item, key, len(sub)))
            elif isinstance(sub, (dict, list)):
                stack.append(sub)
    leaves.sort(key=lambda entry: entry[2], reverse=True)
    return leaves


def _collect_bulk_leaves(container: Any) -> list:
    """(parent, key, size) for every non-text leaf, biggest first.

    Text leaves are handled by ``_collect_text_leaves``; this covers the payloads
    that pass leaves alone cannot shrink - long numeric lists, arrays of records.
    """
    leaves = []
    stack = [container]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            pairs = item.items()
        elif isinstance(item, list):
            pairs = enumerate(item)
        else:
            continue
        for key, sub in list(pairs):
            if isinstance(sub, (str, bytes, bytearray)):
                continue
            if isinstance(sub, (dict, list)):
                stack.append(sub)
                leaves.append((item, key, estimate_chars(sub)))
    leaves.sort(key=lambda entry: entry[2], reverse=True)
    return leaves


_TAIL_NOTE = '...[truncated:'


def _drop_tail(seq: list, excess: int, before: int) -> None:
    """Drop trailing elements, sized from the mean element cost.

    Arithmetic rather than re-measuring each candidate length: a binary search over
    ``estimate_chars`` would make bounding itself the CPU-bound step this feature
    exists to avoid.
    """
    if seq and isinstance(seq[-1], str) and seq[-1].startswith(_TAIL_NOTE):
        seq.pop()  # do not stack a second note on a re-trimmed list
    if not seq:
        return
    drop = min(len(seq), int(excess / max(before / len(seq), 1)) + 1)
    if drop > 0:
        del seq[len(seq) - drop:]
        seq.append(f'{_TAIL_NOTE} {drop} more items]')


def _trim_root(container: Any, excess: int) -> None:
    """Shrink the root container itself when the leaf passes under-shot.

    Size estimates are deliberately approximate, and a root list of many small
    records has no single leaf big enough to trim, so without this a result could be
    stamped as truncated while still being oversized.
    """
    if isinstance(container, list) and container:
        _drop_tail(container, excess, estimate_chars(container))
        return
    if not isinstance(container, dict):
        return
    sizes = sorted(
        ((key, estimate_chars(sub)) for key, sub in container.items()
         if key != TOOL_RESULT_MARKER_KEY),
        key=lambda entry: entry[1], reverse=True,
    )
    trimmable = [entry for entry in sizes if entry[1] >= _MIN_TRIM_CHARS]
    if not trimmable:
        # Thousands of individually tiny values: drop entries instead of mangling
        # the small structural ones, which is what a caller indexes into.
        keys = [key for key, _ in sizes]
        for dead in reversed(keys[1:]):
            if excess <= 0:
                break
            excess -= estimate_chars(container.pop(dead)) + len(str(dead))
        return
    for key, size in trimmable:
        if excess <= 0:
            break
        sub = container[key]
        # Proportional first: a small overshoot must not cost a whole value.
        if isinstance(sub, list) and sub:
            _drop_tail(sub, excess, size)
        elif isinstance(sub, str):
            container[key] = sub[:max(len(sub) - excess - 32, 0)] + f'{_TAIL_NOTE} {len(sub)} chars]'
        else:
            container[key] = f'[oversized value dropped: {size} chars]'
        excess -= size - estimate_chars(container[key])


def _trim_bulk_leaf(parent: Any, key: Any, excess: int) -> int:
    """Shrink one non-text leaf, returning the characters reclaimed.

    Lists only: a dict is either shrunk via its own children (already queued as
    leaves) or dropped whole by the root pass.
    """
    leaf = parent[key]
    before = estimate_chars(leaf)
    if not isinstance(leaf, list) or not leaf:
        return 0
    _drop_tail(leaf, excess, before)
    return before - estimate_chars(parent[key])


def cap_tool_result_structure(
    value: Any,
    limit: int = TOOL_RESULT_MAX_CHARS,
    tool_name: Any = None,
    original: Optional[int] = None,
) -> Any:
    """Bound a dict/list result IN PLACE, preserving its type and its small keys.

    Deliberately not ``cap_trace_json``: that replaces the whole value with a
    preview envelope, and downstream pipeline nodes index into this shape.
    Mutating in place is safe because the caller is the first consumer of a value
    that has just come back from ``tool.invoke``.
    """
    size = original if original is not None else estimate_chars(value)
    marker = _structured_marker(size, limit, tool_name)
    # Budget the marker itself (plus its key and a small margin), so the bounded
    # result respects the contract rather than landing just over it.
    excess = size - limit + estimate_chars(marker) + len(TOOL_RESULT_MARKER_KEY) + 64

    for parent, key, leaf_size in _collect_text_leaves(value):
        if excess <= 0:
            break
        if leaf_size < _MIN_TRIM_CHARS:
            continue
        leaf = parent[key]
        if looks_like_encoded_blob(leaf):
            parent[key] = f'[binary content dropped: {leaf_size} chars]'
        elif isinstance(leaf, (bytes, bytearray)):
            parent[key] = f'[binary content dropped: {leaf_size} bytes]'
        else:
            # The per-leaf note counts against the budget too, or the leftover
            # excess bleeds into the next (small, structurally meaningful) leaf.
            note = f'...[truncated: {leaf_size} chars]'
            keep = max(leaf_size - excess - len(note), 0)
            parent[key] = leaf[:keep] + note
        excess -= leaf_size - len(parent[key])

    if excess > 0:
        # Text leaves alone were not enough (a huge numeric list, arrays of records),
        # so the result would stay oversized while stamped as truncated.
        for parent, key, _ in _collect_bulk_leaves(value):
            if excess <= 0:
                break
            # Trimming a list shortens it, so a later entry's index can be stale.
            if isinstance(parent, list) and not 0 <= key < len(parent):
                continue
            if isinstance(parent, dict) and key not in parent:
                continue
            if isinstance(parent[key], (str, bytes, bytearray)):
                continue  # already replaced by an earlier trim
            excess -= _trim_bulk_leaf(parent, key, excess)

    # Estimates are approximate by design, so confirm the contract instead of
    # trusting the arithmetic: never stamp a result truncated while still oversized.
    reserve = estimate_chars(marker) + len(TOOL_RESULT_MARKER_KEY) + 64
    for _ in range(4):
        current = estimate_chars(value)
        if current + reserve <= limit:
            break
        _trim_root(value, current + reserve - limit)

    if isinstance(value, dict):
        value[TOOL_RESULT_MARKER_KEY] = marker
        return value
    if isinstance(value, list):
        # No reserved-key slot on a list: the marker becomes the last element so
        # the reason for the shorter payload travels with the data.
        value.append({TOOL_RESULT_MARKER_KEY: marker})
    return value


def structure_is_bounded(value: Any) -> bool:
    return isinstance(value, dict) and TOOL_RESULT_MARKER_KEY in value


def text_is_bounded(value: Any) -> bool:
    return isinstance(value, str) and TOOL_RESULT_TEXT_SENTINEL in value
