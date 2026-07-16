"""Shared bounds for trace-step fields transported to persistence."""

import json
from typing import Any


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
