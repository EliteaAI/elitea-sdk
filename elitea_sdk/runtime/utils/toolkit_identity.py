"""Compatibility helpers for persisted toolkit identity names."""

from __future__ import annotations

import re
from typing import Any


_LEGACY_TOOLKIT_NAME_GAP = re.compile(r"[\s_]+")


def toolkit_identity_key(value: Any) -> str:
    """Collapse historical space/underscore drift in persisted toolkit names."""
    return _LEGACY_TOOLKIT_NAME_GAP.sub("", str(value or "").strip().lower())


def toolkit_names_match(left: Any, right: Any) -> bool:
    """Match current and legacy toolkit names without accepting empty aliases."""
    left_key = toolkit_identity_key(left)
    return bool(left_key) and left_key == toolkit_identity_key(right)
