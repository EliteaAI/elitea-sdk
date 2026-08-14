"""Filename filter helpers for SharePoint file listing."""

import re
from typing import Callable, Optional, Sequence


def skip_reporter(on_file_skipped: Optional[Callable[[str], None]]) -> Callable[[str], None]:
    """Callback for names the filters excluded, usable without a None check."""
    return on_file_skipped or (lambda file_name: None)


def normalize_extension_filters(extensions: Optional[Sequence[str]]) -> list[str]:
    """Normalize SharePoint filename/extension filters for glob-style matching."""
    if not extensions:
        return []

    normalized = []
    for extension in extensions:
        if not extension or not extension.strip():
            continue
        extension = extension.strip()

        if extension.startswith('*'):
            if extension.startswith('*.') or '.' in extension:
                pattern = extension
            else:
                pattern = f'*.{extension.lstrip("*")}'
        elif extension.startswith('.') and '/' not in extension:
            pattern = '*' + extension
        elif '.' in extension:
            pattern = extension
        else:
            pattern = f'*.{extension}'

        normalized.append(pattern.lower())
    return normalized


def matches_extension_filter(filename: str, normalized_extensions: Optional[Sequence[str]]) -> bool:
    """Return True when a SharePoint filename matches any normalized filter."""
    if not normalized_extensions:
        return False

    return any(
        re.match(re.escape(pattern).replace(r'\*', '.*') + '$', filename, re.IGNORECASE)
        for pattern in normalized_extensions
    )
