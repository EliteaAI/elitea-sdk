#!/usr/bin/env python3
"""
Sanitize JSON files by escaping control characters inside string values.

Control characters (U+0000 through U+001F) inside JSON strings must be escaped.
This script fixes JSON files where an LLM has written raw control characters
(like actual newlines or tabs) inside string values instead of escape sequences.

Usage:
    python sanitize_json.py <file.json> [file2.json ...]
"""

import re
import sys
from pathlib import Path


def escape_control_chars_in_strings(content: str) -> str:
    """
    Escape control characters inside JSON string values.

    This uses a regex to find JSON strings (text between unescaped quotes)
    and replaces any raw control characters with their escape sequences.
    """
    # Control character mapping (U+0000 to U+001F)
    control_char_map = {
        '\x00': '\\u0000', '\x01': '\\u0001', '\x02': '\\u0002', '\x03': '\\u0003',
        '\x04': '\\u0004', '\x05': '\\u0005', '\x06': '\\u0006', '\x07': '\\u0007',
        '\x08': '\\b',     '\x09': '\\t',     '\x0a': '\\n',     '\x0b': '\\u000b',
        '\x0c': '\\f',     '\x0d': '\\r',     '\x0e': '\\u000e', '\x0f': '\\u000f',
        '\x10': '\\u0010', '\x11': '\\u0011', '\x12': '\\u0012', '\x13': '\\u0013',
        '\x14': '\\u0014', '\x15': '\\u0015', '\x16': '\\u0016', '\x17': '\\u0017',
        '\x18': '\\u0018', '\x19': '\\u0019', '\x1a': '\\u001a', '\x1b': '\\u001b',
        '\x1c': '\\u001c', '\x1d': '\\u001d', '\x1e': '\\u001e', '\x1f': '\\u001f',
    }

    def escape_string_content(match: re.Match) -> str:
        """Escape control characters within a matched JSON string."""
        string_content = match.group(1)
        # Replace each control character with its escape sequence
        for char, escape in control_char_map.items():
            string_content = string_content.replace(char, escape)
        return f'"{string_content}"'

    # Pattern to match JSON strings: "..." where the content can include escaped quotes
    # This handles: "text", "text with \"escaped\" quotes", etc.
    # It matches: opening quote, then (escaped char OR non-quote char)*, then closing quote
    json_string_pattern = r'"((?:[^"\\]|\\.)*)(?:")'

    return re.sub(json_string_pattern, escape_string_content, content, flags=re.DOTALL)


def sanitize_json_file(filepath: Path) -> bool:
    """
    Sanitize a JSON file in place.

    Returns True if the file was modified, False otherwise.
    """
    if not filepath.exists():
        print(f"File not found: {filepath}", file=sys.stderr)
        return False

    try:
        original_content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return False

    sanitized_content = escape_control_chars_in_strings(original_content)

    if sanitized_content != original_content:
        try:
            filepath.write_text(sanitized_content, encoding='utf-8')
            print(f"Sanitized: {filepath}")
            return True
        except Exception as e:
            print(f"Error writing {filepath}: {e}", file=sys.stderr)
            return False
    else:
        print(f"No changes needed: {filepath}")
        return False


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.json> [file2.json ...]", file=sys.stderr)
        sys.exit(1)

    modified_count = 0
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        if sanitize_json_file(filepath):
            modified_count += 1

    print(f"Sanitized {modified_count} file(s)")


if __name__ == "__main__":
    main()
