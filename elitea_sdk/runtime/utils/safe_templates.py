"""Hardened template helpers for author-controlled text (#7681).

- ``safe_format``: ``str.format`` that refuses private/dunder attribute access (``{x.__class__}``).
- ``HardenedSandboxedEnvironment``: Jinja sandbox that also caps ``*`` / ``**`` results and
  ``%`` format widths.

The size caps are a partial mitigation: format-spec widths (``'{:>N}'.format``), padding filters
(``center``/``ljust``/``indent``) and large loops are not bounded here; full CPU/memory bounding
belongs to worker-level limits.
"""

import re
import string
from typing import Any

from _string import formatter_field_name_split
from jinja2.exceptions import SecurityError
from jinja2.sandbox import SandboxedEnvironment

MAX_REPEAT_ITEMS = 1_000_000
MAX_POW_EXPONENT = 10_000
MAX_POW_BITS = 1_000_000
MAX_FORMAT_WIDTH = 10_000

_PERCENT_WIDTH = re.compile(r'%(?:\([^)]*\))?[-#0 +]*(\*|\d*)(?:\.(\*|\d*))?')


class TemplateRejectedError(Exception):
    """An author-supplied template was rejected by the sandbox."""


class UnsafePlaceholderError(KeyError):
    """A ``{}`` placeholder tried to read a private/dunder attribute.

    Subclasses ``KeyError`` so callers' existing missing-placeholder fallbacks apply.
    """


class _SafeFormatter(string.Formatter):
    def get_field(self, field_name, args, kwargs):
        _, rest = formatter_field_name_split(field_name)
        for is_attr, key in rest:
            if is_attr and key.startswith('_'):
                raise UnsafePlaceholderError(f"unsafe placeholder: {{{field_name}}}")
        return super().get_field(field_name, args, kwargs)


_FORMATTER = _SafeFormatter()


def safe_format(template: str, /, *args: Any, **values: Any) -> str:
    """``template.format(*args, **values)`` that rejects ``_``-prefixed attribute access."""
    return _FORMATTER.vformat(template, args, values)


class HardenedSandboxedEnvironment(SandboxedEnvironment):
    """SandboxedEnvironment that rejects ``*`` / ``**`` / ``%`` producing very large values."""

    intercepted_binops = frozenset({'*', '**', '%'})

    def call_binop(self, context, operator, left, right):
        if operator == '*':
            for seq, count in ((left, right), (right, left)):
                if isinstance(seq, (str, bytes, list, tuple)) and isinstance(count, int):
                    if len(seq) * max(count, 0) > MAX_REPEAT_ITEMS:
                        raise SecurityError("repetition result is too large")
        elif operator == '**':
            if isinstance(left, int) and isinstance(right, int) and abs(left) > 1:
                if right > MAX_POW_EXPONENT or right * abs(left).bit_length() > MAX_POW_BITS:
                    raise SecurityError("exponent result is too large")
        elif operator == '%' and isinstance(left, str):
            for spec in _PERCENT_WIDTH.findall(left):
                if any(v == '*' or int(v or 0) > MAX_FORMAT_WIDTH for v in spec):
                    raise SecurityError("format width is too large")
        return super().call_binop(context, operator, left, right)
