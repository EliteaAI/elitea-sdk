"""Hardened template helpers for author-controlled text (#7681).

- ``safe_format``: ``str.format`` that refuses attribute/index traversal in placeholders.
- ``HardenedSandboxedEnvironment``: Jinja sandbox that also caps ``*`` / ``**`` results.
"""

import string
from typing import Any

from jinja2.exceptions import SecurityError
from jinja2.sandbox import SandboxedEnvironment

MAX_REPEAT_ITEMS = 1_000_000
MAX_POW_EXPONENT = 10_000
MAX_POW_BITS = 1_000_000


class _NoTraversalFormatter(string.Formatter):
    """Resolves only plain ``{name}`` fields; ``{x.attr}`` / ``{x[0]}`` stay as written."""

    def vformat(self, format_string, args, kwargs):
        result = []
        auto_index = 0
        for literal, field_name, spec, conversion in self.parse(format_string):
            result.append(literal)
            if field_name is None:
                continue
            if '.' in field_name or '[' in field_name:
                result.append(_rebuild(field_name, spec, conversion))
                continue
            if field_name == '':  # auto-numbered {} as in str.format
                field_name = str(auto_index)
                auto_index += 1
            obj, _ = self.get_field(field_name, args, kwargs)
            obj = self.convert_field(obj, conversion)
            result.append(self.format_field(obj, self.vformat(spec, args, kwargs) if spec else ''))
        return ''.join(result)


def _rebuild(field_name: str, spec: str, conversion: str) -> str:
    return '{' + field_name + (f'!{conversion}' if conversion else '') + (f':{spec}' if spec else '') + '}'


_FORMATTER = _NoTraversalFormatter()


def safe_format(template: str, *args: Any, **values: Any) -> str:
    """``template.format(*args, **values)`` without attribute/index traversal.

    Missing names raise ``KeyError`` / ``IndexError`` exactly as ``str.format`` does.
    """
    return _FORMATTER.vformat(template, args, values)


class HardenedSandboxedEnvironment(SandboxedEnvironment):
    """SandboxedEnvironment that rejects ``*`` / ``**`` producing very large values."""

    intercepted_binops = frozenset({'*', '**'})

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
        return super().call_binop(context, operator, left, right)
