"""
Serialization contract for tool results that must become text.

Toolkit methods return native Python data; every consumer that needs a string --
the LLM tool message, pipeline state messages -- converts here, so no boundary
falls back to a Python repr.
"""
import dataclasses
import json
import logging
import re
from datetime import date, datetime, time
from math import isfinite
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

_COLLECTION_TYPES = (dict, list, tuple, set, frozenset)

# Bare NaN/Infinity tokens, i.e. outside any string literal.
NON_FINITE = re.compile(r'(?<![\w"])(?:-?Infinity|NaN)(?![\w"])')

# "<module.Class object at 0x7f...>" -- the default object repr, address and all.
DEFAULT_REPR = re.compile(r'^<.+ object at 0x[0-9a-f]+>$')


def serialize_tool_result(result: Any) -> str:
    """Render a tool result as text: compact JSON for collections, str() otherwise.

    Compact rather than indented because this text is what the model reads on
    every subsequent turn; the toolkit test panel re-indents for humans on its
    own. Non-collections keep str() so objects whose __str__ is their documented
    LLM rendering (see sharepoint OnenotePageItems) are left alone.
    """
    if isinstance(result, str):
        return result
    if not isinstance(result, _COLLECTION_TYPES):
        if isinstance(result, float) and not isfinite(result):
            return 'null'
        # A bare dataclass, model, datetime or bytes deserves its JSON form as much
        # as a nested one; only wrapping it in a list used to get it. An object with
        # its own __str__ is still untouched -- to_json_primitive falls through to it.
        converted = to_json_primitive(result)
        if isinstance(converted, _COLLECTION_TYPES):
            return serialize_tool_result(converted)
        return converted if isinstance(converted, str) else str(converted)
    try:
        rendered = json.dumps(result, ensure_ascii=False, default=to_json_primitive)
    except Exception:
        # An encoder-level failure (a tuple key, say) rejects the whole document
        # before `default` is ever consulted, so repair the structure and retry
        # rather than degrading the entire payload to a repr.
        return repair_and_dump(result)
    # json.dumps emits bare NaN/Infinity, which no JSON parser accepts -- and the
    # UI parses before it will render a payload as JSON. `default` cannot catch
    # these: float is serializable, so it is never consulted for them.
    if NON_FINITE.search(rendered):
        return repair_and_dump(result, rendered)
    return rendered


def repair_and_dump(result: Any, rendered: str = None) -> str:
    """Serialize again with the values json.dumps rejects replaced, or give up to text."""
    try:
        repaired = json.dumps(make_json_safe(result), ensure_ascii=False, default=to_json_primitive)
        if repaired != rendered and NON_FINITE.search(repaired):
            # The walk should have reached every value; if a bare token survives,
            # say so rather than shipping silently unparseable JSON. An unchanged
            # result just means the token was inside a string all along.
            logger.warning("Non-finite value survived repair for %s", type(result).__name__)
        return repaired
    except Exception:
        # This runs on every tool call of every agent; a serialization failure
        # must degrade to text, never abort the turn.
        logger.debug("Falling back to str() for %s", type(result).__name__, exc_info=True)
        return describe_unrenderable(result)


def make_json_safe(value: Any) -> Any:
    """Replace the values json.dumps mishandles: non-finite floats and exotic keys."""
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, dict):
        return {normalize_key(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [make_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # A dataclass or pydantic model is expanded by `default=` INSIDE json.dumps,
    # i.e. after this walk would have finished — so its own fields would keep any
    # NaN. Expand it here instead and walk what comes back.
    converted = to_json_primitive(value)
    return make_json_safe(converted) if converted is not value else converted


def normalize_key(key: Any) -> Any:
    """json.dumps accepts only scalar keys; a tuple key raises before `default` runs."""
    if key is None or isinstance(key, (str, int, float, bool)):
        return key
    return str(key)


def to_json_primitive(value: Any) -> Any:
    """Convert a value json.dumps rejects into something it accepts."""
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        # float() would silently change the number the toolkit read
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return describe_binary(value)
    if isinstance(value, (set, frozenset)):
        return list(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    dumped = dump_model(value)
    if dumped is not None:
        return dumped
    return describe_unrenderable(value)


def describe_unrenderable(value: Any) -> str:
    """Last resort for a value with no useful text form of its own.

    An object relying on the default repr contributes only a memory address, which
    is noise to the model and makes two identical results diff against each other
    (boto3's StreamingBody reaches this path). Keep the class name, drop the address.
    """
    try:
        text = str(value)
    except Exception:
        return f"<unserializable {type(value).__name__}>"
    return f"<{type(value).__name__}>" if DEFAULT_REPR.match(text) else text


def describe_binary(value: bytes) -> str:
    """Binary payloads must never reach an LLM as pages of mojibake."""
    try:
        return value.decode('utf-8')
    except UnicodeDecodeError:
        return f"<{len(value)} bytes>"


def dump_model(value: Any) -> Any:
    """Extract a mapping from a pydantic v2/v1 model without importing pydantic."""
    dumper = getattr(value, 'model_dump', None)
    if callable(dumper):
        try:
            return dumper(mode='json')
        except Exception:
            logger.debug("model_dump failed for %s", type(value).__name__, exc_info=True)
    if hasattr(value, '__fields__'):
        dumper = getattr(value, 'dict', None)
        if callable(dumper):
            try:
                return dumper()
            except Exception:
                logger.debug("dict() failed for %s", type(value).__name__, exc_info=True)
    return None
