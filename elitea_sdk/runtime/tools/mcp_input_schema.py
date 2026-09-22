"""The JSON Schema an MCP tool advertises, prepared for binding to an LLM.

The server's ``inputSchema`` is handed to LangChain as a dict ``args_schema``
rather than rebuilt as a Pydantic model: a rebuild can only express the subset of
JSON Schema it maps, and silently drops the rest (#6690). The schema is changed
only where binding it as declared would fail.
"""
import json
import math
import re
from logging import getLogger
from typing import Any
from json.encoder import encode_basestring_ascii
from urllib.parse import unquote

from langchain_core.utils.json_schema import dereference_refs
from pydantic import TypeAdapter, ValidationError

logger = getLogger(__name__)

# Anthropic (and other LLM providers) require tool schema property names to
# match this pattern. MCP servers are free to expose arbitrary property
# names (e.g. "fname[]"). See
# https://github.com/EliteaAI/elitea_issues/issues/6274
PROPERTY_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")
_INVALID_PROPERTY_CHARS = re.compile(r"[^a-zA-Z0-9_.-]")
# Keywords whose values are instance data, not subschemas, so a "$ref" key
# inside them is a literal, not a reference. Shared with the server's document,
# not copied: nothing that consumes the schema mutates it.
_DATA_KEYWORDS = frozenset({"const", "default", "enum", "examples"})
# Keywords whose values map arbitrary names to subschemas, so their keys are
# never keywords themselves.
_NAMED_SUBSCHEMA_KEYWORDS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})
_DEFINITION_KEYWORDS = ("$defs", "definitions")
_FALLBACK_PROPERTY_KEYWORDS = ("type", "description")
# Growth allowed beyond the declared schema when inlining $refs, in JSON characters
# (what the model is sent). Heavily reused real models grow by ~10 KB; past this the
# tool binds untyped rather than letting a hostile schema exhaust the shared worker.
_EXPANSION_ALLOWANCE_CHARS = 128 * 1024
_DIGITS_PER_BIT = math.log10(2)
# How json.dumps spells the floats float.__repr__ spells differently.
_FLOAT_SPELLINGS = {math.inf: "Infinity", -math.inf: "-Infinity"}
# Pydantic's lax scalar validators, i.e. the conversions the Pydantic args_schema
# applied before #6690, e.g. "5" -> 5 and "true" -> True.
_SCALAR_VALIDATORS = {"integer": TypeAdapter(int), "number": TypeAdapter(float), "boolean": TypeAdapter(bool)}


def sanitize_property_name(name: str) -> str:
    """Sanitize a schema property name so it matches ^[a-zA-Z0-9_.-]{1,64}$.

    Invalid characters (e.g. "[", "]") are stripped. If sanitization yields
    an empty string, a generic fallback name is used. The result is
    truncated to 64 characters.
    """
    if PROPERTY_NAME_PATTERN.match(name):
        return name
    sanitized = _INVALID_PROPERTY_CHARS.sub("", name)
    if not sanitized:
        sanitized = "field"
    return sanitized[:64]


def build_mcp_args_schema(input_schema: Any, tool_name: str = "") -> dict[str, Any]:
    """Return the ``args_schema`` and ``property_name_map`` fields of an MCP tool.

    The map (sanitized name → declared name) lets the tool translate the model's
    argument names back to the ones the server declared before calling it.
    """
    document = input_schema if isinstance(input_schema, dict) else {}
    try:
        schema = _RefInliner(document).inline(
            {keyword: value for keyword, value in document.items() if keyword not in _DEFINITION_KEYWORDS}
        )
    except (_ExpansionBudgetExceeded, RecursionError) as error:
        _warn_binding_untyped(tool_name, error)
        schema = _flat_schema(document)
    schema.setdefault("type", "object")
    properties = schema.get("properties")
    # BaseTool.args indexes schema["properties"] directly, so it must exist even
    # for a tool the server declared without arguments.
    if not isinstance(properties, dict):
        properties = {}
    schema["properties"], property_name_map = _sanitize_property_names(properties)
    if isinstance(schema.get("required"), list):
        original_to_sanitized = {original: sanitized for sanitized, original in property_name_map.items()}
        schema["required"] = [original_to_sanitized.get(name, name) for name in schema["required"]]
    return {"args_schema": _bindable(schema, tool_name), "property_name_map": property_name_map}


def conform_mcp_arguments(schema: dict, arguments: dict) -> dict:
    """Convert string arguments to the scalar type their parameter declares, and drop
    arguments the schema forbids (``additionalProperties: false``).

    A dict args_schema is not validated by LangChain, and pipeline input mappings
    render every value as a string, so without this "5" reaches a server that
    declared an integer.
    """
    properties = schema.get("properties") or {}
    if schema.get("additionalProperties") is False:
        arguments = {name: value for name, value in arguments.items() if name in properties}
    return {name: _coerce_scalar(properties.get(name), value) for name, value in arguments.items()}


def _coerce_scalar(prop: Any, value: Any) -> Any:
    validator = _SCALAR_VALIDATORS.get(_declared_scalar_type(prop)) if isinstance(value, str) else None
    if validator is None:
        return value
    try:
        return validator.validate_python(value)
    except ValidationError:
        return value


def _declared_scalar_type(prop: Any) -> Any:
    if not isinstance(prop, dict):
        return None
    declared = prop.get("type")
    types = list(declared) if isinstance(declared, list) else [declared]
    for alternative in prop.get("anyOf") or prop.get("oneOf") or ():
        if isinstance(alternative, dict):
            types.append(alternative.get("type"))
    non_null = {declared_type for declared_type in types if isinstance(declared_type, str) and declared_type != "null"}
    return non_null.pop() if len(non_null) == 1 else None


def _sanitize_property_names(properties: dict) -> tuple[dict, dict[str, str]]:
    sanitized_properties = {}
    property_name_map = {}
    for name, prop in properties.items():
        sanitized_name = sanitize_property_name(name)
        if sanitized_name != name:
            property_name_map[sanitized_name] = name
        sanitized_properties[sanitized_name] = prop
    return sanitized_properties, property_name_map


class _ExpansionBudgetExceeded(Exception):
    pass


class _RefInliner:
    """Inlines same-document ``$ref``s, against the document the server sent.

    Done here rather than left to LangChain: its resolver raises on refs it cannot
    follow (missing targets, "~1"/"%20" escapes, anchors, network URIs), and one
    raise fails the bind of every tool on the agent. A ref that cannot be inlined
    is dropped, leaving that one node untyped; a recursive one is cut where it
    recurses. Network refs must not be fetched (SEP-2106).

    Every use of a ref is a fresh copy, so refs that branch grow the schema
    exponentially; a server-controlled schema must not be able to exhaust the
    worker, hence the budget on the size of what is emitted.
    """

    def __init__(self, document: dict):
        self._document = document
        self._remaining_chars = _json_size(document) + _EXPANSION_ALLOWANCE_CHARS
        self._targets: dict[str, Any] = {}
        self._resolving: set[str] = set()

    def inline(self, node: Any) -> Any:
        self._spend(_shallow_json_size(node))
        if isinstance(node, list):
            return [self.inline(item) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if not isinstance(ref, str):
            return self._inline_under_keywords(node)
        inlined = self._inline_under_keywords({key: value for key, value in node.items() if key != "$ref"})
        target = self._target(ref)
        if target is None or ref in self._resolving:
            return inlined
        self._resolving.add(ref)
        try:
            resolved = self.inline(target)
        finally:
            self._resolving.discard(ref)
        return {**resolved, **inlined} if isinstance(resolved, dict) else inlined

    def _inline_under_keywords(self, node: dict) -> dict:
        inlined = {}
        for keyword, value in node.items():
            if keyword in _DATA_KEYWORDS:
                self._spend(_json_size(value))
                inlined[keyword] = value
            elif keyword in _NAMED_SUBSCHEMA_KEYWORDS and isinstance(value, dict):
                self._spend(_shallow_json_size(value))
                inlined[keyword] = {name: self.inline(sub) for name, sub in value.items()}
            else:
                inlined[keyword] = self.inline(value)
        return inlined

    def _target(self, ref: str) -> Any:
        # Cached: a ref is used once per place it expands into, and resolving
        # re-parses the whole ref string.
        if ref not in self._targets:
            self._targets[ref] = _resolve_pointer(self._document, ref)
        return self._targets[ref]

    def _spend(self, chars: int) -> None:
        self._remaining_chars -= chars
        if self._remaining_chars < 0:
            raise _ExpansionBudgetExceeded("inlining its $refs exceeds the schema size limit")


def _json_size(node: Any) -> int:
    """Length of ``node`` serialized by ``json.dumps`` with its defaults (integers may overcount by one)."""
    try:
        return len(json.dumps(node, default=str))
    except ValueError:
        # json.dumps refuses an integer past 4300 digits; a server's JSON parser
        # need not enforce that limit.
        return _walked_json_size(node)


def _walked_json_size(node: Any) -> int:
    if isinstance(node, dict):
        return _shallow_json_size(node) + sum(_walked_json_size(value) for value in node.values())
    if isinstance(node, list):
        return _shallow_json_size(node) + sum(_walked_json_size(item) for item in node)
    return _shallow_json_size(node)


def _shallow_json_size(node: Any) -> int:
    """``_json_size`` of ``node`` itself, excluding the values it contains.

    Called for every emitted node, so it uses what ``json.dumps`` uses underneath
    rather than calling it per value.
    """
    if isinstance(node, str):
        # Non-ASCII and control characters serialize as 6-12 character escapes.
        return len(encode_basestring_ascii(node))
    if node is None or node is True:
        return len("null")
    if node is False:
        return len("false")
    if isinstance(node, int):
        return _integer_json_size(node)
    if isinstance(node, float):
        return len(_FLOAT_SPELLINGS.get(node, "") or float.__repr__(node))
    if isinstance(node, dict):
        return 2 + sum(len(encode_basestring_ascii(key)) + len(": ") for key in node) + _separators_size(len(node))
    if isinstance(node, list):
        return 2 + _separators_size(len(node))
    return len(json.dumps(node, default=str))


def _integer_json_size(node: int) -> int:
    try:
        return len(int.__repr__(node))
    except ValueError:
        return int(abs(node).bit_length() * _DIGITS_PER_BIT) + 1 + (node < 0)


def _separators_size(items: int) -> int:
    return len(", ") * max(items - 1, 0)


def _resolve_pointer(document: dict, ref: str) -> Any:
    """Follow a same-document JSON Pointer ref (RFC 6901); None if it is not one or has no target."""
    pointer = unquote(ref[1:]) if ref.startswith("#") else None
    if pointer is None or (pointer and not pointer.startswith("/")):
        return None
    target: Any = document
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(target, dict) and token in target:
            target = target[token]
        elif isinstance(target, list) and token.isdigit() and int(token) < len(target):
            target = target[int(token)]
        else:
            return None
    return target


def _bindable(schema: dict, tool_name: str) -> dict:
    # LangChain still walks the whole schema when binding, including literal
    # instance data and nested property names, and raises on a "$ref" key it
    # finds there. Run its resolver now so a schema that would fail degrades
    # only this tool, instead of failing the bind for the agent.
    try:
        dereference_refs(schema)
        return schema
    except Exception as error:
        _warn_binding_untyped(tool_name, error)
        return _flat_schema(schema)


def _warn_binding_untyped(tool_name: str, error: BaseException) -> None:
    logger.warning(
        "MCP tool '%s' declares an input schema that cannot be bound as declared (%s: %s); "
        "binding its parameters untyped",
        tool_name, type(error).__name__, error,
    )


def _flat_schema(schema: dict) -> dict:
    declared_properties = schema.get("properties")
    properties = {
        name: {keyword: prop[keyword] for keyword in _FALLBACK_PROPERTY_KEYWORDS if isinstance(prop.get(keyword), str)}
        for name, prop in (declared_properties.items() if isinstance(declared_properties, dict) else ())
        if isinstance(prop, dict)
    }
    flat = {"type": "object", "properties": properties}
    if isinstance(schema.get("required"), list):
        flat["required"] = [name for name in schema["required"] if name in properties]
    return flat
