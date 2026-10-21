"""Credential slots, applied in the SDK at the last moment (C-1009 WP-1D, #6946).

A governed payload carries, instead of secret values, a *slot* inside each configuration
dict of a toolkit's settings::

    settings["github_configuration"] = {
        "base_url": "...",
        "credential_slot": {"v": 1, "credential_ref": "...", "credential_generation": 1,
                            "decision_id": "...", "fields": ["access_token"]},
    }

`apply_credential_slots` asks an injected resolver for the material and writes it, as plain
``str``, into a NEW copy of the settings. The caller's dict (and every nested dict) is never
modified, so the material never reaches the caller's payload, task kwargs, stashes or
checkpoints. A slot either resolves or the call fails with `GovernedCredentialError`; a
missing value is never dropped, because several toolkits treat a missing secret as anonymous
access.

This module has no Pylon imports and no network code: the resolver is supplied by the
embedding worker (the indexer). All slots of one toolkit are released in ONE call when the
resolver offers it::

    resolver.resolve_many(toolkit_instance_id=<str>, items=[{"field": <settings key>, "slot": <slot dict>}, ...])
        -> {<settings key>: CredentialEnvelope}

and otherwise one slot at a time, as
``resolver(toolkit_instance_id=<str>, field=<settings key>, slot=<slot dict>)``. A resolver that also offers::

    resolver.resolve_all(items=[{"toolkit_instance_id": <str>, "field": <settings key>, "slot": <slot dict>}, ...])
        -> {(<toolkit_instance_id>, <settings key>): CredentialEnvelope}

is called once for the slots of every toolkit of a run (`resolve_run_slots`); the envelopes are then handed to
`apply_credential_slots(released=...)` toolkit by toolkit.
Either way each reply must be a `CredentialEnvelope` (or the call raises `GovernedCredentialError`
with the release refusal code), and every envelope is validated against its own slot here.

Batching releases (and so audits) every slot of the toolkit before any envelope is validated.
One bad envelope still fails the whole build with a typed error, and nothing is used partially:
the new settings are only returned once every slot has been validated and written.
"""

from __future__ import annotations

import functools
import json
import re
from contextvars import ContextVar
from datetime import datetime
from types import MappingProxyType
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, NamedTuple, Optional, Protocol, Tuple, TypeVar, Union

from langchain_core.tools import ToolException
from pydantic import SecretStr

SLOT_KEY = "credential_slot"
SLOT_VERSION = 1
REDACTED = "[REDACTED]"

_MAX_SCAN_DEPTH = 6
_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_FALLBACK_CODE = "DEPENDENCY_UNAVAILABLE"
_DEFAULT_MESSAGE = "The governed credential could not be applied; the run was not started."
_STORE_MESSAGE = "The credential store is unavailable; the run was not started. Try again later."

# Fixed, readable texts per reason code. No titles, uuids, field names or values.
_MESSAGES: Dict[str, str] = {
    "WORKER_CAPABILITY_UNAVAILABLE": (
        "This worker cannot apply governed credentials; the run was not started."
    ),
    "UNSUPPORTED_AUTH_MODE": (
        "This toolkit cannot use a governed credential yet; the run was not started."
    ),
    "RATE_LIMITED": "This run asked for its credentials too often and was stopped.",
    "STALE_CONTEXT": "The run's security context is no longer valid; the run was not started.",
    "CREDENTIAL_REFERENCE_REQUIRED": (
        "A governed credential reference is required; the run was not started."
    ),
    "CREDENTIAL_REFERENCE_INACTIVE": (
        "The credential reference is not active; the run was not started."
    ),
    "CREDENTIAL_GENERATION_MISMATCH": (
        "The credential changed since the run was approved; the run was not started."
    ),
    "DENIED_GRANT": "The credential grant is no longer in force; the run was not started.",
    "DENIED_RBAC": "Permission to use this credential was denied; the run was not started.",
    "DENIED_BINDING": "This agent is no longer bound to the credential; the run was not started.",
    "EPOCH_MISMATCH": "Access was revoked while the run was starting; the run was not started.",
    "MANDATE_INACTIVE": "The mandate for this run is not active; the run was not started.",
    "MANDATE_EXPIRED": "The mandate for this run has expired; the run was not started.",
    "AUDIT_UNAVAILABLE": "Auditing is unavailable; the run was not started.",
    "DEPENDENCY_UNAVAILABLE": (
        "A required platform service is unavailable; the run was not started. Try again later."
    ),
    "STORE_DENIED": _STORE_MESSAGE,
    "STORE_SEALED": _STORE_MESSAGE,
    "STORE_UNAVAILABLE": _STORE_MESSAGE,
    "STORE_TLS_INVALID": _STORE_MESSAGE,
    "STORE_NOT_FOUND": _STORE_MESSAGE,
    "STORE_VERSION_CONFLICT": _STORE_MESSAGE,
}


class GovernedCredentialError(ToolException):
    """A governed credential slot could not be resolved or applied.

    Carries only a reason code and the fixed readable message for it. Never an id, a field
    name or a value, so it is safe to log, return to the UI and pickle across a task boundary.
    """

    def __init__(self, reason_code: str):
        code = reason_code if isinstance(reason_code, str) and _CODE_RE.match(reason_code) else _FALLBACK_CODE
        self.reason_code = code
        self.message = _MESSAGES.get(code, _DEFAULT_MESSAGE)
        super().__init__(self.message)

    @property
    def code(self) -> str:
        return self.reason_code

    def __reduce__(self):
        return (type(self), (self.reason_code,))


class CredentialEnvelope:
    """Released material for one slot. Carrier only: redacted everywhere, never pickled.

    `fields` maps password-schema paths (``.``-joined for nesting) to `SecretStr`.
    """

    __slots__ = ("_fields", "generation", "expires_at")

    def __init__(self, fields: Mapping[str, Union[str, SecretStr]], generation: int,
                 expires_at: Optional[datetime] = None):
        wrapped: Dict[str, SecretStr] = {}
        for path, value in fields.items():
            if isinstance(value, SecretStr):
                wrapped[str(path)] = value
            elif isinstance(value, str):
                wrapped[str(path)] = SecretStr(value)
            else:
                raise TypeError("credential envelope values must be str or SecretStr")
        object.__setattr__(self, "_fields", MappingProxyType(wrapped))
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "expires_at", expires_at)

    @property
    def fields(self) -> Mapping[str, SecretStr]:
        return self._fields

    def __setattr__(self, name, value):
        raise AttributeError("CredentialEnvelope is immutable")

    def __repr__(self) -> str:
        return f"CredentialEnvelope(fields={sorted(self._fields)}, generation={self.generation})"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("CredentialEnvelope must not be pickled or copied")

    def __reduce_ex__(self, protocol):
        raise TypeError("CredentialEnvelope must not be pickled or copied")

    def __getstate__(self):
        raise TypeError("CredentialEnvelope must not be pickled or copied")


class CredentialResolver(Protocol):
    """Injected by the embedding worker. Returns the envelope or raises `GovernedCredentialError`.

    A resolver MAY also expose ``resolve_many(*, toolkit_instance_id, items) -> {field: envelope}``
    (items: ``[{"field": str, "slot": dict}]``); it is then called once per toolkit build instead
    of ``__call__`` once per slot. It MAY also expose ``resolve_all(*, items) -> {(toolkit_instance_id, field): envelope}``
    (items: ``[{"toolkit_instance_id": str, "field": str, "slot": dict}]``) for the slots of a whole run.
    """

    def __call__(self, *, toolkit_instance_id: str, field: str, slot: Mapping[str, Any]) -> CredentialEnvelope:
        ...


def _contains_slot(value: Any, depth: int) -> bool:
    if isinstance(value, dict):
        if SLOT_KEY in value:
            return True
        if depth <= 0:
            return False
        return any(_contains_slot(v, depth - 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        if depth <= 0:
            return False
        return any(_contains_slot(v, depth - 1) for v in value)
    return False


class SlotScan(NamedTuple):
    """What one traversal of a toolkit's settings found. Key names only, never values."""

    found: bool                # a slot exists anywhere (bounded depth)
    slotted: Tuple[str, ...]   # top-level keys whose value is a dict carrying a slot
    nested: bool               # a slot exists somewhere other than directly under a top-level key


_NO_SLOTS = SlotScan(False, (), False)
_F = TypeVar("_F", bound=Callable[..., Any])


class _ScanScope:
    """Scan results memoized for the duration of one `get_tools` (see `memoize_slot_scans`).

    Each entry keeps the scanned dict alive so an `id` cannot be reused by another dict while it
    is cached. `active` is cleared when the call ends, so a task that copied the context and
    outlives the call (an executor running `copy_context().run`) never reads a stale scan.
    """

    __slots__ = ("active", "scans")

    def __init__(self) -> None:
        self.active = True
        self.scans: Dict[int, Tuple[Any, SlotScan]] = {}


_SCAN_SCOPE: ContextVar[Optional[_ScanScope]] = ContextVar("c1009_slot_scan_scope", default=None)


def _scan_settings(settings: Dict[str, Any]) -> SlotScan:
    """The ONE traversal: find the slotted top-level keys and whether any other slot exists.

    Depth: with the root at level 0, a slot is found down to level 7 on both paths (a non-slotted
    top-level value is scanned at depth 6, the children of a slotted config at depth 5).
    """
    slotted: List[str] = []
    nested = SLOT_KEY in settings
    for key, value in settings.items():
        if isinstance(value, dict) and SLOT_KEY in value:
            slotted.append(key)
            if not nested:
                nested = any(
                    _contains_slot(v, _MAX_SCAN_DEPTH - 1) for k, v in value.items() if k != SLOT_KEY
                )
        elif not nested:
            nested = _contains_slot(value, _MAX_SCAN_DEPTH)
    return SlotScan(bool(slotted) or nested, tuple(slotted), nested)


def scan_slots(settings: Any) -> SlotScan:
    """Scan `settings` for credential slots, once per dict inside a `memoize_slot_scans` scope."""
    if not isinstance(settings, dict):
        return _NO_SLOTS
    scope = _SCAN_SCOPE.get()
    if scope is None or not scope.active:
        return _scan_settings(settings)
    hit = scope.scans.get(id(settings))
    if hit is not None and hit[0] is settings:
        return hit[1]
    scan = _scan_settings(settings)
    scope.scans[id(settings)] = (settings, scan)
    return scan


def memoize_slot_scans(fn: _F) -> _F:
    """Decorator: inside `fn`, every settings dict is deep-scanned for slots at most once.

    Nested decorated calls share the outermost scope. The scope is closed when the outermost call
    ends (also on an exception); a context copied inside it no longer serves cached scans.
    Assumes no slot is added to a settings dict while the call runs (the governed payload is
    sealed before the build starts).
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        outer = _SCAN_SCOPE.get()
        if outer is not None and outer.active:
            return fn(*args, **kwargs)
        scope = _ScanScope()
        token = _SCAN_SCOPE.set(scope)
        try:
            return fn(*args, **kwargs)
        finally:
            scope.active = False
            _SCAN_SCOPE.reset(token)

    return wrapper  # type: ignore[return-value]


def has_slots(settings: Any) -> bool:
    """True when `settings` holds a credential slot anywhere (bounded depth). A plain dict scan."""
    return scan_slots(settings).found


def _copy_containers(value: Any) -> Any:
    """Copy dict/list structure; leaves are shared (they are immutable scalars for slot configs)."""
    if isinstance(value, dict):
        return {k: _copy_containers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_containers(v) for v in value]
    return value


def _validate_slot(slot: Any) -> Tuple[int, list]:
    if not isinstance(slot, dict) or slot.get("v") != SLOT_VERSION:
        raise GovernedCredentialError("STALE_CONTEXT")
    generation = slot.get("credential_generation")
    fields = slot.get("fields")
    if (
        isinstance(generation, bool) or not isinstance(generation, int)
        or not isinstance(fields, list)
        or not all(isinstance(f, str) and f and all(f.split(".")) for f in fields)
    ):
        raise GovernedCredentialError("STALE_CONTEXT")
    return generation, fields


def _write_path(config: Dict[str, Any], path: str, value: str) -> None:
    *parents, leaf = path.split(".")
    node = config
    for part in parents:
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        elif not isinstance(child, dict):
            raise GovernedCredentialError("STALE_CONTEXT")
        node = child
    node[leaf] = value


def _resolve_slots(
    resolver: CredentialResolver, toolkit_instance_id: str, items: List[Dict[str, Any]]
) -> Mapping[str, Any]:
    """Release every item: one `resolve_many` call, else one call per slot. Typed errors only."""
    many = getattr(resolver, "resolve_many", None)
    try:
        if callable(many):
            replies = many(toolkit_instance_id=toolkit_instance_id, items=items)
        else:
            replies = {
                item["field"]: resolver(
                    toolkit_instance_id=toolkit_instance_id, field=item["field"], slot=item["slot"]
                )
                for item in items
            }
    except GovernedCredentialError:
        raise
    except Exception:
        # The original text may carry ids or values; the typed code is the contract.
        raise GovernedCredentialError("DEPENDENCY_UNAVAILABLE") from None
    if not isinstance(replies, Mapping) or set(replies) != {item["field"] for item in items}:
        raise GovernedCredentialError("DEPENDENCY_UNAVAILABLE")
    return replies


def resolve_run_slots(resolver: CredentialResolver, items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Release the items of a whole run with one ``resolver.resolve_all`` call; ``{toolkit_instance_id: {field: reply}}``.

    Typed errors only; a reply that is not exactly the requested (toolkit, field) set is ``DEPENDENCY_UNAVAILABLE``.
    """
    try:
        replies = resolver.resolve_all(items=items)  # type: ignore[attr-defined]
    except GovernedCredentialError:
        raise
    except Exception:
        raise GovernedCredentialError("DEPENDENCY_UNAVAILABLE") from None
    if not isinstance(replies, Mapping) or set(replies) != {(i["toolkit_instance_id"], i["field"]) for i in items}:
        raise GovernedCredentialError("DEPENDENCY_UNAVAILABLE")
    by_toolkit: Dict[str, Dict[str, Any]] = {}
    for (toolkit_instance_id, field), envelope in replies.items():
        by_toolkit.setdefault(toolkit_instance_id, {})[field] = envelope
    return by_toolkit


def _checked_scan(settings: Dict[str, Any], scan: Optional[SlotScan]) -> SlotScan:
    if scan is None:
        return scan_slots(settings)
    # A caller's scan is a hint, never trusted: its top-level shape must match these settings.
    top = tuple(k for k, v in settings.items() if isinstance(v, dict) and SLOT_KEY in v)
    if top != scan.slotted or (not scan.found and (top or SLOT_KEY in settings)):
        raise GovernedCredentialError("STALE_CONTEXT")
    return scan


def _plan_slots(settings: Dict[str, Any], toolkit_instance_id: Any, scan: SlotScan):
    """Validate every slot: ``(plan, items)``, plan = ``(key, new_config, generation, paths)`` per slotted key."""
    if not isinstance(toolkit_instance_id, str) or not toolkit_instance_id:
        raise GovernedCredentialError("STALE_CONTEXT")
    plan = []
    items: List[Dict[str, Any]] = []
    for key in scan.slotted:
        config = settings[key]
        slot = config[SLOT_KEY]
        generation, paths = _validate_slot(slot)
        plan.append((key, _copy_containers({k: v for k, v in config.items() if k != SLOT_KEY}), generation, paths))
        if paths:
            items.append({"field": key, "slot": _copy_containers(slot)})
    return plan, items


def collect_slot_items(
    settings: Dict[str, Any], toolkit_instance_id: str, *, scan: Optional[SlotScan] = None
) -> List[Dict[str, Any]]:
    """The ``resolve_all`` items of one toolkit (``[]`` when it needs no release), validated like `apply_credential_slots`."""
    scan = _checked_scan(settings, scan)
    if not scan.found:
        return []
    if scan.nested:
        raise GovernedCredentialError("UNSUPPORTED_AUTH_MODE")
    _plan, items = _plan_slots(settings, toolkit_instance_id, scan)
    return [{"toolkit_instance_id": toolkit_instance_id, **item} for item in items]


def apply_credential_slots(
    settings: Dict[str, Any],
    toolkit_instance_id: str,
    resolver: Optional[CredentialResolver],
    *,
    scan: Optional[SlotScan] = None,
    released: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], FrozenSet[str]]:
    """Resolve every slot in `settings` and return ``(new_settings, released_values)``.

    `settings` and every dict nested in it are left untouched. The new dict replaces each
    slot-bearing configuration dict with a copy, minus the slot, plus the released paths as
    plain ``str``. `released_values` exists only to `scrub` error texts: never log, store or
    return it. `scan` is the caller's `scan_slots` result for this same settings structure (or
    its shallow copy); its top-level shape is re-checked here and a mismatch is ``STALE_CONTEXT``.
    Without it the settings are scanned here.

    All slots are collected and validated first, then released with one resolver call
    (`resolve_many`, or one call per slot for a resolver that lacks it). `released` is ``{field: envelope}``
    already released for this toolkit (`resolve_run_slots`): the resolver is then not called, and each
    envelope is validated like a fresh reply; it must cover exactly this toolkit's slotted fields.

    Raises `GovernedCredentialError`: no resolver -> ``WORKER_CAPABILITY_UNAVAILABLE``; a slot
    anywhere but directly under a top-level settings key -> ``UNSUPPORTED_AUTH_MODE``; the
    resolver's refusal as is; a reply that does not match the slot (generation, field set) ->
    ``CREDENTIAL_GENERATION_MISMATCH`` / ``DEPENDENCY_UNAVAILABLE``.
    """
    result = dict(settings)
    scan = _checked_scan(settings, scan)
    if not scan.found:
        return result, frozenset()
    # Slots below the top level cannot be resolved here; never leave one behind unresolved.
    if scan.nested:
        raise GovernedCredentialError("UNSUPPORTED_AUTH_MODE")
    if resolver is None and released is None:
        raise GovernedCredentialError("WORKER_CAPABILITY_UNAVAILABLE")

    plan, items = _plan_slots(settings, toolkit_instance_id, scan)
    if released is not None:
        if not isinstance(released, Mapping) or set(released) != {item["field"] for item in items}:
            raise GovernedCredentialError("DEPENDENCY_UNAVAILABLE")
        replies = released
    else:
        replies = _resolve_slots(resolver, toolkit_instance_id, items) if items else {}

    values = set()
    for key, new_config, generation, paths in plan:
        if paths:
            envelope = replies[key]
            if not isinstance(envelope, CredentialEnvelope):
                raise GovernedCredentialError("DEPENDENCY_UNAVAILABLE")
            if envelope.generation != generation:
                raise GovernedCredentialError("CREDENTIAL_GENERATION_MISMATCH")
            if set(envelope.fields) != set(paths):
                raise GovernedCredentialError("DEPENDENCY_UNAVAILABLE")
            for path, secret in envelope.fields.items():
                value = str(secret.get_secret_value())
                _write_path(new_config, path, value)
                if value:
                    values.add(value)
        result[key] = new_config
    return result, frozenset(values)


def _variants(value: str) -> Iterable[str]:
    """The value as it may appear in an error text: raw, repr-escaped, json-escaped."""
    yield value
    yield repr(value)[1:-1]
    yield json.dumps(value)[1:-1]


def scrub(text: str, values: FrozenSet[str]) -> str:
    """Remove released values from `text`.

    pydantic v2 shows ``input_value='<repr>'`` in `ValidationError` texts and, for strings
    longer than ~50 chars, only a head and a tail joined by ``...``; those fragments are
    removed too so a long key or token is not half-leaked.
    """
    if not text or not values:
        return text
    needles = set()
    for value in values:
        if not value:
            continue
        for form in _variants(value):
            needles.add(form)
            if len(form) > 40:
                for n in range(8, 41):
                    needles.add(form[:n])
                    needles.add(form[-n:])
    for needle in sorted(needles, key=len, reverse=True):
        text = text.replace(needle, REDACTED)
    return text
