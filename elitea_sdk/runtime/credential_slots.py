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
embedding worker (the indexer) and is called as
``resolver(toolkit_instance_id=<str>, field=<settings key>, slot=<slot dict>)``.
It must return a `CredentialEnvelope`, or raise `GovernedCredentialError` with the release
refusal code.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Protocol, Tuple, Union

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
    """Injected by the embedding worker. Returns the envelope or raises `GovernedCredentialError`."""

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


def has_slots(settings: Any) -> bool:
    """True when `settings` holds a credential slot anywhere (bounded depth). A plain dict scan."""
    return isinstance(settings, dict) and _contains_slot(settings, _MAX_SCAN_DEPTH)


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


def apply_credential_slots(
    settings: Dict[str, Any],
    toolkit_instance_id: str,
    resolver: Optional[CredentialResolver],
) -> Tuple[Dict[str, Any], FrozenSet[str]]:
    """Resolve every slot in `settings` and return ``(new_settings, released_values)``.

    `settings` and every dict nested in it are left untouched. The new dict replaces each
    slot-bearing configuration dict with a copy, minus the slot, plus the released paths as
    plain ``str``. `released_values` exists only to `scrub` error texts: never log, store or
    return it.

    Raises `GovernedCredentialError`: no resolver -> ``WORKER_CAPABILITY_UNAVAILABLE``; a slot
    anywhere but directly under a top-level settings key -> ``UNSUPPORTED_AUTH_MODE``; the
    resolver's refusal as is; a reply that does not match the slot (generation, field set) ->
    ``CREDENTIAL_GENERATION_MISMATCH`` / ``DEPENDENCY_UNAVAILABLE``.
    """
    result = dict(settings)
    slotted = [k for k, v in settings.items() if isinstance(v, dict) and SLOT_KEY in v]
    if not slotted:
        if has_slots(settings):
            raise GovernedCredentialError("UNSUPPORTED_AUTH_MODE")
        return result, frozenset()
    # Slots below the top level cannot be resolved here; never leave one behind unresolved.
    if any(has_slots(v) for k, v in settings.items() if k not in slotted) or any(
        has_slots({k2: v2 for k2, v2 in settings[k].items() if k2 != SLOT_KEY}) for k in slotted
    ):
        raise GovernedCredentialError("UNSUPPORTED_AUTH_MODE")
    if resolver is None:
        raise GovernedCredentialError("WORKER_CAPABILITY_UNAVAILABLE")
    if not isinstance(toolkit_instance_id, str) or not toolkit_instance_id:
        raise GovernedCredentialError("STALE_CONTEXT")

    released = set()
    for key in slotted:
        config = settings[key]
        slot = config[SLOT_KEY]
        generation, paths = _validate_slot(slot)
        new_config = _copy_containers({k: v for k, v in config.items() if k != SLOT_KEY})
        if paths:
            try:
                envelope = resolver(
                    toolkit_instance_id=toolkit_instance_id,
                    field=key,
                    slot=_copy_containers(slot),
                )
            except GovernedCredentialError:
                raise
            except Exception:
                # The original text may carry ids or values; the typed code is the contract.
                raise GovernedCredentialError("DEPENDENCY_UNAVAILABLE") from None
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
                    released.add(value)
        result[key] = new_config
    return result, frozenset(released)


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
