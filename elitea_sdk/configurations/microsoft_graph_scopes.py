"""Microsoft Graph delegated permission lists for the Outlook and Teams credentials.

Each credential offers a fixed list of permissions, rendered as checkboxes. ELITEA
passes the ticked ones to Microsoft sign-in as-is: Microsoft decides what the token
holds, and a token always carries every permission the user already consented to
for the Entra app, so unticking a box never removes a granted permission.
"""
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

GRAPH_RESOURCE_PREFIX = "https://graph.microsoft.com/"

_SPLIT_RE = re.compile(r"[,\s]+")

SCOPES_NOTE = (
    "Microsoft Graph permissions requested at sign-in. ELITEA passes them to Microsoft as-is.\n\n"
    "- The token always holds every permission you already approved for this Entra app, "
    "so unticking a box does not remove access granted before.\n"
    "- To remove a granted permission, revoke it on the app's Manage your application page "
    "at https://myapps.microsoft.com (or ask your Entra admin), then sign in again.\n"
    "- To strictly limit what a toolkit can do, use a separate Entra app that has only "
    "the permissions you need.\n"
    "- offline_access is always added so the token can be refreshed."
)


def scope_options(descriptions: Dict[str, str], admin_consent: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """Checkbox options for the UI: one {value, label, description} per permission.

    Permissions in ``admin_consent`` need Entra admin consent: their option carries
    ``admin_consent: true``, which the UIs and the sign-in prompt read from the schema.
    """
    consent = set(admin_consent)
    unknown = consent - set(descriptions)
    if unknown:
        raise ValueError(f"Admin consent set for permissions not offered: {', '.join(sorted(unknown))}")
    options: List[Dict[str, Any]] = []
    for name, text in descriptions.items():
        option: Dict[str, Any] = {"value": name, "label": name, "description": text}
        if name in consent:
            option["admin_consent"] = True
        options.append(option)
    return options


def scopes_field_extra(descriptions: Dict[str, str], admin_consent: Iterable[str] = ()) -> Dict[str, Any]:
    """json_schema_extra for a scopes field rendered as a checkbox list."""
    return {"ui_component": "checkbox_list", "checkbox_options": scope_options(descriptions, admin_consent)}


def admin_consent_scopes(model: Any, scopes: Optional[Iterable[str]]) -> List[str]:
    """The given scopes that the credential model's ``scopes`` options mark as needing admin consent."""
    field = getattr(model, "model_fields", {}).get("scopes")
    extra = getattr(field, "json_schema_extra", None)
    options = extra.get("checkbox_options", []) if isinstance(extra, dict) else []
    flagged = {option["value"] for option in options if option.get("admin_consent")}
    return [scope for scope in (scopes or []) if scope in flagged]


def normalize_scopes(
    value: Any,
    allowed: List[str],
    default: List[str],
    source: str = "Microsoft Graph",
) -> List[str]:
    """Keep only the allowed permissions, in canonical casing and order.

    Accepts a list or a comma / space separated string, matches case-insensitively
    and strips the https://graph.microsoft.com/ prefix. Unknown values are dropped
    (logged). Returns a copy of ``default`` when nothing allowed is left.
    """
    if value is None:
        items: List[Any] = []
    elif isinstance(value, str):
        items = _SPLIT_RE.split(value)
    elif isinstance(value, (list, tuple, set)):
        items = [part for item in value for part in _SPLIT_RE.split(str(item))]
    else:
        items = [value]

    canonical = {name.lower(): name for name in allowed}
    picked = set()
    dropped: List[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key.startswith(GRAPH_RESOURCE_PREFIX):
            key = key[len(GRAPH_RESOURCE_PREFIX):]
        if key in canonical:
            picked.add(canonical[key])
        else:
            dropped.append(text)
    if dropped:
        log.warning("%s: ignoring permissions not offered by this credential: %s", source, ", ".join(dropped))
    result = [name for name in allowed if name in picked]
    return result or list(default)


def missing_permission_hint(needed: Optional[str], admin_consent: bool = False) -> str:
    """403 hint naming the permission a tool needs; ``admin_consent`` when only an Entra admin can grant it."""
    if not needed:
        return (" The signed-in account lacks the Microsoft Graph permission this tool needs: "
                "tick it in the credential scopes and sign in again (an Entra admin may need to grant it).")
    if admin_consent:
        return (f" This tool needs the Microsoft Graph permission {needed}, which requires Entra admin "
                "consent: ask your Entra admin to grant it, tick it in the credential scopes and sign in again.")
    return (f" This tool needs the Microsoft Graph permission {needed}: tick it in the credential "
            "scopes and sign in again (an Entra admin may need to grant it).")
