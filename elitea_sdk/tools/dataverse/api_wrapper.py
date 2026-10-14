import logging
import re
from typing import Any, Optional

from langchain_core.tools import ToolException
from pydantic import Field, SecretStr, create_model

from ..elitea_base import BaseToolApiWrapper
from ..utils.tool_groups import tool_group, with_tool_groups
from ...configurations.dataverse import DataverseError, dataverse_get, dataverse_patch

logger = logging.getLogger(__name__)

MAX_TOP = 50
MAX_UPDATE_FIELDS = 20
_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
MAX_DESCRIBE_COLUMNS = 150
MAX_OPTIONS_PER_COLUMN = 60
# System-owned columns; ownership and state changes need dedicated Dataverse actions
_READONLY_COLUMNS = {"createdon", "createdby", "modifiedon", "modifiedby", "ownerid", "statecode", "statuscode",
                     "versionnumber", "overriddencreatedon", "importsequencenumber", "timezoneruleversionnumber"}
_ENTITY_SET_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_GUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_SELECT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(,[A-Za-z_][A-Za-z0-9_]*)*$")
_BIND_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}@odata\.bind$")
_BIND_VALUE_RE = re.compile(r"^/[A-Za-z_][A-Za-z0-9_]{0,127}\([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\)$")
_OPTION_TYPES = ("Picklist", "MultiSelectPicklist", "Boolean")


def _clean_select(select: Optional[str]) -> Optional[str]:
    select = (select or "").replace(" ", "")
    if not select:
        return None
    if not _SELECT_RE.match(select):
        raise ToolException("select must be a comma-separated list of column logical names")
    return select


def _check_record_id(record_id: str) -> str:
    record_id = (record_id or "").strip().strip("{}")
    if not _GUID_RE.match(record_id):
        raise ToolException("record_id must be a GUID")
    return record_id


def _check_update_fields(fields: dict) -> dict:
    if not isinstance(fields, dict) or not fields:
        raise ToolException("fields must be a non-empty object of column logical names to new values")
    if len(fields) > MAX_UPDATE_FIELDS:
        raise ToolException(f"At most {MAX_UPDATE_FIELDS} columns per update")
    for column, value in fields.items():
        if _BIND_KEY_RE.match(column or ""):
            if not isinstance(value, str) or not _BIND_VALUE_RE.match(value):
                raise ToolException(f"'{column}' must be '/<entity_set>(<guid>)', e.g. '/accounts(00000000-...)'")
            continue
        if not _COLUMN_RE.match(column or "") or column.startswith("_"):
            raise ToolException(f"'{column}' is not a column logical name; set lookups as '<navigation>@odata.bind'")
        if column.lower() in _READONLY_COLUMNS:
            raise ToolException(f"'{column}' cannot be updated with this tool")
        if isinstance(value, (dict, list)):
            raise ToolException(f"'{column}': use a plain value (multi-select choices as '1,2')")
    return fields


def _label(metadata: dict) -> Optional[str]:
    return ((metadata or {}).get("UserLocalizedLabel") or {}).get("Label")


def _options(attribute: dict) -> list:
    if attribute.get("_kind") == "Boolean":
        option_set = attribute.get("OptionSet") or {}
        pairs = [option_set.get("FalseOption"), option_set.get("TrueOption")]
        return [{"value": bool(o.get("Value")), "label": _label(o.get("Label"))} for o in pairs if o]
    option_set = attribute.get("OptionSet") or attribute.get("GlobalOptionSet") or {}
    return [{"value": o.get("Value"), "label": _label(o.get("Label"))} for o in option_set.get("Options") or []]


def _check_entity_set(entity_set: str) -> str:
    entity_set = (entity_set or "").strip()
    if not _ENTITY_SET_RE.match(entity_set):
        raise ToolException("entity_set must be a Dataverse entity set name, e.g. 'opportunities' or 'accounts'")
    return entity_set


class DataverseApiWrapper(BaseToolApiWrapper):
    url: Optional[str] = None
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[SecretStr] = None
    refresh_token: Optional[SecretStr] = None

    def _settings(self) -> dict:
        return {
            "url": self.url, "tenant_id": self.tenant_id, "client_id": self.client_id,
            "client_secret": self.client_secret, "refresh_token": self.refresh_token,
        }

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        try:
            return dataverse_get(self._settings(), path, params)
        except DataverseError as e:
            raise ToolException(str(e)) from None

    @tool_group('read')
    def who_am_i(self) -> dict:
        """Return the Dataverse identity the toolkit acts as (UserId, BusinessUnitId, OrganizationId)."""
        data = self._get("WhoAmI")
        return {k: v for k, v in data.items() if not k.startswith("@odata")}

    @tool_group('read')
    def get_record(self, entity_set: str, record_id: str, select: Optional[str] = None) -> dict:
        """Read one Dataverse record by entity set and GUID. Use select to limit columns, e.g. 'name,statecode'.
        Lookup columns use the _<name>_value form; formatted labels are returned as <column>@OData.Community.Display.V1.FormattedValue."""
        entity_set = _check_entity_set(entity_set)
        record_id = _check_record_id(record_id)
        select = _clean_select(select)
        return self._get(f"{entity_set}({record_id})", {"$select": select} if select else None)

    @tool_group('read')
    def query_records(self, entity_set: str, filter: Optional[str] = None, select: Optional[str] = None,
                      orderby: Optional[str] = None, top: int = 10) -> dict:
        """Query Dataverse records with OData $filter/$select/$orderby, e.g. filter "statecode eq 0", orderby "createdon desc".
        Returns at most top records (max 50) and whether more exist."""
        entity_set = _check_entity_set(entity_set)
        top = max(1, min(int(top or 10), MAX_TOP))
        # $top suppresses @odata.nextLink, so fetch one extra row to know whether more exist
        params = {"$top": str(top + 1)}
        select = _clean_select(select)
        if select:
            params["$select"] = select
        if filter:
            params["$filter"] = filter.strip()
        if orderby:
            params["$orderby"] = orderby.strip()
        data = self._get(entity_set, params)
        records = data.get("value", [])
        return {"records": records[:top], "count": min(len(records), top), "more_available": len(records) > top}

    @tool_group('read')
    def describe_entity(self, entity_set: str, search: Optional[str] = None) -> dict:
        """List the columns of a Dataverse table that can be updated: logical name, display label, type, required level,
        choice values for option columns and the bind target for lookups. Use search (matches name or label) to narrow it down.
        Call this before update_record to get exact column names and option codes."""
        entity_set = _check_entity_set(entity_set)
        found = self._get("EntityDefinitions", {"$select": "LogicalName", "$filter": f"EntitySetName eq '{entity_set}'"})
        if not found.get("value"):
            raise ToolException(f"No table with entity set '{entity_set}'")
        entity = f"EntityDefinitions(LogicalName='{found['value'][0]['LogicalName']}')"
        attributes = self._get(f"{entity}/Attributes", {"$select": "LogicalName,AttributeTypeName,DisplayName,"
                                                         "IsValidForUpdate,RequiredLevel,AttributeOf"}).get("value", [])
        needle = (search or "").strip().lower()
        columns = []
        for a in attributes:
            if not a.get("IsValidForUpdate") or a.get("AttributeOf") or a["LogicalName"] in _READONLY_COLUMNS:
                continue
            label = _label(a.get("DisplayName"))
            if needle and needle not in a["LogicalName"].lower() and needle not in (label or "").lower():
                continue
            kind = ((a.get("AttributeTypeName") or {}).get("Value") or "").removesuffix("Type")
            columns.append({"name": a["LogicalName"], "label": label, "type": kind,
                            "required": (a.get("RequiredLevel") or {}).get("Value") in ("ApplicationRequired", "SystemRequired")})
        total = len(columns)
        columns = sorted(columns, key=lambda c: c["name"])[:MAX_DESCRIBE_COLUMNS]
        by_name = {c["name"]: c for c in columns}
        # Unfiltered listings omit option lists to stay small; a search returns them
        with_options = bool(needle)
        for kind in _OPTION_TYPES if with_options else ():
            if not any(c["type"] == kind for c in columns):
                continue
            expand = "OptionSet($select=TrueOption,FalseOption)" if kind == "Boolean" else \
                "OptionSet($select=Options),GlobalOptionSet($select=Options)"
            meta = self._get(f"{entity}/Attributes/Microsoft.Dynamics.CRM.{kind}AttributeMetadata",
                             {"$select": "LogicalName", "$expand": expand}).get("value", [])
            for a in meta:
                if a["LogicalName"] in by_name:
                    opts = _options({**a, "_kind": kind})
                    by_name[a["LogicalName"]]["options"] = opts[:MAX_OPTIONS_PER_COLUMN]
        if any(c["type"] in ("Lookup", "Customer") for c in columns):
            rels = self._get(f"{entity}/ManyToOneRelationships", {"$select": "ReferencingAttribute,"
                             "ReferencingEntityNavigationPropertyName,ReferencedEntity"}).get("value", [])
            tables = {r["ReferencedEntity"] for r in rels if r["ReferencingAttribute"] in by_name}
            sets = self._entity_sets(tables)
            for r in rels:
                col = by_name.get(r["ReferencingAttribute"])
                if col is not None:
                    target = sets.get(r["ReferencedEntity"], r["ReferencedEntity"])
                    col.setdefault("bind_targets", []).append({
                        "key": f"{r['ReferencingEntityNavigationPropertyName']}@odata.bind",
                        "value_format": f"/{target}(<guid>)", "table": r["ReferencedEntity"]})
        note = None if with_options else "Option values are omitted here; call again with search to get them."
        return {"entity_set": entity_set, "columns": columns, "count": len(columns), "total_matching": total,
                **({"note": note} if note else {}),
                "how_to_set": {"text/memo": "string", "integer/decimal/money": "number", "boolean": "true/false",
                               "datetime": "ISO 8601 string", "picklist": "option value",
                               "multiselectpicklist": "comma-separated option values, e.g. '1,2'",
                               "lookup": "bind_targets key with value_format", "clear": "null"}}

    def _entity_sets(self, tables: set) -> dict:
        if not tables:
            return {}
        flt = " or ".join(f"LogicalName eq '{t}'" for t in sorted(tables) if _COLUMN_RE.match(t))
        rows = self._get("EntityDefinitions", {"$select": "LogicalName,EntitySetName", "$filter": flt}).get("value", [])
        return {r["LogicalName"]: r["EntitySetName"] for r in rows}

    @tool_group('write')
    def update_record(self, entity_set: str, record_id: str, fields: dict) -> dict:
        """Update columns of one existing Dataverse record, acting as the signed-in user (their OneCRM rights apply).
        fields maps column logical names to new values, e.g. {"epam_description": "New text", "epam_prioritycode": 895450001}.
        Use describe_entity first for exact names, option codes and lookup bind keys ({"<nav>@odata.bind": "/accounts(<guid>)"}).
        Ownership and status columns are not supported. Never creates records."""
        entity_set = _check_entity_set(entity_set)
        record_id = _check_record_id(record_id)
        fields = _check_update_fields(fields)
        data = self._patch(f"{entity_set}({record_id})", fields)
        updated = {k: data.get(k) for k in fields if k in data} if data else {}
        return {"message": f"Updated {len(fields)} column(s) on {entity_set}({record_id})", "record": updated}

    def _patch(self, path: str, fields: dict) -> Any:
        try:
            return dataverse_patch(self._settings(), path, fields)
        except DataverseError as e:
            raise ToolException(str(e)) from None

    @with_tool_groups
    def get_available_tools(self):
        return [
            {
                "name": "who_am_i",
                "ref": self.who_am_i,
                "description": self.who_am_i.__doc__,
                "args_schema": create_model("DataverseWhoAmI"),
            },
            {
                "name": "get_record",
                "ref": self.get_record,
                "description": self.get_record.__doc__,
                "args_schema": create_model(
                    "DataverseGetRecord",
                    entity_set=(str, Field(description="Entity set name, e.g. 'opportunities', 'accounts'")),
                    record_id=(str, Field(description="Record GUID")),
                    select=(Optional[str], Field(default=None, description="Comma-separated column logical names")),
                ),
            },
            {
                "name": "query_records",
                "ref": self.query_records,
                "description": self.query_records.__doc__,
                "args_schema": create_model(
                    "DataverseQueryRecords",
                    entity_set=(str, Field(description="Entity set name, e.g. 'opportunities', 'accounts'")),
                    filter=(Optional[str], Field(default=None, description="OData $filter expression")),
                    select=(Optional[str], Field(default=None, description="Comma-separated column logical names")),
                    orderby=(Optional[str], Field(default=None, description="OData $orderby, e.g. 'createdon desc'")),
                    top=(int, Field(default=10, ge=1, le=MAX_TOP, description="Maximum records to return")),
                ),
            },
            {
                "name": "describe_entity",
                "ref": self.describe_entity,
                "description": self.describe_entity.__doc__,
                "args_schema": create_model(
                    "DataverseDescribeEntity",
                    entity_set=(str, Field(description="Entity set name, e.g. 'epam_engagements', 'opportunities'")),
                    search=(Optional[str], Field(default=None, description="Filter by column name or label, e.g. 'priority'")),
                ),
            },
            {
                "name": "update_record",
                "ref": self.update_record,
                "description": self.update_record.__doc__,
                "args_schema": create_model(
                    "DataverseUpdateRecord",
                    entity_set=(str, Field(description="Entity set name, e.g. 'epam_engagements', 'opportunities'")),
                    record_id=(str, Field(description="GUID of the record to update")),
                    fields=(dict, Field(description="Column logical names (or '<nav>@odata.bind' for lookups) mapped to new values")),
                ),
            },
        ]
