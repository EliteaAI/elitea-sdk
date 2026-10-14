import logging
import re
from typing import Any, Optional

from langchain_core.tools import ToolException
from pydantic import Field, SecretStr, create_model

from ..elitea_base import BaseToolApiWrapper
from ..utils.tool_groups import tool_group, with_tool_groups
from ...configurations.dataverse import DataverseError, dataverse_get

logger = logging.getLogger(__name__)

MAX_TOP = 50
_ENTITY_SET_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_GUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_SELECT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(,[A-Za-z_][A-Za-z0-9_]*)*$")


def _clean_select(select: Optional[str]) -> Optional[str]:
    select = (select or "").replace(" ", "")
    if not select:
        return None
    if not _SELECT_RE.match(select):
        raise ToolException("select must be a comma-separated list of column logical names")
    return select


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

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        settings = {
            "url": self.url, "tenant_id": self.tenant_id, "client_id": self.client_id,
            "client_secret": self.client_secret, "refresh_token": self.refresh_token,
        }
        try:
            return dataverse_get(settings, path, params)
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
        record_id = (record_id or "").strip().strip("{}")
        if not _GUID_RE.match(record_id):
            raise ToolException("record_id must be a GUID")
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
        ]
