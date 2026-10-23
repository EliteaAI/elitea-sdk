"""Tests for the fields returned by the WIQL-based search_work_items (#7141).

  * System.WorkItemType is returned by default and when requested explicitly.
  * Without explicit `fields`, the WIQL SELECT columns decide what is returned.
  * Explicit `fields` override the SELECT columns.
"""

import pytest
from pydantic import SecretStr

from elitea_sdk.tools.ado.work_item.ado_wrapper import ADOWorkItemsSearch, AzureDevOpsApiWrapper

DEFAULT_FIELDS = [
    "System.Title",
    "System.State",
    "System.AssignedTo",
    "System.WorkItemType",
    "System.CreatedDate",
    "System.ChangedDate",
]

STORED_FIELDS = {
    "System.Id": 7,
    "System.Title": "Login times out",
    "System.State": "Active",
    "System.AssignedTo": "John Doe <jodoe@contoso.com>",
    "System.WorkItemType": "Bug",
    "System.CreatedDate": "2026-10-01T00:00:00Z",
    "System.ChangedDate": "2026-10-02T00:00:00Z",
    "Microsoft.VSTS.Common.Priority": 2,
}


class FakeRef:
    def __init__(self, id=None, reference_name=None):
        self.id = id
        self.reference_name = reference_name


class FakeQueryResult:
    def __init__(self, work_items, columns=None):
        self.work_items = work_items
        self.columns = columns


class FakeWorkItem:
    def __init__(self, id, fields):
        self.id = id
        self.fields = fields


class FakeClient:
    """Answers get_work_item like ADO does: only the requested fields, or all when none are requested."""

    def __init__(self, columns=None, ids=(7,)):
        self._columns = columns
        self._ids = ids
        self.requested_fields = []

    def query_by_wiql(self, wiql, top=None, team_context=None):
        return FakeQueryResult([FakeRef(id=item_id) for item_id in self._ids], self._columns)

    def get_work_item(self, id, project=None, fields=None, as_of=None, expand=None):
        self.requested_fields.append(fields)
        stored = {**STORED_FIELDS, "System.Id": id}
        if fields is None:
            return FakeWorkItem(id, stored)
        return FakeWorkItem(id, {name: stored[name] for name in fields if name in stored})


def build_wrapper(client):
    wrapper = AzureDevOpsApiWrapper.model_construct(
        organization_url="https://dev.azure.com/org",
        project="MyProject",
        token=SecretStr("token"),
        limit=5,
    )
    wrapper._client = client
    return wrapper


def columns(*names):
    return [FakeRef(reference_name=name) for name in names]


def search(client, **kwargs):
    kwargs.setdefault("query", "SELECT [System.Id] FROM WorkItems WHERE [System.State] = 'Active'")
    return build_wrapper(client).search_work_items(**kwargs)


@pytest.mark.parametrize("query_columns", [None, []])
def test_default_fields_include_work_item_type(query_columns):
    client = FakeClient(columns=query_columns, ids=(7, 8))

    items = search(client)

    assert client.requested_fields == [DEFAULT_FIELDS, DEFAULT_FIELDS]
    assert [item["System.WorkItemType"] for item in items] == ["Bug", "Bug"]
    assert set(items[0]) == {"id", "url", *DEFAULT_FIELDS}


def test_select_with_only_system_id_falls_back_to_defaults():
    client = FakeClient(columns=columns("System.Id"))

    items = search(client)

    assert client.requested_fields == [DEFAULT_FIELDS]
    assert items[0]["System.WorkItemType"] == "Bug"


def test_explicit_fields_return_work_item_type():
    client = FakeClient(columns=columns("System.Id", "System.State"))

    items = search(client, fields=["System.Title", "System.WorkItemType"])

    assert client.requested_fields == [["System.Title", "System.WorkItemType"]]
    assert items == [{
        "id": 7,
        "url": "https://dev.azure.com/org/_workitems/edit/7",
        "System.Title": "Login times out",
        "System.WorkItemType": "Bug",
    }]


def test_select_columns_are_returned_when_fields_are_not_given():
    client = FakeClient(columns=columns("System.Id", "System.Title", "System.WorkItemType"))

    items = search(client, query="SELECT [System.Id], [System.Title], [System.WorkItemType] FROM WorkItems")

    assert client.requested_fields == [["System.Title", "System.WorkItemType"]]
    assert set(items[0]) == {"id", "url", "System.Title", "System.WorkItemType"}
    assert items[0]["System.WorkItemType"] == "Bug"


def test_select_columns_keep_custom_fields():
    client = FakeClient(columns=columns("System.Id", "Microsoft.VSTS.Common.Priority"))

    items = search(client)

    assert items[0]["Microsoft.VSTS.Common.Priority"] == 2


def test_missing_field_is_reported_as_na():
    client = FakeClient()

    items = search(client, fields=["System.WorkItemType", "Custom.Missing"])

    assert items[0]["System.WorkItemType"] == "Bug"
    assert items[0]["Custom.Missing"] == "N/A"


def test_no_results_message_is_unchanged():
    client = FakeClient(ids=())

    assert search(client) == "No work items found."
    assert client.requested_fields == []


def test_schema_descriptions_explain_returned_fields():
    fields_description = ADOWorkItemsSearch.model_fields["fields"].description
    query_description = ADOWorkItemsSearch.model_fields["query"].description

    assert "System.WorkItemType" in fields_description
    assert "SELECT" in query_description
    assert "System.WorkItemType" in AzureDevOpsApiWrapper.search_work_items.__doc__
