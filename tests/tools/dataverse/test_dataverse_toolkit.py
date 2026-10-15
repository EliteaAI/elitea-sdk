"""Dataverse (OneCRM) toolkit: refresh-token auth, token cache, URL/argument guards and error mapping.

All HTTP is faked: `requests.post` stands in for the Entra token endpoint (via the SDK's
refresh_oauth_token) and `requests.get` for the Dataverse Web API.
"""
import pytest
from langchain_core.tools import ToolException

from elitea_sdk.configurations import dataverse as dv
from elitea_sdk.configurations.dataverse import DataverseConfiguration, DataverseError, normalize_org_url
from elitea_sdk.runtime.utils.mcp_oauth import McpAuthorizationRequired
from elitea_sdk.tools.dataverse import DataverseToolkit

ORG = "https://onecrmepam.crm4.dynamics.com"
RECORD_ID = "4bc255b0-be02-467c-9f6e-bdcff31242db"
SETTINGS = {
    "url": ORG, "tenant_id": "b41b72d0-4e9f-4c26-8a69-f949f367c91d",
    "client_id": "client-1", "client_secret": "secret-1", "refresh_token": "refresh-1",
}


class FakeResponse:
    def __init__(self, status, body=None, text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


@pytest.fixture
def http(monkeypatch):
    """Records token and API calls; tests set .token_response / .api_responses."""
    class State:
        token_calls = []
        api_calls = []
        token_response = FakeResponse(200, {"access_token": "at-1", "expires_in": 3600, "refresh_token": "r2"})
        api_responses = []

    def fake_post(url, data=None, headers=None, timeout=None):
        State.token_calls.append({"url": url, "data": data})
        return State.token_response

    def fake_request(method, url, headers=None, json=None, timeout=None):
        State.api_calls.append({"method": method, "url": url, "headers": headers, "json": json})
        return State.api_responses.pop(0) if State.api_responses else FakeResponse(200, {"UserId": "u-1"})

    monkeypatch.setattr("elitea_sdk.runtime.utils.mcp_oauth.requests.post", fake_post)
    monkeypatch.setattr(dv.requests, "request", fake_request)
    dv._token_cache.clear()
    yield State
    dv._token_cache.clear()


def tools(settings=None):
    kit = DataverseToolkit.get_toolkit(dataverse_configuration=dict(settings or SETTINGS), toolkit_name="crm")
    return {t.name: t for t in kit.get_tools()}


def test_check_connection_refreshes_with_dataverse_scope(http):
    assert DataverseConfiguration.check_connection(SETTINGS) is None
    call = http.token_calls[0]
    assert call["url"] == f"https://login.microsoftonline.com/{SETTINGS['tenant_id']}/oauth2/v2.0/token"
    assert call["data"]["grant_type"] == "refresh_token"
    assert call["data"]["scope"] == f"{ORG}/user_impersonation offline_access"
    assert call["data"]["client_secret"] == "secret-1"
    assert http.api_calls[0]["url"] == f"{ORG}/api/data/v9.2/WhoAmI"
    assert http.api_calls[0]["headers"]["Authorization"] == "Bearer at-1"


def test_access_token_is_cached_between_calls(http):
    t = tools()
    t["who_am_i"].run({})
    t["who_am_i"].run({})
    assert len(http.token_calls) == 1
    assert len(http.api_calls) == 2


def test_expired_cached_token_is_refreshed(http, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(dv.time, "time", lambda: clock[0])
    dv.get_access_token(SETTINGS)
    clock[0] += 3600 - dv.TOKEN_SKEW_SECONDS + 1
    dv.get_access_token(SETTINGS)
    assert len(http.token_calls) == 2


def test_wrong_client_secret_is_a_plain_error(http):
    http.token_response = FakeResponse(401, {"error": "invalid_client",
                                             "error_description": "AADSTS7000215: Invalid client secret provided"})
    message = DataverseConfiguration.check_connection(SETTINGS)
    assert "client secret is wrong or expired" in message
    assert "refresh-1" not in message and "secret-1" not in message


@pytest.mark.parametrize("error, expected", [
    ("AADSTS65001: The user or administrator has not consented", "no consent for Dataverse"),
    ("AADSTS700082: The refresh token has expired due to inactivity", "expired or was revoked"),
])
def test_expired_or_unconsented_login_asks_ui_to_log_in_again(http, error, expected):
    # An unusable refresh token must reopen the Login popup rather than show a dead-end error
    http.token_response = FakeResponse(400, {"error": "invalid_grant", "error_description": error})
    with pytest.raises(McpAuthorizationRequired) as exc:
        DataverseConfiguration.check_connection({**SETTINGS, "configuration_uuid": "uuid-1"})
    assert expected in str(exc.value)
    assert "refresh-1" not in str(exc.value.to_dict())


def test_missing_refresh_token_requests_login_with_entra_endpoints(http):
    with pytest.raises(McpAuthorizationRequired) as exc:
        DataverseConfiguration.check_connection({**SETTINGS, "refresh_token": None, "configuration_uuid": "uuid-1"})
    meta = exc.value.resource_metadata
    authority = f"https://login.microsoftonline.com/{SETTINGS['tenant_id']}"
    assert meta["authorization_servers"] == [authority]
    assert meta["oauth_authorization_server"]["token_endpoint"] == f"{authority}/oauth2/v2.0/token"
    assert meta["oauth_authorization_server"]["authorization_endpoint"] == f"{authority}/oauth2/v2.0/authorize"
    assert meta["scopes_supported"] == [f"{ORG}/user_impersonation", "offline_access"]
    assert meta["configuration_uuid"] == "uuid-1"
    assert meta["provided_settings"]["mcp_client_id"] == "client-1"
    assert "mcp_client_secret" not in meta["provided_settings"]
    assert http.token_calls == []


def test_tool_call_without_login_explains_how_to_log_in(http):
    with pytest.raises(ToolException, match="Login button"):
        tools({**SETTINGS, "refresh_token": None})["who_am_i"].run({})


@pytest.mark.parametrize("url", [
    "http://onecrmepam.crm4.dynamics.com",
    "https://evil.example.com",
    "https://onecrmepam.crm4.dynamics.com.evil.com",
    "https://onecrmepam.crm4.dynamics.com/api",
    "https://user@onecrmepam.crm4.dynamics.com",
    "https://onecrmepam.crm4.dynamics.com:8443",
])
def test_non_dataverse_urls_are_rejected_before_any_token_request(http, url):
    assert "must look like" in DataverseConfiguration.check_connection({**SETTINGS, "url": url})
    assert http.token_calls == []


def test_url_is_normalized():
    assert normalize_org_url(" https://OneCRMepam.crm4.dynamics.com/ ") == ORG


def test_missing_fields_are_reported(http):
    assert "all required" in DataverseConfiguration.check_connection({**SETTINGS, "client_secret": ""})
    assert http.token_calls == []


def test_tenant_cannot_inject_a_path(http):
    with pytest.raises(DataverseError, match="tenant_id"):
        dv.get_access_token({**SETTINGS, "tenant_id": "evil/../x"})


def test_get_record_builds_select_url(http):
    http.api_responses = [FakeResponse(200, {"name": "Deal", "statecode": 0})]
    result = tools()["get_record"].run({"entity_set": "opportunities", "record_id": "{%s}" % RECORD_ID,
                                         "select": "name, statecode"})
    assert result == {"name": "Deal", "statecode": 0}
    assert http.api_calls[0]["url"] == f"{ORG}/api/data/v9.2/opportunities({RECORD_ID})?%24select=name%2Cstatecode"


@pytest.mark.parametrize("args, message", [
    ({"entity_set": "opportunities", "record_id": "x"}, "GUID"),
    ({"entity_set": "opportunities?$x=1", "record_id": RECORD_ID}, "entity set"),
    ({"entity_set": "opportunities", "record_id": RECORD_ID, "select": "name;drop"}, "select"),
])
def test_get_record_rejects_bad_arguments_without_calling_dataverse(http, args, message):
    with pytest.raises(ToolException, match=message):
        tools()["get_record"].run(args)
    assert http.api_calls == []


def test_query_records_caps_top_and_reports_more(http):
    http.api_responses = [FakeResponse(200, {"value": [{"name": str(i)} for i in range(4)]})]
    result = tools()["query_records"].run({"entity_set": "opportunities", "filter": "statecode eq 0",
                                            "orderby": "createdon desc", "top": 3})
    assert result == {"records": [{"name": "0"}, {"name": "1"}, {"name": "2"}], "count": 3, "more_available": True}
    url = http.api_calls[0]["url"]
    assert "%24top=4" in url and "%24filter=statecode%20eq%200" in url and "%24orderby=createdon%20desc" in url


def test_query_records_last_page(http):
    http.api_responses = [FakeResponse(200, {"value": [{"name": "only"}]})]
    result = tools()["query_records"].run({"entity_set": "accounts", "top": 5})
    assert result["count"] == 1 and result["more_available"] is False


@pytest.mark.parametrize("status, message", [
    (401, "rejected the access token"),
    (403, "no permission"),
    (404, "not found"),
    (429, "rate limit"),
])
def test_dataverse_http_errors_map_to_tool_errors(http, status, message):
    http.api_responses = [FakeResponse(status, {"error": {"message": "x"}})]
    with pytest.raises(ToolException, match=message):
        tools()["who_am_i"].run({})


def test_dataverse_error_body_is_surfaced(http):
    http.api_responses = [FakeResponse(400, {"error": {"message": "Could not find a property named 'nme'"}})]
    with pytest.raises(ToolException, match="property named 'nme'"):
        tools()["query_records"].run({"entity_set": "opportunities", "select": "nme"})


def test_selected_tools_filter_and_metadata():
    kit = DataverseToolkit.get_toolkit(dataverse_configuration=SETTINGS, toolkit_name="crm",
                                       selected_tools=["get_record"])
    [tool] = kit.get_tools()
    assert tool.name == "get_record"
    assert tool.metadata["toolkit_type"] == "dataverse"
    assert tool.description.startswith("Toolkit: crm\n")


def test_config_schema_exposes_configuration_and_tools():
    schema = DataverseToolkit.toolkit_config_schema().model_json_schema()
    assert set(schema["properties"]) == {"dataverse_configuration", "selected_tools"}
    fields = DataverseConfiguration.model_json_schema()["properties"]
    assert fields["client_secret"]["format"] == "password"
    # Stored in the vault (password format) but never rendered: only the Login flow sets it
    assert fields["refresh_token"]["hidden"] is True
    assert {"format": "password", "type": "string", "writeOnly": True} in fields["refresh_token"]["anyOf"]
    assert "refresh_token" not in DataverseConfiguration.model_json_schema().get("required", [])


def test_update_record_patches_existing_record_only(http):
    # If-Match: * makes Dataverse refuse the PATCH instead of upserting a brand-new record.
    http.api_responses = [FakeResponse(200, {"epam_description": "New text", "epam_name": "Deal"})]
    result = tools()["update_record"].run({"entity_set": "epam_engagements", "record_id": RECORD_ID,
                                            "fields": {"epam_description": "New text"}})
    call = http.api_calls[0]
    assert call["method"] == "PATCH"
    assert call["url"] == f"{ORG}/api/data/v9.2/epam_engagements({RECORD_ID})"
    assert call["json"] == {"epam_description": "New text"}
    assert call["headers"]["If-Match"] == "*"
    assert call["headers"]["Prefer"] == "return=representation"
    assert result == {"message": f"Updated 1 column(s) on epam_engagements({RECORD_ID})",
                      "record": {"epam_description": "New text"}}


def test_update_record_accepts_no_content_reply(http):
    http.api_responses = [FakeResponse(204)]
    result = tools()["update_record"].run({"entity_set": "accounts", "record_id": RECORD_ID,
                                            "fields": {"description": None}})
    assert result["record"] == {}


def test_update_record_on_missing_record_never_creates_one(http):
    http.api_responses = [FakeResponse(412, {"error": {"message": "precondition failed"}})]
    with pytest.raises(ToolException, match="never create"):
        tools()["update_record"].run({"entity_set": "accounts", "record_id": RECORD_ID, "fields": {"name": "x"}})


@pytest.mark.parametrize("fields, message", [
    ({}, "non-empty"),
    ({"_parentaccountid_value": "x"}, "not a column logical name"),
    ({"parentaccountid@odata.bind": "/accounts(1)"}, "must be '/<entity_set>"),
    ({"parentaccountid@odata.bind": "https://evil/accounts(%s)" % RECORD_ID}, "must be '/<entity_set>"),
    ({"name; drop": "x"}, "not a column logical name"),
    ({"statecode": 1}, "cannot be updated"),
    ({"OwnerId": "x"}, "cannot be updated"),
    ({"description": {"nested": 1}}, "plain value"),
    ({f"c{i}": i for i in range(21)}, "At most 20"),
])
def test_update_record_rejects_unsupported_fields_without_calling_dataverse(http, fields, message):
    with pytest.raises(ToolException, match=message):
        tools()["update_record"].run({"entity_set": "accounts", "record_id": RECORD_ID, "fields": fields})
    assert http.api_calls == []


def test_update_record_is_in_write_group():
    groups = DataverseToolkit.toolkit_config_schema().model_json_schema()["properties"]["selected_tools"]["tool_groups"]
    assert groups["update_record"] == "write"
    assert groups["get_record"] == "read"
    assert groups["describe_entity"] == "read"


def test_update_record_sets_lookup_and_choice_columns(http):
    # Lookups go through @odata.bind with a relative /<entity set>(<guid>) reference; choices by option value.
    http.api_responses = [FakeResponse(204)]
    tools()["update_record"].run({"entity_set": "epam_engagements", "record_id": RECORD_ID, "fields": {
        "epam_accountid@odata.bind": f"/accounts({RECORD_ID})",
        "epam_prioritycode": 895450001,
        "epam_servicescode": "895450000,895450002",
    }})
    assert http.api_calls[0]["json"] == {
        "epam_accountid@odata.bind": f"/accounts({RECORD_ID})",
        "epam_prioritycode": 895450001,
        "epam_servicescode": "895450000,895450002",
    }


def _metadata_responses():
    attributes = {"value": [
        {"LogicalName": "epam_description", "AttributeTypeName": {"Value": "MemoType"}, "IsValidForUpdate": True,
         "DisplayName": {"UserLocalizedLabel": {"Label": "Description"}}, "RequiredLevel": {"Value": "None"}},
        {"LogicalName": "epam_prioritycode", "AttributeTypeName": {"Value": "PicklistType"}, "IsValidForUpdate": True,
         "DisplayName": {"UserLocalizedLabel": {"Label": "Priority"}}, "RequiredLevel": {"Value": "ApplicationRequired"}},
        {"LogicalName": "epam_accountid", "AttributeTypeName": {"Value": "LookupType"}, "IsValidForUpdate": True,
         "DisplayName": {"UserLocalizedLabel": {"Label": "Account"}}, "RequiredLevel": {"Value": "None"}},
        {"LogicalName": "epam_accountidname", "AttributeTypeName": {"Value": "StringType"}, "IsValidForUpdate": True,
         "AttributeOf": "epam_accountid"},
        {"LogicalName": "createdon", "AttributeTypeName": {"Value": "DateTimeType"}, "IsValidForUpdate": True},
        {"LogicalName": "epam_number", "AttributeTypeName": {"Value": "StringType"}, "IsValidForUpdate": False},
    ]}
    picklists = {"value": [{"LogicalName": "epam_prioritycode", "OptionSet": {"Options": [
        {"Value": 895450001, "Label": {"UserLocalizedLabel": {"Label": "P1 - High"}}}]}}]}
    relations = {"value": [{"ReferencingAttribute": "epam_accountid",
                            "ReferencingEntityNavigationPropertyName": "epam_accountid", "ReferencedEntity": "account"}]}
    return [
        FakeResponse(200, {"value": [{"LogicalName": "epam_engagement"}]}),
        FakeResponse(200, attributes),
        FakeResponse(200, picklists),
        FakeResponse(200, relations),
        FakeResponse(200, {"value": [{"LogicalName": "account", "EntitySetName": "accounts"}]}),
    ]


def test_describe_entity_lists_updatable_columns_with_options_and_bind_keys(http):
    http.api_responses = _metadata_responses()
    result = tools()["describe_entity"].run({"entity_set": "epam_engagements", "search": "e"})
    cols = {c["name"]: c for c in result["columns"]}
    # Read-only, system and helper (AttributeOf) columns are hidden
    assert set(cols) == {"epam_description", "epam_prioritycode", "epam_accountid"}
    assert cols["epam_prioritycode"]["required"] is True
    assert cols["epam_prioritycode"]["options"] == [{"value": 895450001, "label": "P1 - High"}]
    assert cols["epam_accountid"]["bind_targets"] == [
        {"key": "epam_accountid@odata.bind", "value_format": "/accounts(<guid>)", "table": "account"}]
    assert "note" not in result
    assert "EntitySetName eq 'epam_engagements'" in http.api_calls[0]["url"].replace("%20", " ").replace("%27", "'")


def test_describe_entity_without_search_skips_option_lists(http):
    responses = _metadata_responses()
    http.api_responses = [responses[0], responses[1], responses[3], responses[4]]
    result = tools()["describe_entity"].run({"entity_set": "epam_engagements"})
    assert "options" not in {c["name"]: c for c in result["columns"]}["epam_prioritycode"]
    assert "search" in result["note"]
    assert not any("PicklistAttributeMetadata" in c["url"] for c in http.api_calls)


def test_describe_entity_unknown_table(http):
    http.api_responses = [FakeResponse(200, {"value": []})]
    with pytest.raises(ToolException, match="No table"):
        tools()["describe_entity"].run({"entity_set": "nope"})
