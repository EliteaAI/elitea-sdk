"""C-1009 #6756: a SharePoint credential that acts with the signed-in person's Microsoft sign-in.

The platform mints a Graph token on the person's behalf when a governed run starts and releases it into the
configuration's ``access_token``. The toolkit uses that bearer for Graph and nothing else, and refuses (never falls back)
when no token was released. Graph is a fake; every value here is made up for the tests.
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException
from pydantic import ValidationError

from elitea_sdk.configurations.sharepoint import SharepointConfiguration
from elitea_sdk.tools.sharepoint import SharepointToolkit
from elitea_sdk.tools.sharepoint.graph_wrapper import SharepointGraphWrapper

SITE_URL = "https://contoso.sharepoint.com/sites/sandbox"
RELEASED = "graph-token-made-up-for-tests"


def delegated(**over):
    config = {"site_url": SITE_URL, "auth_mode": "entra_delegated", "access_token": RELEASED,
              "configuration_uuid": "cfg-1", "configuration_type": "sharepoint"}
    config.update(over)
    return config


class Graph:
    """``requests.get`` of the Graph wrapper: the site, then its lists."""

    def __init__(self):
        self.calls = []

    def __call__(self, url, headers=None, params=None, timeout=None, **_kwargs):
        self.calls.append((url, dict(headers or {})))
        body = {"id": "contoso.sharepoint.com,site-1,web-1"} if url.endswith("sites/contoso.sharepoint.com:/sites/sandbox") \
            else {"value": [{"id": "list-1", "displayName": "Tasks", "description": "", "list": {"template": "genericList"}}]}
        response = MagicMock(status_code=200, ok=True, url=url, text="")
        response.json.return_value = body
        return response


def build(config, **kwargs):
    return SharepointToolkit.get_toolkit(selected_tools=["get_lists"], toolkit_name="sp", sharepoint_configuration=config,
                                         **kwargs)


def test_the_released_token_is_the_graph_bearer_of_the_read_tools():
    graph = Graph()
    with patch("elitea_sdk.tools.sharepoint.graph_wrapper.requests.get", graph):
        toolkit = build(delegated(), llm=None)
        [tool] = toolkit.get_tools()
        result = tool.api_wrapper.get_lists()
    assert isinstance(tool.api_wrapper._backend, SharepointGraphWrapper)   # pylint: disable=W0212
    assert [item["Title"] for item in result] == ["Tasks"]
    assert [url for url, _ in graph.calls] == [
        "https://graph.microsoft.com/v1.0/sites/contoso.sharepoint.com:/sites/sandbox",
        "https://graph.microsoft.com/v1.0/sites/contoso.sharepoint.com,site-1,web-1/lists"]
    assert {headers["Authorization"] for _, headers in graph.calls} == {f"Bearer {RELEASED}"}


def test_nothing_but_the_released_token_is_used():
    with patch("elitea_sdk.tools.sharepoint.SharepointApiWrapper") as wrapper_cls:
        wrapper_cls.return_value = MagicMock(site_url=SITE_URL, get_available_tools=MagicMock(return_value=[]))
        build(delegated(client_id="stray-id", client_secret="stray-secret", refresh_token="stray-refresh",
                        oauth_discovery_endpoint="https://login.microsoftonline.com/t"),
              tokens={"cfg-1:https://login.microsoftonline.com/t": {"access_token": "browser-token"}})
    payload = wrapper_cls.call_args.kwargs
    assert payload["token"] == RELEASED and payload["scopes"] == ["https://graph.microsoft.com/.default"]
    for key in ("client_id", "client_secret", "refresh_token", "oauth_token_endpoint", "oauth_discovery_endpoint",
                "tokens", "access_token", "auth_mode"):
        assert key not in payload


@pytest.mark.parametrize("token", [None, ""])
def test_without_a_released_token_the_toolkit_refuses_and_never_falls_back(token):
    with patch("elitea_sdk.tools.sharepoint.SharepointApiWrapper") as wrapper_cls:
        with pytest.raises(ToolException) as refused:
            build(delegated(access_token=token, client_id="id", oauth_discovery_endpoint="https://login.microsoftonline.com/t"),
                  tokens={"cfg-1:https://login.microsoftonline.com/t": {"access_token": "browser-token"}})
    assert "DENIED_DELEGATION_UNSUPPORTED_CALLER" in str(refused.value)
    wrapper_cls.assert_not_called()


def test_the_other_modes_are_unchanged():
    with patch("elitea_sdk.tools.sharepoint.SharepointApiWrapper") as wrapper_cls:
        wrapper_cls.return_value = MagicMock(site_url=SITE_URL, get_available_tools=MagicMock(return_value=[]))
        build({"site_url": SITE_URL, "client_id": "app", "client_secret": "app-secret"})
    payload = wrapper_cls.call_args.kwargs
    assert payload["client_id"] == "app" and "token" not in payload


def test_the_configuration_of_a_delegated_credential_holds_no_secret():
    assert SharepointConfiguration(site_url=SITE_URL, auth_mode="entra_delegated").client_secret is None
    with pytest.raises(ValidationError):
        SharepointConfiguration(site_url=SITE_URL, auth_mode="entra_delegated", client_secret="a-secret")
    with pytest.raises(ValidationError):
        SharepointConfiguration(site_url=SITE_URL, auth_mode="entra_delegated", refresh_token="a-refresh")
    with pytest.raises(ValidationError):
        SharepointConfiguration(site_url=SITE_URL, auth_mode="something_else", client_id="a", client_secret="b")
    with pytest.raises(ValidationError):   # every other mode still needs its app credentials
        SharepointConfiguration(site_url=SITE_URL)
    assert SharepointConfiguration(site_url=SITE_URL, client_id="a", client_secret="b").auth_mode is None


def test_access_token_and_auth_mode_are_hidden_platform_fields():
    properties = SharepointConfiguration.model_json_schema()["properties"]
    assert properties["access_token"].get("hidden") is True and properties["auth_mode"].get("hidden") is True
    formats = {item.get("format") for item in properties["access_token"].get("anyOf", [properties["access_token"]])}
    assert "password" in formats   # a secret path: the seal strips it from payloads


def test_check_connection_of_a_delegated_credential_needs_a_released_token():
    assert SharepointConfiguration.check_connection(delegated()) is None
    assert "DENIED_DELEGATION_UNSUPPORTED_CALLER" in SharepointConfiguration.check_connection(delegated(access_token=None))
