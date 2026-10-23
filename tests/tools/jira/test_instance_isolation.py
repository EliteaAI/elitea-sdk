"""Per-instance isolation for JiraApiWrapper (llm and shared-client headers)."""

from unittest.mock import MagicMock, patch

import pytest

from elitea_sdk.tools.jira.api_wrapper import JiraApiWrapper
from elitea_sdk.tools.client_registry import ClientRegistry

JIRA = "elitea_sdk.tools.jira.api_wrapper"


@pytest.fixture(autouse=True)
def _fresh_registry():
    ClientRegistry().clear()
    yield
    ClientRegistry().clear()


def _make(llm=None, **kwargs):
    params = dict(base_url="https://jira.example.com", token="tok", llm=llm, cloud=False)
    params.update(kwargs)
    with patch(f"{JIRA}.JiraClient", side_effect=lambda **kw: MagicMock(name="JiraClient")):
        return JiraApiWrapper(**params)


def test_llm_is_per_instance_and_not_class_state():
    llm_a, llm_b = object(), object()
    a = _make(llm=llm_a)
    b = _make(llm=llm_b)

    assert a.llm is llm_a
    assert b.llm is llm_b
    assert "llm" not in JiraApiWrapper.__dict__


def test_same_credentials_different_custom_headers_get_distinct_clients():
    a = _make(custom_headers={"X-Tenant": "a"})
    b = _make(custom_headers={"X-Tenant": "b"})

    assert a._client is not b._client
    a._client._update_header.assert_called_once_with("X-Tenant", "a")
    b._client._update_header.assert_called_once_with("X-Tenant", "b")


def test_same_credentials_same_headers_share_client():
    a = _make(custom_headers={"X-Tenant": "a"})
    b = _make(custom_headers={"X-Tenant": "a"})

    assert a._client is b._client
