"""#6980: metered_call — check before, record after, fail closed, retries reuse the permit.

The real EliteAClient is used with its HTTP session patched, so the test covers the client
methods (URLs, headers, raise_for_status) as well as the context manager around them.
"""
from unittest.mock import MagicMock

import pytest
import requests
from langchain_core.tools import ToolException

from elitea_sdk.runtime.exceptions import budget_exceeded_from
from elitea_sdk.tools.utils import metering

BASE = "https://platform.example.com"
CRED = {"configuration_uuid": "u-shared", "configuration_project_id": 1}
OWN = {"configuration_uuid": "u-own", "configuration_project_id": 3}


def response(status=200, body=None):
    resp = MagicMock(status_code=status)
    resp.json.return_value = body or {}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    metering._exempt_until.clear()  # pylint: disable=W0212
    monkeypatch.setattr(metering, "_sleep", lambda _s: None)


def client_with(*responses):
    from elitea_sdk.runtime.clients.client import EliteAClient
    client = EliteAClient(base_url=BASE, project_id=3, auth_token="tok",
                          api_extra_headers={"X-Elitea-Run-Id": "run-1"})
    client._session = MagicMock()
    client._session.request.side_effect = list(responses)
    return client


def calls(client):
    return [(c.args[0], c.args[1].rsplit("/elitea_core/", 1)[1], c.kwargs.get("json"))
            for c in client._session.request.call_args_list]


ALLOWED = {"allowed": True, "exempt": False, "permit": "P1"}


class TestHappyPath:
    def test_one_block_is_one_check_and_one_record(self):
        client = client_with(response(body=ALLOWED), response(body={"recorded": True}))
        with metering.metered_call(client, "web_search", "tavily", "search", CRED,
                                   plan={"depth": "basic"}, tool_name="web_search") as m:
            m.units = 1
            m.meta = {"credits_source": "reported"}
        made = calls(client)
        assert [path for _m, path, _b in made] == ["tool_spend_check/prompt_lib/3", "tool_spend_record/prompt_lib/3"]
        record = made[1][2]
        assert record["permit"] == "P1" and record["units"] == 1 and record["ok"] is True
        assert record["configuration_uuid"] == "u-shared" and record["tool_name"] == "web_search"
        assert client._session.request.call_args_list[0].kwargs["headers"]["X-Elitea-Run-Id"] == "run-1"

    def test_exception_inside_records_ok_false_and_reraises(self):
        client = client_with(response(body=ALLOWED), response(body={}))
        with pytest.raises(RuntimeError):
            with metering.metered_call(client, "web_search", "tavily", "search", CRED):
                raise RuntimeError("provider 500")
        assert calls(client)[1][2]["ok"] is False and calls(client)[1][2]["units"] == 0

    def test_no_client_is_a_no_op(self):
        with metering.metered_call(None, "web_search", "tavily", "search", CRED) as m:
            m.units = 1


class TestDenials:
    def test_denied_raises_tool_spend_denied_without_calling_provider(self):
        client = client_with(response(body={"allowed": False, "code": metering.CODE_LIMIT,
                                            "message": "Web search monthly limit reached for this project ($1.00)."}))
        provider = MagicMock()
        with pytest.raises(metering.ToolSpendDenied) as info:
            with metering.metered_call(client, "web_search", "tavily", "search", CRED):
                provider()
        provider.assert_not_called()
        assert info.value.code == metering.CODE_LIMIT
        assert isinstance(info.value, ToolException)
        assert len(calls(client)) == 1

    @pytest.mark.parametrize("code", [metering.CODE_LIMIT, metering.CODE_DISABLED, metering.CODE_UNAVAILABLE])
    def test_denial_never_trips_the_run_abort_matcher(self, code):
        error = metering.ToolSpendDenied(code, "Web search is temporarily unavailable.")
        assert budget_exceeded_from(error) is None
        assert "budget_exceeded" not in f"{error} {error.code}"

    @pytest.mark.parametrize("failure", [
        requests.ConnectionError("down"), requests.Timeout("slow"), response(503), response(404),
    ])
    def test_unreachable_check_fails_closed(self, failure):
        client = client_with(failure)
        with pytest.raises(metering.ToolSpendDenied) as info:
            with metering.metered_call(client, "web_search", "tavily", "search", CRED):
                pass
        assert info.value.code == metering.CODE_UNAVAILABLE

    def test_unreachable_check_allows_a_cached_exempt_credential_without_record(self):
        client = client_with(
            response(body={"allowed": True, "exempt": True, "permit": "P0"}), response(body={}),
            requests.ConnectionError("down"),
        )
        with metering.metered_call(client, "web_search", "tavily", "search", OWN) as m:
            m.units = 1
        with metering.metered_call(client, "web_search", "tavily", "search", OWN) as m:
            m.units = 1
        assert m.exempt is True and m.permit is None
        assert len(calls(client)) == 3

    def test_cached_exempt_does_not_cover_a_shared_credential(self):
        client = client_with(response(body={"allowed": True, "exempt": True, "permit": "P0"}), response(body={}),
                             requests.ConnectionError("down"))
        with metering.metered_call(client, "web_search", "tavily", "search", OWN):
            pass
        with pytest.raises(metering.ToolSpendDenied):
            with metering.metered_call(client, "web_search", "tavily", "search", CRED):
                pass

    def test_a_4xx_check_is_a_denial(self):
        with pytest.raises(metering.ToolSpendDenied):
            with metering.metered_call(client_with(response(403)), "web_search", "tavily", "search", CRED):
                pass


class TestRecordRetries:
    def test_retries_reuse_the_same_permit(self):
        client = client_with(response(body=ALLOWED), response(503), requests.Timeout("t"), response(body={}))
        with metering.metered_call(client, "web_search", "tavily", "search", CRED) as m:
            m.units = 2
        records = [body for _m, path, body in calls(client) if path.startswith("tool_spend_record")]
        assert len(records) == 3 and {r["permit"] for r in records} == {"P1"}

    def test_record_failure_never_fails_the_tool(self):
        client = client_with(response(body=ALLOWED), response(500), response(500), response(500))
        with metering.metered_call(client, "web_search", "tavily", "search", CRED) as m:
            result = "provider result"
            m.units = 1
        assert result == "provider result"
        assert len(calls(client)) == 1 + metering.RECORD_ATTEMPTS

    def test_pydantic_like_credential(self):
        class Cfg:
            configuration_uuid = "u-shared"
            configuration_project_id = 1
        client = client_with(response(body=ALLOWED), response(body={}))
        with metering.metered_call(client, "web_search", "tavily", "search", Cfg()):
            pass
        assert calls(client)[0][2]["configuration_uuid"] == "u-shared"
