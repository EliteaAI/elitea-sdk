"""
EL-7011 US-44: EliteAClient gets ten schedule methods on the one
`elitea_core/schedules/prompt_lib/{project_id}` resource (owner change
request 2026-10-21). Each sends one call through `_request` and returns the
response JSON; a status of 400 or more raises ApiDetailsRequestError with
`status_code` and the parsed `body`.
"""
from unittest.mock import MagicMock, patch

import pytest

from elitea_sdk.runtime.clients.client import (
    ApiDetailsRequestError,
    AuthSessionExpiredError,
    EliteAClient,
)

BASE = "https://platform.example.com"
PROJECT_ID = 42
SCHEDULE_ID = 7


@pytest.fixture
def client():
    return EliteAClient(base_url=BASE, project_id=PROJECT_ID, auth_token="tok")


@pytest.fixture
def base(client):
    return f"{BASE}{client.api_v2_path}/elitea_core/schedules/prompt_lib/{PROJECT_ID}"


def make_response(status=200, payload=None, text=""):
    resp = MagicMock(status_code=status, history=[], text=text)
    if payload is None:
        resp.json.side_effect = ValueError("no json")
    else:
        resp.json.return_value = payload
    return resp


@pytest.fixture
def send():
    with patch("requests.Session.request") as mock_request:
        mock_request.return_value = make_response(200, {"ok": True})
        yield mock_request


def test_el7011_r1_1_create_schedule_posts_data(client, send, base):
    data = {"name": "nightly", "cron": "0 3 * * *"}
    assert client.create_schedule(data) == {"ok": True}
    args, kwargs = send.call_args
    assert args == ("post", base)
    assert kwargs["json"] == data


def test_el7011_r1_2_get_schedule(client, send, base):
    assert client.get_schedule(SCHEDULE_ID) == {"ok": True}
    assert send.call_args[0] == ("get", f"{base}/{SCHEDULE_ID}")


def test_el7011_r1_3_update_schedule_puts_data(client, send, base):
    data = {"name": "renamed"}
    assert client.update_schedule(SCHEDULE_ID, data) == {"ok": True}
    args, kwargs = send.call_args
    assert args == ("put", f"{base}/{SCHEDULE_ID}")
    assert kwargs["json"] == data


def test_el7011_r1_4_delete_schedule_returns_none_on_204(client, send, base):
    send.return_value = make_response(204)
    assert client.delete_schedule(SCHEDULE_ID) is None
    assert send.call_args[0] == ("delete", f"{base}/{SCHEDULE_ID}")


def test_el7011_r1_5_list_schedules_sends_paging_and_filters(client, send, base):
    client.list_schedules(limit=10, offset=20, enabled=True, name=None)
    args, kwargs = send.call_args
    assert args == ("get", base)
    assert kwargs["params"] == {"limit": 10, "offset": 20, "enabled": True}


def test_el7011_r1_5_list_schedules_returns_json(client, send):
    assert client.list_schedules() == {"ok": True}


@pytest.mark.parametrize("call", [
    lambda c: c.create_schedule({"name": "x"}),
    lambda c: c.get_schedule(SCHEDULE_ID),
    lambda c: c.update_schedule(SCHEDULE_ID, {}),
    lambda c: c.delete_schedule(SCHEDULE_ID),
    lambda c: c.list_schedules(),
    lambda c: c.pause_schedule(SCHEDULE_ID),
    lambda c: c.resume_schedule(SCHEDULE_ID),
    lambda c: c.run_schedule(SCHEDULE_ID),
    lambda c: c.list_schedule_runs(SCHEDULE_ID),
    lambda c: c.change_schedule_owner(SCHEDULE_ID, 20),
])
def test_el7011_r1_6_every_method_uses_request_headers_and_timeout(client, send, call):
    send.return_value = make_response(200, {"ok": True})
    call(client)
    kwargs = send.call_args[1]
    assert kwargs["headers"] == client.headers
    assert kwargs["timeout"] == (5, 30)


def test_el7011_r1_6_one_url_helper_replaces_the_old_url_attributes(client):
    assert not hasattr(client, "schedule_url")
    assert not hasattr(client, "schedules_url")


@pytest.mark.parametrize("data", [{"enabled": False}, {"name": "x", "enabled": True}])
def test_el7011_r1_3_update_refuses_enabled_before_any_request(client, send, data):
    with pytest.raises(ValueError, match=r"pause_schedule.*resume_schedule"):
        client.update_schedule(SCHEDULE_ID, data)
    send.assert_not_called()


@pytest.mark.parametrize("method, action", [("pause_schedule", "pause"), ("resume_schedule", "resume")])
def test_el7011_r2_1_r2_2_pause_and_resume_post_without_body(client, send, base, method, action):
    assert getattr(client, method)(SCHEDULE_ID) == {"ok": True}
    args, kwargs = send.call_args
    assert args == ("post", f"{base}/{SCHEDULE_ID}/{action}")
    assert "json" not in kwargs and "data" not in kwargs


def test_el7011_r2_3_run_schedule_posts_manual_trigger(client, send, base):
    send.return_value = make_response(202, {"run_id": 99})
    assert client.run_schedule(SCHEDULE_ID) == {"run_id": 99}
    args, kwargs = send.call_args
    assert args == ("post", f"{base}/{SCHEDULE_ID}/run")
    assert kwargs["json"] == {"trigger": "manual"}


def test_el7011_r2_3_run_schedule_allows_test_trigger(client, send):
    send.return_value = make_response(202, {"run_id": 100})
    client.run_schedule(SCHEDULE_ID, trigger="test")
    assert send.call_args[1]["json"] == {"trigger": "test"}


def test_el7011_change_schedule_owner_posts_owner_id(client, send, base):
    assert client.change_schedule_owner(SCHEDULE_ID, 20) == {"ok": True}
    args, kwargs = send.call_args
    assert args == ("post", f"{base}/{SCHEDULE_ID}/owner")
    assert kwargs["json"] == {"owner_id": 20}


def test_el7011_r2_4_list_runs_omits_status_when_not_given(client, send, base):
    client.list_schedule_runs(SCHEDULE_ID, limit=5, offset=10)
    args, kwargs = send.call_args
    assert args == ("get", f"{base}/{SCHEDULE_ID}/runs")
    assert kwargs["params"] == {"limit": 5, "offset": 10}


def test_el7011_r2_4_list_runs_sends_status_when_given(client, send):
    client.list_schedule_runs(SCHEDULE_ID, limit=5, offset=10, status="failed")
    assert send.call_args[1]["params"] == {"limit": 5, "offset": 10, "status": "failed"}


def test_el7011_r3_1_status_400_or_more_raises_with_status_code(client, send):
    send.return_value = make_response(500, {"error": "boom"})
    with pytest.raises(ApiDetailsRequestError) as exc:
        client.get_schedule(SCHEDULE_ID)
    assert exc.value.status_code == 500


def test_el7011_r3_2_json_error_body_is_parsed(client, send):
    send.return_value = make_response(409, {"active_run_id": 5})
    with pytest.raises(ApiDetailsRequestError) as exc:
        client.run_schedule(SCHEDULE_ID)
    assert exc.value.status_code == 409
    assert exc.value.body["active_run_id"] == 5


def test_el7011_r3_3_text_error_body_is_kept(client, send):
    send.return_value = make_response(502, None, text="Bad gateway")
    with pytest.raises(ApiDetailsRequestError) as exc:
        client.list_schedules()
    assert exc.value.body == "Bad gateway"


def test_el7011_r3_4_session_auth_401_raises_session_expired():
    client = EliteAClient(
        base_url=BASE, project_id=PROJECT_ID, auth_token=None,
        auth_session="sess", session_cookie_name="auth_session_id",
    )
    with patch("requests.Session.request") as mock_request:
        mock_request.return_value = make_response(401, {"error": "no"})
        with pytest.raises(AuthSessionExpiredError):
            client.get_schedule(SCHEDULE_ID)


def test_el7011_r4_1_body_fields_pass_through_unchanged(client, send):
    data = {"description": "d", "run_mode": "bogus", "prompt": "hi {{x}}", "variables": {"x": 1}}
    client.create_schedule(data)
    assert send.call_args[1]["json"] == data
    client.update_schedule(SCHEDULE_ID, data)
    assert send.call_args[1]["json"] == data


@pytest.mark.parametrize("call", [
    lambda c: c.create_schedule({"run_mode": "bogus"}),
    lambda c: c.update_schedule(SCHEDULE_ID, {"run_mode": "bogus"}),
])
def test_el7011_r4_2_server_400_surfaces_field_error(client, send, call):
    send.return_value = make_response(400, {"run_mode": "invalid"})
    with pytest.raises(ApiDetailsRequestError) as exc:
        call(client)
    assert exc.value.status_code == 400
    assert exc.value.body == {"run_mode": "invalid"}
