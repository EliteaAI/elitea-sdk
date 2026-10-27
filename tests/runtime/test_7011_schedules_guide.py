"""
The schedules guide `docs/guides/schedules.md`.

The test runs every Python block of the guide against EliteAClient with
`requests.Session.request` mocked, so the examples cannot drift from the
client. It also checks the sections and the rules the text must state.
"""
import re
import runpy
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from elitea_sdk.runtime.clients.client import ApiDetailsRequestError, EliteAClient

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "guides" / "schedules.md"
SECTIONS = ("Setup", "Methods", "Who sees what", "Errors", "End-to-end example")

BASE = "https://platform.example.com"
API_KEY = "secret-key-123"
PROJECT_ID = 42
SCHEDULE_ID = 7
API = f"{BASE}/api/v2/elitea_core"
APPS = f"{API}/applications/prompt_lib/{PROJECT_ID}"
APP = f"{API}/application/prompt_lib/{PROJECT_ID}/"
SCHEDULES = f"{API}/schedules/prompt_lib/{PROJECT_ID}"

# Method name -> HTTP verb and path after the schedules base (owner change request 2026-10-21)
ENDPOINTS = {
    "list_schedules": ("GET", ""),
    "create_schedule": ("POST", ""),
    "get_schedule": ("GET", "/{schedule_id}"),
    "update_schedule": ("PUT", "/{schedule_id}"),
    "delete_schedule": ("DELETE", "/{schedule_id}"),
    "pause_schedule": ("POST", "/{schedule_id}/pause"),
    "resume_schedule": ("POST", "/{schedule_id}/resume"),
    "run_schedule": ("POST", "/{schedule_id}/run"),
    "list_schedule_runs": ("GET", "/{schedule_id}/runs"),
    "change_schedule_owner": ("POST", "/{schedule_id}/owner"),
    "check_schedule": ("POST", "/check"),
    "get_schedule_run": ("GET", "/{schedule_id}/runs/{run_id}"),
    "get_schedule_settings": ("GET", "/settings"),
    "update_schedule_settings": ("PUT", "/settings"),
}
NO_ID_METHODS = ("list_schedules", "create_schedule", "check_schedule",
                 "get_schedule_settings", "update_schedule_settings")
RISK_409 = {"code": "risk_confirmation_required", "fingerprint": "abc123", "blocking": []}
ROUTES = {(verb.lower(), SCHEDULES + path.format(schedule_id=SCHEDULE_ID, run_id=55)): name
          for name, (verb, path) in ENDPOINTS.items()}
ROUTES[("get", APPS)] = "get_list_of_apps"

AGENT = {"id": 5, "name": "Digest agent", "version_id": 91, "agent_type": "agent"}
PIPELINE = {"id": 12, "name": "Nightly ETL", "version_id": 92, "agent_type": "pipeline"}
SCHEDULE = {"id": SCHEDULE_ID, "name": "SDK example", "enabled": False, "cron": "0 9 * * 1",
            "owner": {"id": 3, "name": "Ann"}, "next_run": None}
RUN = {"id": 55, "trigger": "manual", "status": "running", "started_at": "2026-10-21T09:00:00+00:00"}

FENCE = re.compile(r"^```(\S*)\n(.*?)^```", re.MULTILINE | re.DOTALL)


def response(url, status, payload=None):
    resp = MagicMock(status_code=status, history=[], ok=status < 400, url=url, text="")
    if payload is None:
        resp.json.side_effect = ValueError("no json")
    else:
        resp.json.return_value = payload
    return resp


class FakeApi:
    """Answers the client calls by method name and records each call in order."""

    def __init__(self, apps=(AGENT, PIPELINE)):
        self.apps = {app["id"]: app for app in apps}
        self.calls = []
        self.bodies = {}
        self.answers = {
            "get_list_of_apps": [(200, {"total": len(apps), "rows": [
                {"id": a["id"], "name": a["name"]} for a in apps]})],
            "list_schedules": [(200, {"total": 1, "rows": [SCHEDULE]})],
            "create_schedule": [(201, SCHEDULE)],
            "get_schedule": [(200, SCHEDULE)],
            "update_schedule": [(200, {**SCHEDULE, "cron": "0 10 * * 1"})],
            "delete_schedule": [(204, None)],
            "pause_schedule": [(200, {**SCHEDULE, "enabled": False})],
            "resume_schedule": [(200, {**SCHEDULE, "enabled": True})],
            "run_schedule": [(202, {"run_id": 55})],
            "list_schedule_runs": [(200, {"total": 1, "rows": [RUN]})],
            "change_schedule_owner": [(200, {**SCHEDULE, "owner": {"id": 15, "name": "Bob"}})],
            "check_schedule": [(200, {"risk": "high", "fingerprint": "abc123", "blocking": []})],
            "get_schedule_run": [(200, RUN)],
            "get_schedule_settings": [(200, {"allow_run_without_asking": False})],
            "update_schedule_settings": [(200, {"allow_run_without_asking": True})],
        }

    def answer(self, name, *answers):
        self.answers[name] = list(answers)

    def __call__(self, method, url, **kwargs):
        if method == "get" and url.startswith(APP):
            app = self.apps[int(url[len(APP):])]
            self.calls.append("get_app_details")
            return response(url, 200, {"id": app["id"], "name": app["name"], "version_details": {
                "id": app["version_id"], "agent_type": app["agent_type"]}})
        name = ROUTES[(method, url)]
        self.calls.append(name)
        self.bodies.setdefault(name, []).append(kwargs.get("json"))
        queue = self.answers[name]
        status, payload = queue.pop(0) if len(queue) > 1 else queue[0]
        return response(url, status, payload)


def guide_text():
    return GUIDE.read_text(encoding="utf-8")


def split(text, marker):
    """Map each heading of one level to the text below it, up to the next heading of that level."""
    parts = re.split(rf"^{marker} (.+)$", text, flags=re.MULTILINE)
    return {heading.strip(): body for heading, body in zip(parts[1::2], parts[2::2])}


def section(name):
    return split(guide_text(), "##")[name]


def method_sections():
    return {heading.strip("`"): body for heading, body in split(section("Methods"), "###").items()}


def python_blocks(text):
    return [code for lang, code in FENCE.findall(text) if lang == "python"]


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_URL", BASE)
    monkeypatch.setenv("API_KEY", API_KEY)
    monkeypatch.setenv("PROJECT_ID", str(PROJECT_ID))
    return monkeypatch


def run(api, *blocks):
    """Run blocks in one namespace, as a reader runs Setup and then one example."""
    namespace = {}
    with tempfile.TemporaryDirectory() as folder, patch("requests.Session.request", side_effect=api):
        for index, code in enumerate(blocks):
            script = Path(folder) / f"block_{index}.py"
            script.write_text(code, encoding="utf-8")
            namespace = runpy.run_path(str(script), init_globals=namespace, run_name="__main__")
    return namespace


def setup_block():
    (code,) = python_blocks(section("Setup"))
    return code


def end_to_end_block():
    (code,) = python_blocks(section("End-to-end example"))
    return code


def errors_block():
    (code,) = python_blocks(section("Errors"))
    return code


# Checks: sections, setup and access

def test_guide_has_the_sections_in_order():
    headings = list(split(guide_text(), "##"))
    assert [h for h in headings if h in SECTIONS] == list(SECTIONS)


def test_setup_names_the_three_variables():
    text = section("Setup")
    for name in ("DEPLOYMENT_URL", "API_KEY", "PROJECT_ID"):
        assert name in text


def test_access_section_states_caller_and_own_schedules():
    text = section("Who sees what")
    assert "acts as its caller" in text
    assert "only their own schedules" in text


def test_methods_table_has_arguments_and_endpoint():
    rows = {}
    for line in section("Methods").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        match = re.fullmatch(r"`(\w+)\((.*)\)`", cells[0]) if len(cells) >= 2 else None
        if match:
            rows[match.group(1)] = (match.group(2), cells[1])
    assert set(rows) == set(ENDPOINTS)
    for name, (verb, path) in ENDPOINTS.items():
        arguments, endpoint = rows[name]
        if name not in NO_ID_METHODS:
            assert "schedule_id" in arguments, name
        assert endpoint == f"`{verb} {path or '/'}`", name


@pytest.mark.parametrize("name", list(ENDPOINTS))
def test_each_method_example_runs_and_calls_it(env, name):
    sections = method_sections()
    assert name in sections
    (code,) = python_blocks(sections[name])
    api = FakeApi()
    run(api, setup_block(), code)
    assert name in api.calls


# Checks: examples stay in step with the client

def test_every_client_call_in_the_guide_exists():
    called = set(re.findall(r"\bclient\.(\w+)\(", guide_text()))
    assert set(ENDPOINTS) <= called
    for name in called:
        assert callable(getattr(EliteAClient, name, None)), name


def test_every_code_block_names_its_language():
    fences = re.findall(r"^```(\S*)$", guide_text(), re.MULTILINE)
    assert fences
    openings = fences[0::2]
    assert all(openings), openings


def test_examples_never_print_the_api_key(env, capsys):
    run(FakeApi(), end_to_end_block())
    for code in python_blocks(section("Methods")):
        run(FakeApi(), setup_block(), code)
    output = capsys.readouterr()
    assert API_KEY not in output.out + output.err


# Checks: the pipeline example and the run mode rules

def test_create_example_sends_a_pipeline_in_entry_point_mode(env):
    text = method_sections()["create_schedule"]
    (code,) = python_blocks(text)
    api = FakeApi()
    run(api, setup_block(), code)
    (body,) = api.bodies["create_schedule"]
    assert body["entity_type"] == "pipeline"
    assert body["application_id"] == PIPELINE["id"]
    assert body["version_id"] == PIPELINE["version_id"]
    assert body["run_mode"] == "entry_point"
    assert "prompt" not in body
    assert body["variables"] == []
    assert body["enabled"] is False
    assert "YAML `entry_point`" in text
    assert "empty input" in text


def test_create_section_names_the_run_mode_rules():
    text = method_sections()["create_schedule"]
    assert "A pipeline without `run_mode` gets 400" in text
    assert "`entry_point` with a `prompt` or `variables` gets 400" in text


def test_no_example_sends_entry_point_with_input(env):
    api = FakeApi()
    run(api, end_to_end_block())
    run(api, setup_block(), python_blocks(method_sections()["create_schedule"])[0])
    for body in api.bodies["create_schedule"]:
        if body.get("run_mode") == "entry_point":
            assert not body.get("prompt") and not body.get("variables")


# Checks: the end-to-end example

def test_end_to_end_creates_runs_lists_and_deletes(env, capsys):
    api = FakeApi()
    run(api, end_to_end_block())
    assert api.calls == ["get_list_of_apps", "get_app_details", "create_schedule",
                         "run_schedule", "list_schedule_runs", "delete_schedule"]
    (body,) = api.bodies["create_schedule"]
    assert body["enabled"] is False
    assert api.bodies["run_schedule"] == [{"trigger": "manual"}]
    out = capsys.readouterr().out
    assert "55" in out and "manual" in out


def test_end_to_end_body_fills_one_placeholder(env):
    api = FakeApi()
    run(api, end_to_end_block())
    (body,) = api.bodies["create_schedule"]
    assert body == {
        "name": "SDK example", "description": "Weekly news digest",
        "entity_type": "agent", "application_id": AGENT["id"], "version_id": AGENT["version_id"],
        "prompt": "Summarize the news about {{topic}}",
        "variables": [{"name": "topic", "value": "AI"}],
        "cron": "0 9 * * 1", "timezone": "UTC", "enabled": False,
    }


def test_end_to_end_skips_a_pipeline_listed_first(env):
    api = FakeApi(apps=(PIPELINE, AGENT))
    run(api, end_to_end_block())
    (body,) = api.bodies["create_schedule"]
    assert body["application_id"] == AGENT["id"]


def test_text_states_the_placeholder_rule(env):
    text = section("End-to-end example")
    assert "gets 400" in text
    assert "placeholder" in text
    api = FakeApi()
    run(api, end_to_end_block())
    (body,) = api.bodies["create_schedule"]
    placeholders = set(re.findall(r"\{\{\s*(\w+)\s*\}\}", body["prompt"]))
    assert placeholders == {v["name"] for v in body["variables"]} == {"topic"}


def test_missing_variable_stops_before_any_call(env):
    env.delenv("API_KEY")
    api = FakeApi()
    with pytest.raises(SystemExit) as stop:
        run(api, end_to_end_block())
    assert "API_KEY" in str(stop.value.code)
    assert api.calls == []


@pytest.mark.parametrize("apps", [(), (PIPELINE,)], ids=["empty", "pipeline_only"])
def test_no_agent_stops_before_create(env, apps):
    api = FakeApi(apps=apps)
    with pytest.raises(SystemExit) as stop:
        run(api, end_to_end_block())
    assert stop.value.code == "No agent in this project"
    assert "create_schedule" not in api.calls


# Checks: cleanup in finally

def test_failed_run_still_deletes(env):
    api = FakeApi()
    api.answer("run_schedule", (500, {"error": "boom"}))
    with pytest.raises(ApiDetailsRequestError) as error:
        run(api, end_to_end_block())
    assert error.value.status_code == 500
    assert api.calls[-2:] == ["run_schedule", "delete_schedule"]


def test_failed_runs_list_still_deletes_once(env):
    api = FakeApi()
    api.answer("list_schedule_runs", (500, {"error": "boom"}))
    with pytest.raises(ApiDetailsRequestError):
        run(api, end_to_end_block())
    assert api.calls.count("delete_schedule") == 1
    assert api.calls[-1] == "delete_schedule"


def test_failed_create_sends_no_delete(env):
    api = FakeApi()
    api.answer("create_schedule", (400, {"field": "cron", "error": "bad cron"}))
    with pytest.raises(ApiDetailsRequestError) as error:
        run(api, end_to_end_block())
    assert error.value.status_code == 400
    assert "delete_schedule" not in api.calls


def test_failed_delete_keeps_the_first_error(env, capsys):
    api = FakeApi()
    api.answer("run_schedule", (500, {"error": "boom"}))
    api.answer("delete_schedule", (503, {"error": "down"}))
    with pytest.raises(ApiDetailsRequestError) as error:
        run(api, end_to_end_block())
    assert error.value.status_code == 500
    assert str(SCHEDULE_ID) in capsys.readouterr().out


# Checks: the error example

def test_second_run_now_prints_409_and_active_run_id(env, capsys):
    api = FakeApi()
    api.answer("run_schedule", (202, {"run_id": 54}),
               (409, {"code": "run_active", "error": "A run of this schedule is active", "active_run_id": 54}))
    run(api, setup_block(), errors_block())
    assert api.calls.count("run_schedule") == 2
    out = capsys.readouterr().out
    assert "409" in out and "54" in out


def test_other_errors_reach_the_caller(env, capsys):
    api = FakeApi()
    api.answer("run_schedule", (404, {"code": "not_found", "error": "Schedule not found"}))
    with pytest.raises(ApiDetailsRequestError) as error:
        run(api, setup_block(), errors_block())
    assert error.value.status_code == 404
    assert "active_run_id" not in capsys.readouterr().out


def test_a_409_without_active_run_reaches_the_caller(env, capsys):
    api = FakeApi()
    deleted = {"code": "version_deleted", "error": "The version of this schedule is deleted"}
    api.answer("run_schedule", (202, {"run_id": 54}), (409, deleted))
    with pytest.raises(ApiDetailsRequestError) as error:
        run(api, setup_block(), errors_block())
    assert error.value.body == deleted
    assert "still active" not in capsys.readouterr().out


def test_errors_text_explains_403_and_404():
    text = section("Errors")
    assert "403" in text and "right" in text
    assert "404" in text and "another user" in text


def test_errors_text_names_the_shared_refusal_body():
    """Every refusal body carries code and error, plus field when one input is wrong."""
    text = section("Errors")
    for key in ('`code`', '`error`', '`field`'):
        assert key in text
    for code in ('invalid_input', 'not_owner', 'not_found', 'run_active'):
        assert f'`{code}`' in text


def test_errors_example_checks_the_run_active_code():
    (code,) = python_blocks(section("Errors"))
    assert 'error.body.get("code") != "run_active"' in code


def test_risk_text_says_every_refusal_has_a_code():
    assert "Every refusal has a `code`" in section("Risk refusals")


# Checks: risk_ack, blocked, approval_rule, install

def test_setup_shows_the_install_step():
    assert "pip install '.[all]'" in section("Setup")


def test_settings_text_names_the_settings_field():
    text = method_sections()["get_schedule_settings"] + method_sections()["update_schedule_settings"]
    assert "allow_run_without_asking" in text and "approval_rule" not in text
    assert "approval_rule" in method_sections()["create_schedule"]


def test_risk_section_runs_the_risk_ack_retry(env):
    api = FakeApi()
    api.answer("create_schedule", (409, RISK_409), (201, SCHEDULE))
    (code,) = python_blocks(section("Risk refusals"))
    run(api, setup_block(), code)
    first, second = api.bodies["create_schedule"]
    assert "risk_ack" not in first
    assert second["risk_ack"] == {"fingerprint": RISK_409["fingerprint"],
                                  "confirmed_name": first["name"]}
    assert api.calls.count("create_schedule") == 2


def test_blocked_refusal_is_not_retried(env, capsys):
    api = FakeApi()
    api.answer("create_schedule", (409, {"code": "schedule_blocked", "fingerprint": "x",
                                         "blocking": ["secret in prompt"]}))
    (code,) = python_blocks(section("Risk refusals"))
    run(api, setup_block(), code)
    assert api.calls.count("create_schedule") == 1
    assert "secret in prompt" in capsys.readouterr().out
    assert "schedule_blocked" in section("Risk refusals")


def test_setup_explains_tls_check_and_ca_bundle():
    text = section("Setup")
    assert "TLS" in text and "REQUESTS_CA_BUNDLE" in text
