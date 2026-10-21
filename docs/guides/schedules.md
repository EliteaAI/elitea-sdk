# Schedules

A schedule runs an agent or a pipeline on a cron timetable.
`EliteAClient` has one method for each Schedules API call.
This guide shows each method, who may call it, the errors, and one full example.

## Setup

Install the SDK from its folder:

```bash
pip install '.[all]'
```

The client needs the same three variables as the rest of the SDK (see "Environment Setup" in the `README.md`).

```bash
export DEPLOYMENT_URL="https://your-elitea-host"
export API_KEY="your-personal-access-token"
export PROJECT_ID="42"
```

Create the client once. The method examples below use this `client` and `schedule_id`.

```python
import os

from elitea_sdk.runtime.clients.client import ApiDetailsRequestError, EliteAClient

client = EliteAClient(
    base_url=os.environ["DEPLOYMENT_URL"],
    project_id=int(os.environ["PROJECT_ID"]),
    auth_token=os.environ["API_KEY"],
)
schedule_id = 7  # the id of one of your schedules
```

Never print or log `API_KEY`.

## Methods

All calls go to one resource: `/api/v2/elitea_core/schedules/prompt_lib/{project_id}`.
The endpoint column shows the HTTP method and the path after this base.

| Method and arguments | Endpoint | Answer |
|---|---|---|
| `list_schedules(limit=20, offset=0, **filters)` | `GET /` | One page: `total` and `rows` |
| `create_schedule(data)` | `POST /` | The new schedule (status 201) |
| `get_schedule(schedule_id)` | `GET /{schedule_id}` | The schedule |
| `update_schedule(schedule_id, data)` | `PUT /{schedule_id}` | The changed schedule |
| `delete_schedule(schedule_id)` | `DELETE /{schedule_id}` | `None` (status 204) |
| `pause_schedule(schedule_id)` | `POST /{schedule_id}/pause` | The schedule, `enabled` false |
| `resume_schedule(schedule_id)` | `POST /{schedule_id}/resume` | The schedule, `enabled` true |
| `run_schedule(schedule_id, trigger='manual')` | `POST /{schedule_id}/run` | `{"run_id": ...}` (status 202) |
| `list_schedule_runs(schedule_id, limit=20, offset=0, status=None)` | `GET /{schedule_id}/runs` | One page: `total` and `rows` |
| `change_schedule_owner(schedule_id, owner_id)` | `POST /{schedule_id}/owner` | The schedule with its new owner |
| `check_schedule(data)` | `POST /check` | The risk check of `data` |
| `get_schedule_run(schedule_id, run_id)` | `GET /{schedule_id}/runs/{run_id}` | One run |
| `get_schedule_settings()` | `GET /settings` | The project setting |
| `update_schedule_settings(data)` | `PUT /settings` | The changed project setting |

Each method returns the JSON answer as a `dict`.
When the API answers 400 or more, the method raises `ApiDetailsRequestError` (see [Errors](#errors)).

### `list_schedules`

Lists one page of the project schedules. `limit` is 1 to 100.
Filters go to the query: `entity_type`, `status` (of the last run), `enabled`, `owner_ids`, `entity_id`, `search`, `sort_by` and `sort_order`.
A filter set to `None` is not sent.

```python
page = client.list_schedules(limit=10, enabled=True)
print(page["total"])
for row in page["rows"]:
    print(row["id"], row["name"], row["next_run"])
```

### `create_schedule`

Creates a schedule. The client sends `data` as it is; the API checks it.

Keys: `name`, `description`, `entity_type` (`agent` or `pipeline`), `application_id`, `version_id`, `run_mode`, `prompt`, `variables`, `cron`, `timezone`, `expires_at`, `approval_rule` and `enabled`.
`enabled` false saves the schedule paused. The default is true.
`approval_rule` is `pause_and_ask` or `run_without_asking` (see [`get_schedule_settings`](#get_schedule_settings)).
A save can answer 409 for a risk. See [Risk refusals](#risk-refusals).

The run input depends on the entity:

- An agent takes a `prompt` and no `run_mode`. The [end-to-end example](#end-to-end-example) creates one.
- A pipeline needs `run_mode`. A pipeline without `run_mode` gets 400.
- `run_mode` `entry_point` starts the pipeline from its YAML `entry_point` with an empty input. Send no `prompt` and empty `variables`: `entry_point` with a `prompt` or `variables` gets 400.
- `run_mode` `prompt` sends `prompt` to the pipeline as its input.

This example creates a paused pipeline schedule in Entry point mode.
`get_app_details` without a version name answers the default version.

```python
pipeline_id = 12  # the id of one of your pipelines
version_id = client.get_app_details(pipeline_id)["version_details"]["id"]

schedule = client.create_schedule({
    "name": "Nightly ETL",
    "entity_type": "pipeline",
    "application_id": pipeline_id,
    "version_id": version_id,
    "run_mode": "entry_point",
    "variables": [],
    "cron": "0 2 * * *",
    "timezone": "UTC",
    "enabled": False,
})
print(schedule["id"], schedule["enabled"])
```

### `get_schedule`

Reads one schedule.

```python
schedule = client.get_schedule(schedule_id)
print(schedule["name"], schedule["cron"], schedule["enabled"])
```

### `update_schedule`

Changes the fields you send; the other fields stay.
The owner and the entity do not change here.

`update_schedule` refuses `enabled`: it raises `ValueError` and sends no request.
Use `pause_schedule` and `resume_schedule` instead.

```python
schedule = client.update_schedule(schedule_id, {"cron": "0 10 * * 1"})
print(schedule["cron"])

try:
    client.update_schedule(schedule_id, {"enabled": False})
except ValueError as error:
    print(error)
```

### `delete_schedule`

Deletes a schedule. You cannot undo it.

```python
client.delete_schedule(schedule_id)
```

### `pause_schedule`

Pauses a schedule, so it starts no future runs.
A run in progress goes on. No call stops a running run.

```python
schedule = client.pause_schedule(schedule_id)
print(schedule["enabled"])
```

### `resume_schedule`

Resumes a paused schedule, so it runs on its timetable again.
It gets 400 when the schedule has no version, and 409 when the project has its maximum number of active schedules.

```python
schedule = client.resume_schedule(schedule_id)
print(schedule["enabled"], schedule["next_run"])
```

### `run_schedule`

Starts one run now. It works on a paused schedule too.
`trigger` is `manual` (Run now, the default) or `test` (Test run): `client.run_schedule(schedule_id, trigger="test")`.
The answer holds the `run_id`. The run itself goes on in the background.

```python
answer = client.run_schedule(schedule_id)
print(answer["run_id"])
```

### `list_schedule_runs`

Lists one page of the runs of a schedule, newest first.
`status` keeps only runs with that status: `running`, `waiting_for_approval`, `success`, `failed`, `skipped` or `blocked`.

```python
page = client.list_schedule_runs(schedule_id, status="failed")
for run in page["rows"]:
    print(run["id"], run["trigger"], run["status"], run["started_at"])
```

### `change_schedule_owner`

Gives a schedule to another project member. Only a project admin may do it.
The old and the new owner get a notice.

```python
schedule = client.change_schedule_owner(schedule_id, owner_id=15)
print(schedule["owner"]["name"])
```

### `check_schedule`

Checks the risk of a schedule before you save it. Send the same `data` as `create_schedule`; the client sends it as it is.
The answer holds `fingerprint` and `blocking`. It needs the right to create schedules.

```python
check = client.check_schedule({
    "name": "Digest", "entity_type": "agent", "application_id": 5, "version_id": 91,
    "prompt": "Summarize the news", "cron": "0 9 * * 1", "timezone": "UTC",
})
print(check["fingerprint"], check["blocking"])
```

### `get_schedule_run`

Reads one run of a schedule.

```python
run = client.get_schedule_run(schedule_id, 55)
print(run["id"], run["status"])
```

### `get_schedule_settings`

Reads the project setting for schedules. `approval_rule` tells what a run does when it needs approval:
`pause_and_ask` pauses the run and waits for an answer, and `run_without_asking` goes on without asking.

```python
settings = client.get_schedule_settings()
print(settings["approval_rule"])
```

### `update_schedule_settings`

Changes the project setting. Only a project admin may do it.
Send `approval_rule` as `pause_and_ask` or `run_without_asking`.

```python
settings = client.update_schedule_settings({"approval_rule": "run_without_asking"})
print(settings["approval_rule"])
```

## Who sees what

The client acts as its caller: the user who owns the `API_KEY` token.
It has the same rights as that user has in the UI.

- A regular user sees only their own schedules. Another user's schedule answers 404.
- A project admin sees every schedule of the project. An admin may edit, pause, resume and delete any of them, and change the owner.
- Only the owner may run a schedule. An admin who runs another user's schedule gets 403.
- Each schedule has `allowed_actions`: the actions the caller may take on it.

## Errors

When the API answers 400 or more, the method raises `ApiDetailsRequestError`.
The error has two fields:

- `status_code`: the HTTP status.
- `body`: the parsed JSON answer, or the text when the answer is not JSON.

| Status | Meaning |
|---|---|
| 400 | A field is not valid. `body` names the problem, for example a bad cron or a `run_mode` rule. |
| 403 | The caller lacks the right for this action. Examples: a user without schedule rights, or an admin who runs another user's schedule. |
| 404 | The schedule does not exist, or it belongs to another user and the caller is not an admin. |
| 409 | Run now: a run of this schedule is still active, and `body["active_run_id"]` holds its id. Resume: the project has its maximum number of active schedules. |

`update_schedule` with `enabled` raises `ValueError` before any request.

This example handles one case: a second Run now while the first run is still active.
Any other error goes on to the caller.

```python
client.run_schedule(schedule_id)  # starts a run
try:
    client.run_schedule(schedule_id)  # the first run is still active
except ApiDetailsRequestError as error:
    if error.status_code != 409 or "active_run_id" not in error.body:
        raise
    print(f"Status {error.status_code}: run {error.body['active_run_id']} is still active")
```

## Risk refusals

A save (`create_schedule` or `update_schedule`) can answer 409 with a `code` in `body`:

| `code` | Meaning | What to do |
|---|---|---|
| `risk_confirmation_required` | The schedule carries a risk. `body["fingerprint"]` names it. | Read the risk, then send the same `data` again with `risk_ack` set to the fingerprint. |
| `schedule_blocked` | A rule blocks the schedule. `body["blocking"]` lists the reasons. | Change the schedule. A retry with `risk_ack` does not help. |

```python
data = {
    "name": "Digest", "entity_type": "agent", "application_id": 5, "version_id": 91,
    "prompt": "Summarize the news", "cron": "0 9 * * 1", "timezone": "UTC",
}
try:
    schedule = client.create_schedule(data)
except ApiDetailsRequestError as error:
    body = error.body if isinstance(error.body, dict) else {}
    if error.status_code != 409:
        raise
    if body.get("code") == "risk_confirmation_required":
        schedule = client.create_schedule({**data, "risk_ack": body["fingerprint"]})
    elif body.get("code") == "schedule_blocked":
        print("Blocked:", body["blocking"])
    else:
        raise
```

## End-to-end example

This script needs only the three variables of [Setup](#setup) and one agent in the project.
It does these steps:

1. It takes the first agent of the project and the agent's default version.
2. It creates a paused schedule (`enabled` false), so no cron run starts.
3. It starts one run now. Run now works on a paused schedule.
4. It prints the runs of the schedule.
5. It deletes the schedule in `finally`, so nothing stays behind, even when a step fails.

The agent prompt has one placeholder, `{{topic}}`, and one schedule variable, `topic`, fills it.
Each variable name must equal a placeholder of the prompt.
A variable that is not a placeholder, or a placeholder without a variable, gets 400.
An agent schedule has no `run_mode`.

When a variable is missing or the project has no agent, the script stops and creates nothing.

```python
import os
import sys

from elitea_sdk.runtime.clients.client import ApiDetailsRequestError, EliteAClient

for name in ("DEPLOYMENT_URL", "API_KEY", "PROJECT_ID"):
    if not os.environ.get(name):
        sys.exit(f"Set the {name} environment variable")

client = EliteAClient(
    base_url=os.environ["DEPLOYMENT_URL"],
    project_id=int(os.environ["PROJECT_ID"]),
    auth_token=os.environ["API_KEY"],
)

# The first agent and its default version; pipelines are skipped.
agent = None
for app in client.get_list_of_apps():
    version = client.get_app_details(app["id"])["version_details"]
    if version.get("agent_type") != "pipeline":
        agent = app
        break
if agent is None:
    sys.exit("No agent in this project")

schedule = client.create_schedule({
    "name": "SDK example",
    "description": "Weekly news digest",
    "entity_type": "agent",
    "application_id": agent["id"],
    "version_id": version["id"],
    "prompt": "Summarize the news about {{topic}}",
    "variables": [{"name": "topic", "value": "AI"}],
    "cron": "0 9 * * 1",
    "timezone": "UTC",
    "enabled": False,
})
print(f"Created schedule {schedule['id']} for agent {agent['name']}")

try:
    answer = client.run_schedule(schedule["id"])
    print(f"Started run {answer['run_id']}")
    for run in client.list_schedule_runs(schedule["id"])["rows"]:
        print(run["id"], run["trigger"], run["status"], run["started_at"])
finally:
    try:
        client.delete_schedule(schedule["id"])
        print(f"Deleted schedule {schedule['id']}")
    except ApiDetailsRequestError as error:
        # Keep the first error visible; tell the reader what is left behind.
        print(f"Could not delete schedule {schedule['id']} (status {error.status_code}). Delete it in the UI.")
```
