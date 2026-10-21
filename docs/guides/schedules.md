# Schedules

A schedule runs an agent or a pipeline on a cron timetable.
`EliteAClient` has one method for each Schedules API call.
This guide shows each method, who may call it, the errors, and one full example.

## Setup

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

## Who sees what

The client acts as its caller: the user who owns the `API_KEY` token.
It has the same rights as that user has in the UI.

- A regular user sees only their own schedules. Another user's schedule answers 404.
- A project admin sees every schedule of the project. An admin may edit, pause, resume and delete any of them, and change the owner.
- Only the owner may run a schedule. An admin who runs another user's schedule gets 403.
- Each schedule has `allowed_actions`: the actions the caller may take on it.
