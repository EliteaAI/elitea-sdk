# ms_graph toolkit — requirements and implementation plan

Version: 2026-10-28 · Status: draft for review · Builds on OAuth Apps (#7149, `docs/guides/oauth_apps/`)

## 1. Requirements (after the interview)

| # | Requirement | Decision |
|---|---|---|
| F1 | Low-level access to any Microsoft Graph endpoint, with arbitrary data | Tools `graph_request` + `graph_batch` |
| F2 | Arbitrary scopes | Free-text scope list on the credential (Custom app) and on the admin app (Default app) |
| F3 | OAuth Apps integration | New credential type `ms_graph`, a 4th credential type of the Microsoft Entra provider |
| F4 | Scope ownership in Default app mode | Same as SharePoint: admin's scopes apply, read-only for the user |
| F5 | Auth | Delegated only (user sign-in, refresh token). No app-only |
| F6 | Write safety | Allowed methods + path allow/deny prefixes in **toolkit settings** |
| F7 | Defaults | Methods `[GET]`; path lists empty = all paths; deny wins over allow |
| F8 | Content | JSON/text responses, binary download to artifact, binary upload from artifact, `v1.0` and `beta` |
| F9 | Empty scopes | `User.Read` (+ `offline_access`, always added) |
| F10 | Clouds | Global only: `graph.microsoft.com`, `login.microsoftonline.com` |
| F11 | Relation to M365 | Independent toolkit; its HTTP client is written so it can become `microsoft365/core/graph_client.py` later |
| F12 | Branching | Stacked on the #7149 branches in every repo |

### Assumptions (not asked; change if wrong)

| # | Assumption |
|---|---|
| A1 | Scopes are Graph delegated permissions. Bare names (`Mail.Read`) are sent as is; values with the `https://graph.microsoft.com/` prefix are accepted; other resources (`*.sharepoint.com`, `.default` of another API) are rejected — the token is Graph-only. `.default` of Graph is allowed as a literal scope. |
| A2 | Changing scopes does not invalidate the cached sign-in. A 403 returns a hint: "add `<scope>` and sign in again" (same as Teams). Token key stays `<credential uuid>:<endpoint>`. |
| A3 | Path rules are not a security boundary for the token (Graph scopes are), they constrain the LLM. They are enforced in the SDK only. |
| A4 | `$batch` itself is exempt from the method rule; each sub-request is checked, so a batch of GETs works with the `[GET]` default. Read-style `POST` endpoints (`/search/query`) need `POST` in allowed methods (can be narrowed with `allowed_paths`). |
| A5 | Large-file upload sessions (`createUploadSession`, chunked) are out of v1; simple upload up to a configurable limit (default 25 MB). |

## 2. Design

### 2.1 Credential `ms_graph` (`elitea_sdk/configurations/ms_graph.py`)

Same shape as `TeamsConfiguration`, but scopes are free text:

| Field | Notes |
|---|---|
| `oauth_app_id` | `OAUTH_APP_ID_FIELD`, "Default app" subsection |
| `client_id`, `client_secret`, `oauth_discovery_endpoint`, `scopes`, `auto_refresh_token` | "Custom app" subsection |
| `scopes` | `List[str]`, free text (no `checkbox_list`); validator `normalize_free_scopes()` (split on comma/space, strip Graph prefix, dedupe, reject non-Graph resources, empty → `["User.Read"]`) |
| `refresh_token` | hidden, set by Login |

`check_connection`: delegated, `GET /me?$select=id,displayName`. 401 → `McpAuthorizationRequired`; 403 → "token lacks User.Read". `_build_mcp_authorization_required` is a copy of the Teams one with `resource_name="Microsoft Graph"`, `credential_type="ms_graph"` and the free scope normalizer.

Refactor (small, in scope): move the duplicated `_build_mcp_authorization_required` of Teams/Outlook into `configurations/microsoft_graph_oauth.py` and use it from all three. Optional — skip if it blows up the diff of #7149.

### 2.2 Toolkit (`elitea_sdk/tools/ms_graph/`)

```
__init__.py       MsGraphToolkit, get_tools; token lookup + proactive OAuth guard (copy of Teams)
api_wrapper.py    MsGraphApiWrapper(BaseModel): settings, rules, tools, run()
graph_client.py   GraphClient: base URL, auth header, 429/503 retry with Retry-After,
                  401 refresh via stored refresh token, paging, host guard, size limits
rules.py          RequestRules: allowed methods, allow/deny prefixes, matching
models.py         args schemas
```

**Toolkit settings** (`toolkit_config_schema`):

| Setting | Default | Notes |
|---|---|---|
| `ms_graph_configuration` | — | `configuration_types: ['ms_graph']` |
| `allowed_methods` | `["GET"]` | multiselect of GET, POST, PATCH, PUT, DELETE |
| `allowed_paths` | `[]` | prefixes; empty = all |
| `denied_paths` | `[]` | prefixes; deny wins |
| `max_response_chars` | 20000 | JSON/text truncated with a note |
| `max_upload_mb` / `max_download_mb` | 25 / 50 | |
| `selected_tools` | all | `graph_request`, `graph_batch` |

**Path matching** (`rules.py`): normalise the path (drop scheme/host, drop `/v1.0` or `/beta`, drop query, collapse `//`, lowercase, percent-decode once, reject `..`). A rule matches when it is a segment-prefix of the path (`/me/messages` matches `/me/messages/123`, not `/me/messagesX`); `*` matches one segment (`/users/*/messages`). Order: method check → deny → allow. Errors name the rule: "DELETE is not allowed by this toolkit (allowed: GET)".

**Tool `graph_request`**:

```
method: GET|POST|PATCH|PUT|DELETE = GET
path: str                    "/me/messages" or "https://graph.microsoft.com/v1.0/me/messages"
query: dict | None           {"$top": 10, "$select": "id,subject"}
body: dict | list | str | None
headers: dict | None         e.g. ConsistencyLevel, Prefer; Authorization/Host are dropped
api_version: v1.0|beta = v1.0   (ignored when path is an absolute URL, which carries its own)
follow_next_link: int = 0    extra pages to fetch; values merged into "value"
upload_from_artifact: {bucket, file_name, content_type?} | None   body taken from the artifact (binary)
download_to_artifact: {bucket, file_name} | None   save the binary response there
```

Result: `{status, headers: <selected: content-type, location, request-id, retry-after>, body | artifact, next_link, truncated}`.
- Non-JSON binary response without `download_to_artifact` → saved to a default bucket/name (`ms_graph/<timestamp>-<name>`) and the reference returned; never raw bytes to the LLM.
- 4xx/5xx → `ToolException` with Graph `error.code` / `message`, plus the 403 scope hint.
- 202 + `Location` (async operations) returned as is.

**Tool `graph_batch`**: `requests: [{id, method, url, body?, headers?}]`, max 20 (Graph limit), `api_version`. Every sub-request goes through the rules; results returned per id; no `dependsOn` validation beyond passing it through.

**Host guard**: the token is only ever sent to `https://graph.microsoft.com/{v1.0|beta}/…`. Absolute `path`, `nextLink` and `Location` follow-ups on other hosts are rejected (`@microsoft.graph.downloadUrl` pre-authenticated URLs are fetched **without** the Authorization header).

**Tool descriptions**: short usage guide (path forms, `$select`/`$filter`, `ConsistencyLevel: eventual` for `$count`/`$search`, paging) + the toolkit's current allowed methods and path rules, so the LLM doesn't try forbidden calls.

### 2.3 Platform (pylon_main, on #7149 branches)

| Repo / file | Change |
|---|---|
| `configurations/models/pd/oauth_app.py` | `credential_types=('teams', 'outlook', 'sharepoint', 'ms_graph')`; scopes validation for `ms_graph` = free text, Graph-only (same rule as A1) |
| `configurations/oauth_apps.py` | nothing type-specific expected; verify `resolve_oauth_app` / `decorate_credentials` pass `scopes['ms_graph']` |
| `elitea_core/utils/mcp_oauth.py` | `OAUTH_CONFIGURATION_KEYS += ('ms_graph_configuration',)` |
| `configurations/api/v2/oauth_apps.py` | enum comes from `OAUTH_APP_TOOLKIT_TYPES`, no change |
| seeded Elitea app | `ms_graph` stays **off** on the seed app by default (scopes unknown) — admin turns it on |
| pylon_indexer | grep for hardcoded `teams/outlook/sharepoint` type lists (token injection, stored-login refresh) and add `ms_graph` |

### 2.4 UI

| Repo / file | Change |
|---|---|
| admin_ui `OAuthAppsSection/constants/oauthApps.constants.js` | `{ type: 'ms_graph', label: 'Microsoft Graph', options: null, recommended: ['User.Read'] }` (free text like SharePoint) |
| EliteaUI `credentials/lib/helpers/oauthApp.helpers.js` | `OAUTH_APP_CREDENTIAL_TYPES` add `ms_graph` |
| EliteaUI `storedLogin.helpers.js`, `common/toolkitUtils.jsx` (cases at ~296, ~510) | add `ms_graph` next to `teams` |
| EliteaUI credential form | scopes as free-text chips (reuse the SharePoint scopes input); read-only in Default app mode |
| icon | `ms_graph.svg` (Graph logo) |

## 3. Implementation steps

1. **SDK credential**: `configurations/ms_graph.py`, free scope normalizer in `microsoft_graph_scopes.py`, register in `configurations/__init__.py`. Unit tests: normalizer, `require_app_client`, authority validator, `check_connection` (200/401/403).
2. **SDK client + rules**: `graph_client.py`, `rules.py`. Unit tests: path normalisation and matching (prefix, `*`, deny wins, `..`, encoded paths, absolute URLs), host guard, retry/Retry-After, 401 refresh, paging cap, truncation.
3. **SDK toolkit**: `api_wrapper.py`, `models.py`, `__init__.py`, register in `tools/__init__.py`. Unit tests with a fake Graph (`requests_mock`): GET/POST/PATCH/DELETE, rules rejections, batch (20 limit, per-request rules), artifact upload/download, binary auto-save, OAuth guard raising `McpAuthorizationRequired` with `oauth_app_id` and no secret.
4. **Platform**: provider registry, proxy keys, scopes validation, indexer type lists. Extend `test_7149_oauth_apps.py` and `test_7149_oauth_proxy_app.py` for `ms_graph`.
5. **Admin UI + EliteaUI**: constants, form, icon; component tests next to the existing OAuth app tests.
6. **Local E2E** (dev tenant): Custom app credential → Login → `GET /me`, `GET /me/messages` with paging, `POST` blocked by default, then allowed; Default app credential with admin scopes, scope field read-only; download a drive file to an artifact and upload it back; denied path rejected; `beta` call.
7. **Docs**: `docs/guides/ms_graph/README.md` (setup, scopes, rules examples), update `oauth_apps/spec.md` (credential types list, sections 3.1 and 9) and `checks.md`.

## 4. Risks

| Risk | Mitigation |
|---|---|
| Users request scopes the shared admin app isn't consented for (AADSTS65001) | Default app scopes are admin-only (F4); clear sign-in error text |
| LLM issues destructive calls | GET-only default, method + path rules, descriptions list the rules |
| Token sent to a non-Graph host | Host guard on every request, nextLink and Location |
| Huge responses blow up context | `max_response_chars`, `$select` hint, binary to artifact |
| Rules bypass via path tricks (`%2F`, `..`, case) | Normalisation tests (step 2) |
| #7149 changes under review | Stacked branches; rebase after #7149 merges |
