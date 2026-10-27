# #7149 — OAuth Apps (Microsoft Entra ID): implementation summary

Issue: https://github.com/EliteaAI/elitea_issues/issues/7149 (parent #6823, depends on #7148)
Status: implemented locally, nothing committed. Verified on the local stack (`http://localhost`).

> **2026-10-26 — generalized for other OAuth providers.** The feature is now named **OAuth Apps**:
> `entra_app` → `oauth_app`, `entra_app_id` → `oauth_app_id`, `configuration.entra_apps.*` →
> `configuration.oauth_apps.*`, and each app has a **`provider`** field (only `microsoft_entra` exists
> today). Nothing is committed, so no migration ships; local data was converted by hand (section 8).
> Background: [provider_generalization_research.md](provider_generalization_research.md).

> **2026-10-26 — requirements clarified for per-project apps.** The full, current specification is
> **[spec.md](spec.md)** (with a TL;DR for review). This file records what is **built**; items marked
> ⟶ *spec* below behave differently in the spec and are still to be implemented (spec section 11).
> Interview record: [requirements_interview.md](requirements_interview.md).

## 1. What it does

Admins configure OAuth apps once, in the Admin Portal. Each app belongs to one **provider**; a credential
type uses only apps of its provider. Today the only provider is **Microsoft Entra ID**, used by the Teams,
Outlook and SharePoint credentials instead of carrying their own client ID, secret and OAuth endpoint.

### Admin Portal — Configuration → OAuth Apps (after MCP Servers)

- List of apps: name, provider, client ID, projects (All / N selected), status (Enabled, Default).
- Create / edit / delete in a dialog. A **General** block at the top holds the behaviour settings:
  Provider, Name, Enabled, Default app and Projects (described below). Below it come the app's own settings:
  - **Provider**: chosen on create (only "Microsoft Entra ID" today), read-only on edit. It sets the
    credential types the app serves, the scope lists and the accepted discovery endpoint.
  - Client ID, OAuth discovery endpoint (Entra default `https://login.microsoftonline.com/organizations`;
    only the provider's hosts are accepted, for Entra `https://login.microsoftonline.com/...`).
  - Client secret is **write-only**: shown as `•••• (set)` with **Replace**; never returned by any API.
  - **Scopes** per credential type of the provider: checkboxes for Teams and Outlook (same lists as the SDK,
    recommended set preselected), free text for SharePoint. Empty = the credential type's recommended set.
    Scopes for a type outside the provider are rejected.
  - **Auto refresh token** (offline_access).
  - **Projects**: allow-list; empty = every project. ⟶ *spec*: required, no preselect; "Public — all projects"
    or team projects (never both); personal projects not offered; a project's apps override Public apps.
  - **Enabled**.
  - **Default app**: users get this app and cannot choose a custom one. One default **per provider**
    (the newest choice wins). ⟶ *spec*: one default per provider **per scope** (Public, each team project);
    a conflicting save is rejected.
  - **Test connection** (per provider; Entra): sends a client-credentials token request to the app's own authority and reports
    a wrong secret, an expired secret, an unknown client ID or an unknown tenant. A tenant whose policy
    blocks app-only tokens (common with `/organizations`) is reported as a warning: the secret could not
    be fully confirmed. Unsaved form values are tested; an omitted secret uses the stored one.
- No "Supported toolkits": an app serves every credential type of its provider (Entra: all three).

### Credential form (Teams, Outlook, SharePoint) — driven by the admin's apps

| Admin configuration for the project | What the user sees |
|---|---|
| An enabled app is marked **Default app** | No Default/Custom switcher. Only "Sign-in settings are managed by your administrator." (an app selector only when several apps apply). |
| Enabled app(s), **none is the default** | Default app / Custom app switcher. |
| **No** enabled app (none created, all disabled, deleted, or not allowed for the project) | No switcher: the plain Custom app form with every field. |

- **Default app** mode stores only `{oauth_app_id}`. Client, secret, endpoint, scopes and auto refresh come
  from the app at use time, so an admin change reaches every credential without user action.
  ⟶ *spec*: stores `"default"` (follows the admin's current choice) or an explicit app id; resolved on every
  use by the credential's own project; a sign-in is bound to its app.
- **Custom app** mode shows every field (client ID, secret, endpoint, scopes, auto refresh), unchanged from
  before. Existing credentials keep working and open in Custom app.
- A credential still pointing at a removed, disabled or not-allowed app keeps the switcher and shows the
  error, so the user can switch to Custom app.
- **Shared** is removed from these three credential forms (other credential types unchanged).
- In the **Configuration OAuth** sign-in window, the Scope field is read-only for a Default app credential:
  it shows the admin's scopes. Custom app credentials keep an editable Scope field.
- An existing credential saved with its own app stays on Custom app even when the admin later marks an
  app as the default; it is never switched to the default app silently. ⟶ *spec* (changed): a Default app
  overrides Custom on next use; the credential returns to its own app if the Default goes away.
- SharePoint also keeps its **App-only** option.

## 2. Requirements — status

| # | Requirement (issue #7149 + follow-ups) | Status |
|---|---|---|
| 1 | Admin Portal section for multiple Entra ID apps | Done — Configuration → OAuth Apps (provider Microsoft Entra ID) |
| 2 | Client secret write-only, never returned by any API | Done — `client_secret_set` only; checked across all read APIs |
| 3 | OAuth discovery endpoint per app (`/organizations` or tenant) | Done — provider's hosts only (Entra: `login.microsoftonline.com`) |
| 4 | Supported toolkits multi-select | Removed by follow-up — apps serve all three types |
| 5 | Default scopes per type | Done — app-level scopes per type; users no longer set them in Default mode |
| 6 | Elitea default multi-tenant app pre-configured | Done — seeded once from `oauth_default_app` config (#7148 provides the app); can be disabled, not deleted |
| 7 | One default app | Done — single "Default app" switch, one default per provider |
| 8 | Default / Custom tabs in the credential UI | Done — and now driven by the admin config (table above) |
| 9 | "Ask your admin" when no app | Done — with no enabled app the form is plain Custom app |
| 10 | Existing credentials keep working (Custom), no migration | Done |
| 11 | Credential stores a reference, not a copy of the secret | Done — `oauth_app_id` only |
| 12 | Secret resolved server-side at OAuth / refresh time; never in UI, API or SDK client | Done — OAuth proxy resolves `oauth_app_id`, sends the secret only to the app's own token endpoint |
| 13 | Token cache keys stay per credential (`uuid:endpoint`) | Done — unchanged |
| 14 | Disabled / deleted app → clear error | Done — on save, test connection, sign-in and runtime |
| 15 | Rotating the secret applies to all Default-mode credentials | Done |
| 16 | Per-project availability (research question) | Done — project allow-list on each app. ⟶ *spec*: tiers (project apps win Public), required Projects, per-scope Default |
| 17 | Move to Configuration after MCP Servers, rename "MS Entra ID Apps" | Done — later renamed "OAuth Apps" (#24) |
| 18 | Default app: no options for the user (scopes, auto refresh in Admin Portal) | Done |
| 19 | Custom app: all options | Done |
| 20 | Remove "Shared" from the credential form | Done (Teams/Outlook/SharePoint) |
| 21 | Credential form depends on Enabled / Default in the Admin Portal | Done (table above) |
| 22 | Test connection for admins | Done |
| 23 | Feature toggle "default app only" | Postponed — not built; the original plan used the old `entra_*` names and was not kept (per provider when built) |
| 24 | Reusable for other OAuth providers (Google etc.) | Done (foundation) — renamed to `oauth_app` / `oauth_app_id` / `oauth_apps.*`, `provider` field, provider registry; no second provider yet |
| 25 | Per-project resolution, follow-the-admin, sign-in binding, delete impact, full server enforcement | Specified — [spec.md](spec.md) sections 3–8; not implemented |

## 3. Answers to the research questions

**Per-project apps — makes sense and is feasible.** Built as an allow-list on each environment-wide app
(empty = all projects), enforced on the project app list, on credential save and at runtime. The clarified
requirements replace "empty = all" with an explicit **Public** scope and make a project's own apps win over
Public apps, mirroring how the platform already resolves shared credentials and default models (project
first, then the public project `ai_project_id`). See [spec.md](spec.md) section 4.

**Where the secret is stored.** As an `oauth_app` configuration in the public project (`ai_project_id`):
the secret goes to that project's **hidden vault**, the record keeps only `{{secret.<uuid>}}`, and no API
returns either. Existing admin features that store secrets are not write-only: plugin config (`format:
password`, e.g. OIDC client secret) is plain text in the plugin YAML and returned by the admin GET; the Admin
Secrets page uses the administration vault and its Reveal returns plaintext.

## 4. Design notes

- Apps are `oauth_app` configurations, `shared=False`, in the public project. The generic configuration
  APIs refuse to list, read, create, edit or delete them, and an app can't be resolved by its title.
- **Provider.** `data.provider` (`microsoft_entra`; records without it are read as Entra). It cannot change
  after creation. `models/pd/oauth_app.py` holds the provider registry `OAUTH_PROVIDERS`: label, credential
  types, allowed hosts, default discovery endpoint, token path. `CREDENTIAL_PROVIDERS` maps a credential
  type to its provider; an app of another provider is refused for that credential ("… is a <provider> app
  and cannot be used by this credential"). Test connection dispatches per provider (`_CONNECTION_CHECKS`).
- **Adding a provider** (e.g. Google): an `OAUTH_PROVIDERS` entry + `ProviderId` literal value, a
  connection check, an admin UI entry in `OAUTH_PROVIDERS` (`oauthApps.constants.js`), and credential types
  in the SDK that carry `oauth_app_id`. Storage, write-only secret, allow-list, Default/Custom modes, proxy
  pinning and the credential form are reused unchanged. Still Entra-only: the SDK runtime helpers
  (`is_entra_url`, `access_token_from_stored_login`) and the UI's `isMicrosoftEntraEndpoint` sign-in
  workarounds (see the research report).
- One server-side hook, `apply_oauth_app`, runs inside `expand_configuration`, so test connection,
  OAuth sign-in, token refresh, stored logins and toolkit runs all get the app's settings.
- `GET /configurations/oauth_apps/{project_id}?type=` returns the usable apps of that type's provider
  (no secrets, with `provider`) and a `mode` (`default` / `choice` / `custom`) that drives the credential form.
- The OAuth proxy accepts `oauth_app_id` instead of a client secret, and refuses any token endpoint that
  isn't the app's own (built by the provider: `resolve_app_client` → `token_endpoint`).
- The sign-in prompt for a Default app credential carries the app's client ID and reference, never the
  secret (not even masked).

## 5. Changed files

**configurations** (`centry2/pylon_main/plugins/configurations`)
- New: `models/pd/oauth_app.py`, `oauth_apps.py`, `methods/oauth_apps.py`, `oauth_app_seed.py`,
  `api/v2/oauth_apps.py`, `tests/unit/test_7149_oauth_apps.py`
- Changed: `utils.py`, `module.py`, `models/pd/registry.py`, `api/v2/{available,check_connection,check_connections,configuration,configurations}.py`

**elitea_core** — `api/v2/mcp_oauth_proxy.py`, `models/pd/mcp_oauth.py`; new
`tests/integration/test_7149_oauth_proxy_app.py`

**admin** (repo `admin/` and copy `centry2/pylon_main/plugins/admin/`) — new `api/v2/oauth_apps.py`,
`api/v2/oauth_app_check.py`, `tests/test_oauth_apps_api.py` (routes `/api/v2/admin/oauth_apps/administration`,
`/api/v2/admin/oauth_app_check/administration`)

**admin_ui** (repo `admin_ui/` and copy `centry2/pylon_main/plugins/admin_ui/`) — new
`frontend/src/api/oauthApps.api.js`, `frontend/src/pages/ConfigurationPage/components/OAuthAppsSection/`
(provider registry in `constants/oauthApps.constants.js`);
changed `ConfigurationPage.jsx`, `api/admin.api.js`, `constants/permissions.constants.js`

**elitea-sdk** — new `elitea_sdk/configurations/oauth_app_mode.py` (`OAUTH_APP_ID_FIELD`, UI component
`oauth_app_select`), `tests/configurations/test_7149_oauth_app_mode.py`; changed `configurations/{teams,outlook,sharepoint}.py`,
`tools/{teams,outlook,sharepoint}/__init__.py`

**EliteaUI** — new `features/credentials/api/oauthAppsApi.js`, `features/credentials/ui/oauth-app-select/`
(`OAuthAppSelect`), `features/credentials/lib/helpers/oauthApp.helpers.js`,
`features/credentials/lib/hooks/useOAuthAppSubsections.hooks.js`
(+ tests); changed `features/toolkits/ui/form/tool-base/{ToolSection,ToolBaseProperty,ToolBase}.jsx`,
`features/mcp/{lib/helpers/mcpAuthFlow.helpers.js,lib/hooks/useMcpAuthModal.hooks.js,ui/modal/McpAuthModal.jsx}`,
`features/sharepoint/ui/*`, `pages/Credentials/CredentialForm.jsx`

**Permissions** — `configuration.oauth_apps.view`, `configuration.oauth_apps.edit` (admin, super_admin);
seeded automatically on startup.

## 6. Verification

- Unit/integration tests (after the rename + provider, 2026-10-26): configurations 579 passed (7 new
  provider tests); elitea_core OAuth proxy 4 passed; elitea-sdk 537 passed (M365/OAuth subset); EliteaUI
  966 passed (credentials, toolkits, mcp, sharepoint); admin 6 passed. elitea_core's full suite has
  failures, but there are more of them without this change (32) than with it (28), so they predate it.
- Builds: EliteaUI and admin_ui build and lint clean.
- Live checks (local stack):
  - Secret and its vault reference absent from every response; generic APIs can't reach apps.
  - Credential form in all three admin states (default / choice / custom).
  - Default credential stored as `{oauth_app_id}` only; test connection prompt carries the app's scopes
    and client ID, no secret.
  - Proxy refuses a foreign token endpoint.
  - Admin Test connection reports a wrong secret, an unknown client ID and a policy block.
  - After the rename: `/admin/oauth_apps` lists the app with `provider: microsoft_entra` and no secret (the
    old `/admin/entra_apps` returns 404); unknown provider, changed provider and scopes for a foreign type
    are rejected; project 2 gets mode `default`, project 1 (not allowed) `custom`; the Default credential
    reads back with the app's client ID, endpoint and scopes; the proxy refuses a foreign token endpoint
    ("does not belong to the selected OAuth app"), reaches Microsoft with the app's own one, and refuses a
    project outside the allow-list.

## 7. Open items

- **Real Microsoft sign-in not yet tested**: needs the real client secret entered on "Tenant test app"
  (it holds a placeholder) and `http://localhost/app/mcp-auth-callback` registered as a redirect URI.
- Not tested yet: toolkit-page Login button and an agent run with a Default-mode toolkit.
- Admin plugin's full test suite fails to load locally (missing `croniter`), unrelated to this change.
- Local test data to clean up: app "Tenant test app", credentials `ui_teams_default`, `ui_teams_default_2`.
- The default-only feature toggle is postponed, not built.
- Per-project resolution and related rules: specified in [spec.md](spec.md), to be implemented
  (status table in its section 11). The earlier resolution plan predated the clarified requirements and was dropped.
- Not re-checked in the browser after the rename (the Playwright browser was held by another session):
  Admin Portal "OAuth Apps" dialog with the Provider select, and the credential form.

## 8. Local data migration done for the rename (not shipped)

Nothing is released, so the code has no migration. The local DB was converted once:

- `p_*.configuration`: `type 'entra_app'` → `'oauth_app'` with `provider: microsoft_entra` added; the
  `entra_app_id` key in credential data → `oauth_app_id` (19 credentials); title `entra_app_elitea_default` →
  `oauth_app_elitea_default` (not present locally).
- `public.auth_core__role_permission`: `configuration.entra_apps.*` → `configuration.oauth_apps.*` (12 rows).
- The indexer's site-packages had a stale `elitea_sdk/configurations/entra_app_mode.py` from the old install;
  removed after reinstalling the SDK.
- Any other environment that ran the uncommitted build needs the same SQL.
