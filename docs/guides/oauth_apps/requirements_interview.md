# #7149 — OAuth app settings (AS) per project: requirements

Status: requirements agreed in interview (2026-10-26). Not implemented.
Terms: **AS** = an OAuth app configuration in Admin Portal → Configuration → OAuth Apps.
**Public** = the platform public project (`ai_project_id`, `get_public_project_id()`); its AS apply to every
project. **Provider** = identity provider of the AS (today only Microsoft Entra ID → Teams, Outlook, SharePoint).

## 1. Admin Portal — AS project scope

| # | Requirement |
|---|---|
| A1 | **Projects is required.** Nothing is preselected; Save is disabled until at least one entry is chosen. |
| A2 | The project picker lists **Public** (labelled "Public — all projects") **and team projects**. Personal (private) projects are not offered. The picker supports search. |
| A3 | An AS is **either Public, or one or more team projects** — never both. Mixing is rejected on save: "An app is either for all projects (Public) or for selected team projects, not both." |
| A4 | **At most one enabled Default AS per provider per scope**: one for Public, and one per team project. Saving a Default AS that would give any of its projects a second Default is rejected: "pr_1 already has a default app: <name>". Applies also when enabling, re-assigning projects or switching Default on. |
| A5 | The list shows each AS's scope ("Public" / project names) and a Default chip. |
| A6 | The seeded Elitea default app (#7148) is created with Projects = Public. No migration: the feature is new. |
| A7 | **Delete** shows a confirmation with the number of credentials that use the AS and what happens to them: "N credentials use this app. After deletion they will use <resolved app> / will need another app or Custom app." Delete is allowed. |
| A8 | Disable is offered as the reversible alternative to delete (same warning text, "can be re-enabled"). |

## 2. Resolution rule (create form, test connection, sign-in, token refresh, agent runs)

For a credential of provider **P** whose **own project is X** (R6: never the project an agent runs in):

1. **Project tier** = enabled AS of P assigned to X (empty when X is Public or a personal project).
2. **Public tier** = enabled AS of P assigned to Public.
3. **Active tier** = project tier if it has any AS, else Public tier. A non-empty project tier **fully hides** Public AS in X, Default or not.
4. Active tier empty → **Custom app only** (no switcher).
5. Active tier has a Default → **Default used, no switcher**.
6. No Default → **switcher**: one AS → preselected; several → the user must pick one (Save disabled until picked).

| Case (from the original list) | Result |
|---|---|
| 1. No AS for Public or X | Custom app fields only |
| 2. Only Public AS, no Default | Switcher; Public AS |
| 3. Only Public AS, one Default | No switcher; Default preselected |
| 4. Only X AS | As 2 / 3 with X's AS |
| 5. Public AS and X AS | X tier wins, then as 2 / 3 |
| 6. Several X AS | One Default → it wins; no Default → user picks from X's AS (same for several Public AS) |

## 3. What a credential stores and how it follows admin changes

| # | Requirement |
|---|---|
| R1 | A Default-mode credential stores either **"default"** (took what the admin designates) or an **explicit AS id** (user picked from a list). It never stores client ID, secret, endpoint, scopes or refresh settings. |
| R2 | **Resolve on use**: every use (opening the credential, test connection, sign-in, token refresh, agent run) applies section 2 with the current admin settings. No background job rewrites credentials. |
| R3 | **"default"** always follows the admin: adding a project AS that now wins, changing the Default, deleting or disabling the AS — the credential uses the newly resolved AS automatically. If nothing resolves → it behaves as "needs another app or Custom app" (R5). |
| R4 | **Explicit pick** is used only while no Default AS resolves for the credential's project and the AS stays enabled and in scope. Once a Default resolves, the credential switches to it (R8); if the Default goes away, the pick applies again. |
| R5 | Explicit AS deleted, disabled or out of scope → credential shows: "<name> was removed/disabled by your administrator. Pick another app or switch to Custom app." Agent runs fail with the same message. Nothing is deleted from the credential; re-enabling the AS restores it. |
| R6 | Resolution always uses the **credential's own project**. A personal credential used by an agent in pr_1 still resolves by its personal project (→ Public tier). |
| R7 | **Sign-in follows the app**: a stored sign-in (refresh token / browser token) is bound to the AS (id + client ID) it was made with. When the resolved AS or its client ID / endpoint changes, the old sign-in is ignored and the user signs in once again. A token is never used with another app. |
| R8 | **A Default AS overrides Custom.** When the credential's active tier has a Default AS, every credential of that provider in the scope uses it on its next use (R2), including credentials saved with the user's own Custom app. The user's own client ID, secret, endpoint and scopes are kept in the credential but not used, and the user signs in once with the Default app (R7). If the Default AS is later removed, disabled or no longer covers the project, such a credential returns to its own Custom app. New Custom credentials cannot be created while a Default AS applies. |

## 4. Server enforcement (every operation, not only the UI)

| # | Requirement |
|---|---|
| S1 | Create / update of a credential is rejected when: Custom app is chosen in a scope with a Default AS; an explicit AS is outside the credential's scope, disabled or of another provider; several AS apply, there is no Default and none is picked. |
| S2 | Test connection, OAuth sign-in (proxy), token refresh, stored-login refresh and toolkit/agent runs apply section 2 and R5 on the server; the browser's client ID, scopes or endpoint are never trusted for a Default-mode credential. |
| S3 | The AS secret is sent only to the AS's own token endpoint; never returned by any API (unchanged). |
| S4 | Admin AS writes (create, update, delete, Default, projects) re-validate A1, A3, A4 on the server; only Administration admins may perform them (unchanged). |
| S5 | Save-time checks use the current admin settings, not what the form showed (an admin may change AS while a user has the form open). |
| S6 | Direct API / MCP calls go through the same checks; the UI only mirrors them. |

## 5. Edge cases

| Event | Behaviour |
|---|---|
| Project deleted | Its id is removed from AS project lists; an AS left with no projects is inactive and shown as "No projects". |
| AS projects changed so X is no longer covered | Same as disabled for credentials in X (R3 / R5). |
| AS client ID or endpoint changed | All credentials resolving to it need one new sign-in (R7). |
| Two admins save conflicting Defaults at once | Server check in a transaction; the second save is rejected (A4). |
| Credential's project is Public | Only the Public tier applies. |
| Default AS appears in a scope | All credentials of the provider there (default, explicit pick or Custom) use it on next use; one new sign-in each (R7, R8). |
| That Default AS is removed later | Credentials go back to what they stored: their explicit AS, "default" re-resolution, or their own Custom app. |
| User opens a credential while the AS list is loading | Form shows no fields until resolution is known (current behaviour). |

## 6. Out of scope (separate decisions)

- Feature toggle "default app only" (postponed).
- Project admins managing their own project's AS (today only platform admins).
- Providers other than Microsoft Entra ID.

## 7. Acceptance checks

1. Each of the six cases in section 2 on the create form, for Public, a team project and a personal project.
2. Adding a pr_1 AS moves existing "default" credentials in pr_1 to it on next use and asks for a new sign-in;
   explicit picks stay.
3. Deleting the Default AS: delete dialog shows the count; "default" credentials resolve to the next AS or
   show the "pick another app or Custom" message; explicit picks show the R5 message.
4. Personal credential used by an agent in pr_1 uses the Public AS.
5. Admin save rejected for: no projects, Public + team projects, second Default in a project.
6. API calls that bypass the form are rejected for every S1 case.
7. Re-enabling a disabled AS restores credentials without user action.
8. A Custom credential in pr_1 switches to a newly added pr_1 Default AS on its next use (asks for one sign-in),
   and returns to its own app when that Default AS is deleted.
