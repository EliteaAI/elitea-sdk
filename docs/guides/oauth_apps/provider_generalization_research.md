# Research: generalizing admin-managed Entra ID apps (#7149) to all OAuth providers

Date: 2026-10-26 · Scope: research only, no implementation · Based on the local, uncommitted #7149 code
(see [implementation_summary.md](implementation_summary.md)).

## TL;DR

- **Yes, it can be reused.** About 70% of #7149 does not depend on Microsoft: write-only secret in the hidden
  vault, a credential that stores only a reference, server-side resolution in `expand_configuration`, project
  allow-list, Enabled/Default, the `default/choice/custom` form modes, and the proxy rule that sends the secret
  only to the app's own token endpoint.
- **What depends on Microsoft:** the data model (`entra_app`, `login.microsoftonline.com` only, a
  `teams|outlook|sharepoint` Literal), how endpoints are built (`{authority}/oauth2/v2.0/token`), refresh
  (`offline_access`), Test connection (client credentials + AADSTS codes), the SDK helpers
  (`is_entra_url`, `access_token_from_stored_login`), the UI hostname checks (`isMicrosoftEntraEndpoint`), and the
  names (`entra_app_id`, `configuration.entra_apps.*`, `entra_default_app`).
- **Main recommendation:** before #7149 merges, rename and reshape it as a generic **OAuth App** with a
  `provider` field, and keep `microsoft_entra` as the only provider for now. Doing this before release costs
  little. After release it needs a data migration for stored `entra_app_id` references and for the permissions.
- **Big gap:** there are no non-Microsoft delegated-OAuth credentials yet. Gmail uses `credentials_json`,
  BigQuery uses an API key, and GitHub/Jira/Confluence/Slack use tokens. A Google app is only useful once a
  toolkit can consume a Google user token. So supporting a provider takes two pieces of work: the **provider
  profile** and a **consumer credential**.

---

## 1. What #7149 has today, by how generic it is

| Area | Where | Generic? | Notes |
|---|---|---|---|
| Storage in the public project, `shared=False`, hidden from generic config APIs | `configurations/entra_apps.py`, `utils.py:384` | ✅ | Reuse as is |
| Write-only secret in the hidden vault, `client_secret_set` on read, purge on replace | `entra_apps.py` (`admin_view`, `validated_app_data`, `purge_replaced_secret`) | ✅ | Reuse as is |
| Credential stores the reference only; settings resolved at use time | `apply_entra_app`, `strip_app_client_fields`, `decorate_credentials` | ✅ logic / ❌ names | `APP_CLIENT_FIELDS` and `entra_app_id` are fixed |
| Project allow-list, enabled, availability reasons | `unavailable_reason` | ✅ | |
| Form modes `default / choice / custom` | `credential_mode`, `GET /configurations/entra_apps/{pid}?type=` | ⚠️ | "One default" is global; with several providers it must be one default **per provider / credential type** |
| Proxy pins the secret to the app's token endpoint | `elitea_core/api/v2/mcp_oauth_proxy.py` `_resolve_entra_app_client` | ⚠️ | The idea is generic; `_entra_token_endpoint()` builds the Entra URL shape |
| Model | `models/pd/entra_app.py` | ❌ | `ENTRA_HOSTS`, `is_entra_authority`, `ToolkitType = Literal['teams','outlook','sharepoint']` |
| Consumers list | `ENTRA_APP_TOOLKIT_TYPES`; `elitea_core/utils/mcp_oauth.py` `OAUTH_CONFIGURATION_KEYS` | ❌ | Hard-coded lists in two plugins |
| Test connection | `check_app_connection` | ❌ | `client_credentials` + `graph/.default` + AADSTS codes. Google, GitHub and Slack user-OAuth clients cannot be checked this way |
| SDK runtime | `elitea_sdk/runtime/utils/mcp_oauth.py:1018-1066`, `configurations/entra_app_mode.py` | ❌ | `is_entra_url`, `entra_token_endpoint`, `offline_access` in the refresh, Graph default scope |
| UI | `EntraAppSelect`, `entraApp.helpers.js`, `useEntraAppSubsections`, `mcpAuthFlow.helpers.js:23,43,490` | ⚠️ | The select and helpers can be generalized; provider workarounds are hostname checks (no `resource`, no `prompt=consent` on Entra) |
| Seed, permissions | `entra_app_seed.py` (`entra_default_app`), `configuration.entra_apps.view/edit` | ❌ names | |

**Related mechanism to merge, not duplicate:** pre-built MCP servers already carry platform-owned OAuth clients
(`client_id`, `client_secret`, `scopes`) in **pylon YAML**, merged in by
`elitea_core/methods/mcp_prebuilt_config.py:resolve_mcp_prebuilt_settings`. Those secrets are plain YAML, cannot
be edited in the UI, and have no allow-list. A generic OAuth App could take this over: a pre-built MCP server
would reference an `oauth_app_id`. The OpenAPI toolkit's delegated flow (`oauth_discovery_endpoint`) is a
natural consumer of a `generic_oidc` app.

---

## 2. Provider differences the design must cover

| Concern | Microsoft Entra | Google | GitHub | Atlassian (Jira/Confluence Cloud) | Slack | Salesforce |
|---|---|---|---|---|---|---|
| Endpoints | Built from the authority (`/{tenant}/oauth2/v2.0/...`), OIDC discovery | Fixed URLs, OIDC discovery | Fixed, **no discovery**; GHES host varies | Fixed `auth.atlassian.com` | Fixed `slack.com/oauth/v2/...` | `login.salesforce.com` or My Domain, discovery |
| Refresh token | `offline_access` scope | `access_type=offline`; re-issued only with `prompt=consent`; **expires in 7 days if the app is in "Testing"** | OAuth App: no expiry, no refresh. GitHub App: 8 h token, refresh token valid ~6 months | `offline_access`, **rotating** refresh tokens | Optional token rotation (12 h) | `refresh_token` scope |
| Scope format | Resource URIs or short Graph names, separated by spaces | Full URLs, separated by spaces | Separated by spaces (commas also accepted) | Separated by spaces + `audience=api.atlassian.com` | **Commas**; `scope` (bot) and `user_scope` are separate parameters | Separated by spaces |
| Extra authorize params | Do not send `prompt=consent` (forces admin approval) | `access_type`, `include_granted_scopes`, `hd` | `allow_signup` | `audience`, `prompt=consent` | `user_scope`, `team` | `prompt` |
| Token response | Standard JSON | Standard + `id_token` | Form-encoded unless `Accept: json` (the proxy already sends it) | Standard; API base needs `cloudid` from `accessible-resources` | HTTP 200 with `ok:false` on error; user token nested in `authed_user` | Includes **`instance_url`**, which must be saved |
| Tenancy / instance | Tenant is in the authority (`/organizations`, `/{tid}`) | Workspace `hd`; Internal vs External app | GHES host | Site chosen after sign-in (cloud id) | Workspace install | My Domain / sandbox host |
| Consent | Tenant admin consent | App verification for sensitive/restricted scopes; 100 test users while unverified | Org approval of OAuth apps | User consent per site | **Workspace admin installs the app** | Connected App policies |
| Test without a user | Client credentials (partial, see #7149) | ❌ not for OAuth clients | ❌ (only `/applications/{id}/token` check with Basic auth) | ❌ | ❌ | Client credentials only if the Connected App enables it |
| Client auth | Secret or **certificate** (recommended) | Secret | Secret | Secret (POST body) | Secret | Secret, JWT bearer |
| Secret expiry | Max 24 months, must be tracked | No expiry | No expiry | Can rotate | Can rotate | – |

What this means for the design:

1. **Endpoints are data, not a formula.** Store `authorization_endpoint`, `token_endpoint`, and optional
   `issuer`/`discovery_url`, `revocation_endpoint`, `userinfo_endpoint`. The provider profile fills them in
   (Entra builds them from the tenant; Google/GitHub/Slack have fixed values; `generic_oidc` uses discovery).
2. **Host allow-list per provider.** Replace `ENTRA_HOSTS` with "the hosts of this app's configured endpoints".
   For named providers they are fixed in code (`oauth2.googleapis.com`, `github.com`, …). For `generic_oidc`
   they come from the admin-entered URL (admins are trusted, users are not). The proxy keeps today's rule: send
   the secret only to `app.token_endpoint`.
3. **Each provider gets its own authorize parameters, scope separator and refresh strategy,** declared in the
   profile and delivered to the UI by the backend. This replaces hostname checks such as
   `isMicrosoftEntraEndpoint`.
4. **Normalize the token response** in each provider: Slack `ok`/`authed_user`, Salesforce `instance_url`,
   Atlassian cloud id.
5. **Rotating refresh tokens** (Atlassian, Slack, Entra sometimes): every refresh must save the new refresh
   token on the credential. Check that the stored-login path writes it back. Today
   `access_token_from_stored_login` returns only the access token.
6. **Test connection depends on the provider.** Offer only what each provider supports: (a) discovery/metadata
   reachable, (b) client credentials where allowed, (c) GitHub app-token check, (d) **"Test by signing in"**:
   the admin completes the real authorization-code flow in a popup. (d) works for every provider and is also
   the only real check of delegated scopes.
7. **Some settings belong to the credential, not the app.** The app holds the client, the endpoints, the default
   scopes and the refresh policy. The credential holds what the sign-in returns (Salesforce `instance_url`,
   Atlassian cloud id, GHES base URL if not fixed per app, Entra tenant if the app is `/organizations`).
8. **Show the redirect URI.** The admin form should display the exact callback URL to register
   (`{host}/app/mcp-auth-callback`) with a copy button. GitHub OAuth Apps allow only one callback URL.
9. **Certificate / `private_key_jwt` client auth.** Leave it out now, but leave room for it:
   `token_endpoint_auth_method: client_secret_post | client_secret_basic | private_key_jwt | none`. Entra advises
   certificates in production; a public PKCE-only client (`none`) is also valid.
10. **Always use PKCE**, and add the secret for confidential clients. Entra and Google support PKCE; check
    the other providers.

---

## 3. Proposed generic design

### 3.1 Data model: one configuration type with a provider field

```text
type: oauth_app            (was entra_app)
data:
  provider: microsoft_entra | google | github | atlassian | slack | salesforce | generic_oidc | generic_oauth2
  # common
  client_id, client_secret (write-only, vault), token_endpoint_auth_method
  enabled, allowed_project_ids
  default_for: [credential_type, ...]      # replaces the single global is_default
  scopes: {<credential_type>: [..]}        # str keys, not a Literal
  auto_refresh_token
  # provider-specific, validated by a Pydantic discriminated union on `provider`
  settings: { tenant/authority | hd | ghes_host | my_domain | discovery_url | endpoints... }
```

- Use Pydantic `Field(discriminator='provider')` so that each provider validates its own `settings`. This keeps
  `extra='forbid'` and the Entra host check where they belong.
- `default_for` (or "one default per provider") is needed because a Google default must not clear the
  Microsoft default. Today `_clear_other_defaults` clears every other app.

### 3.2 Provider profile (code, not stored data)

One small class per provider, ideally in the **SDK** so the backend and the runtime share it:

```text
OAuthProviderProfile
  id, label, allowed_hosts
  endpoints(settings) -> authorize/token/revoke/userinfo
  authorize_params(settings, consumer) -> dict      # access_type, audience, prompt rules, user_scope…
  scope_separator, refresh_scope_rule              # offline_access / access_type=offline / none
  normalize_token_response(raw) -> {access_token, refresh_token, expires_in, extra: {instance_url…}}
  test_strategies -> [discovery, client_credentials, sign_in]
  map_error(body) -> admin message                 # AADSTS map moves here
```

This replaces `is_entra_url`, `entra_token_endpoint`, `_entra_token_endpoint`, the AADSTS table, and the UI
hostname checks.

### 3.3 Consumers declare themselves

Credentials in the SDK declare which app they accept, instead of hard-coded lists
(`ENTRA_APP_TOOLKIT_TYPES`, `OAUTH_CONFIGURATION_KEYS`):

```python
model_config = ConfigDict(json_schema_extra={"metadata": {
    "oauth_app": {"providers": ["microsoft_entra"], "recommended_scopes": [...]}}})
```

The backend reads this from the schema it already collects, the `available` API exposes it, and the UI decides
whether to show the app selector. Adding a consumer then needs changes in the SDK only.

### 3.4 Rename everywhere (cheap now, a migration later)

| Today | Generic |
|---|---|
| `entra_app` type, `entra_apps` section | `oauth_app`, `oauth_apps` |
| `entra_app_id` on credentials and in `McpOAuthTokenRequest` | `oauth_app_id` (accept `entra_app_id` as an alias for one release if needed) |
| `GET /configurations/entra_apps/{pid}` | `GET /configurations/oauth_apps/{pid}?type=` (+ `provider` in each item) |
| `configuration.entra_apps.view/edit` | `configuration.oauth_apps.view/edit` |
| `entra_default_app` seed | `oauth_default_apps: [{provider: microsoft_entra, ...}]` |
| Admin "MS Entra ID Apps" | "OAuth Apps": a provider picker, then a form for that provider (the admin can filter by provider) |
| `EntraAppSelect`, `entraApp.helpers.js`, `useEntraAppSubsections` | `OAuthAppSelect`, … (provider label and icon) |
| SDK `entra_app_mode.py` | `oauth_app_mode.py` (`require_app_client` is already generic) |

Users can still see "Microsoft Entra ID" as the provider label, so the UI wording asked for in #7149 stays.

---

## 4. Other points to cover

- **Security**
  - SSRF: Test connection, discovery and `generic_oidc` fetch admin-entered URLs from the server. Require https
    and block private/link-local addresses.
  - Keep the current guarantees for every provider: the secret is never returned or seen by the SDK client, the
    proxy pins the token endpoint, and the generic config APIs cannot read apps.
  - Changing the `client_id` makes every stored refresh token of that app invalid. Warn the admin, and expect
    users to sign in again.
- **Token cache keys:** today they are `uuid:endpoint`. That is fine for several providers, but the key must use
  the endpoint *resolved from the app*, as `decorate_credentials` already does for Entra.
- **Secret lifecycle:** add `secret_expires_at` (entered by the admin or read from the provider where possible,
  e.g. Graph `passwordCredentials`) and warn in the Admin Portal before expiry. Entra secrets always expire.
- **Provider policies that affect operations:** Google app verification and the 7-day refresh tokens in
  "Testing" mode; Slack workspace install by an admin; GitHub org approval of OAuth apps. Document these in the
  admin form help text, because they are the usual "it works for me but not for users" problems.
- **Sign-in identity:** keep `id_token`/userinfo (email, tenant/workspace) on the credential to show "signed in
  as …". This also helps catch a sign-in to the wrong tenant or site.
- **Revocation:** optionally call the revoke endpoint when a credential is deleted (Google, GitHub and Slack have
  one; Entra has none per app).
- **Feature toggle "default app only"** (postponed) should become a per-provider
  setting, not a global one.
- **Back-compat:** Custom app credentials stay unchanged for every provider. That rule from #7149 applies as is.

---

## 5. Suggested phasing

| Phase | Content | Effort |
|---|---|---|
| **0 — before #7149 merges** | Rename to `oauth_app` / `oauth_app_id` / `oauth_apps.*`; add `provider: microsoft_entra`; scopes keyed by `str`; default per provider; move the Entra-specific code behind an `OAuthProviderProfile` that has only one implementation. No new providers. | S–M, mostly renames, plus the existing tests |
| 1 — generic OIDC + MCP pre-built | `generic_oidc` profile; OpenAPI delegated credential and pre-built MCP servers reference an app; YAML secrets move into the hidden vault | M |
| 2 — first non-Microsoft provider | Pick one with a real consumer. **Google** needs a new delegated Google credential (Gmail/Drive/Calendar). **Atlassian** needs Jira/Confluence to support an OAuth token + cloud id API base. **GitHub** is easiest: tokens work with the existing wrapper; a GitHub App gives refresh. | M–L each, mostly consumer work |
| 3 — hardening | "Test by signing in", secret expiry warnings, certificate auth, revocation | M |

## 6. Open questions for the team

1. Should Phase 0 go into #7149 now (no migration) or into a follow-up (migration of `entra_app_id` and the permissions)?
2. Which non-Microsoft provider comes first, and which toolkit will use it? Without a consumer the provider has no value.
3. Should pre-built MCP OAuth clients move from pylon YAML into OAuth Apps (the admin owns them, secrets in the vault)?
4. One default per provider, or per credential type (e.g. a different Entra app for SharePoint than for Teams)?
5. Should apps allow certificate auth (`private_key_jwt`) from the start, given Microsoft's production guidance?
