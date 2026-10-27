# OAuth Apps (#7149) — verification checklist

Checks for issues found and fixed during development. Each line is one behaviour to verify; run them on
every change to the feature. Spec: [spec.md](spec.md).

Setup: an OAuth app in Admin Portal → Configuration → OAuth Apps, a user with a personal (private)
project and, where noted, a team project.

## Admin Portal — OAuth Apps

- [ ] The OAuth Apps section opens without "Unexpected Application Error" (generated RTK Query hook names
      `useOauthApp…` are exported under the names the components import).
- [ ] The edit icons sit in their own column and do not wrap onto a second line.
- [ ] Provider dropdown is disabled and offers only "Microsoft Entra ID".
- [ ] General block at the top holds Name, Enabled, Default app and Projects.
- [ ] Client secret is never shown: `•••• (set)` with Replace; saving without Replace keeps the stored secret.
- [ ] Test connection reports a wrong secret, an unknown client ID and a tenant that blocks app-only tokens
      (warning), and never contacts a host other than the provider's.
- [ ] Each credential type (Teams, Outlook, SharePoint) has an on/off switch; switching all off is rejected.
- [ ] The list shows a Credentials column with the types switched on.
- [ ] "+ New Configuration for App" button label.
- [ ] Opening an existing app for edit: Save is **disabled** until something changes; reverting the change
      disables it again. Test connection is available without changes.
- [ ] Projects is required: Save stays disabled with no project selected ("Required. The app applies only in
      the selected projects."), for a new app and when editing.

## Project scope

- [ ] An app assigned to the **public project** is available to users in **every** project, including their
      personal (private) project — not only inside the public project.
- [ ] An app assigned to a team project is not offered in other projects.
- [ ] **Team project wins Public:** in a team project with its own app for a credential type, Public apps are
      not offered for that type, even when the Public app is Default and the team app is not. Example: Public
      Default app (Outlook, SharePoint) + team app (Teams, Outlook, SharePoint, not Default) → in the team
      project Outlook shows the Default/Custom switcher with only the team app (no app selector); other
      projects still get the Public Default.
- [ ] The project tier counts only types switched on: a team app with Outlook off does not hide the Public
      Outlook app in that project.
- [ ] A credential in a team project that references a Public app reports it as not available once the
      project has its own app for that type.
- [ ] There is no "all projects" shortcut: an empty project list is never treated as every project.
- [ ] The server rejects create, update and Test connection without projects ("Select at least one project"),
      including an update that removes all projects, and rejects unknown project ids ("Unknown project: <id>").
- [ ] An app with no projects (e.g. its only project was deleted) applies nowhere.
- [ ] A lookup without a project never resolves an app.
- [ ] The seeded Elitea default app is created with the public project assigned.

## Credential form — what the user sees

- [ ] When one credential type (toolkit) is switched off on the app, the Default app is no longer available
      for that type's credentials: the form shows only Custom app fields.
- [ ] Existing credentials of a switched-off type show "… is no longer offered for <Type> credentials by your
      administrator. Switch to Custom app."; switching the type on again restores them.
- [ ] App marked **Default app** → no Default/Custom switcher, only "Sign-in settings are managed by your
      administrator."
- [ ] App not marked Default → Default app / Custom app switcher.
- [ ] No usable app (none, disabled, deleted, not in scope) → no switcher, plain Custom app form.
- [ ] A new credential opens on the Default app tab, not on Custom app.
- [ ] Default app tab shows no scopes, auto refresh or client fields.
- [ ] Custom app tab shows every field and starts empty (the default app's discovery endpoint is not carried
      over when switching tabs).
- [ ] An existing credential saved with its own app opens on Custom app (not silently switched).
- [ ] "Shared" is not offered on Teams, Outlook and SharePoint credentials.
- [ ] Disabled app message reads "<name> was disabled by your administrator. Switch to Custom app."

## Saving

- [ ] Creating a credential with the Default app and only a name saves it, returns to the list and shows **no**
      "unsaved changes" warning.
- [ ] Before typing anything, Save is disabled (values the form fills in itself don't mark it dirty).
- [ ] A saved Default app credential stores only `{oauth_app_id}` (no client ID, secret, endpoint or scopes).
- [ ] Opening an existing credential and leaving without edits shows no warning.

## Sign-in and runtime

- [ ] Configuration OAuth window: Scope is read-only for a Default app credential and shows the admin's
      scopes; editable for Custom app credentials.
- [ ] Test connection of a Default app credential returns a sign-in prompt with the app's client ID and
      reference, never the secret (not even masked).
- [ ] The OAuth proxy refuses a token endpoint that isn't the app's own (403).
- [ ] An admin app can't be reached through the generic configuration APIs: list, read, create, edit, delete,
      lookup by title and lookup by `configuration_uuid` are all refused.
