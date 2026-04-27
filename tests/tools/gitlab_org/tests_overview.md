# GitLab Org Toolkit — Tests Overview

## Package Structure

```
tests/tools/gitlab_org/
├── __init__.py
├── test_gitlab_org_repository_access.py   # Repository access control tests
└── tests_overview.md                      # This file
```

---

## Coverage Matrix

### `test_gitlab_org_repository_access.py`

**Scope:** `_get_repo()` method — repository access control logic in `GitLabWorkspaceAPIWrapper`  
**Related Bug:** [#3330] Single-configured-repo did not restrict access to other repositories

| # | Test Class | Test Method | Scenario | Expected Outcome |
|---|------------|-------------|----------|-----------------|
| 1 | `TestGitLabOrgRepositoryAccess` | `test_access_configured_repository_succeeds` | Request a repo that **is** in `repo_instances` | Returns the configured repo instance ✅ |
| 2 | `TestGitLabOrgRepositoryAccess` | `test_access_unconfigured_repository_raises_error` | Request a repo that is **not** in a non-empty `repo_instances` | Raises `ToolException` with repo name and allowed list ❌ |
| 3 | `TestGitLabOrgRepositoryAccess` | `test_access_with_none_returns_first_configured_repo` | `repository_name=None` with at least one configured repo | Returns the first configured repo ✅ |
| 4 | `TestGitLabOrgRepositoryAccess` | `test_no_configured_repos_and_none_raises_error` | `repository_name=None` with **empty** `repo_instances` | Raises `ToolException` — no repos configured ❌ |
| 5 | `TestGitLabOrgRepositoryAccess` | `test_no_configured_repos_allows_any_repository` | Named repo request with **empty** `repo_instances` | Fetches repo dynamically; adds it to `repo_instances` ✅ |
| 6 | `TestGitLabOrgRepositoryAccess` | `test_multiple_configured_repos_restricts_access` | Two repos configured; access configured ones (ok) and unconfigured one (fail) | Configured repos accessible; unconfigured raises `ToolException` ✅/❌ |
| 7 | `TestGitLabOrgRepositoryAccess` | `test_error_message_includes_all_allowed_repos` | Access blocked repo; three repos configured | Error message contains `"Allowed repositories:"` and the blocked repo name ❌ |
| 8 | `TestGitLabOrgRepositoryAccess` | `test_single_configured_repo_blocks_other_repos` | **Bug #3330 core case:** exactly one configured repo; access another | Raises `ToolException`; other repo blocked ❌ |
| 9 | `TestBug3330Regression` | `test_bug_3330_single_repo_scenario` | Exact bug reproduction: one configured repo, agent accesses different repo | Access denied with proper error message ❌ |
| 10 | `TestBug3330Regression` | `test_bug_3330_distinguishes_empty_from_configured` | Distinguish: empty `repo_instances` (allow all) vs. one configured (restrict) | Empty → dynamic fetch allowed; configured → others blocked ✅/❌ |

---

## Key Behaviors Under Test

| Behavior | Tests |
|----------|-------|
| **Happy path** — configured repo is returned | #1, #3, #6 (partial) |
| **Access restriction** — non-configured repos are blocked when config is non-empty | #2, #6, #7, #8, #9 |
| **Fallback to first repo** — `None` name with configured repos | #3 |
| **Empty config error** — `None` name with no repos configured | #4 |
| **Open access** — empty config allows dynamic fetch of any repo | #5, #10 |
| **Error message quality** — error includes blocked repo name + allowed list | #2, #7, #8, #9 |
| **Bug #3330 regression** — single-repo config must still restrict access | #8, #9, #10 |

---

## Method Under Test

```
GitLabWorkspaceAPIWrapper._get_repo(repository_name)
```

Implemented via a standalone `simulate_get_repo()` helper that mirrors the exact production logic, enabling unit testing without live GitLab connectivity.

### Access Control Decision Tree

```
_get_repo(repository_name)
├── repository_name is None / empty
│   ├── repo_instances is EMPTY  → ToolException("haven't configured any repositories")
│   └── repo_instances non-empty → return first configured repo
└── repository_name is provided
    ├── name IS in repo_instances  → return repo_instances[name]
    └── name NOT in repo_instances
        ├── repo_instances non-empty → ToolException("not in configured repositories list")
        └── repo_instances EMPTY    → fetch dynamically, cache in repo_instances, return
```

---

## What Is NOT Yet Tested

| Area | Notes |
|------|-------|
| Live GitLab API integration | Requires real credentials; not covered by unit tests |
| `_get_repo()` error wrapping for GitLab API errors | Non-`ToolException` errors are wrapped — not directly tested here |
| Other `GitLabWorkspaceAPIWrapper` methods | `create_file`, `update_file`, `read_file`, etc. have no unit tests yet |
| Authentication / token handling | Not covered |
| GitLab non-org toolkit | Separate toolkit; no tests yet |

---

*Last updated: 2026-04-27*

