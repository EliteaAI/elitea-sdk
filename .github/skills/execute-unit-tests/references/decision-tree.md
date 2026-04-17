# Decision Tree for Test Execution

## Decision Tree

```
User prompt received
       │
       ▼
Does it mention a specific test name/class?
  YES ─► use ::ClassName::method_name targeting
  NO  ─► extract topic keywords
             │
             ▼
       Map keywords → file path(s)
       (see keyword-mapping.md)
             │
             ▼
       Single file? ──YES──► runTests with that file
             │
            NO
             ▼
       Multiple files ──► runTests with list OR chain pytest paths
             │
             ▼
       Loader tests involved? ──YES──► check if marker -m filter applies
             │                         (see loader-tags-and-filtering.md)
            NO
             ▼
       Run and report results
```

---

## After Running Tests

| Outcome | Action |
|---------|--------|
| **All pass** | Confirm to the user and list the test count |
| **Some fail** | Show the failure summary; point to file + class + method; suggest `-x` (stop-on-first-fail) or `--lf` (last-failed) |
| **Import / collection error** | Virtualenv is likely not activated or a dependency is missing — instruct the user to activate it and re-run |
| **Credential / network error** (integration tests only) | Advise the user to verify `.env` values before re-running |
| **Loader baseline mismatch** | See [baseline-management.md](baseline-management.md) for diagnosis and regeneration |

---

## Useful Flags Reference

| Flag | Effect |
|------|--------|
| `-v` | Verbose output — show each test name |
| `-x` | Stop on first failure |
| `--lf` | Re-run only last-failed tests |
| `-s` | Show stdout/stderr (useful for debugging loader output) |
| `-k "<expr>"` | Filter tests by keyword expression |
| `-m "<mark>"` | Filter tests by marker (loader tests only) |
| `--reportportal` | Upload results to ReportPortal (see [reportportal.md](reportportal.md)) |
| `-p no:reportportal` | Explicitly disable ReportPortal plugin |
