# Bitbucket Toolkit — Tests Overview

## Package Structure

```
tests/tools/bitbucket/
├── __init__.py
├── test_bitbucket_delete_file.py     # delete_file tool tests (wrapper + Cloud/Server backends)
├── test_bitbucket_rate_limit.py      # Rate limit handling & retry logic tests
├── test_bitbucket_read_file_cap.py   # read_file/read_multiple_files output-cap tests
└── tests_overview.md                 # This file
```

---

## Coverage Matrix

### `test_bitbucket_delete_file.py`

**Scope:** New `delete_file` tool — wrapper orchestration in `api_wrapper.py`, Cloud backend's `files`-field deletion via the `src` endpoint, and Server backend's explicit unsupported-operation error in `cloud_api_wrapper.py`
**Related Issue:** [#3401] Add Delete File Tool to BitBucket Toolkit (parity with GitHub/GitLab/ADO Repos)

#### Class: `TestBitbucketAPIWrapperDeleteFile` — `BitbucketAPIWrapper.delete_file` orchestration

| Test Method | Scenario | Expected Outcome |
|-------------|----------|-----------------|
| `test_deletes_existing_file` | File exists on branch | Delegates to backend with a default `Delete {path}` commit message ✅ |
| `test_uses_provided_commit_message` | Caller supplies `commit_message` | Backend receives the caller-supplied message ✅ |
| `test_falls_back_to_active_branch_when_branch_empty` | `branch=None` | Uses `self._active_branch` ✅ |
| `test_raises_when_file_does_not_exist` | File missing on branch | Raises `ToolException` before calling the backend delete ❌ |
| `test_wraps_backend_error_from_delete` | Backend raises a generic `Exception` | Wrapped into a descriptive `ToolException` ❌ |
| `test_propagates_tool_exception_from_backend_unchanged` | Backend raises `ToolException` (e.g. Server limitation) | Re-raised unchanged, message preserved ❌ |

#### Class: `TestBitbucketCloudApiDeleteFile` — `BitbucketCloudApi.delete_file`

| Test Method | Scenario | Expected Outcome |
|-------------|----------|-----------------|
| `test_posts_files_field_without_content_to_mark_deletion` | Delete a file, no commit message | POSTs to `src` with `{"branch", "files"}` and no content field for that path (Bitbucket Cloud's documented deletion convention) ✅ |
| `test_includes_commit_message_when_provided` | Delete with a `commit_message` | Adds a `message` field to the form data ✅ |

#### Class: `TestBitbucketServerApiDeleteFile` — `BitbucketServerApi.delete_file`

| Test Method | Scenario | Expected Outcome |
|-------------|----------|-----------------|
| `test_raises_tool_exception_explaining_platform_limitation` | Any delete attempt | Raises `ToolException` explaining Bitbucket Server's REST API has no file-deletion endpoint ❌ |

### `test_bitbucket_rate_limit.py`

**Scope:** `is_rate_limit_error()` helper + `retry_on_rate_limit()` decorator — rate limit detection and exponential-backoff retry logic in `cloud_api_wrapper.py`  
**Related Bug:** [#4056] Bitbucket toolkit lacked rate limit handling; 429 errors failed immediately without retry

#### Class: `TestIsRateLimitError` — Detection logic for 429 errors

| # | Test Method | Scenario | Expected Outcome |
|---|-------------|----------|-----------------|
| 1 | `test_detects_429_status_code` | Error message contains `"429"` | Returns `True` ✅ |
| 2 | `test_detects_too_many_requests` | Error message contains `"too many requests"` | Returns `True` ✅ |
| 3 | `test_detects_rate_limit_text` | Error message contains `"rate limit"` | Returns `True` ✅ |
| 4 | `test_does_not_match_other_errors` | Timeout / auth / 404 errors | Returns `False` ✅ |
| 5 | `test_case_insensitive_detection` | Mixed-case variants of rate limit strings | Returns `True` regardless of case ✅ |

#### Class: `TestRetryOnRateLimitDecorator` — Retry decorator behaviour

| # | Test Method | Scenario | Expected Outcome |
|---|-------------|----------|-----------------|
| 6 | `test_successful_call_returns_immediately` | Function succeeds on first try | Returns result; called exactly once ✅ |
| 7 | `test_retries_on_rate_limit_error` | Fails twice with 429, succeeds on 3rd attempt | Retries until success; returns result ✅ |
| 8 | `test_raises_tool_exception_after_max_retries` | Always returns 429; `max_retries=3` | Raises `ToolException` after 4 total attempts (1 + 3) ❌ |
| 9 | `test_non_rate_limit_errors_raised_immediately` | 401 Unauthorized error | Re-raises immediately; no retry; called once ❌ |
| 10 | `test_exponential_backoff_timing` | 3 retries with `base_delay=0.1` | Delays follow `0.1 → 0.2 → 0.4` pattern ✅ |
| 11 | `test_max_delay_cap` | `base_delay=100`, `max_delay=0.1`, 10 retries | Total elapsed < 5s (cap prevents huge waits) ✅ |
| 12 | `test_preserves_function_metadata` | Decorated function inspected for `__name__` / `__doc__` | Metadata preserved via `@wraps` ✅ |
| 13 | `test_works_with_arguments` | Decorated function accepts positional + keyword args | Args passed through correctly on retry ✅ |
| 14 | `test_works_with_class_methods` | Decorator applied to instance method | Works with `self`; retries correctly ✅ |

#### Class: `TestBug4056Regression` — Regression tests for bug #4056

| # | Test Method | Scenario | Expected Outcome |
|---|-------------|----------|-----------------|
| 15 | `test_bug_4056_scenario_list_branches` | Simulated `list_branches` hits 429 twice then succeeds | Returns branch list after 3 total calls ✅ |
| 16 | `test_bug_4056_error_message_quality` | Max retries exceeded with 429 | Error message contains `"rate limit"`, `"retries"`, and `"429"` ❌ |
| 17 | `test_bug_4056_recovery_after_backoff` | 4 failures then success within `max_retries=5` | Returns `"recovered"` after 5 total calls ✅ |

---

## Key Behaviours Under Test

| Behaviour | Tests |
|-----------|-------|
| **Rate limit detection** — 429 / "too many requests" / "rate limit" strings | #1, #2, #3, #4, #5 |
| **Retry on 429** — transparent retry until success | #7, #15, #17 |
| **No retry on other errors** — non-429 errors bubble up immediately | #9 |
| **Exponential backoff** — delays double each attempt | #10 |
| **Max delay cap** — delay never exceeds `max_delay` | #11 |
| **Max retries exhausted** — raises descriptive `ToolException` | #8, #16 |
| **Attempt count** — correct number of calls (initial + retries) | #6, #7, #8, #9, #15, #17 |
| **Function transparency** — args, kwargs, metadata preserved | #12, #13, #14 |
| **Bug #4056 regression** — 429 no longer fails immediately | #15, #16, #17 |

---

## Functions Under Test

```
is_rate_limit_error(error: Exception) -> bool
retry_on_rate_limit(max_retries, base_delay, max_delay) -> Callable
```

Both functions are lifted verbatim from `cloud_api_wrapper.py` into a standalone test module to avoid live Bitbucket API dependencies.

### Retry Decision Flow

```
retry_on_rate_limit decorated call
├── attempt N (0 … max_retries)
│   ├── SUCCESS  → return result immediately
│   └── EXCEPTION
│       ├── NOT a rate limit error → re-raise immediately (no retry)
│       └── IS a rate limit error
│           ├── attempt < max_retries → sleep(min(base_delay * 2^attempt, max_delay)) → retry
│           └── attempt == max_retries → raise ToolException("rate limit exceeded after N retries")
```

---

## What Is NOT Yet Tested

| Area | Notes |
|------|-------|
| Live Bitbucket API integration | Requires real credentials; not covered by unit tests |
| `cloud_api_wrapper.py` direct import | Tests use an inline copy of the logic to avoid import-time dependency issues |
| Other Bitbucket toolkit methods (`list_repos`, `create_pr`, etc.) | No unit tests yet |
| Authentication / OAuth token handling | Not covered |
| Cloud vs. Server/DC URL routing | Not covered |
| Pagination handling | Not covered |

---

*Last updated: 2026-08-13*

