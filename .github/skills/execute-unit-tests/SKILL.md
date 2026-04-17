---
name: "execute-unit-tests"
description: "Interpret user prompts and run only the relevant pytest unit tests for the Alita SDK, covering all unit, document loader, and integration test suites"
---

# Unit Test Executor Skill

This skill interprets natural-language prompts and maps them to specific pytest invocations covering the `tests/` directory of the Alita SDK. It covers all runtime unit tests, parametrized document loader tests, and integration tests.

Always run only the tests that match the user's intent — never the full suite unless explicitly asked.

---

## When to Use This Skill

- Running unit, loader, or integration tests in `tests/`
- Re-running only tests related to a specific change (impact analysis via tags)
- Diagnosing test failures (including loader baseline mismatches)
- Updating or regenerating baseline output files for document loaders
- Adding new test cases or new test suites following the established pattern

---

## Workflow

1. **Parse** the user's prompt to determine scope (file, class, method, keyword, tag).
2. **Map** keywords to concrete test paths — see [references/keyword-mapping.md](references/keyword-mapping.md).
3. **Build** the minimal pytest command that covers only the requested tests.
4. **Run** with `runTests` tool (preferred) or terminal fallback.
5. **Report** results and suggest next steps on failures — see [references/decision-tree.md](references/decision-tree.md).

---

## Prerequisites

Always activate the virtualenv before running in terminal:

```bash
# Windows (Git Bash / bash)
source venv/Scripts/activate

# Linux / macOS
source venv/bin/activate
```

---

## Running Tests

### Preferred method — `runTests` tool

Use the `runTests` tool when possible. Pass the absolute file path(s) and optionally the test names.
See [references/keyword-mapping.md](references/keyword-mapping.md) for `runTests` examples by prompt type.

### Terminal commands

```bash
# All unit tests (no credentials needed)
python -m pytest tests/runtime/ -v

# All document loader tests
python -m pytest tests/runtime/langchain/document_loaders/ -v

# Single test file
python -m pytest tests/runtime/test_blocked_tools.py -v

# Specific class
python -m pytest tests/runtime/test_alita_llm.py::TestMaxRetriesExceededError -v

# Specific method
python -m pytest tests/runtime/test_logging_utils.py::test_evaluate_template_context_variables -v

# Keyword filter
python -m pytest tests/runtime/ -k "token" -v

# Multiple files
python -m pytest tests/runtime/test_alita_llm.py tests/runtime/test_prompt_client.py -v

# Stop on first failure
python -m pytest tests/runtime/test_blocked_tools.py -x -v

# Re-run last failed only
python -m pytest tests/runtime/ --lf -v
```

### Loader tests — tag filtering

Document loader tests support pytest marker filtering for impact analysis:

```bash
# By loader
python -m pytest tests/runtime/langchain/document_loaders/ -m "loader_csv" -v

# By feature
python -m pytest tests/runtime/langchain/document_loaders/ -m "feature_chunking" -v

# Combined
python -m pytest tests/runtime/langchain/document_loaders/ -m "loader_csv and feature_chunking" -v

# Skip slow tests
python -m pytest tests/runtime/langchain/document_loaders/ -m "not performance" -v
```

Full tag reference and all filter examples → [references/loader-tags-and-filtering.md](references/loader-tags-and-filtering.md)

---

## References

| Reference | Contents |
|-----------|----------|
| [test-inventory.md](references/test-inventory.md) | Full test file inventory — runtime unit tests, loader tests, integration tests, and key paths |
| [keyword-mapping.md](references/keyword-mapping.md) | Keyword → test path mapping table; `runTests` tool examples; ambiguous prompt heuristics |
| [loader-tags-and-filtering.md](references/loader-tags-and-filtering.md) | Tag reference table; filter commands by loader, feature, content type, and combined expressions |
| [loader-structure.md](references/loader-structure.md) | Repeatable test suite structure; loader name → test file mapping; adding new test cases and suites |
| [baseline-management.md](references/baseline-management.md) | Regenerating single and bulk baselines; diagnosing loader test failures |
| [reportportal.md](references/reportportal.md) | ReportPortal setup, credentials, and upload commands |
| [decision-tree.md](references/decision-tree.md) | Decision tree for ambiguous prompts; post-run action guide; pytest flags reference |
