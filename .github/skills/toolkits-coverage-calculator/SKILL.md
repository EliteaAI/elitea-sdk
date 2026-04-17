---
name: toolkits-coverage-calculator
description: Calculate which toolkit API-wrapper tools have YAML test cases in .elitea/tests/test_pipelines/suites/, update test_coverage.md, and identify untested tools. Use ONLY for toolkit-level tool coverage (counting tools vs test cases). Do NOT use for running coverage.py or measuring Python code execution - use pytest-coverage for that.
license: Apache-2.0
compatibility: Requires access to elitea_sdk/tools/ and .elitea/tests/test_pipelines/
metadata:
  author: elitea-sdk
  version: "2.0"
---

# Coverage Calculator

Analyze test coverage for ELITEA SDK toolkits by counting tools vs test cases.

## When to Use

- Calculating coverage metrics for specific toolkits
- Updating the test coverage report
- Identifying coverage gaps and untested tools
- Tracking coverage trends over time

**NOT for:** Python code execution coverage (use pytest-coverage skill instead)

---

## Key Paths

| Path | Purpose |
|------|---------|
| `elitea_sdk/tools/` | Toolkit source code |
| `.elitea/tests/test_pipelines/suites/` | Test suites (YAML) |
| `.elitea/tests/test_pipelines/test_coverage.md` | Coverage report |

---

## Quick Workflow

1. **Scan** `elitea_sdk/tools/` for toolkit directories
2. **Categorize** each toolkit (user-facing vs framework utility)
3. **Count tools** from `get_available_tools()` in each wrapper
4. **Count tests** from YAML files in test suites
5. **Calculate** coverage percentages
6. **Update** report with timestamp

---

## Coverage Formulas

**Toolkit Coverage:**
```
Coverage % = (Tested Tools / Total Tools) x 100
```

**Overall Coverage:**
```
Overall Coverage % = (Toolkits with Tests / Total User-Facing Toolkits) x 100
```

---

## Procedures

| Task | Reference |
|------|-----------|
| Count tools in a toolkit | [references/count-tools.md](references/count-tools.md) |
| Count test cases in a suite | [references/count-tests.md](references/count-tests.md) |
| Categorize a toolkit | [references/categorize-toolkit.md](references/categorize-toolkit.md) |
| Toolkit category lists | [references/toolkit-categories.md](references/toolkit-categories.md) |

---

## Report Sections

The coverage report at `.elitea/tests/test_pipelines/test_coverage.md` should contain:

1. **Executive Summary** - High-level metrics table
2. **Toolkits With Tests** - Detailed coverage with status indicators
3. **Toolkits Without Tests** - Organized by priority
4. **Framework Utilities** - List (no coverage expected)
5. **Coverage Trend** - Historical data with dates
6. **Recommendations** - Next steps

---

## Key Principles

- **Accuracy**: All counts must match actual files (never estimate)
- **Verification**: Cross-check source against test suites
- **Separation**: Keep user-facing toolkits separate from utilities
- **Timestamp**: Include date on all report updates

---

## Example Output

```json
{
  "toolkit": "github",
  "tool_count": 15,
  "test_count": 12,
  "coverage": "80.0%",
  "untested_tools": ["create_gist", "delete_gist", "fork_repo"]
}
```
