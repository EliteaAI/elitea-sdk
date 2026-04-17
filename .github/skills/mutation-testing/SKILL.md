---
name: mutation-testing
description: Run cosmic-ray mutation tests against Python modules using Docker workers. Use when asked to check mutation coverage, verify test quality, or find gaps in test assertions. Measures whether tests actually validate logic (not just execute code).
license: Apache-2.0
compatibility: Requires Docker, cosmic-ray v8.4.4+, and Windows with Git Bash (Linux also supported)
metadata:
  author: elitea-sdk
  version: "2.0"
---

# Mutation Testing

Run mutation tests to verify that tests actually **assert** correct behavior, not just execute code.

## CRITICAL Constraints

**This skill must NEVER:**
- Modify source code without explicit developer confirmation
- "Fix" tests by changing source to match assertions
- Apply code changes to resolve failures without being asked

**When baseline tests fail:** Document failures, describe the fix, and **ask the developer** before proceeding.

---

## Quick Start

```bash
# 1. Start workers
docker compose -f docker-compose.mutation.yml up -d

# 2. Create config (cosmic-ray-<module>.toml)
# 3. Initialize and prune
cosmic-ray init cosmic-ray-<module>.toml cr-<module>.sqlite
python .github/skills/mutation-testing/scripts/cr_prune.py cr-<module>.sqlite

# 4. Run
cosmic-ray --verbosity=ERROR baseline cosmic-ray-<module>.toml
cosmic-ray exec cosmic-ray-<module>.toml cr-<module>.sqlite

# 5. Report
python .github/skills/mutation-testing/scripts/cr_survivors.py cr-<module>.sqlite
```

---

## Prerequisites

### One-Time Windows Fix (REQUIRED)

Patch `cosmic_ray/distribution/http.py` to fix backslash paths:

```bash
python -c "import cosmic_ray.distribution.http as m; print(m.__file__)"
```

Change:
```python
"module_path": str(mutation.module_path),
```
To:
```python
"module_path": str(mutation.module_path).replace("\\", "/"),
```

---

## Workflow

### 1. Create Config

Create `cosmic-ray-<module>.toml`:

```toml
[cosmic-ray]
module-path = "elitea_sdk/path/to/module.py"
timeout = 90.0
excluded-modules = []
test-command = "python -m pytest tests/path/to/test_module.py -x -q --no-header --tb=no"

[cosmic-ray.distributor]
name = "http"

[cosmic-ray.distributor.http]
worker-urls = [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
]
```

### 2. Start Workers

```bash
docker compose -f docker-compose.mutation.yml up -d
curl http://localhost:8001  # Expect: 405 (healthy)
```

### 3. Initialize and Prune

```bash
cosmic-ray init cosmic-ray-<module>.toml cr-<module>.sqlite
python .github/skills/mutation-testing/scripts/cr_prune.py cr-<module>.sqlite
```

### 4. Baseline and Execute

```bash
cosmic-ray --verbosity=ERROR baseline cosmic-ray-<module>.toml
cosmic-ray exec cosmic-ray-<module>.toml cr-<module>.sqlite
```

### 5. Report Results

```bash
cr-report cr-<module>.sqlite --show-pending
python .github/skills/mutation-testing/scripts/cr_survivors.py cr-<module>.sqlite
cr-html cr-<module>.sqlite > report-<module>.html
```

---

## Interpreting Results

| Outcome | Meaning |
|---------|---------|
| **KILLED** | Test suite caught the mutation |
| **SURVIVED** | Tests passed with broken logic |
| **INCOMPETENT** | Worker crashed (see [troubleshooting](references/troubleshooting.md)) |

| Score | Interpretation |
|-------|----------------|
| Many survived | Tests execute code but do not assert correctly |
| Few survived | Tests are strong for covered code |
| 100% killed | Covered code is well-tested |

---

## Analyzing Survivors

See [references/survivor-analysis.md](references/survivor-analysis.md) for the full protocol.

**Key rule:** Never write a test purely to kill a mutation. A survivor is a question: Does this logic change matter?

| Situation | Action |
|-----------|--------|
| Intent clear, code correct | Write test asserting correct behavior |
| Intent clear, code wrong | Write failing test, file bug report |
| Intent unclear | Report ambiguity, wait for decision |

---

## Helper Scripts

| Script | Purpose |
|--------|---------|
| [scripts/cr_prune.py](scripts/cr_prune.py) | Prune to logic operators only |
| [scripts/cr_survivors.py](scripts/cr_survivors.py) | Report survivors with line numbers |

---

## References

- [references/troubleshooting.md](references/troubleshooting.md) - Common issues and fixes
- [references/db-schema.md](references/db-schema.md) - SQLite database schema
- [references/survivor-analysis.md](references/survivor-analysis.md) - How to analyze surviving mutations

---

## Cleanup

```bash
docker compose -f docker-compose.mutation.yml down
rm cr-<module>.sqlite cosmic-ray-<module>.toml report-<module>.html
```
