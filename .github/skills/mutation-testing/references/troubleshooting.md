# Troubleshooting

Common issues and solutions for cosmic-ray mutation testing on Windows with Docker workers.

---

## `TypeError: Arguments provided for operator X which accepts no arguments`

**Cause:** Listing no-arg operators in `[cosmic-ray.operators]` with any value (even `[]` or `[{}]`).

**Fix:** Remove the `[cosmic-ray.operators]` section entirely. No-arg operators run automatically.

---

## `EXCEPTION/INCOMPETENT: N` — all mutations fail with `FileNotFoundError`

**Symptom:** Error like `FileNotFoundError: alita_sdk\runtime\...\module.py`

**Cause:** Windows backslash path bug. `str(mutation.module_path)` on Windows produces backslash paths; Linux workers receive them via HTTP and can't resolve them.

**Fix:** Patch one line in `venv/Lib/site-packages/cosmic_ray/distribution/http.py`:

```python
# Find the file
python -c "import cosmic_ray.distribution.http as m; print(m.__file__)"

# Change this line:
"module_path": str(mutation.module_path),

# To:
"module_path": str(mutation.module_path).replace("\\", "/"),
```

> Patching SQLite paths is **not sufficient** — `work_db.py` re-wraps every path via `pathlib.Path()` on every read, restoring backslashes before the HTTP layer sees them.

---

## `405 Method Not Allowed` from `curl http://localhost:8001`

**This is correct behavior.** Cosmic-ray HTTP workers only accept POST requests. The worker is healthy.

---

## Workers show 0 results after `exec`

**Symptom:** The `work_results` table is empty.

**Checklist:**
1. Are containers running? → `docker ps`
2. Can the coordinator reach workers? → `curl http://localhost:8001` should return `405`
3. Is the `http.py` patch applied? → If not, all mutations will be `INCOMPETENT`

---

## Inline Python (`-c` or `<<'EOF'`) produces no output on Windows Git Bash

**Cause:** Git Bash on Windows does **not** reliably support Python heredocs or multi-line `-c` string commands. The command appears to run but produces no output, or the heredoc is never terminated.

**Fix:** Always write Python analysis/query code to a `.py` script file and run it with `python script.py`. Never use `python -c "...multiline..."` or `python - <<'EOF'` in the terminal.

The helper scripts in `scripts/` (`cr_prune.py`, `cr_survivors.py`) exist for this reason.

---

## `sqlite3.OperationalError: no such column: ms.start_pos`

**Cause:** The column names are different than expected.

**Fix:** Use the correct column names. See [db-schema.md](db-schema.md) for the full schema.

The `mutation_specs` table uses:
- `start_pos_row` and `start_pos_col` (NOT `start_pos` / `end_pos`)

---

## SQLite query for `test_outcome = 'survived'` returns 0 rows

**Cause:** Outcome values are UPPERCASE in the database.

**Fix:** Use uppercase in WHERE clauses:
```sql
WHERE wr.test_outcome = 'SURVIVED'
```

Valid values: `'SURVIVED'`, `'KILLED'`, `'INCOMPETENT'`
