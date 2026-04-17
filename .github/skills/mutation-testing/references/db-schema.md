# Cosmic-Ray SQLite Database Schema

Reference for the cosmic-ray session database (`cr-*.sqlite`).

---

## Tables

### `mutation_specs`

Stores mutation definitions.

| Column | Description |
|--------|-------------|
| `module_path` | Path to the Python module |
| `operator_name` | Mutation operator (e.g., `core/AddNot`) |
| `operator_args` | Arguments for parameterized operators |
| `occurrence` | Which occurrence of the operator in the file |
| `start_pos_row` | Line number (1-based) |
| `start_pos_col` | Column number |
| `end_pos_row` | End line number |
| `end_pos_col` | End column number |
| `definition_name` | Function/method name containing the mutation |
| `job_id` | Unique job identifier |

### `work_items`

Tracks mutation jobs.

| Column | Description |
|--------|-------------|
| `job_id` | Unique job identifier (FK to `mutation_specs`) |

### `work_results`

Stores execution results.

| Column | Description |
|--------|-------------|
| `job_id` | FK to `work_items` |
| `worker_outcome` | Worker status: `'NORMAL'`, `'EXCEPTION'`, `'TIMEOUT'` |
| `test_outcome` | Test result: `'KILLED'`, `'SURVIVED'`, `'INCOMPETENT'` |

---

## Important Notes

1. **Outcome values are UPPERCASE** — queries like `WHERE test_outcome = 'survived'` return nothing
2. **Position columns** are `start_pos_row`/`start_pos_col` — there is no `start_pos` or `end_pos`
3. **Job IDs** link all three tables together via `job_id`

---

## Example Queries

### Count mutations by status
```sql
SELECT worker_outcome, test_outcome, COUNT(*)
FROM work_results
GROUP BY worker_outcome, test_outcome;
```

### Get surviving mutants with locations
```sql
SELECT ms.operator_name, ms.start_pos_row, ms.start_pos_col, ms.definition_name
FROM mutation_specs ms
JOIN work_items wi ON ms.job_id = wi.job_id
JOIN work_results wr ON wi.job_id = wr.job_id
WHERE wr.test_outcome = 'SURVIVED'
ORDER BY ms.start_pos_row;
```

### Check progress
```sql
SELECT
    (SELECT COUNT(*) FROM work_items) AS total,
    (SELECT COUNT(*) FROM work_results) AS done;
```
