#!/usr/bin/env python3
"""
cr_prune.py — Prune a cosmic-ray SQLite session to logic operators only.

Usage:
    python .github/skills/mutation-testing/scripts/cr_prune.py cr-<module>.sqlite

Keeps only boolean/comparison operators that matter for logic coverage and
removes all arithmetic, string, and other noise operators.
"""

import sqlite3
import sys

KEEP = {
    "core/AddNot",
    "core/ReplaceTrueWithFalse",
    "core/ReplaceFalseWithTrue",
    "core/ReplaceAndWithOr",
    "core/ReplaceOrWithAnd",
    "core/ReplaceComparisonOperator_Eq_NotEq",
    "core/ReplaceComparisonOperator_NotEq_Eq",
    "core/ReplaceComparisonOperator_Is_IsNot",
    "core/ReplaceComparisonOperator_IsNot_Is",
}

if len(sys.argv) < 2:
    print("Usage: python scripts/cr_prune.py cr-<module>.sqlite")
    sys.exit(1)

db = sys.argv[1]
conn = sqlite3.connect(db)
before = conn.execute("SELECT COUNT(*) FROM work_items").fetchone()[0]

placeholders = ",".join(f"'{o}'" for o in KEEP)
conn.execute(f"""
    DELETE FROM work_items WHERE job_id IN (
        SELECT job_id FROM mutation_specs
        WHERE operator_name NOT IN ({placeholders})
    )
""")
conn.execute(f"""
    DELETE FROM mutation_specs WHERE operator_name NOT IN ({placeholders})
""")
conn.commit()

after = conn.execute("SELECT COUNT(*) FROM work_items").fetchone()[0]
print(f"Pruned: {before} → {after} mutations remaining")
conn.close()
