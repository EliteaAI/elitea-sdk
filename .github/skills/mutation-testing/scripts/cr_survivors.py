#!/usr/bin/env python3
"""
cr_survivors.py — Report surviving mutants with source location details.

Usage:
    python .github/skills/mutation-testing/scripts/cr_survivors.py cr-<module>.sqlite

Prints each surviving mutant's line number, column, operator, and function name,
then shows a summary count.
"""

import sqlite3
import sys

if len(sys.argv) < 2:
    print("Usage: python scripts/cr_survivors.py cr-<module>.sqlite")
    sys.exit(1)

db = sys.argv[1]
conn = sqlite3.connect(db)

# Summary
total = conn.execute("SELECT COUNT(*) FROM work_items").fetchone()[0]
done  = conn.execute("SELECT COUNT(*) FROM work_results").fetchone()[0]
outcomes = conn.execute(
    "SELECT worker_outcome, test_outcome, COUNT(*) FROM work_results "
    "GROUP BY worker_outcome, test_outcome"
).fetchall()

print(f"Progress: {done}/{total}")
for r in outcomes:
    print(f"  {r[0]}/{r[1]}: {r[2]}")

# Survivor details
# NOTE: test_outcome values in the DB are UPPERCASE: 'SURVIVED', 'KILLED'
# NOTE: column names are start_pos_row / start_pos_col (NOT start_pos / end_pos)
survivors = conn.execute(
    "SELECT ms.operator_name, ms.occurrence, ms.start_pos_row, ms.start_pos_col, ms.definition_name "
    "FROM mutation_specs ms "
    "JOIN work_items wi ON ms.job_id = wi.job_id "
    "JOIN work_results wr ON wi.job_id = wr.job_id "
    "WHERE wr.test_outcome = 'SURVIVED' "
    "ORDER BY ms.start_pos_row, ms.start_pos_col"
).fetchall()
conn.close()

if survivors:
    print(f"\nSurviving mutants ({len(survivors)}):")
    for s in survivors:
        print(f"  line={s[2]:3d}  col={s[3]:3d}  operator={s[0]}  occurrence={s[1]}  def={s[4]}")
else:
    print("\nAll mutations killed — test suite is strong for covered logic.")
