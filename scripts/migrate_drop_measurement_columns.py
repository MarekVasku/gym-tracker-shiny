"""Migration helper: drop legacy columns from Measurements table safely.

Usage:
  python scripts/migrate_drop_measurement_columns.py --dry-run
  python scripts/migrate_drop_measurement_columns.py --apply

The script will:
 - Inspect the Measurements table columns
 - Report extra/legacy columns (e.g., hips_cm, arm_cm, notes)
 - By default (dry-run) only print actions
 - With --apply it will:
     * backup the DB file to DB_PATH.bak-<timestamp>
     * create a temp table with the desired schema
     * copy data for the desired columns
     * drop the old table and rename the temp table

This is SQLite-friendly and does not require external libs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
from datetime import datetime

from gymtracker.config import db_path as config_db_path
from gymtracker.utils import REQUIRED_TABS

WANTED = REQUIRED_TABS["Measurements"]

def col_type_for(col: str) -> str:
    if col == "id":
        return "TEXT PRIMARY KEY"
    if col.endswith("_cm") or col.endswith("_kg"):
        return "REAL"
    if col in {"reps"}:
        return "INTEGER"
    return "TEXT"


def inspect_db(path: str) -> list[str]:
    with sqlite3.connect(path) as con:
        cur = con.cursor()
        cur.execute("PRAGMA table_info(Measurements)")
        rows = cur.fetchall()
    return [r[1] for r in rows]


def backup_db(path: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest = f"{path}.bak-{ts}"
    shutil.copy2(path, dest)
    return dest


def apply_migration(path: str, existing_cols: list[str]):
    # Build CREATE TABLE statement for temp table
    col_defs = ",\n    ".join([f"{c} {col_type_for(c)}" for c in WANTED])
    create_sql = f"CREATE TABLE Measurements_new (\n    {col_defs}\n)"

    want_cols_csv = ",".join(WANTED)
    want_cols_select = ",".join([c for c in WANTED if c in existing_cols])

    with sqlite3.connect(path) as con:
        cur = con.cursor()
        cur.execute(create_sql)
        # Copy data for columns that exist in the old table into the new table
        cur.execute(f"INSERT INTO Measurements_new ({want_cols_csv}) SELECT {want_cols_select} FROM Measurements")
        cur.execute("DROP TABLE Measurements")
        cur.execute("ALTER TABLE Measurements_new RENAME TO Measurements")
        con.commit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="Perform the migration. Without this flag a dry-run is shown.")
    p.add_argument("--db", type=str, default=None, help="Path to SQLite DB (overrides DB_PATH env)")
    args = p.parse_args()

    path = args.db or config_db_path()
    if not os.path.exists(path):
        print(f"DB not found at {path}")
        return 1

    existing = inspect_db(path)
    print("Existing Measurements columns:", existing)
    extras = [c for c in existing if c not in WANTED]
    missing = [c for c in WANTED if c not in existing]
    print("Desired columns:", WANTED)
    print("Missing desired columns (will be created):", missing)
    print("Extra/legacy columns (candidate for removal):", extras)

    if not extras:
        print("No legacy columns detected. Nothing to do.")
        return 0

    if not args.apply:
        print("Dry-run: nothing changed. Re-run with --apply to perform migration.")
        return 0

    # apply
    bak = backup_db(path)
    print(f"Backed up DB to: {bak}")
    apply_migration(path, existing)
    print("Migration complete. Measurements table now matches REQUIRED_TABS.")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
