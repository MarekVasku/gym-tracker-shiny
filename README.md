# Gym Tracker (Shiny for Python) — multi-file layout
# --------------------------------------------------
# Features
# - Track Big 3 lifts, bodyweight, and measurements
# - Edit/Delete records in-app (stable UUID row IDs)
# - Pluggable persistence: Google Sheets, SQLite, or both (write-through)
#
# Quick start
# 1) `python -m venv .venv && source .venv/bin/activate` (or uv/poetry)
# 2) `pip install -r requirements.txt`
# 3) export env vars (examples below)
# 4) `shiny run --reload app.py`
#
# Env vars
#   PERSIST_TARGET=sheet|sqlite|both
#   GYM_SHEET_URL=https://docs.google.com/...
#   GOOGLE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'
#   DB_PATH=./gym_tracker.db
#
# Deploy: shinyapps.io / Posit Connect (set secrets); for SQLite, mount a volume for DB_PATH.

## Schema migration: removing legacy measurement columns

If you have older databases that still contain legacy columns (for example `hips_cm`, `arm_cm`, or `notes`) in the `Measurements` table, a safe migration helper is provided.

Usage (dry-run):

```bash
python scripts/migrate_drop_measurement_columns.py --db ./gym_tracker.db
```

To apply the migration (will create a backup):

```bash
python scripts/migrate_drop_measurement_columns.py --apply --db ./gym_tracker.db
```

The script will back up your DB to `./gym_tracker.db.bak-<timestamp>` before making changes. It reconstructs the `Measurements` table to match the expected columns in `gymtracker.utils.REQUIRED_TABS`.
