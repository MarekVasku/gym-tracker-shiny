# Gym Tracker (Shiny for Python)

A compact, local-first gym tracking app built with Shiny for Python. Track Big 3 lifts, bodyweight (with timestamp), and a small set of body measurements. Designed for data scientists who want a clean UI and simple persistence (Google Sheets, SQLite, or both).

Table of Contents
-----------------
- [Quick start](#quick-start)
- [Environment variables](#environment-variables)
- [Schema migration](#schema-migration-removing-legacy-measurement-columns)
- [Development & tests](#development--tests)
- [License](#license)

Quick start
-----------

Create a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app locally (development reload enabled):

```bash
shiny run --reload app.py
```

By default the app will look for `DB_PATH` (SQLite) or a Google Sheets config depending on `PERSIST_TARGET` (see below).

Environment variables
---------------------
Provide these as needed for your chosen persistence backend. Example (bash):

```bash
# Use sqlite, sheet, or both
export PERSIST_TARGET=sqlite
# Path to sqlite DB (default: ./gym_tracker.db)
export DB_PATH=./gym_tracker.db

# If using Google Sheets (set PERSIST_TARGET=sheet or both):
export GYM_SHEET_URL='https://docs.google.com/...'
export GOOGLE_SERVICE_ACCOUNT_JSON='{"type":"service_account", ... }'
```

Schema migration: removing legacy measurement columns
---------------------------------------------------
If you have an older SQLite DB that still includes legacy columns (for example `hips_cm`, `arm_cm`, or `notes`) in the `Measurements` table, a safe migration helper is included at `scripts/migrate_drop_measurement_columns.py`.

Dry-run (no changes):

```bash
python scripts/migrate_drop_measurement_columns.py --db ./gym_tracker.db
```

Apply migration (creates a timestamped backup first):

```bash
python scripts/migrate_drop_measurement_columns.py --apply --db ./gym_tracker.db
```

The migration recreates the `Measurements` table to match the fields defined in `gymtracker.utils.REQUIRED_TABS` (currently: `id, date, neck_cm, shoulder_cm, chest_cm, waist_cm, biceps_cm, thigh_cm, calf_cm`). The script will back up the DB to `./gym_tracker.db.bak-<timestamp>` before altering data.

Development & tests
-------------------

- Tests (if added) can be run with pytest. Example:

```bash
pytest -q
```

- The project provides both `pyproject.toml` and `requirements.txt`. Use whichever suits your workflow (poetry/uv/venv/pip).

License
-------

MIT-style (adjust as needed).

If you'd like, I can also add a small `CONTRIBUTING.md` and a few GitHub-friendly badges (CI, python version) — tell me which services you use and I'll add them.
