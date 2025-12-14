# Gym Tracker (Shiny for Python)

Gym Tracker is a compact, local-first application built with Shiny for Python. It records Big 3 lifts, timestamped bodyweight entries, and a focused set of body measurements. Persistence uses SQLite (local file) only.

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

# Optional: install developer tools
pip install -r requirements.txt pre-commit mypy ruff isort black
```

Run the app locally (development reload enabled):

```bash
shiny run --reload app.py
```

By default the app will look for `DB_PATH` (SQLite).

Environment variables
---------------------
Configure the SQLite database path via environment variable. Example (bash):

```bash
# Path to sqlite DB (default: ./gym_tracker.db)
export DB_PATH=./gym_tracker.db

Enable Google Sheets mirror (Bodyweight only)
--------------------------------------------
If you want bodyweight entries to also write to a Google Sheet, provide a service account and target sheet:

```bash
export SHEETS_SPREADSHEET_ID=<your-spreadsheet-id>
export SHEETS_BODYWEIGHT_SHEET=Sheet1       # optional, defaults to Bodyweight

# Either point to a service account file...
export SHEETS_CREDENTIALS_FILE=/path/to/creds.json
# ...or inline the JSON (use with care)
# export SHEETS_CREDENTIALS_JSON='{"type": "service_account", ...}'
```

On startup the app will keep SQLite as primary storage and best-effort mirror Bodyweight writes to the sheet.
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

- Quick dev loop with Makefile:

```bash
make install      # create venv and install deps
make test         # run tests
make coverage     # run tests with coverage report
make run          # start the Shiny app with reload
```

- Or run pytest directly:

```bash
pytest -q
```

Pre-commit
---------
Install pre-commit hooks locally and run them before commits:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

- The project provides both `pyproject.toml` and `requirements.txt`. Use whichever suits your workflow (poetry/uv/venv/pip).

License
-------

MIT-style (adjust as needed).
