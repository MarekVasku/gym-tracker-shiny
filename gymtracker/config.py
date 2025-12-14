from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, cast

PersistTarget = Literal["sqlite"]

def persist_target() -> PersistTarget:
    """Return the persistence target. Project uses SQLite-only."""
    return cast(PersistTarget, "sqlite")


def db_path() -> str:
    return os.environ.get("DB_PATH", "./gym_tracker.db")


@dataclass
class SheetsConfig:
    spreadsheet_id: str
    worksheet: str
    credentials_file: str | None
    credentials_json: str | None


def sheets_config() -> SheetsConfig | None:
    """Return Google Sheets config if all required values are present.

    Requires either SHEETS_CREDENTIALS_FILE (path) or SHEETS_CREDENTIALS_JSON
    (JSON string), plus SHEETS_SPREADSHEET_ID. The worksheet defaults to
    "Bodyweight" but can be overridden via SHEETS_BODYWEIGHT_SHEET.
    """
    spreadsheet_id = os.environ.get("SHEETS_SPREADSHEET_ID")
    cred_file = os.environ.get("SHEETS_CREDENTIALS_FILE")
    cred_json = os.environ.get("SHEETS_CREDENTIALS_JSON")
    worksheet = os.environ.get("SHEETS_BODYWEIGHT_SHEET", "Bodyweight")

    if not spreadsheet_id or not (cred_file or cred_json):
        return None
    return SheetsConfig(
        spreadsheet_id=spreadsheet_id,
        worksheet=worksheet,
        credentials_file=cred_file,
        credentials_json=cred_json,
    )
