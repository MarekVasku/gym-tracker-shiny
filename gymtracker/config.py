from __future__ import annotations
import os, json
from typing import Literal, cast

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

PersistTarget = Literal["sheet", "sqlite", "both"]

def persist_target() -> PersistTarget:
    val = os.environ.get("PERSIST_TARGET", "sheet").lower().strip()
    if val in {"sheet", "sqlite", "both"}:
        return cast(PersistTarget, val)
    return cast(PersistTarget, "sheet")


def sheet_config() -> tuple[str, str]:
    url = os.environ.get("GYM_SHEET_URL", "")
    creds = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not url or not creds:
        raise RuntimeError("Missing GYM_SHEET_URL or GOOGLE_SERVICE_ACCOUNT_JSON for Sheets mode.")
    return url, creds


def db_path() -> str:
    return os.environ.get("DB_PATH", "./gym_tracker.db")
