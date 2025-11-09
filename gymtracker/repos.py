from __future__ import annotations
import uuid, json, sqlite3
from datetime import date, datetime
from dataclasses import dataclass
from typing import Protocol
import pandas as pd
from pydantic import ValidationError

from .utils import REQUIRED_TABS
from .config import db_path
from .logger import logger
from .models import LiftEntry, BodyweightEntry, MeasurementEntry, InBodyEntry

class Repo(Protocol):
    def read_df(self, tab: str) -> pd.DataFrame: ...
    def append(self, tab: str, payload: dict) -> str: ...
    def update(self, tab: str, row_id: str, payload: dict) -> None: ...
    def delete(self, tab: str, row_id: str) -> None: ...


# Google Sheets support removed — this project now uses SQLite only.


class SQLiteRepo:
    def __init__(self, path: str | None = None):
        self.path = path or db_path()
        self._init_db()
        logger.info(f"SQLiteRepo initialized at {self.path}")

    def _conn(self):
        return sqlite3.connect(self.path, check_same_thread=False)

    def _init_db(self):
        with self._conn() as con:
            cur = con.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS Lifts (
                    id TEXT PRIMARY KEY,
                    date TEXT,
                    exercise TEXT,
                    weight_kg REAL,
                    reps INTEGER,
                    notes TEXT
                )""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS Bodyweight (
                    id TEXT PRIMARY KEY,
                    date TEXT,
                    time TEXT,
                    weight_kg REAL,
                    notes TEXT
                )""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS Measurements (
                    id TEXT PRIMARY KEY,
                    date TEXT,
                    weight_kg REAL,
                    neck_cm REAL,
                    shoulder_cm REAL,
                    chest_cm REAL,
                    waist_cm REAL,
                    biceps_cm REAL,
                    thigh_cm REAL,
                    calf_cm REAL
                )""")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS InBody (
                    id TEXT PRIMARY KEY,
                    date TEXT,
                    inbody_score REAL,
                    weight_kg REAL,
                    skeletal_muscle_kg_total REAL,
                    body_fat_kg_total REAL,
                    body_fat_percent REAL,
                    visceral_fat_level REAL,
                    bmr_kcal REAL,
                    muscle_right_arm_kg REAL,
                    muscle_left_arm_kg REAL,
                    muscle_trunk_kg REAL,
                    muscle_right_leg_kg REAL,
                    muscle_left_leg_kg REAL,
                    fat_right_arm_kg REAL,
                    fat_left_arm_kg REAL,
                    fat_trunk_kg REAL,
                    fat_right_leg_kg REAL,
                    fat_left_leg_kg REAL,
                    notes TEXT
                )""")
            # Ensure any missing columns are added to existing tables (simple migrations)
            def ensure_columns(table: str, required: list[str]):
                cur.execute(f"PRAGMA table_info({table})")
                existing = {row[1] for row in cur.fetchall()}
                for col in required:
                    if col not in existing:
                        # choose a type (REAL for *_cm/_kg, INTEGER for reps, TEXT otherwise)
                        if col.endswith("_cm") or col.endswith("_kg"):
                            typ = "REAL"
                        elif col in {"reps"}:
                            typ = "INTEGER"
                        else:
                            typ = "TEXT"
                        try:
                            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
                        except Exception:
                            pass

            ensure_columns("Lifts", REQUIRED_TABS["Lifts"])
            ensure_columns("Bodyweight", REQUIRED_TABS["Bodyweight"])
            ensure_columns("Measurements", REQUIRED_TABS["Measurements"])
            ensure_columns("InBody", REQUIRED_TABS["InBody"])

    def read_df(self, tab: str) -> pd.DataFrame:
        logger.debug(f"SQLite read from table: {tab}")
        with self._conn() as con:
            df = pd.read_sql_query(f"SELECT * FROM {tab}", con)
        if df.empty:
            logger.debug(f"SQLite table {tab} is empty")
            return pd.DataFrame(columns=REQUIRED_TABS[tab])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        logger.debug(f"SQLite read {len(df)} rows from {tab}")
        return df

    def append(self, tab: str, payload: dict) -> str:
        logger.debug(f"SQLite append to {tab}: {payload}")
        # Validate on insert
        _validate_payload(tab, payload)

        row_id = payload.get("id") or str(uuid.uuid4())
        payload = {**payload, "id": row_id}
        cols = REQUIRED_TABS[tab]
        placeholders = ",".join(["?"] * len(cols))
        # Serialize date/datetime values to ISO format to avoid sqlite3 date adapter deprecation
        values = []
        for c in cols:
            v = payload.get(c)
            if isinstance(v, (date, datetime)):
                values.append(v.isoformat())
            else:
                values.append(v)
        with self._conn() as con:
            con.execute(
                f"INSERT OR REPLACE INTO {tab} ({','.join(cols)}) VALUES ({placeholders})",
                tuple(values),
            )
        logger.info(f"SQLite appended row id={row_id} to {tab}")
        return row_id

    def update(self, tab: str, row_id: str, payload: dict) -> None:
        logger.debug(f"SQLite update {tab} id={row_id} with {payload}")
        sets = [f"{k} = ?" for k in payload.keys() if k != "id"]
        if not sets:
            logger.debug("No fields provided for update; skipping")
            return
        # Serialize date/datetime values to ISO format when updating
        params = []
        for k in (k for k in payload.keys() if k != "id"):
            v = payload[k]
            if isinstance(v, (date, datetime)):
                params.append(v.isoformat())
            else:
                params.append(v)
        params = tuple(params) + (row_id,)
        with self._conn() as con:
            con.execute(
                f"UPDATE {tab} SET {', '.join(sets)} WHERE id = ?",
                params,
            )
        logger.info(f"SQLite updated {tab} id={row_id}")

    def delete(self, tab: str, row_id: str) -> None:
        logger.debug(f"SQLite delete from {tab} id={row_id}")
        with self._conn() as con:
            con.execute(f"DELETE FROM {tab} WHERE id = ?", (row_id,))
        logger.info(f"SQLite deleted {tab} id={row_id}")


class CombinedRepo:
    def __init__(self, primary: Repo, secondary: Repo):
        self.primary = primary
        self.secondary = secondary

    def read_df(self, tab: str) -> pd.DataFrame:
        return self.primary.read_df(tab)

    def append(self, tab: str, payload: dict) -> str:
        row_id = self.primary.append(tab, payload)
        try:
            self.secondary.append(tab, {**payload, "id": row_id})
        except Exception:
            logger.warning(f"Secondary append failed for {tab} id={row_id}", exc_info=True)
        return row_id

    def update(self, tab: str, row_id: str, payload: dict) -> None:
        self.primary.update(tab, row_id, payload)
        try:
            self.secondary.update(tab, row_id, payload)
        except Exception:
            logger.warning(f"Secondary update failed for {tab} id={row_id}", exc_info=True)

    def delete(self, tab: str, row_id: str) -> None:
        self.primary.delete(tab, row_id)
        try:
            self.secondary.delete(tab, row_id)
        except Exception:
            logger.warning(f"Secondary delete failed for {tab} id={row_id}", exc_info=True)


def repo_factory() -> Repo:
    """Return a SQLite-only repository (Sheets support removed)."""
    return SQLiteRepo()


# --- Validation helper ---
def _validate_payload(tab: str, payload: dict) -> None:
    """Validate payload using Pydantic models for inserts.

    Raises ValidationError if invalid. For updates we skip strict validation
    as partial updates may not include required fields.
    """
    model_map = {
        "Lifts": LiftEntry,
        "Bodyweight": BodyweightEntry,
        "Measurements": MeasurementEntry,
        "InBody": InBodyEntry,
    }
    model = model_map.get(tab)
    if not model:
        logger.debug(f"No validation model registered for tab {tab}")
        return
    try:
        model(**payload)  # type: ignore[arg-type]
    except ValidationError:
        logger.exception(f"Validation failed for tab={tab} payload={payload}")
        raise
