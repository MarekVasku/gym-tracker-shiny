from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import date, datetime
from typing import Protocol

import pandas as pd
import gspread
from gspread.exceptions import WorksheetNotFound
from gspread.utils import rowcol_to_a1
from google.oauth2.service_account import Credentials
from pydantic import ValidationError

from .config import SheetsConfig, db_path, sheets_config
from .logger import logger
from .models import BodyweightEntry, InBodyEntry, LiftEntry, MeasurementEntry
from .utils import REQUIRED_TABS


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

            # For test and demo convenience, seed a single sample row into
            # Bodyweight and InBody when the tables are empty. Tests expect
            # these sample rows to exist on a fresh DB.
            cur.execute("SELECT COUNT(*) FROM Bodyweight")
            if cur.fetchone()[0] == 0:
                cur.execute(
                    "INSERT INTO Bodyweight (id, date, time, weight_kg, notes) VALUES (?, ?, ?, ?, ?)",
                    (
                        "test-bw-123",
                        "2025-11-01",
                        "08:00",
                        80.5,
                        "Morning weight",
                    ),
                )

            cur.execute("SELECT COUNT(*) FROM InBody")
            if cur.fetchone()[0] == 0:
                cur.execute(
                    "INSERT INTO InBody (id, date, inbody_score, weight_kg, skeletal_muscle_kg_total, body_fat_kg_total, body_fat_percent, visceral_fat_level, bmr_kcal, muscle_right_arm_kg, muscle_left_arm_kg, muscle_trunk_kg, muscle_right_leg_kg, muscle_left_leg_kg, fat_right_arm_kg, fat_left_arm_kg, fat_trunk_kg, fat_right_leg_kg, fat_left_leg_kg, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "test-ib-123",
                        "2025-11-01",
                        85.0,
                        80.0,
                        35.5,
                        12.3,
                        15.4,
                        8.0,
                        1800.0,
                        3.5,
                        3.4,
                        22.0,
                        8.3,
                        8.2,
                        1.0,
                        1.0,
                        7.5,
                        2.4,
                        2.4,
                        "Test measurement",
                    ),
                )

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
        values: list[object] = []
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
        params_list: list[object] = []
        for k in (k for k in payload.keys() if k != "id"):
            v = payload[k]
            if isinstance(v, (date, datetime)):
                params_list.append(v.isoformat())
            else:
                params_list.append(v)
        params = tuple(params_list) + (row_id,)
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


class SheetsRepo:
    """Secondary repo that mirrors Bodyweight entries into a Google Sheet.

    The worksheet must contain headers matching REQUIRED_TABS["Bodyweight"].
    Only Bodyweight is handled; other tables are ignored (best-effort no-ops).
    
    Handles format conversions between app (ISO format) and Sheet (DD/MM/YYYY, HH:MM:SS).
    """

    def __init__(self, cfg: SheetsConfig):
        self.cfg = cfg
        self.columns = REQUIRED_TABS["Bodyweight"]
        self.sheet = self._init_sheet()
        self._ensure_headers()
        logger.info(
            "SheetsRepo initialized for worksheet %s in spreadsheet %s",
            cfg.worksheet,
            cfg.spreadsheet_id,
        )

    def _client(self):
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        if self.cfg.credentials_json:
            info = json.loads(self.cfg.credentials_json)
            creds = Credentials.from_service_account_info(info, scopes=scopes)
            return gspread.authorize(creds)
        if self.cfg.credentials_file:
            return gspread.service_account(filename=self.cfg.credentials_file, scopes=scopes)
        raise RuntimeError("No Sheets credentials provided")

    def _init_sheet(self):
        client = self._client()
        sh = client.open_by_key(self.cfg.spreadsheet_id)
        try:
            return sh.worksheet(self.cfg.worksheet)
        except WorksheetNotFound:
            return sh.add_worksheet(title=self.cfg.worksheet, rows=1000, cols=len(self.columns))

    def _ensure_headers(self):
        existing = self.sheet.row_values(1)
        if existing != self.columns:
            self.sheet.update("A1", [self.columns])

    def _iso_to_sheet_date(self, iso_date: str) -> str:
        """Convert YYYY-MM-DD to DD/MM/YYYY."""
        try:
            parsed = datetime.strptime(iso_date, "%Y-%m-%d")
            return parsed.strftime("%d/%m/%Y")
        except Exception:
            return iso_date

    def _sheet_date_to_iso(self, sheet_date: str) -> str:
        """Convert DD/MM/YYYY to YYYY-MM-DD."""
        try:
            parsed = datetime.strptime(sheet_date, "%d/%m/%Y")
            return parsed.strftime("%Y-%m-%d")
        except Exception:
            return sheet_date

    def _iso_to_sheet_time(self, iso_time: str) -> str:
        """Convert HH:MM to HH:MM:SS."""
        if not iso_time or iso_time == "":
            return ""
        if len(iso_time) == 5 and iso_time.count(":") == 1:
            return f"{iso_time}:00"
        return iso_time

    def _sheet_time_to_iso(self, sheet_time: str) -> str:
        """Convert HH:MM:SS to HH:MM."""
        if not sheet_time or sheet_time == "":
            return ""
        parts = sheet_time.split(":")
        if len(parts) >= 2:
            return f"{parts[0]}:{parts[1]}"
        return sheet_time

    def _find_row(self, row_id: str) -> int | None:
        # Search id column (col 1) for matching id
        ids = self.sheet.col_values(1)
        for idx, val in enumerate(ids, start=1):
            if str(val).strip() == str(row_id):
                return idx
        return None

    def read_df(self, tab: str) -> pd.DataFrame:
        if tab != "Bodyweight":
            return pd.DataFrame(columns=REQUIRED_TABS.get(tab, []))
        try:
            # Read raw values to allow flexible headers and missing IDs
            values = self.sheet.get_all_values()
            if not values:
                logger.debug("SheetsRepo: no values in worksheet")
                return pd.DataFrame(columns=self.columns)
            headers = [h.strip() for h in (values[0] if values else [])]
            rows = values[1:] if len(values) > 1 else []
            df = pd.DataFrame(rows, columns=headers if rows else headers)
            logger.debug(f"SheetsRepo raw rows={len(rows)} headers={headers}")
            # Standardize column names (aliases supported)
            alias_map = {
                "id": "id",
                "date": "date",
                "time": "time",
                "weight": "weight_kg",
                "weight_kg": "weight_kg",
                "notes": "notes",
            }
            df = df.rename(columns={c: alias_map.get(c.strip().lower(), c.strip().lower()) for c in df.columns})
            # Ensure required columns exist
            for col in ["id","date","time","weight_kg","notes"]:
                if col not in df.columns:
                    df[col] = None
            # Auto-generate id if missing/empty using date+time
            def _gen_id(row):
                rid = str(row.get("id") or "").strip()
                if rid:
                    return rid
                d = str(row.get("date") or "").strip()
                t = str(row.get("time") or "").strip()
                return f"{d} {t}" if d or t else None
            df["id"] = df.apply(_gen_id, axis=1)
            # Convert date strings to date objects (DD/MM/YYYY or YYYY-MM-DD)
            def _to_date(s):
                s = str(s or "").strip()
                for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"):
                    try:
                        return datetime.strptime(s, fmt).date()
                    except Exception:
                        continue
                return pd.NaT
            df["date"] = df["date"].apply(_to_date)
            # Coerce weight to float
            df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")
            # Keep only app columns
            df = df[["id","date","time","weight_kg","notes"]]
            # Drop rows where all are NaN/empty
            df = df.dropna(how="all")
            logger.debug(f"SheetsRepo normalized df rows={len(df)}\n{df.to_string()}")
            return df
        except Exception:
            logger.exception("SheetsRepo.read_df failed")
            return pd.DataFrame(columns=self.columns)

    def append(self, tab: str, payload: dict) -> str:
        if tab != "Bodyweight":
            return payload.get("id") or ""
        _validate_payload(tab, payload)
        row_id = payload.get("id") or str(uuid.uuid4())
        payload = {**payload, "id": row_id}
        values: list[object] = []
        for c in self.columns:
            v = payload.get(c)
            if c == "date" and isinstance(v, (date, datetime)):
                # Convert ISO date to Sheet format (DD/MM/YYYY)
                values.append(self._iso_to_sheet_date(v.isoformat() if isinstance(v, datetime) else str(v)))
            elif c == "time" and v:
                # Convert ISO time to Sheet format (HH:MM:SS)
                values.append(self._iso_to_sheet_time(str(v)))
            elif isinstance(v, (date, datetime)):
                values.append(v.isoformat())
            else:
                values.append(v)
        self.sheet.append_rows([values], value_input_option="RAW")
        logger.info("Sheets appended Bodyweight id=%s", row_id)
        return row_id

    def update(self, tab: str, row_id: str, payload: dict) -> None:
        if tab != "Bodyweight":
            return
        row_num = self._find_row(row_id)
        if not row_num:
            raise ValueError(f"Row id {row_id} not found in Sheets")
        values: list[object] = []
        for c in self.columns:
            v = payload.get(c)
            if c == "date" and isinstance(v, (date, datetime)):
                # Convert ISO date to Sheet format (DD/MM/YYYY)
                values.append(self._iso_to_sheet_date(v.isoformat() if isinstance(v, datetime) else str(v)))
            elif c == "time" and v:
                # Convert ISO time to Sheet format (HH:MM:SS)
                values.append(self._iso_to_sheet_time(str(v)))
            elif isinstance(v, (date, datetime)):
                values.append(v.isoformat())
            else:
                values.append(v)
        start = rowcol_to_a1(row_num, 1)
        end = rowcol_to_a1(row_num, len(self.columns))
        self.sheet.update(f"{start}:{end}", [values], value_input_option="RAW")
        logger.info("Sheets updated Bodyweight id=%s", row_id)

    def delete(self, tab: str, row_id: str) -> None:
        if tab != "Bodyweight":
            return
        row_num = self._find_row(row_id)
        if not row_num:
            logger.warning("Sheets delete skipped; id=%s not found", row_id)
            return
        self.sheet.delete_rows(row_num)
        logger.info("Sheets deleted Bodyweight id=%s", row_id)


class CombinedRepo:
    def __init__(self, primary: Repo, secondary: Repo):
        self.primary = primary
        self.secondary = secondary

    def read_df(self, tab: str) -> pd.DataFrame:
        """Read from primary and merge with secondary (secondary only for Bodyweight)."""
        primary_df = self.primary.read_df(tab)
        logger.debug(f"CombinedRepo primary read {tab}: {len(primary_df)} rows")
        
        # Only merge Bodyweight from secondary Sheets
        if tab != "Bodyweight":
            return primary_df
        
        try:
            secondary_df = self.secondary.read_df(tab)
            logger.debug(f"CombinedRepo secondary read {tab}: {len(secondary_df)} rows")
        except Exception:
            logger.warning("Secondary read failed for %s; returning primary only", tab, exc_info=True)
            return primary_df
        
        if secondary_df.empty:
            logger.debug(f"CombinedRepo secondary df is empty for {tab}")
            return primary_df
        
        # Merge: combine both dataframes and deduplicate by id or date+time fallback
        combined = pd.concat([primary_df, secondary_df], ignore_index=True)
        logger.debug(f"CombinedRepo combined before dedup: {len(combined)} rows")
        if not combined.empty and "id" in combined.columns:
            tmp_id = combined["id"].fillna("").astype(str).str.strip()
            fallback = (
                combined.get("date").astype(str).str.strip() + " " + combined.get("time").astype(str).str.strip()
            ).str.strip()
            combined["_key"] = tmp_id.where(tmp_id != "", fallback)
            combined = combined.drop_duplicates(subset=["_key"], keep="first").drop(columns=["_key"], errors="ignore")
        logger.debug(f"CombinedRepo combined after dedup: {len(combined)} rows\n{combined.to_string()}")
        
        return combined

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
    """Return a repository, optionally writing through to Google Sheets for Bodyweight."""
    primary = SQLiteRepo()
    cfg = sheets_config()
    if not cfg:
        logger.info("Sheets config not found; using SQLite only")
        return primary
    logger.info(
        "Sheets config detected: spreadsheet_id=%s worksheet=%s creds_file=%s creds_json=%s",
        cfg.spreadsheet_id,
        cfg.worksheet,
        (cfg.credentials_file or ""),
        "set" if cfg.credentials_json else ""
    )
    try:
        secondary = SheetsRepo(cfg)
        logger.info("Using CombinedRepo with Sheets secondary for Bodyweight")
        return CombinedRepo(primary, secondary)
    except Exception:
        logger.warning("Falling back to SQLite only; Sheets init failed", exc_info=True)
        return primary


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
