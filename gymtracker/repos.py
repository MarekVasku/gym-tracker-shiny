from __future__ import annotations
import uuid, json, sqlite3
from dataclasses import dataclass
from typing import Protocol
import pandas as pd

from .utils import REQUIRED_TABS
from .config import SCOPES, sheet_config, db_path, persist_target

# Lazy imports for Sheets
try:
    import gspread  # type: ignore
    from google.oauth2.service_account import Credentials  # type: ignore
except Exception:  # pragma: no cover
    gspread = None
    Credentials = None

class Repo(Protocol):
    def read_df(self, tab: str) -> pd.DataFrame: ...
    def append(self, tab: str, payload: dict) -> str: ...
    def update(self, tab: str, row_id: str, payload: dict) -> None: ...
    def delete(self, tab: str, row_id: str) -> None: ...


class SheetsRepo:
    def __init__(self, sheet_url: str, creds_json: str):
        if gspread is None or Credentials is None:
            raise RuntimeError("gspread/google-auth not installed. pip install gspread google-auth")
        info = json.loads(creds_json)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        self.gc = gspread.authorize(creds)
        self.sh = self.gc.open_by_url(sheet_url)
        self._ensure_tabs()

    def _ensure_tabs(self):
        for tab, headers in REQUIRED_TABS.items():
            try:
                ws = self.sh.worksheet(tab)
            except Exception:
                ws = self.sh.add_worksheet(title=tab, rows=1000, cols=len(headers) + 2)
                ws.append_row(headers)
            row1 = self.sh.worksheet(tab).row_values(1)
            if row1[: len(headers)] != headers:
                try:
                    self.sh.worksheet(tab).delete_rows(1)
                except Exception:
                    pass
                self.sh.worksheet(tab).insert_row(headers, 1)

    def _ws(self, tab: str):
        return self.sh.worksheet(tab)

    def read_df(self, tab: str) -> pd.DataFrame:
        rows = self._ws(tab).get_all_records()
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=REQUIRED_TABS[tab])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        return df

    def append(self, tab: str, payload: dict) -> str:
        ws = self._ws(tab)
        headers = REQUIRED_TABS[tab]
        if not payload.get("id"):
            payload["id"] = str(uuid.uuid4())
        ws.append_row([payload.get(h, "") for h in headers])
        return payload["id"]

    def update(self, tab: str, row_id: str, payload: dict) -> None:
        ws = self._ws(tab)
        data = ws.get_all_records()
        headers = REQUIRED_TABS[tab]
        idx = None
        for i, r in enumerate(data, start=2):
            if str(r.get("id")) == str(row_id):
                idx = i
                break
        if idx is None:
            raise KeyError(f"Row id {row_id} not found in {tab}")
        current = data[idx - 2]
        merged = {h: payload.get(h, current.get(h, "")) for h in headers}
        merged["id"] = row_id
        ws.update(f"A{idx}:{chr(64+len(headers))}{idx}", [[merged.get(h, "") for h in headers]])

    def delete(self, tab: str, row_id: str) -> None:
        ws = self._ws(tab)
        data = ws.get_all_records()
        for i, r in enumerate(data, start=2):
            if str(r.get("id")) == str(row_id):
                ws.delete_rows(i)
                return
        raise KeyError(f"Row id {row_id} not found in {tab}")


class SQLiteRepo:
    def __init__(self, path: str | None = None):
        self.path = path or db_path()
        self._init_db()

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
                    neck_cm REAL,
                    shoulder_cm REAL,
                    chest_cm REAL,
                    waist_cm REAL,
                    biceps_cm REAL,
                    thigh_cm REAL,
                    calf_cm REAL,
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

    def read_df(self, tab: str) -> pd.DataFrame:
        with self._conn() as con:
            df = pd.read_sql_query(f"SELECT * FROM {tab}", con)
        if df.empty:
            return pd.DataFrame(columns=REQUIRED_TABS[tab])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        return df

    def append(self, tab: str, payload: dict) -> str:
        row_id = payload.get("id") or str(uuid.uuid4())
        payload = {**payload, "id": row_id}
        cols = REQUIRED_TABS[tab]
        placeholders = ",".join(["?"] * len(cols))
        with self._conn() as con:
            con.execute(
                f"INSERT OR REPLACE INTO {tab} ({','.join(cols)}) VALUES ({placeholders})",
                tuple(payload.get(c) for c in cols),
            )
        return row_id

    def update(self, tab: str, row_id: str, payload: dict) -> None:
        sets = [f"{k} = ?" for k in payload.keys() if k != "id"]
        if not sets:
            return
        with self._conn() as con:
            con.execute(
                f"UPDATE {tab} SET {', '.join(sets)} WHERE id = ?",
                tuple(payload[k] for k in payload.keys() if k != "id") + (row_id,),
            )

    def delete(self, tab: str, row_id: str) -> None:
        with self._conn() as con:
            con.execute(f"DELETE FROM {tab} WHERE id = ?", (row_id,))


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
            pass
        return row_id

    def update(self, tab: str, row_id: str, payload: dict) -> None:
        self.primary.update(tab, row_id, payload)
        try:
            self.secondary.update(tab, row_id, payload)
        except Exception:
            pass

    def delete(self, tab: str, row_id: str) -> None:
        self.primary.delete(tab, row_id)
        try:
            self.secondary.delete(tab, row_id)
        except Exception:
            pass


def repo_factory() -> Repo:
    target = persist_target()
    if target == "sqlite":
        return SQLiteRepo()
    if target == "sheet":
        url, creds = sheet_config()
        return SheetsRepo(url, creds)
    # both
    url, creds = sheet_config()
    return CombinedRepo(SheetsRepo(url, creds), SQLiteRepo())
