#!/usr/bin/env python3
"""
Super-simple script: open Google Sheet worksheet, build a pandas DataFrame,
then print columns and all values to the terminal.

Reads .env for:
- SHEETS_SPREADSHEET_ID
- SHEETS_BODYWEIGHT_SHEET (defaults to Bodyweight)
- SHEETS_CREDENTIALS_FILE or SHEETS_CREDENTIALS_JSON
"""

import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv

# Put project root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import gspread  # type: ignore


def _get_client():
    # Prefer file-based creds; fall back to inline JSON
    cred_file = os.environ.get("SHEETS_CREDENTIALS_FILE")
    cred_json = os.environ.get("SHEETS_CREDENTIALS_JSON")
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    if cred_file:
        return gspread.service_account(filename=cred_file, scopes=scopes)
    if cred_json:
        data = json.loads(cred_json)
        return gspread.service_account_from_dict(data, scopes=scopes)
    raise RuntimeError("Missing credentials: set SHEETS_CREDENTIALS_FILE or SHEETS_CREDENTIALS_JSON in .env")


def main():
    load_dotenv(os.path.join(ROOT, ".env"), override=False)
    ssid = os.environ.get("SHEETS_SPREADSHEET_ID")
    sheet_name = os.environ.get("SHEETS_BODYWEIGHT_SHEET", "Bodyweight")
    if not ssid:
        print("[ERROR] SHEETS_SPREADSHEET_ID missing in .env")
        sys.exit(1)

    try:
        client = _get_client()
    except Exception as e:
        print(f"[ERROR] Failed to init gspread client: {e}")
        sys.exit(2)

    try:
        sh = client.open_by_key(ssid)
        ws = sh.worksheet(sheet_name)
    except Exception as e:
        print(f"[ERROR] Failed to open worksheet '{sheet_name}': {e}")
        sys.exit(3)

    # Fetch all values
    values = ws.get_all_values()
    if not values:
        print("[INFO] Worksheet is completely empty.")
        sys.exit(0)

    # Build DataFrame: use first row as headers if it looks like strings
    headers = values[0]
    data_rows = values[1:] if len(values) > 1 else []

    # If there are no data rows, still print columns and state
    df = pd.DataFrame(data_rows, columns=headers if data_rows else range(len(headers)))

    # Print columns
    print("[INFO] Columns:")
    print(" | ".join(map(str, headers)))

    # Print row count
    print(f"[INFO] Rows: {len(df)}")

    # Print values (all rows)
    if df.empty:
        print("[INFO] No data rows under header.")
    else:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        print("[INFO] DataFrame:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
