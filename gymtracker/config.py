from __future__ import annotations

import os
from typing import Literal, cast

PersistTarget = Literal["sqlite"]

def persist_target() -> PersistTarget:
    """Return the persistence target. Project uses SQLite-only."""
    return cast(PersistTarget, "sqlite")


def db_path() -> str:
    return os.environ.get("DB_PATH", "./gym_tracker.db")
