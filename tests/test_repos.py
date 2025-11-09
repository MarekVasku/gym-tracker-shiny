"""Tests for gymtracker.repos module."""
import pytest
import pandas as pd
from datetime import date
from gymtracker.repos import SQLiteRepo, repo_factory
from gymtracker.utils import REQUIRED_TABS


class TestSQLiteRepo:
    """Test cases for SQLiteRepo class."""
    
    def test_init_creates_tables(self, temp_db):
        """Test that initialization creates all required tables."""
        repo = SQLiteRepo(temp_db)
        
        # Check all tables exist
        with repo._conn() as con:
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cur.fetchall()}
        
        assert "Lifts" in tables
        assert "Bodyweight" in tables
        assert "Measurements" in tables
        assert "InBody" in tables
    
    def test_read_empty_df(self, temp_db):
        """Test reading from empty table returns correct schema."""
        repo = SQLiteRepo(temp_db)
        df = repo.read_df("Lifts")
        
        assert df.empty
        assert list(df.columns) == REQUIRED_TABS["Lifts"]
    
    def test_append_lift(self, temp_db, sample_lift_data):
        """Test appending a lift entry."""
        repo = SQLiteRepo(temp_db)
        row_id = repo.append("Lifts", sample_lift_data)
        
        assert row_id == sample_lift_data["id"]
        
        df = repo.read_df("Lifts")
        assert len(df) == 1
        assert df.iloc[0]["exercise"] == "Squat"
        assert df.iloc[0]["weight_kg"] == 100.0
        assert df.iloc[0]["reps"] == 5
    
    def test_append_generates_id(self, temp_db):
        """Test that append generates ID if not provided."""
        repo = SQLiteRepo(temp_db)
        data = {
            "date": date(2025, 11, 1),
            "exercise": "Bench",
            "weight_kg": 80.0,
            "reps": 8,
            "notes": ""
        }
        row_id = repo.append("Lifts", data)
        
        assert row_id is not None
        assert len(row_id) > 0
        
        df = repo.read_df("Lifts")
        assert len(df) == 1
        assert df.iloc[0]["id"] == row_id
    
    def test_update_lift(self, temp_db, sample_lift_data):
        """Test updating a lift entry."""
        repo = SQLiteRepo(temp_db)
        row_id = repo.append("Lifts", sample_lift_data)
        
        # Update weight and reps
        repo.update("Lifts", row_id, {
            "weight_kg": 110.0,
            "reps": 3
        })
        
        df = repo.read_df("Lifts")
        assert df.iloc[0]["weight_kg"] == 110.0
        assert df.iloc[0]["reps"] == 3
        # Other fields should remain unchanged
        assert df.iloc[0]["exercise"] == "Squat"
        assert df.iloc[0]["notes"] == "Felt strong"
    
    def test_delete_lift(self, temp_db, sample_lift_data):
        """Test deleting a lift entry."""
        repo = SQLiteRepo(temp_db)
        row_id = repo.append("Lifts", sample_lift_data)
        
        assert len(repo.read_df("Lifts")) == 1
        
        repo.delete("Lifts", row_id)
        
        assert len(repo.read_df("Lifts")) == 0
    
    def test_read_df_converts_dates(self, temp_db, sample_lift_data):
        """Test that dates are properly converted."""
        repo = SQLiteRepo(temp_db)
        repo.append("Lifts", sample_lift_data)
        
        df = repo.read_df("Lifts")
        assert isinstance(df.iloc[0]["date"], date)
    
    def test_append_bodyweight(self, temp_db, sample_bodyweight_data):
        """Test appending bodyweight entry."""
        repo = SQLiteRepo(temp_db)
        row_id = repo.append("Bodyweight", sample_bodyweight_data)
        
        df = repo.read_df("Bodyweight")
        assert len(df) == 1
        assert df.iloc[0]["weight_kg"] == 80.5
        assert df.iloc[0]["time"] == "08:00"
    
    def test_append_inbody(self, temp_db, sample_inbody_data):
        """Test appending InBody entry."""
        repo = SQLiteRepo(temp_db)
        row_id = repo.append("InBody", sample_inbody_data)
        
        df = repo.read_df("InBody")
        assert len(df) == 1
        assert df.iloc[0]["inbody_score"] == 85.0
        assert df.iloc[0]["skeletal_muscle_kg_total"] == 35.5
        assert df.iloc[0]["body_fat_percent"] == 15.4
    
    def test_multiple_entries(self, temp_db):
        """Test storing and retrieving multiple entries."""
        repo = SQLiteRepo(temp_db)
        
        # Add multiple lifts
        for i in range(5):
            repo.append("Lifts", {
                "date": date(2025, 11, i+1),
                "exercise": "Squat",
                "weight_kg": 100.0 + i * 5,
                "reps": 5,
                "notes": f"Set {i+1}"
            })
        
        df = repo.read_df("Lifts")
        assert len(df) == 5
        assert df["weight_kg"].tolist() == [100.0, 105.0, 110.0, 115.0, 120.0]


class TestRepoFactory:
    """Test cases for repo_factory function."""
    
    def test_factory_creates_sqlite_repo(self, mock_env_sqlite):
        """Test factory creates SQLite repo when configured."""
        repo = repo_factory()
        assert isinstance(repo, SQLiteRepo)
    
    def test_factory_respects_db_path(self, temp_db, monkeypatch):
        """Test factory uses correct DB path."""
        monkeypatch.setenv("DB_PATH", temp_db)
        
        repo = repo_factory()
        assert isinstance(repo, SQLiteRepo)
        assert repo.path == temp_db
