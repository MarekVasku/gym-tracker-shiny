"""Pytest fixtures and configuration for test suite."""
import os
import tempfile
from datetime import date

import pytest


@pytest.fixture
def sample_lift_data():
    """Sample lift data for testing."""
    return {
        "id": "test-lift-123",
        "date": date(2025, 11, 1),
        "exercise": "Squat",
        "weight_kg": 100.0,
        "reps": 5,
        "notes": "Felt strong"
    }


@pytest.fixture
def sample_bodyweight_data():
    """Sample bodyweight data for testing."""
    return {
        "id": "test-bw-123",
        "date": date(2025, 11, 1),
        "time": "08:00",
        "weight_kg": 80.5,
        "notes": "Morning weight"
    }


@pytest.fixture
def sample_inbody_data():
    """Sample InBody data for testing."""
    return {
        "id": "test-ib-123",
        "date": date(2025, 11, 1),
        "inbody_score": 85.0,
        "weight_kg": 80.0,
        "skeletal_muscle_kg_total": 35.5,
        "body_fat_kg_total": 12.3,
        "body_fat_percent": 15.4,
        "visceral_fat_level": 8.0,
        "bmr_kcal": 1800.0,
        "muscle_right_arm_kg": 3.5,
        "muscle_left_arm_kg": 3.4,
        "muscle_trunk_kg": 22.0,
        "muscle_right_leg_kg": 8.3,
        "muscle_left_leg_kg": 8.2,
        "fat_right_arm_kg": 1.0,
        "fat_left_arm_kg": 1.0,
        "fat_trunk_kg": 7.5,
        "fat_right_leg_kg": 2.4,
        "fat_left_leg_kg": 2.4,
        "notes": "Test measurement"
    }


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_env_sqlite(temp_db, monkeypatch):
    """Mock environment for SQLite mode."""
    monkeypatch.setenv("DB_PATH", temp_db)
    return temp_db
