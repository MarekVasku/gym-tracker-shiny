from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, Field, validator

from gymtracker.utils import BIG3

"""Pydantic v1 models for data validation.

This project currently pins `pydantic<2.0` in the project manifests, so
we keep the v1 API (`validator`, `Config`) to match the pinned dependency
and avoid language-server/runtime mismatches.
"""

class LiftEntry(BaseModel):
    """Validation model for lift entries."""

    id: Optional[str] = None
    date: datetime.date
    exercise: str
    weight_kg: float = Field(gt=0, description="Weight must be positive")
    reps: int = Field(gt=0, le=50, description="Reps must be between 1-50")
    notes: Optional[str] = None

    @validator("exercise")
    def validate_exercise(cls, v: str) -> str:
        if v not in BIG3:
            raise ValueError(f"Exercise must be one of {BIG3}")
        return v

    @validator("date")
    def no_future_dates(cls, v: datetime.date) -> datetime.date:
        if v > datetime.date.today():
            raise ValueError("Cannot log future dates")
        return v

    class Config:
        schema_extra = {
            "examples": [
                {
                    "date": "2025-11-01",
                    "exercise": "Squat",
                    "weight_kg": 100.0,
                    "reps": 5,
                    "notes": "Felt strong",
                }
            ]
        }


class BodyweightEntry(BaseModel):
    """Validation model for bodyweight entries."""

    id: Optional[str] = None
    date: datetime.date
    time: str = Field(pattern=r"^\d{2}:\d{2}$", description="Time in HH:MM format")
    weight_kg: float = Field(gt=0, lt=500, description="Weight must be between 0-500kg")
    notes: Optional[str] = None

    @validator("date")
    def no_future_dates(cls, v: datetime.date) -> datetime.date:
        if v > datetime.date.today():
            raise ValueError("Cannot log future dates")
        return v

    class Config:
        schema_extra = {
            "examples": [
                {
                    "date": "2025-11-01",
                    "time": "08:00",
                    "weight_kg": 80.5,
                    "notes": "Morning weight",
                }
            ]
        }


class MeasurementEntry(BaseModel):
    """Validation model for body measurement entries."""

    id: Optional[str] = None
    date: datetime.date
    weight_kg: float = Field(gt=0, lt=500)
    neck_cm: Optional[float] = Field(None, gt=0, lt=100)
    shoulder_cm: Optional[float] = Field(None, gt=0, lt=200)
    chest_cm: Optional[float] = Field(None, gt=0, lt=200)
    waist_cm: Optional[float] = Field(None, gt=0, lt=200)
    biceps_cm: Optional[float] = Field(None, gt=0, lt=100)
    thigh_cm: Optional[float] = Field(None, gt=0, lt=150)
    calf_cm: Optional[float] = Field(None, gt=0, lt=100)

    @validator("date")
    def no_future_dates(cls, v: datetime.date) -> datetime.date:
        if v > datetime.date.today():
            raise ValueError("Cannot log future dates")
        return v


class InBodyEntry(BaseModel):
    """Validation model for InBody scan entries."""

    id: Optional[str] = None
    date: datetime.date
    inbody_score: float = Field(ge=0, le=100, description="InBody score 0-100")
    weight_kg: float = Field(gt=0, lt=500)
    skeletal_muscle_kg_total: float = Field(gt=0, lt=200)
    body_fat_kg_total: float = Field(ge=0, lt=300)
    body_fat_percent: float = Field(ge=0, le=100)
    visceral_fat_level: float = Field(ge=0, lt=50)
    bmr_kcal: float = Field(gt=0, lt=10000)

    # Segmental muscle mass
    muscle_right_arm_kg: Optional[float] = Field(None, ge=0, lt=50)
    muscle_left_arm_kg: Optional[float] = Field(None, ge=0, lt=50)
    muscle_trunk_kg: Optional[float] = Field(None, ge=0, lt=100)
    muscle_right_leg_kg: Optional[float] = Field(None, ge=0, lt=50)
    muscle_left_leg_kg: Optional[float] = Field(None, ge=0, lt=50)

    # Segmental fat mass
    fat_right_arm_kg: Optional[float] = Field(None, ge=0, lt=50)
    fat_left_arm_kg: Optional[float] = Field(None, ge=0, lt=50)
    fat_trunk_kg: Optional[float] = Field(None, ge=0, lt=100)
    fat_right_leg_kg: Optional[float] = Field(None, ge=0, lt=50)
    fat_left_leg_kg: Optional[float] = Field(None, ge=0, lt=50)

    notes: Optional[str] = None

    @validator("date")
    def no_future_dates(cls, v: datetime.date) -> datetime.date:
        if v > datetime.date.today():
            raise ValueError("Cannot log future dates")
        return v

    @validator("skeletal_muscle_kg_total")
    def validate_muscle_total(cls, v: float, values) -> float:
        weight = values.get("weight_kg")
        if weight is not None and v > weight:
            raise ValueError("Muscle mass cannot exceed total body weight")
        return v

    @validator("body_fat_kg_total")
    def validate_fat_total(cls, v: float, values) -> float:
        weight = values.get("weight_kg")
        if weight is not None and v > weight:
            raise ValueError("Fat mass cannot exceed total body weight")
        return v
