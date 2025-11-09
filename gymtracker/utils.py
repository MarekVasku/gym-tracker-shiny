from __future__ import annotations

import math
from typing import Dict, List


def epley_1rm(weight_kg: float, reps: int) -> float:
    """Estimate one-repetition maximum (1RM) using the Epley formula.

    1RM = weight_kg * (1 + reps / 30)

    - Caps reps at 12 to avoid unrealistic extrapolations
    - Rounds to 1 decimal place for readability
    - Returns NaN on invalid input
    """
    try:
        w = float(weight_kg)
        r = max(1, min(int(reps), 12))
        est = w * (1.0 + r / 30.0)
        return round(est, 1)
    except Exception:
        return float("nan")

def epley_training_max(weight_kg: float, reps: int) -> float:
    """Return a 90% training max based on Epley 1RM estimate.

    training_max = epley_1rm(weight_kg, reps) * 0.9
    Rounded to 1 decimal place.
    """
    orm = epley_1rm(weight_kg, reps)
    if math.isnan(orm):
        return orm
    return round(orm * 0.9, 1)

BIG3: List[str] = ["Squat", "Bench", "Deadlift"]

REQUIRED_TABS: Dict[str, List[str]] = {
    "Lifts": ["id", "date", "exercise", "weight_kg", "reps", "notes"],
    "Bodyweight": ["id", "date", "time", "weight_kg", "notes"],
    "Measurements": [
        "id", "date", "weight_kg", "neck_cm", "shoulder_cm", "chest_cm", "waist_cm", "biceps_cm", "thigh_cm", "calf_cm"
    ],
    "InBody": [
        "id", "date", "inbody_score", "weight_kg", "skeletal_muscle_kg_total", "body_fat_kg_total",
        "body_fat_percent", "visceral_fat_level", "bmr_kcal",
        "muscle_right_arm_kg", "muscle_left_arm_kg", "muscle_trunk_kg", "muscle_right_leg_kg", "muscle_left_leg_kg",
        "fat_right_arm_kg", "fat_left_arm_kg", "fat_trunk_kg", "fat_right_leg_kg", "fat_left_leg_kg",
        "notes"
    ],
}
