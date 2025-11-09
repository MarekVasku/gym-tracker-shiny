"""Tests for gymtracker.utils module."""
import math
from gymtracker.utils import epley_1rm, epley_training_max, BIG3, REQUIRED_TABS


class TestEpley1RM:
    """Test cases for epley_1rm function."""
    
    def test_basic_calculation(self):
        """Test basic 1RM calculation."""
        # 100kg x 5 reps = 100 * (1 + 5/30) = 116.7kg
        result = epley_1rm(100.0, 5)
        assert result == 116.7
    
    def test_one_rep(self):
        """Test that 1 rep returns the weight itself."""
        result = epley_1rm(150.0, 1)
        assert result == 155.0  # 150 * (1 + 1/30) = 155
    
    def test_max_reps_capped_at_12(self):
        """Test that reps are capped at 12."""
        result_12 = epley_1rm(100.0, 12)
        result_20 = epley_1rm(100.0, 20)
        assert result_12 == result_20  # Both should use 12 reps
    
    def test_zero_reps_treated_as_one(self):
        """Test that zero reps is treated as 1."""
        result = epley_1rm(100.0, 0)
        assert result == epley_1rm(100.0, 1)
    
    def test_negative_reps_treated_as_one(self):
        """Test that negative reps is treated as 1."""
        result = epley_1rm(100.0, -5)
        assert result == epley_1rm(100.0, 1)
    
    def test_float_weight(self):
        """Test with float weight values."""
        result = epley_1rm(87.5, 8)
        expected = round(87.5 * (1 + 8/30), 1)
        assert result == expected
    
    def test_invalid_weight_returns_nan(self):
        """Test that invalid weight returns NaN."""
        result = epley_1rm("invalid", 5)  # type: ignore[arg-type]
        assert math.isnan(result)
    
    def test_invalid_reps_returns_nan(self):
        """Test that invalid reps returns NaN."""
        result = epley_1rm(100.0, "invalid")  # type: ignore[arg-type]
        assert math.isnan(result)
    
    def test_none_values_return_nan(self):
        """Test that None values return NaN."""
        assert math.isnan(epley_1rm(None, 5))  # type: ignore[arg-type]
        assert math.isnan(epley_1rm(100.0, None))  # type: ignore[arg-type]


class TestEpleyTrainingMax:
    """Test cases for epley_training_max function."""
    
    def test_basic_calculation(self):
        """Test basic training max calculation (90% of 1RM)."""
        # 100kg x 5 reps 1RM = 116.7kg, TM = 105.0kg
        result = epley_training_max(100.0, 5)
        expected = round(116.7 * 0.9, 1)
        assert result == expected
    
    def test_one_rep(self):
        """Test training max with 1 rep."""
        result = epley_training_max(150.0, 1)
        expected = round(155.0 * 0.9, 1)  # 90% of 1RM
        assert result == expected
    
    def test_invalid_input_returns_nan(self):
        """Test that invalid input propagates NaN."""
        result = epley_training_max("invalid", 5)  # type: ignore[arg-type]
        assert math.isnan(result)
    
    def test_preserves_nan_from_1rm(self):
        """Test that NaN from 1RM calculation is preserved."""
        result = epley_training_max(None, None)  # type: ignore[arg-type]
        assert math.isnan(result)
    
    def test_training_max_always_less_than_1rm(self):
        """Test that training max is always 90% of 1RM."""
        for weight in [50, 100, 150, 200]:
            for reps in [1, 5, 10]:
                tm = epley_training_max(weight, reps)
                orm = epley_1rm(weight, reps)
                assert tm == round(orm * 0.9, 1)


class TestConstants:
    """Test module constants."""
    
    def test_big3_exercises(self):
        """Test BIG3 constant contains expected exercises."""
        assert BIG3 == ["Squat", "Bench", "Deadlift"]
        assert len(BIG3) == 3
    
    def test_required_tabs_structure(self):
        """Test REQUIRED_TABS has correct structure."""
        assert isinstance(REQUIRED_TABS, dict)
        assert "Lifts" in REQUIRED_TABS
        assert "Bodyweight" in REQUIRED_TABS
        assert "Measurements" in REQUIRED_TABS
        assert "InBody" in REQUIRED_TABS
    
    def test_lifts_schema(self):
        """Test Lifts table schema."""
        expected = ["id", "date", "exercise", "weight_kg", "reps", "notes"]
        assert REQUIRED_TABS["Lifts"] == expected
    
    def test_bodyweight_schema(self):
        """Test Bodyweight table schema."""
        expected = ["id", "date", "time", "weight_kg", "notes"]
        assert REQUIRED_TABS["Bodyweight"] == expected
    
    def test_measurements_schema(self):
        """Test Measurements table schema."""
        schema = REQUIRED_TABS["Measurements"]
        assert "id" in schema
        assert "date" in schema
        assert "weight_kg" in schema
        assert "chest_cm" in schema
        assert "waist_cm" in schema
        # Should NOT have legacy columns
        assert "hips_cm" not in schema
        assert "arm_cm" not in schema
    
    def test_inbody_schema(self):
        """Test InBody table schema."""
        schema = REQUIRED_TABS["InBody"]
        assert "id" in schema
        assert "date" in schema
        assert "inbody_score" in schema
        assert "skeletal_muscle_kg_total" in schema
        assert "body_fat_kg_total" in schema
        assert "visceral_fat_level" in schema
        # Check segmental data
        assert "muscle_right_arm_kg" in schema
        assert "fat_trunk_kg" in schema
