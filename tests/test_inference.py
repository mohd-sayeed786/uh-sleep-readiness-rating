"""
Unit and functional tests for the Inference Service.
Verifies API contract adherence, output bounding, dynamic feature lists, cold-start fallback resilience,
and deterministic reproducible training/inference.
"""
import numpy as np
import pytest

from src.config import DEFAULT_XGB_PARAMS, RANDOM_SEED
from src.inference import ReadinessPredictor, get_predictor


@pytest.fixture
def predictor():
    return get_predictor()


def test_prediction_output_bounds(predictor):
    """Verify raw prediction is clamped strictly within [1.0, 5.0] and rounded is in {1..5}."""
    sample = {
        "alcohol_units": 0.0,
        "had_alcohol": 0.0,
        "alcohol_level": 0.0,
        "week_of_year": 10,
        "total_sleep_minutes_zscore": 1.5,
        "avg_hr_bpm_zscore": -1.2,
        "avg_hrv_rmssd_ms_zscore": 1.4,
        "subjective_feeling_lag1": 4.0,
        "days_since_bad_sleep": 12.0,
        "days_since_great_sleep": 1.0,
        "checkin_seq_num": 30,
        "deep_rem_total": 160.0,
        "sleep_debt": 25.0,
    }
    result = predictor.predict_single(sample)
    assert 1.0 <= result["pred_raw"] <= 5.0
    assert result["pred_rounded"] in [1, 2, 3, 4, 5]
    assert result["readiness_tier"] in ["Recovery", "Moderate", "Optimal"]
    assert isinstance(result["guidance"], str) and len(result["guidance"]) > 0


def test_cold_start_resilience(predictor):
    """Verify inference pipeline handles Day-1 cold-start user (all history is NaN) gracefully."""
    empty_features = {}
    result = predictor.predict_single(empty_features)
    assert 1.0 <= result["pred_raw"] <= 5.0
    assert result["pred_rounded"] in [1, 2, 3, 4, 5]
    assert result["is_cold_start"] is True


def test_prediction_reproducibility(predictor):
    """Verify identical inputs produce identical deterministic readiness outputs."""
    sample = {
        "alcohol_units": 3.0,
        "had_alcohol": 1.0,
        "alcohol_level": 2.0,
        "week_of_year": 12,
        "total_sleep_minutes_zscore": -1.0,
        "avg_hr_bpm_zscore": 1.5,
        "avg_hrv_rmssd_ms_zscore": -1.2,
        "subjective_feeling_lag1": 2.0,
        "days_since_bad_sleep": 1.0,
        "days_since_great_sleep": 15.0,
        "checkin_seq_num": 15,
        "deep_rem_total": 80.0,
        "sleep_debt": -60.0,
    }
    res1 = predictor.predict_single(sample)
    res2 = predictor.predict_single(sample)
    assert res1["pred_raw"] == res2["pred_raw"]
    assert res1["pred_rounded"] == res2["pred_rounded"]
    assert res1["readiness_tier"] == res2["readiness_tier"]


def test_batch_prediction_length_and_order(predictor):
    """Verify batch inference preserves input count, order, and schema."""
    batch_records = [
        {"alcohol_units": 0.0, "total_sleep_minutes_zscore": 0.5},
        {"alcohol_units": 4.0, "total_sleep_minutes_zscore": -1.5},
        {"alcohol_units": 1.0, "total_sleep_minutes_zscore": 0.0},
    ]
    results = predictor.predict_batch(batch_records)
    assert len(results) == 3
    for res in results:
        assert 1.0 <= res["pred_raw"] <= 5.0
        assert res["pred_rounded"] in [1, 2, 3, 4, 5]
        assert res["readiness_tier"] in ["Recovery", "Moderate", "Optimal"]


def test_dynamic_feature_names_resilience():
    """Verify predictor dynamically extracts only configured features and ignores extra input keys."""
    custom_predictor = ReadinessPredictor()
    # Payload with extra arbitrary keys and missing optional keys
    payload = {
        "alcohol_units": 1.0,
        "random_unrelated_feature": 999.0,
        "extra_sensor_channel": "accelerometer_raw",
        "total_sleep_minutes_zscore": 0.2,
    }
    result = custom_predictor.predict_single(payload)
    assert 1.0 <= result["pred_raw"] <= 5.0
    # Ensure extra keys are not in features_used
    assert "random_unrelated_feature" not in result["features_used"]
    assert "extra_sensor_channel" not in result["features_used"]
    assert "alcohol_units" in result["features_used"]


def test_training_and_config_seed_determinism():
    """Verify constant seed is enforced in config and default XGBoost parameters."""
    assert RANDOM_SEED == 42
    assert DEFAULT_XGB_PARAMS.get("random_state") == RANDOM_SEED
