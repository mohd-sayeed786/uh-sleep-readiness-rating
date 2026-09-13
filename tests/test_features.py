"""
Unit tests for feature engineering logic.
Verifies temporal integrity, zero-division safeguards, and feature presence.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_NAMES
from src.features import compute_user_zscore, engineer_features


def test_compute_user_zscore_zero_variance():
    """Verify z-score does not crash or divide by zero when series has zero variance."""
    constant_series = pd.Series([450.0, 450.0, 450.0, 450.0])
    zscore = compute_user_zscore(constant_series)
    assert not zscore.isna().any()
    assert (zscore == 0.0).all()


def test_autoregressive_lag_shift_integrity():
    """Verify subjective_feeling_lag1 strictly shifts by 1 and never peeks ahead."""
    sample_data = pd.DataFrame({
        "user_id": ["UH-001", "UH-001", "UH-001"],
        "checkin_date": ["2026-03-01", "2026-03-02", "2026-03-03"],
        "subjective_feeling": [2.0, 4.0, 5.0],
        "alcohol_units": [0.0, 1.0, 0.0],
        "deep_minutes": [60.0, 70.0, 80.0],
        "rem_minutes": [90.0, 80.0, 90.0],
        "total_sleep_minutes": [420.0, 440.0, 460.0],
        "avg_hr_bpm": [60.0, 62.0, 58.0],
        "avg_hrv_rmssd_ms": [50.0, 48.0, 55.0],
    })
    feat_df = engineer_features(sample_data)
    lags = feat_df["subjective_feeling_lag1"].tolist()
    assert pd.isna(lags[0]), "First day lag must be NaN (no history)"
    assert lags[1] == 2.0, "Second day lag must equal first day's feeling"
    assert lags[2] == 4.0, "Third day lag must equal second day's feeling"


def test_restorative_deep_rem_total():
    """Verify deep_rem_total is strictly deep_minutes + rem_minutes."""
    sample = pd.DataFrame({
        "user_id": ["UH-001"],
        "checkin_date": ["2026-03-01"],
        "subjective_feeling": [3.0],
        "alcohol_units": [0.0],
        "deep_minutes": [65.0],
        "rem_minutes": [85.0],
    })
    feat = engineer_features(sample)
    assert feat["deep_rem_total"].iloc[0] == 150.0


def test_all_21_features_produced():
    """Ensure all 21 final selected features exist in output dataframe."""
    sample = pd.DataFrame({
        "user_id": ["UH-001", "UH-001"],
        "checkin_date": ["2026-03-01", "2026-03-02"],
        "subjective_feeling": [3.0, 4.0],
        "alcohol_units": [0.0, 0.0],
        "deep_minutes": [60.0, 60.0],
        "rem_minutes": [80.0, 80.0],
        "total_sleep_minutes": [420.0, 430.0],
        "avg_hr_bpm": [60.0, 61.0],
        "avg_hrv_rmssd_ms": [45.0, 46.0],
    })
    feat = engineer_features(sample)
    for expected_col in FEATURE_NAMES:
        assert expected_col in feat.columns, f"Missing feature: {expected_col}"
