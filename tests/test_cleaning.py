"""
Unit tests for data cleaning functions.
Protects critical cleaning invariants and Decision Log rules (D-008 to D-022).
"""
import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    clean_checkins,
    clean_daily_context,
    clean_profiles,
    clean_sleep_sessions,
    parse_timestamp_series,
)


def test_user_id_normalization():
    """D-008: Ensure user IDs with whitespace or lowercase are standardized."""
    raw = pd.DataFrame({
        "user_id": [" uh-001 ", "UH-002", "uh-003"],
        "date": ["2026-03-01", "2026-03-01", "2026-03-01"],
        "subjective_feeling": [3, 4, 2]
    })
    cleaned = clean_checkins(raw)
    assert list(cleaned["user_id"]) == ["UH-001", "UH-002", "UH-003"]


def test_epoch_millis_timestamp_parsing():
    """D-009: Verify epoch-millisecond timestamps parse to identical dates as ISO strings."""
    # 1772499600000 ms is 2026-03-03 01:00:00 UTC
    series = pd.Series(["2026-03-03T01:00:00Z", 1772499600000])
    parsed = parse_timestamp_series(series)
    assert parsed.iloc[0] == parsed.iloc[1]
    assert parsed.dt.year.tolist() == [2026, 2026]


def test_duration_seconds_to_minutes_correction():
    """D-015: Detect duration logged in seconds (~60x timestamp diff) and convert to minutes."""
    raw = pd.DataFrame({
        "session_id": ["sess_1", "sess_2"],
        "user_id": ["UH-001", "UH-001"],
        "session_start": ["2026-03-01T23:00:00Z", "2026-03-02T23:00:00Z"],
        "session_end": ["2026-03-02T07:00:00Z", "2026-03-03T07:00:00Z"],  # 8 hours = 480 mins
        "time_in_bed_minutes": [480.0, 28800.0],  # sess_2 logged in seconds (480 * 60)
        "total_sleep_minutes": [420.0, 25200.0],  # 420 * 60
        "deep_minutes": [60.0, 3600.0],
        "rem_minutes": [90.0, 5400.0],
        "light_minutes": [270.0, 16200.0],
        "awake_minutes": [60.0, 3600.0],
        "sleep_efficiency": [0.875, 0.875],
    })
    cleaned = clean_sleep_sessions(raw)
    assert cleaned.loc[cleaned["session_id"] == "sess_2", "time_in_bed_minutes"].iloc[0] == 480.0
    assert cleaned.loc[cleaned["session_id"] == "sess_2", "total_sleep_minutes"].iloc[0] == 420.0


def test_negative_duration_correction():
    """D-017: Negative awake or light minutes should be corrected using absolute value."""
    raw = pd.DataFrame({
        "session_id": ["sess_neg"],
        "user_id": ["UH-001"],
        "session_start": ["2026-03-01T23:00:00Z"],
        "session_end": ["2026-03-02T07:00:00Z"],
        "time_in_bed_minutes": [480.0],
        "total_sleep_minutes": [420.0],
        "awake_minutes": [-25.0],
        "light_minutes": [-180.0],
        "deep_minutes": [60.0],
        "rem_minutes": [90.0],
    })
    cleaned = clean_sleep_sessions(raw)
    assert cleaned["awake_minutes"].iloc[0] == 25.0
    assert cleaned["light_minutes"].iloc[0] == 180.0


def test_physiological_sentinel_handling():
    """D-018: Sentinel values (HR=0/250, HRV=999) must be converted to NaN."""
    raw = pd.DataFrame({
        "session_id": ["sess_sentinel"],
        "user_id": ["UH-001"],
        "session_start": ["2026-03-01T23:00:00Z"],
        "session_end": ["2026-03-02T07:00:00Z"],
        "time_in_bed_minutes": [480.0],
        "total_sleep_minutes": [420.0],
        "avg_hr_bpm": [250.0],  # Error sentinel
        "avg_hrv_rmssd_ms": [999.0],  # Error sentinel
        "sleep_efficiency": [1.45],  # Impossible efficiency
    })
    cleaned = clean_sleep_sessions(raw)
    assert pd.isna(cleaned["avg_hr_bpm"].iloc[0])
    assert pd.isna(cleaned["avg_hrv_rmssd_ms"].iloc[0])
    assert cleaned["sleep_efficiency"].iloc[0] <= 1.0


def test_multi_session_night_primary_selection():
    """D-011: Multi-session nights keep the longest session as primary and set fragmented_night=1."""
    raw = pd.DataFrame({
        "session_id": ["sess_short", "sess_long"],
        "user_id": ["UH-001", "UH-001"],
        "session_start": ["2026-03-01T22:00:00Z", "2026-03-01T23:30:00Z"],
        "session_end": ["2026-03-01T23:00:00Z", "2026-03-02T07:00:00Z"],
        "time_in_bed_minutes": [60.0, 450.0],
        "total_sleep_minutes": [50.0, 410.0],
    })
    cleaned = clean_sleep_sessions(raw)
    assert len(cleaned) == 1
    assert cleaned["session_id"].iloc[0] == "sess_long"
    assert cleaned["fragmented_night"].iloc[0] == 1


def test_daily_context_caffeine_imputation_and_orphan_cleanup():
    """D-022: Impute missing caffeine hours with median; clear orphaned hours where caffeine is zero."""
    raw = pd.DataFrame({
        "userId": ["UH-001", "UH-002"],
        "date": ["2026-03-01", "2026-03-01"],
        "caffeine_mg": [150.0, 0.0],
        "last_caffeine_hours_before_bed": [np.nan, 8.0],  # First missing, second orphaned
        "alcohol_units": [2.0, 0.0]
    })
    cleaned = clean_daily_context(raw)
    row1 = cleaned[cleaned["user_id"] == "UH-001"].iloc[0]
    row2 = cleaned[cleaned["user_id"] == "UH-002"].iloc[0]
    assert pd.notna(row1["last_caffeine_hours_before_bed"])
    assert pd.isna(row2["last_caffeine_hours_before_bed"])
