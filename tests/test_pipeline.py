"""
Integration tests for the full data pipeline.
Verifies table shapes, cohort retention, join integrity, and null tolerances.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import run_data_pipeline


def test_pipeline_table_shape_and_users():
    """Verify data pipeline processes all 5 raw datasets and retains 120 target users."""
    df = run_data_pipeline()
    assert df["user_id"].nunique() == 120, f"Expected 120 users, got {df['user_id'].nunique()}"
    assert len(df) >= 4900, f"Expected ~4989 check-ins, got {len(df)}"


def test_pipeline_no_duplicate_checkins():
    """Verify that every row in the modelling table is unique by user_id and checkin_date."""
    df = run_data_pipeline()
    duplicates = df.duplicated(subset=["user_id", "checkin_date"])
    assert not duplicates.any(), f"Found {duplicates.sum()} duplicate user-date pairs"


def test_has_session_data_flag_integrity():
    """D-021: Verify has_session_data accurately reflects sleep session presence."""
    df = run_data_pipeline()
    expected_flag = df["session_id"].notna().astype(int)
    assert (df["has_session_data"] == expected_flag).all()
    # Ensure between 70% and 80% of rows have matching sleep sessions
    sess_rate = df["has_session_data"].mean()
    assert 0.70 <= sess_rate <= 0.80, f"Session match rate out of expected bounds: {sess_rate:.2f}"
