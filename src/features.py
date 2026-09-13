"""
Feature Engineering module for Readiness Score prediction.
Computes the 13 final selected features from the base modelling table.
"""
from typing import Optional

import numpy as np
import pandas as pd

from src.config import FEATURE_NAMES
from src.logger import get_logger

logger = get_logger("features")


def compute_user_zscore(group_series: pd.Series) -> pd.Series:
    """Compute per-user z-score with safe denominator."""
    mean_val = group_series.mean()
    std_val = group_series.std()
    safe_std = max(std_val, 0.01) if pd.notna(std_val) and std_val > 0 else 1.0
    return (group_series - mean_val) / safe_std


def compute_days_since_event_series(
    df: pd.DataFrame,
    col: str,
    threshold: float,
    direction: str = "above"
) -> pd.Series:
    """
    Computes days since condition was last true across user groups.
    Assigns by index to guarantee shape compatibility across all pandas versions.
    Uses strictly past history to prevent target leakage.
    """
    output = pd.Series(np.nan, index=df.index, dtype=float)
    dates = pd.to_datetime(df["checkin_date"])

    for uid, group in df.groupby("user_id"):
        last_event = pd.NaT
        for idx in group.index:
            curr_date = dates.loc[idx]

            # Assign days since prior event before updating with today's record
            if pd.notna(last_event):
                output.loc[idx] = float((curr_date - last_event).days)

            val = df.loc[idx, col] if col in df.columns else np.nan
            if pd.notna(val):
                if (direction == "above" and val >= threshold) or (direction == "below" and val <= threshold):
                    last_event = curr_date

    return output
def engineer_features(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the merged base dataframe, compute the 13 features required for inference.

    Output includes all original columns plus:
      1. alcohol_units (continuous)
      2. had_alcohol (binary 0/1)
      3. alcohol_level (ordinal 0/1/2)
      4. week_of_year (temporal)
      5. total_sleep_minutes_zscore (user-normalised duration)
      6. avg_hr_bpm_zscore (user-normalised heart rate)
      7. avg_hrv_rmssd_ms_zscore (user-normalised HRV)
      8. subjective_feeling_lag1 (yesterday's feeling)
      9. days_since_bad_sleep (recency <= 2)
      10. days_since_great_sleep (recency >= 4)
      11. checkin_seq_num (engagement order)
      12. deep_rem_total (restorative sleep volume)
      13. sleep_debt (sleep deficit vs user baseline)
    """
    logger.info(f"Starting feature engineering on {len(base_df)} rows across {base_df['user_id'].nunique()} users...")
    df = base_df.sort_values(["user_id", "checkin_date"]).copy()

    # 1. Alcohol features
    if "alcohol_units" in df.columns:
        df["alcohol_units"] = df["alcohol_units"].fillna(0.0)
        df["had_alcohol"] = (df["alcohol_units"] > 0).astype(float)
        df["alcohol_level"] = np.where(
            df["alcohol_units"] <= 0, 0.0,
            np.where(df["alcohol_units"] <= 2.0, 1.0, 2.0)
        ).astype(float)
    else:
        df["alcohol_units"] = 0.0
        df["had_alcohol"] = 0.0
        df["alcohol_level"] = 0.0

    # 2. Temporal feature
    dates = pd.to_datetime(df["checkin_date"])
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)

    # 3. User physiological z-scores
    grouped = df.groupby("user_id")

    if "total_sleep_minutes" in df.columns:
        df["total_sleep_minutes_zscore"] = grouped["total_sleep_minutes"].transform(compute_user_zscore)
        user_mean_sleep = grouped["total_sleep_minutes"].transform("mean")
        df["sleep_debt"] = df["total_sleep_minutes"] - user_mean_sleep
    else:
        df["total_sleep_minutes_zscore"] = np.nan
        df["sleep_debt"] = np.nan

    if "avg_hr_bpm" in df.columns:
        df["avg_hr_bpm_zscore"] = grouped["avg_hr_bpm"].transform(compute_user_zscore)
    else:
        df["avg_hr_bpm_zscore"] = np.nan

    if "avg_hrv_rmssd_ms" in df.columns:
        df["avg_hrv_rmssd_ms_zscore"] = grouped["avg_hrv_rmssd_ms"].transform(compute_user_zscore)
    else:
        df["avg_hrv_rmssd_ms_zscore"] = np.nan

    # 4. Restorative sleep volume (Deep + REM)
    deep = df["deep_minutes"] if "deep_minutes" in df.columns else pd.Series(np.nan, index=df.index)
    rem = df["rem_minutes"] if "rem_minutes" in df.columns else pd.Series(np.nan, index=df.index)
    df["deep_rem_total"] = deep + rem

    # 5. Autoregressive and sequence features
    if "subjective_feeling" in df.columns:
        df["subjective_feeling_lag1"] = grouped["subjective_feeling"].shift(1)
        df["days_since_bad_sleep"] = compute_days_since_event_series(
            df, col="subjective_feeling", threshold=2.0, direction="below"
        )
        df["days_since_great_sleep"] = compute_days_since_event_series(
            df, col="subjective_feeling", threshold=4.0, direction="above"
        )
    else:
        df["subjective_feeling_lag1"] = np.nan
        df["days_since_bad_sleep"] = np.nan
        df["days_since_great_sleep"] = np.nan

    df["checkin_seq_num"] = grouped.cumcount() + 1

    # Verify all 13 features exist
    for f in FEATURE_NAMES:
        if f not in df.columns:
            logger.warning(f"Feature '{f}' missing from engineered dataframe; filling with NaN")
            df[f] = np.nan

    logger.info(f"Feature engineering completed successfully: table shape {df.shape} with all {len(FEATURE_NAMES)} features present")
    return df


if __name__ == "__main__":
    from src.data_pipeline import run_data_pipeline
    base = run_data_pipeline()
    feats = engineer_features(base)
    print("Engineered features shape:", feats[FEATURE_NAMES].shape)
