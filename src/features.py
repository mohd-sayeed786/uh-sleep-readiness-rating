"""
Feature Engineering module for Readiness Score prediction.
Computes the 21 final selected features (Tier 1+2 enriched model) from the base modelling table.
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
    Given the merged base dataframe, compute the 21 features required for inference.

    Output includes all original columns plus:
      1. had_alcohol (binary 0/1)
      2. alcohol_level (ordinal 0/1/2)
      3. deep_rem_total (restorative sleep volume)
      4. total_sleep_minutes_zscore (user-normalised duration)
      5. stress_index_z (hr_z_7d - hrv_z_7d)
      6. alcohol_units (continuous)
      7. alcohol_x_hrv_z (alcohol_units * avg_hrv_rmssd_ms_zscore)
      8. sleep_debt (sleep deficit vs user baseline)
      9. rem_minutes_zscore (user-normalised REM)
      10. sleep_user_ratio (sleep / user expanding mean)
      11. recovery_score (hrv_z_7d - hr_z_7d)
      12. avg_hr_bpm_zscore (user-normalised heart rate)
      13. deep_minutes_zscore (user-normalised deep sleep)
      14. restorative_pct (deep_rem_total / total_sleep_minutes)
      15. avg_hrv_rmssd_ms_zscore (user-normalised HRV)
      16. feeling_roll5_mean (rolling mean of past 5 feelings)
      17. feeling_ewm_7 (exponential weighted mean, span=7)
      18. hrv_user_ratio (HRV / user expanding mean)
      19. deep_user_ratio (deep / user expanding mean)
      20. user_expanding_mean (expanding past mean feeling)
      21. hr_user_ratio (HR / user expanding mean)
      plus auxiliary tracking columns (subjective_feeling_lag1, checkin_seq_num, etc.)
    """
    logger.info(f"Starting feature engineering on {len(base_df)} rows across {base_df['user_id'].nunique()} users...")
    df = base_df.sort_values(["user_id", "checkin_date"]).copy().reset_index(drop=True)
    grouped = df.groupby("user_id")

    # ---- 1. ALCOHOL FEATURES ----
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

    # ---- 2. OVERNIGHT PHYSIOLOGY & USER Z-SCORES ----
    for col in ["total_sleep_minutes", "deep_minutes", "rem_minutes", "avg_hr_bpm", "avg_hrv_rmssd_ms"]:
        if col in df.columns:
            df[f"{col}_zscore"] = grouped[col].transform(compute_user_zscore)
        else:
            df[f"{col}_zscore"] = np.nan

    deep_m = df["deep_minutes"] if "deep_minutes" in df.columns else pd.Series(0.0, index=df.index)
    rem_m = df["rem_minutes"] if "rem_minutes" in df.columns else pd.Series(0.0, index=df.index)
    df["deep_rem_total"] = deep_m.fillna(0.0) + rem_m.fillna(0.0)

    if "total_sleep_minutes" in df.columns:
        df["restorative_pct"] = (
            df["deep_rem_total"] / df["total_sleep_minutes"].replace(0, np.nan)
        ).round(3)
        user_mean_sleep = grouped["total_sleep_minutes"].transform("mean")
        df["sleep_debt"] = df["total_sleep_minutes"] - user_mean_sleep
    else:
        df["restorative_pct"] = np.nan
        df["sleep_debt"] = np.nan

    # Alcohol x HRV interaction
    df["alcohol_x_hrv_z"] = df["alcohol_units"].fillna(0.0) * df["avg_hrv_rmssd_ms_zscore"].fillna(0.0)

    # ---- 3. TIER 1+2 PHYSIOLOGICAL (7-DAY BASELINE DEVIATIONS & RATIOS) ----
    for raw_col in ["avg_hr_bpm", "avg_hrv_rmssd_ms"]:
        if raw_col in df.columns:
            mean_col = f"{raw_col}_roll7_mean"
            std_col = f"{raw_col}_roll7_std"
            df[mean_col] = grouped[raw_col].transform(
                lambda x: x.shift(1).rolling(7, min_periods=1).mean()
            )
            df[std_col] = grouped[raw_col].transform(
                lambda x: x.shift(1).rolling(7, min_periods=2).std()
            )

    if "avg_hr_bpm_roll7_std" in df.columns and "avg_hr_bpm" in df.columns:
        hr_std_safe = df["avg_hr_bpm_roll7_std"].replace(0, np.nan)
        hr_z_7d = ((df["avg_hr_bpm"] - df["avg_hr_bpm_roll7_mean"]) / hr_std_safe).clip(-5, 5)
    else:
        hr_z_7d = pd.Series(np.nan, index=df.index)

    if "avg_hrv_rmssd_ms_roll7_std" in df.columns and "avg_hrv_rmssd_ms" in df.columns:
        hrv_std_safe = df["avg_hrv_rmssd_ms_roll7_std"].replace(0, np.nan)
        hrv_z_7d = ((df["avg_hrv_rmssd_ms"] - df["avg_hrv_rmssd_ms_roll7_mean"]) / hrv_std_safe).clip(-5, 5)
    else:
        hrv_z_7d = pd.Series(np.nan, index=df.index)

    df["stress_index_z"] = hr_z_7d - hrv_z_7d
    df["recovery_score"] = hrv_z_7d - hr_z_7d

    # User expanding ratios
    for new_col, src_col in [("sleep_user_ratio", "total_sleep_minutes"),
                              ("hr_user_ratio", "avg_hr_bpm"),
                              ("hrv_user_ratio", "avg_hrv_rmssd_ms"),
                              ("deep_user_ratio", "deep_minutes")]:
        if src_col in df.columns:
            user_exp_mean = grouped[src_col].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            df[new_col] = (df[src_col] / user_exp_mean.replace(0, np.nan)).round(4)
        else:
            df[new_col] = np.nan

    # ---- 4. AUTOREGRESSIVE & LONGITUDINAL FEELING ----
    TARGET = "subjective_feeling"
    if TARGET in df.columns:
        df["feeling_roll5_mean"] = grouped[TARGET].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        df["feeling_ewm_7"] = grouped[TARGET].transform(
            lambda x: x.shift(1).ewm(span=7, min_periods=1).mean()
        )
        df["user_expanding_mean"] = grouped[TARGET].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        # Auxiliary lag & recency
        df["subjective_feeling_lag1"] = grouped[TARGET].shift(1)
        df["days_since_bad_sleep"] = compute_days_since_event_series(
            df, col=TARGET, threshold=2.0, direction="below"
        )
        df["days_since_great_sleep"] = compute_days_since_event_series(
            df, col=TARGET, threshold=4.0, direction="above"
        )
    else:
        df["feeling_roll5_mean"] = np.nan
        df["feeling_ewm_7"] = np.nan
        df["user_expanding_mean"] = np.nan
        df["subjective_feeling_lag1"] = np.nan
        df["days_since_bad_sleep"] = np.nan
        df["days_since_great_sleep"] = np.nan

    # Auxiliary temporal and engagement features
    dates = pd.to_datetime(df["checkin_date"])
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["checkin_seq_num"] = grouped.cumcount() + 1

    # Verify all 21 features exist
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
