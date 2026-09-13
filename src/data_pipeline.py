"""
Data ingestion, cleaning, and preprocessing pipeline.
Implements Decision Log rules D-008 through D-022.
"""
from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.config import (
    DAILY_CONTEXT_FILE,
    DATA_DIR,
    MORNING_CHECKINS_FILE,
    PHYSIOLOGICAL_BOUNDS,
    SLEEP_SESSIONS_FILE,
    USER_PROFILES_FILE,
)
from src.logger import get_logger

logger = get_logger("data_pipeline")


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    """
    Safely parse timestamps handling both ISO strings and epoch-millisecond ints/floats.
    (Decision D-009: 1,731 epoch-millis timestamps in sleep_sessions)
    """
    # Detect numeric values that look like epoch milliseconds (> 1e11)
    numeric_converted = pd.to_numeric(series, errors="coerce")
    is_epoch_ms = numeric_converted.notna() & (numeric_converted > 1e11)

    result = pd.Series(index=series.index, dtype="datetime64[ns, UTC]")

    if is_epoch_ms.any():
        logger.debug(f"Parsing {is_epoch_ms.sum()} epoch-millisecond timestamps to UTC (D-009)")
        result.loc[is_epoch_ms] = pd.to_datetime(
            numeric_converted.loc[is_epoch_ms].astype("int64"),
            unit="ms",
            utc=True,
            errors="coerce",
        )

    non_epoch = ~is_epoch_ms
    if non_epoch.any():
        result.loc[non_epoch] = pd.to_datetime(
            series.loc[non_epoch],
            utc=True,
            errors="coerce",
        )

    return result


def clean_checkins(raw_checkins: pd.DataFrame) -> pd.DataFrame:
    """
    Clean morning_checkins:
      1. Normalise user_id (strip + uppercase) (D-008)
      2. Parse checkin date and submitted_at timestamp
      3. Drop duplicate checkins per (user_id, date) (D-010)
      4. Derive submit_hour, early_submit (D-016), and context_date (date - 1)
    """
    df = raw_checkins.copy()
    raw_users_count = df["user_id"].nunique()
    df["user_id"] = df["user_id"].astype(str).str.strip().str.upper()
    norm_users_count = df["user_id"].nunique()
    if raw_users_count != norm_users_count:
        logger.debug(f"D-008: Normalised user_ids from {raw_users_count} variations to {norm_users_count} canonical users")

    df["checkin_date"] = pd.to_datetime(df["date"]).dt.date

    if "submitted_at" in df.columns:
        df["submitted_at_dt"] = parse_timestamp_series(df["submitted_at"])
        # Drop duplicate check-ins per user-date, keeping latest
        dups = df.duplicated(subset=["user_id", "checkin_date"], keep=False).sum()
        if dups > 0:
            logger.debug(f"D-010: Found {dups} duplicate check-in entries; retaining latest submitted")
            df = df.sort_values("submitted_at_dt").drop_duplicates(
                subset=["user_id", "checkin_date"],
                keep="last"
            )
        # Submission timing indicators (D-016)
        df["submit_hour"] = df["submitted_at_dt"].dt.hour + df["submitted_at_dt"].dt.minute / 60.0
        df["early_submit"] = (df["submit_hour"] < 7.0).astype(int)
    else:
        df = df.drop_duplicates(subset=["user_id", "checkin_date"], keep="last")
        df["submit_hour"] = np.nan
        df["early_submit"] = 0

    # Reference date for linking daytime context (the day prior to waking checkin)
    df["context_date"] = df["checkin_date"].apply(lambda d: d - pd.Timedelta(days=1))

    logger.debug(f"Cleaned {len(df)} check-in records across {df['user_id'].nunique()} users")
    return df


def clean_sleep_sessions(
    raw_sessions: pd.DataFrame,
    target_users: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Clean sleep_sessions:
      1. Normalise user_id (D-008) and filter to study users
      2. Parse start/end timestamps (ISO and epoch millis) (D-009)
      3. Fix seconds-to-minutes duration anomaly (D-015)
      4. Correct negative durations (D-017)
      5. Fix sleep efficiency (> 1.0) and clip physiology / sentinel values (D-018)
      6. Consolidate multi-session nights based on session_start_dt night_date (D-011)
      7. Filter sessions outside reasonable duration bounds (60 to 900 minutes) (D-014)
    """
    df = raw_sessions.copy()
    df["user_id"] = df["user_id"].astype(str).str.strip().str.upper()

    if target_users is not None:
        df = df[df["user_id"].isin(target_users)].copy()

    # Parse timestamps
    df["session_start_dt"] = parse_timestamp_series(df["session_start"])
    df["session_end_dt"] = parse_timestamp_series(df["session_end"])
    df["session_end_date"] = df["session_end_dt"].dt.date
    df["night_date"] = df["session_start_dt"].dt.date

    # Drop corrupted sessions missing valid timestamps
    df = df.dropna(subset=["session_start_dt", "session_end_dt"]).copy()

    # Calculate actual duration between timestamps in minutes
    elapsed_minutes = (df["session_end_dt"] - df["session_start_dt"]).dt.total_seconds() / 60.0

    # Fix seconds-to-minutes duration bug (D-015)
    if "time_in_bed_minutes" in df.columns:
        tib = df["time_in_bed_minutes"].copy()
        ratio = tib / np.maximum(elapsed_minutes, 1.0)
        in_seconds_mask = (ratio > 50.0) & (ratio < 70.0)
        if in_seconds_mask.any():
            n_fixed = in_seconds_mask.sum()
            logger.debug(f"D-015: Correcting {n_fixed} sleep sessions recorded in seconds instead of minutes")
            duration_cols = [
                "time_in_bed_minutes",
                "total_sleep_minutes",
                "deep_minutes",
                "rem_minutes",
                "light_minutes",
                "awake_minutes",
            ]
            for col in duration_cols:
                if col in df.columns:
                    df.loc[in_seconds_mask, col] = df.loc[in_seconds_mask, col] / 60.0

    # Fix negative durations via absolute value (D-017)
    for col in ["awake_minutes", "light_minutes", "deep_minutes", "rem_minutes"]:
        if col in df.columns:
            neg_mask = df[col] < 0
            if neg_mask.any():
                logger.debug(f"D-017: Fixed {neg_mask.sum()} negative values in {col}")
                df.loc[neg_mask, col] = df.loc[neg_mask, col].abs()

    # Sentinel value handling: replace physiological impossibilities with NaN (D-018)
    if "avg_hr_bpm" in df.columns:
        hr_invalid = (df["avg_hr_bpm"] < PHYSIOLOGICAL_BOUNDS["avg_hr_bpm"][0]) | (
            df["avg_hr_bpm"] > PHYSIOLOGICAL_BOUNDS["avg_hr_bpm"][1]
        )
        if hr_invalid.any():
            logger.debug(f"D-018: Flagged {hr_invalid.sum()} sentinel/out-of-bounds heart rate entries as NaN")
            df.loc[hr_invalid, "avg_hr_bpm"] = np.nan

    if "avg_hrv_rmssd_ms" in df.columns:
        hrv_invalid = (df["avg_hrv_rmssd_ms"] < PHYSIOLOGICAL_BOUNDS["avg_hrv_rmssd_ms"][0]) | (
            df["avg_hrv_rmssd_ms"] > PHYSIOLOGICAL_BOUNDS["avg_hrv_rmssd_ms"][1]
        )
        if hrv_invalid.any():
            logger.debug(f"D-018: Flagged {hrv_invalid.sum()} sentinel/out-of-bounds HRV entries as NaN")
            df.loc[hrv_invalid, "avg_hrv_rmssd_ms"] = np.nan

    # Sleep efficiency correction: recalculate if > 1.0 (D-018)
    if "sleep_efficiency" in df.columns:
        eff_invalid = (df["sleep_efficiency"] > 1.0) | (df["sleep_efficiency"] < 0.0)
        if eff_invalid.any():
            logger.debug(f"D-018: Recalculated {eff_invalid.sum()} invalid sleep efficiency values")
            if "total_sleep_minutes" in df.columns and "time_in_bed_minutes" in df.columns:
                calc_eff = df["total_sleep_minutes"] / np.maximum(df["time_in_bed_minutes"], 1.0)
                df.loc[eff_invalid, "sleep_efficiency"] = np.clip(calc_eff.loc[eff_invalid], 0.0, 1.0)

    # Bedtime hour
    df["bedtime_hour"] = df["session_start_dt"].dt.hour + df["session_start_dt"].dt.minute / 60.0

    # Multi-session night handling (D-011):
    # Group by (user_id, night_date) to count sessions per night
    if "session_id" not in df.columns:
        df["session_id"] = [f"sess_{i}" for i in range(len(df))]

    n_sessions_per_night = df.groupby(["user_id", "night_date"]).size().reset_index(name="n_sessions")

    # Sort so primary (longest total sleep) is first, then deduplicate
    sort_cols = ["user_id", "night_date"]
    ascending = [True, True]
    if "total_sleep_minutes" in df.columns:
        sort_cols.append("total_sleep_minutes")
        ascending.append(False)

    df = df.sort_values(sort_cols, ascending=ascending).drop_duplicates(
        subset=["user_id", "night_date"], keep="first"
    ).copy()
    df = df.merge(n_sessions_per_night, on=["user_id", "night_date"], how="left")
    df["fragmented_night"] = (df["n_sessions"] > 1).astype(int)

    # Filter duration bounds (60 <= TIB <= 900) (D-014)
    if "time_in_bed_minutes" in df.columns:
        dur_mask = (df["time_in_bed_minutes"] >= 60.0) & (df["time_in_bed_minutes"] <= 900.0)
        df = df[dur_mask].copy()

    logger.debug(f"Cleaned {len(df)} primary sleep sessions across {df['user_id'].nunique()} users")
    return df


def clean_profiles(raw_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Clean user_profiles:
      1. Normalise user_id (D-008)
      2. Standardise column names
      3. Clip demographic ranges
    """
    df = raw_profiles.copy()
    if "userId" in df.columns:
        df = df.rename(columns={"userId": "user_id"})
    df["user_id"] = df["user_id"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset=["user_id"]).copy()

    # Plausibility bounds on demographics
    if "age_years" in df.columns:
        df["age_years"] = df["age_years"].clip(18, 100)
    if "height_cm" in df.columns:
        df["height_cm"] = df["height_cm"].clip(120, 240)
    if "weight_kg" in df.columns:
        df["weight_kg"] = df["weight_kg"].clip(35, 250)

    logger.debug(f"Cleaned user profiles for {len(df)} users")
    return df


def clean_daily_context(
    raw_context: pd.DataFrame,
    target_users: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Clean daily_context:
      1. Normalise user_id (D-008)
      2. Filter out non-study user orphans (D-013)
      3. Drop duplicate user-date rows
      4. Encode alcohol indicators (D-019) and workout indicators (D-020)
      5. Fix caffeine inconsistencies (D-022)
    """
    df = raw_context.copy()
    if "userId" in df.columns:
        df = df.rename(columns={"userId": "user_id"})
    df["user_id"] = df["user_id"].astype(str).str.strip().str.upper()
    if target_users is not None:
        init_len = len(df)
        df = df[df["user_id"].isin(target_users)].copy()
        logger.debug(f"D-013: Filtered context from {init_len} to {len(df)} rows belonging to active users")

    df["context_date"] = pd.to_datetime(df["date"]).dt.date
    df = df.drop_duplicates(subset=["user_id", "context_date"], keep="last").copy()

    # Alcohol indicators
    if "alcohol_units" in df.columns:
        df["had_alcohol"] = (df["alcohol_units"] > 0).astype(float)
        df["had_alcohol"] = df["had_alcohol"].where(df["alcohol_units"].notna(), np.nan)
        au = df["alcohol_units"].fillna(0)
        df["alcohol_level"] = np.where(au <= 0, 0.0, np.where(au <= 2.0, 1.0, 2.0)).astype(float)
    else:
        df["had_alcohol"] = np.nan
        df["alcohol_level"] = np.nan

    # Caffeine indicators & fix (D-022)
    if "caffeine_mg" in df.columns:
        df["had_caffeine"] = (df["caffeine_mg"] > 0).astype(float)
        df["had_caffeine"] = df["had_caffeine"].where(df["caffeine_mg"].notna(), np.nan)

        if "last_caffeine_hours_before_bed" in df.columns:
            # Impute missing hours_before_bed with population median (7.5h) when caffeine > 0
            valid_hours = df.loc[
                (df["caffeine_mg"] > 0) & df["last_caffeine_hours_before_bed"].notna(),
                "last_caffeine_hours_before_bed"
            ]
            med_hours = valid_hours.median() if len(valid_hours) > 0 else 7.5
            fix_caffeine_mask = (df["caffeine_mg"] > 0) & df["last_caffeine_hours_before_bed"].isna()
            df.loc[fix_caffeine_mask, "last_caffeine_hours_before_bed"] = med_hours

            # Clear orphaned hours where caffeine is zero or NaN
            orphaned_mask = (
                (df["caffeine_mg"].isna()) | (df["caffeine_mg"] == 0)
            ) & df["last_caffeine_hours_before_bed"].notna()
            df.loc[orphaned_mask, "last_caffeine_hours_before_bed"] = np.nan
    else:
        df["had_caffeine"] = np.nan

    # Workout indicators (D-020: distinct from explicit rest)
    if "workout_notes" in df.columns:
        wc = df["workout_notes"].fillna("").astype(str).str.strip().str.lower()
        df["had_workout"] = (
            ~wc.isin(["", "-", "n/a", "none", "rest", "rest day", "no workout"])
        ).astype(int)
    else:
        df["had_workout"] = 0

    logger.debug(f"Cleaned daily context: {len(df)} records across {df['user_id'].nunique()} users")
    return df


def build_modelling_table(
    checkins: pd.DataFrame,
    sessions: pd.DataFrame,
    context: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge cleaned tables via LEFT JOIN anchored on checkins (D-021):
      checkins <- profiles (on user_id)
      checkins <- daily_context (on user_id and context_date = checkin_date - 1)
      checkins <- sleep_sessions (on user_id and session_end_date = checkin_date)
    """
    base = checkins.copy()

    # 1. Profiles
    pcols = [c for c in ["user_id", "age_years", "sex", "height_cm", "weight_kg", "timezone", "plan_tier"] if c in profiles.columns]
    base = base.merge(profiles[pcols], on="user_id", how="left")

    # 2. Daily context
    ctx_cols = [c for c in [
        "user_id", "context_date", "steps", "active_minutes", "alcohol_units",
        "caffeine_mg", "last_caffeine_hours_before_bed", "stress_score",
        "travel_flag", "had_alcohol", "had_workout", "alcohol_level", "legacy_readiness_shown"
    ] if c in context.columns]
    base = base.merge(context[ctx_cols], on=["user_id", "context_date"], how="left")

    # 3. Sleep sessions
    sess_cols = [c for c in [
        "user_id", "session_end_date", "session_id",
        "time_in_bed_minutes", "total_sleep_minutes", "deep_minutes", "rem_minutes",
        "light_minutes", "awake_minutes", "sleep_efficiency", "avg_hr_bpm",
        "min_hr_bpm", "avg_hrv_rmssd_ms", "avg_spo2_pct", "avg_resp_rate_bpm",
        "temperature_deviation_c", "awakenings_count", "movement_index",
        "fragmented_night", "n_sessions", "bedtime_hour"
    ] if c in sessions.columns]

    base = base.merge(
        sessions[sess_cols],
        left_on=["user_id", "checkin_date"],
        right_on=["user_id", "session_end_date"],
        how="left"
    )

    # Multi-match deduplication: retain longest session if duplicate
    if base.duplicated(subset=["user_id", "checkin_date"]).any():
        sort_col = "total_sleep_minutes" if "total_sleep_minutes" in base.columns else base.columns[0]
        base = base.sort_values(sort_col, ascending=False).drop_duplicates(
            subset=["user_id", "checkin_date"], keep="first"
        )

    base["has_session_data"] = base["session_id"].notna().astype(int) if "session_id" in base.columns else 0
    base = base.sort_values(["user_id", "checkin_date"]).reset_index(drop=True)

    match_rate = base["has_session_data"].mean() * 100
    logger.info(f"Modelling table assembled: {len(base)} rows, {base['user_id'].nunique()} users | Sleep match rate: {match_rate:.1f}%")
    return base


def run_data_pipeline(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Execute full ingestion and cleaning pipeline from raw CSV files.
    Returns cleaned, joined modelling table.
    """
    d_dir = data_dir or DATA_DIR
    logger.info(f"Starting data ingestion and cleaning from {d_dir}...")

    raw_checkins = pd.read_csv(d_dir / "morning_checkins.csv")
    raw_sessions = pd.read_csv(d_dir / "sleep_sessions.csv")
    raw_profiles = pd.read_csv(d_dir / "user_profiles.csv")
    raw_context = pd.read_csv(d_dir / "daily_context.csv")

    checkins = clean_checkins(raw_checkins)
    study_users = set(checkins["user_id"].unique())
    logger.info(f"Anchoring study on {len(study_users)} active users from morning check-ins")

    profiles = clean_profiles(raw_profiles)
    sessions = clean_sleep_sessions(raw_sessions, target_users=study_users)
    context = clean_daily_context(raw_context, target_users=study_users)

    modelling_table = build_modelling_table(checkins, sessions, context, profiles)
    return modelling_table


if __name__ == "__main__":
    df = run_data_pipeline()
    print(f"Pipeline executed successfully. Table shape: {df.shape}")
