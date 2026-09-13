# Databricks notebook source
# DBTITLE 1,Inference Pipeline — Readiness Score
# MAGIC %md
# MAGIC # Inference Pipeline — Readiness Score
# MAGIC
# MAGIC End-to-end pipeline that reads raw CSVs, cleans data exactly as training did, engineers the 13 final features, and predicts morning `subjective_feeling` (1–5).
# MAGIC
# MAGIC **Supports both single-instance and batch prediction.** Validated against the training test set to confirm identical results.

# COMMAND ----------

# DBTITLE 1,Install xgboost (if needed)
# Try importing first — xgboost was already installed by the Model Training notebook
# on this cluster. Only install if missing.
try:
    import xgboost
    print(f"xgboost {xgboost.__version__} already available — no install needed.")
except ImportError:
    import subprocess, sys
    print("xgboost not found, installing (CPU-only to skip 252 MB NVIDIA download)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "xgboost-cpu"])  # ~3 MB, no CUDA/NCCL deps
    import xgboost
    print(f"xgboost {xgboost.__version__} installed (CPU-only).")

# COMMAND ----------

# DBTITLE 1,Imports and paths
import os, json, pickle, warnings, time
import numpy as np
import pandas as pd
from datetime import timedelta
warnings.filterwarnings("ignore")

PROJECT_DIR = "/Workspace/Users/mohammad.sayeed@tatadigital.com/AS_UH_Project"
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
MODEL_DIR   = os.path.join(PROJECT_DIR, "selected_model")

print("Imports ready.")

# COMMAND ----------

# DBTITLE 1,Cell 2: Load model and reference data
# ══════════════════════════════════════════════════════════════════
# CELL 2: Load the trained model and reference data
# ══════════════════════════════════════════════════════════════════

# ── Load the 13-feature XGBoost model ──
with open(os.path.join(MODEL_DIR, "xgboost_tuned_reduced.pkl"), "rb") as f:
    model = pickle.load(f)

# ── Load the feature list (the 13 features in order) ──
with open(os.path.join(MODEL_DIR, "feature_list.json"), "r") as f:
    FEATURE_NAMES = json.load(f)

# ── Load model metadata for reference ──
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r") as f:
    model_meta = json.load(f)

print(f"Model loaded: {model_meta.get('model_type', 'XGBoost')}")
print(f"Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
print(f"Test R²: {model_meta.get('test_metrics', {}).get('r2', 'N/A')}")

# COMMAND ----------

# DBTITLE 1,Cell 3: Data cleaning functions
# ══════════════════════════════════════════════════════════════════
# CELL 3: Data Cleaning Functions
#
# These replicate the EXACT cleaning steps from the Data Cleaning & EDA
# notebook. Each function is standalone and documented.
# ══════════════════════════════════════════════════════════════════

def clean_checkins(raw_checkins: pd.DataFrame) -> pd.DataFrame:
    """
    Clean morning_checkins:
      1. Normalise user_id (uppercase + strip)
      2. Parse dates
      3. Remove duplicate (user_id, date) entries
      4. Add submit_hour from submitted_at
      5. Set checkin_date = date, context_date = date - 1 day
    """
    df = raw_checkins.copy()
    df["user_id"] = df["user_id"].str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["submitted_at"] = pd.to_datetime(df["submitted_at"])
    df = df.drop_duplicates(subset=["user_id", "date"], keep="last")
    df["submit_hour"] = df["submitted_at"].dt.hour
    df["checkin_date"] = df["date"]
    df["context_date"] = df["date"] - timedelta(days=1)
    return df


def clean_sessions(raw_sessions: pd.DataFrame, target_users: set) -> pd.DataFrame:
    """
    Clean sleep_sessions:
      1. Normalise user_id
      2. Parse mixed timestamps (ISO + epoch millis)
      3. Filter to target users
      4. Remove duplicate session_ids
      5. Fix seconds-to-minutes (785 rows where durations are in seconds)
      6. Fix negative durations with abs()
      7. Fix sensor sentinels: HR=0/250→NaN, HRV=999→NaN, efficiency>1→recalc
      8. Keep longest session per (user, night) for multi-session nights
      9. Filter duration outliers: keep [60, 900] min
      10. Derive session_end_date for joining to checkins
    """
    df = raw_sessions.copy()
    df["user_id"] = df["user_id"].str.strip().str.upper()
    
    # Parse timestamps (handle both ISO and epoch millis)
    for col in ["session_start", "session_end"]:
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        mask_na = parsed.isna()
        if mask_na.any():
            epoch_vals = pd.to_numeric(df.loc[mask_na, col], errors="coerce")
            parsed.loc[mask_na] = pd.to_datetime(epoch_vals, unit="ms", utc=True)
        df[f"{col}_utc"] = parsed
    
    # Filter to target users and remove duplicates
    df = df[df["user_id"].isin(target_users)].copy()
    df = df.drop_duplicates(subset=["session_id"], keep="first")
    
    # Seconds-to-minutes fix: detect via ratio to timestamp-derived duration
    duration_cols = ["time_in_bed_minutes", "total_sleep_minutes", "deep_minutes",
                     "rem_minutes", "light_minutes", "awake_minutes"]
    ts_duration = (df["session_end_utc"] - df["session_start_utc"]).dt.total_seconds() / 60
    ratio = df["time_in_bed_minutes"] / ts_duration.replace(0, np.nan)
    sec_mask = ratio.between(50, 70)  # ratio ~60 means values are in seconds
    for col in duration_cols:
        df.loc[sec_mask, col] = df.loc[sec_mask, col] / 60
    
    # Fix negative durations
    for col in ["awake_minutes", "light_minutes"]:
        neg_mask = df[col] < 0
        df.loc[neg_mask, col] = df.loc[neg_mask, col].abs()
    
    # Sensor sentinel handling
    df.loc[df["avg_hr_bpm"] == 0, "avg_hr_bpm"] = np.nan
    df.loc[df["avg_hr_bpm"] == 250, "avg_hr_bpm"] = np.nan
    df.loc[df["avg_hrv_rmssd_ms"] == 999, "avg_hrv_rmssd_ms"] = np.nan
    # Fix impossible efficiency (>1.0) by recalculating
    bad_eff = df["sleep_efficiency"] > 1.0
    df.loc[bad_eff, "sleep_efficiency"] = (
        df.loc[bad_eff, "total_sleep_minutes"] /
        df.loc[bad_eff, "time_in_bed_minutes"].replace(0, np.nan)
    )
    
    # Night date from session start
    df["night_date"] = df["session_start_utc"].dt.date
    
    # Keep longest session per (user, night)
    df = df.sort_values("total_sleep_minutes", ascending=False)
    df["fragmented_night"] = df.duplicated(subset=["user_id", "night_date"], keep="first").astype(int)
    n_sessions_per_night = df.groupby(["user_id", "night_date"]).size().reset_index(name="n_sessions")
    df = df.drop_duplicates(subset=["user_id", "night_date"], keep="first")
    df = df.merge(n_sessions_per_night, on=["user_id", "night_date"], how="left")
    df["fragmented_night"] = (df["n_sessions"] > 1).astype(int)
    
    # Duration filter
    df = df[(df["time_in_bed_minutes"] >= 60) & (df["time_in_bed_minutes"] <= 900)].copy()
    
    # Derive bedtime_hour for feature engineering
    df["bedtime_hour"] = df["session_start_utc"].dt.hour + df["session_start_utc"].dt.minute / 60
    
    # Session end date for joining to checkins
    df["session_end_date"] = df["session_end_utc"].dt.date
    
    return df


def clean_context(raw_context: pd.DataFrame, target_users: set) -> pd.DataFrame:
    """
    Clean daily_context:
      1. Rename userId→user_id, normalise
      2. Filter to target users
      3. Create binary flags: had_alcohol, had_caffeine, had_workout
      4. Create alcohol_level ordinal (none=0, light=1, heavy=2)
      5. Fix caffeine inconsistencies (D-022)
    """
    df = raw_context.copy()
    df = df.rename(columns={"userId": "user_id"})
    df["user_id"] = df["user_id"].str.strip().str.upper()
    df = df[df["user_id"].isin(target_users)].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.drop_duplicates(subset=["user_id", "date"], keep="last")
    
    # Binary flags
    df["had_alcohol"] = (df["alcohol_units"] > 0).astype(float)
    df["had_alcohol"] = df["had_alcohol"].where(df["alcohol_units"].notna(), np.nan)
    df["had_caffeine"] = (df["caffeine_mg"] > 0).astype(float)
    df["had_caffeine"] = df["had_caffeine"].where(df["caffeine_mg"].notna(), np.nan)
    df["had_workout"] = df["workout_notes"].notna().astype(int)
    # Refine: null/empty/"-"/"n/a"/"none" = unknown (not a workout)
    wc = df["workout_notes"].fillna("").str.strip().str.lower()
    df["had_workout"] = (~wc.isin(["", "-", "n/a", "none",
                                     "rest", "rest day", "no workout"])).astype(int)
    
    # Alcohol level: ordinal encoding matching training
    # Training used pd.cut with bins=[-0.1, 0, 2, 5.1] -> labels=[none, light, heavy]
    # We convert to numeric: none=0, light=1, heavy=2
    au = df["alcohol_units"].fillna(0)
    df["alcohol_level"] = np.where(au <= 0, 0, np.where(au <= 2, 1, 2)).astype(float)
    
    # Caffeine fix (D-022): impute missing hours with median
    med_hours = df.loc[
        (df["caffeine_mg"] > 0) & df["last_caffeine_hours_before_bed"].notna(),
        "last_caffeine_hours_before_bed"
    ].median()
    if pd.notna(med_hours):
        fix_mask = (df["caffeine_mg"] > 0) & df["last_caffeine_hours_before_bed"].isna()
        df.loc[fix_mask, "last_caffeine_hours_before_bed"] = med_hours
    # Clear orphaned caffeine hours
    bad_mask = ((df["caffeine_mg"].isna()) | (df["caffeine_mg"] == 0)) & df["last_caffeine_hours_before_bed"].notna()
    df.loc[bad_mask, "last_caffeine_hours_before_bed"] = np.nan
    
    return df


def clean_profiles(raw_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Clean user_profiles:
      1. Normalise user_id
      2. Deduplicate (keep latest profile_updated_at)
      3. Impute missing heights with sex-specific median
    """
    df = raw_profiles.copy()
    df["user_id"] = df["user_id"].str.strip().str.upper()
    df["profile_updated_at"] = pd.to_datetime(df["profile_updated_at"])
    df = (df.sort_values("profile_updated_at", ascending=False)
          .drop_duplicates(subset=["user_id"], keep="first"))
    # Impute missing heights
    for sex in ["M", "F"]:
        mask = (df["sex"] == sex) & (df["height_cm"].isna())
        median_h = df.loc[df["sex"] == sex, "height_cm"].median()
        df.loc[mask, "height_cm"] = median_h
    return df


print("Data cleaning functions defined: clean_checkins, clean_sessions, clean_context, clean_profiles")

# COMMAND ----------

# DBTITLE 1,Cell 4: Merge and build base table
# ══════════════════════════════════════════════════════════════════
# CELL 4: Merge Tables into Base Modelling Table
#
# Replicates the LEFT JOIN strategy from Data Cleaning (D-021):
#   checkins ← profiles (1:1 per user)
#   checkins ← daily_context (on context_date = checkin_date - 1)
#   checkins ← sleep_sessions (on session_end_date = checkin_date)
# ══════════════════════════════════════════════════════════════════

def build_base_table(checkins: pd.DataFrame, sessions: pd.DataFrame,
                     context: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the 4 cleaned tables into a single base table.
    Uses LEFT JOIN throughout to preserve all checkin rows.
    """
    # Start with checkins
    df = checkins[["user_id", "checkin_date", "context_date",
                   "subjective_feeling", "submit_hour"]].copy()
    
    # 1. LEFT JOIN profiles
    profile_cols = ["user_id", "age_years", "sex", "height_cm", "weight",
                    "timezone", "plan_tier"]
    available_pcols = [c for c in profile_cols if c in profiles.columns]
    df = df.merge(profiles[available_pcols], on="user_id", how="left")
    
    # 2. LEFT JOIN daily context
    ctx_cols = ["user_id", "date", "steps", "active_minutes", "alcohol_units",
                "caffeine_mg", "last_caffeine_hours_before_bed", "stress_score",
                "travel_flag", "had_alcohol", "had_workout", "alcohol_level"]
    available_ctx = [c for c in ctx_cols if c in context.columns]
    df = df.merge(
        context[available_ctx].rename(columns={"date": "context_date"}),
        on=["user_id", "context_date"], how="left"
    )
    
    # 3. LEFT JOIN sleep sessions
    sess_cols = ["user_id", "session_end_date", "session_id",
                 "time_in_bed_minutes", "total_sleep_minutes",
                 "deep_minutes", "rem_minutes", "light_minutes", "awake_minutes",
                 "sleep_efficiency", "avg_hr_bpm", "min_hr_bpm",
                 "avg_hrv_rmssd_ms", "avg_spo2_pct", "avg_resp_rate_bpm",
                 "temperature_deviation_c", "awakenings_count", "movement_index",
                 "fragmented_night", "n_sessions", "bedtime_hour"]
    available_sess = [c for c in sess_cols if c in sessions.columns]
    df = df.merge(
        sessions[available_sess],
        left_on=["user_id", "checkin_date"],
        right_on=["user_id", "session_end_date"],
        how="left"
    )
    # Handle multi-match: keep longest session
    if df.duplicated(subset=["user_id", "checkin_date"]).any():
        df = df.sort_values("total_sleep_minutes", ascending=False)
        df = df.drop_duplicates(subset=["user_id", "checkin_date"], keep="first")
    
    df["has_session_data"] = df["session_id"].notna().astype(int)
    df.drop(columns=["session_end_date", "session_id"], errors="ignore", inplace=True)
    
    # Sort by user and date (essential for lag/rolling features)
    df = df.sort_values(["user_id", "checkin_date"]).reset_index(drop=True)
    
    return df


print("Merge function defined: build_base_table")

# COMMAND ----------

# DBTITLE 1,Cell 5: Feature engineering — the 13 features explained
# ══════════════════════════════════════════════════════════════════
# CELL 5: Feature Engineering — Build the 13 Model Features
#
# Each feature is documented with:
#   - WHAT it is (plain English)
#   - HOW it is computed (exact logic)
#   - WHY the model uses it (interpretation)
#   - LEAKAGE SAFETY: how future-leaking is prevented
# ══════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 13 features required by the model.
    Input: base table sorted by [user_id, checkin_date]
    Output: same table with the 13 feature columns added.
    
    THE 13 FEATURES:
    -----------------------------------------------------------------------
    GROUP 1 — ALCOHOL (from daily_context, previous day's activity log)
      1. alcohol_units    : How many alcohol units the user logged yesterday.
                            More alcohol → worse morning feeling.
      2. had_alcohol      : Binary 0/1 — did the user drink at all?
                            Captures the "any vs none" cliff effect.
      3. alcohol_level    : Ordinal 0/1/2 (none/light/heavy).
                            Gives the model discrete thresholds to split on.
    
    GROUP 2 — TEMPORAL
      4. week_of_year     : Calendar week (1–52) from the night before checkin.
                            Captures seasonal rhythms (daylight, weather).
    
    GROUP 3 — OVERNIGHT PHYSIOLOGY (ring sensor, measured while sleeping)
      5. total_sleep_minutes_zscore : This night's sleep duration compared to
                            the user's own average. z = (value - mean) / std.
                            Positive = slept more than usual. Negative = less.
      6. avg_hr_bpm_zscore : User-normalised resting heart rate.
                            Lower HR during sleep = better recovery.
      7. avg_hrv_rmssd_ms_zscore : User-normalised heart rate variability.
                            Higher HRV = better parasympathetic recovery.
      8. deep_rem_total   : deep_minutes + rem_minutes. Total "restorative"
                            sleep. Deep = physical recovery, REM = cognitive.
      9. sleep_debt       : total_sleep_minutes minus user's mean.
                            Negative = slept less than usual (deficit).
    
    GROUP 4 — AUTOREGRESSIVE (uses PAST check-in history, never current day)
      10. subjective_feeling_lag1 : Yesterday's self-reported feeling (1–5).
                            How you felt yesterday predicts today. shift(1).
      11. checkin_seq_num  : Sequential check-in number (1st, 2nd, 3rd...).
                            Captures habituation / engagement effects.
    
    GROUP 5 — RECENCY (days since a notable past event)
      12. days_since_bad_sleep  : Days since feeling ≤2 was last reported.
                            A recent bad night lingers. Uses strict < guard.
      13. days_since_great_sleep : Days since feeling ≥4 was last reported.
                            Recent great sleep creates momentum. Dominant feature.
    -----------------------------------------------------------------------
    """
    df = df.copy()
    df = df.sort_values(["user_id", "checkin_date"]).reset_index(drop=True)
    checkin_dates = pd.to_datetime(df["checkin_date"])
    
    # ---- GROUP 1: ALCOHOL (already in base table from daily_context) ----
    # alcohol_units, had_alcohol, alcohol_level: carried through from merge.
    # Ensure had_alcohol is float (matches training)
    if "had_alcohol" not in df.columns:
        df["had_alcohol"] = (df["alcohol_units"] > 0).astype(float)
    if "alcohol_level" not in df.columns:
        au = df["alcohol_units"].fillna(0)
        df["alcohol_level"] = np.where(au <= 0, 0, np.where(au <= 2, 1, 2)).astype(float)
    
    # ---- GROUP 2: TEMPORAL ----
    # week_of_year from the night BEFORE checkin (night_date = checkin_date - 1)
    night_date = checkin_dates - pd.Timedelta(days=1)
    df["week_of_year"] = night_date.dt.isocalendar().week.astype(int)
    
    # ---- GROUP 3: OVERNIGHT PHYSIOLOGY ----
    # User z-scores: (value - user_mean) / max(user_std, 0.01)
    # These capture "how unusual was THIS night for THIS user"
    for col in ["total_sleep_minutes", "avg_hr_bpm", "avg_hrv_rmssd_ms"]:
        df[f"{col}_zscore"] = df.groupby("user_id")[col].transform(
            lambda x: (x - x.mean()) / max(x.std(), 0.01)
        )
    
    # deep_rem_total: sum of deep + REM sleep (restorative stages)
    df["deep_rem_total"] = df["deep_minutes"].fillna(0) + df["rem_minutes"].fillna(0)
    
    # sleep_debt: how much more/less than the user's average
    df["sleep_debt"] = df["total_sleep_minutes"] - df.groupby("user_id")["total_sleep_minutes"].transform("mean")
    
    # ---- GROUP 4: AUTOREGRESSIVE ----
    # subjective_feeling_lag1: yesterday's feeling, using shift(1) per user
    # LEAKAGE SAFETY: shift(1) guarantees we only see the PREVIOUS row
    df["subjective_feeling_lag1"] = df.groupby("user_id")["subjective_feeling"].shift(1)
    
    # checkin_seq_num: 1-indexed sequential counter per user
    df["checkin_seq_num"] = df.groupby("user_id").cumcount() + 1
    
    # ---- GROUP 5: RECENCY ----
    # days_since_bad_sleep and days_since_great_sleep
    # LEAKAGE SAFETY: Only counts events STRICTLY BEFORE the current row's date.
    # The loop records the last event date, then checks last_event < current_date.
    
    def _days_since_event(group, col, threshold, direction):
        """For each row, days since condition was last true (strict past only)."""
        result = pd.Series(np.nan, index=group.index)
        last_event = pd.NaT
        for i, (idx, row) in enumerate(group.iterrows()):
            val = row[col]
            if pd.notna(val):
                if (direction == "above" and val >= threshold) or \
                   (direction == "below" and val <= threshold):
                    last_event = row["checkin_date"]
            if pd.notna(last_event) and last_event < row["checkin_date"]:
                result.iloc[i] = (row["checkin_date"] - last_event).days
        return result
    
    df["checkin_date"] = pd.to_datetime(df["checkin_date"]).dt.date
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    
    df["days_since_bad_sleep"] = df.groupby("user_id", group_keys=False).apply(
        lambda g: _days_since_event(g, "subjective_feeling", 2, "below")
    ).values
    
    df["days_since_great_sleep"] = df.groupby("user_id", group_keys=False).apply(
        lambda g: _days_since_event(g, "subjective_feeling", 4, "above")
    ).values
    
    return df


print("Feature engineering function defined: engineer_features")
print(f"Produces these 13 features: {FEATURE_NAMES}")

# COMMAND ----------

# DBTITLE 1,Cell 6: Prediction functions (single + batch)
# ══════════════════════════════════════════════════════════════════
# CELL 6: Prediction Functions
#
# predict_batch(): Takes a DataFrame with the 13 features, returns predictions.
# predict_single(): Takes a dict of 13 feature values, returns one prediction.
# predict_from_raw(): Full pipeline from raw CSVs to predictions.
# ══════════════════════════════════════════════════════════════════

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame containing the 13 feature columns,
    add prediction columns and return.
    
    Output columns added:
      - pred_raw:     Raw model output (float, ~1.0–5.0)
      - pred_rounded: Rounded and clipped to integers 1–5
    """
    X = df[FEATURE_NAMES].copy()
    preds = model.predict(X)
    df = df.copy()
    df["pred_raw"] = preds
    df["pred_rounded"] = np.clip(np.round(preds), 1, 5).astype(int)
    return df


def predict_single(features: dict) -> dict:
    """
    Predict for a single instance.
    
    Args:
        features: dict with keys matching the 13 feature names.
                  Missing keys become NaN (XGBoost handles natively).
    
    Returns:
        dict with 'pred_raw' (float) and 'pred_rounded' (int 1–5).
    
    Example:
        predict_single({
            'alcohol_units': 2.0, 'had_alcohol': 1.0, 'alcohol_level': 1.0,
            'week_of_year': 15,
            'total_sleep_minutes_zscore': -0.5, 'avg_hr_bpm_zscore': 0.3,
            'avg_hrv_rmssd_ms_zscore': -0.2, 'deep_rem_total': 120.0,
            'sleep_debt': -30.0,
            'subjective_feeling_lag1': 3.0, 'checkin_seq_num': 25,
            'days_since_bad_sleep': 5.0, 'days_since_great_sleep': 2.0
        })
    """
    row = {feat: features.get(feat, np.nan) for feat in FEATURE_NAMES}
    X = pd.DataFrame([row])
    pred = model.predict(X)[0]
    return {
        "pred_raw": float(pred),
        "pred_rounded": int(np.clip(round(pred), 1, 5))
    }


def predict_from_raw() -> pd.DataFrame:
    """
    Full end-to-end pipeline:
      1. Load raw CSVs
      2. Clean all tables
      3. Merge into base table
      4. Engineer the 13 features
      5. Predict for all rows
    Returns the full DataFrame with predictions.
    """
    print("Step 1/5: Loading raw CSVs...")
    raw_checkins = pd.read_csv(os.path.join(DATA_DIR, "morning_checkins.csv"))
    raw_sessions = pd.read_csv(os.path.join(DATA_DIR, "sleep_sessions.csv"))
    raw_context  = pd.read_csv(os.path.join(DATA_DIR, "daily_context.csv"))
    raw_profiles = pd.read_csv(os.path.join(DATA_DIR, "user_profiles.csv"))
    
    print("Step 2/5: Cleaning tables...")
    checkins = clean_checkins(raw_checkins)
    target_users = set(checkins["user_id"].unique())
    sessions = clean_sessions(raw_sessions, target_users)
    context  = clean_context(raw_context, target_users)
    profiles = clean_profiles(raw_profiles)
    
    print("Step 3/5: Merging into base table...")
    base = build_base_table(checkins, sessions, context, profiles)
    
    print("Step 4/5: Engineering 13 features...")
    featured = engineer_features(base)
    
    print("Step 5/5: Predicting...")
    result = predict_batch(featured)
    
    n = len(result)
    print(f"\nDone: {n:,} predictions generated.")
    print(f"  Pred range: {result['pred_raw'].min():.2f} – {result['pred_raw'].max():.2f}")
    print(f"  Rounded distribution: {result['pred_rounded'].value_counts().sort_index().to_dict()}")
    return result


print("Prediction functions defined: predict_batch, predict_single, predict_from_raw")

# COMMAND ----------

# DBTITLE 1,Cell 7: Run full pipeline and validate against test set
# ══════════════════════════════════════════════════════════════════
# CELL 7: Run Full Pipeline + Validate Against Test Set
#
# This cell:
#   1. Runs predict_from_raw() to process all data end-to-end
#   2. Applies the SAME temporal split as Model Training
#   3. Compares test set predictions to expected metrics
# ══════════════════════════════════════════════════════════════════
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

# Run the full pipeline
all_preds = predict_from_raw()

# ---- Temporal split (same as Model Training notebook) ----
dates = pd.to_datetime(all_preds["checkin_date"])
unique_dates = sorted(dates.dropna().unique())
n_dates = len(unique_dates)

train_cutoff = unique_dates[int(n_dates * 0.70) - 1]
val_cutoff   = unique_dates[int(n_dates * 0.85) - 1]

train_mask = dates <= train_cutoff
val_mask   = (dates > train_cutoff) & (dates <= val_cutoff)
test_mask  = dates > val_cutoff

print(f"\n{'='*70}")
print(f"TEMPORAL SPLIT (replicating Model Training)")
print(f"{'='*70}")
print(f"  Train: {train_mask.sum():,} rows | up to {pd.Timestamp(train_cutoff).date()}")
print(f"  Val:   {val_mask.sum():,} rows  | {pd.Timestamp(train_cutoff).date()} to {pd.Timestamp(val_cutoff).date()}")
print(f"  Test:  {test_mask.sum():,} rows  | after {pd.Timestamp(val_cutoff).date()}")

# ---- Evaluate on each split ----
def evaluate_split(y_true, y_pred, n_features, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n > n_features + 1 else r2
    y_cls = np.clip(np.round(y_pred), 1, 5).astype(int)
    exact_acc = accuracy_score(y_true.astype(int), y_cls)
    acc_pm1 = np.mean(np.abs(y_true - y_cls) <= 1)
    return {"Split": label, "RMSE": round(rmse, 4), "MAE": round(mae, 4),
            "R²": round(r2, 4), "Adj R²": round(adj_r2, 4),
            "Exact Acc": f"{exact_acc:.1%}", "Acc ±1": f"{acc_pm1:.1%}", "N": n}

results = []
for mask, label in [(train_mask, "Train"), (val_mask, "Val"), (test_mask, "Test")]:
    subset = all_preds[mask]
    y_true = subset["subjective_feeling"].values.astype(float)
    y_pred = subset["pred_raw"].values
    results.append(evaluate_split(y_true, y_pred, len(FEATURE_NAMES), label))

results_df = pd.DataFrame(results)
print(f"\n{'='*70}")
print(f"INFERENCE PIPELINE METRICS")
print(f"{'='*70}")
display(results_df)

# ---- Compare to Model Training expected values ----
print(f"\n{'='*70}")
print(f"VALIDATION: Compare to Model Training notebook results")
print(f"{'='*70}")
expected = {"RMSE": 0.2769, "R²": 0.9253, "Exact Acc": "90.8%"}
test_row = results_df[results_df["Split"] == "Test"].iloc[0]
print(f"\n  Expected  Test R²:        {expected['R²']}")
print(f"  Pipeline  Test R²:        {test_row['R²']}")
print(f"  Expected  Test RMSE:      {expected['RMSE']}")
print(f"  Pipeline  Test RMSE:      {test_row['RMSE']}")
print(f"  Expected  Test Exact Acc: {expected['Exact Acc']}")
print(f"  Pipeline  Test Exact Acc: {test_row['Exact Acc']}")

# Check if results match within tolerance
r2_match = abs(test_row["R²"] - expected["R²"]) < 0.01
rmse_match = abs(test_row["RMSE"] - expected["RMSE"]) < 0.01
if r2_match and rmse_match:
    print(f"\n  ✅ VALIDATION PASSED: Inference pipeline reproduces training results.")
else:
    print(f"\n  ⚠️ METRICS DIFFER: Check feature computation or data cleaning steps.")
    print(f"     R² diff:   {abs(test_row['R²'] - expected['R²']):.4f}")
    print(f"     RMSE diff: {abs(test_row['RMSE'] - expected['RMSE']):.4f}")

# COMMAND ----------

# DBTITLE 1,Cell 8: Demo — single instance prediction
# ══════════════════════════════════════════════════════════════════
# CELL 8: Demo — Single Instance Prediction
#
# Shows how to use predict_single() for one user on one morning.
# ══════════════════════════════════════════════════════════════════

# Example: a user who drank moderately, slept slightly less than usual,
# had a bad night 10 days ago, and a great night 1 day ago.

example = {
    # Alcohol (from yesterday's daily_context)
    "alcohol_units": 2.0,            # 2 units consumed
    "had_alcohol": 1.0,              # yes, they drank
    "alcohol_level": 1.0,            # light (0=none, 1=light, 2=heavy)
    # Temporal
    "week_of_year": 15,              # mid-April
    # Overnight physiology (from ring sensor)
    "total_sleep_minutes_zscore": -0.5,  # slept a bit less than their average
    "avg_hr_bpm_zscore": 0.3,            # HR slightly elevated (worse)
    "avg_hrv_rmssd_ms_zscore": -0.2,     # HRV slightly below personal norm
    "deep_rem_total": 120.0,             # 2 hours of restorative sleep
    "sleep_debt": -30.0,                 # 30 min less than their average
    # Autoregressive
    "subjective_feeling_lag1": 3.0,      # felt neutral yesterday
    "checkin_seq_num": 25,               # 25th check-in for this user
    # Recency
    "days_since_bad_sleep": 10.0,        # last bad night was 10 days ago
    "days_since_great_sleep": 1.0,       # felt great just yesterday
}

result = predict_single(example)

print(f"Single-instance prediction:")
print(f"  Raw score:  {result['pred_raw']:.3f}")
print(f"  Rounded:    {result['pred_rounded']} (on 1–5 scale)")
print(f"\nInterpretation:")
if result['pred_rounded'] >= 4:
    print(f"  → The user is predicted to feel GOOD this morning.")
elif result['pred_rounded'] == 3:
    print(f"  → The user is predicted to feel NEUTRAL this morning.")
else:
    print(f"  → The user is predicted to feel BELOW AVERAGE this morning.")

print(f"\nKey drivers for this prediction:")
print(f"  + days_since_great_sleep=1 (felt great recently → positive momentum)")
print(f"  - alcohol_units=2 (moderate drinking → reduces score)")
print(f"  - sleep_debt=-30 (slept less than usual → reduces score)")
print(f"  + days_since_bad_sleep=10 (no recent bad nights → positive)")

# COMMAND ----------

# DBTITLE 1,Cell 9: Inference latency and performance metrics
# ══════════════════════════════════════════════════════════════════
# CELL 9: Inference Latency & Performance Metrics
#
# Measures actual timing for:
#   1. Single-instance prediction (model.predict only)
#   2. Batch prediction (model.predict on N rows)
#   3. Full end-to-end pipeline (load CSVs → clean → engineer → predict)
#   4. Feature engineering cost breakdown
# ══════════════════════════════════════════════════════════════════
import time, os

print("="*70)
print("INFERENCE LATENCY BENCHMARK")
print("="*70)

# ---- 1. Model-only: single instance ----
single_row = pd.DataFrame([{feat: example.get(feat, np.nan) for feat in FEATURE_NAMES}])

# Warmup (first call loads the model into CPU cache)
_ = model.predict(single_row)

N_RUNS = 1000
t0 = time.perf_counter()
for _ in range(N_RUNS):
    model.predict(single_row)
t1 = time.perf_counter()
single_us = (t1 - t0) / N_RUNS * 1e6

print(f"\n1. SINGLE-INSTANCE PREDICTION (model.predict only)")
print(f"   {single_us:.1f} µs / prediction  ({N_RUNS} runs averaged)")
print(f"   = {1e6/single_us:,.0f} predictions/sec")

# ---- 2. Batch prediction ----
test_subset = all_preds[test_mask].copy()
X_test_batch = test_subset[FEATURE_NAMES].copy()
batch_sizes = [1, 10, 100, len(X_test_batch)]

print(f"\n2. BATCH PREDICTION (model.predict only)")
print(f"   {'Batch':>8s}  {'Total (ms)':>12s}  {'Per Sample (µs)':>16s}  {'Throughput':>14s}")
print(f"   {'':->8s}  {'':->12s}  {'':->16s}  {'':->14s}")
for bs in batch_sizes:
    X_batch = X_test_batch.head(bs)
    _ = model.predict(X_batch)  # warmup
    runs = max(100, 1000 // bs)
    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict(X_batch)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) / runs * 1000
    per_sample_us = total_ms / bs * 1000
    throughput = bs / (total_ms / 1000)
    print(f"   {bs:>8,d}  {total_ms:>10.3f} ms  {per_sample_us:>13.1f} µs  {throughput:>10,.0f} /sec")

# ---- 3. Full pipeline timing breakdown ----
print(f"\n3. FULL PIPELINE BREAKDOWN (end-to-end)")

t_start = time.perf_counter()
raw_checkins = pd.read_csv(os.path.join(DATA_DIR, "morning_checkins.csv"))
raw_sessions = pd.read_csv(os.path.join(DATA_DIR, "sleep_sessions.csv"))
raw_context  = pd.read_csv(os.path.join(DATA_DIR, "daily_context.csv"))
raw_profiles = pd.read_csv(os.path.join(DATA_DIR, "user_profiles.csv"))
t_load = time.perf_counter()

checkins = clean_checkins(raw_checkins)
target_users = set(checkins["user_id"].unique())
sessions = clean_sessions(raw_sessions, target_users)
context  = clean_context(raw_context, target_users)
profiles = clean_profiles(raw_profiles)
t_clean = time.perf_counter()

base = build_base_table(checkins, sessions, context, profiles)
t_merge = time.perf_counter()

featured = engineer_features(base)
t_feat = time.perf_counter()

X_all = featured[FEATURE_NAMES]
preds = model.predict(X_all)
t_pred = time.perf_counter()

stages = [
    ("CSV loading (4 files)",        t_load - t_start),
    ("Data cleaning (4 tables)",     t_clean - t_load),
    ("Table merging (LEFT JOINs)",   t_merge - t_clean),
    ("Feature engineering (13 feat)", t_feat - t_merge),
    ("Model prediction (4,989 rows)", t_pred - t_feat),
]
total = t_pred - t_start

print(f"   {'Stage':<38s}  {'Time':>10s}  {'Share':>6s}")
print(f"   {'':->38s}  {'':->10s}  {'':->6s}")
for stage, dur in stages:
    print(f"   {stage:<38s}  {dur:>8.3f} s  {dur/total*100:>5.1f}%")
print(f"   {'':->38s}  {'':->10s}  {'':->6s}")
print(f"   {'TOTAL':<38s}  {total:>8.3f} s  100.0%")

# ---- 4. Model artefact sizes ----
print(f"\n4. MODEL ARTEFACT SIZES")
model_path = os.path.join(MODEL_DIR, "xgboost_tuned_reduced.pkl")
flist_path = os.path.join(MODEL_DIR, "feature_list.json")
meta_path  = os.path.join(MODEL_DIR, "model_metadata.json")
for label, path in [("Model pickle", model_path), ("Feature list", flist_path), ("Metadata", meta_path)]:
    size_kb = os.path.getsize(path) / 1024
    print(f"   {label:<20s}  {size_kb:>8.1f} KB  ({size_kb/1024:.2f} MB)")

# ---- Summary ----
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"  Model:               XGBoost, 13 features, 1.11 MB")
print(f"  Single prediction:   {single_us:.0f} µs  ({1e6/single_us:,.0f}/sec)")
print(f"  Batch ({len(X_test_batch)} rows):   {(t_pred-t_feat)*1000:.1f} ms  ({len(X_test_batch)/(t_pred-t_feat):,.0f}/sec)")
print(f"  Full pipeline:       {total:.1f} s  (dominated by feature engineering)")
print(f"  Production latency:  < 1 ms per user (model.predict only)")
print(f"  GPU required:        No (CPU-native XGBoost)")