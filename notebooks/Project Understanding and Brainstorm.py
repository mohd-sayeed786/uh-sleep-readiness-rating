# Databricks notebook source
# DBTITLE 1,Project Overview
# MAGIC %md
# MAGIC # Readiness Score — Project Understanding & Brainstorm
# MAGIC
# MAGIC Predict how a user will report feeling when they wake up (`subjective_feeling`, 1–5) from overnight ring sensor data + daily context. 120 users × ~90 nights. This notebook explores data quality, relationships, risks, and recommends a modelling approach.

# COMMAND ----------

# DBTITLE 1,Load all datasets
import pandas as pd
import numpy as np
import os

DATA_DIR = "/Workspace/Users/mohammad.sayeed@tatadigital.com/AS_UH_Project/data"

# Load all five files
user_profiles = pd.read_csv(os.path.join(DATA_DIR, "user_profiles.csv"))
sleep_sessions = pd.read_csv(os.path.join(DATA_DIR, "sleep_sessions.csv"))
daily_context = pd.read_csv(os.path.join(DATA_DIR, "daily_context.csv"))
morning_checkins = pd.read_csv(os.path.join(DATA_DIR, "morning_checkins.csv"))
nightly_signals = pd.read_csv(os.path.join(DATA_DIR, "nightly_signals.csv.gz"))

datasets = {
    "user_profiles": user_profiles,
    "sleep_sessions": sleep_sessions,
    "daily_context": daily_context,
    "morning_checkins": morning_checkins,
    "nightly_signals": nightly_signals,
}

for name, df in datasets.items():
    print(f"\n{'='*60}")
    print(f"  {name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"{'='*60}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes.to_string()}")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
    print(f"\nSample (3 rows):")
    display(df.head(3))

# COMMAND ----------

# DBTITLE 1,Target distribution & basic stats
# ── TARGET DISTRIBUTION ──
print("=" * 60)
print("  TARGET: morning_checkins.subjective_feeling")
print("=" * 60)
print(f"\nValue counts:")
vc = morning_checkins["subjective_feeling"].value_counts().sort_index()
for val, cnt in vc.items():
    pct = cnt / len(morning_checkins) * 100
    print(f"  {val}: {cnt:>5} ({pct:5.1f}%)")

print(f"\nMean:   {morning_checkins['subjective_feeling'].mean():.3f}")
print(f"Median: {morning_checkins['subjective_feeling'].median():.1f}")
print(f"Std:    {morning_checkins['subjective_feeling'].std():.3f}")
print(f"Users:  {morning_checkins['user_id'].nunique()}")
print(f"Dates:  {morning_checkins['date'].nunique()}")
print(f"Date range: {morning_checkins['date'].min()} to {morning_checkins['date'].max()}")

# Per-user stats
per_user = morning_checkins.groupby("user_id")["subjective_feeling"].agg(["count", "mean", "std"])
print(f"\nPer-user check-in counts:")
print(per_user["count"].describe().to_string())
print(f"\nPer-user mean feeling:")
print(per_user["mean"].describe().to_string())
print(f"\nPer-user std feeling:")
print(per_user["std"].describe().to_string())

# COMMAND ----------

# DBTITLE 1,Sleep sessions profiling
# ── SLEEP SESSIONS ──
print("Unique users:", sleep_sessions["user_id"].nunique())
print("Unique sessions:", sleep_sessions["session_id"].nunique())
print("Duplicate session_ids:", sleep_sessions["session_id"].duplicated().sum())

# Check date parsing — mixed tz-aware and tz-naive formats
sleep_sessions["_start"] = pd.to_datetime(sleep_sessions["session_start"], errors="coerce", utc=True)
sleep_sessions["_end"] = pd.to_datetime(sleep_sessions["session_end"], errors="coerce", utc=True)
print(f"\nsession_start parse failures: {sleep_sessions['_start'].isna().sum()}")
print(f"session_end parse failures: {sleep_sessions['_end'].isna().sum()}")

# Show sample of unparseable timestamps
bad_starts = sleep_sessions[sleep_sessions["_start"].isna()]["session_start"].head(5)
print(f"\nSample unparseable session_start values: {bad_starts.tolist()}")

# Date range (only parsed rows)
valid = sleep_sessions[sleep_sessions["_start"].notna()]
print(f"Date range (parsed): {valid['_start'].min()} to {valid['_end'].max()}")

# Duration consistency
sleep_sessions["_calc_duration"] = (sleep_sessions["_end"] - sleep_sessions["_start"]).dt.total_seconds() / 60
print(f"\nCalculated vs reported time_in_bed_minutes:")
diff = (sleep_sessions["_calc_duration"] - sleep_sessions["time_in_bed_minutes"]).dropna()
print(f"  Mean diff: {diff.mean():.2f} min")
print(f"  Max diff:  {diff.max():.2f} min")
print(f"  Rows with >5min diff: {(diff.abs() > 5).sum()}")

# user_id casing check
print(f"\nuser_id casing: {sleep_sessions['user_id'].str[:3].value_counts().to_dict()}")
sleep_sessions["user_id_norm"] = sleep_sessions["user_id"].str.upper()
print(f"Unique users after normalizing case: {sleep_sessions['user_id_norm'].nunique()}")

# Multiple sessions per night?
sleep_sessions["_night_date"] = sleep_sessions["_start"].dt.date
multi = sleep_sessions.groupby(["user_id_norm", "_night_date"]).size()
print(f"\nMultiple sessions per user-night: {(multi > 1).sum()} user-nights")
print(f"Max sessions per user-night: {multi.max()}")

# Numeric distributions
numeric_cols = ["time_in_bed_minutes", "total_sleep_minutes", "deep_minutes", "rem_minutes",
                "light_minutes", "awake_minutes", "sleep_efficiency", "avg_hr_bpm",
                "min_hr_bpm", "avg_hrv_rmssd_ms", "avg_spo2_pct", "avg_resp_rate_bpm",
                "temperature_deviation_c", "awakenings_count", "movement_index"]
print("\nNumeric summary:")
display(sleep_sessions[numeric_cols].describe().T)

# COMMAND ----------

# DBTITLE 1,Daily context profiling
# ── DAILY CONTEXT ──
print("Unique users:", daily_context["userId"].nunique())
print("Date range:", daily_context["date"].min(), "to", daily_context["date"].max())
print("Unique dates:", daily_context["date"].nunique())
print(f"Note: user_id column is named 'userId' (camelCase vs snake_case elsewhere)")

# Numeric distributions
for col in ["steps", "active_minutes", "alcohol_units", "caffeine_mg",
            "last_caffeine_hours_before_bed", "stress_score", "travel_flag",
            "legacy_readiness_shown"]:
    if col in daily_context.columns:
        s = daily_context[col]
        print(f"\n{col}: nulls={s.isna().sum()}, mean={s.mean():.2f}, "
              f"min={s.min():.2f}, max={s.max():.2f}, std={s.std():.2f}")

# Workout notes
print(f"\nworkout_notes: nulls={daily_context['workout_notes'].isna().sum()}, "
      f"non-null={daily_context['workout_notes'].notna().sum()}")
if daily_context["workout_notes"].notna().any():
    print("Sample workout_notes:")
    print(daily_context["workout_notes"].dropna().head(5).tolist())

# COMMAND ----------

# DBTITLE 1,User profiles & nightly signals profiling
# ── USER PROFILES ──
print("=" * 60)
print("  USER PROFILES")
print("=" * 60)
print(f"Users: {len(user_profiles)}")
print(f"Duplicate user_ids: {user_profiles['user_id'].duplicated().sum()}")
print(f"\nAge: {user_profiles['age_years'].describe().to_string()}")
print(f"\nSex distribution:\n{user_profiles['sex'].value_counts().to_string()}")
print(f"\nWeight: {user_profiles['weight'].describe().to_string()}")
print(f"\nHeight: {user_profiles['height_cm'].describe().to_string()}")
print(f"\nPlan tier:\n{user_profiles['plan_tier'].value_counts().to_string()}")
print(f"\nTimezones: {user_profiles['timezone'].nunique()} unique")
print(user_profiles['timezone'].value_counts().head(10).to_string())
print(f"\nNulls:\n{user_profiles.isnull().sum().to_string()}")

# ── NIGHTLY SIGNALS (sample-based) ──
print("\n" + "=" * 60)
print("  NIGHTLY SIGNALS")
print("=" * 60)
print(f"Rows: {len(nightly_signals):,}")
print(f"Users: {nightly_signals['user_id'].nunique()}")
print(f"Sessions: {nightly_signals['session_id'].nunique()}")

# Readings per session
rps = nightly_signals.groupby("session_id").size()
print(f"\nReadings per session: mean={rps.mean():.1f}, min={rps.min()}, max={rps.max()}, median={rps.median():.0f}")

# Signal distributions
for col in ["hr_bpm", "hrv_rmssd_ms", "motion_index", "temp_delta_c", "spo2_pct"]:
    s = nightly_signals[col]
    print(f"\n{col}: nulls={s.isna().sum():,}, mean={s.mean():.2f}, "
          f"min={s.min():.2f}, max={s.max():.2f}, std={s.std():.2f}")

# COMMAND ----------

# DBTITLE 1,Data joins & alignment check
# ── KEY ALIGNMENT CHECKS ──
print("=" * 60)
print("  DATA JOINS & ALIGNMENT")
print("=" * 60)

# User coverage across tables
users_profiles = set(user_profiles["user_id"])
users_sessions = set(sleep_sessions["user_id"])
users_context = set(daily_context["userId"])
users_checkins = set(morning_checkins["user_id"])
users_signals = set(nightly_signals["user_id"])

print(f"Users in profiles:     {len(users_profiles)}")
print(f"Users in sessions:     {len(users_sessions)}")
print(f"Users in daily_context: {len(users_context)}")
print(f"Users in checkins:     {len(users_checkins)}")
print(f"Users in signals:      {len(users_signals)}")
print(f"\nIn checkins but NOT in profiles: {len(users_checkins - users_profiles)}")
print(f"In profiles but NOT in checkins: {len(users_profiles - users_checkins)}")
print(f"In sessions but NOT in checkins: {len(users_sessions - users_checkins)}")
print(f"In checkins but NOT in sessions: {len(users_checkins - users_sessions)}")

# Date alignment: morning_checkins.date vs sleep_sessions night
# The checkin date = morning the user answered. The sleep session should be the NIGHT BEFORE.
# i.e., checkin date 2024-03-15 should match session that ENDED on 2024-03-15 morning
morning_checkins["_date"] = pd.to_datetime(morning_checkins["date"])
sleep_sessions["_end_date"] = sleep_sessions["_end"].dt.date
sleep_sessions["_end_date_str"] = sleep_sessions["_end_date"].astype(str)

# Try matching checkin date to session end date
merge_test = morning_checkins.merge(
    sleep_sessions[["user_id", "session_id", "_end_date_str"]],
    left_on=["user_id", "date"],
    right_on=["user_id", "_end_date_str"],
    how="left"
)
print(f"\nCheckins matched to session (by end_date): {merge_test['session_id'].notna().sum()} / {len(merge_test)}")
print(f"Unmatched checkins: {merge_test['session_id'].isna().sum()}")

# Daily context date alignment
# daily_context.date = "waking day" (before that night's sleep)
# So for a checkin on morning of 2024-03-15, the relevant daily_context is date=2024-03-14 (previous day)
morning_checkins["_prev_date"] = (morning_checkins["_date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
daily_context["_date_str"] = daily_context["date"].astype(str)

merge_ctx = morning_checkins.merge(
    daily_context[["userId", "_date_str"]],
    left_on=["user_id", "_prev_date"],
    right_on=["userId", "_date_str"],
    how="left"
)
print(f"\nCheckins matched to daily_context (prev day): {merge_ctx['userId'].notna().sum()} / {len(merge_ctx)}")

# Also try same-day match for comparison
merge_ctx_same = morning_checkins.merge(
    daily_context[["userId", "_date_str"]],
    left_on=["user_id", "date"],
    right_on=["userId", "_date_str"],
    how="left"
)
print(f"Checkins matched to daily_context (same day):  {merge_ctx_same['userId'].notna().sum()} / {len(merge_ctx_same)}")

# Nightly signals coverage
sessions_with_signals = set(nightly_signals["session_id"])
sessions_all = set(sleep_sessions["session_id"])
print(f"\nSessions with signals: {len(sessions_with_signals)} / {len(sessions_all)}")
print(f"Sessions in signals but not in sleep_sessions: {len(sessions_with_signals - sessions_all)}")

# COMMAND ----------

# DBTITLE 1,Deep dive: leakage, outliers, and signal quality
# ── DEEP DIVE: LEAKAGE, OUTLIERS, SIGNAL QUALITY ──
import warnings; warnings.filterwarnings('ignore')

# 1. LEAKAGE CHECK: legacy_readiness_shown correlation with target
merged_leak = morning_checkins.merge(
    daily_context.rename(columns={"userId": "user_id"}),
    left_on=["user_id", "date"],   # same-day context (legacy_readiness is shown "on the morning after this date")
    right_on=["user_id", "date"],
    how="inner"
)
print("=" * 60)
print("  LEAKAGE RISK: legacy_readiness_shown")
print("=" * 60)
# But wait - daily_context.date = "waking day" and legacy_readiness_shown is "on the morning after this date"
# So if checkin date = 2026-02-15 morning, the legacy score for THAT morning is in daily_context date=2026-02-14
morning_checkins["_prev_date"] = (pd.to_datetime(morning_checkins["date"]) - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
merged_leak2 = morning_checkins.merge(
    daily_context.rename(columns={"userId": "user_id"}),
    left_on=["user_id", "_prev_date"],
    right_on=["user_id", "date"],
    how="inner",
    suffixes=("_checkin", "_ctx")
)
valid_leak = merged_leak2[merged_leak2["legacy_readiness_shown"].notna()]
corr = valid_leak["subjective_feeling"].corr(valid_leak["legacy_readiness_shown"])
print(f"Correlation (target vs legacy_readiness): {corr:.4f}")
print(f"Rows with legacy_readiness: {len(valid_leak)} / {len(morning_checkins)}")
print(f"\n⚠ The legacy score is what users SEE before answering → potential anchoring effect")
print(f"⚠ Using it as a feature = information leakage if we're replacing this heuristic")

# 2. PHYSIOLOGICAL OUTLIERS in sleep_sessions
print(f"\n{'='*60}")
print(f"  OUTLIER DETECTION")
print(f"{'='*60}")
for col, lo, hi, label in [
    ("avg_hr_bpm", 30, 120, "Heart rate"),
    ("min_hr_bpm", 25, 100, "Min heart rate"),
    ("avg_hrv_rmssd_ms", 1, 300, "HRV"),
    ("avg_spo2_pct", 80, 100, "SpO2"),
    ("avg_resp_rate_bpm", 6, 30, "Resp rate"),
    ("time_in_bed_minutes", 60, 900, "Time in bed"),
    ("sleep_efficiency", 0.3, 1.0, "Sleep efficiency"),
]:
    s = sleep_sessions[col]
    bad = ((s < lo) | (s > hi)).sum()
    print(f"  {label} ({col}): {bad} rows outside [{lo}, {hi}]")

# 3. NIGHTLY SIGNALS null pattern — are nulls session-level or random?
null_per_session = nightly_signals.groupby("session_id").apply(
    lambda g: g[["hr_bpm", "hrv_rmssd_ms", "spo2_pct"]].isna().all().all()
)
print(f"\n{'='*60}")
print(f"  NIGHTLY SIGNALS NULL PATTERN")
print(f"{'='*60}")
print(f"Sessions where ALL signal values are null: {null_per_session.sum()}")
print(f"Total sessions in signals: {nightly_signals['session_id'].nunique()}")

# 4. MOOD TAGS analysis
print(f"\n{'='*60}")
print(f"  MOOD TAGS")
print(f"{'='*60}")
all_tags = morning_checkins["mood_tags"].str.split(";").explode().str.strip().str.lower()
tag_counts = all_tags.value_counts()
print(f"Unique tags: {tag_counts.shape[0]}")
print(f"Top 20 tags:")
for tag, cnt in tag_counts.head(20).items():
    print(f"  {tag}: {cnt}")

# 5. PER-USER VARIANCE — is this more within-user or between-user?
print(f"\n{'='*60}")
print(f"  VARIANCE DECOMPOSITION")
print(f"{'='*60}")
global_mean = morning_checkins["subjective_feeling"].mean()
per_user_stats = morning_checkins.groupby("user_id")["subjective_feeling"].agg(["mean", "count", "var"])
between_var = np.average((per_user_stats["mean"] - global_mean)**2, weights=per_user_stats["count"])
within_var = np.average(per_user_stats["var"].dropna(), weights=per_user_stats.loc[per_user_stats["var"].notna(), "count"])
total_var = morning_checkins["subjective_feeling"].var()
print(f"Total variance:   {total_var:.4f}")
print(f"Between-user var: {between_var:.4f} ({between_var/total_var*100:.1f}%)")
print(f"Within-user var:  {within_var:.4f} ({within_var/total_var*100:.1f}%)")
print(f"\n→ {'Within' if within_var > between_var else 'Between'}-user variance dominates")
print(f"→ Implication: {'Personalised baselines important' if between_var/total_var > 0.1 else 'User baselines less critical'}")

# COMMAND ----------

# DBTITLE 1,Epoch timestamp and weight unit investigation
# ── DEEPER DATA QUALITY CHECKS ──
print("=" * 60)
print("  EPOCH TIMESTAMP INVESTIGATION")
print("=" * 60)

# Those unparseable timestamps look like epoch millis
epoch_mask = sleep_sessions["_start"].isna()
epoch_rows = sleep_sessions[epoch_mask]
print(f"Rows with epoch-like timestamps: {len(epoch_rows)}")
if len(epoch_rows) > 0:
    sample_ts = epoch_rows["session_start"].head(5).tolist()
    print(f"Sample values: {sample_ts}")
    # Try parsing as epoch millis
    for ts in sample_ts[:3]:
        try:
            dt = pd.to_datetime(int(ts), unit="ms")
            print(f"  {ts} → {dt}")
        except:
            print(f"  {ts} → FAILED")

# Are epoch-timestamp sessions from specific users?
print(f"\nUsers with epoch timestamps: {epoch_rows['user_id'].nunique()}")
print(f"User_id pattern: {epoch_rows['user_id'].str[:5].value_counts().to_dict()}")

# Weight units investigation
print(f"\n{'='*60}")
print(f"  WEIGHT UNITS INVESTIGATION")
print(f"{'='*60}")
print(f"Weight distribution:")
print(user_profiles["weight"].describe().to_string())
print(f"\nWeights > 120 (possibly in lbs):")
heavy = user_profiles[user_profiles["weight"] > 120]
print(f"  Count: {len(heavy)}")
if len(heavy) > 0:
    display(heavy[["user_id", "weight", "height_cm", "sex"]].head(10))

# Firmware version distribution
print(f"\n{'='*60}")
print(f"  FIRMWARE VERSIONS")
print(f"{'='*60}")
print(sleep_sessions["firmware_version"].value_counts().to_string())

# Duplicate session_id investigation
print(f"\n{'='*60}")
print(f"  DUPLICATE SESSION IDS")
print(f"{'='*60}")
dupes = sleep_sessions[sleep_sessions["session_id"].duplicated(keep=False)]
print(f"Rows involved in duplicates: {len(dupes)}")
print(f"Unique duplicated session_ids: {dupes['session_id'].nunique()}")
# Are they exact duplicates?
sample_dup_id = dupes["session_id"].value_counts().index[0]
sample_dup = dupes[dupes["session_id"] == sample_dup_id]
print(f"\nSample duplicate (session_id={sample_dup_id}):")
display(sample_dup[["session_id", "user_id", "session_start", "session_end", "total_sleep_minutes"]])

# COMMAND ----------

# DBTITLE 1,Summary of findings
# MAGIC %md
# MAGIC ## Summary of Confirmed Observations
# MAGIC
# MAGIC ### Business Objective
# MAGIC * **Goal**: Predict morning subjective feeling (1–5) from overnight ring sensor data + daily context
# MAGIC * **Motivation**: Replace hand-tuned readiness heuristic with a learned model
# MAGIC * **Deployment**: Must work on-device (phone) or server, shown each morning
# MAGIC
# MAGIC ### Dataset Shape
# MAGIC | Table | Rows | Users | Key |
# MAGIC | --- | --- | --- | --- |
# MAGIC | morning\_checkins (TARGET) | 5,048 | 120 | user\_id × date |
# MAGIC | sleep\_sessions | 11,438 | 469 raw → 120 relevant | session\_id |
# MAGIC | nightly\_signals | 942,796 | 120 | session\_id × timestamp |
# MAGIC | daily\_context | 10,800 | 120 | userId × date |
# MAGIC | user\_profiles | 126 | 120 unique + 6 dupes | user\_id |
# MAGIC
# MAGIC ### Target Distribution
# MAGIC * Heavily centred on 3 (37.2%), roughly symmetric: 1(4.9%), 2(16.3%), 3(37.2%), 4(30.4%), 5(11.2%)
# MAGIC * Mean=3.27, Std=1.02
# MAGIC * Per-user mean range: 2.67–3.91 (narrow between-user spread)
# MAGIC * **Variance decomposition**: 95.5% within-user, 6.8% between-user → night-to-night variation is what matters, not who the user is
# MAGIC
# MAGIC ### Data Quality Issues Found
# MAGIC 1. **user\_id casing**: `UH-`, `uh-`, `  U` (leading spaces) → must normalise before joins
# MAGIC 2. **Epoch timestamps**: 1,731 sessions have millis-epoch instead of ISO format (75 users affected)
# MAGIC 3. **Duplicate session\_ids**: 221 exact duplicates → simple dedup
# MAGIC 4. **349 extra users** in sleep\_sessions not in checkins → noise/test users to filter
# MAGIC 5. **Mixed timezone formats**: some ISO with offset, some naive → parse with `utc=True` then convert
# MAGIC 6. **Weight units**: 12 users > 120 likely in lbs vs kg
# MAGIC 7. **height\_cm**: 5 nulls
# MAGIC 8. **daily\_context.userId**: camelCase naming inconsistency
# MAGIC 9. **Physiological outliers**: 34 HR, 57 HRV (999 sentinels), 1,154 time-in-bed, 29 efficiency
# MAGIC 10. **High nulls**: stress\_score (65.7%), caffeine (20%), alcohol (4.8%), legacy\_readiness (3%)
# MAGIC 11. **Multi-session nights**: 984 user-nights with 2–4 sessions (naps, fragmented sleep)
# MAGIC 12. **Firmware versions**: 3 versions — potential sensor calibration differences
# MAGIC
# MAGIC ### Date Alignment Logic
# MAGIC * morning\_checkins.date = morning the user answered
# MAGIC * daily\_context.date = the **waking day** (before that night's sleep)
# MAGIC * Therefore: checkin on 2026-02-15 morning → daily\_context date=2026-02-14 (100% match confirmed)
# MAGIC * sleep\_session should have ended on the checkin morning date
# MAGIC
# MAGIC ### Critical Leakage Risk
# MAGIC * `legacy_readiness_shown` correlates **0.77** with the target
# MAGIC * This IS the current heuristic score shown to users each morning
# MAGIC * Using it as a feature = circular (we're trying to replace it)
# MAGIC * Additionally: users may anchor their self-report to the score they saw → label contamination
# MAGIC * **Decision**: Exclude from modelling features, but use as a **benchmark** ("can we beat the heuristic?")

# COMMAND ----------

# DBTITLE 1,Assumptions and hypotheses
# MAGIC %md
# MAGIC ## Assumptions (to validate)
# MAGIC
# MAGIC 1. **Single "main" session per night**: For multi-session nights, the longest session is the actual sleep session; shorter ones are naps or false recordings
# MAGIC 2. **Epoch timestamps are recoverable**: The millis-epoch values parse to valid dates within the dataset range
# MAGIC 3. **Weights > 120 are in lbs**: Can convert using BMI reasonableness check against height
# MAGIC 4. **HRV=999 is a sentinel**: Not a real measurement; same for HR=0
# MAGIC 5. **Missing stress\_score means not answered**: Safe to impute with median or create a binary "reported" flag
# MAGIC 6. **Nightly signals nulls are not session-level**: Confirmed — no session has ALL nulls, so nulls are sporadic sensor dropout
# MAGIC 7. **Mood tags are informative but NOT prediction-time inputs**: They're recorded at the same time as the target → can't use as features
# MAGIC 8. **`submitted_at` timing is not a feature**: It reveals when the user answered, not how they slept
# MAGIC
# MAGIC ## Hypotheses (to test in modelling)
# MAGIC
# MAGIC 1. **Sleep quality signals (HRV, HR, efficiency) will be strongest predictors** — they directly measure recovery
# MAGIC 2. **Previous-day context (steps, caffeine timing, alcohol) will add marginal value** — especially caffeine close to bed
# MAGIC 3. **Temperature deviation may capture illness/inflammation** — deviation from personal baseline is meaningful
# MAGIC 4. **Sequence patterns in nightly signals may help but won't dramatically improve over session-level aggregates** — the 5-min resolution captures HRV trajectories but the low sample size (5k rows) limits deep learning
# MAGIC 5. **Day-of-week effects may exist** — weekend sleep differs from weekday
# MAGIC 6. **User-level random effects are small** (6.8% of variance) but still worth encoding as user embeddings or personalised baselines in V2

# COMMAND ----------

# DBTITLE 1,Modelling approach comparison
# ── MODELLING APPROACH COMPARISON (Decision Matrix) ──
import pandas as pd

approaches = pd.DataFrame([
    {
        "Approach": "0. Naive Baseline (predict user mean)",
        "Type": "Baseline",
        "Pros": "No features needed; captures user tendency; unbeatable if between-user var dominates",
        "Cons": "Between-user var is only 6.8%; ignores nightly variation",
        "Data Needs": "~10+ nights per user",
        "Complexity": "Trivial",
        "Recommended": "YES — Floor to beat"
    },
    {
        "Approach": "0b. Predict global mean (3.27)",
        "Type": "Baseline",
        "Pros": "Simplest possible; works for new users",
        "Cons": "Ignores everything",
        "Data Needs": "None",
        "Complexity": "Trivial",
        "Recommended": "YES — Absolute floor"
    },
    {
        "Approach": "1. Linear/Ridge Regression",
        "Type": "Regression",
        "Pros": "Interpretable; fast; handles continuous target well; reveals feature importance directly",
        "Cons": "Assumes linearity; ordinal nature ignored; can predict outside 1-5",
        "Data Needs": "~5k rows is adequate",
        "Complexity": "Low",
        "Recommended": "YES — V1 candidate"
    },
    {
        "Approach": "2. Ordinal Regression (mord/proportional odds)",
        "Type": "Ordinal",
        "Pros": "Respects ordered nature of 1-5; natural cutpoints; interpretable thresholds",
        "Cons": "Less common; fewer libraries; proportional odds assumption may not hold",
        "Data Needs": "5k rows sufficient",
        "Complexity": "Medium",
        "Recommended": "CONSIDER — V1 alt"
    },
    {
        "Approach": "3. Classification (5-class)",
        "Type": "Classification",
        "Pros": "No distributional assumptions; can capture non-linear thresholds",
        "Cons": "Ignores ordering (predicting 1 vs 2 penalised same as 1 vs 5); class imbalance (class 1 = 4.9%)",
        "Data Needs": "5k rows marginal for 5 classes",
        "Complexity": "Low",
        "Recommended": "NO for V1 — ordinal is better"
    },
    {
        "Approach": "4. Gradient Boosted Trees (XGBoost/LightGBM)",
        "Type": "Tree ensemble",
        "Pros": "Handles nonlinearity, interactions, missing values natively; strong baseline; interpretable via SHAP",
        "Cons": "Can overfit on 5k rows; needs careful CV; treats target as continuous or classification",
        "Data Needs": "5k rows is tight but feasible with regularisation",
        "Complexity": "Medium",
        "Recommended": "YES — V1 primary"
    },
    {
        "Approach": "5. Personalised Baseline + Residual Model",
        "Type": "Hybrid",
        "Pros": "Captures user baseline + nightly deviations; conceptually clean; handles cold-start by falling back to global",
        "Cons": "Two-stage complexity; user baseline needs 10+ nights",
        "Data Needs": "Median 41 nights/user — enough",
        "Complexity": "Medium",
        "Recommended": "YES — V1 enhancement"
    },
    {
        "Approach": "6. 1D-CNN / LSTM on nightly signals",
        "Type": "Sequence/DL",
        "Pros": "Captures temporal patterns in 5-min signals (HRV trajectory, sleep stage transitions)",
        "Cons": "Only 5k labelled sequences; 942k signal rows but grouped into ~5k sessions; high risk of overfitting; slow to iterate",
        "Data Needs": "Marginal — typically need 10k+ sequences for DL",
        "Complexity": "High",
        "Recommended": "V2 only — compare feature extraction approach first"
    },
    {
        "Approach": "7. Mixed-Effects Model",
        "Type": "Statistical",
        "Pros": "Explicit random intercepts per user; handles repeated measures correctly; best for causal understanding",
        "Cons": "Linear; harder to deploy in production; fewer non-linear interactions",
        "Data Needs": "Ideal for this panel structure",
        "Complexity": "Medium",
        "Recommended": "V2 — for analysis, not deployment"
    },
])

display(approaches)

# COMMAND ----------

# DBTITLE 1,Recommended approach
# MAGIC %md
# MAGIC ## Recommended Approach
# MAGIC
# MAGIC ### 1. Naive Baseline (Floor)
# MAGIC **Predict the user's historical mean** (or global mean=3.27 for cold-start).
# MAGIC * Why: 95.5% of variance is within-user, so this captures only 6.8% — but it's the simplest floor to beat
# MAGIC * Metric: MAE, RMSE, and quadratic weighted kappa
# MAGIC * Expected: MAE \~0.8–0.9 (roughly 1 point off on average)
# MAGIC
# MAGIC ### 2. Best V1 Approach: LightGBM Regressor on Session-Level Features
# MAGIC **Why this combination:**
# MAGIC * **LightGBM** handles missing values natively (critical: stress\_score 65% missing, caffeine 20%) — no imputation guesswork
# MAGIC * **Regression** framing: target is ordinal 1–5, but treating as continuous regression with MAE/RMSE works well in practice and simplifies deployment. Round to nearest int for display.
# MAGIC * **Session-level features**: Aggregate nightly signals per session (mean, std, min, max, slope of HRV/HR/temp over the night) — this captures 80–90% of what a sequence model would find, at 1/100th the complexity
# MAGIC * **5,048 rows** is comfortable for gradient boosting with regularisation — well within the sweet spot
# MAGIC
# MAGIC **Feature groups for V1:**
# MAGIC 1. Sleep architecture: total\_sleep, deep\_min, rem\_min, efficiency, awakenings
# MAGIC 2. Physiological: avg/min HR, avg HRV, SpO2, resp rate, temp deviation
# MAGIC 3. Signal dynamics: HRV slope (first half vs second half of night), HR recovery curve, motion trajectory
# MAGIC 4. Daily context: steps, active\_minutes, alcohol, caffeine timing, travel\_flag
# MAGIC 5. Temporal: day-of-week, session duration relative to user's mean
# MAGIC 6. User-normalised: z-score features relative to user's own historical distribution (e.g., "HRV is 1.2σ above your 14-day mean")
# MAGIC
# MAGIC **Validation strategy:**
# MAGIC * **Time-based split** (last 2 weeks held out) — NOT random split, because temporal autocorrelation exists
# MAGIC * **Group-aware**: ensure some users are in test-only (to evaluate cold-start)
# MAGIC * **Nested CV for hyperparameter tuning** on the train portion
# MAGIC
# MAGIC ### 3. Advanced Options (V2+)
# MAGIC * **Ordinal regression layer**: Replace MSE loss with ordinal-aware loss (CORN/CORAL loss)
# MAGIC * **1D-CNN on nightly signals**: Extract learned features from the raw 5-min time series as an alternative to hand-crafted aggregates
# MAGIC * **User embeddings**: Learn a small (dim=4–8) embedding per user that captures personal baseline — handles the 6.8% between-user variance
# MAGIC * **Multi-task learning**: Predict both the feeling AND the mood tag category — may improve feature learning
# MAGIC * **Ensemble**: Stack LightGBM + ordinal regression + signal-CNN
# MAGIC
# MAGIC ### Why NOT Classification?
# MAGIC The target 1–5 is ordinal. Classification ignores that predicting 1 when truth is 2 is far better than predicting 1 when truth is 5. Ordinal regression or continuous regression naturally respect this.
# MAGIC
# MAGIC ### Why NOT Deep Learning First?
# MAGIC With only 5,048 labelled samples (and ~120 unique users), a 1D-CNN/LSTM over 942k signal rows is grouping into ~5k sequences. This is at the lower bound for deep learning. Hand-crafted signal features fed into LightGBM will likely match or exceed DL performance with 10x less effort. We can revisit in V2.

# COMMAND ----------

# DBTITLE 1,Open questions and quick experiments
# MAGIC %md
# MAGIC ## Open Questions Requiring Deeper Exploration
# MAGIC
# MAGIC 1. **How much does the legacy score anchor user responses?** — If correlation drops when we look at users who answer before seeing the score, anchoring is real
# MAGIC 2. **Is firmware version a confounder?** — v2.0.3 may have different sensor calibration; check if target distribution differs by firmware
# MAGIC 3. **What fraction of usable rows remain after cleaning?** — After removing epoch timestamps, deduplication, outlier filtering, and restricting to 120 target users
# MAGIC 4. **Do the 984 multi-session nights produce different targets?** — May indicate disrupted sleep → lower scores
# MAGIC 5. **Is there temporal drift in the target?** — Do users rate higher/lower as they get used to the ring?
# MAGIC 6. **How much information do the 5-min signals add over session-level aggregates?** — Quick test: LightGBM with session features only vs session + signal-derived features
# MAGIC
# MAGIC ## Quick Validation Experiments (Before Full Pipeline)
# MAGIC
# MAGIC 1. **Correlation matrix**: sleep\_session numeric features vs target — identify top 5 predictors
# MAGIC 2. **Legacy readiness as predictor**: Ridge regression with legacy\_readiness\_shown only → this IS the current heuristic, so its MAE is the bar to beat
# MAGIC 3. **User-mean baseline MAE**: For each user, predict their mean → measure MAE
# MAGIC 4. **LightGBM with just session features**: No signal features, no context → how far does raw sleep data get?
# MAGIC 5. **Feature importance from #4**: Which features matter most? HRV? Deep sleep? Efficiency?