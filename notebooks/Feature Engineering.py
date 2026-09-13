# Databricks notebook source
# DBTITLE 1,Feature Engineering — Readiness Score
# MAGIC %md
# MAGIC # Feature Engineering — Readiness Score
# MAGIC
# MAGIC Adds temporal lag, rolling window, recency/frequency/value, trend, and interaction features to the cleaned modelling table. Outputs `feature_engineered_table.parquet` for Model Selection.

# COMMAND ----------

# DBTITLE 1,Setup and load modelling table
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings("ignore")

DATA_DIR = "/Workspace/Users/mohammad.sayeed@tatadigital.com/AS_UH_Project/data"
df = pd.read_csv(f"{DATA_DIR}/modelling_table.csv")
df["checkin_date"] = pd.to_datetime(df["checkin_date"])
df = df.sort_values(["user_id", "checkin_date"]).reset_index(drop=True)

print(f"Loaded: {df.shape[0]:,} rows \u00d7 {df.shape[1]} cols")
print(f"Users: {df['user_id'].nunique()}, Date: {df['checkin_date'].min().date()} to {df['checkin_date'].max().date()}")
print(f"Target mean: {df['subjective_feeling'].mean():.3f}, std: {df['subjective_feeling'].std():.3f}")
initial_cols = df.shape[1]

# COMMAND ----------

# DBTITLE 1,Lag features (1, 2, 3-night lookback)
# ── 1. LAG FEATURES ──
# Previous nights' sleep metrics — strong predictors because sleep quality
# is auto-correlated (a bad night often follows a bad night).

lag_cols = ["total_sleep_minutes", "deep_minutes", "rem_minutes", "sleep_efficiency",
            "avg_hr_bpm", "avg_hrv_rmssd_ms", "movement_index", "awakenings_count",
            "bedtime_hour", "subjective_feeling"]

for lag in [1, 2, 3]:
    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_lag{lag}"] = df.groupby("user_id")[col].shift(lag)

# Lag of target = previous morning's self-reported feeling
# (not leakage because it was reported BEFORE this night's sleep)

n_lag = sum(1 for c in df.columns if "_lag" in c)
print(f"Lag features: {n_lag} columns (3 lags \u00d7 {len(lag_cols)} metrics)")
print(f"  NaN in lag1 (first checkin per user): {df['total_sleep_minutes_lag1'].isna().sum()}")

# COMMAND ----------

# DBTITLE 1,Rolling window features (3 and 7-night)
# ── 2. ROLLING WINDOW FEATURES (3-night and 7-night) ──
# Capture recent trends and stability. A person who slept well 3 of last
# 7 nights may feel differently than one who slept well 7/7.

roll_cols = ["total_sleep_minutes", "deep_minutes", "rem_minutes", "sleep_efficiency",
             "avg_hr_bpm", "avg_hrv_rmssd_ms", "movement_index", "steps",
             "alcohol_units", "bedtime_hour"]

for window in [3, 7]:
    for col in roll_cols:
        if col in df.columns:
            grp = df.groupby("user_id")[col]
            # Shift by 1 so we don't include the current night (leakage)
            shifted = grp.shift(1)
            df[f"{col}_roll{window}_mean"] = shifted.rolling(window, min_periods=1).mean()
            df[f"{col}_roll{window}_std"]  = shifted.rolling(window, min_periods=1).std()

# Rolling target (how has the person been feeling recently)
for window in [3, 7]:
    shifted_target = df.groupby("user_id")["subjective_feeling"].shift(1)
    df[f"feeling_roll{window}_mean"] = shifted_target.rolling(window, min_periods=1).mean()

n_roll = sum(1 for c in df.columns if "_roll" in c)
print(f"Rolling features: {n_roll} columns")
print(f"  Example: total_sleep_minutes_roll7_mean nulls = {df['total_sleep_minutes_roll7_mean'].isna().sum()}")

# COMMAND ----------

# DBTITLE 1,Recency features
# ── 3. RECENCY FEATURES ──
# How many days since a notable event? Recent bad sleep or alcohol may
# still affect this morning's mood.

def days_since_event(group, condition_col, threshold, direction="above"):
    """For each row, compute days since condition was last true."""
    result = pd.Series(np.nan, index=group.index)
    last_event = pd.NaT
    for i, (idx, row) in enumerate(group.iterrows()):
        val = row[condition_col]
        if pd.notna(val):
            if (direction == "above" and val >= threshold) or \
               (direction == "below" and val <= threshold):
                last_event = row["checkin_date"]
        if pd.notna(last_event) and last_event < row["checkin_date"]:
            result.iloc[i] = (row["checkin_date"] - last_event).days
    return result

print("Computing recency features (may take ~30s)...")

# Days since last bad sleep (feeling <= 2)
df["days_since_bad_sleep"] = df.groupby("user_id", group_keys=False).apply(
    lambda g: days_since_event(g, "subjective_feeling", 2, "below")
).values

# Days since last great sleep (feeling >= 4)
df["days_since_great_sleep"] = df.groupby("user_id", group_keys=False).apply(
    lambda g: days_since_event(g, "subjective_feeling", 4, "above")
).values

# Days since last alcohol
df["days_since_alcohol"] = df.groupby("user_id", group_keys=False).apply(
    lambda g: days_since_event(g, "alcohol_units", 0.5, "above")
).values

# Days since last workout
df["days_since_workout"] = df.groupby("user_id", group_keys=False).apply(
    lambda g: days_since_event(g, "had_workout", 1, "above")
).values

recency_cols = [c for c in df.columns if c.startswith("days_since_") and c != "days_since_last_checkin"]
print(f"Recency features: {len(recency_cols)} columns")
for c in recency_cols:
    print(f"  {c}: mean={df[c].mean():.1f}, nulls={df[c].isna().sum()}")

# COMMAND ----------

# DBTITLE 1,Frequency features (rolling counts in windows)
# ── 4. FREQUENCY FEATURES (rolling event counts) ──
# How often have events occurred in recent history?

# Count of workouts in last 3 and 7 check-ins
for window in [3, 7]:
    shifted = df.groupby("user_id")["had_workout"].shift(1)
    df[f"workout_count_{window}d"] = shifted.rolling(window, min_periods=1).sum()
    
    shifted_alc = df.groupby("user_id")["had_alcohol"].shift(1)
    df[f"alcohol_count_{window}d"] = shifted_alc.rolling(window, min_periods=1).sum()
    
    shifted_frag = df.groupby("user_id")["fragmented_night"].shift(1)
    df[f"fragmented_count_{window}d"] = shifted_frag.rolling(window, min_periods=1).sum()

# Cumulative check-in count per user (engagement proxy)
df["checkin_seq_num"] = df.groupby("user_id").cumcount() + 1

# Consecutive good/bad nights streak (LAGGED — uses only PREVIOUS nights,
# never the current row's target, to avoid data leakage)
def streak_count_lagged(group, col, threshold, direction):
    """Streak of PREVIOUS rows meeting condition. Current row excluded."""
    result = []
    streak = 0
    for val in group[col]:
        result.append(streak)          # record streak BEFORE seeing this row
        if pd.notna(val):
            if (direction == "above" and val >= threshold) or \
               (direction == "below" and val <= threshold):
                streak += 1
            else:
                streak = 0
    return pd.Series(result, index=group.index)

df["good_sleep_streak"] = df.groupby("user_id", group_keys=False).apply(
    lambda g: streak_count_lagged(g, "subjective_feeling", 4, "above")
).values
df["bad_sleep_streak"] = df.groupby("user_id", group_keys=False).apply(
    lambda g: streak_count_lagged(g, "subjective_feeling", 2, "below")
).values

n_freq = sum(1 for c in df.columns if "_count_" in c or "streak" in c or c == "checkin_seq_num")
print(f"Frequency features: {n_freq} columns")

# COMMAND ----------

# DBTITLE 1,Trend features (7-night slopes)
# ── 5. TREND FEATURES (linear slope over last 7 nights) ──
# Is sleep quality improving or degrading? Slope captures direction.

from scipy.stats import linregress

def rolling_slope(series, window=7):
    """Compute slope of linear fit over rolling window."""
    result = pd.Series(np.nan, index=series.index)
    vals = series.values
    for i in range(window - 1, len(vals)):
        chunk = vals[i - window + 1:i + 1]
        valid = ~np.isnan(chunk)
        if valid.sum() >= 3:  # need at least 3 points
            x = np.arange(window)[valid]
            y = chunk[valid]
            try:
                slope, _, _, _, _ = linregress(x, y)
                result.iloc[i] = slope
            except:
                pass
    return result

trend_cols = ["total_sleep_minutes", "deep_minutes", "sleep_efficiency",
              "avg_hr_bpm", "avg_hrv_rmssd_ms", "bedtime_hour", "subjective_feeling"]

print("Computing 7-night trend slopes...")
for col in trend_cols:
    if col in df.columns:
        # Shift by 1 to exclude current night
        df[f"{col}_trend7"] = df.groupby("user_id")[col].transform(
            lambda x: rolling_slope(x.shift(1), 7)
        )

n_trend = sum(1 for c in df.columns if "_trend7" in c)
print(f"Trend features: {n_trend} columns")
for c in [c for c in df.columns if "_trend7" in c]:
    print(f"  {c}: mean={df[c].mean():.4f}, nulls={df[c].isna().sum()}")

# COMMAND ----------

# DBTITLE 1,Interaction and ratio features
# ── 6. INTERACTION & RATIO FEATURES ──
# Domain-informed combinations that capture physiological relationships.

# Sleep quality composites
df["deep_rem_total"] = df["deep_minutes"].fillna(0) + df["rem_minutes"].fillna(0)
df["restorative_pct"] = (df["deep_rem_total"] / df["total_sleep_minutes"].replace(0, np.nan)).round(3)

# HR recovery proxy: lower HR + higher HRV = better recovery
df["hr_hrv_ratio"] = (df["avg_hr_bpm"] / df["avg_hrv_rmssd_ms"].replace(0, np.nan)).round(3)

# Sleep debt: deviation from user's own mean sleep
df["sleep_debt"] = df["total_sleep_minutes"] - df.groupby("user_id")["total_sleep_minutes"].transform("mean")

# Bedtime consistency: abs deviation from user's median bedtime
df["bedtime_consistency"] = abs(df["bedtime_hour"] - df.groupby("user_id")["bedtime_hour"].transform("median"))

# Alcohol on workout day interaction
df["alcohol_workout_combo"] = ((df["had_alcohol"].fillna(0) > 0) & (df["had_workout"] == 1)).astype(int)

# Weekend + alcohol interaction
df["weekend_alcohol"] = (df["is_weekend"] * df["alcohol_units"].fillna(0)).round(2)

# Sleep efficiency × duration (both matter for recovery)
df["efficiency_duration"] = (df["sleep_efficiency"].fillna(0) * df["total_sleep_minutes"].fillna(0) / 60).round(2)

# Temperature instability (deviation from 0 is bad)
df["temp_instability"] = df["temperature_deviation_c"].abs()

# Night-before-feeling change (delta from lag)
df["feeling_delta"] = df["subjective_feeling"] - df["subjective_feeling_lag1"]

n_interact = 10
print(f"Interaction features: {n_interact} columns")
for c in ["deep_rem_total", "restorative_pct", "hr_hrv_ratio", "sleep_debt",
          "bedtime_consistency", "efficiency_duration"]:
    print(f"  {c}: mean={df[c].mean():.2f}, nulls={df[c].isna().sum()}")

# COMMAND ----------

# DBTITLE 1,Stress categorical features (preserve 65% null signal)
# ── 6b. STRESS CATEGORICAL FEATURES ──
# stress_score is 65% null (MNAR — users report only when stressed).
# Instead of dropping it, convert to categorical: the MISSINGNESS itself
# is informative (not reporting ≈ lower stress).

def stress_to_category(val):
    if pd.isna(val):   return "not_reported"
    if val <= 3:       return "low"
    if val <= 6:       return "moderate"
    return "high"

df["stress_category"] = df["stress_score"].apply(stress_to_category)

# Also create a numeric version for tree models
stress_map = {"not_reported": 0, "low": 1, "moderate": 2, "high": 3}
df["stress_level_ord"] = df["stress_category"].map(stress_map)

# Interaction: stressed + alcohol (compounding negative)
df["stress_alcohol"] = (df["stress_level_ord"] * df["alcohol_units"].fillna(0)).round(2)

# Interaction: stressed + poor sleep (double hit)
df["stress_poor_sleep"] = (df["stress_level_ord"] * (1 - df["sleep_efficiency"].fillna(0.85))).round(4)

print(f"Stress features added:")
print(f"  stress_category distribution:")
for cat, n in df["stress_category"].value_counts().items():
    print(f"    {cat:15s}: {n:,} ({n/len(df)*100:.1f}%)")
print(f"  stress_level_ord: mean={df['stress_level_ord'].mean():.2f}")
print(f"  stress_alcohol: mean={df['stress_alcohol'].mean():.3f}, >0: {(df['stress_alcohol']>0).sum()}")
print(f"  stress_poor_sleep: mean={df['stress_poor_sleep'].mean():.4f}")

# COMMAND ----------

# DBTITLE 1,Consolidation, summary, and save
# ── 7. CONSOLIDATION & SAVE ──
import os

final_cols = df.shape[1]
new_features = final_cols - initial_cols

print(f"{'='*70}")
print(f"FEATURE ENGINEERING COMPLETE")
print(f"{'='*70}")
print(f"  Initial columns:  {initial_cols}")
print(f"  Final columns:    {final_cols}")
print(f"  New features:     {new_features}")
print(f"  Rows:             {len(df):,} (unchanged)")

# Feature group summary
groups = {
    "Lag features":        [c for c in df.columns if "_lag" in c],
    "Rolling features":    [c for c in df.columns if "_roll" in c],
    "Recency features":    [c for c in df.columns if c.startswith("days_since_") and c != "days_since_last_checkin"],
    "Frequency features":  [c for c in df.columns if "_count_" in c or "streak" in c or c == "checkin_seq_num"],
    "Trend features":      [c for c in df.columns if "_trend7" in c],
    "Interaction features": ["deep_rem_total", "restorative_pct", "hr_hrv_ratio", "sleep_debt",
                            "bedtime_consistency", "alcohol_workout_combo", "weekend_alcohol",
                            "efficiency_duration", "temp_instability", "feeling_delta"],
}
print(f"\nFeature groups:")
for name, cols in groups.items():
    existing = [c for c in cols if c in df.columns]
    print(f"  {name:25s}  {len(existing):>3} cols")

# Null summary
total_nulls = df.isnull().sum()
high_null = total_nulls[total_nulls > len(df) * 0.5]
if len(high_null):
    print(f"\n\u26a0 Columns with >50% nulls ({len(high_null)}):")
    for col, n in high_null.items():
        print(f"    {col}: {n} ({n/len(df)*100:.1f}%)")

# Save
out_path = os.path.join(DATA_DIR, "feature_engineered_table.parquet")
df.to_parquet(out_path, index=False)
fsize = os.path.getsize(out_path) / 1024 / 1024
print(f"\n\u2713 Saved: {out_path} ({fsize:.1f} MB)")

# Also CSV for inspection
df.to_csv(os.path.join(DATA_DIR, "feature_engineered_table.csv"), index=False)
print(f"\u2713 Saved: feature_engineered_table.csv")

# Show sample
print(f"\nSample (3 rows):")
display(df.head(3))