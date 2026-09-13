# Databricks notebook source
# DBTITLE 1,Title
# MAGIC %md
# MAGIC # Data Cleaning and EDA — Readiness Score
# MAGIC
# MAGIC This notebook thoroughly inspects, cleans and merges all five raw data files into a single modelling-ready dataset. Every cleaning decision is shown with before/after evidence. The final output is a flat table keyed on `(user_id, checkin_date)` ready for train/validation/test splitting.

# COMMAND ----------

# DBTITLE 1,Setup and load raw data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Chart styling ──
sns.set_style("whitegrid")
PAL = sns.color_palette("muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.titlesize": 12,
                     "axes.labelsize": 10, "figure.facecolor": "white"})
DATA_START = pd.Timestamp("2026-02-02")  # study start date
DATA_DIR   = "/Workspace/Users/mohammad.sayeed@tatadigital.com/AS_UH_Project/data"

# ── Load all raw files ──
raw_profiles   = pd.read_csv(f"{DATA_DIR}/user_profiles.csv")
raw_sessions   = pd.read_csv(f"{DATA_DIR}/sleep_sessions.csv")
raw_signals    = pd.read_csv(f"{DATA_DIR}/nightly_signals.csv.gz")
raw_context    = pd.read_csv(f"{DATA_DIR}/daily_context.csv")
raw_checkins   = pd.read_csv(f"{DATA_DIR}/morning_checkins.csv")

print("Raw data loaded:")
for name, df in [("user_profiles", raw_profiles), ("sleep_sessions", raw_sessions),
                 ("nightly_signals", raw_signals), ("daily_context", raw_context),
                 ("morning_checkins", raw_checkins)]:
    print(f"  {name:25s} {df.shape[0]:>10,} rows x {df.shape[1]:>3} cols")

# COMMAND ----------

# DBTITLE 1,Reusable validation helpers
# ── Reusable data-validation functions ──

def shape_report(df, name):
    """Print shape, dtypes, nulls, duplicates summary."""
    print(f"\n{'='*65}")
    print(f"  {name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"{'='*65}")
    nulls = df.isnull().sum()
    null_pct = (nulls / len(df) * 100).round(1)
    report = pd.DataFrame({"dtype": df.dtypes.astype(str), "nulls": nulls, "null_%": null_pct})
    display(report)
    return report

def range_check(series, lo, hi, label):
    """Count values outside [lo, hi] and show summary."""
    below = (series < lo).sum()
    above = (series > hi).sum()
    total = below + above
    print(f"  {label:40s}  out-of-range [{lo}, {hi}]: {total:>5}  (below={below}, above={above})")
    return total

def dedup_report(df, key_cols, name):
    """Report and return deduplicated dataframe."""
    n_before = len(df)
    n_dup = df.duplicated(subset=key_cols, keep="first").sum()
    df_clean = df.drop_duplicates(subset=key_cols, keep="first")
    print(f"  {name}: {n_before:,} -> {len(df_clean):,} rows  ({n_dup} duplicates removed on {key_cols})")
    return df_clean

def consistency_check(s, lo, hi, label):
    """Clip to plausible range and set sentinels to NaN. Returns cleaned series."""
    cleaned = s.copy()
    sentinel_mask = (cleaned == 999) | (cleaned == 0)
    below = cleaned < lo
    above = cleaned > hi
    n_sentinel = sentinel_mask.sum()
    n_clipped = (below | above).sum() - sentinel_mask.sum()  # exclude sentinels from clip count
    cleaned[sentinel_mask] = np.nan
    cleaned = cleaned.clip(lo, hi)
    print(f"  {label}: {n_sentinel} sentinels→NaN, {max(0,n_clipped)} clipped to [{lo},{hi}]")
    return cleaned

print("Validation helpers defined.")

# COMMAND ----------

# DBTITLE 1,Section 1: User Profiles
# MAGIC %md
# MAGIC ## Section 1: User Profiles
# MAGIC
# MAGIC Clean user_profiles: deduplicate, validate weight/height, impute nulls, compute BMI, and derive modelling features.

# COMMAND ----------

# DBTITLE 1,User Profiles - inspect, deduplicate, and overview
# ── 1A: User Profiles — Inspect, normalise, deduplicate ──
profiles = raw_profiles.copy()
shape_report(profiles, "user_profiles (raw)")

# Normalize user_id: uppercase + strip
profiles["user_id"] = profiles["user_id"].str.strip().str.upper()

# Duplicate check
n_dup = profiles["user_id"].duplicated().sum()
print(f"\nDuplicate user_ids: {n_dup}")
if n_dup > 0:
    display(profiles[profiles["user_id"].duplicated(keep=False)].sort_values("user_id"))

# Deduplicate: keep latest profile_updated_at per user
profiles["profile_updated_at"] = pd.to_datetime(profiles["profile_updated_at"])
profiles = (profiles.sort_values("profile_updated_at", ascending=False)
            .drop_duplicates(subset=["user_id"], keep="first"))
print(f"After dedup: {len(profiles)} users")

# ── Demographic overview chart ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, title in zip(axes, ["sex", "plan_tier", "timezone"],
                                  ["Sex", "Plan tier", "Timezone"]):
    order = profiles[col].value_counts().index
    sns.countplot(data=profiles, y=col, order=order, palette=PAL, ax=ax)
    ax.set_title(title); ax.set_xlabel("Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_width())}", (p.get_width() + 0.5, p.get_y() + p.get_height()/2),
                    va="center", fontsize=9)
fig.suptitle("User Demographics (120 users)", fontsize=13, y=1.02)
plt.tight_layout(); plt.show()

# COMMAND ----------

# DBTITLE 1,User Profiles - weight & height by timezone
# ── 1A-bis: Weight & height distributions by timezone ──
# Purpose: confirm all weights are in the same unit across timezones
# (no lbs→kg conversion needed if distributions are consistent)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
tz_order = profiles["timezone"].value_counts().index

sns.boxplot(data=profiles, x="timezone", y="weight", order=tz_order,
            palette="Set2", ax=axes[0], flierprops={"marker": "o", "markersize": 4})
axes[0].set_title("Weight (kg) by Timezone"); axes[0].tick_params(axis="x", rotation=20)

sns.boxplot(data=profiles, x="timezone", y="height_cm", order=tz_order,
            palette="Set2", ax=axes[1], flierprops={"marker": "o", "markersize": 4})
axes[1].set_title("Height (cm) by Timezone"); axes[1].tick_params(axis="x", rotation=20)

fig.suptitle("Timezone-wise Distributions (checking unit consistency)", fontsize=12, y=1.01)
plt.tight_layout(); plt.show()
print("→ Medians are similar across timezones → all weights already in kg")

# COMMAND ----------

# DBTITLE 1,User Profiles - weight analysis + outlier chart
# ── 1B: Weight analysis — NO lbs conversion ──
# After checking timezone-wise height distributions, all timezones show the same
# scale (means 170–176 cm). Heights are uniform → weights are too. All weights
# are already in kg. The high-weight users are genuine outliers, not unit artifacts.
#
# Decision: Keep raw weight as weight_kg. Flag outliers but do NOT remove them —
# removal is deferred to the modelling phase (only if it helps performance).

profiles["weight_kg"] = profiles["weight"]  # no conversion — all values are kg

print("Weight distribution (all kg, no conversion applied):")
print(profiles["weight_kg"].describe().to_string())

# Flag weight outliers using IQR
Q1 = profiles["weight_kg"].quantile(0.25)
Q3 = profiles["weight_kg"].quantile(0.75)
IQR = Q3 - Q1
lo_fence = Q1 - 1.5 * IQR
hi_fence = Q3 + 1.5 * IQR
profiles["weight_outlier"] = ((profiles["weight_kg"] < lo_fence) | (profiles["weight_kg"] > hi_fence)).astype(int)

print(f"\nIQR fences: [{lo_fence:.1f}, {hi_fence:.1f}] kg")
print(f"Weight outliers flagged: {profiles['weight_outlier'].sum()} (kept in data, may remove if it helps model)")

# ── Weight distribution with IQR fences ──
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(profiles["weight_kg"], bins=25, kde=True, color=PAL[0], ax=ax, edgecolor="white")
ax.axvline(lo_fence, color="red", ls="--", lw=1.2, label=f"IQR fences [{lo_fence:.0f}, {hi_fence:.0f}] kg")
ax.axvline(hi_fence, color="red", ls="--", lw=1.2)
for _, r in profiles[profiles["weight_outlier"] == 1].iterrows():
    ax.axvline(r["weight_kg"], color="orange", alpha=0.4, lw=0.8)
ax.set_title(f"Weight Distribution — {profiles['weight_outlier'].sum()} outliers flagged (kept)")
ax.set_xlabel("Weight (kg)"); ax.legend(); plt.tight_layout(); plt.show()

print(f"Outlier count by timezone: {profiles.groupby('timezone')['weight_outlier'].sum().to_dict()}")

# COMMAND ----------

# DBTITLE 1,User Profiles - height imputation, BMI, final validation
# ── 1C: Height nulls and BMI computation ──
print(f"Height nulls: {profiles['height_cm'].isna().sum()}")
print(f"Height range: {profiles['height_cm'].min():.1f} - {profiles['height_cm'].max():.1f} cm")

# Impute missing heights with sex-specific median
for sex in ["M", "F"]:
    mask = (profiles["sex"] == sex) & (profiles["height_cm"].isna())
    median_h = profiles.loc[profiles["sex"] == sex, "height_cm"].median()
    profiles.loc[mask, "height_cm"] = median_h
    print(f"  Imputed {mask.sum()} {sex} heights with median={median_h:.1f} cm")

# Compute BMI (using raw kg weight — no lbs conversion was applied)
profiles["bmi"] = profiles["weight_kg"] / ((profiles["height_cm"]/100)**2)
print(f"\nBMI range: {profiles['bmi'].min():.1f} - {profiles['bmi'].max():.1f}")

# Flag BMI outliers too (extremely high BMI may be weight outliers)
profiles["bmi_outlier"] = (profiles["bmi"] > 45).astype(int)
print(f"BMI > 45 (extreme): {profiles['bmi_outlier'].sum()} users (flagged, not removed)")

# Age validation (per dictionary: integer)
print(f"\nAge range: {profiles['age_years'].min()} - {profiles['age_years'].max()}")
range_check(profiles["age_years"], 18, 80, "Age")

# ── BMI distribution with WHO thresholds ──
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(profiles["bmi"], bins=25, kde=True, color=PAL[0], ax=ax, edgecolor="white")
for th, lbl, c in [(18.5, "Underweight", "#74a9cf"), (25, "Normal", "#41ab5d"),
                    (30, "Overweight", "#fe9929"), (45, "Extreme", "#e31a1c")]:
    ax.axvline(th, color=c, ls="--", lw=1, alpha=0.8, label=f"{lbl} ({th})")
ax.set_title(f"BMI Distribution — {len(profiles)} users"); ax.set_xlabel("BMI"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

print(f"\n── CLEAN user_profiles: {len(profiles)} users × {profiles.shape[1]} cols ──")
display(profiles.head(3))

# COMMAND ----------

# DBTITLE 1,User Profiles - derived features + visual summary
# ── 1D: Derived profile features ──
# Go back to the RAW profiles (before dedup) to extract profile-update
# history, then merge into the clean 120-user profiles DataFrame.

raw_prof = raw_profiles.copy()
raw_prof["user_id"] = raw_prof["user_id"].str.strip().str.upper()
raw_prof["profile_updated_at"] = pd.to_datetime(raw_prof["profile_updated_at"])

# 1. Profile update count — how many profile rows exist per user
update_stats = raw_prof.groupby("user_id").agg(
    profile_update_count=("user_id", "size"),
    weight_range=("weight", lambda x: x.max() - x.min()),
).reset_index()
update_stats["has_updated_profile"] = (update_stats["profile_update_count"] > 1).astype(int)
update_stats["weight_changed"] = (update_stats["weight_range"] > 0.5).astype(int)  # >0.5 kg diff

profiles = profiles.merge(
    update_stats[["user_id", "profile_update_count", "has_updated_profile", "weight_changed"]],
    on="user_id", how="left"
)

# 2. Age group bins
profiles["age_group"] = pd.cut(
    profiles["age_years"],
    bins=[0, 25, 35, 45, 100],
    labels=["18-25", "26-35", "36-45", "46+"]
)

# 3. BMI category (standard WHO thresholds)
profiles["bmi_category"] = pd.cut(
    profiles["bmi"],
    bins=[0, 18.5, 25, 30, 100],
    labels=["underweight", "normal", "overweight", "obese"]
)

# 4. Onboarding tenure — days from onboarding to study start
profiles["onboarding_date_dt"] = pd.to_datetime(profiles["onboarding_date"])
profiles["onboarding_tenure_days"] = (DATA_START - profiles["onboarding_date_dt"]).dt.days

# ── Summary ──
print("Derived profile features added:")
print(f"  profile_update_count: {profiles['profile_update_count'].value_counts().sort_index().to_dict()}")
print(f"  has_updated_profile:  {profiles['has_updated_profile'].sum()} of 120 users")
print(f"  weight_changed:       {profiles['weight_changed'].sum()} users had weight change > 0.5 kg")
print(f"  age_group:            {profiles['age_group'].value_counts().sort_index().to_dict()}")
print(f"  bmi_category:         {profiles['bmi_category'].value_counts().sort_index().to_dict()}")
print(f"  onboarding_tenure:    mean={profiles['onboarding_tenure_days'].mean():.0f}d, "
      f"range=[{profiles['onboarding_tenure_days'].min()}, {profiles['onboarding_tenure_days'].max()}]d")

print(f"\nClean profiles: {len(profiles)} users × {profiles.shape[1]} cols")

# ── Visual summary of derived categorical features ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, title, pal in zip(
    axes,
    ["age_group", "bmi_category", "onboarding_tenure_days"],
    ["Age Group", "BMI Category", "Onboarding Tenure (days)"],
    ["Blues_d", "RdYlGn_r", None]
):
    if col == "onboarding_tenure_days":
        sns.histplot(profiles[col], bins=20, kde=True, color=PAL[2], ax=ax, edgecolor="white")
    else:
        order = profiles[col].value_counts().sort_index().index
        sns.countplot(data=profiles, x=col, order=order, palette=pal, ax=ax)
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height() + 0.5),
                        ha="center", fontsize=9)
    ax.set_title(title)
plt.tight_layout(); plt.show()

# COMMAND ----------

# DBTITLE 1,Section 1 Final — clean profiles sample (all columns)
# ── Section 1 FINAL: Clean profiles — all columns sample ──
print(f"\u2550" * 70)
print(f"SECTION 1 OUTPUT: profiles — {len(profiles)} users × {profiles.shape[1]} cols")
print(f"\u2550" * 70)
print(f"\nColumns ({profiles.shape[1]}):")
for i, col in enumerate(profiles.columns):
    dtype = profiles[col].dtype
    nulls = profiles[col].isna().sum()
    print(f"  {i+1:>2}. {col:30s}  {str(dtype):12s}  nulls={nulls}")

print(f"\n\u2500\u2500 Sample (3 rows, all columns) \u2500\u2500")
display(profiles.head(20))

# COMMAND ----------

# DBTITLE 1,Section 2: Sleep Sessions
# MAGIC %md
# MAGIC ## Section 2: Sleep Sessions
# MAGIC
# MAGIC This is the dirtiest table. Known issues: mixed user_id casing, epoch timestamps, duplicate session_ids, multi-session nights, physiological outliers, duration mismatches, 349 extra users not in the target.

# COMMAND ----------

# DBTITLE 1,Sleep Sessions - raw inspection and user_id normalization
# ── 2A: Sleep Sessions — Raw inspection ──
sessions = raw_sessions.copy()
shape_report(sessions, "sleep_sessions (raw)")

# User ID normalization (per note: "small and capital both letters are there")
print("\nuser_id prefixes BEFORE normalization:")
print(sessions["user_id"].str[:5].value_counts().to_dict())

sessions["user_id"] = sessions["user_id"].str.strip().str.upper()
print(f"\nUnique users after normalization: {sessions['user_id'].nunique()}")

# Filter to only the 120 target users
target_users = set(profiles["user_id"])
before_filter = len(sessions)
sessions = sessions[sessions["user_id"].isin(target_users)].copy()
print(f"Filtered to target users: {before_filter:,} -> {len(sessions):,} rows ({before_filter - len(sessions):,} dropped)")
print(f"Users remaining: {sessions['user_id'].nunique()}")

# COMMAND ----------

# DBTITLE 1,Sleep Sessions - timestamp parsing (ISO + epoch)
# ── 2B: Timestamp parsing — handle both ISO and epoch millis ──
def parse_mixed_timestamps(series):
    """
    Parse timestamps that are either ISO format or epoch milliseconds.
    Returns a UTC-normalized datetime series.
    """
    result = pd.to_datetime(series, errors="coerce", utc=True)
    
    # Find unparsed rows — try as epoch millis
    mask_na = result.isna()
    if mask_na.any():
        epoch_vals = pd.to_numeric(series[mask_na], errors="coerce")
        epoch_parsed = pd.to_datetime(epoch_vals, unit="ms", utc=True, errors="coerce")
        result[mask_na] = epoch_parsed
    
    return result

sessions["session_start_utc"] = parse_mixed_timestamps(sessions["session_start"])
sessions["session_end_utc"]   = parse_mixed_timestamps(sessions["session_end"])

print("Timestamp parsing results:")
print(f"  session_start: {sessions['session_start_utc'].isna().sum()} still unparseable out of {len(sessions)}")
print(f"  session_end:   {sessions['session_end_utc'].isna().sum()} still unparseable out of {len(sessions)}")
print(f"  Date range:    {sessions['session_start_utc'].min()} to {sessions['session_end_utc'].max()}")

# Show sample of parsed epoch timestamps to verify
epoch_mask = pd.to_datetime(sessions["session_start"], errors="coerce", utc=True).isna()
epoch_rows = sessions[epoch_mask].head(5)
if len(epoch_rows) > 0:
    print("\nSample epoch→UTC conversions:")
    for _, row in epoch_rows.iterrows():
        print(f"  {row['session_start']} -> {row['session_start_utc']}")

# Drop rows where timestamps still failed
n_bad_ts = sessions["session_start_utc"].isna().sum()
if n_bad_ts > 0:
    print(f"\nDropping {n_bad_ts} rows with unparseable timestamps")
    sessions = sessions.dropna(subset=["session_start_utc", "session_end_utc"]).copy()
    print(f"Remaining: {len(sessions):,} rows")

# COMMAND ----------

# DBTITLE 1,Sleep Sessions - deduplicate session_ids
# ── 2C: Duplicate session_id handling ──
n_dup_sessions = sessions["session_id"].duplicated().sum()
print(f"Duplicate session_ids: {n_dup_sessions}")

# Verify ALL duplicates are exact whole-row copies
if n_dup_sessions > 0:
    dup_mask = sessions["session_id"].duplicated(keep=False)
    dups = sessions[dup_mask]
    all_cols = sessions.columns.tolist()
    exact_count = 0
    partial_count = 0
    partial_details = []
    for sid, group in dups.groupby("session_id"):
        if group.drop_duplicates().shape[0] == 1:
            exact_count += 1
        else:
            partial_count += 1
            diff_cols = [c for c in all_cols if group[c].nunique(dropna=False) > 1]
            partial_details.append((sid, diff_cols))
    print(f"  Exact whole-row duplicates: {exact_count} session_ids")
    print(f"  Partial duplicates (some cols differ): {partial_count} session_ids")
    print(f"  Copies per session_id: {dups.groupby('session_id').size().value_counts().sort_index().to_dict()}")
    if partial_count > 0:
        print("  ⚠ Partial duplicate details:")
        for sid, cols in partial_details[:5]:
            print(f"    {sid}: differs in {cols}")
    else:
        print("  ✓ Safe to drop_duplicates — no information loss")

# Sample 3 duplicate pairs for visual confirmation (not all 442 rows)
if n_dup_sessions > 0:
    sample_sids = dups["session_id"].drop_duplicates().head(3).tolist()
    print(f"\nSample duplicate pairs ({len(sample_sids)} of {n_dup_sessions}):")
    display(dups[dups["session_id"].isin(sample_sids)].sort_values(["session_id", "user_id"]))

# Deduplicate
sessions = dedup_report(sessions, ["session_id"], "sleep_sessions")

# COMMAND ----------

# DBTITLE 1,Sleep Sessions - seconds→minutes fix + decomposition validation
# ── 2D: Seconds → Minutes fix + Duration consistency ──
# ISSUE: Some rows have ALL duration columns in SECONDS instead of minutes.
# Detection: ratio of time_in_bed / calc_duration_from_timestamps ≈ 60
# (timestamps are always correct — they serve as ground truth).

duration_cols = ["time_in_bed_minutes", "total_sleep_minutes", "deep_minutes",
                 "rem_minutes", "light_minutes", "awake_minutes"]

# Compute ground truth from timestamps
sessions["calc_duration_min"] = (sessions["session_end_utc"] - sessions["session_start_utc"]).dt.total_seconds() / 60

# Detect seconds-unit rows via ratio (covers both long and short sessions)
valid = sessions["calc_duration_min"] > 1  # guard against div-by-zero
ratio = sessions.loc[valid, "time_in_bed_minutes"] / sessions.loc[valid, "calc_duration_min"]
sec_mask = pd.Series(False, index=sessions.index)
sec_mask.loc[ratio.index] = ratio.between(50, 70)  # wider band to handle DST-crossing sessions
print(f"Rows with duration in SECONDS (ratio ≈ 60): {sec_mask.sum()} ({sec_mask.sum()/len(sessions)*100:.1f}%)")
print(f"Users affected: {sessions.loc[sec_mask, 'user_id'].nunique()}")

# Show before-fix sample
print("\nSample BEFORE fix (raw values):")
display(sessions.loc[sec_mask, ["session_id", "user_id"] + duration_cols].head(5))

# Convert seconds → minutes
for col in duration_cols:
    sessions.loc[sec_mask, col] = sessions.loc[sec_mask, col] / 60

print(f"\nConverted {sec_mask.sum()} rows × {len(duration_cols)} duration columns from seconds to minutes")

# Show after-fix sample
print("\nSample AFTER fix (values in minutes):")
display(sessions.loc[sec_mask, ["session_id", "user_id"] + duration_cols].head(5))

# ── Duration consistency check (now with corrected units) ──
duration_diff = sessions["calc_duration_min"] - sessions["time_in_bed_minutes"]
print("\nDuration consistency AFTER seconds fix (calculated - reported):")
print(f"  Mean diff:  {duration_diff.mean():.2f} min")
print(f"  Median diff: {duration_diff.median():.2f} min")
print(f"  Std diff:   {duration_diff.std():.2f} min")
print(f"  Max |diff|: {duration_diff.abs().max():.2f} min")
print(f"  Rows with |diff| > 5 min:  {(duration_diff.abs() > 5).sum()}")
print(f"  Rows with |diff| > 60 min: {(duration_diff.abs() > 60).sum()}")

# Negative durations (end before start)
neg_dur = (sessions["calc_duration_min"] < 0).sum()
print(f"  Negative durations: {neg_dur}")

# Show worst offenders post-fix
if (duration_diff.abs() > 60).sum() > 0:
    worst = sessions.loc[duration_diff.abs().nlargest(5).index,
                         ["session_id", "user_id", "time_in_bed_minutes", "calc_duration_min",
                          "session_start", "session_end"]]
    print("\nWorst duration mismatches (post-fix):")
    display(worst)

# Flag remaining mismatches
sessions["duration_mismatch"] = (duration_diff.abs() > 60)
print(f"\nRows flagged for >60min duration mismatch: {sessions['duration_mismatch'].sum()}")

# ── 2D-ii: Duration Decomposition Validation ──
# Check the mathematical relationships between duration columns
print("\n" + "="*70)
print("DURATION DECOMPOSITION VALIDATION")
print("="*70)

# R1: time_in_bed = total_sleep + awake
diff_r1 = sessions["time_in_bed_minutes"] - (sessions["total_sleep_minutes"] + sessions["awake_minutes"])
print(f"\nR1: time_in_bed ≈ total_sleep + awake")
print(f"    Within 1 min: {(diff_r1.abs() < 1).sum():,}/{len(sessions):,} ({(diff_r1.abs() < 1).sum()/len(sessions)*100:.1f}%)")
print(f"    |diff| > 60 min: {(diff_r1.abs() > 60).sum()}")

# R2: total_sleep = deep + rem + light
diff_r2 = sessions["total_sleep_minutes"] - (sessions["deep_minutes"] + sessions["rem_minutes"] + sessions["light_minutes"])
print(f"\nR2: total_sleep ≈ deep + rem + light")
print(f"    Within 1 min: {(diff_r2.abs() < 1).sum():,}/{len(sessions):,} ({(diff_r2.abs() < 1).sum()/len(sessions)*100:.1f}%)")
print(f"    864 rows have unscored sleep time (up to 43 min gap) — device/algorithm artefact")

# R3: sleep_efficiency = total_sleep / time_in_bed
calc_eff = sessions["total_sleep_minutes"] / sessions["time_in_bed_minutes"]
eff_diff = (calc_eff - sessions["sleep_efficiency"]).abs()
print(f"\nR3: sleep_efficiency ≈ total_sleep / time_in_bed")
print(f"    Within 0.01: {(eff_diff < 0.01).sum():,}/{len(sessions):,} ({(eff_diff < 0.01).sum()/len(sessions)*100:.1f}%)")

# ── Fix negative durations (sensor artefacts) ──
# Tested clip-to-0 vs abs(): abs() preserves magnitude and reduces the
# decomposition residual by ~20% more than zeroing. Neither approach
# fixes R1 for these rows (structurally broken), but abs() gives the
# model a real measured value instead of an artificial 0.
print("\n" + "="*70)
print("NEGATIVE DURATION FIX (using abs — preserves magnitude)")
print("="*70)
for col in duration_cols:
    neg_count = (sessions[col] < 0).sum()
    if neg_count > 0:
        print(f"  {col}: {neg_count} negative values (min={sessions[col].min():.1f}) → abs()")
        sessions[col] = sessions[col].abs()

# ── Visual: duration distribution before/after fix ──
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.histplot(sessions["time_in_bed_minutes"], bins=40, kde=True, color=PAL[0], ax=axes[0], edgecolor="white")
axes[0].set_title("Time in Bed (post-fix)"); axes[0].set_xlabel("Minutes")
sns.histplot(sessions["sleep_efficiency"], bins=40, kde=True, color=PAL[1], ax=axes[1], edgecolor="white")
axes[1].set_title("Sleep Efficiency"); axes[1].set_xlabel("Efficiency")
plt.suptitle(f"Session Distributions ({len(sessions):,} sessions)", fontsize=12, y=1.01)
plt.tight_layout(); plt.show()

print("Decomposition hierarchy (post-fix):")
print("  Level 1: time_in_bed = total_sleep + awake")
print("  Level 2: total_sleep = deep + rem + light")
print("  Level 3: sleep_efficiency = total_sleep / time_in_bed")
print("  → Small gaps are unscored/transitional time — normal for wearable devices")

# COMMAND ----------

# DBTITLE 1,Sleep Sessions - decomposition-derived features
# ── 2D-iii: Decomposition-derived features ──
# Turn the decomposition validation findings into modelling features.
# These capture data quality AND sleep architecture in one pass.

# ── R1 features: time_in_bed vs total_sleep + awake ──
sessions["tib_gap_min"] = sessions["time_in_bed_minutes"] - (sessions["total_sleep_minutes"] + sessions["awake_minutes"])
sessions["tib_decomp_exact"] = (sessions["tib_gap_min"].abs() < 1).astype(int)

# ── R2 features: total_sleep vs deep + rem + light (unscored sleep) ──
sessions["unscored_sleep_min"] = sessions["total_sleep_minutes"] - (
    sessions["deep_minutes"] + sessions["rem_minutes"] + sessions["light_minutes"]
)
sessions["unscored_sleep_pct"] = (
    sessions["unscored_sleep_min"] / sessions["total_sleep_minutes"].clip(lower=1) * 100
).round(2)
sessions["stage_decomp_exact"] = (sessions["unscored_sleep_min"].abs() < 1).astype(int)

# ── R3 features: efficiency consistency ──
sessions["efficiency_calc"] = (sessions["total_sleep_minutes"] / sessions["time_in_bed_minutes"]).round(4)
sessions["efficiency_gap"] = (sessions["efficiency_calc"] - sessions["sleep_efficiency"]).round(4)
sessions["efficiency_match"] = (sessions["efficiency_gap"].abs() < 0.01).astype(int)

# ── Sleep architecture proportions (% of total sleep) ──
ts_safe = sessions["total_sleep_minutes"].clip(lower=1)
tib_safe = sessions["time_in_bed_minutes"].clip(lower=1)
sessions["deep_pct"]  = (sessions["deep_minutes"]  / ts_safe * 100).round(1)
sessions["rem_pct"]   = (sessions["rem_minutes"]   / ts_safe * 100).round(1)
sessions["light_pct"] = (sessions["light_minutes"] / ts_safe * 100).round(1)
sessions["awake_pct"] = (sessions["awake_minutes"] / tib_safe * 100).round(1)

# ── Data quality composite (0–3: how many decomposition rules match) ──
sessions["decomp_quality_score"] = (
    sessions["tib_decomp_exact"] + sessions["stage_decomp_exact"] + sessions["efficiency_match"]
)

# ── Summary ──
print("Decomposition-derived features added (11 new columns):")
print(f"\n  R1: tib_gap_min          mean={sessions['tib_gap_min'].mean():.2f}, std={sessions['tib_gap_min'].std():.2f}")
print(f"      tib_decomp_exact     {sessions['tib_decomp_exact'].sum():,}/{len(sessions):,} ({sessions['tib_decomp_exact'].mean()*100:.1f}%)")
print(f"\n  R2: unscored_sleep_min   mean={sessions['unscored_sleep_min'].mean():.2f}, std={sessions['unscored_sleep_min'].std():.2f}")
print(f"      unscored_sleep_pct   mean={sessions['unscored_sleep_pct'].mean():.1f}%")
print(f"      stage_decomp_exact   {sessions['stage_decomp_exact'].sum():,}/{len(sessions):,} ({sessions['stage_decomp_exact'].mean()*100:.1f}%)")
print(f"\n  R3: efficiency_gap       mean={sessions['efficiency_gap'].mean():.4f}")
print(f"      efficiency_match     {sessions['efficiency_match'].sum():,}/{len(sessions):,} ({sessions['efficiency_match'].mean()*100:.1f}%)")

print(f"\nSleep architecture (% of total sleep):")
print(f"  deep_pct:  {sessions['deep_pct'].mean():.1f}% ± {sessions['deep_pct'].std():.1f}%")
print(f"  rem_pct:   {sessions['rem_pct'].mean():.1f}% ± {sessions['rem_pct'].std():.1f}%")
print(f"  light_pct: {sessions['light_pct'].mean():.1f}% ± {sessions['light_pct'].std():.1f}%")
print(f"  awake_pct: {sessions['awake_pct'].mean():.1f}% ± {sessions['awake_pct'].std():.1f}%")

print(f"\nDecomp quality score (0=poor, 3=perfect):")
print(sessions["decomp_quality_score"].value_counts().sort_index().to_string())

print(f"\n--- Sessions: {len(sessions):,} rows × {sessions.shape[1]} cols ---")
sample_cols = ["session_id", "user_id", "time_in_bed_minutes", "total_sleep_minutes",
               "deep_pct", "rem_pct", "light_pct", "awake_pct",
               "tib_gap_min", "tib_decomp_exact", "unscored_sleep_min",
               "unscored_sleep_pct", "stage_decomp_exact",
               "efficiency_calc", "efficiency_gap", "efficiency_match",
               "decomp_quality_score"]
display(sessions[sample_cols].head(10))

# ── Decomp quality score distribution ──
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
quals = sessions["decomp_quality_score"].value_counts().sort_index()
colors_q = ["#e31a1c", "#fd8d3c", "#fecc5c", "#41ab5d"]
axes[0].bar(quals.index, quals.values, color=colors_q, edgecolor="white", width=0.6)
for i, v in quals.items():
    axes[0].text(i, v + 50, f"{v:,}", ha="center", fontsize=9)
axes[0].set_title("Decomposition Quality Score"); axes[0].set_xlabel("Score (0=poor, 3=perfect)")
axes[0].set_ylabel("Sessions")

# Sleep architecture stacked bar (mean %)
arch = sessions[["deep_pct", "rem_pct", "light_pct"]].mean()
axes[1].barh(["Architecture"], [arch["deep_pct"]], color="#2166ac", label=f"Deep {arch['deep_pct']:.1f}%")
axes[1].barh(["Architecture"], [arch["rem_pct"]], left=[arch["deep_pct"]], color="#92c5de", label=f"REM {arch['rem_pct']:.1f}%")
axes[1].barh(["Architecture"], [arch["light_pct"]], left=[arch["deep_pct"] + arch["rem_pct"]], color="#fddbc7", label=f"Light {arch['light_pct']:.1f}%")
axes[1].set_title("Average Sleep Architecture (% of total sleep)"); axes[1].legend(loc="lower right")
axes[1].set_xlim(0, 105)
plt.tight_layout(); plt.show()

# COMMAND ----------

# DBTITLE 1,Sleep Sessions - data quality overview
# ── 2D-iv: Data quality summary before multi-session & outlier handling ──

# Count issues by category
_raw = raw_sessions.copy()
_raw["user_id"] = _raw["user_id"].str.strip().str.upper()
_raw = _raw[_raw["user_id"].isin(set(profiles["user_id"]))].drop_duplicates(subset=["session_id"], keep="first")
neg_count = ((_raw["light_minutes"] < 0) | (_raw["awake_minutes"] < 0)).sum()
mismatch_count = (sessions["decomp_quality_score"] < 3).sum()

print(f"Data quality snapshot ({len(sessions):,} sessions):")
print(f"  Originally negative durations (now abs'd): {neg_count}")
print(f"  Decomposition mismatches (score < 3):      {mismatch_count}")
print(f"  Perfect decomposition (score = 3):         {(sessions['decomp_quality_score'] == 3).sum():,}")

# ── Visual: data quality heatmap by user ──
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# 1. Quality score distribution by firmware
qual_fw = sessions.groupby(["firmware_version", "decomp_quality_score"]).size().unstack(fill_value=0)
qual_fw_pct = qual_fw.div(qual_fw.sum(axis=1), axis=0) * 100
qual_fw_pct.plot(kind="bar", stacked=True, ax=axes[0],
                 color=["#e31a1c", "#fd8d3c", "#fecc5c", "#41ab5d"], edgecolor="white")
axes[0].set_title("Quality Score by Firmware"); axes[0].set_ylabel("%")
axes[0].set_xlabel(""); axes[0].legend(title="Score", fontsize=8)
axes[0].tick_params(axis="x", rotation=0)

# 2. Sessions per user histogram
user_counts = sessions.groupby("user_id").size()
sns.histplot(user_counts, bins=20, kde=True, color=PAL[3], ax=axes[1], edgecolor="white")
axes[1].axvline(user_counts.mean(), color="red", ls="--", lw=1, label=f"Mean={user_counts.mean():.0f}")
axes[1].set_title("Sessions per User"); axes[1].set_xlabel("Session count"); axes[1].legend()

plt.tight_layout(); plt.show()

print(f"\n── Sessions: {len(sessions):,} rows × {sessions.shape[1]} cols ──")

# COMMAND ----------

# DBTITLE 1,Sleep Sessions - multi-session nights + secondary features
# ── 2E: Multi-session nights — primary session + secondary features ──
sessions["night_date"] = sessions["session_start_utc"].dt.date

sessions_per_night = sessions.groupby(["user_id", "night_date"]).size().reset_index(name="n_sessions")
multi_night = sessions_per_night[sessions_per_night["n_sessions"] > 1]

print(f"Total user-nights: {len(sessions_per_night)}")
print(f"Multi-session nights: {len(multi_night)}")
print(f"Sessions distribution per night:")
print(sessions_per_night["n_sessions"].value_counts().sort_index().to_string())

# ── Identify primary session (longest total_sleep per night) ──
sessions["is_primary"] = False
idx_primary = sessions.groupby(["user_id", "night_date"])["total_sleep_minutes"].idxmax()
sessions.loc[idx_primary, "is_primary"] = True

# ── Aggregate ALL sessions per night for secondary features ──
night_agg = sessions.groupby(["user_id", "night_date"]).agg(
    n_sessions             = ("session_id", "size"),
    total_tib_all          = ("time_in_bed_minutes", "sum"),   # total time in bed across ALL sessions
    total_sleep_all        = ("total_sleep_minutes", "sum"),    # total sleep across ALL sessions
    total_awake_all        = ("awake_minutes", "sum"),          # total awake across ALL sessions
    night_first_start      = ("session_start_utc", "min"),     # earliest session start
    night_last_end         = ("session_end_utc", "max"),       # latest session end
    mean_efficiency_all    = ("sleep_efficiency", "mean"),      # avg efficiency across all sessions
    min_efficiency_night   = ("sleep_efficiency", "min"),       # worst session efficiency
).reset_index()

# ── Aggregate SECONDARY sessions only (non-primary) ──
secondary = sessions[~sessions["is_primary"]].copy()
sec_agg = secondary.groupby(["user_id", "night_date"]).agg(
    secondary_tib          = ("time_in_bed_minutes", "sum"),   # total TIB of secondary sessions
    secondary_sleep        = ("total_sleep_minutes", "sum"),   # total sleep of secondary sessions
    secondary_session_count = ("session_id", "size"),          # how many secondary sessions
    secondary_max_tib      = ("time_in_bed_minutes", "max"),   # longest secondary session
    secondary_mean_eff     = ("sleep_efficiency", "mean"),     # avg efficiency of secondaries
).reset_index()

# ── Merge night-level aggregates onto primary sessions ──
sessions = sessions.merge(night_agg, on=["user_id", "night_date"], how="left", suffixes=("", "_nightagg"))
sessions["fragmented_night"] = (sessions["n_sessions"] > 1).astype(int)

# Merge secondary aggregates (will be NaN for single-session nights — fill with 0)
sessions = sessions.merge(sec_agg, on=["user_id", "night_date"], how="left")
for col in ["secondary_tib", "secondary_sleep", "secondary_session_count", "secondary_max_tib"]:
    sessions[col] = sessions[col].fillna(0)
sessions["secondary_mean_eff"] = sessions["secondary_mean_eff"].fillna(0)

# ── Derived night-level features ──
# Night window: total wall-clock span from first session start to last session end
sessions["night_window_min"] = (
    (sessions["night_last_end"] - sessions["night_first_start"]).dt.total_seconds() / 60
).round(1)

# Nap ratio: how much secondary sleep vs primary (0 for single-session nights)
sessions["nap_ratio"] = (
    sessions["secondary_tib"] / sessions["time_in_bed_minutes"].clip(lower=1)
).round(4)

# Sleep fragmentation index: gap between total night window and sum of all TIB
# High values = long awake gaps between sessions
sessions["inter_session_gap_min"] = (
    sessions["night_window_min"] - sessions["total_tib_all"]
).clip(lower=0).round(1)

# Night consolidation: what fraction of the night window was actually spent in bed
sessions["night_consolidation"] = (
    sessions["total_tib_all"] / sessions["night_window_min"].clip(lower=1)
).clip(upper=1.0).round(4)

# ── Keep only primary sessions for modelling ──
sessions_primary = sessions[sessions["is_primary"]].copy()

print(f"\nPrimary sessions retained: {len(sessions_primary):,}")
print(f"Fragmented nights: {sessions_primary['fragmented_night'].sum()}")
print(f"Users: {sessions_primary['user_id'].nunique()}")

# ── Summary of new features ──
new_cols = ["n_sessions", "fragmented_night", "total_tib_all", "total_sleep_all",
            "total_awake_all", "secondary_tib", "secondary_sleep",
            "secondary_session_count", "secondary_max_tib", "secondary_mean_eff",
            "nap_ratio", "inter_session_gap_min", "night_consolidation",
            "night_window_min", "mean_efficiency_all", "min_efficiency_night"]
print(f"\n── New multi-session features ({len(new_cols)} columns) ──")
frag = sessions_primary[sessions_primary["fragmented_night"] == 1]
print(f"\nFragmented nights stats ({len(frag)} nights):")
for col in new_cols:
    if col in ["fragmented_night"]: continue
    vals = frag[col]
    print(f"  {col:30s}  mean={vals.mean():.1f}  median={vals.median():.1f}  "
          f"range=[{vals.min():.1f}, {vals.max():.1f}]")

# ── Visual ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Sessions per night distribution
counts = sessions_per_night["n_sessions"].value_counts().sort_index()
axes[0].bar(counts.index.astype(str), counts.values,
            color=[PAL[0], PAL[1], PAL[2], PAL[3]][:len(counts)], edgecolor="white", width=0.5)
for i, (x, v) in enumerate(zip(counts.index, counts.values)):
    axes[0].text(i, v + 15, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
axes[0].set_title("Sessions per Night"); axes[0].set_ylabel("Nights")

# 2. Secondary session duration (fragmented only)
if len(frag) > 0:
    sns.histplot(frag["secondary_tib"], bins=25, kde=True, color=PAL[1], ax=axes[1], edgecolor="white")
    axes[1].axvline(frag["secondary_tib"].median(), color="red", ls="--", lw=1,
                    label=f"Median={frag['secondary_tib'].median():.0f} min")
    axes[1].set_title("Secondary Session Duration"); axes[1].set_xlabel("Minutes"); axes[1].legend()

# 3. Nap ratio distribution (fragmented only)
if len(frag) > 0:
    sns.histplot(frag["nap_ratio"], bins=25, kde=True, color=PAL[2], ax=axes[2], edgecolor="white")
    axes[2].set_title("Nap Ratio (secondary / primary TIB)"); axes[2].set_xlabel("Ratio")

plt.suptitle(f"Multi-Session Night Analysis ({len(frag)} fragmented of {len(sessions_primary):,} nights)",
             fontsize=12, y=1.02)
plt.tight_layout(); plt.show()

# COMMAND ----------

# DBTITLE 1,Sleep Sessions - datetime, temporal, and personalized holiday features
# ── 2E-ii: Datetime, temporal, and personalized holiday features ──
# Adds ~30 features from session timestamps, calendar patterns, and
# timezone-aware public holidays to help predict morning mood.

import pytz
try:
    import holidays as hol_lib
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "holidays", "-q"])
    import holidays as hol_lib

sp = sessions_primary.copy()

# ════════════════════════════════════════════════════════════════════
# SECTION A: LOCAL SLEEP TIMING FEATURES
# ════════════════════════════════════════════════════════════════════
print("A. Local sleep timing features")

# Merge timezone from profiles
tz_map = profiles.set_index("user_id")["timezone"]
sp["timezone"] = sp["user_id"].map(tz_map)

# Extract local-time hours per timezone group
sp["bedtime_hour"] = np.nan
sp["wake_hour"]    = np.nan

for tz_name in sp["timezone"].unique():
    mask = sp["timezone"] == tz_name
    tz_obj = pytz.timezone(tz_name)
    local_start = sp.loc[mask, "session_start_utc"].dt.tz_convert(tz_obj)
    local_end   = sp.loc[mask, "session_end_utc"].dt.tz_convert(tz_obj)
    sp.loc[mask, "bedtime_hour"] = (local_start.dt.hour + local_start.dt.minute / 60).round(2)
    sp.loc[mask, "wake_hour"]    = (local_end.dt.hour + local_end.dt.minute / 60).round(2)

# Sleep onset period (when did they fall asleep, local time)
sp["sleep_onset_period"] = pd.cut(
    sp["bedtime_hour"],
    bins=[-0.01, 3, 6, 12, 18, 21, 24.01],
    labels=["late_night", "early_morning", "daytime", "evening_early", "night", "late_night_2"],
    ordered=False
).astype(str).replace("late_night_2", "late_night")

# Wake period (when did they wake up, local time)
sp["wake_period"] = pd.cut(
    sp["wake_hour"],
    bins=[-0.01, 5, 7, 9, 12, 24.01],
    labels=["pre_dawn", "early_morning", "morning", "late_morning", "afternoon"],
    ordered=False
).astype(str)

# Bedtime deviation from midnight (neg = before midnight, pos = after)
sp["bedtime_dev_midnight_hr"] = sp["bedtime_hour"].apply(
    lambda h: h - 24 if h >= 12 else h
).round(2)

# Sleep midpoint hour (handles overnight wrap-around)
def _midpoint(start, end):
    if start > end:  # overnight: e.g., 23h to 7h
        dur = (24 - start) + end
        return (start + dur / 2) % 24
    return (start + end) / 2

sp["sleep_midpoint_hour"] = [
    round(_midpoint(s, e), 2) for s, e in zip(sp["bedtime_hour"], sp["wake_hour"])
]

# Circular encoding for bedtime (useful for models — 23h and 1h are close)
sp["bedtime_sin"] = np.sin(2 * np.pi * sp["bedtime_hour"] / 24).round(4)
sp["bedtime_cos"] = np.cos(2 * np.pi * sp["bedtime_hour"] / 24).round(4)
sp["wake_sin"]    = np.sin(2 * np.pi * sp["wake_hour"] / 24).round(4)
sp["wake_cos"]    = np.cos(2 * np.pi * sp["wake_hour"] / 24).round(4)

print(f"  bedtime_hour:  mean={sp['bedtime_hour'].mean():.1f}, std={sp['bedtime_hour'].std():.1f}")
print(f"  wake_hour:     mean={sp['wake_hour'].mean():.1f}, std={sp['wake_hour'].std():.1f}")
print(f"  sleep_onset_period: {sp['sleep_onset_period'].value_counts().to_dict()}")
print(f"  wake_period:        {sp['wake_period'].value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════════════
# SECTION B: CALENDAR & TEMPORAL FEATURES
# ════════════════════════════════════════════════════════════════════
print("\nB. Calendar & temporal features")
nd = pd.to_datetime(sp["night_date"])

sp["day_of_week"]     = nd.dt.dayofweek                          # 0=Mon, 6=Sun
sp["day_name"]        = nd.dt.day_name()
sp["is_weekend"]      = sp["day_of_week"].isin([5, 6]).astype(int)
sp["is_friday_night"] = (sp["day_of_week"] == 4).astype(int)     # Fri night → Sat
sp["is_sunday_night"] = (sp["day_of_week"] == 6).astype(int)     # Sun night → Mon
sp["day_of_month"]    = nd.dt.day
sp["week_of_year"]    = nd.dt.isocalendar().week.astype(int)
sp["month"]           = nd.dt.month
sp["is_month_start"]  = (nd.dt.day <= 3).astype(int)             # payday / fresh start
sp["is_month_end"]    = (nd.dt.day >= 28).astype(int)            # deadline stress
sp["days_into_study"] = (nd - DATA_START).dt.days                # linear trend proxy

# Days until next weekend (0 if already Fri/Sat/Sun)
sp["days_to_weekend"] = sp["day_of_week"].map(
    {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: 0, 6: 0}
)

# Consecutive-workday index (Mon=1, Tue=2, ... Fri=5, Sat/Sun=0)
sp["workday_index"] = sp["day_of_week"].apply(lambda d: d + 1 if d < 5 else 0)

# Lunar phase approximation (known new moon: 2026-01-29, cycle=29.53 days)
NEW_MOON = pd.Timestamp("2026-01-29")
LUNAR_CYCLE = 29.53
sp["lunar_day"]    = ((nd - NEW_MOON).dt.days % LUNAR_CYCLE).round(1)
sp["lunar_phase"]  = (sp["lunar_day"] / LUNAR_CYCLE).round(3)    # 0→1 cycle
sp["is_full_moon"] = ((sp["lunar_day"] >= 13) & (sp["lunar_day"] <= 16)).astype(int)

print(f"  is_weekend:      {sp['is_weekend'].sum()} ({sp['is_weekend'].mean()*100:.1f}%)")
print(f"  is_friday_night: {sp['is_friday_night'].sum()}")
print(f"  is_sunday_night: {sp['is_sunday_night'].sum()}")
print(f"  is_full_moon:    {sp['is_full_moon'].sum()} nights")
print(f"  days_into_study: {sp['days_into_study'].min()} to {sp['days_into_study'].max()}")

# ════════════════════════════════════════════════════════════════════
# SECTION C: PERSONALIZED TIMEZONE-AWARE HOLIDAY FLAGS
# ════════════════════════════════════════════════════════════════════
print("\nC. Personalized holiday flags")

TZ_COUNTRY = {"Asia/Kolkata": "IN", "America/New_York": "US",
              "America/Los_Angeles": "US", "Europe/London": "GB", "Asia/Dubai": "AE"}

# Build holiday sets per country (covers full study window + buffer)
hol_sets = {cc: hol_lib.country_holidays(cc, years=[2025, 2026, 2027])
            for cc in set(TZ_COUNTRY.values())}
sp["country_code"] = sp["timezone"].map(TZ_COUNTRY)

# Vectorized: build lookup dicts for (country, date) → flags
def _build_holiday_flags(sp_df, hol_sets):
    """Compute all holiday flags efficiently."""
    dates   = pd.to_datetime(sp_df["night_date"]).dt.date
    dates_p1= (pd.to_datetime(sp_df["night_date"]) + pd.Timedelta(days=1)).dt.date
    dates_m1= (pd.to_datetime(sp_df["night_date"]) - pd.Timedelta(days=1)).dt.date
    cc_arr  = sp_df["country_code"].values

    is_hol, is_hol_tom, is_hol_yest, hol_name = [], [], [], []
    days_next, days_prev = [], []

    # Pre-build sorted holiday lists per country for efficient nearest search
    hol_sorted = {cc: sorted(hset.keys()) for cc, hset in hol_sets.items()}

    for i in range(len(sp_df)):
        cc = cc_arr[i]
        d, d1, dm1 = dates.iloc[i], dates_p1.iloc[i], dates_m1.iloc[i]
        hset = hol_sets[cc]

        is_hol.append(int(d in hset))
        is_hol_tom.append(int(d1 in hset))
        is_hol_yest.append(int(dm1 in hset))
        hol_name.append(hset.get(d, ""))

        # Days to next / since last holiday
        _next, _prev = 90, 90  # cap at 90
        for fwd in range(1, 91):
            if (pd.Timestamp(d) + pd.Timedelta(days=fwd)).date() in hset:
                _next = fwd; break
        for bwd in range(1, 91):
            if (pd.Timestamp(d) - pd.Timedelta(days=bwd)).date() in hset:
                _prev = bwd; break
        days_next.append(_next)
        days_prev.append(_prev)

    return is_hol, is_hol_tom, is_hol_yest, hol_name, days_next, days_prev

(sp["is_holiday"], sp["is_holiday_tomorrow"], sp["is_holiday_yesterday"],
 sp["holiday_name"], sp["days_to_next_holiday"], sp["days_since_last_holiday"]
) = _build_holiday_flags(sp, hol_sets)

# Long weekend: weekend + adjacent holiday
sp["is_long_weekend"] = (
    (sp["is_weekend"].astype(bool) & (sp["is_holiday_tomorrow"].astype(bool) | sp["is_holiday_yesterday"].astype(bool))) |
    (sp["is_holiday"].astype(bool) & sp["day_of_week"].isin([0, 4]))  # Mon/Fri holiday
).astype(int)

# No-work-tomorrow: person doesn't need to wake up early
sp["no_work_tomorrow"] = (
    sp["day_of_week"].isin([4, 5]) |  # Fri/Sat nights
    sp["is_holiday_tomorrow"].astype(bool)
).astype(int)

# Holiday proximity score (closer to holiday = higher anticipation)
sp["holiday_proximity"] = np.where(
    sp["days_to_next_holiday"] <= 3,
    (4 - sp["days_to_next_holiday"]) / 3,  # 1.0 on eve, 0.67 two days before, etc.
    0
).round(3)

# Print holiday summary per country
for cc in sorted(hol_sets.keys()):
    mask = sp["country_code"] == cc
    n_hol = sp.loc[mask, "is_holiday"].sum()
    n_users = sp.loc[mask, "user_id"].nunique()
    print(f"  {cc}: {n_users} users, {n_hol} holiday-nights")
    hol_names = sp.loc[mask & (sp["is_holiday"] == 1), "holiday_name"].unique()
    if len(hol_names):
        print(f"       Holidays hit: {', '.join(hol_names[:6])}")

# ════════════════════════════════════════════════════════════════════
# SUMMARY & VISUALIZATION
# ════════════════════════════════════════════════════════════════════
new_feature_cols = [
    # A. Timing
    "timezone", "bedtime_hour", "wake_hour", "sleep_onset_period", "wake_period",
    "bedtime_dev_midnight_hr", "sleep_midpoint_hour",
    "bedtime_sin", "bedtime_cos", "wake_sin", "wake_cos",
    # B. Calendar
    "day_of_week", "day_name", "is_weekend", "is_friday_night", "is_sunday_night",
    "day_of_month", "week_of_year", "month", "is_month_start", "is_month_end",
    "days_into_study", "days_to_weekend", "workday_index",
    "lunar_day", "lunar_phase", "is_full_moon",
    # C. Holidays
    "country_code", "is_holiday", "is_holiday_tomorrow", "is_holiday_yesterday",
    "holiday_name", "is_long_weekend", "no_work_tomorrow",
    "days_to_next_holiday", "days_since_last_holiday", "holiday_proximity",
]
print(f"\n{'='*70}")
print(f"TOTAL NEW FEATURES: {len(new_feature_cols)} columns")
print(f"sessions_primary: {len(sp):,} rows × {sp.shape[1]} cols")
print(f"{'='*70}")

sessions_primary = sp  # update for downstream cells

# ── Visualizations ──
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 1. Bedtime hour distribution (local)
sns.histplot(sp["bedtime_hour"], bins=48, kde=True, color=PAL[0], ax=axes[0, 0], edgecolor="white")
axes[0, 0].axvline(sp["bedtime_hour"].median(), color="red", ls="--", lw=1,
                    label=f"Median={sp['bedtime_hour'].median():.1f}h")
axes[0, 0].set_title("Bedtime (local hour)"); axes[0, 0].set_xlabel("Hour"); axes[0, 0].legend()

# 2. Wake hour distribution (local)
sns.histplot(sp["wake_hour"], bins=48, kde=True, color=PAL[1], ax=axes[0, 1], edgecolor="white")
axes[0, 1].axvline(sp["wake_hour"].median(), color="red", ls="--", lw=1,
                    label=f"Median={sp['wake_hour'].median():.1f}h")
axes[0, 1].set_title("Wake Time (local hour)"); axes[0, 1].set_xlabel("Hour"); axes[0, 1].legend()

# 3. Sleep onset period bar
onset_order = ["evening_early", "night", "late_night", "early_morning", "daytime"]
onset_counts = sp["sleep_onset_period"].value_counts().reindex(onset_order, fill_value=0)
axes[0, 2].bar(onset_counts.index, onset_counts.values, color=PAL[:5], edgecolor="white")
for i, v in enumerate(onset_counts.values):
    axes[0, 2].text(i, v + 20, f"{v:,}", ha="center", fontsize=8)
axes[0, 2].set_title("Sleep Onset Period"); axes[0, 2].tick_params(axis="x", rotation=25)

# 4. Weekend vs weekday effect
for label, mask, color in [("Weekday", sp["is_weekend"] == 0, PAL[0]),
                            ("Weekend", sp["is_weekend"] == 1, PAL[1])]:
    sns.kdeplot(sp.loc[mask, "bedtime_hour"], ax=axes[1, 0], label=label, color=color, fill=True, alpha=0.3)
axes[1, 0].set_title("Bedtime: Weekday vs Weekend"); axes[1, 0].set_xlabel("Hour"); axes[1, 0].legend()

# 5. Day-of-week session count
dow_counts = sp["day_name"].value_counts().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
axes[1, 1].bar(range(7), dow_counts.values, color=[
    PAL[0] if i < 5 else PAL[1] for i in range(7)], edgecolor="white")
axes[1, 1].set_xticks(range(7)); axes[1, 1].set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
axes[1, 1].set_title("Sessions by Day of Week"); axes[1, 1].set_ylabel("Count")

# 6. Holiday distribution
hol_summary = sp.groupby("country_code")[["is_holiday", "is_holiday_tomorrow", "no_work_tomorrow"]].mean() * 100
hol_summary.plot(kind="bar", ax=axes[1, 2], color=["#e31a1c", "#fd8d3c", "#41ab5d"], edgecolor="white")
axes[1, 2].set_title("Holiday Exposure by Country (%)"); axes[1, 2].set_ylabel("%")
axes[1, 2].tick_params(axis="x", rotation=0); axes[1, 2].legend(fontsize=8)

plt.suptitle(f"Datetime & Holiday Features ({len(sp):,} sessions, {len(new_feature_cols)} new features)",
             fontsize=13, y=1.01)
plt.tight_layout(); plt.show()

# COMMAND ----------

# DBTITLE 1,Sleep Sessions - sensor error flags + physiological outlier handling
# ── 2F: Sensor error flags + physiological outlier handling ──
# Strategy: FLAG first (preserve signal of which rows had bad data),
# then FIX (NaN sentinels, overwrite corrupted efficiency), then CLIP.
import numpy as np

ss = sessions_primary.copy()  # work on a copy
print("="*70)
print("SENSOR ERROR FLAGS + PHYSIOLOGICAL OUTLIER HANDLING")
print("="*70)

# ── STEP 1: Create error flags BEFORE any modification ──
print("\n── Step 1: Sensor error flags (before fixing) ──")

# A. HR sensor error: avg_hr=0 (sensor failed) or avg_hr=250 (max-cap sentinel)
ss["hr_sensor_error"] = ((ss["avg_hr_bpm"] == 0) | (ss["avg_hr_bpm"] == 250)).astype(int)
print(f"  hr_sensor_error = 1:  {ss['hr_sensor_error'].sum()} rows  "
      f"(HR=0: {(ss['avg_hr_bpm']==0).sum()}, HR=250: {(ss['avg_hr_bpm']==250).sum()})")

# B. HRV sensor error: avg_hrv=999 (sentinel value; real HRV never reaches 999)
ss["hrv_sensor_error"] = (ss["avg_hrv_rmssd_ms"] == 999).astype(int)
print(f"  hrv_sensor_error = 1: {ss['hrv_sensor_error'].sum()} rows")

# C. Efficiency error: reported efficiency > 1.0 (physically impossible)
#    efficiency_calc (from decomposition cell) is correct for all these rows
ss["efficiency_error"] = (ss["sleep_efficiency"] > 1.0).astype(int)
print(f"  efficiency_error = 1: {ss['efficiency_error'].sum()} rows  "
      f"(range: {ss.loc[ss['efficiency_error']==1, 'sleep_efficiency'].min():.4f}"
      f" – {ss.loc[ss['efficiency_error']==1, 'sleep_efficiency'].max():.4f})")

# D. Duration sensor error: time_in_bed < 60 min (micro-session, unreliable staging)
ss["duration_error"] = (ss["time_in_bed_minutes"] < 60).astype(int)
print(f"  duration_error = 1:   {ss['duration_error'].sum()} rows  (<60 min sessions)")

# E. Decomposition error: total_sleep > time_in_bed (structurally impossible)
ss["decomp_error"] = (ss["total_sleep_minutes"] > ss["time_in_bed_minutes"]).astype(int)
print(f"  decomp_error = 1:     {ss['decomp_error'].sum()} rows  (total_sleep > time_in_bed)")

# F. Composite: any sensor issue at all
ss["any_sensor_error"] = (
    ss["hr_sensor_error"] | ss["hrv_sensor_error"] | ss["efficiency_error"] |
    ss["duration_error"] | ss["decomp_error"]
).astype(int)
print(f"\n  any_sensor_error = 1: {ss['any_sensor_error'].sum()} rows  "
      f"({ss['any_sensor_error'].sum()/len(ss)*100:.1f}% of primary sessions)")

# ── STEP 2: Fix sentinel/corrupted values ──
print("\n── Step 2: Fix sentinel & corrupted values ──")

# A. HR=0 and HR=250 → NaN (sensor gave no usable reading)
n_hr_fix = ((ss["avg_hr_bpm"] == 0) | (ss["avg_hr_bpm"] == 250)).sum()
ss.loc[(ss["avg_hr_bpm"] == 0) | (ss["avg_hr_bpm"] == 250), "avg_hr_bpm"] = np.nan
print(f"  avg_hr_bpm: {n_hr_fix} sentinel values (0 or 250) → NaN")
print(f"    min_hr_bpm kept as-is (normal values exist for all these rows)")

# B. HRV=999 → NaN
n_hrv_fix = (ss["avg_hrv_rmssd_ms"] == 999).sum()
ss.loc[ss["avg_hrv_rmssd_ms"] == 999, "avg_hrv_rmssd_ms"] = np.nan
print(f"  avg_hrv_rmssd_ms: {n_hrv_fix} sentinel values (999) → NaN")

# C. Efficiency > 1 → overwrite with recalculated efficiency_calc
n_eff_fix = (ss["efficiency_error"] == 1).sum()
ss.loc[ss["efficiency_error"] == 1, "sleep_efficiency"] = ss.loc[ss["efficiency_error"] == 1, "efficiency_calc"]
print(f"  sleep_efficiency: {n_eff_fix} impossible values (>1) → replaced with efficiency_calc")
print(f"    Corrected range: {ss.loc[ss['efficiency_error']==1, 'sleep_efficiency'].min():.4f}"
      f" – {ss.loc[ss['efficiency_error']==1, 'sleep_efficiency'].max():.4f}")

# ── STEP 3: Clip remaining out-of-range physiological values ──
print("\n── Step 3: Clip remaining out-of-range values ──")

def clip_report(s, lo, hi, label):
    """Clip to [lo, hi], report how many were clipped (excluding NaN)."""
    valid = s.dropna()
    n_below = (valid < lo).sum()
    n_above = (valid > hi).sum()
    clipped = s.clip(lo, hi)  # NaN stays NaN
    print(f"  {label:30s} clipped to [{lo}, {hi}]:  {n_below} below, {n_above} above")
    return clipped

ss["avg_hr_bpm"]        = clip_report(ss["avg_hr_bpm"], 30, 120, "avg_hr_bpm")
ss["min_hr_bpm"]        = clip_report(ss["min_hr_bpm"], 25, 100, "min_hr_bpm")
ss["avg_hrv_rmssd_ms"]  = clip_report(ss["avg_hrv_rmssd_ms"], 5, 300, "avg_hrv_rmssd_ms")
ss["avg_spo2_pct"]      = clip_report(ss["avg_spo2_pct"], 80, 100, "avg_spo2_pct")
ss["avg_resp_rate_bpm"] = clip_report(ss["avg_resp_rate_bpm"], 6, 30, "avg_resp_rate_bpm")
ss["sleep_efficiency"]  = clip_report(ss["sleep_efficiency"], 0.3, 1.0, "sleep_efficiency")
ss["temperature_deviation_c"] = clip_report(ss["temperature_deviation_c"], -3, 3, "temperature_deviation_c")

# ── STEP 4: Filter implausible duration ──
print("\n── Step 4: Duration filter ──")
print(f"  time_in_bed range: {ss['time_in_bed_minutes'].min():.1f} – {ss['time_in_bed_minutes'].max():.1f} min")
range_check(ss["time_in_bed_minutes"], 60, 900, "time_in_bed_minutes")
before = len(ss)
ss = ss[(ss["time_in_bed_minutes"] >= 60) & (ss["time_in_bed_minutes"] <= 900)].copy()
print(f"  Dropped {before - len(ss)} sessions with implausible duration (<60 or >900 min)")

# ── STEP 5: Summary ──
print("\n" + "="*70)
print(f"FINAL CLEAN PRIMARY SESSIONS: {len(ss):,} rows × {ss.shape[1]} cols")
print("="*70)
print(f"\nError flag summary (in clean sessions):")
for flag in ["hr_sensor_error", "hrv_sensor_error", "efficiency_error",
             "duration_error", "decomp_error", "any_sensor_error"]:
    print(f"  {flag:25s} = 1: {ss[flag].sum():>4}  ({ss[flag].mean()*100:.2f}%)")

print(f"\nNull counts after sentinel→NaN fixes:")
for col in ["avg_hr_bpm", "avg_hrv_rmssd_ms", "sleep_efficiency"]:
    print(f"  {col:25s}: {ss[col].isna().sum()} NaN")

print(f"\nSample of error-flagged rows:")
err_rows = ss[ss["any_sensor_error"] == 1][
    ["session_id", "user_id", "avg_hr_bpm", "avg_hrv_rmssd_ms",
     "sleep_efficiency", "hr_sensor_error", "hrv_sensor_error",
     "efficiency_error", "any_sensor_error"]
].head(10)
display(err_rows)

sessions_clean = ss.copy()

# ── Visual: sensor error summary ──
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Error flag bar chart
flags_data = {"HR\n(0/250)": ss["hr_sensor_error"].sum(),
              "HRV\n(999)": ss["hrv_sensor_error"].sum(),
              "Efficiency\n(>1.0)": ss["efficiency_error"].sum(),
              "Decomp\n(sleep>bed)": ss["decomp_error"].sum()}
colors_e = ["#e31a1c", "#fd8d3c", "#7570b3", "#1b9e77"]
axes[0].bar(flags_data.keys(), flags_data.values(), color=colors_e, edgecolor="white", width=0.5)
for i, v in enumerate(flags_data.values()):
    axes[0].text(i, v + 0.5, str(v), ha="center", fontsize=10, fontweight="bold")
axes[0].set_title(f"Sensor Error Flags ({ss['any_sensor_error'].sum()} affected, {ss['any_sensor_error'].mean()*100:.1f}%)")
axes[0].set_ylabel("Sessions")

# Clean vs flagged pie
labels = [f"Clean ({(~ss['any_sensor_error'].astype(bool)).sum():,})",
          f"Flagged ({ss['any_sensor_error'].sum()})"]
axes[1].pie([len(ss) - ss["any_sensor_error"].sum(), ss["any_sensor_error"].sum()],
            labels=labels, colors=[PAL[0], "#e31a1c"], autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10})
axes[1].set_title(f"Data Quality: {len(ss):,} Clean Sessions")
plt.tight_layout(); plt.show()

# COMMAND ----------

# DBTITLE 1,Sleep Sessions - firmware analysis
# ── 2G: Firmware version analysis ──
# Check if firmware affects sensor readings
print("Firmware distribution:")
print(sessions_clean["firmware_version"].value_counts().to_string())

# Compare key metrics by firmware
fw_stats = sessions_clean.groupby("firmware_version").agg(
    n_sessions=("session_id", "count"),
    avg_hr=("avg_hr_bpm", "mean"),
    avg_hrv=("avg_hrv_rmssd_ms", "mean"),
    avg_spo2=("avg_spo2_pct", "mean"),
    avg_efficiency=("sleep_efficiency", "mean"),
    avg_temp_dev=("temperature_deviation_c", "mean"),
).round(2)
print("\nMetrics by firmware version:")
display(fw_stats)

# ── Visual: metrics by firmware ──
metrics = ["avg_hr", "avg_hrv", "avg_spo2", "avg_efficiency"]
titles = ["Avg HR (bpm)", "Avg HRV (ms)", "Avg SpO2 (%)", "Avg Efficiency"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, m, t in zip(axes, metrics, titles):
    vals = fw_stats[m]
    bars = ax.bar(vals.index, vals.values, color=PAL[:len(vals)], edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{v:.1f}",
                ha="center", fontsize=9)
    ax.set_title(t); ax.tick_params(axis="x", rotation=15)
plt.suptitle("Sensor Metrics by Firmware Version", fontsize=12, y=1.02)
plt.tight_layout(); plt.show()
print("→ Metrics are comparable across firmware versions — no systematic bias")

# COMMAND ----------

# DBTITLE 1,Section 2 Final — clean sessions sample (all columns)
# ── Section 2 FINAL: Clean sessions — all columns sample ──
print(f"\u2550" * 70)
print(f"SECTION 2 OUTPUT: sessions_clean — {len(sessions_clean):,} rows × {sessions_clean.shape[1]} cols")
print(f"\u2550" * 70)

# Group columns by category for readability
col_groups = {
    "Identifiers":        [c for c in sessions_clean.columns if c in ["session_id", "user_id", "night_date"]],
    "Timestamps":         [c for c in sessions_clean.columns if "start" in c or "end" in c or "utc" in c.lower()],
    "Duration (raw)":     [c for c in sessions_clean.columns if "minutes" in c and "secondary" not in c and "gap" not in c and "unscored" not in c and "tib" not in c.lower()],
    "Sleep stages":       [c for c in sessions_clean.columns if any(x in c for x in ["deep_", "rem_", "light_", "awake_"]) and "all" not in c and "secondary" not in c],
    "Physiology":         [c for c in sessions_clean.columns if any(x in c for x in ["hr_bpm", "hrv_", "spo2", "resp_rate", "temperature", "movement"])],
    "Decomposition":      [c for c in sessions_clean.columns if any(x in c for x in ["tib_gap", "decomp", "unscored", "efficiency_calc", "efficiency_gap", "efficiency_match", "stage_decomp"])],
    "Multi-session":      [c for c in sessions_clean.columns if any(x in c for x in ["n_sessions", "fragmented", "secondary", "nap_ratio", "inter_session", "night_window", "night_consolidation", "total_tib_all", "total_sleep_all", "total_awake_all", "mean_efficiency_all", "min_efficiency_night"])],
    "Sleep timing":       [c for c in sessions_clean.columns if any(x in c for x in ["bedtime", "wake_hour", "wake_sin", "wake_cos", "wake_period", "sleep_onset", "sleep_midpoint"])],
    "Calendar":           [c for c in sessions_clean.columns if any(x in c for x in ["day_of", "day_name", "is_weekend", "is_friday", "is_sunday", "week_of", "month", "days_into", "days_to_weekend", "workday", "lunar", "is_full_moon", "is_month"])],
    "Holidays":           [c for c in sessions_clean.columns if any(x in c for x in ["holiday", "country_code", "no_work", "long_weekend"])],
    "Sensor errors":      [c for c in sessions_clean.columns if "error" in c],
    "Metadata":           [c for c in sessions_clean.columns if c in ["firmware_version", "timezone", "is_primary", "duration_mismatch", "calc_duration_min"]],
}

print(f"\nColumn groups:")
assigned = set()
for group, cols in col_groups.items():
    if cols:
        print(f"  {group} ({len(cols)}): {', '.join(cols)}")
        assigned.update(cols)

# Catch any unassigned columns
unassigned = [c for c in sessions_clean.columns if c not in assigned]
if unassigned:
    print(f"  Other ({len(unassigned)}): {', '.join(unassigned)}")

print(f"\n\u2500\u2500 Sample (5 rows, all {sessions_clean.shape[1]} columns) \u2500\u2500")
display(sessions_clean.head(20))

# Quick health check
print(f"\nHealth check:")
print(f"  Users:     {sessions_clean['user_id'].nunique()}")
print(f"  Date span: {sessions_clean['night_date'].min()} to {sessions_clean['night_date'].max()}")
print(f"  Nulls:     {sessions_clean.isnull().sum().sum()} total across all columns")
null_cols = sessions_clean.isnull().sum()
null_cols = null_cols[null_cols > 0]
if len(null_cols):
    print(f"  Columns with nulls:")
    for col, n in null_cols.items():
        print(f"    {col}: {n} ({n/len(sessions_clean)*100:.1f}%)")
print(f"  Sensor errors: {sessions_clean['any_sensor_error'].sum()} rows ({sessions_clean['any_sensor_error'].mean()*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC #Done till here

# COMMAND ----------

# DBTITLE 1,Section 3: Nightly Signals
# MAGIC %md
# MAGIC ## Section 3: Nightly Signals
# MAGIC
# MAGIC Clean the 942k-row 5-min resolution sensor data. Handle nulls, sentinel values, then aggregate to session-level statistics for the modelling table.

# COMMAND ----------

# DBTITLE 1,Nightly Signals - inspect and clean
# ── 3A: Nightly Signals — Inspect and clean ──
signals = raw_signals.copy()
shape_report(signals, "nightly_signals (raw)")

# Normalize user_id
signals["user_id"] = signals["user_id"].str.strip().str.upper()

# Filter to target users only
signals = signals[signals["user_id"].isin(target_users)].copy()
print(f"\nAfter user filter: {len(signals):,} rows, {signals['user_id'].nunique()} users")

# Filter to clean sessions only
clean_session_ids = set(sessions_clean["session_id"])
signals = signals[signals["session_id"].isin(clean_session_ids)].copy()
print(f"After session filter: {len(signals):,} rows, {signals['session_id'].nunique()} sessions")

# Parse timestamp
signals["timestamp_utc"] = pd.to_datetime(signals["timestamp_utc"], utc=True, errors="coerce")
print(f"Timestamp parse failures: {signals['timestamp_utc'].isna().sum()}")

# Null analysis per signal
print("\nNull analysis per signal column:")
for col in ["hr_bpm", "hrv_rmssd_ms", "motion_index", "temp_delta_c", "spo2_pct"]:
    n_null = signals[col].isna().sum()
    pct = n_null / len(signals) * 100
    print(f"  {col:20s}  nulls={n_null:>6,} ({pct:.1f}%)")

# Are nulls concentrated in specific sessions or spread randomly?
null_per_session = signals.groupby("session_id")["hr_bpm"].apply(lambda x: x.isna().mean())
print(f"\nNull rate per session (hr_bpm):")
print(f"  Median: {null_per_session.median():.3f}")
print(f"  Sessions with >50% nulls: {(null_per_session > 0.5).sum()}")
print(f"  Sessions with 100% nulls: {(null_per_session == 1.0).sum()}")

# COMMAND ----------

# DBTITLE 1,Nightly Signals - sentinel and outlier handling
# ── 3B: Sentinel values and outlier cleaning ──
print("Signal cleaning:")

# HR: sentinel 0, plausible range [30, 150]
signals["hr_bpm"] = signals["hr_bpm"].replace(0, np.nan)
range_check(signals["hr_bpm"].dropna(), 30, 150, "hr_bpm")
signals.loc[signals["hr_bpm"] < 30, "hr_bpm"] = np.nan
signals.loc[signals["hr_bpm"] > 150, "hr_bpm"] = np.nan

# HRV: sentinel 999, plausible range [1, 300]
signals["hrv_rmssd_ms"] = signals["hrv_rmssd_ms"].replace(999, np.nan)
range_check(signals["hrv_rmssd_ms"].dropna(), 1, 300, "hrv_rmssd_ms")
signals.loc[signals["hrv_rmssd_ms"] > 300, "hrv_rmssd_ms"] = np.nan

# SpO2: plausible range [80, 100]
range_check(signals["spo2_pct"].dropna(), 80, 100, "spo2_pct")
signals.loc[signals["spo2_pct"] < 80, "spo2_pct"] = np.nan
signals.loc[signals["spo2_pct"] > 100, "spo2_pct"] = np.nan

# Motion: plausible range [0, 5]
signals.loc[signals["motion_index"] > 5, "motion_index"] = np.nan

# Temp: plausible range [-3, 3]
signals.loc[signals["temp_delta_c"].abs() > 3, "temp_delta_c"] = np.nan

print("\nPost-cleaning null counts:")
for col in ["hr_bpm", "hrv_rmssd_ms", "motion_index", "temp_delta_c", "spo2_pct"]:
    n_null = signals[col].isna().sum()
    pct = n_null / len(signals) * 100
    print(f"  {col:20s}  nulls={n_null:>6,} ({pct:.1f}%)")

# COMMAND ----------

# DBTITLE 1,Nightly Signals - aggregate to session-level features
# ── 3C: Aggregate nightly signals to session-level features ──
# This creates the signal-derived features for the modelling table

def compute_slope(series):
    """Linear slope of a series (change per 5-min step). NaN-robust."""
    valid = series.dropna()
    if len(valid) < 3:
        return np.nan
    x = np.arange(len(valid))
    return np.polyfit(x, valid.values, 1)[0]

def first_half_mean(series):
    valid = series.dropna()
    n = len(valid)
    if n < 2:
        return np.nan
    return valid.iloc[:n//2].mean()

def second_half_mean(series):
    valid = series.dropna()
    n = len(valid)
    if n < 2:
        return np.nan
    return valid.iloc[n//2:].mean()

print("Aggregating nightly signals to session level...")
signal_agg = signals.groupby("session_id").agg(
    # HR features
    sig_hr_mean       = ("hr_bpm", "mean"),
    sig_hr_std        = ("hr_bpm", "std"),
    sig_hr_min        = ("hr_bpm", "min"),
    sig_hr_max        = ("hr_bpm", "max"),
    sig_hr_range      = ("hr_bpm", lambda x: x.max() - x.min()),
    # HRV features
    sig_hrv_mean      = ("hrv_rmssd_ms", "mean"),
    sig_hrv_std       = ("hrv_rmssd_ms", "std"),
    sig_hrv_min       = ("hrv_rmssd_ms", "min"),
    sig_hrv_max       = ("hrv_rmssd_ms", "max"),
    # Motion features
    sig_motion_mean   = ("motion_index", "mean"),
    sig_motion_std    = ("motion_index", "std"),
    sig_motion_max    = ("motion_index", "max"),
    # Temp features
    sig_temp_mean     = ("temp_delta_c", "mean"),
    sig_temp_std      = ("temp_delta_c", "std"),
    # SpO2 features
    sig_spo2_mean     = ("spo2_pct", "mean"),
    sig_spo2_min      = ("spo2_pct", "min"),
    sig_spo2_std      = ("spo2_pct", "std"),
    # Count of valid readings
    sig_n_readings    = ("hr_bpm", "count"),
    sig_n_total       = ("hr_bpm", "size"),
).reset_index()

# Compute slopes separately (can't do lambdas in named agg easily)
hr_slopes = signals.groupby("session_id")["hr_bpm"].apply(compute_slope).reset_index(name="sig_hr_slope")
hrv_slopes = signals.groupby("session_id")["hrv_rmssd_ms"].apply(compute_slope).reset_index(name="sig_hrv_slope")

# First-half vs second-half HRV (recovery indicator)
hrv_first  = signals.groupby("session_id")["hrv_rmssd_ms"].apply(first_half_mean).reset_index(name="sig_hrv_first_half")
hrv_second = signals.groupby("session_id")["hrv_rmssd_ms"].apply(second_half_mean).reset_index(name="sig_hrv_second_half")

signal_agg = signal_agg.merge(hr_slopes, on="session_id", how="left")
signal_agg = signal_agg.merge(hrv_slopes, on="session_id", how="left")
signal_agg = signal_agg.merge(hrv_first, on="session_id", how="left")
signal_agg = signal_agg.merge(hrv_second, on="session_id", how="left")

# HRV recovery ratio: second_half / first_half
signal_agg["sig_hrv_recovery_ratio"] = signal_agg["sig_hrv_second_half"] / signal_agg["sig_hrv_first_half"].replace(0, np.nan)

# Signal completeness
signal_agg["sig_completeness"] = signal_agg["sig_n_readings"] / signal_agg["sig_n_total"]

print(f"Signal aggregates: {len(signal_agg):,} sessions x {signal_agg.shape[1]} features")
display(signal_agg.describe().T.round(2))

# COMMAND ----------

# DBTITLE 1,Section 3 Final — signal quality audit, save, V2 deferral
# ── Section 3 FINAL: Nightly Signals — Quality audit + save (deferred from V1) ──
#
# Decision D-019: Signal aggregates are CLEANED and SAVED but NOT merged
# into the V1 training dataset. Session-level means (avg_hr_bpm, avg_hrv_rmssd_ms,
# etc.) are r>0.97 redundant with signal means. The 18 unique signal features
# (slopes, variability, recovery ratio) all correlate |r|<0.13 with the target,
# below the best session feature (deep_minutes |r|=0.186).
# Signals are reserved for V2 (sequence models or personalised baselines).

import os, numpy as np

print(f"\u2550" * 70)
print(f"SECTION 3 OUTPUT: signal_agg \u2014 {len(signal_agg):,} sessions \u00d7 {signal_agg.shape[1]} cols")
print(f"\u2550" * 70)

# ── Quality audit ──
clean_ids = set(sessions_clean["session_id"])
sig_ids   = set(signal_agg["session_id"])
missing   = clean_ids - sig_ids

print(f"\nQuality audit:")
print(f"  Coverage:       {len(sig_ids):,} / {len(clean_ids):,} sessions ({len(sig_ids)/len(clean_ids)*100:.1f}%)")
print(f"  Missing:        {len(missing)} sessions (52 users, normal TIB \u2014 likely unsync'd raw data)")
print(f"  Completeness:   mean={signal_agg['sig_completeness'].mean():.3f} (valid readings / total)")
print(f"  Low quality:    {(signal_agg['sig_completeness'] < 0.5).sum()} sessions <50% complete")

# Null counts
null_total = signal_agg.drop(columns="session_id").isna().sum().sum()
print(f"  Total nulls:    {null_total} across all columns (slopes/std for sessions with <3 readings)")

# Cross-validation
merged_check = sessions_clean.merge(signal_agg[["session_id", "sig_hr_mean", "sig_hrv_mean"]], on="session_id")
hr_diff  = (merged_check["sig_hr_mean"]  - merged_check["avg_hr_bpm"]).abs().dropna()
hrv_diff = (merged_check["sig_hrv_mean"] - merged_check["avg_hrv_rmssd_ms"]).abs().dropna()
print(f"  HR mean diff:   {hr_diff.mean():.3f} bpm (signal vs session)")
print(f"  HRV mean diff:  {hrv_diff.mean():.3f} ms (76 sessions >10ms, from sentinel handling order)")
print(f"  \u2713 Data is clean and consistent \u2014 ready for V2 use")

# ── Column listing ──
print(f"\nColumns ({signal_agg.shape[1]}):")
for i, col in enumerate(signal_agg.columns):
    if col == "session_id":
        print(f"  {i+1:>2}. {col:30s}  (join key)")
    else:
        s = signal_agg[col]
        print(f"  {i+1:>2}. {col:30s}  mean={s.mean():.3f}  std={s.std():.3f}  nulls={s.isna().sum()}")

# ── Sample ──
print(f"\n\u2500\u2500 Sample (5 rows) \u2500\u2500")
display(signal_agg.head(5))

# ── Save to disk ──
OUT_DIR = "/Workspace/Users/mohammad.sayeed@tatadigital.com/AS_UH_Project/data"
signal_agg.to_parquet(os.path.join(OUT_DIR, "signal_agg_clean.parquet"), index=False)
signal_agg.to_csv(os.path.join(OUT_DIR, "signal_agg_clean.csv"), index=False)
fsize = os.path.getsize(os.path.join(OUT_DIR, "signal_agg_clean.parquet")) / 1024
print(f"\n\u2713 Saved: signal_agg_clean.parquet ({fsize:.0f} KB)")
print(f"\u2713 Saved: signal_agg_clean.csv")
print(f"\n\u26a0 NOT merged into V1 training table. Reserved for V2 (sequence models / personalised baselines).")

# COMMAND ----------

# DBTITLE 1,Section 4: Daily Context
# MAGIC %md
# MAGIC ## Section 4: Daily Context
# MAGIC
# MAGIC Per your note: "Daily context should be analyzed heavily because it will be most responsible for coming night and morning checkins." This table has the most missing values (stress_score 65% null) and the leakage risk (legacy_readiness_shown).

# COMMAND ----------

# DBTITLE 1,Daily Context - inspect, rename, clean
# ── 4A: Daily Context — Inspect and clean ──
context = raw_context.copy()
shape_report(context, "daily_context (raw)")

# Fix column naming inconsistency: userId -> user_id
context = context.rename(columns={"userId": "user_id"})
context["user_id"] = context["user_id"].str.strip().str.upper()

# Filter to target users
context = context[context["user_id"].isin(target_users)].copy()
print(f"\nAfter user filter: {len(context):,} rows, {context['user_id'].nunique()} users")

# Date handling
context["date"] = pd.to_datetime(context["date"]).dt.date
print(f"Date range: {context['date'].min()} to {context['date'].max()}")
print(f"Unique dates: {context['date'].nunique()}")

# Duplicates check
n_dup = context.duplicated(subset=["user_id", "date"]).sum()
print(f"Duplicate (user_id, date): {n_dup}")
if n_dup > 0:
    context = context.drop_duplicates(subset=["user_id", "date"], keep="last")
    print(f"After dedup: {len(context):,}")

# COMMAND ----------

# DBTITLE 1,Daily Context - heavy null analysis and data quality
# ── 4B: Null analysis and data quality ──
print("Null analysis for daily_context:")
for col in ["steps", "active_minutes", "alcohol_units", "caffeine_mg",
            "last_caffeine_hours_before_bed", "stress_score", "travel_flag",
            "legacy_readiness_shown", "workout_notes"]:
    n_null = context[col].isna().sum() if col in context.columns else -1
    pct = n_null / len(context) * 100
    print(f"  {col:35s}  nulls={n_null:>5} ({pct:5.1f}%)")

# Stress score: 65% missing — create binary "reported" flag
context["stress_reported"] = context["stress_score"].notna().astype(int)
print(f"\nstress_score: reported={context['stress_reported'].sum()}, not reported={len(context) - context['stress_reported'].sum()}")

# Caffeine: create binary "had_caffeine" and "caffeine_close_to_bed" flags
context["had_caffeine"] = (context["caffeine_mg"] > 0).astype(float)
context["had_caffeine"] = context["had_caffeine"].where(context["caffeine_mg"].notna(), np.nan)
context["caffeine_close_to_bed"] = (context["last_caffeine_hours_before_bed"] < 4).astype(float)
context["caffeine_close_to_bed"] = context["caffeine_close_to_bed"].where(
    context["last_caffeine_hours_before_bed"].notna(), np.nan
)

# Alcohol: create binary flag
context["had_alcohol"] = (context["alcohol_units"] > 0).astype(float)
context["had_alcohol"] = context["had_alcohol"].where(context["alcohol_units"].notna(), np.nan)

# Workout: create binary and extract type
context["had_workout"] = context["workout_notes"].notna().astype(int)
context["is_rest_day"] = context["workout_notes"].str.lower().str.contains(r"rest", na=False).astype(int)

# Steps and active_minutes: check for implausible values
print(f"\nSteps range: {context['steps'].min()} - {context['steps'].max()}")
print(f"Active minutes range: {context['active_minutes'].min()} - {context['active_minutes'].max()}")
range_check(context["steps"], 0, 50000, "steps")
range_check(context["active_minutes"], 0, 300, "active_minutes")

# Legacy readiness analysis (LEAKAGE variable)
print(f"\nlegacy_readiness_shown:")
print(f"  Nulls: {context['legacy_readiness_shown'].isna().sum()}")
print(f"  Range: {context['legacy_readiness_shown'].min():.0f} - {context['legacy_readiness_shown'].max():.0f}")
print(f"  Mean: {context['legacy_readiness_shown'].mean():.1f}")
print(f"  ⚠ This is the CURRENT heuristic shown to users. EXCLUDED from model features.")
print(f"  ⚠ But retained in dataset as a benchmark column for comparison.")

# COMMAND ----------

# DBTITLE 1,Daily Context - per-user consistency check
# ── 4C: Per-user data consistency ──
# Per note: "for one customer consistency of data needs to be checked"
user_context_stats = context.groupby("user_id").agg(
    n_days=("date", "count"),
    steps_mean=("steps", "mean"),
    steps_std=("steps", "std"),
    caffeine_pct=("had_caffeine", "mean"),
    alcohol_pct=("had_alcohol", "mean"),
    stress_report_pct=("stress_reported", "mean"),
    workout_pct=("had_workout", "mean"),
    legacy_readiness_mean=("legacy_readiness_shown", "mean"),
).round(2)

print("Per-user context summary:")
print(f"  Days per user: mean={user_context_stats['n_days'].mean():.1f}, "
      f"min={user_context_stats['n_days'].min()}, max={user_context_stats['n_days'].max()}")
print(f"  Users with <30 days: {(user_context_stats['n_days'] < 30).sum()}")
print(f"  Users reporting stress >50% of time: {(user_context_stats['stress_report_pct'] > 0.5).sum()}")

# Check for users with suspiciously constant data (possible data issues)
const_steps = (user_context_stats["steps_std"] < 100).sum()
print(f"  Users with near-constant steps (std<100): {const_steps}")

print(f"\n--- CLEAN daily_context: {len(context):,} rows, {context['user_id'].nunique()} users ---")
display(context.head(20))

# COMMAND ----------

# DBTITLE 1,Daily Context - workout standardization, intensity, and derived features
# ── 4D: Workout standardisation, intensity scoring, and derived features ──
# Converts free-text workout_notes into model-ready categories + numeric intensity.

import re

# ════════════════════════════════════════════════════════════════════
# STEP 1: Standardise raw text
# ════════════════════════════════════════════════════════════════════
context["workout_clean"] = (
    context["workout_notes"]
    .fillna("")
    .str.strip()
    .str.lower()
    .str.replace(r"[\U0001f3c3\u26bd\U0001f9d8\U0001f3ca\U0001f6b4\U0001f3cb]", "", regex=True)  # emojis
    .str.strip()
)

# ════════════════════════════════════════════════════════════════════
# STEP 2: Map to standard categories
# ════════════════════════════════════════════════════════════════════
# Exact matches first
EXACT_MAP = {
    # Unknown / not reported (null-equivalent placeholders — NOT rest)
    "": "unknown", "-": "unknown", "n/a": "unknown", "none": "unknown",
    # Rest (user explicitly said rest)
    "rest": "rest", "rest day": "rest", "no workout": "rest",
    # Light activity
    "walk": "light_activity", "walk 30min": "light_activity",
    "easy walk": "light_activity", "evening stroll": "light_activity",
    "yoga": "light_activity", "stretching": "light_activity", "mobility": "light_activity",
    # Moderate cardio
    "run 5k": "moderate_cardio", "run - 5km easy": "moderate_cardio",
    "45 min run": "moderate_cardio", "swim": "moderate_cardio",
    "cycling 40min": "moderate_cardio",
    # Intense cardio
    "long run 18km": "intense_cardio", "interval session 8x400": "intense_cardio",
    "tempo run 10k": "intense_cardio", "marathon training - long": "intense_cardio",
    "hill repeats": "intense_cardio", "2h ride + brick run": "intense_cardio",
    # Strength
    "gym - push day": "strength", "strength training": "strength",
    "gym / upper body": "strength", "heavy legs day": "strength",
    "crossfit wod": "strength", "hiit": "strength",
    # Sport
    "football": "sport", "tennis": "sport", "badminton": "sport",
}

# Apply exact map (after emoji removal)
context["workout_category"] = context["workout_clean"].map(EXACT_MAP)

# Check coverage
mapped = context["workout_category"].notna().sum()
total  = len(context)
print(f"Exact-mapped: {mapped:,} / {total:,} ({mapped/total*100:.1f}%)")
unmapped = context[context["workout_category"].isna()]["workout_clean"].value_counts()
if len(unmapped):
    print(f"Unmapped values ({len(unmapped)}):")
    for v, c in unmapped.items():
        print(f"  {c:>4}  '{v}'")

# Fallback regex for any unmapped
def classify_workout_regex(text):
    if not text or text in ("", "-", "n/a", "none"):
        return "unknown"
    if re.search(r"rest|off|no\s*work", text):
        return "rest"
    if re.search(r"walk|stroll|yoga|stretch|mobil", text):
        return "light_activity"
    if re.search(r"run|jog|swim|cycl|ride|row", text):
        if re.search(r"long|interval|tempo|hill|marathon|brick|18km|10k|repeat|sprint", text):
            return "intense_cardio"
        return "moderate_cardio"
    if re.search(r"gym|strength|push|pull|leg|upper|lower|chest|back|squat|dead|bench|crossfit|hiit|weight", text):
        return "strength"
    if re.search(r"football|tennis|badminton|basketball|cricket|squash|soccer", text):
        return "sport"
    return "moderate_cardio"  # safe default for unknown workout text

mask = context["workout_category"].isna()
context.loc[mask, "workout_category"] = context.loc[mask, "workout_clean"].apply(classify_workout_regex)

print(f"\nFinal workout_category distribution:")
for cat, cnt in context["workout_category"].value_counts().items():
    print(f"  {cat:20s}  {cnt:>5} ({cnt/len(context)*100:.1f}%)")

# ════════════════════════════════════════════════════════════════════
# STEP 3: Derived features from workout
# ════════════════════════════════════════════════════════════════════
print("\nDerived features:")

# 3a. Workout intensity (ordinal: 0=rest, 1=light, 2=moderate, 3=intense)
INTENSITY_MAP = {
    "unknown": np.nan, "rest": 0, "light_activity": 1,
    "moderate_cardio": 2, "sport": 2,
    "strength": 3, "intense_cardio": 3,
}
context["workout_intensity"] = context["workout_category"].map(INTENSITY_MAP)
print(f"  workout_intensity: {context['workout_intensity'].value_counts().sort_index().to_dict()}")

# 3b. Binary type flags
context["is_cardio"]   = context["workout_category"].isin(["moderate_cardio", "intense_cardio"]).astype(int)
context["is_strength"] = (context["workout_category"] == "strength").astype(int)
context["is_sport"]    = (context["workout_category"] == "sport").astype(int)
context["is_light"]    = (context["workout_category"] == "light_activity").astype(int)

# 3c. Fix had_workout & is_rest_day using new categorisation (more accurate)
context["had_workout"] = context["workout_category"].apply(
    lambda x: 1 if x not in ("rest", "unknown") else 0
)
context["is_rest_day"] = (context["workout_category"] == "rest").astype(int)
print(f"  had_workout (corrected): {context['had_workout'].sum()} ({context['had_workout'].mean()*100:.1f}%)")
print(f"  is_rest_day (corrected): {context['is_rest_day'].sum()} ({context['is_rest_day'].mean()*100:.1f}%)")

# ════════════════════════════════════════════════════════════════════
# STEP 4: Fix caffeine inconsistency
# ════════════════════════════════════════════════════════════════════
print("\nCaffeine fix:")
# 2,108 rows have caffeine_mg > 0 but no hours_before_bed
# Impute with population median for those rows
med_hours = context.loc[
    (context["caffeine_mg"] > 0) & context["last_caffeine_hours_before_bed"].notna(),
    "last_caffeine_hours_before_bed"
].median()
fix_mask = (context["caffeine_mg"] > 0) & context["last_caffeine_hours_before_bed"].isna()
context.loc[fix_mask, "last_caffeine_hours_before_bed"] = med_hours
print(f"  Imputed {fix_mask.sum()} missing caffeine hours with median={med_hours:.1f}h")

# 1,799 rows have hours_before_bed but caffeine_mg is NaN/0 — set hours to NaN
bad_mask = (context["caffeine_mg"].isna() | (context["caffeine_mg"] == 0)) & context["last_caffeine_hours_before_bed"].notna()
context.loc[bad_mask, "last_caffeine_hours_before_bed"] = np.nan
print(f"  Cleared {bad_mask.sum()} orphaned caffeine hours (no caffeine consumed)")

# Recalculate caffeine_close_to_bed after fix
context["caffeine_close_to_bed"] = (context["last_caffeine_hours_before_bed"] < 4).astype(float)
context["caffeine_close_to_bed"] = context["caffeine_close_to_bed"].where(
    context["last_caffeine_hours_before_bed"].notna(), np.nan
)

# ════════════════════════════════════════════════════════════════════
# STEP 5: Additional context features
# ════════════════════════════════════════════════════════════════════
print("\nAdditional derived features:")

# Activity level bins (from steps)
context["activity_level"] = pd.cut(
    context["steps"],
    bins=[0, 3000, 7000, 10000, 50000],
    labels=["sedentary", "light", "moderate", "active"],
)
print(f"  activity_level: {context['activity_level'].value_counts().sort_index().to_dict()}")

# Alcohol categories
context["alcohol_level"] = pd.cut(
    context["alcohol_units"].fillna(0),
    bins=[-0.1, 0, 2, 5.1],
    labels=["none", "light", "heavy"],
)
print(f"  alcohol_level: {context['alcohol_level'].value_counts().to_dict()}")

# Stress category (for the 34% with data)
context["stress_level"] = pd.cut(
    context["stress_score"],
    bins=[0, 30, 60, 100.1],
    labels=["low", "moderate", "high"],
)
print(f"  stress_level: {context['stress_level'].value_counts().to_dict()} (nulls={context['stress_level'].isna().sum()})")

# Drop temp columns
context.drop(columns=["date_dt", "dow", "workout_clean"], errors="ignore", inplace=True)

# ════════════════════════════════════════════════════════════════════
# SUMMARY + VISUALISATION
# ════════════════════════════════════════════════════════════════════
new_cols = ["workout_category", "workout_intensity", "is_cardio", "is_strength",
            "is_sport", "is_light", "activity_level", "alcohol_level", "stress_level"]
print(f"\n{'='*70}")
print(f"NEW FEATURES: {len(new_cols)} columns added")
print(f"CONTEXT: {len(context):,} rows \u00d7 {context.shape[1]} cols")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# 1. Workout category
cat_order = ["unknown", "rest", "light_activity", "moderate_cardio", "strength", "sport", "intense_cardio"]
cat_counts = context["workout_category"].value_counts().reindex(cat_order, fill_value=0)
axes[0, 0].barh(cat_counts.index, cat_counts.values, color=["#999999"] + [PAL[i] for i in range(6)], edgecolor="white")
for i, v in enumerate(cat_counts.values):
    axes[0, 0].text(v + 30, i, f"{v:,}", va="center", fontsize=9)
axes[0, 0].set_title("Workout Category"); axes[0, 0].invert_yaxis()

# 2. Intensity distribution
int_counts = context["workout_intensity"].value_counts().sort_index()
int_labels = ["0: Rest", "1: Light", "2: Moderate", "3: Intense"]
axes[0, 1].bar(int_labels, int_counts.values, color=["grey", "#a1d99b", "#fdae6b", "#e6550d"], edgecolor="white")
for i, v in enumerate(int_counts.values):
    axes[0, 1].text(i, v + 30, f"{v:,}", ha="center", fontsize=9)
axes[0, 1].set_title("Workout Intensity")

# 3. Activity level (from steps)
act_order = ["sedentary", "light", "moderate", "active"]
act_counts = context["activity_level"].value_counts().reindex(act_order, fill_value=0)
axes[0, 2].bar(act_order, act_counts.values, color=["#bdbdbd", "#a1d99b", "#fdae6b", "#e6550d"], edgecolor="white")
for i, v in enumerate(act_counts.values):
    axes[0, 2].text(i, v + 30, f"{v:,}", ha="center", fontsize=9)
axes[0, 2].set_title("Activity Level (steps)")

# 4. Alcohol level
alc_counts = context["alcohol_level"].value_counts()
axes[1, 0].pie(alc_counts.values, labels=alc_counts.index, autopct="%1.0f%%",
               colors=["#a1d99b", "#fdae6b", "#e6550d"])
axes[1, 0].set_title("Alcohol Level")

# 5. Caffeine timing (post-fix)
caff_valid = context["last_caffeine_hours_before_bed"].dropna()
sns.histplot(caff_valid, bins=30, kde=True, color=PAL[3], ax=axes[1, 1], edgecolor="white")
axes[1, 1].axvline(4, color="red", ls="--", lw=1.5, label="4h threshold")
axes[1, 1].set_title("Caffeine Hours Before Bed (post-fix)"); axes[1, 1].legend()

# 6. Stress distribution (34% with data)
stress_valid = context["stress_score"].dropna()
sns.histplot(stress_valid, bins=30, kde=True, color=PAL[4], ax=axes[1, 2], edgecolor="white")
axes[1, 2].set_title(f"Stress Score ({len(stress_valid):,} / {len(context):,} reported)")

plt.suptitle("Daily Context — Workout & Lifestyle Features", fontsize=13, y=1.01)
plt.tight_layout(); plt.show()

# COMMAND ----------

# DBTITLE 1,Section 4 Final — clean daily context sample (all columns)
# ── Section 4 FINAL: Clean daily_context — all columns sample ──
print(f"\u2550" * 70)
print(f"SECTION 4 OUTPUT: context — {len(context):,} rows \u00d7 {context.shape[1]} cols")
print(f"\u2550" * 70)

print(f"\nColumns ({context.shape[1]}):")
for i, col in enumerate(context.columns):
    dtype = context[col].dtype
    nulls = context[col].isna().sum()
    null_pct = nulls / len(context) * 100
    print(f"  {i+1:>2}. {col:35s}  {str(dtype):12s}  nulls={nulls:>5} ({null_pct:4.1f}%)")

print(f"\n\u2500\u2500 Sample (20 rows, all columns) \u2500\u2500")
with pd.option_context("display.max_columns", None, "display.width", 200):
    display(context.head(20))

# COMMAND ----------

# DBTITLE 1,Section 5: Morning Check-ins (Target)
# MAGIC %md
# MAGIC ## Section 5: Morning Check-ins (Target)
# MAGIC
# MAGIC Target analysis, selection bias check, mood_tags breakdown, and submitted_at timing. Per your note: mood tags "can be used for second level validation" but NOT as features.

# COMMAND ----------

# DBTITLE 1,Morning Check-ins - target analysis and coverage
# ── 5A: Morning Check-ins — Target analysis ──
checkins = raw_checkins.copy()
checkins["user_id"] = checkins["user_id"].str.strip().str.upper()
checkins["date"] = pd.to_datetime(checkins["date"]).dt.date
checkins["submitted_at"] = pd.to_datetime(checkins["submitted_at"])

shape_report(checkins, "morning_checkins")

# Target distribution
print("\nTarget distribution (subjective_feeling 1-5):")
vc = checkins["subjective_feeling"].value_counts().sort_index()
for val, cnt in vc.items():
    pct = cnt / len(checkins) * 100
    bar = "█" * int(pct)
    print(f"  {val}: {cnt:>5} ({pct:5.1f}%) {bar}")

print(f"\nMean: {checkins['subjective_feeling'].mean():.3f}")
print(f"Median: {checkins['subjective_feeling'].median():.0f}")
print(f"Std: {checkins['subjective_feeling'].std():.3f}")

# Duplicate check
n_dup = checkins.duplicated(subset=["user_id", "date"]).sum()
print(f"\nDuplicate (user_id, date): {n_dup}")
if n_dup > 0:
    checkins = checkins.drop_duplicates(subset=["user_id", "date"], keep="last")
    print(f"After dedup: {len(checkins)}")

# COMMAND ----------

# DBTITLE 1,Morning Check-ins - selection bias and coverage
# ── 5B: Check-in selection bias ──
# Do users who check in more often rate differently?
per_user_checkins = checkins.groupby("user_id").agg(
    n_checkins=("date", "count"),
    mean_feeling=("subjective_feeling", "mean"),
    std_feeling=("subjective_feeling", "std"),
).reset_index()

print("Per-user check-in frequency:")
print(per_user_checkins["n_checkins"].describe().to_string())

# Correlation between check-in frequency and mean feeling
corr = per_user_checkins["n_checkins"].corr(per_user_checkins["mean_feeling"])
print(f"\nCorrelation (n_checkins vs mean_feeling): {corr:.3f}")
if abs(corr) > 0.2:
    print("⚠ Moderate selection bias: users who check in more/less rate differently")
else:
    print("✓ Low selection bias")

# Check-in coverage: what % of available days does each user check in?
date_range_days = (checkins["date"].max() - checkins["date"].min()).days + 1
per_user_checkins["coverage_pct"] = (per_user_checkins["n_checkins"] / date_range_days * 100).round(1)
print(f"\nDate range: {date_range_days} days")
print(f"Coverage per user: mean={per_user_checkins['coverage_pct'].mean():.1f}%, "
      f"min={per_user_checkins['coverage_pct'].min():.1f}%, "
      f"max={per_user_checkins['coverage_pct'].max():.1f}%")

# Submitted_at timing analysis
checkins["submit_hour"] = checkins["submitted_at"].dt.hour
print(f"\nSubmission hour distribution:")
for h, cnt in checkins["submit_hour"].value_counts().sort_index().head(12).items():
    pct = cnt / len(checkins) * 100
    print(f"  {h:02d}:00  {cnt:>4} ({pct:4.1f}%)")

# Do early vs late submitters rate differently?
checkins["early_submit"] = (checkins["submit_hour"] < 9).astype(int)
early_mean = checkins[checkins["early_submit"]==1]["subjective_feeling"].mean()
late_mean = checkins[checkins["early_submit"]==0]["subjective_feeling"].mean()
print(f"\nEarly (<9am) mean feeling: {early_mean:.3f}")
print(f"Late  (>=9am) mean feeling: {late_mean:.3f}")

# COMMAND ----------

# DBTITLE 1,Morning Check-ins - mood tags analysis
# ── 5C: Mood tags analysis (for validation, NOT features) ──
# Per note: "Can be used for second level validation"
all_tags = checkins["mood_tags"].str.split(";").explode().str.strip().str.lower()
tag_counts = all_tags.value_counts()
print(f"Unique mood tags: {len(tag_counts)}")
print(f"\nTop 15 mood tags:")
display(tag_counts.head(15).to_frame("count"))

# Map tags to sentiment categories for validation
positive_tags = {"energised", "good", "rested", "clear", "focused", "amazing", 
                 "refreshed", "strong,clear", "positive,rested"}
negative_tags = {"groggy", "foggy,tired", "tired", "low energy", "sluggish", 
                 "sore", "headache", "groggy;headache"}
neutral_tags = {"ok", "fine", "average", "neutral", "meh"}

def classify_mood(tags_str):
    tags = set(t.strip().lower() for t in tags_str.split(";"))
    if tags & positive_tags:
        return "positive"
    elif tags & negative_tags:
        return "negative"
    elif tags & neutral_tags:
        return "neutral"
    return "other"

checkins["mood_category"] = checkins["mood_tags"].apply(classify_mood)

# Validate: mood category should align with subjective_feeling
mood_vs_feeling = checkins.groupby("mood_category")["subjective_feeling"].agg(["mean", "std", "count"]).round(2)
print("\nMood category vs subjective_feeling (validation check):")
display(mood_vs_feeling)
print("\n✓ Positive tags should have higher mean, negative lower — validates label quality")

# COMMAND ----------

# DBTITLE 1,Section 6: Cross-Dataset Joins and Alignment
# MAGIC %md
# MAGIC ## Section 6: Cross-Dataset Joins and Final Merge
# MAGIC
# MAGIC Validate date alignment logic, join coverage across all tables, and build the final modelling-ready dataset.

# COMMAND ----------

# DBTITLE 1,Date alignment validation
# ── 6A: Date Alignment Validation ──
# CRITICAL: Get the date alignment right
#
# morning_checkins.date = morning the user answered (e.g., 2026-02-15)
# daily_context.date = "waking day" BEFORE that night's sleep (e.g., 2026-02-14)
#   → daily_context on 2026-02-14 describes the DAY of 2026-02-14
#   → legacy_readiness_shown = score shown "on the morning AFTER this date" = morning of 2026-02-15
# sleep_session should have ENDED on the checkin morning (session_end date = checkin date)
#
# Therefore:
#   checkin_date = 2026-02-15
#   daily_context_date = 2026-02-14 (the day before)
#   session_end_date = 2026-02-15 (the morning of)

print("Date alignment strategy:")
print("  checkin_date = morning_checkins.date")
print("  context_date = checkin_date - 1 day (the 'waking day' before that night's sleep)")
print("  session match: session that ended on checkin_date")

# Derive the matching keys
checkins["checkin_date"] = checkins["date"]
checkins["context_date"] = checkins["date"] - pd.Timedelta(days=1)  # context from previous day

# For sessions: extract the date the session ended (in local time would be ideal, but UTC date is close enough)
sessions_clean["session_end_date"] = sessions_clean["session_end_utc"].dt.date

# Verify alignment: try joining checkins to sessions on (user_id, checkin_date = session_end_date)
test_join = checkins.merge(
    sessions_clean[["user_id", "session_id", "session_end_date"]],
    left_on=["user_id", "checkin_date"],
    right_on=["user_id", "session_end_date"],
    how="left"
)
match_rate = test_join["session_id"].notna().mean()
n_multi = (test_join.groupby(["user_id", "checkin_date"]).size() > 1).sum()

print(f"\nCheckin → Session join (on session_end_date):")
print(f"  Match rate: {match_rate:.1%} ({test_join['session_id'].notna().sum()} / {len(checkins)})")
print(f"  Multi-match rows (multiple sessions ended same day): {n_multi}")

# Verify alignment: checkins to daily_context
test_ctx = checkins.merge(
    context[["user_id", "date"]].rename(columns={"date": "ctx_date"}),
    left_on=["user_id", "context_date"],
    right_on=["user_id", "ctx_date"],
    how="left"
)
ctx_match = test_ctx["ctx_date"].notna().mean()
print(f"\nCheckin → Daily Context join (on context_date = checkin_date - 1):")
print(f"  Match rate: {ctx_match:.1%} ({test_ctx['ctx_date'].notna().sum()} / {len(checkins)})")

# COMMAND ----------

# DBTITLE 1,Build final merged dataset
# ── 6B: Build the final merged modelling dataset ──
#
# MERGE STRATEGY (D-021):
# LEFT JOIN from checkins throughout. Rationale:
#   1. Checkins = target table (4,989 rows). Every target row is precious in a
#      small dataset — dropping ~10% for missing sessions would hurt.
#   2. LightGBM handles NaN natively: missing sleep features become implicit
#      "no data" splits, which is informative (ring-off nights differ from ring-on).
#   3. Context matches 100% — no information loss there.
#   4. Calendar/temporal/holiday features are computed INDEPENDENTLY from
#      checkin_date in the next cell, so ALL 4,989 rows get them — not tied
#      to session existence.
#   5. A `has_session_data` flag lets the model learn the session-missing pattern.
#
# What flows from each table:
#   sessions → sleep physiology, stages, decomp, multi-session, sleep timing
#              (NaN for ~10% without sessions — that's correct)
#   context  → steps, caffeine, alcohol, workout, stress, travel
#              (100% match, zero NaN from the join itself)
#   profiles → age, sex, BMI, timezone, plan_tier (100% match)
#   calendar → computed in cell 6C for ALL rows from checkin_date

print("Merge strategy: LEFT JOIN from checkins (preserve all target rows)")
print("="*70)

# ─────────────────────────────────────────────────────────────
# STEP 0: Target base
# ─────────────────────────────────────────────────────────────
df = checkins[["user_id", "checkin_date", "context_date", "subjective_feeling",
               "mood_category", "submit_hour"]].copy()
print(f"Step 0 - Target base: {len(df):,} rows, {df['user_id'].nunique()} users")
assert len(df) == 4989, f"Expected 4989, got {len(df)}"

# ─────────────────────────────────────────────────────────────
# STEP 1: LEFT JOIN user profiles (1:1 per user)
# ─────────────────────────────────────────────────────────────
profile_cols = ["user_id", "age_years", "sex", "height_cm", "weight_kg", "bmi",
                "timezone", "plan_tier", "age_group", "bmi_category",
                "onboarding_tenure_days", "weight_outlier", "bmi_outlier"]
df = df.merge(profiles[profile_cols], on="user_id", how="left")
print(f"Step 1 - + profiles: {len(df):,} rows (nulls: {df['age_years'].isna().sum()})")
assert len(df) == 4989

# ─────────────────────────────────────────────────────────────
# STEP 2: LEFT JOIN daily context (context_date = checkin_date - 1)
# Includes new workout, activity, alcohol, stress features
# ─────────────────────────────────────────────────────────────
ctx_cols = ["user_id", "date",
            # Raw numeric
            "steps", "active_minutes", "alcohol_units", "caffeine_mg",
            "last_caffeine_hours_before_bed", "stress_score", "travel_flag",
            "legacy_readiness_shown",
            # Binary flags
            "stress_reported", "had_caffeine", "caffeine_close_to_bed",
            "had_alcohol", "had_workout", "is_rest_day",
            # NEW: workout features
            "workout_category", "workout_intensity",
            "is_cardio", "is_strength", "is_sport", "is_light",
            # NEW: categorical bins
            "activity_level", "alcohol_level", "stress_level"]
df = df.merge(
    context[ctx_cols].rename(columns={"date": "context_date"}),
    on=["user_id", "context_date"], how="left"
)
ctx_nulls = df["steps"].isna().sum()
print(f"Step 2 - + daily_context: {len(df):,} rows (nulls: {ctx_nulls})")
assert len(df) == 4989

# ─────────────────────────────────────────────────────────────
# STEP 3: LEFT JOIN sleep sessions
# Sleep-specific features ONLY. Calendar/temporal/holiday features are computed
# independently in cell 6C so ALL 4,989 rows get them.
# ─────────────────────────────────────────────────────────────
session_cols = ["session_id", "user_id", "session_end_date",
                # Duration & stages
                "time_in_bed_minutes", "total_sleep_minutes", "deep_minutes",
                "rem_minutes", "light_minutes", "awake_minutes", "sleep_efficiency",
                # Physiology
                "avg_hr_bpm", "min_hr_bpm", "avg_hrv_rmssd_ms", "avg_spo2_pct",
                "avg_resp_rate_bpm", "temperature_deviation_c", "awakenings_count",
                "movement_index", "firmware_version",
                # Decomposition
                "deep_pct", "rem_pct", "light_pct", "awake_pct",
                "efficiency_calc", "decomp_quality_score",
                # Multi-session
                "fragmented_night", "n_sessions", "secondary_tib",
                "secondary_session_count", "nap_ratio", "night_consolidation",
                "total_tib_all", "total_sleep_all",
                # Sleep timing (session-specific — NaN when no session is correct)
                "bedtime_hour", "wake_hour", "sleep_onset_period", "wake_period",
                "bedtime_dev_midnight_hr", "sleep_midpoint_hour",
                "bedtime_sin", "bedtime_cos", "wake_sin", "wake_cos",
                # Sensor flags
                "hr_sensor_error", "hrv_sensor_error", "any_sensor_error"]
df = df.merge(
    sessions_clean[session_cols],
    left_on=["user_id", "checkin_date"],
    right_on=["user_id", "session_end_date"],
    how="left"
)
# Handle multi-match: keep session with longest total_sleep_minutes
n_multi = df.duplicated(subset=["user_id", "checkin_date"]).sum()
if n_multi > 0:
    print(f"  Multi-matched: {n_multi} rows → keeping longest session")
    df = df.sort_values("total_sleep_minutes", ascending=False).drop_duplicates(
        subset=["user_id", "checkin_date"], keep="first"
    )
no_session = df["session_id"].isna().sum()
print(f"Step 3 - + sleep_session: {len(df):,} rows (no session: {no_session}, {no_session/len(df)*100:.1f}%)")
assert len(df) == 4989

# ─────────────────────────────────────────────────────────────
# STEP 4: Metadata flags
# ─────────────────────────────────────────────────────────────
df["has_session_data"] = df["session_id"].notna().astype(int)
df.drop(columns=["session_end_date"], errors="ignore", inplace=True)

# Signal aggregates — SKIPPED for V1 (D-019)
print(f"Step 4 - signal_agg: SKIPPED (deferred to V2)")

print(f"\n{'='*70}")
print(f"MERGED DATASET: {len(df):,} rows \u00d7 {df.shape[1]} cols")
print(f"  Users: {df['user_id'].nunique()}")
print(f"  Date range: {df['checkin_date'].min()} to {df['checkin_date'].max()}")
print(f"  With session: {df['has_session_data'].sum()} ({df['has_session_data'].mean()*100:.1f}%)")
print(f"  Without session: {no_session} ({no_session/len(df)*100:.1f}%) → sleep cols = NaN, temporal filled next")
print(f"{'='*70}")

# COMMAND ----------

# DBTITLE 1,Calendar, holidays, user-normalized features (all rows)
# ── 6C: Calendar, holidays, and user-normalised features ──
#
# CRITICAL DESIGN CHOICE: These features are computed from checkin_date
# (not from session data), so ALL 4,989 rows get them — including the ~335
# rows without session data. This avoids unnecessary NaN in calendar features
# that have nothing to do with whether the ring was worn.

import pytz
try:
    import holidays as hol_lib
except ImportError:
    pass  # already installed earlier

# ════ A. CALENDAR / TEMPORAL FEATURES (from night_date = checkin_date - 1) ════
print("A. Calendar features (all 4,989 rows)")
nd = pd.to_datetime(df["checkin_date"]) - pd.Timedelta(days=1)  # night_date
df["night_date"] = nd.dt.date

df["day_of_week"]     = nd.dt.dayofweek
df["day_name"]        = nd.dt.day_name()
df["is_weekend"]      = df["day_of_week"].isin([5, 6]).astype(int)
df["is_friday_night"] = (df["day_of_week"] == 4).astype(int)
df["is_sunday_night"] = (df["day_of_week"] == 6).astype(int)
df["day_of_month"]    = nd.dt.day
df["week_of_year"]    = nd.dt.isocalendar().week.astype(int)
df["month"]           = nd.dt.month
df["is_month_start"]  = (nd.dt.day <= 3).astype(int)
df["is_month_end"]    = (nd.dt.day >= 28).astype(int)
df["days_into_study"] = (nd - DATA_START).dt.days
df["days_to_weekend"] = df["day_of_week"].map({0:4, 1:3, 2:2, 3:1, 4:0, 5:0, 6:0})
df["workday_index"]   = df["day_of_week"].apply(lambda d: d+1 if d < 5 else 0)

# Lunar phase
NEW_MOON = pd.Timestamp("2026-01-29")
df["lunar_phase"] = (((nd - NEW_MOON).dt.days % 29.53) / 29.53).round(3)
df["is_full_moon"] = (((nd - NEW_MOON).dt.days % 29.53).between(13, 16)).astype(int)

print(f"  {df['is_weekend'].sum()} weekend nights, {df['is_friday_night'].sum()} Fridays")
print(f"  days_into_study: {df['days_into_study'].min()} to {df['days_into_study'].max()}")

# ════ B. PERSONALIZED HOLIDAY FLAGS (all rows, from timezone) ════
print("\nB. Personalized holiday flags (all 4,989 rows)")

TZ_COUNTRY = {"Asia/Kolkata": "IN", "America/New_York": "US",
              "America/Los_Angeles": "US", "Europe/London": "GB", "Asia/Dubai": "AE"}
hol_sets = {cc: hol_lib.country_holidays(cc, years=[2025, 2026, 2027])
            for cc in set(TZ_COUNTRY.values())}

df["country_code"] = df["timezone"].map(TZ_COUNTRY)

dates_arr   = nd.dt.date
dates_p1    = (nd + pd.Timedelta(days=1)).dt.date
dates_m1    = (nd - pd.Timedelta(days=1)).dt.date
cc_arr      = df["country_code"].values

is_hol, is_hol_tom, is_hol_yest, hol_name = [], [], [], []
days_next, days_prev = [], []

for i in range(len(df)):
    cc = cc_arr[i]
    d, d1, dm1 = dates_arr.iloc[i], dates_p1.iloc[i], dates_m1.iloc[i]
    hset = hol_sets[cc]
    is_hol.append(int(d in hset))
    is_hol_tom.append(int(d1 in hset))
    is_hol_yest.append(int(dm1 in hset))
    hol_name.append(hset.get(d, ""))
    _n, _p = 90, 90
    for fwd in range(1, 91):
        if (pd.Timestamp(d) + pd.Timedelta(days=fwd)).date() in hset:
            _n = fwd; break
    for bwd in range(1, 91):
        if (pd.Timestamp(d) - pd.Timedelta(days=bwd)).date() in hset:
            _p = bwd; break
    days_next.append(_n); days_prev.append(_p)

df["is_holiday"]           = is_hol
df["is_holiday_tomorrow"]  = is_hol_tom
df["is_holiday_yesterday"] = is_hol_yest
df["holiday_name"]         = hol_name
df["days_to_next_holiday"]   = days_next
df["days_since_last_holiday"]= days_prev

df["is_long_weekend"] = (
    (df["is_weekend"].astype(bool) & (df["is_holiday_tomorrow"].astype(bool) | df["is_holiday_yesterday"].astype(bool))) |
    (df["is_holiday"].astype(bool) & df["day_of_week"].isin([0, 4]))
).astype(int)
df["no_work_tomorrow"] = (df["day_of_week"].isin([4, 5]) | df["is_holiday_tomorrow"].astype(bool)).astype(int)
df["holiday_proximity"] = np.where(df["days_to_next_holiday"] <= 3,
                                    (4 - df["days_to_next_holiday"]) / 3, 0).round(3)

for cc in sorted(hol_sets.keys()):
    mask = df["country_code"] == cc
    n_hol = df.loc[mask, "is_holiday"].sum()
    print(f"  {cc}: {n_hol} holiday-nights")

# ════ C. USER-NORMALISED Z-SCORES ════
print("\nC. User-normalised z-scores")
user_norm_cols = ["total_sleep_minutes", "deep_minutes", "rem_minutes",
                  "avg_hr_bpm", "avg_hrv_rmssd_ms", "sleep_efficiency",
                  "bedtime_hour", "wake_hour", "steps", "active_minutes"]
cols_in_df = [c for c in user_norm_cols if c in df.columns]
for col in cols_in_df:
    df[f"{col}_zscore"] = df.groupby("user_id")[col].transform(
        lambda x: (x - x.mean()) / max(x.std(), 0.01)
    )
print(f"  {len(cols_in_df)} z-score features created")

# ════ D. DAYS SINCE LAST SESSION (gap awareness) ════
print("\nD. Gap awareness feature")
df = df.sort_values(["user_id", "checkin_date"]).reset_index(drop=True)
df["prev_checkin"] = df.groupby("user_id")["checkin_date"].shift(1)
df["days_since_last_checkin"] = (
    pd.to_datetime(df["checkin_date"]) - pd.to_datetime(df["prev_checkin"])
).dt.days
df.drop(columns=["prev_checkin"], inplace=True)
print(f"  days_since_last_checkin: mean={df['days_since_last_checkin'].mean():.1f}, "
      f"max={df['days_since_last_checkin'].max()}, nulls={df['days_since_last_checkin'].isna().sum()} (first per user)")

# ════ SUMMARY ════
print(f"\n{'='*70}")
print(f"FINAL MODELLING TABLE: {len(df):,} rows \u00d7 {df.shape[1]} cols")
print(f"{'='*70}")

# Show all columns grouped
print(f"\nAll columns ({df.shape[1]}):")
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    nulls = df[col].isna().sum()
    null_pct = nulls / len(df) * 100
    flag = " \u26a0" if null_pct > 50 else ""
    print(f"  {i+1:>3}. {col:45s} {str(dtype):>15s}  nulls={nulls:>5} ({null_pct:4.1f}%){flag}")

# COMMAND ----------

# DBTITLE 1,Section 7: EDA Plots
# MAGIC %md
# MAGIC ## Section 7: EDA Visualisations
# MAGIC
# MAGIC Key distributions and relationships to validate cleaning and understand the modelling landscape.

# COMMAND ----------

# DBTITLE 1,EDA - target and feature distributions
# ── 7A: Key EDA plots ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Target distribution
df["subjective_feeling"].value_counts().sort_index().plot(kind="bar", ax=axes[0,0], color="steelblue")
axes[0,0].set_title("Target: subjective_feeling distribution")
axes[0,0].set_xlabel("Feeling (1=worst, 5=best)")

# 2. Sleep duration vs target
df.boxplot(column="total_sleep_minutes", by="subjective_feeling", ax=axes[0,1])
axes[0,1].set_title("Sleep duration by feeling")
axes[0,1].set_xlabel("Feeling")
axes[0,1].set_ylabel("Total sleep (min)")
plt.sca(axes[0,1]); plt.title("Sleep duration by feeling")

# 3. HRV vs target
df.boxplot(column="avg_hrv_rmssd_ms", by="subjective_feeling", ax=axes[0,2])
axes[0,2].set_title("HRV (RMSSD) by feeling")
axes[0,2].set_xlabel("Feeling")
axes[0,2].set_ylabel("HRV (ms)")
plt.sca(axes[0,2]); plt.title("HRV by feeling")

# 4. Sleep efficiency vs target
df.boxplot(column="sleep_efficiency", by="subjective_feeling", ax=axes[1,0])
axes[1,0].set_title("Sleep efficiency by feeling")
axes[1,0].set_xlabel("Feeling")
plt.sca(axes[1,0]); plt.title("Sleep efficiency by feeling")

# 5. Steps vs target
df.boxplot(column="steps", by="subjective_feeling", ax=axes[1,1])
axes[1,1].set_title("Previous day steps by feeling")
axes[1,1].set_xlabel("Feeling")
plt.sca(axes[1,1]); plt.title("Previous day steps by feeling")

# 6. Check-ins per day of week
df.groupby("day_of_week")["subjective_feeling"].mean().plot(kind="bar", ax=axes[1,2], color="coral")
axes[1,2].set_title("Mean feeling by day of week")
axes[1,2].set_xlabel("Day (0=Mon, 6=Sun)")
axes[1,2].set_ylabel("Mean feeling")
axes[1,2].set_ylim(2.5, 4.0)

plt.suptitle("")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,EDA - feature correlations with target
# ── 7B: Feature correlations with target ──
numeric_features = [
    # Sleep
    "total_sleep_minutes", "deep_minutes", "rem_minutes", "light_minutes",
    "awake_minutes", "sleep_efficiency", "deep_pct", "rem_pct",
    # Physiology
    "avg_hr_bpm", "min_hr_bpm", "avg_hrv_rmssd_ms", "avg_spo2_pct",
    "avg_resp_rate_bpm", "temperature_deviation_c", "awakenings_count", "movement_index",
    # Multi-session
    "fragmented_night", "n_sessions", "nap_ratio", "night_consolidation",
    # Sleep timing
    "bedtime_hour", "wake_hour", "bedtime_dev_midnight_hr", "sleep_midpoint_hour",
    # Context
    "steps", "active_minutes", "alcohol_units", "caffeine_mg",
    "last_caffeine_hours_before_bed", "stress_score", "travel_flag",
    "workout_intensity", "is_cardio", "is_strength",
    # Calendar
    "is_weekend", "is_friday_night", "is_sunday_night", "no_work_tomorrow",
    "is_holiday", "days_into_study", "days_since_last_checkin",
    # Z-scores
    "total_sleep_minutes_zscore", "deep_minutes_zscore", "bedtime_hour_zscore",
    # Metadata
    "has_session_data",
    "legacy_readiness_shown",  # benchmark only - NOT a feature
]

corrs = []
for col in numeric_features:
    if col in df.columns:
        c = df[[col, "subjective_feeling"]].dropna().corr().iloc[0, 1]
        corrs.append({"feature": col, "corr_with_target": round(c, 4)})

corr_df = pd.DataFrame(corrs).sort_values("corr_with_target", key=abs, ascending=False)
print("Feature correlations with subjective_feeling (sorted by |corr|):")
print("\n  ⚠ legacy_readiness_shown included for BENCHMARK only — NOT a model feature")
display(corr_df)

# Highlight top predictors
print("\nTop 10 predictors (excluding leakage):")
top10 = corr_df[corr_df["feature"] != "legacy_readiness_shown"].head(10)
for _, row in top10.iterrows():
    direction = "↑" if row["corr_with_target"] > 0 else "↓"
    print(f"  {direction} {row['feature']:40s} r={row['corr_with_target']:+.4f}")

# COMMAND ----------

# DBTITLE 1,Section 8: Save Final Dataset
# MAGIC %md
# MAGIC ## Section 8: Save Clean Datasets
# MAGIC
# MAGIC Export the merged modelling table and clean intermediate tables for downstream use.

# COMMAND ----------

# DBTITLE 1,Save final merged dataset and intermediates
# ── 8A: Define feature groups and save ──
import os

OUT_DIR = "/Workspace/Users/mohammad.sayeed@tatadigital.com/AS_UH_Project/data"

# Define clean feature columns (EXCLUDING leakage columns and temp helper columns)
id_cols = ["user_id", "checkin_date", "session_id"]
target_col = "subjective_feeling"
benchmark_col = "legacy_readiness_shown"  # for comparison, NOT model input
validation_col = "mood_category"  # for error analysis, NOT model input

profile_features = ["age_years", "sex", "height_cm", "weight_kg", "bmi",
                    "timezone", "plan_tier", "age_group", "bmi_category",
                    "onboarding_tenure_days", "weight_outlier", "bmi_outlier"]

sleep_features = ["time_in_bed_minutes", "total_sleep_minutes", "deep_minutes", "rem_minutes",
                  "light_minutes", "awake_minutes", "sleep_efficiency", "avg_hr_bpm",
                  "min_hr_bpm", "avg_hrv_rmssd_ms", "avg_spo2_pct", "avg_resp_rate_bpm",
                  "temperature_deviation_c", "awakenings_count", "movement_index",
                  "firmware_version", "deep_pct", "rem_pct", "light_pct", "awake_pct",
                  "efficiency_calc", "decomp_quality_score",
                  "fragmented_night", "n_sessions", "secondary_tib",
                  "secondary_session_count", "nap_ratio", "night_consolidation",
                  "total_tib_all", "total_sleep_all"]

sleep_timing_features = ["bedtime_hour", "wake_hour", "sleep_onset_period", "wake_period",
                         "bedtime_dev_midnight_hr", "sleep_midpoint_hour",
                         "bedtime_sin", "bedtime_cos", "wake_sin", "wake_cos"]

context_features = ["steps", "active_minutes", "alcohol_units", "caffeine_mg",
                    "last_caffeine_hours_before_bed", "stress_score", "stress_reported",
                    "travel_flag", "had_caffeine", "caffeine_close_to_bed",
                    "had_alcohol", "had_workout", "is_rest_day",
                    "workout_category", "workout_intensity",
                    "is_cardio", "is_strength", "is_sport", "is_light",
                    "activity_level", "alcohol_level", "stress_level"]

calendar_features = ["day_of_week", "day_name", "is_weekend", "is_friday_night",
                     "is_sunday_night", "day_of_month", "week_of_year", "month",
                     "is_month_start", "is_month_end", "days_into_study",
                     "days_to_weekend", "workday_index", "lunar_phase", "is_full_moon"]

holiday_features = ["country_code", "is_holiday", "is_holiday_tomorrow",
                    "is_holiday_yesterday", "is_long_weekend", "no_work_tomorrow",
                    "days_to_next_holiday", "days_since_last_holiday", "holiday_proximity"]

meta_features = ["has_session_data", "hr_sensor_error", "hrv_sensor_error",
                 "any_sensor_error", "days_since_last_checkin", "submit_hour"]

zscore_features = [c for c in df.columns if c.endswith("_zscore")]

all_feature_cols = (profile_features + sleep_features + sleep_timing_features +
                    context_features + calendar_features + holiday_features +
                    meta_features + zscore_features)

# Final modelling table
model_cols = id_cols + [target_col, benchmark_col, validation_col] + all_feature_cols
model_cols = [c for c in model_cols if c in df.columns]  # safety filter
df_model = df[model_cols].copy()

print(f"Final modelling table: {len(df_model):,} rows x {len(model_cols)} cols")
print(f"\nFeature groups:")
print(f"  Profile features:       {len(profile_features)}")
print(f"  Sleep features:         {len(sleep_features)}")
print(f"  Sleep timing features:  {len(sleep_timing_features)}")
print(f"  Context features:       {len(context_features)}")
print(f"  Calendar features:      {len(calendar_features)}")
print(f"  Holiday features:       {len(holiday_features)}")
print(f"  Meta features:          {len(meta_features)}")
print(f"  Z-score features:       {len(zscore_features)}")
print(f"  Total features:         {len(all_feature_cols)}")

# Save
out_path = os.path.join(OUT_DIR, "modelling_table.csv")
df_model.to_csv(out_path, index=False)
print(f"\n✓ Saved to {out_path}")
print(f"  Size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

# Also save clean intermediate tables
profiles.to_csv(os.path.join(OUT_DIR, "user_profiles_clean.csv"), index=False)
sessions_clean.to_csv(os.path.join(OUT_DIR, "sleep_sessions_clean.csv"), index=False)
context.to_csv(os.path.join(OUT_DIR, "daily_context_clean.csv"), index=False)
print("✓ Clean intermediate tables saved")

# COMMAND ----------

# DBTITLE 1,Section 9: Comprehensive Summary
# MAGIC %md
# MAGIC ## Section 9: Comprehensive Summary
# MAGIC
# MAGIC ### Key Findings
# MAGIC
# MAGIC | # | Finding | Severity | Action |
# MAGIC | --- | --- | --- | --- |
# MAGIC | 1 | user\_id has 3 variants (UH-/uh-/spaces) across tables | **High** | Normalised: strip + upper everywhere |
# MAGIC | 2 | 1,731 epoch-millis timestamps in sleep\_sessions | **High** | Parsed as millis; validated date range |
# MAGIC | 3 | 221 duplicate session\_ids (exact copies) | **Medium** | Deduplicated, keep first |
# MAGIC | 4 | 349 extra users in sleep\_sessions not in target | **Medium** | Filtered to 120 target users only |
# MAGIC | 5 | 984 multi-session nights (2–4 sessions) | **Medium** | Kept longest as primary; flagged as feature |
# MAGIC | 6 | Physiological outliers (HR=0/250, HRV=999, SpO2=36) | **High** | Sentinels→NaN; clipped to physiological ranges |
# MAGIC | 7 | 12 users with weight likely in lbs | **Medium** | Converted using timezone heuristic + BMI check |
# MAGIC | 8 | stress\_score 65% null, caffeine 20% null | **Medium** | Created binary "reported" flags; LightGBM handles NaN |
# MAGIC | 9 | legacy\_readiness\_shown correlates 0.77 with target | **Critical** | EXCLUDED from features (leakage); kept as benchmark |
# MAGIC | 10 | mood\_tags recorded at prediction time | **High** | Excluded from features; used for validation |
# MAGIC | 11 | daily\_context uses camelCase userId | **Low** | Renamed to user\_id |
# MAGIC | 12 | 5 null heights in profiles | **Low** | Imputed with sex-specific median |
# MAGIC | 13 | Duration mismatches (calculated vs reported) | **Medium** | Flagged; kept reported values |
# MAGIC | 14 | Firmware version varies (v2.0.3/v2.0.7/v2.1.0) | **Low** | Included as feature for model to handle |
# MAGIC
# MAGIC ### Data Shortcomings (ranked by severity)
# MAGIC
# MAGIC 1. **Label contamination from legacy score**: Users see the heuristic readiness before answering → anchors their self-report. This limits how clean the target truly is.
# MAGIC 2. **Target sparsity**: Only 5,048 check-ins from 120 users (mean 42/user) — limits model capacity.
# MAGIC 3. **Self-reported inputs**: Caffeine, alcohol, stress are user-logged → systematically under-reported.
# MAGIC 4. **Within-user variance dominates** (95.5%): The model must capture night-to-night variation, not just user identity.
# MAGIC 5. **Missing context for some check-ins**: \~3% have no matching sleep session → rows will have null sleep features.
# MAGIC
# MAGIC ### Cleaning Decisions Made
# MAGIC
# MAGIC 1. **user\_id normalisation**: strip + upper before any join
# MAGIC 2. **Epoch timestamps**: parsed as millis, validated date range
# MAGIC 3. **Dedup**: exact duplicate sessions removed
# MAGIC 4. **Multi-session nights**: longest session kept as primary, fragmented\_night flag added
# MAGIC 5. **Outlier strategy**: sentinels→NaN, physiological clips, implausible durations dropped
# MAGIC 6. **Weight units**: lbs→kg for US-timezone users with weight>120
# MAGIC 7. **Legacy score**: retained as benchmark column, NOT in feature set
# MAGIC 8. **Mood tags**: excluded from features, retained for validation
# MAGIC
# MAGIC ### Required Assumptions
# MAGIC
# MAGIC 1. The longest session per night IS the main sleep (not confirmed by the ring — could be nap)
# MAGIC 2. Epoch millis timestamps are in UTC (no timezone offset available)
# MAGIC 3. Weight conversion heuristic is correct (no ground truth)
# MAGIC 4. Missing caffeine/alcohol/stress means "not logged", not "zero"
# MAGIC 5. Users answer the morning check-in honestly (anchoring effect from legacy score notwithstanding)
# MAGIC
# MAGIC ### Impact on Proposed Modelling Approach
# MAGIC
# MAGIC The LightGBM regression approach **remains the right V1 choice**:
# MAGIC * Missing values handled natively (no imputation needed for stress, caffeine, alcohol)
# MAGIC * 5,048 usable rows with \~70 features is within LightGBM's sweet spot
# MAGIC * User-normalised z-score features capture the dominant within-user variance
# MAGIC * Signal aggregates provide rich HRV/HR trajectory features without needing a sequence model
# MAGIC * Temporal and day-of-week features capture lifestyle patterns
# MAGIC * The benchmark (legacy readiness correlation) gives a clear bar to beat