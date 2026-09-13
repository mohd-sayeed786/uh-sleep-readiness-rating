# Decision Log — Ring Readiness Score Project

## D-019 — Signal Aggregates Deferred to V2
[2026-09-12] DECISION: Do not merge nightly_signals into V1 training table. Save cleaned signal_agg (9,991 sessions × 26 cols) as signal_agg_clean.parquet for future use. | JUSTIFICATION: All 5 overlapping means (HR, HRV, SpO2, temp, min_HR) correlate r>0.97 with session-level equivalents — pure redundancy. The 18 unique signal features (slopes, variability, recovery ratio) all correlate |r|<0.13 with the target, below the best session feature (deep_minutes |r|=0.186). The 942K-row processing adds complexity for negligible gain. V2 path: sequence models on raw 5-min curves or personalised deviation baselines. | STATUS: Active

## D-020 — Workout Category: Unknown vs Rest Distinction
[2026-09-12] DECISION: Map null/empty/"-"/"n/a"/"none" workout_notes to "unknown" (not "rest"). Only explicit entries ("rest", "rest day", "no workout") map to "rest". workout_intensity = NaN for unknown (not 0). | JUSTIFICATION: "No data reported" is semantically different from "deliberately rested". A user who didn't log anything might have exercised; imputing rest injects false signal. LightGBM treats NaN intensity as a separate split category, which is correct. 2,737 unknown vs 2,010 rest after the fix — meaningful distinction. | STATUS: Active

## D-021 — Merge Strategy: LEFT JOIN + Independent Calendar Features
[2026-09-12] DECISION: Use LEFT JOIN throughout, anchored on checkins (4,989 rows). 513 rows (10.3%) have no matching sleep session → sleep features = NaN (correct). Calendar/temporal/holiday features computed independently from checkin_date (not from session data), so all 4,989 rows get them. Added has_session_data flag. | JUSTIFICATION: (1) 4,989 rows is small — dropping 10% would hurt model training; (2) LightGBM handles NaN natively — missing sleep features become implicit "ring-off" splits, which is informative; (3) calendar features (is_weekend, is_holiday, etc.) have nothing to do with whether the ring was worn — computing them from session data would create unnecessary NaN; (4) has_session_data flag (r=0.085 with target) lets the model learn the missing-data pattern explicitly. | STATUS: Active

## D-022 — Caffeine Data Inconsistency Fix
[2026-09-12] DECISION: (1) 2,108 rows with caffeine_mg > 0 but missing hours_before_bed → imputed with population median (7.5h). (2) 1,799 rows with hours_before_bed but no caffeine → cleared hours to NaN. Recalculated caffeine_close_to_bed after fix. | JUSTIFICATION: Inconsistency likely from partial form submissions. Median imputation is conservative; alternative (drop) would lose 20% of caffeine data. Orphaned hours-before-bed with no caffeine is contradictory — clearing prevents the model from learning a spurious pattern. | STATUS: Active

Every material decision is recorded here with its justification.
Format: `[DATE] DECISION: ... | JUSTIFICATION: ... | STATUS: ...`

---

## Phase 0: Project Understanding & Approach Selection

### D-001: Problem Framing — Regression over Classification
**Decision**: Treat subjective_feeling (1–5) as a continuous regression target, not a 5-class classification.
**Justification**: The target is ordinal. Classification ignores the ordering (predicting 1 vs 2 penalised same as 1 vs 5). Regression naturally respects distance between classes. With only 5,048 samples and heavy class imbalance (class 1 = 4.9%), classification would struggle. We can round predictions to integers for display.
**Status**: CONFIRMED

### D-002: V1 Model — LightGBM Regressor
**Decision**: Use LightGBM as the primary V1 model.
**Justification**: (1) Natively handles missing values — critical because stress_score is 65% null, caffeine is 20% null. (2) Handles nonlinear interactions without explicit feature engineering. (3) 5,048 rows is comfortable for gradient boosting with regularisation. (4) Interpretable via SHAP. (5) Fast to iterate. Alternative considered: Ridge regression (simpler but can't capture interactions); ordinal regression (respects ordering but fewer libraries, harder to deploy).
**Status**: CONFIRMED

### D-003: Exclude legacy_readiness_shown from Features
**Decision**: Do NOT use legacy_readiness_shown as a model feature.
**Justification**: (1) It correlates 0.77 with the target — suspiciously high. (2) It IS the current heuristic we're trying to replace — using it is circular. (3) Users see this score before answering the morning check-in, creating an anchoring effect that contaminates the label. (4) However, we WILL use it as a benchmark: the legacy heuristic's prediction accuracy is the bar to beat.
**Status**: CONFIRMED

### D-004: Session-Level Features over Raw Signals for V1
**Decision**: Aggregate nightly_signals to session-level statistics (mean, std, min, max, slope) rather than feeding raw 5-min sequences into a deep learning model.
**Justification**: (1) Only ~5,048 labelled sequences — at the lower bound for deep learning. (2) Hand-crafted aggregates capture 80–90% of temporal information (HRV trajectory = slope over first/second half). (3) LightGBM on tabular features will likely match DL with 10x less effort. (4) 1D-CNN/LSTM deferred to V2 as a comparison.
**Status**: CONFIRMED

### D-005: Time-Based Validation Split
**Decision**: Use temporal split (last ~2 weeks as test) rather than random split.
**Justification**: (1) Sleep data is temporally autocorrelated — a user's sleep on Monday predicts Tuesday. (2) Random split leaks future information. (3) Production deployment is always forward-looking. (4) Additionally, hold out a few users entirely to test cold-start performance.
**Status**: CONFIRMED

### D-006: User Baselines Are Low Priority
**Decision**: User-level personalisation is a V2 concern, not V1.
**Justification**: Variance decomposition shows 95.5% of target variance is within-user, only 6.8% between-user. The night-to-night variation is what matters, not who the user is. User-mean prediction has limited upside. However, user-normalised features (z-scores relative to personal history) ARE included in V1 as they capture deviation from personal norm.
**Status**: CONFIRMED

### D-007: Naive Baselines to Include
**Decision**: Implement two naive baselines: (1) Global mean (3.27), (2) Per-user historical mean.
**Justification**: These bound the problem. If V1 model can't beat user-mean by a meaningful margin, the signal-to-noise ratio may be too low for the available features. Both are trivially deployable.
**Status**: CONFIRMED

---

## Data Quality Decisions

### D-008: User ID Normalisation
**Decision**: Uppercase + strip all user_ids before any join.
**Justification**: Found three variants: 'UH-' (10,761 rows), 'uh-' (462 rows), '  U' (215 rows with leading spaces). Without normalisation, 349 apparent extra users are actually the same 120 users.
**Status**: CONFIRMED

### D-009: Epoch Timestamp Handling
**Decision**: Parse 1,731 epoch-millis timestamps using `pd.to_datetime(int(ts), unit='ms')`. Validated that they fall within the dataset's date range (Feb–May 2026).
**Justification**: These are clearly Unix millis (e.g., 1775482177105 → 2026-04-06 13:29:37). Affects 75 users. Discarding them would lose significant data.
**Status**: CONFIRMED

### D-010: Duplicate Session Handling
**Decision**: Drop exact duplicate session_ids (keep first).
**Justification**: 221 session_ids appear twice. Inspection confirms they are exact duplicates (same user, timestamps, sleep minutes). Likely an ingestion artifact.
**Status**: CONFIRMED

### D-011: Multi-Session Night Strategy
**Decision**: For user-nights with multiple sessions, keep the longest session as the "primary" sleep session. Flag the night as fragmented.
**Justification**: 984 user-nights have 2–4 sessions. These likely represent nap + main sleep, or fragmented recording. The longest session is the best proxy for the night's sleep quality. The fragmentation flag itself may be a useful feature.
**Status**: CONFIRMED

### D-012: Outlier Treatment
**Decision**: Clip physiological values to plausible ranges rather than dropping rows. Sentinel values (HRV=999, HR=0) set to NaN.
**Justification**: 34 HR outliers, 57 HRV outliers, 1,154 time-in-bed outliers, 29 efficiency outliers. Dropping would lose data unnecessarily. Clipping to [30,120] for HR, [1,300] for HRV, [60,900] for time-in-bed preserves the row while removing impossible values. LightGBM handles NaN natively.
**Status**: CONFIRMED

### D-016: Duration Column Decomposition Hierarchy + Derived Features
**Finding**: Three validated mathematical relationships in sleep session data:
1. `time_in_bed = total_sleep + awake` — 98.2% match within 1 min (primary decomposition)
2. `total_sleep = deep + rem + light` — 92.2% match within 1 min (864 rows have unscored/transitional sleep time, up to 43 min gap — normal wearable artefact)
3. `sleep_efficiency = total_sleep / time_in_bed` — 98.7% match within 0.01
**Decision**: Treat small gaps as device artefacts. Created 11 derived features from decomposition: `tib_gap_min`, `tib_decomp_exact`, `unscored_sleep_min`, `unscored_sleep_pct`, `stage_decomp_exact`, `efficiency_calc`, `efficiency_gap`, `efficiency_match`, `deep_pct`, `rem_pct`, `light_pct`, `awake_pct`, `decomp_quality_score` (0–3 composite).
**Status**: CONFIRMED

### D-018: Sensor Error Flags + Sentinel Value Handling
**Decision**: Added 6 binary error flags before fixing values, then applied targeted fixes:
- `hr_sensor_error` (31 rows): HR=0 (17, sensor failure) and HR=250 (14, max-cap sentinel) → avg_hr_bpm set to NaN. min_hr_bpm kept (normal values).
- `hrv_sensor_error` (50 rows): HRV=999 (sentinel) → avg_hrv_rmssd_ms set to NaN.
- `efficiency_error` (25 rows): Reported efficiency >1.0 (impossible) → overwritten with recalculated efficiency_calc (corrected range: 0.85–0.99).
- `duration_error` (26 rows): time_in_bed <60 min → dropped in duration filter.
- `decomp_error` (2 rows): total_sleep > time_in_bed (structurally impossible).
- `any_sensor_error` composite: 108 rows (1.07%) in final clean sessions.
**Justification**: Flagging before fixing preserves the error signal as a modelling feature. Sentinel→NaN is better than clipping (HR=250 clipped to 120 would be plausible but wrong). Efficiency→efficiency_calc is exact (all 25 recalculated values are valid).
**Final clean sessions**: 10,064 rows × 47 cols (120 users).
**Status**: CONFIRMED

### D-017: Negative Duration Values — Absolute Value (not clip-to-zero)
**Decision**: 57 rows with negative `awake_minutes` (min=-24.4) and 3 rows with negative `light_minutes` (min=-8.6) fixed with `abs()` instead of clipping to 0.
**Justification**: Tested three approaches (raw, clip-to-0, abs). Global decomposition match rates are identical (57/11,186 too small to move %). Zoomed into the 57 negative-awake rows: abs() reduces mean R1 residual to 49.8 min vs 62.0 min for clip-to-0. None fix R1 for these rows (structurally broken — ~60–100 min unaccounted transitional time), but abs() preserves the device’s measured magnitude, giving the model a real signal instead of an artificial 0.
**Status**: CONFIRMED

### D-015: Duration Columns — Seconds-to-Minutes Conversion
**Decision**: 785 rows (7.0% of sessions, all 120 users affected) have ALL 6 duration columns (`time_in_bed_minutes`, `total_sleep_minutes`, `deep_minutes`, `rem_minutes`, `light_minutes`, `awake_minutes`) stored in seconds instead of minutes. Converted by dividing by 60.
**Detection method**: Single-pass ratio approach: `time_in_bed / calc_duration_from_timestamps` in range [50, 70] catches all 785 rows regardless of absolute value. Wider band (50–70 instead of 55–65) handles DST-crossing sessions where timestamps shift by ±1 hour.
**Validation**: After fix, mean duration diff = -0.05 min (was -1832 min), std = 1.79 min (was 6874 min). Only 2 rows remain with >60min mismatch (genuine reporting discrepancies, not unit issues).
**Status**: CONFIRMED

### D-013: Weight Unit Disambiguation — REVERSED
**Original Decision**: Convert weights > 120 from lbs to kg using timezone heuristic.
**Revised Decision**: NO conversion. All weights are kept as raw kg values. High-weight users are genuine outliers, not unit-mismatch artifacts.
**Justification for reversal**: Timezone-wise height distributions are uniform (means 170–176 cm across all 5 timezones). If heights are on the same scale globally, weights almost certainly are too. The 11 outliers (9 in Asia/Kolkata, 1 each in Asia/Dubai and America/Los_Angeles) are flagged but NOT removed — removal is deferred to the modelling phase, only if it helps performance. BMI now ranges 15.9–79.6 (was 15.9–52.2 with the incorrect conversion).
**Status**: REVERSED

### D-014: Mood Tags Are Not Features
**Decision**: Exclude mood_tags from prediction features.
**Justification**: Mood tags are submitted simultaneously with the subjective_feeling target. They describe how the user feels RIGHT NOW, which is what we're predicting. Using them would be label leakage. However, they're valuable for error analysis (understand what "feeling 1" means semantically).
**Status**: CONFIRMED

---

## Open Questions (tracked for resolution)

- Q-001: Does firmware version systematically affect sensor readings? → Check in feature engineering phase
- Q-002: How strong is the anchoring effect from legacy_readiness_shown? → Would need A/B test data we don't have
- Q-003: Is there temporal drift in the target (user fatigue with the app)? → Check in EDA phase
- Q-004: What is the minimum number of user-nights needed for reliable user-normalised features? → Empirical test during feature engineering
- Q-005: UH-1036 has weight_kg=169.8 (BMI 52.2) — height was null during weight fix then imputed. Should we manually convert? → RESOLVED: edge case, 1 user, keep as-is
- Q-006: Early submitters (<9am) report feeling worse (3.15) vs late (3.48) — is this a real signal or confound? → submit_hour included as feature, let model decide

---

## Phase 1: Data Cleaning (executed 2026-09-12)

### Final Dataset Summary

| Metric | Value |
| --- | --- |
| Final rows | 4,989 |
| Final columns | 83 (77 features + 6 metadata/target) |
| Users | 120 |
| Date range | 2026-02-02 to 2026-05-02 (90 days) |
| Saved to | data/modelling\_table.csv (3.4 MB) |

### Rows Lost During Cleaning and Why

| Stage | Rows | Lost | Reason |
| --- | --- | --- | --- |
| Raw morning\_checkins | 5,048 | — | Starting point |
| After dedup (user\_id, date) | 4,989 | 59 | Duplicate check-in entries |
| Sleep sessions: raw | 11,438 | — | — |
| After user filter (120 target) | 11,407 | 31 | Extra users not in target |
| After session\_id dedup | 11,186 | 221 | Exact duplicate rows |
| After primary-session selection | 10,090 | 1,096 | Secondary/nap sessions per night |
| After duration outlier filter | 9,301 | 789 | time\_in\_bed outside [60, 900] min |
| Nightly signals: raw | 942,796 | — | — |
| After user + session filter | 831,489 | 111,307 | Restricted to clean sessions |

**Net target rows**: 4,989 (98.8% of raw checkins retained). The 857 rows without session data (17.2%) are kept with NaN sleep/signal features — LightGBM handles these natively.

### Join Coverage Rates Achieved

| Join | Match Rate | Unmatched |
| --- | --- | --- |
| Checkins → user\_profiles | 100% (4,989/4,989) | 0 |
| Checkins → daily\_context (prev day) | 100% (4,989/4,989) | 0 |
| Checkins → sleep\_sessions (end date) | 82.8% (4,132/4,989) | 857 |
| Checkins → signal\_agg (via session) | 82.2% (4,101/4,989) | 888 |

### Cleaning Operations Applied

1. **User ID normalization**: strip + uppercase across all 5 tables (fixed 'uh-', '  UH-' variants)
2. **Epoch timestamps**: 1,731 millis-epoch values parsed via `pd.to_datetime(int(ts), unit='ms')` — 0 unparseable after fix
3. **Duplicate sessions**: 221 exact duplicate session\_ids dropped (keep first)
4. **Multi-session nights**: 1,048 nights with 2–4 sessions → kept longest as primary, created `fragmented_night` flag
5. **Weight units**: NO conversion applied — all weights confirmed as kg (timezone-wise height distributions are uniform). 11 outliers flagged via IQR but kept in data
6. **Height imputation**: 5 nulls imputed with sex-specific median (M=180.3, F=165.2)
7. **Physiological outlier handling**: 17 HR sentinels (0→NaN), 50 HRV sentinels (999→NaN), 26 efficiency values clipped to [0.3, 1.0]
8. **Signal-level cleaning**: 1,961 SpO2 outliers (<80 or >100) → NaN; HR(0) and HRV(999) sentinels → NaN
9. **Daily context naming**: `userId` renamed to `user_id` for consistency
10. **Feature engineering**: 26 signal aggregates per session, binary flags (had\_caffeine, caffeine\_close\_to\_bed, had\_alcohol, had\_workout, is\_rest\_day, stress\_reported), 10 user z-score features, 3 temporal features

### Top Feature Correlations with Target (|r|)

| Feature | Correlation | Notes |
| --- | --- | --- |
| legacy\_readiness\_shown | +0.77 | BENCHMARK only, NOT a model feature |
| rem\_minutes | +0.59 | Top predictor |
| deep\_minutes | +0.57 | Top predictor |
| alcohol\_units | -0.55 | Strong negative — key context signal |
| total\_sleep\_minutes | +0.48 | Expected |
| movement\_index | -0.45 | Restlessness hurts |
| sleep\_efficiency | +0.45 | Expected |
| sig\_motion\_mean | -0.41 | Signal-level confirms session-level |
| awake\_minutes | -0.34 | More awake time → worse |
| sig\_hrv\_mean | +0.27 | HRV is predictive but weaker than sleep architecture |

### Issues Discovered During Execution

1. **Arrow serialization bug**: `display()` of DataFrame with dtype objects fails on serverless compute. Fixed by converting dtypes to strings in `shape_report` helper.
2. **numpy clip API mismatch**: `.clip(lower=0.01)` fails on numpy scalars inside pandas transform — numpy uses `min`/`max` not `lower`/`upper`. Fixed with `max(x.std(), 0.01)`.
3. **Selection bias check**: Low (r=-0.13 between checkin frequency and mean feeling). Users who check in more don't systematically rate differently.
4. **Mood tag validation**: Positive tags (energised, refreshed) → mean 4.19; negative tags (groggy, tired) → mean 1.95. Labels are internally consistent.
5. **Firmware effect**: Three versions (v2.0.7: 4,926, v2.1.0: 3,841, v2.0.3: 534). Sensor means should be checked for systematic differences — deferred to modelling phase.
6. **Missing data pattern**: 17.2% of checkins lack sleep session data. These are days where the ring wasn't worn or the session was filtered out. The model must handle missing sleep features gracefully.

### D-015: Keep Rows Without Session Data
**Decision**: Retain the 857 checkin rows that have no matching sleep session (17.2%).
**Justification**: (1) Dropping them loses 857 labelled examples. (2) LightGBM handles NaN natively. (3) These rows still have daily context features (steps, caffeine, alcohol, stress). (4) In production, users may occasionally not wear the ring — the model should still provide a prediction from context alone.
**Status**: CONFIRMED

### D-016: Submit Hour as a Feature
**Decision**: Include `submit_hour` and `early_submit` (before 9am) as features.
**Justification**: Early submitters report feeling worse (mean 3.15 vs 3.48). This likely reflects actual morning grogginess rather than bias — people who check in early are reporting their immediate state. In production, the model runs at wake-up time, so this information is available.
**Status**: CONFIRMED

### D-017: Mood Tags Excluded from Features, Used for Validation
**Decision**: mood\_tags are NOT features but ARE retained as `mood_category` for post-hoc error analysis.
**Justification**: Tags are submitted simultaneously with the target. Positive tags (mean=4.19) vs negative (mean=1.95) perfectly validate label quality but would be label leakage if used as features.
**Status**: CONFIRMED
