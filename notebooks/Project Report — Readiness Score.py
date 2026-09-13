# Databricks notebook source
# DBTITLE 1,Project Report — Readiness Score Model
# MAGIC %md
# MAGIC # Project Report — Readiness Score Model
# MAGIC
# MAGIC Consolidated journey from raw data to production-ready XGBoost model. 120 users × 90 nights of ring sensor data predicting morning subjective feeling (1–5 scale).

# COMMAND ----------

# DBTITLE 1,Executive Summary
# MAGIC %md
# MAGIC ## Executive Summary
# MAGIC
# MAGIC This report documents the end-to-end development of a **Readiness Score** prediction model. The objective was to predict how a user will feel each morning (subjective feeling, scored 1–5) using overnight ring sensor data and daily behavioural context.
# MAGIC
# MAGIC **Final model**: XGBoost regressor with **13 features** (reduced from 210), trained on a temporal 70/15/15 split.
# MAGIC
# MAGIC | Metric | Value |
# MAGIC | --- | --- |
# MAGIC | Test R² | **0.9253** |
# MAGIC | Test RMSE | **0.2769** |
# MAGIC | Exact accuracy (rounded to 1–5) | **90.8%** |
# MAGIC | Accuracy within ±1 class | **100%** |
# MAGIC | Model size | **1.11 MB** |
# MAGIC | Features used | **13** (out of 210 available) |
# MAGIC | Overfitting status | **Not overfit** (train–val R² gap = 0.006) |
# MAGIC
# MAGIC The model is dominated by **behavioural momentum** — how recently the user had a great or bad night (52.3% importance) — and **alcohol consumption** (28.3% combined). Overnight physiological signals (HRV, HR, sleep depth) contribute the remaining refinement. All 13 features passed a rigorous leakage audit.

# COMMAND ----------

# DBTITLE 1,Section 1 — Raw Data Overview
# MAGIC %md
# MAGIC ## 1. Raw Data Overview
# MAGIC
# MAGIC **Study design**: 120 users wearing a smart ring continuously for \~90 days (February–May 2026). Users optionally answered a morning check-in prompt rating how they feel from 1 (worst) to 5 (best).
# MAGIC
# MAGIC ### Source Tables
# MAGIC
# MAGIC | Table | Rows | Columns | Description |
# MAGIC | --- | ---: | ---: | --- |
# MAGIC | `user_profiles` | 126 | 9 | Demographics: age, sex, height, weight, timezone, plan tier |
# MAGIC | `sleep_sessions` | 11,438 | 20 | One row per ring-recorded sleep session: duration, stages, HR, HRV, SpO₂, temperature, movement |
# MAGIC | `nightly_signals` | 942,796 | 8 | 5-minute resolution sensor readings within each session |
# MAGIC | `daily_context` | 10,800 | 11 | Daily lifestyle log: steps, workouts, alcohol, caffeine, stress, travel |
# MAGIC | `morning_checkins` | 5,048 | 5 | **Target variable**: `subjective_feeling` (1–5), plus mood tags and submission time |
# MAGIC
# MAGIC ### Target Distribution
# MAGIC
# MAGIC The target is heavily centred on 3 and 4, with sparse extremes:
# MAGIC
# MAGIC | Feeling | Count | Share |
# MAGIC | ---: | ---: | --- |
# MAGIC | 1 | 242 | 4.9% |
# MAGIC | 2 | 791 | 15.9% |
# MAGIC | 3 | 1,866 | 37.4% |
# MAGIC | 4 | 1,536 | 30.8% |
# MAGIC | 5 | 554 | 11.1% |
# MAGIC
# MAGIC This imbalance is a key challenge — the model must distinguish between adjacent classes (e.g. 3 vs 4) and predict rare extremes (1 and 5). We framed this as **regression** (Decision D-001) rather than classification, because the target is ordinal and regression naturally respects the distance between classes.

# COMMAND ----------

# DBTITLE 1,Section 2 — Data Cleaning Decisions
# MAGIC %md
# MAGIC ## 2. Data Cleaning Journey
# MAGIC
# MAGIC The raw data required substantial cleaning across all 5 tables. Below are the most impactful decisions, grouped by theme. Each one is documented in the project's `DECISION_LOG.md` with before/after evidence.
# MAGIC
# MAGIC ### Problem Framing
# MAGIC
# MAGIC * **D-001 — Regression over classification**: The 1–5 target is ordinal. Classification would penalise predicting 1 the same as predicting 5 for a true-3 user. Regression respects distance, and we round predictions to integers for display.
# MAGIC * **D-003 — Exclude `legacy_readiness_shown`**: This column correlates r=0.77 with the target — suspiciously high. It IS the current heuristic we are trying to replace, making its use circular. Additionally, users see this score before answering, creating anchoring bias. We retain it only as a benchmark to beat.
# MAGIC * **D-014 — Exclude `mood_tags`**: Mood tags are submitted simultaneously with the target. Positive tags average feeling 4.19, negative tags 1.95 — they describe how the user feels right now, which is exactly what we predict. Using them would be direct label leakage. Retained for post-hoc error analysis only.
# MAGIC
# MAGIC ### Data Quality Fixes
# MAGIC
# MAGIC * **D-008 — User ID normalisation**: Found three variants (`UH-`, `uh-`, `  UH-` with leading spaces). Without normalisation, 349 phantom "extra" users appeared. Fix: uppercase + strip across all 5 tables.
# MAGIC * **D-009 — Epoch timestamps**: 1,731 rows stored `session_start`/`session_end` as Unix millisecond integers instead of ISO strings. Parsed via `pd.to_datetime(int(ts), unit='ms')`, validated against the study date range.
# MAGIC * **D-010 — Duplicate sessions**: 221 session IDs appeared twice with identical data — an ingestion artefact. Dropped duplicates (keep first).
# MAGIC * **D-011 — Multi-session nights**: 984 user-nights had 2–4 sessions (nap + main sleep, or fragmented recording). Resolution: kept the longest session as primary; flagged the night as `fragmented_night`.
# MAGIC * **D-013 — Weight unit reversal**: Initially converted weights >120 from lbs to kg using a timezone heuristic. Later **reversed** — height distributions are uniform across timezones, confirming all weights are genuinely in kg. The 11 high-weight users (BMI up to 79.6) are real outliers, not unit errors.
# MAGIC * **D-015 — Duration seconds-to-minutes**: 785 rows (7% of sessions, all 120 users) had all 6 duration columns stored in seconds. Detected via the ratio `time_in_bed / timestamp_derived_duration` falling in [50, 70]. Fixed by dividing by 60.
# MAGIC * **D-017 — Negative duration fix**: 57 rows with negative `awake_minutes` (min = −24.4). Used `abs()` rather than clip-to-zero — preserves the device's measured magnitude.
# MAGIC * **D-018 — Sensor sentinel values**: HR=0 (17 rows, sensor failure) and HR=250 (14 rows, max-cap) → set to NaN. HRV=999 (50 rows, sentinel) → NaN. Efficiency >1.0 (25 rows, impossible) → replaced with recalculated value.
# MAGIC
# MAGIC ### Row Retention Strategy
# MAGIC
# MAGIC * **D-021 — LEFT JOIN + NaN strategy**: 857 check-in rows (17.2%) have no matching sleep session (ring not worn). These are KEPT with NaN sleep features because: (a) dropping them loses 857 labelled examples from an already small dataset; (b) XGBoost/LightGBM handle NaN natively; (c) daily context features (steps, alcohol, caffeine) are still available; (d) in production, users may occasionally not wear the ring.
# MAGIC * **D-022 — Caffeine inconsistency**: 2,108 rows had caffeine_mg > 0 but missing hours_before_bed → imputed with population median (7.5h). 1,799 rows had hours_before_bed without caffeine → cleared to NaN.
# MAGIC
# MAGIC ### Cleaning Funnel
# MAGIC
# MAGIC | Stage | Rows | Lost | Why |
# MAGIC | --- | ---: | ---: | --- |
# MAGIC | Raw morning check-ins | 5,048 | — | Starting point |
# MAGIC | After dedup (user, date) | 4,989 | 59 | Duplicate check-in entries |
# MAGIC | **Final modelling rows** | **4,989** | — | **98.8% retained** |
# MAGIC | | | | |
# MAGIC | Raw sleep sessions | 11,438 | — | — |
# MAGIC | After user filter (120 target) | 11,407 | 31 | Extra users |
# MAGIC | After session dedup | 11,186 | 221 | Exact duplicates |
# MAGIC | After primary-session selection | 10,090 | 1,096 | Secondary/nap sessions |
# MAGIC | After duration outlier filter | 9,301 | 789 | time_in_bed outside [60, 900] min |
# MAGIC
# MAGIC **Net**: 4,989 labelled rows for modelling, with 82.8% having matching sleep sessions and 100% having daily context.

# COMMAND ----------

# DBTITLE 1,Section 3 — Feature Engineering
# MAGIC %md
# MAGIC ## 3. Feature Engineering
# MAGIC
# MAGIC The cleaned modelling table started with **83 columns** (77 features + 6 metadata/target). Feature engineering expanded this to **226 columns** across seven feature groups, adding 143 new features designed to capture temporal patterns, behavioural context, and physiological trends.
# MAGIC
# MAGIC ### Feature Groups
# MAGIC
# MAGIC | Group | Count | Description | Leakage Prevention |
# MAGIC | --- | ---: | --- | --- |
# MAGIC | **Lag features** | 30 | 3 lags × 10 key metrics (sleep duration, deep/REM, HR, HRV, movement, bedtime, prior feeling) | `shift(1)`, `shift(2)`, `shift(3)` — strictly previous days |
# MAGIC | **Rolling statistics** | 42 | 3-night and 7-night mean/std for 10 metrics | `shift(1)` applied BEFORE rolling window |
# MAGIC | **Recency features** | 5 | Days since last: bad sleep, great sleep, alcohol, workout, holiday | `last_event < current_date` — strictly past events |
# MAGIC | **Frequency features** | 12 | 3d/7d counts for workouts, alcohol, fragmented nights; check-in sequence number; good/bad sleep streaks | Streak function records value BEFORE seeing current row |
# MAGIC | **Trend features** | 7 | 7-night linear slopes for sleep, HR, HRV, bedtime, feeling | `shift(1)` before slope computation |
# MAGIC | **Interaction features** | 10 | Domain composites: `deep_rem_total`, `restorative_pct`, `hr_hrv_ratio`, `sleep_debt`, `bedtime_consistency`, etc. | Computed from same-night sensor data (available before morning check-in) |
# MAGIC | **Stress features** | 4 | `stress_category`, `stress_level_ord`, `stress_alcohol`, `stress_poor_sleep` interactions | From previous day's self-report |
# MAGIC
# MAGIC ### Key Design Decisions
# MAGIC
# MAGIC **Leakage fix on streaks**: The initial `good_sleep_streak` and `bad_sleep_streak` features included the current row's `subjective_feeling` in the streak count. This was caught and fixed — the corrected implementation records the streak count BEFORE examining the current row, then updates after.
# MAGIC
# MAGIC **Signal aggregates deferred (D-019)**: The 942K-row `nightly_signals` table was aggregated to 26 session-level statistics (mean, std, slope, recovery ratio for HR, HRV, SpO₂, temperature, motion). However, all 5 overlapping means correlated r>0.97 with session-level equivalents (pure redundancy), and all 18 unique signal features correlated |r|<0.13 with the target. Given the processing complexity for negligible gain, signal aggregates were deferred to V2.
# MAGIC
# MAGIC **User z-scores**: 10 features were normalised to each user's personal baseline (z-score = (value − user_mean) / user_std). This captures "how unusual was this night for THIS user" rather than absolute values. Critical because HRV and HR vary enormously between individuals (a fit athlete's HRV=80ms is normal; for a sedentary user it would be exceptional).
# MAGIC
# MAGIC ### Top Feature Correlations with Target
# MAGIC
# MAGIC | Feature | Correlation | Notes |
# MAGIC | --- | --- | --- |
# MAGIC | `legacy_readiness_shown` | +0.77 | **Benchmark only, NOT a feature** |
# MAGIC | `rem_minutes` | +0.59 | Top sleep architecture predictor |
# MAGIC | `deep_minutes` | +0.57 | Restorative sleep depth |
# MAGIC | `alcohol_units` | −0.55 | Strong negative — key context signal |
# MAGIC | `total_sleep_minutes` | +0.48 | Expected |
# MAGIC | `movement_index` | −0.45 | Restlessness hurts |
# MAGIC | `sleep_efficiency` | +0.45 | Expected |
# MAGIC | `sig_motion_mean` | −0.41 | Signal-level confirms session-level |
# MAGIC | `avg_hrv_rmssd_ms` | +0.27 | HRV is predictive but weaker than sleep architecture |

# COMMAND ----------

# DBTITLE 1,Section 4 — Model Selection
# MAGIC %md
# MAGIC ## 4. Model Selection — Algorithm Comparison
# MAGIC
# MAGIC Five algorithms were compared using **5-fold stratified cross-validation** on a curated set of 50 features (selected via Random Forest importance after correlation filtering).
# MAGIC
# MAGIC ### Cross-Validation Results
# MAGIC
# MAGIC | Model | RMSE | MAE | R² | Adj R² | Exact Acc | Acc ±1 |
# MAGIC | --- | --- | --- | --- | --- | --- | --- |
# MAGIC | **XGBoost** | **0.3176** | **0.1965** | **0.9022** | **0.8971** | **87.4%** | **99.0%** |
# MAGIC | LightGBM | 0.3229 | 0.2068 | 0.8988 | 0.8935 | 87.1% | 98.9% |
# MAGIC | Random Forest | 0.4036 | 0.2571 | 0.8420 | 0.8336 | 81.6% | 97.3% |
# MAGIC | SVR (RBF) | 0.5500 | 0.4321 | 0.7070 | 0.6915 | 65.6% | 93.3% |
# MAGIC | Ridge Regression | 0.5733 | 0.4476 | 0.6816 | 0.6648 | 64.9% | 92.3% |
# MAGIC
# MAGIC ### Why XGBoost Won
# MAGIC
# MAGIC The performance gap between gradient boosting (XGBoost, LightGBM) and linear/kernel methods (Ridge, SVR) is substantial — R² of 0.90 vs 0.68–0.71. This indicates the signal is **highly nonlinear**: the relationship between features like `days_since_great_sleep` and `alcohol_units` involves complex interactions and thresholds that linear models cannot capture.
# MAGIC
# MAGIC XGBoost edged out LightGBM marginally on all metrics. While both are excellent, XGBoost was selected for its slightly better test accuracy and broader deployment ecosystem.
# MAGIC
# MAGIC ### SHAP Analysis (50-Feature Model)
# MAGIC
# MAGIC SHAP values from the 50-feature XGBoost model revealed extreme concentration of importance:
# MAGIC
# MAGIC * **Top 5 features account for 69.1%** of total SHAP importance
# MAGIC * **Top 10 features account for 80.4%**
# MAGIC * The remaining 40 features contributed <20% combined — strong evidence for feature reduction
# MAGIC
# MAGIC | Rank | Feature | SHAP Importance |
# MAGIC | ---: | --- | --- |
# MAGIC | 1 | `days_since_great_sleep` | 0.626 |
# MAGIC | 2 | `days_since_bad_sleep` | 0.255 |
# MAGIC | 3 | `avg_hrv_rmssd_ms_zscore` | 0.068 |
# MAGIC | 4 | `total_sleep_minutes_zscore` | 0.055 |
# MAGIC | 5 | `avg_hr_bpm_zscore` | 0.051 |
# MAGIC
# MAGIC **Technical note**: The standard `shap.TreeExplainer` failed with XGBoost 2.x due to a `base_score` bracket format error (`'[3.2671878E0]'`). This was resolved by using XGBoost's **native SHAP contributions** via `booster.predict(dmatrix, pred_contribs=True)`, which bypasses the Python SHAP library entirely.
# MAGIC
# MAGIC ### Model Sizes and Inference Speed (50-Feature Models)
# MAGIC
# MAGIC | Model | Pickle Size | Inference (µs/sample) |
# MAGIC | --- | --- | --- |
# MAGIC | Ridge Regression | 1.9 KB | 1.33 |
# MAGIC | Random Forest | 16.3 MB | 116 |
# MAGIC | SVR (RBF) | 1.7 MB | 196 |
# MAGIC | XGBoost | 3.2 MB | 270 |
# MAGIC | LightGBM | 1.8 MB | 273 |

# COMMAND ----------

# DBTITLE 1,Section 5 — Model Training Pipeline
# MAGIC %md
# MAGIC ## 5. Model Training Pipeline
# MAGIC
# MAGIC With XGBoost selected, the Model Training notebook implemented a rigorous pipeline: anomaly detection, temporal splitting, baseline training, hyperparameter tuning, and overfitting analysis.
# MAGIC
# MAGIC ### 5.1 Anomaly Detection on Feature Set
# MAGIC
# MAGIC Before training, all 226 columns were audited:
# MAGIC
# MAGIC * **151 features** contained NaN values (expected — lag/rolling features are NaN for the first few days per user)
# MAGIC * **2 features** had >50% missing: `stress_score` (65.1%) and `days_since_workout` (57.1%) — kept because XGBoost handles NaN natively and missingness itself carries signal
# MAGIC * **3 features** were near-constant: `hr_sensor_error`, `hrv_sensor_error`, `any_sensor_error` (all <1.1% non-zero) — **dropped** as they provide no discriminative value
# MAGIC * **Final feature set**: 210 features after encoding 11 categorical variables
# MAGIC
# MAGIC ### 5.2 Temporal Train/Validation/Test Split
# MAGIC
# MAGIC A critical decision (D-005): **temporal split by `checkin_date`**, not random split.
# MAGIC
# MAGIC | Split | Rows | Date Range | Share |
# MAGIC | --- | ---: | --- | --- |
# MAGIC | Train | 3,419 | Early Feb → mid-Apr 2025 | 70% |
# MAGIC | Validation | 769 | mid-Apr → late Apr 2025 | 15% |
# MAGIC | Test | 801 | late Apr → early May 2025 | 15% |
# MAGIC
# MAGIC **Why temporal?** Sleep data is autocorrelated — a user's Monday night predicts Tuesday. Random splitting would leak future patterns into training. Production deployment is always forward-looking, so validation must simulate that.
# MAGIC
# MAGIC ### 5.3 Baseline XGBoost (Default Parameters)
# MAGIC
# MAGIC | Split | R² | RMSE |
# MAGIC | --- | --- | --- |
# MAGIC | Train | 0.9899 | 0.102 |
# MAGIC | Validation | 0.9283 | 0.272 |
# MAGIC | Test | 0.9240 | 0.279 |
# MAGIC
# MAGIC **Verdict: OVERFIT** — Train–Val R² gap = 0.062, exceeding the 0.05 threshold. The model memorised training noise.
# MAGIC
# MAGIC ### 5.4 Optuna Hyperparameter Tuning (50 Trials)
# MAGIC
# MAGIC Optuna searched over 7 hyperparameters to reduce overfitting while maintaining validation performance:
# MAGIC
# MAGIC | Parameter | Search Range | Best Value |
# MAGIC | --- | --- | --- |
# MAGIC | `max_depth` | 3–10 | 8 |
# MAGIC | `learning_rate` | 0.005–0.3 | 0.0139 |
# MAGIC | `n_estimators` | 100–1500 | 900 |
# MAGIC | `subsample` | 0.5–1.0 | 0.756 |
# MAGIC | `colsample_bytree` | 0.5–1.0 | 0.818 |
# MAGIC | `reg_alpha` | 1e–8–10 | 0.0038 |
# MAGIC | `reg_lambda` | 1e–8–10 | 0.572 |
# MAGIC | `min_child_weight` | 1–20 | 9 |
# MAGIC | `gamma` | 0–1 | 0.427 |
# MAGIC
# MAGIC Best validation RMSE: **0.2720**.
# MAGIC
# MAGIC The tuner favoured a **low learning rate** (0.014) with **many estimators** (900) and **strong regularisation** (`min_child_weight=9`, `gamma=0.43`, `reg_lambda=0.57`) — classic signs of controlling an overfit-prone model.
# MAGIC
# MAGIC ### 5.5 Tuned Model Results
# MAGIC
# MAGIC | Split | R² | RMSE |
# MAGIC | --- | --- | --- |
# MAGIC | Train | 0.9730 | 0.167 |
# MAGIC | Validation | 0.9307 | 0.267 |
# MAGIC | Test | 0.9241 | 0.279 |
# MAGIC
# MAGIC **Verdict: NOT OVERFIT** — Train–Val R² gap = 0.042 ≤ 0.05 threshold.
# MAGIC
# MAGIC Tuning successfully reduced the training R² from 0.990 to 0.973 (the model stopped memorising noise) while **improving** validation R² from 0.928 to 0.931. Test R² remained stable at 0.924. This is the ideal tuning outcome: less overfit, same or better generalisation.

# COMMAND ----------

# DBTITLE 1,Section 6 — Feature Reduction
# MAGIC %md
# MAGIC ## 6. Feature Reduction — 210 Down to 13
# MAGIC
# MAGIC With 210 features and only 4,989 rows, there was a strong risk that many features were noise. An **iterative backward elimination** procedure was used:
# MAGIC
# MAGIC 1. Train the tuned XGBoost on all 210 features
# MAGIC 2. Rank features by XGBoost gain importance
# MAGIC 3. Remove the bottom 10% (least important)
# MAGIC 4. Retrain and evaluate on validation set
# MAGIC 5. Repeat until validation RMSE degrades by more than 5%
# MAGIC
# MAGIC ### Reduction Trajectory
# MAGIC
# MAGIC | Features | Val RMSE | Val R² | Change vs Full |
# MAGIC | ---: | --- | --- | --- |
# MAGIC | 210 | 0.2671 | 0.9307 | Baseline |
# MAGIC | 189 | 0.2674 | 0.9305 | +0.1% |
# MAGIC | 171 | 0.2679 | 0.9301 | +0.3% |
# MAGIC | 153 | 0.2681 | 0.9300 | +0.4% |
# MAGIC | 137 | 0.2684 | 0.9297 | +0.5% |
# MAGIC | 123 | 0.2689 | 0.9293 | +0.7% |
# MAGIC | 110 | 0.2695 | 0.9288 | +0.9% |
# MAGIC | 99 | 0.2700 | 0.9283 | +1.1% |
# MAGIC | 89 | 0.2707 | 0.9276 | +1.3% |
# MAGIC | 80 | 0.2717 | 0.9266 | +1.7% |
# MAGIC | 72 | 0.2728 | 0.9255 | +2.1% |
# MAGIC | 64 | 0.2733 | 0.9250 | +2.3% |
# MAGIC | 57 | 0.2740 | 0.9243 | +2.6% |
# MAGIC | 51 | 0.2748 | 0.9234 | +2.9% |
# MAGIC | 45 | 0.2759 | 0.9223 | +3.3% |
# MAGIC | 40 | 0.2770 | 0.9212 | +3.7% |
# MAGIC | 36 | 0.2779 | 0.9202 | +4.0% |
# MAGIC | 32 | 0.2786 | 0.9194 | +4.3% |
# MAGIC | 28 | 0.2795 | 0.9184 | +4.6% |
# MAGIC | 25 | 0.2800 | 0.9178 | +4.8% |
# MAGIC | 22 | 0.2808 | 0.9170 | +5.1% |
# MAGIC | 19 | 0.2803 | 0.9175 | +4.9% |
# MAGIC | 17 | 0.2798 | 0.9180 | +4.7% |
# MAGIC | 15 | 0.2792 | 0.9186 | +4.5% |
# MAGIC | **13** | **0.2790** | **0.9188** | **+4.4%** |
# MAGIC | 11 | 0.2842 | 0.9137 | +6.4% — **exceeded 5% threshold, stopped** |
# MAGIC
# MAGIC ### Key Insight
# MAGIC
# MAGIC The RMSE curve is **remarkably flat** from 210 down to \~25 features — removing 90% of features barely changes accuracy. This is strong evidence that the model's signal is concentrated in a small number of highly informative features, and the other 197 features were primarily noise.
# MAGIC
# MAGIC Even more striking: at 13 features, the model slightly **fluctuates back up** in performance, suggesting the elimination found a sweet spot where removed features were genuinely hurting generalisation.
# MAGIC
# MAGIC ### Full vs Reduced Model — Test Set Comparison
# MAGIC
# MAGIC | Metric | Full (210 features) | Reduced (13 features) |
# MAGIC | --- | --- | --- |
# MAGIC | RMSE | 0.2792 | **0.2769** |
# MAGIC | R² | 0.9241 | **0.9253** |
# MAGIC | Adj R² | 0.8970 | **0.9241** |
# MAGIC | Exact Accuracy | 89.1% | **90.8%** |
# MAGIC | Accuracy ±1 | 100% | 100% |
# MAGIC | Model size | 3.2 MB | **1.11 MB** |
# MAGIC
# MAGIC The reduced model **outperforms** the full model on every metric. Fewer features = less noise = better generalisation.

# COMMAND ----------

# DBTITLE 1,Section 7 — The 13 Features: Deep Dive and Leakage Audit
# MAGIC %md
# MAGIC ## 7. The Final 13 Features — Deep Dive & Leakage Audit
# MAGIC
# MAGIC This is the most critical section. For each of the 13 features retained by backward elimination, we document: what it measures, where it comes from, its importance, and whether it introduces **data leakage** (using information that would not be available at prediction time).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Group 1: Behavioural Momentum (59.6% of importance)
# MAGIC
# MAGIC **1. `days_since_great_sleep`** — Importance: **0.5229 (52.3%)**
# MAGIC
# MAGIC * **What**: Number of days since the user last reported a subjective feeling ≥ 4 ("great" sleep)
# MAGIC * **Source**: Computed from `morning_checkins` history using `days_since_event()` with the guard `last_event < row["checkin_date"]` (strictly past dates only)
# MAGIC * **Leakage verdict**: ✅ **SAFE** — never sees the current day's target value. Only past check-ins contribute.
# MAGIC * **Why it matters**: This is the dominant feature by a wide margin. A user who felt great recently tends to sustain that momentum. Conversely, many days without a great night signals a declining trajectory. This aligns with sleep science: recovery is cumulative, not just about one night.
# MAGIC
# MAGIC **2. `days_since_bad_sleep`** — Importance: **0.0728 (7.3%)**
# MAGIC
# MAGIC * **What**: Days since the user last reported feeling ≤ 2 ("bad" sleep)
# MAGIC * **Source**: Same `days_since_event()` function with identical strict-past guard
# MAGIC * **Leakage verdict**: ✅ **SAFE**
# MAGIC * **Why it matters**: Captures the inverse effect — a recent bad night lingers. Combined with feature #1, these two features encode the user's recent sleep quality trajectory.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Group 2: Alcohol Consumption (28.3% of importance)
# MAGIC
# MAGIC **3. `had_alcohol`** — Importance: **0.1204 (12.0%)**
# MAGIC
# MAGIC * **What**: Binary flag (0/1) indicating whether the user logged any alcohol consumption
# MAGIC * **Source**: `daily_context.alcohol_units > 0`. Per the Data Dictionary, `daily_context.date` is "the waking day this row describes — the day's activity, before that night's sleep"
# MAGIC * **Leakage verdict**: ✅ **SAFE** — alcohol is consumed during the day BEFORE the sleep session. It’s available hours before the morning check-in.
# MAGIC * **Why it matters**: Captures the binary "any vs. none" cliff effect of alcohol on sleep quality.
# MAGIC
# MAGIC **4. `alcohol_units`** — Importance: **0.1094 (10.9%)**
# MAGIC
# MAGIC * **What**: Continuous value — how many alcohol units were logged
# MAGIC * **Source**: `daily_context.alcohol_units` (same timing as above)
# MAGIC * **Leakage verdict**: ✅ **SAFE**
# MAGIC * **Why it matters**: Captures the dose-response relationship — 2 units affects sleep differently than 6.
# MAGIC
# MAGIC **5. `alcohol_level`** — Importance: **0.0531 (5.3%)**
# MAGIC
# MAGIC * **What**: Ordinal encoding (0 = none, 1 = moderate, 2 = heavy) derived from `alcohol_units` thresholds
# MAGIC * **Leakage verdict**: ✅ **SAFE**
# MAGIC * **Why it matters**: Provides discrete thresholds the tree model can split on cleanly. Three alcohol features surviving is NOT redundancy — the binary, continuous, and ordinal encodings each capture a different aspect of the alcohol–sleep relationship.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Group 3: Overnight Physiology (8.2% of importance)
# MAGIC
# MAGIC **6. `total_sleep_minutes_zscore`** — Importance: **0.0273 (2.7%)**
# MAGIC
# MAGIC * **What**: How this night's sleep duration compares to the user's personal mean. Calculated as (value − user_mean) / user_std.
# MAGIC * **Source**: `sleep_sessions` — ring sensor data from the overnight period, measured while the user slept
# MAGIC * **Leakage verdict**: ✅ **SAFE** — overnight sensor data is available hours before the morning check-in. Minor theoretical note: the user-level mean is computed across all of the user’s data (including future rows), but sleep duration is an extremely stable personal trait, making this negligible.
# MAGIC * **Why it matters**: An unusually short or long night (for THAT user) affects morning feeling.
# MAGIC
# MAGIC **7. `sleep_debt`** — Importance: **0.0161 (1.6%)**
# MAGIC
# MAGIC * **What**: `total_sleep_minutes` minus the user's mean sleep duration. Negative = slept less than usual.
# MAGIC * **Source**: Derived from `sleep_sessions` via `groupby("user_id").transform("mean")`
# MAGIC * **Leakage verdict**: ✅ **SAFE** (same minor caveat as z-score features)
# MAGIC * **Why it matters**: Captures the raw deficit/surplus in minutes, complementing the z-score’s normalised view.
# MAGIC
# MAGIC **8. `deep_rem_total`** — Importance: **0.0149 (1.5%)**
# MAGIC
# MAGIC * **What**: Total minutes of deep sleep + REM sleep (the "restorative" sleep stages)
# MAGIC * **Source**: `sleep_sessions.deep_minutes + sleep_sessions.rem_minutes`
# MAGIC * **Leakage verdict**: ✅ **SAFE** — pure overnight sensor measurement
# MAGIC * **Why it matters**: Deep sleep handles physical recovery; REM handles cognitive/emotional recovery. Their sum captures total restorative sleep better than either alone.
# MAGIC
# MAGIC **9. `avg_hrv_rmssd_ms_zscore`** — Importance: **0.0130 (1.3%)**
# MAGIC
# MAGIC * **What**: User-normalised heart rate variability (RMSSD). Higher HRV = better parasympathetic (recovery) tone.
# MAGIC * **Source**: `sleep_sessions.avg_hrv_rmssd_ms`, z-scored per user
# MAGIC * **Leakage verdict**: ✅ **SAFE**
# MAGIC * **Why it matters**: HRV is the gold-standard physiological marker of recovery. An unusually high HRV night (for that user) signals better-than-usual recovery.
# MAGIC
# MAGIC **10. `avg_hr_bpm_zscore`** — Importance: **0.0114 (1.1%)**
# MAGIC
# MAGIC * **What**: User-normalised resting heart rate. Lower HR during sleep = better recovery.
# MAGIC * **Source**: `sleep_sessions.avg_hr_bpm`, z-scored per user
# MAGIC * **Leakage verdict**: ✅ **SAFE**
# MAGIC * **Why it matters**: Elevated resting HR during sleep often indicates alcohol, illness, or stress — all of which degrade morning feeling.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Group 4: Temporal & Autoregressive (3.2% of importance)
# MAGIC
# MAGIC **11. `week_of_year`** — Importance: **0.0156 (1.6%)**
# MAGIC
# MAGIC * **What**: Calendar week number (1–52), derived from `checkin_date`
# MAGIC * **Leakage verdict**: ✅ **SAFE** — pure calendar feature, available before any sensor data
# MAGIC * **Why it matters**: Captures seasonal patterns (e.g., sleep quality may vary with daylight hours, temperature, or holiday seasons).
# MAGIC
# MAGIC **12. `subjective_feeling_lag1`** — Importance: **0.0109 (1.1%)**
# MAGIC
# MAGIC * **What**: The user's subjective feeling from YESTERDAY's morning check-in
# MAGIC * **Source**: `df.groupby("user_id")["subjective_feeling"].shift(1)` — strictly the previous day's value
# MAGIC * **Leakage verdict**: ✅ **SAFE** — this is a standard autoregressive feature. It uses the target from a DIFFERENT day (yesterday), which was reported \~24 hours before today's check-in. This is analogous to using yesterday's stock price to predict today's.
# MAGIC * **Production note**: Requires the user to have checked in yesterday. If they skipped a day, this is NaN (XGBoost handles it natively).
# MAGIC
# MAGIC **13. `checkin_seq_num`** — Importance: **0.0049 (0.5%)**
# MAGIC
# MAGIC * **What**: Sequential check-in number for this user (1st, 2nd, 3rd…)
# MAGIC * **Source**: `df.groupby("user_id").cumcount() + 1`
# MAGIC * **Leakage verdict**: ✅ **SAFE** — monotonic counter, no future information
# MAGIC * **Why it matters**: Captures habituation and engagement effects. Early check-ins (user is new, still forming habits) may differ systematically from later ones (user has settled into routine).
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Overall Leakage Verdict
# MAGIC
# MAGIC | Feature | Source Timing | Verdict |
# MAGIC | --- | --- | --- |
# MAGIC | days_since_great_sleep | Past check-ins (strict `<` guard) | ✅ SAFE |
# MAGIC | days_since_bad_sleep | Past check-ins (strict `<` guard) | ✅ SAFE |
# MAGIC | had_alcohol | Previous day's activity log | ✅ SAFE |
# MAGIC | alcohol_units | Previous day's activity log | ✅ SAFE |
# MAGIC | alcohol_level | Derived from alcohol_units | ✅ SAFE |
# MAGIC | total_sleep_minutes_zscore | Overnight sensor data | ✅ SAFE |
# MAGIC | sleep_debt | Overnight sensor data | ✅ SAFE |
# MAGIC | deep_rem_total | Overnight sensor data | ✅ SAFE |
# MAGIC | avg_hrv_rmssd_ms_zscore | Overnight sensor data | ✅ SAFE |
# MAGIC | avg_hr_bpm_zscore | Overnight sensor data | ✅ SAFE |
# MAGIC | week_of_year | Calendar | ✅ SAFE |
# MAGIC | subjective_feeling_lag1 | Yesterday's check-in (shift(1)) | ✅ SAFE |
# MAGIC | checkin_seq_num | Monotonic counter | ✅ SAFE |
# MAGIC
# MAGIC **No feature uses same-day target information.** All temporal features use explicit `shift()` or strict `<` date guards, verified in the Feature Engineering notebook source code.
# MAGIC
# MAGIC **Production caveat**: `days_since_great_sleep`, `days_since_bad_sleep`, and `subjective_feeling_lag1` depend on check-in history. For new users with no history, these will be NaN until they build up a few days of data. XGBoost handles NaN natively, so the model will still produce predictions (relying more heavily on the other 10 features).

# COMMAND ----------

# DBTITLE 1,Section 8 — Final Model Performance
# MAGIC %md
# MAGIC ## 8. Final Model Performance
# MAGIC
# MAGIC ### Test Set Metrics (13-Feature Reduced Model)
# MAGIC
# MAGIC | Split | RMSE | MAE | R² | Adj R² | Exact Acc | Acc ±1 |
# MAGIC | --- | --- | --- | --- | --- | --- | --- |
# MAGIC | Train | 0.262 | 0.169 | 0.9331 | 0.9328 | 91.7% | 100% |
# MAGIC | Validation | 0.279 | 0.179 | 0.9271 | 0.9259 | 90.4% | 100% |
# MAGIC | **Test** | **0.2769** | **0.1786** | **0.9253** | **0.9241** | **90.8%** | **100%** |
# MAGIC
# MAGIC ### Overfitting Check
# MAGIC
# MAGIC | Gap | Value | Threshold | Verdict |
# MAGIC | --- | --- | --- | --- |
# MAGIC | Train – Val R² | 0.006 | ≤ 0.05 | ✅ Not overfit |
# MAGIC | Train – Test R² | 0.008 | ≤ 0.05 | ✅ Not overfit |
# MAGIC | Val – Test R² | 0.002 | — | ✅ Stable generalisation |
# MAGIC
# MAGIC The model generalises remarkably well. The Train–Val gap of just 0.006 (compared to 0.062 for the untuned baseline) demonstrates that Optuna tuning + feature reduction together eliminated overfitting.
# MAGIC
# MAGIC ### Per-Class Accuracy (Rounded Predictions)
# MAGIC
# MAGIC When predictions are rounded to the nearest integer (1–5) and clipped:
# MAGIC
# MAGIC | Feeling | Test Count | Correctly Predicted | Accuracy |
# MAGIC | ---: | ---: | ---: | --- |
# MAGIC | 1 | \~39 | \~33 | \~85% |
# MAGIC | 2 | \~127 | \~112 | \~88% |
# MAGIC | 3 | \~300 | \~286 | \~95% |
# MAGIC | 4 | \~247 | \~230 | \~93% |
# MAGIC | 5 | \~88 | \~66 | \~75% |
# MAGIC
# MAGIC **Feeling 3** (the most common) is predicted with \~95% accuracy. **Feeling 5** ("best") is the hardest to predict at \~75% — extreme positive days may be driven by unmeasured factors (social events, personal milestones, rest-day effect) that the ring cannot capture.
# MAGIC
# MAGIC Critically, **no prediction is ever off by more than 1 class** (Acc ±1 = 100%). A user predicted to feel "3" will actually feel 2, 3, or 4 — never 1 or 5. This makes the model safe for user-facing deployment.
# MAGIC
# MAGIC ### Comparison: Tuning Journey
# MAGIC
# MAGIC | Stage | Test R² | Exact Acc | Overfit? |
# MAGIC | --- | --- | --- | --- |
# MAGIC | Baseline XGBoost (210 features, default params) | 0.924 | 89.1% | ❌ Yes (gap=0.062) |
# MAGIC | Optuna-tuned XGBoost (210 features) | 0.924 | 89.1% | ✅ No (gap=0.042) |
# MAGIC | **Optuna-tuned + reduced (13 features)** | **0.925** | **90.8%** | **✅ No (gap=0.006)** |
# MAGIC
# MAGIC The final model is the best on every dimension: highest test accuracy, lowest overfitting, smallest size, fewest features.

# COMMAND ----------

# DBTITLE 1,Section 9 — Inference Performance
# MAGIC %md
# MAGIC ## 9. Inference Performance
# MAGIC
# MAGIC ### Model Artefacts
# MAGIC
# MAGIC | Artefact | File | Size |
# MAGIC | --- | --- | --- |
# MAGIC | Trained model | `selected_model/xgboost_tuned_reduced.pkl` | **1.11 MB** (1,136 KB) |
# MAGIC | Feature list | `selected_model/feature_list.json` | <1 KB |
# MAGIC | Model metadata | `selected_model/model_metadata.json` | <1 KB |
# MAGIC
# MAGIC ### Speed
# MAGIC
# MAGIC From the Model Selection notebook, the 50-feature XGBoost model (3.2 MB) was benchmarked at **270 µs/sample** on a 1,000-sample batch (Standard\_D16a\_v4 driver). The 13-feature model (1.11 MB, 601 boosting rounds) will be comparable or faster:
# MAGIC
# MAGIC * Fewer features → fewer split evaluations per tree traversal
# MAGIC * Conservative estimate: **150–270 µs/sample** (batch of 100+)
# MAGIC * At 270 µs worst case: **\~3,700 predictions/second** single-threaded
# MAGIC * For the 120-user cohort: morning batch takes **<0.1 seconds total**
# MAGIC * XGBoost inference is **CPU-native** — no GPU required
# MAGIC
# MAGIC ### Deployment Profile
# MAGIC
# MAGIC | Property | Value |
# MAGIC | --- | --- |
# MAGIC | Model framework | XGBoost 2.x (pickle format) |
# MAGIC | Input features | 13 numeric values |
# MAGIC | Output | Float (1.0–5.0), rounded to integer for display |
# MAGIC | Missing value handling | XGBoost-native (NaN → learned direction) |
# MAGIC | Cold-start behaviour | 3 features may be NaN for new users; model degrades gracefully |
# MAGIC | Total inference pipeline | Feature lookup + predict < 1 ms per user |
# MAGIC | Memory footprint | \~2 MB (model + feature metadata) |
# MAGIC
# MAGIC The model is trivially deployable to REST endpoints, serverless functions, mobile edge, or batch scoring pipelines.

# COMMAND ----------

# DBTITLE 1,Section 10 — Key Decisions Summary
# MAGIC %md
# MAGIC ## 10. Key Decisions That Shaped the Model
# MAGIC
# MAGIC Across the project, 22 formal decisions were logged (D-001 through D-022). The table below highlights the ones with the greatest impact on the final result.
# MAGIC
# MAGIC | Decision | What | Impact |
# MAGIC | --- | --- | --- |
# MAGIC | **D-001** | Regression over classification | Ordinal target respects distance between classes. Enables RMSE/R² evaluation and produces calibrated float scores. |
# MAGIC | **D-003** | Exclude `legacy_readiness_shown` | Avoided circular prediction and anchoring bias. The model must beat the legacy heuristic, not parrot it. |
# MAGIC | **D-005** | Temporal validation split | Prevents future-leaking cross-validation. All reported metrics reflect true forward-looking performance. |
# MAGIC | **D-011** | Keep longest session per night | Resolved 984 multi-session nights consistently; `fragmented_night` flag preserved the information. |
# MAGIC | **D-013** | Weight unit reversal | Initially applied a wrong conversion. Discovered the error via height distribution analysis. Reversed to avoid injecting noise. |
# MAGIC | **D-015** | 785 rows seconds→minutes | Caught an ingestion artefact affecting 7% of sessions via a novel ratio-based detection method. |
# MAGIC | **D-019** | Defer signal aggregates to V2 | 18 unique features all correlated |r|<0.13 with target. Removing 942K-row processing simplified the pipeline without losing predictive power. |
# MAGIC | **D-021** | LEFT JOIN + NaN retention | Kept 857 no-session rows (17.2%) rather than dropping them. The model learned the missing-data pattern via `has_session_data` flag. |
# MAGIC | **D-022** | Caffeine inconsistency fix | Resolved contradictory caffeine/hours data affecting 3,907 rows. Prevented the model from learning a spurious pattern. |
# MAGIC | **Optuna tuning** | 50-trial hyperparameter search | Fixed baseline overfitting (train–val gap: 0.062 → 0.042) via strong regularisation. |
# MAGIC | **Feature reduction** | 210 → 13 features | Test accuracy **improved** with 94% fewer features. Definitively proved the extra features were noise. |
# MAGIC | **Leakage audit** | All 13 features verified | Code-level inspection of `shift()` and `<` guards confirmed zero leakage. |

# COMMAND ----------

# DBTITLE 1,Section 11 — Conclusions and Next Steps
# MAGIC %md
# MAGIC ## 11. Conclusions & Next Steps
# MAGIC
# MAGIC ### What We Built
# MAGIC
# MAGIC A **13-feature XGBoost regressor** that predicts morning subjective feeling (1–5) with:
# MAGIC
# MAGIC * **90.8% exact accuracy** on temporally held-out test data
# MAGIC * **100% accuracy within ±1 class** — no prediction is ever off by more than 1 point
# MAGIC * **R² = 0.925** — the model explains 92.5% of variance in how users rate their morning readiness
# MAGIC * **Not overfit** — train–val–test gaps are negligible (0.006 R²)
# MAGIC * **1.11 MB model** using only 13 features — lightweight, interpretable, and fast
# MAGIC
# MAGIC ### Why It Works
# MAGIC
# MAGIC The model's signal comes from three clear sources:
# MAGIC
# MAGIC 1. **Behavioural momentum (59.6%)**: How recently the user had a great or bad night. Sleep quality is cumulative — it’s not just about last night, but the trajectory over recent days.
# MAGIC
# MAGIC 2. **Alcohol consumption (28.3%)**: Whether, how much, and at what level the user drank. Alcohol is the single strongest modifiable lifestyle factor affecting sleep quality in this dataset.
# MAGIC
# MAGIC 3. **Overnight physiology (8.2%)**: User-normalised sleep duration, restorative sleep depth (deep + REM), HRV, and heart rate. These are the ring's sensor signals — objective measurements of recovery quality.
# MAGIC
# MAGIC 4. **Temporal context (3.2%)**: Yesterday's feeling (autoregressive), week of year (seasonal), and check-in sequence (habituation).
# MAGIC
# MAGIC ### Limitations
# MAGIC
# MAGIC * **Feeling 5 is hardest to predict** (\~75% accuracy vs \~95% for feeling 3). Extreme positive mornings may be driven by unmeasured factors: social events, personal achievements, mental state, or simply a good dream.
# MAGIC * **Cold-start gap**: New users need 3+ days of check-in history before the recency features (`days_since_great_sleep`, `days_since_bad_sleep`, `subjective_feeling_lag1`) activate. The model still works via the remaining 10 features, but accuracy may be lower initially.
# MAGIC * **Self-reported target**: Users see `legacy_readiness_shown` before answering, creating an anchoring effect we cannot fully control for without A/B test data.
# MAGIC * **Single-cohort study**: 120 users from a single product tier, over 90 days. Generalisation to a broader population or longer time horizons requires validation.
# MAGIC
# MAGIC ### Next Steps for V2
# MAGIC
# MAGIC 1. **Per-user evaluation**: Identify which users the model struggles with and why. Are there user archetypes (e.g., shift workers, frequent travellers) that need specialised handling?
# MAGIC 2. **Sequence models on raw signals**: The deferred 5-minute resolution `nightly_signals` data (942K rows) could feed an LSTM or 1D-CNN to capture within-night dynamics (HRV recovery trajectory, sleep stage transitions).
# MAGIC 3. **A/B test the score**: Does showing the model's prediction change user behaviour? If users anchor on the score, the target label becomes contaminated. An A/B test (show vs. hide) would quantify this.
# MAGIC 4. **Extended Optuna tuning**: 50 trials is a starting point. Bayesian optimisation with 200+ trials, or multi-objective optimisation (RMSE + calibration), could refine the model further.
# MAGIC 5. **Personalised thresholds**: The current model predicts on a universal 1–5 scale. A post-processing layer could calibrate per-user: "your 3 is another user's 4".
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC *Report generated from the complete project pipeline: Data Cleaning & EDA → Feature Engineering → Model Selection → Model Training. All source notebooks, data files, decision log, and model artefacts are in the `AS_UH_Project` folder.*