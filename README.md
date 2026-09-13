# Ring AI: Readiness Score Prediction Engine

An end-to-end machine learning system and production API that predicts subjective morning recovery (1–5 scale) from smart ring overnight sensor telemetry and daytime habits.

---

## Executive Summary

| Dimension | Specification / Result |
|---|---|
| **Objective** | Predict subjective recovery feeling (1 to 5) before morning check-in using overnight ring telemetry and daytime habits. |
| **Model** | Tuned 21-feature **XGBoost Regressor** (`xgboost_tuned_reduced.pkl`, depth=3, 1,699 trees, colsample=0.46). |
| **Holdout Metrics** | **Test RMSE: 0.556** (vs 1.016 baseline — **45.3% error reduction**) \| **Test $R^2$: 0.6989** (Val $R^2$: 0.7052, Train $R^2$: 0.7672). |
| **Accuracy** | **65.17% exact class match** \| **98.50% within $\pm 1$ class** (zero severe outliers). |
| **Overfit Gap** | **0.0619** ($	ext{Train } R^2 - 	ext{Val } R^2$), confirming strong generalization under temporal drift. |
| **Inference Latency** | **< 10 ms** per request with native Booster TreeSHAP marginal attribution. |
| **Deployment** | FastAPI service (`run_api.py`) + Interactive Ultrahuman Simulator UI (1-page categorized layout) + Automated model versioning & rollback. |

> [!NOTE]
> **Comprehensive Development & Modeling Report:**  
> For detailed understanding of the data cleaning journey, hardware anomaly resolutions, 260-to-21 feature reduction, Optuna hyperparameter optimization, and the sequential decision history (D-001 to D-022), refer to [`MODEL_TRAINING_REPORT.md`](model_training_notebooks/MODEL_TRAINING_REPORT.md).

---

## 1. Quickstart

### Setup & Pipeline Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Run the pipeline to clean data, engineer 21 features, and validate the model artifact (~3s)
python run_pipeline.py

# 2. Run automated test suite (30 unit & integration tests)
pytest -v
```
*(Requires the 5 raw CSVs in `data/` and OpenMP: `brew install libomp` on macOS).*

### Run the Service & Simulator
1. **Start the API:**
   ```bash
   python run_api.py
   ```
2. **Access Interfaces:**
   - **Interactive UI Simulator:** [http://localhost:8000/simulator](http://localhost:8000/simulator)
   - **FastAPI Interactive Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Service Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

*(Optional: `python run_simulator.py` health-checks the API in the background and opens your browser once the server is ready).*

### Research & Training Notebooks
The data science and modeling workflow is sequentially documented in [`model_training_notebooks/`](model_training_notebooks):
- `Data Cleaning and EDA.py`: Data ingestion, sentinel handling, and exploratory distributions.
- `Feature Engineering.py`: Derivation of candidate features across 4 physiological pillars.
- `Model Training.py`: Optuna Bayesian tuning, temporal validation, and feature reduction to 21 features.
- `Inference Pipeline — Readiness Score.py`: Reference production pipeline and cold-start simulation.
- `MODEL_TRAINING_REPORT.md`: Comprehensive end-to-end report covering all methodology and decision logs.

---

## 2. Data Cleaning: Hardware & Wearable Anomalies

Raw wearable telemetry has edge cases that silently corrupt model training if ingested blindly:

### Key Edge Cases & Resolutions
- **The 60x Duration Bug (Seconds vs. Minutes):** 785 sessions had durations logged in seconds instead of minutes. Rather than using an arbitrary threshold (which would miss short naps), we compared reported duration to the actual timestamp interval (`session_end - session_start`). Records with a ratio near ~60 were normalized back to minutes.
- **Mixed Timestamp Formats:** 1,731 sleep records stored timestamps as 13-digit millisecond epochs (`1772499600000`) mixed with ISO strings. Vectorized type detection parsed these without dropping data.
- **Sensor Disconnect Sentinels:** Ring disconnections produce sentinel codes (`HR = 0` or `250`, `HRV = 999`). Instead of clipping these to false physiological values, they were mapped to `NaN`, allowing tree-based splits to handle sensor absence naturally.
- **Multi-Session Nights:** 984 user-nights contained fragmented recordings or naps. We retained the longest session as the primary sleep and engineered a `fragmented_night` indicator.
- **Deduplication & Normalization:** Stripped whitespace and normalized user ID casing (collapsing 469 variants into 120 canonical users) and retained only the latest submission for duplicate check-ins.
- **Strict Leakage Prevention:** Joins anchor on morning check-ins. Context features represent the preceding day ($D-1$), and autoregressive lags/recency counters strictly reference prior days (`shift(1)`), preventing target leakage.

---

## 3. Feature Engineering: The 21 Production Features

Through extensive domain exploration and feature reduction, a 21-feature schema was distilled from 260 engineered candidate features. The features map into four physiological pillars:

### Physiological Feature Pillars

| Feature | Category | Definition / Formula | Biological Rationale |
|---|---|---|---|
| `had_alcohol` | Alcohol & Lifestyle | Binary: $1$ if `alcohol_units` $> 0$, else $0$ | Distinguishes sober nights from nights with metabolic alcohol processing. |
| `alcohol_level` | Alcohol & Lifestyle | Categorical tier: 0 (None), 1 (Light $\le 2$), 2 (Heavy $> 2$) | Captures dose-dependent degradation of nocturnal autonomic stability. |
| `alcohol_units` | Alcohol & Lifestyle | Total units consumed yesterday evening | Quantifies absolute toxic load suppressing REM sleep and elevating resting heart rate. |
| `alcohol_x_hrv_z` | Alcohol & Lifestyle | $	ext{alcohol\_units} 	imes 	ext{avg\_hrv\_zscore}$ | Interaction term: captures compound vulnerability when alcohol is consumed under low autonomic reserve. |
| `total_sleep_minutes_zscore` | Sleep Architecture | $(S - \mu_S) / \sigma_S$ (user expanding baseline) | Normalizes individual sleep requirements (e.g., 6h natural short sleepers vs 9h sleepers). |
| `deep_minutes_zscore` | Sleep Architecture | $(D - \mu_D) / \sigma_D$ (user expanding baseline) | Normalized deep sleep volume; governs physical cellular repair and growth hormone release. |
| `rem_minutes_zscore` | Sleep Architecture | $(R - \mu_R) / \sigma_R$ (user expanding baseline) | Normalized REM sleep volume; governs emotional regulation and cognitive memory consolidation. |
| `deep_rem_total` | Sleep Architecture | $	ext{deep\_minutes} + 	ext{rem\_minutes}$ | Total absolute volume of restorative sleep stages. |
| `restorative_pct` | Sleep Architecture | $(	ext{deep} + 	ext{rem}) / 	ext{total\_sleep\_minutes}$ | Restorative sleep efficiency; isolates proportion of sleep spent in restorative phases vs light/wake. |
| `sleep_debt` | Sleep Architecture | $	ext{total\_sleep\_minutes} - \mu_{S,	ext{user}}$ | Minute surplus or deficit against personal historical rolling requirement. |
| `avg_hr_bpm_zscore` | Autonomic Recovery | $(	ext{HR} - \mu_{	ext{HR}}) / \sigma_{	ext{HR}}$ | Elevated resting HR signals systemic stress, late meals, infection, or dehydration. |
| `avg_hrv_rmssd_ms_zscore` | Autonomic Recovery | $(	ext{HRV} - \mu_{	ext{HRV}}) / \sigma_{	ext{HRV}}$ | Parasympathetic tone; positive deviation strongly signals nervous system readiness. |
| `stress_index_z` | Autonomic Recovery | $	ext{HR}_{z} - 	ext{HRV}_{z}$ | Composite autonomic stress index; spikes when HR is elevated while HRV is suppressed. |
| `recovery_score` | Autonomic Recovery | $	ext{HRV}_{z} - 	ext{HR}_{z}$ | Net autonomic balance; positive when parasympathetic recovery outpaces cardiovascular strain. |
| `sleep_user_ratio` | Baseline Ratios | $	ext{total\_sleep\_minutes} / \mu_{S,	ext{user}}$ | Tonight's sleep volume relative to expanding personal historical baseline. |
| `deep_user_ratio` | Baseline Ratios | $	ext{deep\_minutes} / \mu_{D,	ext{user}}$ | Tonight's deep stage duration relative to personal historical mean. |
| `hr_user_ratio` | Baseline Ratios | $	ext{avg\_hr\_bpm} / \mu_{	ext{HR},	ext{user}}$ | Heart rate ratio relative to user baseline; values $> 1.0$ indicate incomplete cardiac recovery. |
| `hrv_user_ratio` | Baseline Ratios | $	ext{avg\_hrv\_rmssd\_ms} / \mu_{	ext{HRV},	ext{user}}$ | HRV ratio relative to user baseline; values $> 1.0$ indicate strong parasympathetic activity. |
| `feeling_roll5_mean` | Historical Feeling | 5-day backward rolling mean of `subjective_feeling` | Captures recent multi-day psychological momentum and mood trajectory. |
| `feeling_ewm_7` | Historical Feeling | Exponential weighted moving average ($lpha = 2 / (7 + 1)$) | Exponentially decays older check-ins to prioritize recent recovery experiences. |
| `user_expanding_mean` | Historical Feeling | Cumulative expanding mean of user feelings up to $D-1$ | User subjective baseline anchor; calibrates whether a user naturally rates strictly or leniently. |

**Why 21 features?**
1. **Balanced Representation:** Blends raw sleep architecture, autonomic balance ratios, lifestyle inputs, and autoregressive psychological momentum.
2. **Minimal Overfitting:** Controlled feature depth (`max_depth=3`) and feature subsampling (`colsample_bytree=0.4616`) prevent the model from memorizing individual users, keeping the overfit gap at just **0.0619**.
3. **Sub-10ms Exact TreeSHAP:** Tree structure allows zero-latency exact SHAP attribution calculation for all 21 features simultaneously via XGBoost booster matrix operations.

---

## 4. Validation & Benchmark Results

Evaluated using a chronological 70 / 15 / 15 temporal split by check-in date:
- **Train (70%):** Check-ins $\le$ 2026-04-04 (3,419 rows)
- **Validation (15%):** Check-ins 2026-04-05 to 2026-04-18 (769 rows) — used for Optuna tuning & early stopping.
- **Holdout Test (15%):** Check-ins $>$ 2026-04-18 (801 rows) — final holdout evaluation.

### Model Comparison

| Model | Train $R^2$ | Val $R^2$ | Test RMSE | Test $R^2$ | Exact Acc (%) | Acc $\pm 1$ Class (%) | Overfit Gap |
|---|---|---|---|---|---|---|---|
| **Global Mean Baseline** | 0.0000 | 0.0000 | 1.016 | 0.0000 | 38.2% | 85.6% | 0.0000 |
| **User Historical Mean** | 0.0410 | -0.0020 | 1.018 | -0.0030 | 38.2% | 86.4% | 0.0430 |
| **XGBoost (Tuned 21 Features)** | **0.7672** | **0.7052** | **0.556** | **0.6989** | **65.17%** | **98.50%** | **0.0619** |

### Key Observations
- **45.3% Error Reduction:** Test RMSE drops from 1.016 (global baseline) to **0.556**.
- **Discrete Class Reliability:** **98.50% of predictions fall within $\pm 1$ class**, with **65.17% exact matches** across the 5 discrete score tiers.
- **Strong Generalization:** The difference between Train $R^2$ (0.7672) and Test $R^2$ (0.6989) is only 0.0683, demonstrating that early stopping (best iteration: 510) and shallow tree depth (depth=3) successfully prevented overfitting on historical records.
- **Missing Sensor Resilience:** When ring sensor telemetry is absent on any given night, XGBoost's default split routing gracefully leverages daytime context and historical feeling baselines without crashing or requiring ad-hoc heuristic imputation.

---

## 5. Production Design & User Experience

### Daily Rhythm & Contextual Guidance
Displaying an ungrounded raw decimal invites confusion. The engine couples continuous predictions with actionable, user-friendly guidance:
1. **Last Night's Rest (Recovery Assessment):** Plain-language explanation of overnight autonomic recovery and restorative sleep stages (e.g., *"Rest was a bit choppy (2.86/5). 2.0 drinks kept heart rate elevated & 40m sleep deficit — your body worked harder than usual overnight."*).
2. **Today's Rhythm (Daily Pacing):** Clear, approachable activity guidance (e.g., *"Take things easy today. Stick to gentle walks or light movement, drink plenty of water, and treat yourself to an earlier bedtime tonight."*).
3. **Tonight's Quick Win + Tomorrow's Boost (Model-Linked Counterfactual):** Evaluates real-time counterfactual interventions through the active XGBoost model to project the exact score delta the user can earn tomorrow (e.g., *"Head to bed 45 mins earlier to erase sleep debt and reset energy (+0.44 pts → 3.60)"*).

### Cold-Start Strategy (Day 1 Onboarding)
When a new user begins using the ring:
- **Detection Criteria:** The system flags `is_cold_start = True` whenever a user is on Day-1 onboarding (`checkin_seq_num <= 1`) or when historical personal baselines (autoregressive feelings `feeling_roll5_mean`, `user_expanding_mean` and sensor z-scores `total_sleep_minutes_zscore`, `avg_hr_bpm_zscore`, etc.) are absent (`NaN` / `null`). Established users with accumulated personal baselines evaluate to `is_cold_start = False`.
- **Model Fallback Routing:** XGBoost natively routes missing baseline features along default learned split directions, anchoring readiness predictions safely to the population prior (~3.0–3.3 / Moderate tier).
- **Interactive UI & API Testing:**
  - **Simulator UI (`/simulator`):** Click the `❄️ Day 1 Test` button in the control deck to instantly simulate Day-1 cold start and observe the live `🟡 Day-1 Cold Start` badge. Click `↺ Reset` or adjust any slider to return to `● Baseline Active`.
  - **API Verification:**
    - Established user (`is_cold_start: false`):
      ```bash
      curl -s -X POST http://localhost:8000/features/calculate -H "Content-Type: application/json" \
        -d '{"user_id": "UH-001", "checkin_date": "2026-03-20", "total_sleep_minutes": 450, "avg_hr_bpm": 58, "avg_hrv_rmssd_ms": 55, "recent_feeling_mean": 3.5}' | jq .features.checkin_seq_num
      ```
    - Day-1 onboarding user (`is_cold_start: true`):
      ```bash
      curl -s -X POST http://localhost:8000/features/calculate -H "Content-Type: application/json" \
        -d '{"user_id": "UH-NEW", "checkin_date": "2026-03-20", "total_sleep_minutes": 420, "checkin_seq_num": 1}' | jq .features.checkin_seq_num
      ```

---

## 6. Interactive Simulator & API Endpoints

The interactive simulator (`http://localhost:8000/simulator`) provides a single-page control dashboard:
- **Left Mobile Mirror:** Ultrahuman ring dial, readiness pill, recovery guidance cards, sleep architecture breakdowns, and autonomic biomarker cards.
- **Right Control Deck (Tabbed 1-Page Layout):**
  - **Raw Data Tab:** Sliders for raw sleep minutes, deep sleep, REM sleep, resting HR, HRV, alcohol units, and past feeling baseline; automatically computes and live-syncs all 21 features.
  - **Engineered Features Tab:** Categorized into 3 sub-tabs keeping the UI clean and accessible:
    - 🌙 *Sleep & Restorative (6 features)*: duration z-score, sleep debt, restorative sleep volume, deep z-score, REM z-score, restorative percentage.
    - 💓 *Autonomic & Stress (8 features)*: resting HR z-score, HRV z-score, stress index, recovery score, and expanding baseline ratios.
    - 🍷 *Alcohol & History (7 features)*: alcohol units, alcohol tier, alcohol $\times$ HRV interaction, 5-day rolling feeling, 7-day EWM feeling, expanding historical mean.
  - **Persistent TreeSHAP Section (Vacant Space Below Sliders):** Embedded directly beneath the active sliders in the previously vacant lower space—features live mini-tabs for `Top 10` (highest positive boosters), `Last 10` (most negative drags), and `All 21` features, displaying real-time population prior bias, net TreeSHAP contribution sum, and sorted signed impact bars simultaneously as sliders are adjusted.
- **Live Pipeline Retrain Modal (`⚡ Retrain Pipeline`):** Triggers end-to-end retraining directly from the simulator navbar; executes all 4 pipeline phases with live animated progress bars and displays holdout validation cards: **Holdout RMSE**, **Holdout $R^2$**, **Exact Accuracy**, and **Acc $\pm 1$ Class**.

### Key API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health, active model path, 21 feature list, and rollback versions. |
| `POST` | `/inference/predict` | Single prediction with cold-start detection and model-linked recommendations. |
| `POST` | `/inference/explain` | Real-time prediction with exact 21-feature TreeSHAP attributions. |
| `POST` | `/features/calculate` | Transform raw sensor metrics into the 21 engineered model features. |
| `GET` | `/model/versions` | List archived model checkpoints (`selected_model_vX.X.pkl`). |
| `POST` | `/model/rollback` | Roll back active model to any prior version. |
| `POST` | `/train` | Retrain from raw data with automatic version archiving. |

---

## 7. What Would I Do With Two More Weeks?

1. **Deeper Domain & Feature Engineering:** Spend more time researching sleep science and chronobiology to engineer richer domain features—such as circadian alignment (social jetlag via sleep midpoint variance), multi-day sleep debt decay curves, Sleep Regularity Index (SRI), and pre-bed meal/alcohol cutoff timing.
2. **Sequential Deep Learning on 5-Min Telemetry:** Leverage the granular 5-minute sensor streams (`nightly_signals.csv.gz`) with sequence models (1D-CNNs, BiLSTMs, or TCNs) to capture overnight heart-rate dip curvature, HRV recovery trajectories, and sleep stage transitions.
3. **LLM-Driven Personalized Recommendations:** Integrate an LLM API conditioned on user history, current telemetry, TreeSHAP feature drivers, and model counterfactual deltas to deliver empathetic, context-aware daily coaching tailored to the user's personal routine.
4. **Expanded Pipeline Testing & AI Code Verification:** Conduct deeper manual sanity checks and build broader edge-case test suites across the data and feature pipelines to thoroughly stress-test boundary conditions and establish total confidence in all AI-assisted code implementations.

---

## 8. AI Assistance Disclosure

AI was utilized as an accelerator for clerical, operational, and engineering tasks:
- **Code Generation & Boilerplate:** Drafting initial code scaffolding, function signatures, type annotations, and UI simulator styling boilerplate.
- **Code Optimization:** Refactoring procedural code into modular components and identifying vectorized Pandas/NumPy execution patterns.
- **Documentation & Technical Summaries:** Formatting markdown tables, summarizing API endpoint schemas, and organizing decision logs.
- **Workflow Planning & Ideation:** Structuring documentation outlines and organizing experiment checklists.
- **Debugging & Error Resolution:** Assisting with package version conflicts, environment setup, and resolving runtime tracebacks.
