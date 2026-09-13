# Ring AI: Readiness Score Prediction Engine

An end-to-end machine learning system and production API that predicts subjective morning recovery (1–5 scale) from smart ring overnight sensor telemetry and daytime habits.

---

## Executive Summary

| Dimension | Specification / Result |
|---|---|
| **Objective** | Predict subjective recovery feeling (1 to 5) before morning check-in using overnight ring telemetry and daytime habits. |
| **Model** | Tuned 13-feature **XGBoost Regressor** with early stopping. |
| **Holdout Metrics** | **Test RMSE: 0.286** (vs 1.018 user baseline — **71.9% error reduction**) \| **Test $R^2$: 0.921**. |
| **Accuracy** | **89.5% exact class match** \| **100.0% within $\pm 1$ class** (zero errors $\ge 2$ classes). |
| **Inference Latency** | **< 15 ms** per request with exact TreeSHAP marginal attribution. |
| **Deployment** | FastAPI service (`run_api.py`) + Interactive Ultrahuman Simulator UI + Automated fallback circuit breaker. |

---

## 1. Quickstart

### Setup & Pipeline Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Run the pipeline first to clean data, engineer features, and train/generate the model artifact (~3s)
python run_pipeline.py

# 2. Run automated test suite (29 unit & integration tests)
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

## 3. Feature Selection: The 13 Production Features

A compact 13-feature schema was selected over the full 57-feature candidate pool:

| Feature | Category | Rationale |
|---|---|---|
| `alcohol_units` | Behavioural | Toxic dose suppressing REM sleep and elevating nocturnal resting heart rate. |
| `had_alcohol` | Behavioural | Binary indicator distinguishing zero intake from any alcohol consumption. |
| `alcohol_level` | Behavioural | Categorical tier: 0 (None), 1 (Light $\le 2$ units), 2 (Heavy $> 2$ units). |
| `week_of_year` | Temporal | Controls for seasonal fatigue, holiday clusters, and circadian shifts. |
| `total_sleep_minutes_zscore` | Normalized Sensor | User-standardized sleep duration; normalizes short vs long sleepers. |
| `avg_hr_bpm_zscore` | Normalized Sensor | Elevated resting HR reflects autonomic strain or recovery debt. |
| `avg_hrv_rmssd_ms_zscore` | Normalized Sensor | Parasympathetic tone; positive HRV deviation strongly correlates with recovery. |
| `subjective_feeling_lag1` | Autoregressive | Previous morning's feeling; captures psychological momentum and baseline mood. |
| `days_since_bad_sleep` | Event Recency | Days elapsed since feeling $\le 2$; tracks cumulative fatigue compounding. |
| `days_since_great_sleep` | Event Recency | Days elapsed since feeling $\ge 4$; tracks sustained recovery streaks. |
| `checkin_seq_num` | Behavioural | User engagement count; controls for app onboarding novelty effects. |
| `deep_rem_total` | Sleep Architecture | Total minutes of restorative sleep stages ($\text{Deep} + \text{REM}$). |
| `sleep_debt` | Normalized Sensor | Minutes of sleep surplus or deficit relative to the user's historical norm. |

**Why 13 features?**
1. **Identical Accuracy:** Test $R^2$ with 13 features (`0.921`) matched the full 57-feature baseline (`0.917`).
2. **Missing-Ring Resilience:** When a user skips wearing the ring (`has_session_data = 0`), the model falls back cleanly on daytime context and momentum without missing collinear features.
3. **Sub-4ms TreeSHAP:** Low dimensionality allows real-time exact TreeSHAP attribution on every request.

---

## 4. Validation & Benchmark Results

Evaluated using a chronological 70 / 15 / 15 temporal split by check-in date:
- **Train (70%):** Check-ins $\le$ 2026-04-04 (3,419 rows)
- **Validation (15%):** Check-ins 2026-04-05 to 2026-04-18 (769 rows) — used for Optuna tuning & early stopping.
- **Holdout Test (15%):** Check-ins $>$ 2026-04-18 (801 rows) — final holdout evaluation.

### Model Comparison

| Model | Train RMSE | Val RMSE | Test RMSE | Test $R^2$ | Exact Acc (%) | Acc $\pm 1$ Class (%) |
|---|---|---|---|---|---|---|
| **Global Mean Baseline** | 1.018 | 1.014 | 1.016 | 0.000 | 38.2% | 85.6% |
| **User Historical Mean** | 0.985 | 1.012 | 1.018 | -0.003 | 38.2% | 86.4% |
| **Random Forest (500 Trees)** | 0.412 | 0.385 | 0.379 | 0.860 | 79.4% | 98.9% |
| **LightGBM Regressor** | 0.285 | 0.312 | 0.306 | 0.909 | 86.8% | 99.8% |
| **XGBoost (Full 57 Features)** | 0.224 | 0.301 | 0.292 | 0.917 | 88.5% | 100.0% |
| **XGBoost (Selected 13 Features)** | **0.285** | **0.294** | **0.286** | **0.921** | **89.5%** | **100.0%** |

### Key Observations
- **71.9% Error Reduction:** Test RMSE drops from 1.018 (user baseline) to **0.286**.
- **Bound Guarantees:** 100.0% of predictions are within $\pm 1$ class; zero predictions deviate by $\ge 2$ classes.
- **Generalization:** Train $R^2$ (`0.920`) vs Val $R^2$ (`0.920`) confirms strong regularization.
- **Sensor-Off Nights:** RMSE is **0.287** when the ring was not worn vs **0.285** with full sensor telemetry, demonstrating robust fallback on daytime context.

---

## 5. Production Design & User Experience

### UX: Category Tiers over Raw Numbers
Displaying a raw 1–5 score invites subjective friction. Continuous model predictions are mapped to qualitative tiers with actionable advice:
- **Optimal (4.0–5.0):** Strong autonomic recovery. Recommended for peak training load.
- **Moderate (2.6–3.9):** Baseline recovery. Suitable for standard daily demands.
- **Recovery (1.0–2.5):** Elevated resting HR or sleep deficit. Prioritize hydration and light restorative movement.

### Cold-Start Strategy (Day 1–3)
When historical user baselines ($\mu_u, \sigma_u$) are unavailable:
- The service flags `is_cold_start = True` and substitutes demographic cohort medians (`age`, `sex`).
- The UI communicates baseline calibration status to the user. Rolling personalization activates on Day 4.

### Drift Monitoring & Fallback Circuit Breaker
- **Telemetry Drift:** Population Stability Index (PSI) tracks sensor stream shifts (`avg_hr_bpm`, `avg_hrv_rmssd_ms`).
- **Kill-Switch:** If PSI breaches 0.25 or service errors exceed 0.5%, an automated circuit breaker routes inference to the legacy heuristic score, alerting the on-call engineer.

---

## 6. Interactive Simulator & API Endpoints

The interactive simulator (`http://localhost:8000/simulator`) offers:
- **Left Column:** Ultrahuman mobile readiness widget (radial dial, readiness pill, recovery guidance card, sensor summary).
- **Right Column (Tab-Driven):**
  - **Raw Telemetry Tab:** Sliders for raw sleep minutes, resting HR, HRV, and alcohol; auto-calculates engineered features.
  - **Engineered Features Tab:** Direct control over z-scores, sleep debt, and lags.
  - **TreeSHAP Tab:** Real-time marginal attribution bars showing positive and negative drivers of the prediction.

### Key API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health, active model status, and available versions. |
| `POST` | `/inference/predict` | Single prediction with cold-start detection. |
| `POST` | `/inference/explain` | Real-time prediction with exact TreeSHAP contributions. |
| `POST` | `/features/calculate` | Transform raw sensor metrics into model features. |
| `GET` | `/model/versions` | List archived model checkpoints. |
| `POST` | `/model/rollback` | Roll back active model to any prior version. |
| `POST` | `/train` | Retrain from raw data with automatic version archiving. |

---

## 7. What Would I Do With Two More Weeks?

1. **Deeper Domain & Feature Engineering:** Spend more time researching sleep science and chronobiology to engineer richer domain features—such as circadian alignment (social jetlag via sleep midpoint variance), multi-day sleep debt decay curves, Sleep Regularity Index (SRI), and pre-bed meal/alcohol cutoff timing.
2. **Sequential Deep Learning on 5-Min Telemetry:** Leverage the granular 5-minute sensor streams (`nightly_signals.csv.gz`) with sequence models (1D-CNNs, BiLSTMs, or TCNs) to capture overnight heart-rate dip curvature, HRV recovery trajectories, and sleep stage transitions.
3. **LLM-Driven Personalized Recommendations:** Integrate an LLM API conditioned on user history, current telemetry, TreeSHAP feature drivers, and model counterfactual deltas to deliver empathetic, context-aware daily coaching tailored to the user's personal routine.

---

## 8. AI Assistance Disclosure

AI was utilized as an accelerator for clerical, operational, and engineering tasks:
- **Code Generation & Boilerplate:** Drafting initial code scaffolding, function signatures, type annotations, and UI simulator styling boilerplate.
- **Code Optimization:** Refactoring procedural code into modular components and identifying vectorized Pandas/NumPy execution patterns.
- **Documentation & Technical Summaries:** Formatting markdown tables, summarizing API endpoint schemas, and organizing decision logs.
- **Workflow Planning & Ideation:** Structuring documentation outlines and organizing experiment checklists.
- **Debugging & Error Resolution:** Assisting with package version conflicts, environment setup, and resolving runtime tracebacks.
