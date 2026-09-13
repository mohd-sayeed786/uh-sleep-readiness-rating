"""
FastAPI application providing:
  1. /train: Model retraining pipeline with configurable parameters, custom feature subsets, and version archiving
  2. /model/versions & /model/rollback: Version control and instant rollback management
  3. /features/calculate: Feature extraction service from raw data
  4. /inference/predict & /inference/predict-batch: Single and batch prediction endpoints
  5. /inference/explain: Real-time TreeSHAP feature contribution and readiness breakdown
  6. /simulator: Interactive Ultrahuman UI simulator with live controls and SHAP visualization
  7. /tests/run: Test execution and verification route for CI/CD sanity checks
"""
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_XGB_PARAMS,
    FEATURE_LIST_PATH,
    FEATURE_NAMES,
    MODEL_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODEL_VERSIONS_DIR,
    RANDOM_SEED,
    TARGET_COL,
)
from src.data_pipeline import (
    clean_checkins,
    clean_daily_context,
    clean_profiles,
    clean_sleep_sessions,
    build_modelling_table,
    run_data_pipeline,
)
from src.features import engineer_features
from src.inference import ReadinessPredictor, get_predictor
from src.logger import get_logger
from src.simulator_ui import SIMULATOR_HTML
from src.train import rollback_model, train_model

logger = get_logger("api")

app = FastAPI(
    title="Ring AI - Readiness Score API",
    description="Production-ready REST API for subjective sleep readiness prediction, feature generation, model versioning, and automated testing.",
    version="1.2.0",
)


# ---------------------------------------------------------------------------
# Pydantic Request / Response Models
# ---------------------------------------------------------------------------

class TrainParams(BaseModel):
    max_depth: Optional[int] = Field(default=8, ge=2, le=16)
    learning_rate: Optional[float] = Field(default=0.014, gt=0.0, le=1.0)
    n_estimators: Optional[int] = Field(default=900, ge=50, le=3000)
    subsample: Optional[float] = Field(default=0.75, gt=0.0, le=1.0)
    colsample_bytree: Optional[float] = Field(default=0.82, gt=0.0, le=1.0)
    reg_alpha: Optional[float] = Field(default=0.0038, ge=0.0)
    reg_lambda: Optional[float] = Field(default=0.572, ge=0.0)
    save_model: bool = Field(default=True, description="Whether to archive old model and write new selected_model.pkl")
    feature_cols: Optional[List[str]] = Field(default=None, description="Optional custom feature names to train on. Defaults to standard 13 features.")


class SingleFeatureInput(BaseModel):
    alcohol_units: Optional[float] = Field(default=0.0, description="Alcohol units consumed yesterday")
    had_alcohol: Optional[float] = Field(default=0.0, description="Binary flag (1.0 if alcohol units > 0)")
    alcohol_level: Optional[float] = Field(default=0.0, description="Ordinal: 0 (none), 1 (light <= 2), 2 (heavy > 2)")
    week_of_year: Optional[int] = Field(default=15, ge=1, le=53, description="Calendar ISO week")
    total_sleep_minutes_zscore: Optional[float] = Field(default=0.0, description="User z-score for total sleep minutes")
    avg_hr_bpm_zscore: Optional[float] = Field(default=0.0, description="User z-score for resting heart rate")
    avg_hrv_rmssd_ms_zscore: Optional[float] = Field(default=0.0, description="User z-score for HRV")
    subjective_feeling_lag1: Optional[float] = Field(default=3.0, ge=1.0, le=5.0, description="Yesterday's feeling (1-5)")
    days_since_bad_sleep: Optional[float] = Field(default=None, description="Days since last feeling <= 2")
    days_since_great_sleep: Optional[float] = Field(default=None, description="Days since last feeling >= 4")
    checkin_seq_num: Optional[int] = Field(default=10, ge=1, description="Cumulative check-in count")
    deep_rem_total: Optional[float] = Field(default=120.0, ge=0.0, description="Deep + REM sleep in minutes")
    sleep_debt: Optional[float] = Field(default=0.0, description="Minutes sleep deficit vs user mean")


class BatchFeatureInput(BaseModel):
    records: List[SingleFeatureInput]


class RawContextInput(BaseModel):
    user_id: str
    checkin_date: str
    alcohol_units: Optional[float] = 0.0
    caffeine_mg: Optional[float] = None
    last_caffeine_hours_before_bed: Optional[float] = None
    workout_notes: Optional[str] = None
    total_sleep_minutes: Optional[float] = None
    deep_minutes: Optional[float] = None
    rem_minutes: Optional[float] = None
    avg_hr_bpm: Optional[float] = None
    avg_hrv_rmssd_ms: Optional[float] = None
    # User historical baselines (for single-record feature synthesis)
    user_mean_sleep: Optional[float] = 420.0
    user_std_sleep: Optional[float] = 60.0
    user_mean_hr: Optional[float] = 62.0
    user_std_hr: Optional[float] = 5.0
    user_mean_hrv: Optional[float] = 45.0
    user_std_hrv: Optional[float] = 12.0
    subjective_feeling_lag1: Optional[float] = 3.0
    days_since_bad_sleep: Optional[float] = None
    days_since_great_sleep: Optional[float] = None
    checkin_seq_num: Optional[int] = 1


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Root endpoint detailing API status and capabilities."""
    return {
        "service": "Ring AI Readiness Score API",
        "version": "1.2.0",
        "active_model": str(MODEL_PATH),
        "versions_directory": str(MODEL_VERSIONS_DIR),
        "description": "Predicts subjective morning recovery score (1-5) using ring sensors and waking context.",
        "endpoints": {
            "simulator": "GET /simulator (Interactive Web App)",
            "health": "/health",
            "model_versions": "GET /model/versions",
            "model_rollback": "POST /model/rollback?version=v1.0",
            "train": "POST /train",
            "calculate_features": "POST /features/calculate",
            "predict_single": "POST /inference/predict",
            "explain_single": "POST /inference/explain",
            "predict_batch": "POST /inference/predict-batch",
            "run_tests": "POST /tests/run",
        }
    }


@app.get("/simulator", response_class=HTMLResponse)
def get_simulator_ui():
    """
    Ultrahuman-styled interactive UI Simulator for real-time model controls and SHAP value explainability.
    """
    return HTMLResponse(content=SIMULATOR_HTML, status_code=200)


@app.get("/health")
def health_check():
    """Service health check, active model status, and available versions."""
    model_exists = MODEL_PATH.exists()
    metadata = {}
    if MODEL_METADATA_PATH.exists():
        try:
            with open(MODEL_METADATA_PATH, "r") as f:
                metadata = json.load(f)
        except Exception:
            pass

    # Read active features
    active_features = FEATURE_NAMES
    if FEATURE_LIST_PATH.exists():
        try:
            with open(FEATURE_LIST_PATH, "r") as f:
                active_features = json.load(f)
        except Exception:
            pass

    pattern = re.compile(r"selected_model_(v\d+\.\d+)\.pkl")
    archived_versions = sorted(list(set(
        m.group(1)
        for search_dir in [MODEL_VERSIONS_DIR, MODEL_DIR]
        if search_dir.exists()
        for p in search_dir.glob("selected_model_v*.pkl")
        for m in [pattern.search(p.name)]
        if m
    )))

    return {
        "status": "healthy",
        "active_model": str(MODEL_PATH),
        "model_loaded": model_exists,
        "feature_count": len(active_features),
        "features": active_features,
        "versions_directory": str(MODEL_VERSIONS_DIR),
        "available_rollback_versions": archived_versions,
        "model_info": {
            "type": metadata.get("model_type", "XGBRegressor"),
            "best_iteration": metadata.get("best_iteration"),
            "test_rmse": metadata.get("metrics", {}).get("test", {}).get("rmse"),
            "test_r2": metadata.get("metrics", {}).get("test", {}).get("r2"),
            "exact_accuracy": metadata.get("metrics", {}).get("test", {}).get("exact_accuracy"),
        }
    }


@app.get("/model/versions")
def list_model_versions():
    """List all archived model versions stored in the versions directory for rollback."""
    pattern = re.compile(r"selected_model_(v\d+\.\d+)\.pkl")
    versions = []
    seen = set()

    for search_dir in [MODEL_VERSIONS_DIR, MODEL_DIR]:
        if not search_dir.exists():
            continue
        for p in sorted(search_dir.glob("selected_model_v*.pkl")):
            m = pattern.search(p.name)
            if m:
                ver = m.group(1)
                if ver in seen:
                    continue
                seen.add(ver)

                meta_file = search_dir / f"model_metadata_{ver}.json"
                feat_file = search_dir / f"feature_list_{ver}.json"

                test_rmse = None
                if meta_file.exists():
                    try:
                        with open(meta_file, "r") as f:
                            test_rmse = json.load(f).get("metrics", {}).get("test", {}).get("rmse")
                    except Exception:
                        pass

                ver_features = []
                if feat_file.exists():
                    try:
                        with open(feat_file, "r") as f:
                            ver_features = json.load(f)
                    except Exception:
                        pass

                versions.append({
                    "version": ver,
                    "filename": p.name,
                    "file_path": str(p),
                    "feature_count": len(ver_features),
                    "features": ver_features,
                    "test_rmse": test_rmse,
                })

    return {
        "active_model": "selected_model.pkl",
        "versions_directory": str(MODEL_VERSIONS_DIR),
        "archived_versions": versions,
        "count": len(versions),
    }


@app.post("/model/rollback")
def rollback_model_endpoint(version: str = Query(..., description="Version to restore, e.g. v1.0 or v1.1")):
    """
    Rollback active selected_model.pkl and its feature_list.json to a previously archived version without data loss.
    """
    logger.info(f"API request received: rollback active model to version {version}")
    try:
        res = rollback_model(version, model_dir=MODEL_DIR, versions_dir=MODEL_VERSIONS_DIR)

        # Reset predictor singleton so it loads the restored model and feature set
        global _default_predictor
        _default_predictor = None

        return res
    except FileNotFoundError as e:
        logger.error(f"Rollback version not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Rollback error: {e}")
        raise HTTPException(status_code=500, detail=f"Rollback failed: {str(e)}")


@app.post("/train")
def train_endpoint(params: TrainParams):
    """
    Train a new model using the five raw CSV files in data/:
      1. Ingests and cleans sleep_sessions, user_profiles, daily_context, morning_checkins
      2. Performs feature engineering and temporal 70/15/15 train/val/test split
      3. Trains XGBoost regressor with specified parameters and optional custom feature_cols
      4. Archives existing active model and feature_list into selected_model/versions/
      5. Saves newly trained model as active selected_model/selected_model.pkl and updates feature_list.json
    """
    logger.info(f"API request received: initiate training (n_est={params.n_estimators}, max_depth={params.max_depth}, lr={params.learning_rate})")
    try:
        custom_params = {
            "max_depth": params.max_depth,
            "learning_rate": params.learning_rate,
            "n_estimators": params.n_estimators,
            "subsample": params.subsample,
            "colsample_bytree": params.colsample_bytree,
            "reg_alpha": params.reg_alpha,
            "reg_lambda": params.reg_lambda,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        }

        results = train_model(
            custom_params=custom_params,
            feature_cols=params.feature_cols,
            save_artifacts=params.save_model,
            versions_dir=MODEL_VERSIONS_DIR,
        )

        # Reset predictor singleton to pick up newly trained active model and feature list
        global _default_predictor
        _default_predictor = None

        logger.info(f"Training completed successfully: test RMSE={results['metrics']['test']['rmse']}, exact_acc={results['metrics']['test']['exact_accuracy']*100:.1f}%")
        return {
            "status": "success",
            "message": "Model trained and saved as active selected_model.pkl",
            "active_model": str(MODEL_PATH),
            "feature_count": results["n_features"],
            "features_used": results["features"],
            "versions_directory": str(MODEL_VERSIONS_DIR),
            "previous_model_archived_as": results.get("previous_model_archived_as"),
            "training_results": results
        }
    except Exception as e:
        logger.error(f"Training endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/features/calculate")
def calculate_features(raw: RawContextInput):
    """
    Feature service: Computes the 13 required readiness features from raw sensor & context inputs.
    """
    logger.debug(f"API request received: calculate features for user {raw.user_id} on {raw.checkin_date}")
    try:
        checkin_dt = pd.to_datetime(raw.checkin_date)
        night_dt = checkin_dt - pd.Timedelta(days=1)
        week_of_year = int(night_dt.isocalendar().week)

        # Alcohol
        alcohol_units = raw.alcohol_units if raw.alcohol_units is not None else 0.0
        had_alcohol = 1.0 if alcohol_units > 0 else 0.0
        alcohol_level = 0.0 if alcohol_units <= 0 else (1.0 if alcohol_units <= 2.0 else 2.0)

        # Sleep z-score
        if raw.total_sleep_minutes is not None:
            sleep_std = max(raw.user_std_sleep or 60.0, 0.01)
            sleep_zscore = (raw.total_sleep_minutes - (raw.user_mean_sleep or 420.0)) / sleep_std
            sleep_debt = raw.total_sleep_minutes - (raw.user_mean_sleep or 420.0)
        else:
            sleep_zscore = np.nan
            sleep_debt = np.nan

        # HR z-score
        if raw.avg_hr_bpm is not None:
            hr_std = max(raw.user_std_hr or 5.0, 0.01)
            hr_zscore = (raw.avg_hr_bpm - (raw.user_mean_hr or 62.0)) / hr_std
        else:
            hr_zscore = np.nan

        # HRV z-score
        if raw.avg_hrv_rmssd_ms is not None:
            hrv_std = max(raw.user_std_hrv or 12.0, 0.01)
            hrv_zscore = (raw.avg_hrv_rmssd_ms - (raw.user_mean_hrv or 45.0)) / hrv_std
        else:
            hrv_zscore = np.nan

        # Restorative deep + REM
        deep = raw.deep_minutes or 0.0
        rem = raw.rem_minutes or 0.0
        deep_rem_total = deep + rem if (raw.deep_minutes is not None or raw.rem_minutes is not None) else np.nan

        features_dict = {
            "alcohol_units": float(alcohol_units),
            "had_alcohol": float(had_alcohol),
            "alcohol_level": float(alcohol_level),
            "week_of_year": week_of_year,
            "total_sleep_minutes_zscore": None if np.isnan(sleep_zscore) else round(float(sleep_zscore), 4),
            "avg_hr_bpm_zscore": None if np.isnan(hr_zscore) else round(float(hr_zscore), 4),
            "avg_hrv_rmssd_ms_zscore": None if np.isnan(hrv_zscore) else round(float(hrv_zscore), 4),
            "subjective_feeling_lag1": raw.subjective_feeling_lag1,
            "days_since_bad_sleep": raw.days_since_bad_sleep,
            "days_since_great_sleep": raw.days_since_great_sleep,
            "checkin_seq_num": raw.checkin_seq_num,
            "deep_rem_total": None if np.isnan(deep_rem_total) else round(float(deep_rem_total), 1),
            "sleep_debt": None if np.isnan(sleep_debt) else round(float(sleep_debt), 1),
        }

        return {
            "user_id": raw.user_id,
            "checkin_date": raw.checkin_date,
            "features": features_dict
        }
    except Exception as e:
        logger.error(f"Feature calculation error: {e}")
        raise HTTPException(status_code=400, detail=f"Feature calculation error: {str(e)}")


@app.post("/inference/predict")
def predict_single_endpoint(input_data: SingleFeatureInput):
    """
    Inference endpoint: Accepts the 13 calculated features and outputs the predicted readiness score.
    """
    logger.debug("API request received: single prediction")
    try:
        predictor = get_predictor()
        feat_dict = input_data.model_dump()
        result = predictor.predict_single(feat_dict)
        return result
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/inference/explain")
def explain_single_endpoint(input_data: SingleFeatureInput):
    """
    SHAP explainability endpoint: Computes readiness prediction and exact TreeSHAP feature contributions.
    """
    logger.debug("API request received: SHAP explanation")
    try:
        predictor = get_predictor()
        feat_dict = input_data.model_dump()
        result = predictor.explain_single(feat_dict)
        return result
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"SHAP explanation error: {str(e)}")


@app.post("/inference/predict-batch")
def predict_batch_endpoint(batch_input: BatchFeatureInput):
    """
    Batch inference endpoint: Accepts a list of feature dictionaries and returns readiness scores.
    """
    logger.debug(f"API request received: batch prediction for {len(batch_input.records)} records")
    try:
        predictor = get_predictor()
        records = [rec.model_dump() for rec in batch_input.records]
        results = predictor.predict_batch(records)
        return {
            "batch_size": len(results),
            "predictions": results
        }
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


@app.post("/tests/run")
def run_tests_endpoint(test_suite: str = Query(default="all", enum=["all", "cleaning", "features", "pipeline", "inference"])):
    """
    Execute pytest test suites for automated sanity checks and model validation.
    """
    logger.info(f"API request received: run test suite '{test_suite}'")
    test_files_map = {
        "cleaning": "tests/test_cleaning.py",
        "features": "tests/test_features.py",
        "pipeline": "tests/test_pipeline.py",
        "inference": "tests/test_inference.py",
    }

    if test_suite == "all":
        selected_targets = list(test_files_map.values())
    else:
        selected_targets = [test_files_map[test_suite]]

    class PytestCapturePlugin:
        def __init__(self):
            self.reports = []

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or (report.failed and report.when in ["setup", "teardown"]):
                self.reports.append({
                    "nodeid": report.nodeid,
                    "test_name": report.nodeid.split("::")[-1],
                    "outcome": report.outcome,
                    "duration_seconds": round(report.duration, 4),
                    "error_message": str(report.longrepr) if report.failed else None
                })

    capture_plugin = PytestCapturePlugin()
    pytest_args = ["-q"] + selected_targets
    exit_code = pytest.main(pytest_args, plugins=[capture_plugin])

    total = len(capture_plugin.reports)
    passed = sum(1 for r in capture_plugin.reports if r["outcome"] == "passed")
    failed = sum(1 for r in capture_plugin.reports if r["outcome"] == "failed")
    skipped = sum(1 for r in capture_plugin.reports if r["outcome"] == "skipped")

    logger.info(f"Test suite execution finished: {passed}/{total} passed, {failed} failed")

    justifications = {
        "test_cleaning.py": "Protects data pipeline from corrupted units (e.g. 785 rows with seconds-to-minutes bug D-015), sentinel values (HR=0/250, HRV=999 D-018), and case-sensitivity join failures.",
        "test_features.py": "Guarantees temporal safety by verifying lag shifts and recency event guards never peer into current/future observations.",
        "test_pipeline.py": "Guarantees end-to-end integrity from raw files to modelling table, verifying that all 120 users are retained without catastrophic drops.",
        "test_inference.py": "Guarantees API contracts: output bounded strictly in [1.0, 5.0], graceful cold-start handling for users with no history, and numerical reproducibility."
    }

    return {
        "status": "passed" if exit_code == 0 else "failed",
        "exit_code": int(exit_code),
        "summary": {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "suite_justifications": justifications,
        "test_results": capture_plugin.reports
    }
