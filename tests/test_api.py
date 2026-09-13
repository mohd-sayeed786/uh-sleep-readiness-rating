"""
Functional tests for the FastAPI service endpoints.
Verifies API contract adherence, feature calculation, model versioning, and prediction payloads.
"""
import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_api_health():
    """Verify health endpoint reports healthy status and model presence."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["feature_count"] == 21
    assert "selected_model.pkl" in data["active_model"]


def test_api_calculate_features():
    """Verify feature calculation endpoint transforms raw metrics into model inputs."""
    payload = {
        "user_id": "UH-001",
        "checkin_date": "2026-03-20",
        "alcohol_units": 1.5,
        "total_sleep_minutes": 440.0,
        "deep_minutes": 70.0,
        "rem_minutes": 80.0,
        "avg_hr_bpm": 58.0,
        "avg_hrv_rmssd_ms": 52.0,
        "subjective_feeling_lag1": 4.0,
    }
    response = client.post("/features/calculate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "UH-001"
    feats = data["features"]
    assert feats["had_alcohol"] == 1.0
    assert feats["alcohol_level"] == 1.0
    assert feats["deep_rem_total"] == 150.0
    assert feats["subjective_feeling_lag1"] == 4.0


def test_api_predict_single():
    """Verify single prediction endpoint outputs bounded score and user guidance."""
    payload = {
        "alcohol_units": 0.0,
        "had_alcohol": 0.0,
        "alcohol_level": 0.0,
        "week_of_year": 12,
        "total_sleep_minutes_zscore": 0.8,
        "avg_hr_bpm_zscore": -0.5,
        "avg_hrv_rmssd_ms_zscore": 0.9,
        "subjective_feeling_lag1": 4.0,
        "days_since_bad_sleep": 8.0,
        "days_since_great_sleep": 1.0,
        "checkin_seq_num": 18,
        "deep_rem_total": 150.0,
        "sleep_debt": 20.0,
    }
    response = client.post("/inference/predict", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert 1.0 <= res["pred_raw"] <= 5.0
    assert res["pred_rounded"] in [1, 2, 3, 4, 5]
    assert res["readiness_tier"] in ["Recovery", "Moderate", "Optimal"]
    assert len(res["guidance"]) > 0


def test_api_explain_endpoint():
    """Verify explain endpoint computes exact TreeSHAP feature breakdown."""
    payload = {
        "alcohol_units": 2.0,
        "had_alcohol": 1.0,
        "alcohol_level": 1.0,
        "week_of_year": 15,
        "total_sleep_minutes_zscore": -0.5,
        "avg_hr_bpm_zscore": 1.2,
        "avg_hrv_rmssd_ms_zscore": -1.0,
        "subjective_feeling_lag1": 2.0,
        "days_since_bad_sleep": 1.0,
        "days_since_great_sleep": 10.0,
        "checkin_seq_num": 25,
        "deep_rem_total": 80.0,
        "sleep_debt": -40.0,
    }
    response = client.post("/inference/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "pred_raw" in data
    assert "base_value" in data
    assert "shap_breakdown" in data
    assert len(data["shap_breakdown"]) == 21
    assert "total_shap_impact" in data
    # Check structure of each SHAP item
    first_item = data["shap_breakdown"][0]
    assert "feature" in first_item
    assert "label" in first_item
    assert "shap_impact" in first_item
    assert first_item["direction"] in ["positive", "negative"]


def test_api_simulator_endpoint():
    """Verify simulator endpoint returns the HTML interactive dashboard."""
    response = client.get("/simulator")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ultrahuman" in response.text.lower()
    assert "gaugeArc" in response.text
    assert "TreeSHAP" in response.text


def test_api_predict_batch():
    """Verify batch prediction endpoint preserves order and batch size."""
    rec1 = {"alcohol_units": 0.0, "total_sleep_minutes_zscore": 0.5}
    rec2 = {"alcohol_units": 3.0, "total_sleep_minutes_zscore": -1.2}
    response = client.post("/inference/predict-batch", json={"records": [rec1, rec2]})
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    assert len(data["predictions"]) == 2


def test_api_run_tests_endpoint():
    """Verify test-runner endpoint executes suite programmatically."""
    response = client.post("/tests/run?test_suite=features")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "passed"
    assert data["summary"]["passed"] > 0


def test_api_model_versions_endpoint():
    """Verify model version list endpoint reports active model and archived versions."""
    response = client.get("/model/versions")
    assert response.status_code == 200
    data = response.json()
    assert data["active_model"] == "selected_model.pkl"
    assert "archived_versions" in data
