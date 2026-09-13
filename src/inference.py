"""
Inference module for Ring AI Readiness Score prediction.
Loads the trained model and feature definitions to predict subjective recovery for single or batch inputs.
"""
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import FEATURE_LIST_PATH, FEATURE_NAMES, MODEL_PATH
from src.logger import get_logger

logger = get_logger("inference")


class ReadinessPredictor:
    """Predictor class encapsulating model loading and inference logic."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        feature_names: Optional[List[str]] = None,
        feature_list_path: Optional[Union[str, Path]] = None,
    ):
        self.model_path = Path(model_path or MODEL_PATH)
        self.feature_list_path = Path(feature_list_path or FEATURE_LIST_PATH)

        if feature_names is not None:
            self.feature_names = feature_names
        elif self.feature_list_path.exists():
            try:
                with open(self.feature_list_path, "r") as f:
                    self.feature_names = json.load(f)
                logger.debug(f"Loaded {len(self.feature_names)} features from {self.feature_list_path.name}")
            except Exception as e:
                logger.warning(f"Could not load feature list file: {e}; falling back to default schema")
                self.feature_names = FEATURE_NAMES
        else:
            self.feature_names = FEATURE_NAMES

        self.model = self._load_model()

    def _load_model(self):
        """Load pickled model artifact."""
        if not self.model_path.exists():
            err_msg = f"Model file not found at {self.model_path}. Run training pipeline first."
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        logger.info(f"Loading model artifact from {self.model_path}...")
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def _map_tier_and_guidance(score_rounded: int, has_alcohol: float, avg_hr_zscore: float) -> Tuple[str, str]:
        """Generate human-readable readiness category and actionable guidance."""
        if score_rounded <= 2:
            tier = "Recovery"
            if has_alcohol == 1.0:
                guidance = "Recovery indicated. Alcohol consumption impacted overnight restorative rest; prioritize hydration and lighter activity."
            elif avg_hr_zscore is not None and avg_hr_zscore > 1.0:
                guidance = "Elevated resting heart rate detected overnight. Take it easy and schedule time to recharge."
            else:
                guidance = "Sleep reserves are running low. Prioritize an early bedtime and light restorative movement."
        elif score_rounded == 3:
            tier = "Moderate"
            guidance = "Baseline readiness. Steady capacity for standard daily training and work demands."
        else:
            tier = "Optimal"
            guidance = "High recovery status. Physiological markers show strong restorative sleep; great day for high performance or intense training."
        return tier, guidance

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict readiness for a single observation.

        Args:
            features: Dictionary containing feature values.
                      Any missing features default to np.nan (natively supported by XGBoost).

        Returns:
            Dictionary with:
              - pred_raw: continuous float prediction
              - pred_rounded: integer score (1 to 5)
              - readiness_tier: Recovery / Moderate / Optimal
              - guidance: user-facing recommendation
              - is_cold_start: boolean flag indicating whether user baseline features were absent
              - features_used: dictionary of feature values sent to the model
        """
        row = {feat: features.get(feat, np.nan) for feat in self.feature_names}
        X = pd.DataFrame([row], columns=self.feature_names).astype(float)

        raw_pred = float(self.model.predict(X)[0])
        clamped_raw = float(np.clip(raw_pred, 1.0, 5.0))
        rounded_pred = int(np.clip(np.round(clamped_raw), 1, 5))

        # Cold start detection (e.g. no prior checkins or z-scores available)
        is_cold_start = bool(
            pd.isna(features.get("subjective_feeling_lag1")) and
            pd.isna(features.get("total_sleep_minutes_zscore"))
        )

        has_alc = features.get("had_alcohol")
        has_alc_val = 0.0 if pd.isna(has_alc) else float(has_alc)

        hr_z = features.get("avg_hr_bpm_zscore")
        hr_z_val = 0.0 if pd.isna(hr_z) else float(hr_z)

        tier, guidance = self._map_tier_and_guidance(
            score_rounded=rounded_pred,
            has_alcohol=has_alc_val,
            avg_hr_zscore=hr_z_val
        )

        logger.debug(f"Predict single: raw={clamped_raw:.3f}, rounded={rounded_pred}, tier={tier}, cold_start={is_cold_start}")

        return {
            "pred_raw": round(clamped_raw, 3),
            "pred_rounded": rounded_pred,
            "readiness_tier": tier,
            "guidance": guidance,
            "is_cold_start": is_cold_start,
            "features_used": {k: None if pd.isna(v) else float(v) for k, v in row.items()}
        }

    def explain_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute readiness score alongside exact TreeSHAP marginal feature contributions.
        Uses native Booster pred_contribs=True for zero-latency exact SHAP values.
        """
        import xgboost as xgb

        row = {feat: features.get(feat, np.nan) for feat in self.feature_names}
        X = pd.DataFrame([row], columns=self.feature_names).astype(float)

        # Baseline prediction
        base_res = self.predict_single(features)

        # TreeSHAP via XGBoost booster
        dmat = xgb.DMatrix(X)
        booster = self.model.get_booster() if hasattr(self.model, "get_booster") else self.model
        contribs = booster.predict(dmat, pred_contribs=True)[0]

        base_value = float(contribs[-1])
        shap_raw = contribs[:-1]

        feature_labels = {
            "alcohol_units": "Alcohol Intake (Units)",
            "had_alcohol": "Alcohol Consumed",
            "alcohol_level": "Alcohol Severity Tier",
            "week_of_year": "Seasonality (Week of Year)",
            "total_sleep_minutes_zscore": "Sleep Duration (Z-Score)",
            "avg_hr_bpm_zscore": "Resting Heart Rate (Z-Score)",
            "avg_hrv_rmssd_ms_zscore": "HRV Parasympathetic (Z-Score)",
            "subjective_feeling_lag1": "Yesterday's Feeling (Lag 1)",
            "days_since_bad_sleep": "Days Since Bad Sleep",
            "days_since_great_sleep": "Days Since Great Sleep",
            "checkin_seq_num": "Check-in Habit / Day Count",
            "deep_rem_total": "Restorative Sleep (Deep + REM)",
            "sleep_debt": "Sleep Deficit / Surplus (min)",
        }

        shap_details = []
        for feat_name, shap_val in zip(self.feature_names, shap_raw):
            val = row[feat_name]
            shap_details.append({
                "feature": feat_name,
                "label": feature_labels.get(feat_name, feat_name.replace("_", " ").title()),
                "value": None if pd.isna(val) else round(float(val), 2),
                "shap_impact": round(float(shap_val), 4),
                "direction": "positive" if shap_val >= 0 else "negative",
                "abs_impact": abs(float(shap_val)),
            })

        # Sort by absolute SHAP impact descending so strongest drivers are first
        shap_details.sort(key=lambda x: x["abs_impact"], reverse=True)

        base_res["base_value"] = round(base_value, 4)
        base_res["shap_breakdown"] = shap_details
        base_res["total_shap_impact"] = round(float(sum(shap_raw)), 4)

        return base_res

    def predict_batch(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Predict readiness for a batch of records.
        """
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
        else:
            records = data

        logger.debug(f"Executing batch prediction on {len(records)} records")
        return [self.predict_single(rec) for rec in records]


# Singleton instance cache for fast reuse in API / services
_default_predictor: Optional[ReadinessPredictor] = None


def get_predictor(force_reload: bool = False) -> ReadinessPredictor:
    """Returns a cached singleton instance of ReadinessPredictor."""
    global _default_predictor
    if _default_predictor is None or force_reload:
        logger.debug("Instantiating new singleton ReadinessPredictor")
        _default_predictor = ReadinessPredictor()
    return _default_predictor
