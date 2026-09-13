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
        """Load pickled model artifact, auto-generating at runtime if missing."""
        if not self.model_path.exists():
            logger.warning(f"Model file not found at {self.model_path}. Auto-generating champion model at runtime...")
            try:
                from src.train import train_model
                train_model(save_artifacts=True)
            except Exception as e:
                err_msg = f"Model file not found at {self.model_path} and auto-training failed: {e}"
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)

        logger.info(f"Loading model artifact from {self.model_path}...")
        with open(self.model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def _predict_raw_features(self, feat_dict: Dict[str, Any]) -> float:
        """Helper to quickly evaluate a counterfactual feature set on the active model."""
        row = {f: feat_dict.get(f, np.nan) for f in self.feature_names}
        X = pd.DataFrame([row], columns=self.feature_names).astype(float)
        raw = float(self.model.predict(X)[0])
        return float(np.clip(raw, 1.0, 5.0))

    def generate_recommendation(
        self,
        features: Dict[str, Any],
        pred_raw: float,
        tier: str,
    ) -> Dict[str, Any]:
        """
        Generates contextual recovery assessments and actionable interventions linked directly
        to counterfactual predictions from the active XGBoost model.
        Answers the core brief prompt:
          1. How well did you recover?
          2. What should you do about it today? (with model delta projected from behavioral levers)
        """
        alc_units = float(features.get("alcohol_units") or 0.0)
        had_alc = float(features.get("had_alcohol") or 0.0)
        sleep_debt = float(features.get("sleep_debt") or 0.0)
        sleep_z = float(features.get("total_sleep_minutes_zscore") or 0.0)
        hr_z = float(features.get("avg_hr_bpm_zscore") or 0.0)
        hrv_z = float(features.get("avg_hrv_rmssd_ms_zscore") or 0.0)

        # 1. "Last Night's Rest" (user-friendly recovery summary)
        if tier == "Recovery":
            reasons = []
            if had_alc > 0 or alc_units > 0:
                reasons.append(f"{alc_units:.1f} drinks kept heart rate elevated")
            if sleep_debt < -20 or sleep_z < -0.8:
                reasons.append(f"{abs(int(sleep_debt))}m sleep deficit")
            if hr_z > 0.8:
                reasons.append("higher resting heart rate than normal")
            if hrv_z < -0.8:
                reasons.append("nervous system was working in overdrive")
            if not reasons:
                reasons.append("shorter restorative deep & REM cycles")
            recovery_assessment = "Rest was a bit choppy (" + f"{pred_raw:.2f}/5). " + " & ".join(reasons).capitalize() + " — your body worked harder than usual overnight."
        elif tier == "Moderate":
            recovery_assessment = f"Steady, solid rest ({pred_raw:.2f}/5). Your heart rate, HRV, and sleep depth were right around your normal baseline."
        else:
            recovery_assessment = f"Deep, high-quality recharge ({pred_raw:.2f}/5). Calm resting heart rate and strong restorative stages left your body fully topped up."

        # 2. "Today's Rhythm" (approachable daily pacing)
        if tier == "Recovery":
            what_to_do = "Take things easy today. Stick to gentle walks or light movement, drink plenty of water, and treat yourself to an earlier bedtime tonight."
        elif tier == "Moderate":
            what_to_do = "Keep your regular rhythm. You have good steady energy for normal workouts, focused work blocks, and everyday tasks."
        else:
            what_to_do = "You're primed to go! Perfect day for a challenging workout, aiming for a personal best, or tackling high-focus projects."

        # 3. Model-Linked Score Improvement Lever (Counterfactual Prediction)
        best_action = "Keep up your great sleep schedule; your body is in an ideal groove."
        best_delta = 0.0

        if pred_raw < 4.8:
            candidate_levers = []

            # Lever A: Eliminate alcohol
            if had_alc > 0 or alc_units > 0:
                cf = dict(features)
                cf["alcohol_units"] = 0.0
                cf["had_alcohol"] = 0.0
                cf["alcohol_level"] = 0.0
                cf_pred = self._predict_raw_features(cf)
                delta = cf_pred - pred_raw
                if delta > 0.03:
                    candidate_levers.append((
                        delta,
                        f"Skip alcohol tonight to lower your resting HR and wake up refreshed",
                        cf_pred
                    ))

            # Lever B: Clear sleep deficit (+45m sleep duration)
            if sleep_debt < 15.0 or sleep_z < 0.6:
                cf = dict(features)
                cf["total_sleep_minutes_zscore"] = max(sleep_z + 0.8, 0.8)
                cf["sleep_debt"] = max(sleep_debt + 45.0, 15.0)
                cf["deep_rem_total"] = max(float(features.get("deep_rem_total") or 100.0) + 20.0, 140.0)
                cf_pred = self._predict_raw_features(cf)
                delta = cf_pred - pred_raw
                if delta > 0.03:
                    candidate_levers.append((
                        delta,
                        "Head to bed 45 mins earlier to erase sleep debt and reset energy",
                        cf_pred
                    ))

            # Lever C: Normalize resting HR & parasympathetic tone
            if hr_z > 0.3 or hrv_z < 0.0:
                cf = dict(features)
                cf["avg_hr_bpm_zscore"] = -0.5
                cf["avg_hrv_rmssd_ms_zscore"] = max(hrv_z + 0.8, 0.8)
                cf_pred = self._predict_raw_features(cf)
                delta = cf_pred - pred_raw
                if delta > 0.03:
                    candidate_levers.append((
                        delta,
                        "Wind down with 10m breathwork and keep dinner light before bed",
                        cf_pred
                    ))

            # Lever D: Combined optimization (alcohol + sleep debt recovery)
            if (had_alc > 0) and (sleep_debt < 0 or hr_z > 0):
                cf = dict(features)
                cf["alcohol_units"] = 0.0
                cf["had_alcohol"] = 0.0
                cf["alcohol_level"] = 0.0
                cf["total_sleep_minutes_zscore"] = max(sleep_z + 0.8, 0.8)
                cf["sleep_debt"] = max(sleep_debt + 45.0, 15.0)
                cf["avg_hr_bpm_zscore"] = -0.4
                cf_pred = self._predict_raw_features(cf)
                delta = cf_pred - pred_raw
                if delta > 0.03:
                    candidate_levers.append((
                        delta,
                        "Skip the evening drink + get 45 mins extra sleep to clear debt",
                        cf_pred
                    ))

            if candidate_levers:
                candidate_levers.sort(key=lambda x: x[0], reverse=True)
                top_delta, top_action, top_cf_pred = candidate_levers[0]
                best_delta = round(top_delta, 2)
                best_action = top_action
                projected_score = min(5.0, round(pred_raw + best_delta, 2))
            else:
                projected_score = round(pred_raw, 2)
        else:
            projected_score = round(pred_raw, 2)

        return {
            "how_did_you_recover": recovery_assessment,
            "what_to_do_today": what_to_do,
            "improvement_action": best_action,
            "projected_delta": best_delta,
            "projected_score": projected_score,
        }

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

        # Cold start detection:
        # A user is in cold start if:
        # 1. Day-1 onboarding checkin (checkin_seq_num <= 1)
        # 2. Or missing baseline history (both subjective_feeling_lag1 and total_sleep_minutes_zscore are NaN/absent)
        seq_num = features.get("checkin_seq_num")
        is_day_1 = seq_num is not None and not pd.isna(seq_num) and float(seq_num) <= 1.0

        is_cold_start = bool(
            is_day_1 or (
                pd.isna(features.get("subjective_feeling_lag1")) and
                pd.isna(features.get("total_sleep_minutes_zscore"))
            )
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

        recommendation = self.generate_recommendation(
            features=features,
            pred_raw=clamped_raw,
            tier=tier,
        )

        return {
            "pred_raw": round(clamped_raw, 3),
            "pred_rounded": rounded_pred,
            "readiness_tier": tier,
            "guidance": guidance,
            "recommendation": recommendation,
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
