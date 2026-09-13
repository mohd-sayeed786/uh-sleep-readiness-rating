"""
Inference and Explainability Module for Ring Readiness Prediction.
Loads the trained XGBoost model and produces point predictions, confidence bounds,
actionable contextual guidance, counterfactual score deltas, and TreeSHAP feature attributions.
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
    """
    Production-grade predictor for daily morning readiness scores (1-5 scale).
    Supports single prediction, batch prediction, and exact TreeSHAP attribution.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        feature_list_path: Optional[Path] = None,
    ):
        self.model_path = model_path or MODEL_PATH
        self.feature_list_path = feature_list_path or FEATURE_LIST_PATH
        self.model = None
        self.feature_names: List[str] = []
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained XGBoost model and ordered feature list."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please train a model first using train.py."
            )
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded active model from {self.model_path}")

        if self.feature_list_path.exists():
            with open(self.feature_list_path, "r") as f:
                self.feature_names = json.load(f)
            logger.info(f"Loaded {len(self.feature_names)} features from {self.feature_list_path}")
        else:
            self.feature_names = list(FEATURE_NAMES)
            logger.warning(f"Feature list file not found; falling back to config.FEATURE_NAMES ({len(self.feature_names)} features)")

    def _predict_raw_features(self, feat_dict: Dict[str, Any]) -> float:
        """Helper to run model on a feature dictionary, returning raw clamped score."""
        row = {feat: feat_dict.get(feat, np.nan) for feat in self.feature_names}
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
          1. Last Night's Rest (How well did you recover?)
          2. Today's Rhythm (What should you do about it today?)
          3. Tonight's Quick Win + Tomorrow's Boost (Model-linked counterfactual delta)
        """
        alc_units = float(features.get("alcohol_units") or 0.0)
        had_alc = float(features.get("had_alcohol") or 0.0)
        sleep_debt = float(features.get("sleep_debt") or 0.0)
        sleep_z = float(features.get("total_sleep_minutes_zscore") or 0.0)
        hr_z = float(features.get("avg_hr_bpm_zscore") or 0.0)
        hrv_z = float(features.get("avg_hrv_rmssd_ms_zscore") or 0.0)
        stress_z = float(features.get("stress_index_z") or 0.0)
        recovery_sc = float(features.get("recovery_score") or 0.0)

        # 1. "Last Night's Rest" (user-friendly recovery summary)
        if tier == "Recovery":
            reasons = []
            if had_alc > 0 or alc_units > 0:
                reasons.append(f"{alc_units:.1f} drinks kept heart rate elevated")
            if sleep_debt < -20 or sleep_z < -0.8:
                reasons.append(f"{abs(int(sleep_debt))}m sleep deficit")
            if hr_z > 0.8 or stress_z > 0.8:
                reasons.append("higher autonomic stress than normal")
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

        if pred_raw < 4.85:
            candidate_levers = []

            # Lever A: Eliminate alcohol
            if had_alc > 0 or alc_units > 0:
                cf = dict(features)
                cf["alcohol_units"] = 0.0
                cf["had_alcohol"] = 0.0
                cf["alcohol_level"] = 0.0
                cf["alcohol_x_hrv_z"] = 0.0
                cf_pred = self._predict_raw_features(cf)
                delta = cf_pred - pred_raw
                if delta > 0.02:
                    candidate_levers.append((
                        delta,
                        "Skip alcohol tonight to lower your resting HR and wake up refreshed",
                        cf_pred
                    ))

            # Lever B: Clear sleep deficit (+45m sleep duration & restorative depth)
            if sleep_debt < 15.0 or sleep_z < 0.6:
                cf = dict(features)
                cf["total_sleep_minutes_zscore"] = max(sleep_z + 0.8, 0.8)
                cf["sleep_debt"] = max(sleep_debt + 45.0, 15.0)
                cf["deep_rem_total"] = max(float(features.get("deep_rem_total") or 120.0) + 25.0, 150.0)
                cf["sleep_user_ratio"] = max(float(features.get("sleep_user_ratio") or 1.0) + 0.1, 1.1)
                cf["restorative_pct"] = min(0.45, max(float(features.get("restorative_pct") or 0.35) + 0.05, 0.40))
                cf_pred = self._predict_raw_features(cf)
                delta = cf_pred - pred_raw
                if delta > 0.02:
                    candidate_levers.append((
                        delta,
                        "Head to bed 45 mins earlier to erase sleep debt and reset energy",
                        cf_pred
                    ))

            # Lever C: Normalize autonomic stress & autonomic recovery
            if hr_z > 0.3 or hrv_z < 0.0 or stress_z > 0.2:
                cf = dict(features)
                cf["avg_hr_bpm_zscore"] = -0.5
                cf["avg_hrv_rmssd_ms_zscore"] = max(hrv_z + 0.8, 0.8)
                cf["stress_index_z"] = min(-0.5, stress_z - 1.0)
                cf["recovery_score"] = max(0.5, recovery_sc + 1.0)
                cf["hr_user_ratio"] = 0.95
                cf["hrv_user_ratio"] = 1.10
                cf_pred = self._predict_raw_features(cf)
                delta = cf_pred - pred_raw
                if delta > 0.02:
                    candidate_levers.append((
                        delta,
                        "Wind down with 10m breathwork and keep dinner light before bed",
                        cf_pred
                    ))

            # Lever D: Combined optimization (alcohol + sleep debt recovery)
            if (had_alc > 0 or alc_units > 0) and (sleep_debt < 0 or hr_z > 0):
                cf = dict(features)
                cf["alcohol_units"] = 0.0
                cf["had_alcohol"] = 0.0
                cf["alcohol_level"] = 0.0
                cf["alcohol_x_hrv_z"] = 0.0
                cf["total_sleep_minutes_zscore"] = max(sleep_z + 0.8, 0.8)
                cf["sleep_debt"] = max(sleep_debt + 45.0, 15.0)
                cf["deep_rem_total"] = max(float(features.get("deep_rem_total") or 120.0) + 25.0, 150.0)
                cf["sleep_user_ratio"] = 1.10
                cf["avg_hr_bpm_zscore"] = -0.4
                cf["avg_hrv_rmssd_ms_zscore"] = max(hrv_z + 0.6, 0.6)
                cf["stress_index_z"] = -0.6
                cf["recovery_score"] = 0.6
                cf_pred = self._predict_raw_features(cf)
                delta = cf_pred - pred_raw
                if delta > 0.02:
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
              - recommendation: structured recovery & action guidance
              - is_cold_start: boolean flag indicating whether user baseline features were absent
              - features_used: dictionary of feature values sent to the model
        """
        # Auto-derive missing interactions if components are present
        features_clean = dict(features)
        if features_clean.get("had_alcohol") is None and features_clean.get("alcohol_units") is not None:
            features_clean["had_alcohol"] = 1.0 if float(features_clean["alcohol_units"]) > 0 else 0.0
        if features_clean.get("alcohol_level") is None and features_clean.get("alcohol_units") is not None:
            u = float(features_clean["alcohol_units"])
            features_clean["alcohol_level"] = 0.0 if u <= 0 else (1.0 if u <= 2.0 else 2.0)
        if features_clean.get("alcohol_x_hrv_z") is None:
            alc = float(features_clean.get("alcohol_units") or 0.0)
            hrv_z = float(features_clean.get("avg_hrv_rmssd_ms_zscore") or 0.0)
            features_clean["alcohol_x_hrv_z"] = alc * hrv_z
        if features_clean.get("stress_index_z") is None and features_clean.get("avg_hr_bpm_zscore") is not None and features_clean.get("avg_hrv_rmssd_ms_zscore") is not None:
            features_clean["stress_index_z"] = float(features_clean["avg_hr_bpm_zscore"]) - float(features_clean["avg_hrv_rmssd_ms_zscore"])
        if features_clean.get("recovery_score") is None and features_clean.get("avg_hrv_rmssd_ms_zscore") is not None and features_clean.get("avg_hr_bpm_zscore") is not None:
            features_clean["recovery_score"] = float(features_clean["avg_hrv_rmssd_ms_zscore"]) - float(features_clean["avg_hr_bpm_zscore"])

        row = {feat: features_clean.get(feat, np.nan) for feat in self.feature_names}
        X = pd.DataFrame([row], columns=self.feature_names).astype(float)

        raw_pred = float(self.model.predict(X)[0])
        clamped_raw = float(np.clip(raw_pred, 1.0, 5.0))
        rounded_pred = int(np.clip(np.round(clamped_raw), 1, 5))

        # Cold start detection:
        # A user is in cold start if:
        # 1. Baseline features are completely missing / NaN (no z-scores and no historical feeling anchors)
        # 2. OR explicitly flagged as Day 1 onboarding (checkin_seq_num <= 1 AND without established multi-day feeling history).
        has_baseline_zscores = any(
            features_clean.get(f) is not None and not pd.isna(features_clean.get(f))
            for f in ["total_sleep_minutes_zscore", "avg_hr_bpm_zscore", "avg_hrv_rmssd_ms_zscore", "sleep_debt"]
        )
        has_history_feeling = any(
            features_clean.get(f) is not None and not pd.isna(features_clean.get(f))
            for f in ["feeling_roll5_mean", "user_expanding_mean", "feeling_ewm_7", "subjective_feeling_lag1"]
        )

        seq_num = features_clean.get("checkin_seq_num")
        is_explicit_day_1 = seq_num is not None and not pd.isna(seq_num) and float(seq_num) <= 1.0

        if is_explicit_day_1:
            is_cold_start = True
        elif not has_baseline_zscores and not has_history_feeling:
            is_cold_start = True
        else:
            is_cold_start = False

        has_alc = features_clean.get("had_alcohol")
        has_alc_val = 0.0 if pd.isna(has_alc) else float(has_alc)

        hr_z = features_clean.get("avg_hr_bpm_zscore")
        hr_z_val = None if pd.isna(hr_z) else float(hr_z)

        tier, guidance = self._map_tier_and_guidance(rounded_pred, has_alc_val, hr_z_val)
        recommendation = self.generate_recommendation(features_clean, clamped_raw, tier)

        return {
            "pred_raw": round(clamped_raw, 3),
            "pred_rounded": rounded_pred,
            "readiness_tier": tier,
            "guidance": guidance,
            "recommendation": recommendation,
            "is_cold_start": is_cold_start,
            "features_used": {k: (None if pd.isna(v) else v) for k, v in row.items()},
        }

    def explain_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute readiness score alongside exact TreeSHAP marginal feature contributions.
        Uses native Booster pred_contribs=True for zero-latency exact SHAP values.
        """
        import xgboost as xgb

        # Baseline prediction
        base_res = self.predict_single(features)

        # Retrieve cleaned row with all features used
        cleaned_features = base_res["features_used"]
        row = {feat: cleaned_features.get(feat, np.nan) for feat in self.feature_names}
        X = pd.DataFrame([row], columns=self.feature_names).astype(float)

        # TreeSHAP via XGBoost booster
        dmat = xgb.DMatrix(X)
        booster = self.model.get_booster() if hasattr(self.model, "get_booster") else self.model
        contribs = booster.predict(dmat, pred_contribs=True)[0]

        base_value = float(contribs[-1])
        shap_raw = contribs[:-1]

        feature_labels = {
            "had_alcohol": "Alcohol Consumed",
            "alcohol_level": "Alcohol Severity Tier",
            "deep_rem_total": "Restorative Sleep (Deep + REM)",
            "total_sleep_minutes_zscore": "Sleep Duration (Z-Score)",
            "stress_index_z": "Physiological Stress Index",
            "alcohol_units": "Alcohol Intake (Units)",
            "alcohol_x_hrv_z": "Alcohol × HRV Interaction",
            "sleep_debt": "Sleep Deficit / Surplus (min)",
            "rem_minutes_zscore": "REM Sleep (Z-Score)",
            "sleep_user_ratio": "Sleep vs Baseline Ratio",
            "recovery_score": "Autonomic Recovery Score",
            "avg_hr_bpm_zscore": "Resting Heart Rate (Z-Score)",
            "deep_minutes_zscore": "Deep Sleep (Z-Score)",
            "restorative_pct": "Restorative Sleep Ratio (%)",
            "avg_hrv_rmssd_ms_zscore": "HRV Parasympathetic (Z-Score)",
            "feeling_roll5_mean": "Recent Feeling (5-Day Rolling)",
            "feeling_ewm_7": "Weighted Recent Feeling (7-Day)",
            "hrv_user_ratio": "HRV vs Baseline Ratio",
            "deep_user_ratio": "Deep Sleep vs Baseline Ratio",
            "user_expanding_mean": "Historical Baseline Feeling",
            "hr_user_ratio": "Heart Rate vs Baseline Ratio",
            # Auxiliary / Legacy labels
            "subjective_feeling_lag1": "Yesterday's Feeling (Lag 1)",
            "checkin_seq_num": "Check-in Day Count",
            "days_since_bad_sleep": "Days Since Bad Sleep",
            "days_since_great_sleep": "Days Since Great Sleep",
            "week_of_year": "Seasonality (Week of Year)",
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
        """Predict readiness for a batch of records."""
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient="records")
        else:
            records = data

        logger.debug(f"Executing batch prediction on {len(records)} records")
        return [self.predict_single(rec) for rec in records]


# Singleton pattern for FastAPI process memory efficiency
_predictor_instance: Optional[ReadinessPredictor] = None


def get_predictor() -> ReadinessPredictor:
    """Provides a lazily-instantiated singleton ReadinessPredictor."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ReadinessPredictor()
    return _predictor_instance


def reload_predictor() -> ReadinessPredictor:
    """Forces reloading of the predictor artifacts after retraining or rollback."""
    global _predictor_instance
    _predictor_instance = ReadinessPredictor()
    return _predictor_instance
