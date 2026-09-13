"""
Evaluation and error analysis module.
Computes comprehensive slice breakdowns, error analyses, and baseline comparisons.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import FEATURE_NAMES, TARGET_COL
from src.data_pipeline import run_data_pipeline
from src.features import engineer_features
from src.logger import get_logger
from src.train import compute_metrics, split_temporal

logger = get_logger("evaluate")


def evaluate_slices(df_test: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Perform slice-based error analysis across user segments and contexts:
      - By Target Score (1 to 5)
      - By Sleep Session Availability (has_session_data)
      - By Alcohol Consumption (had_alcohol)
      - By Check-in Timing (early_submit < 9am)
    """
    df = df_test.copy()
    df["pred_raw"] = y_pred
    df["pred_rounded"] = np.clip(np.round(y_pred), 1, 5).astype(int)
    df["error"] = df[TARGET_COL] - df["pred_raw"]
    df["abs_error"] = np.abs(df["error"])
    df["exact_match"] = (df["pred_rounded"] == df[TARGET_COL]).astype(int)
    df["within_1"] = (df["abs_error"] <= 1.0).astype(int)

    slices: Dict[str, Any] = {}

    # 1. By Target Class (1 to 5)
    class_breakdown = {}
    for cls in range(1, 6):
        cls_sub = df[df[TARGET_COL] == cls]
        if len(cls_sub) > 0:
            class_breakdown[f"feeling_{cls}"] = {
                "count": len(cls_sub),
                "exact_accuracy": round(float(cls_sub["exact_match"].mean()), 4),
                "accuracy_pm1": round(float(cls_sub["within_1"].mean()), 4),
                "mae": round(float(cls_sub["abs_error"].mean()), 4),
                "mean_prediction": round(float(cls_sub["pred_raw"].mean()), 4)
            }
    slices["by_class"] = class_breakdown

    # 2. By Session Presence
    if "has_session_data" in df.columns:
        sess_breakdown = {}
        for val, label in [(1, "with_sleep_session"), (0, "ring_not_worn_missing_sleep")]:
            sub = df[df["has_session_data"] == val]
            if len(sub) > 0:
                sess_breakdown[label] = {
                    "count": len(sub),
                    "exact_accuracy": round(float(sub["exact_match"].mean()), 4),
                    "rmse": round(float(np.sqrt(np.mean(sub["error"] ** 2))), 4),
                    "mae": round(float(sub["abs_error"].mean()), 4)
                }
        slices["by_session_data"] = sess_breakdown

    # 3. By Alcohol Consumption
    if "had_alcohol" in df.columns:
        alc_breakdown = {}
        for val, label in [(1.0, "consumed_alcohol"), (0.0, "no_alcohol_logged")]:
            sub = df[df["had_alcohol"] == val]
            if len(sub) > 0:
                alc_breakdown[label] = {
                    "count": len(sub),
                    "exact_accuracy": round(float(sub["exact_match"].mean()), 4),
                    "rmse": round(float(np.sqrt(np.mean(sub["error"] ** 2))), 4),
                    "mae": round(float(sub["abs_error"].mean()), 4)
                }
        slices["by_alcohol"] = alc_breakdown

    # 4. By Check-in Timing (early < 7am)
    if "early_submit" in df.columns:
        timing_breakdown = {}
        for val, label in [(1, "early_waking_submits"), (0, "standard_submits")]:
            sub = df[df["early_submit"] == val]
            if len(sub) > 0:
                timing_breakdown[label] = {
                    "count": len(sub),
                    "exact_accuracy": round(float(sub["exact_match"].mean()), 4),
                    "rmse": round(float(np.sqrt(np.mean(sub["error"] ** 2))), 4),
                    "mae": round(float(sub["abs_error"].mean()), 4)
                }
        slices["by_submission_timing"] = timing_breakdown

    logger.debug(f"Computed slice breakdowns across {len(df)} test observations")
    return slices


def run_full_evaluation(model, feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run full evaluation on temporal test split."""
    logger.info("Executing full model evaluation on temporal holdout split...")
    base_df = run_data_pipeline()
    feature_df = engineer_features(base_df).dropna(subset=[TARGET_COL])

    train_df, val_df, test_df = split_temporal(feature_df)

    feats = feature_cols or FEATURE_NAMES
    X_test = test_df[feats]
    y_test = test_df[TARGET_COL]

    preds = model.predict(X_test)
    overall_metrics = compute_metrics(y_test, preds, split_name="Test", n_features=len(feats))
    slices = evaluate_slices(test_df, preds)

    # Comparison against legacy readiness shown heuristic
    legacy_comp = None
    if "legacy_readiness_shown" in test_df.columns:
        valid_legacy = test_df.dropna(subset=["legacy_readiness_shown", TARGET_COL])
        if len(valid_legacy) > 0:
            corr = float(valid_legacy["legacy_readiness_shown"].corr(valid_legacy[TARGET_COL]))
            scaled_legacy = 1.0 + (valid_legacy["legacy_readiness_shown"] - 1.0) * 4.0 / 99.0
            legacy_rmse = float(np.sqrt(np.mean((valid_legacy[TARGET_COL] - scaled_legacy) ** 2)))
            legacy_comp = {
                "legacy_correlation": round(corr, 4),
                "legacy_scaled_rmse": round(legacy_rmse, 4),
                "model_rmse": overall_metrics["rmse"]
            }
            logger.info(f"Legacy comparison: Legacy Corr={corr:.4f}, Scaled RMSE={legacy_rmse:.4f} vs XGBoost RMSE={overall_metrics['rmse']:.4f}")

    logger.info(f"Holdout Test RMSE: {overall_metrics['rmse']} | R²: {overall_metrics['r2']} | Exact Acc: {overall_metrics['exact_accuracy']*100:.1f}%")
    return {
        "overall_test_metrics": overall_metrics,
        "slice_analysis": slices,
        "legacy_heuristic_comparison": legacy_comp
    }
