"""
Training pipeline for Readiness Score prediction.
Includes temporal splitting, baseline models, XGBoost training, artifact versioning, and rollback.
"""
import json
import pickle
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

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
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data_pipeline import run_data_pipeline
from src.features import engineer_features
from src.logger import get_logger

logger = get_logger("train")


# ---------------------------------------------------------------------------
# Version Control & Rollback Utilities
# ---------------------------------------------------------------------------

def get_next_archive_version(versions_dir: Optional[Path] = None) -> str:
    """Determine the next incremental version string (v1.0, v1.1, v1.2...)."""
    vdir = versions_dir or MODEL_VERSIONS_DIR
    pattern = re.compile(r"selected_model_v(\d+)\.(\d+)\.pkl$")
    max_major, max_minor = 1, 0
    found = False

    # Check both versions_dir and parent MODEL_DIR for existing version files
    search_dirs = [vdir]
    if vdir != MODEL_DIR and MODEL_DIR.exists():
        search_dirs.append(MODEL_DIR)

    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.glob("selected_model_v*.pkl"):
            m = pattern.search(p.name)
            if m:
                found = True
                maj, min_ = int(m.group(1)), int(m.group(2))
                if (maj, min_) > (max_major, max_minor):
                    max_major, max_minor = maj, min_

    if not found:
        return "v1.0"
    next_ver = f"v{max_major}.{max_minor + 1}"
    logger.debug(f"Determined next archive version: {next_ver}")
    return next_ver


def archive_existing_model(
    model_dir: Optional[Path] = None,
    versions_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    If selected_model.pkl exists, renames/moves it to the versions folder
    as selected_model_vX.X.pkl (e.g. selected_model/versions/selected_model_v1.0.pkl).
    Also archives its corresponding model_metadata.json and feature_list.json.
    Returns the archived version string, or None if no prior model existed.
    """
    mdir = model_dir or MODEL_DIR
    vdir = versions_dir or MODEL_VERSIONS_DIR
    vdir.mkdir(parents=True, exist_ok=True)

    current_model = mdir / "selected_model.pkl"
    current_meta = mdir / "model_metadata.json"
    current_features = mdir / "feature_list.json"

    # Fallback to legacy file if selected_model.pkl was never initialized
    if not current_model.exists():
        legacy_model = mdir / "xgboost_tuned_reduced.pkl"
        if legacy_model.exists():
            logger.info("Initializing active selected_model.pkl from benchmark xgboost_tuned_reduced.pkl")
            shutil.copy(str(legacy_model), str(current_model))

    if current_model.exists():
        archive_version = get_next_archive_version(vdir)
        archived_model = vdir / f"selected_model_{archive_version}.pkl"
        archived_meta = vdir / f"model_metadata_{archive_version}.json"
        archived_feats = vdir / f"feature_list_{archive_version}.json"

        # Move active model to versioned folder (never delete)
        shutil.move(str(current_model), str(archived_model))
        if current_meta.exists():
            shutil.copy(str(current_meta), str(archived_meta))
        if current_features.exists():
            shutil.copy(str(current_features), str(archived_feats))

        logger.info(f"Archived previous active model to versions folder as: {archived_model.name}")
        return archive_version
    return None


def rollback_model(
    target_version: str,
    model_dir: Optional[Path] = None,
    versions_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Roll back the active selected_model.pkl and feature_list.json to a previously archived version in versions/ (e.g. 'v1.0').
    """
    mdir = model_dir or MODEL_DIR
    vdir = versions_dir or MODEL_VERSIONS_DIR
    clean_ver = target_version if target_version.startswith("v") else f"v{target_version}"

    logger.info(f"Initiating model rollback to version {clean_ver}...")

    # Search in versions directory first, then root model directory
    target_model = vdir / f"selected_model_{clean_ver}.pkl"
    target_meta = vdir / f"model_metadata_{clean_ver}.json"
    target_feats = vdir / f"feature_list_{clean_ver}.json"

    if not target_model.exists():
        fallback_model = mdir / f"selected_model_{clean_ver}.pkl"
        if fallback_model.exists():
            target_model = fallback_model
            target_meta = mdir / f"model_metadata_{clean_ver}.json"
            target_feats = mdir / f"feature_list_{clean_ver}.json"
        else:
            available = [p.name for p in vdir.glob("selected_model_v*.pkl")] + \
                        [p.name for p in mdir.glob("selected_model_v*.pkl")]
            err_msg = f"Archived model selected_model_{clean_ver}.pkl not found. Available versions: {available}"
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

    # Archive current active model before restoring
    archived_curr = archive_existing_model(mdir, vdir)

    # Restore target version as active selected_model.pkl
    active_model = mdir / "selected_model.pkl"
    active_meta = mdir / "model_metadata.json"
    active_feats = mdir / "feature_list.json"

    shutil.copy(str(target_model), str(active_model))
    if target_meta.exists():
        shutil.copy(str(target_meta), str(active_meta))
    if target_feats.exists():
        shutil.copy(str(target_feats), str(active_feats))

    # Read restored feature list if available
    restored_features = []
    if active_feats.exists():
        try:
            with open(active_feats, "r") as f:
                restored_features = json.load(f)
        except Exception:
            pass

    logger.info(f"Rollback successful: {clean_ver} restored to {active_model.name} with {len(restored_features)} features")
    return {
        "status": "success",
        "message": f"Successfully rolled back active model to {clean_ver}",
        "active_model": str(active_model),
        "restored_from": str(target_model),
        "restored_version": clean_ver,
        "features": restored_features,
        "previous_active_archived_as": archived_curr,
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class GlobalMeanBaseline:
    """Naive baseline predicting the training population mean."""
    def __init__(self):
        self.mean_value = 3.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.mean_value = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_value)


class UserHistoricalMeanBaseline:
    """Predicts the user's historical training mean, with fallback to global mean."""
    def __init__(self):
        self.global_mean = 3.0
        self.user_means: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, user_ids: pd.Series):
        self.global_mean = float(y.mean())
        user_grouped = pd.DataFrame({"user_id": user_ids, "y": y}).groupby("user_id")["y"].mean()
        self.user_means = user_grouped.to_dict()
        return self

    def predict(self, user_ids: pd.Series) -> np.ndarray:
        return np.array([self.user_means.get(uid, self.global_mean) for uid in user_ids])


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    split_name: str = "Test",
    n_features: int = len(FEATURE_NAMES),
) -> Dict[str, Any]:
    """Compute regression and discrete class accuracy metrics."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))

    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    n = len(y_true_arr)
    p = n_features
    adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else r2

    rounded_preds = np.clip(np.round(y_pred_arr), 1, 5).astype(int)
    exact_acc = float(np.mean(rounded_preds == y_true_arr))
    acc_pm1 = float(np.mean(np.abs(rounded_preds - y_true_arr) <= 1))

    return {
        "split": split_name,
        "n_samples": n,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "adj_r2": round(adj_r2, 4),
        "exact_accuracy": round(exact_acc, 4),
        "accuracy_pm1": round(acc_pm1, 4),
    }


def split_temporal(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe chronologically by checkin_date to prevent forward leakage (D-005)."""
    unique_dates = np.sort(df["checkin_date"].unique())
    n_dates = len(unique_dates)

    train_cutoff_idx = int(n_dates * train_ratio) - 1
    val_cutoff_idx = int(n_dates * (train_ratio + val_ratio)) - 1

    train_cutoff = unique_dates[train_cutoff_idx]
    val_cutoff = unique_dates[val_cutoff_idx]

    train_df = df[df["checkin_date"] <= train_cutoff].copy()
    val_df = df[(df["checkin_date"] > train_cutoff) & (df["checkin_date"] <= val_cutoff)].copy()
    test_df = df[df["checkin_date"] > val_cutoff].copy()

    logger.debug(f"Temporal Split: Train <= {train_cutoff} ({len(train_df)} rows) | Val <= {val_cutoff} ({len(val_df)} rows) | Test > {val_cutoff} ({len(test_df)} rows)")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------

def train_model(
    custom_params: Optional[Dict[str, Any]] = None,
    feature_cols: Optional[List[str]] = None,
    save_artifacts: bool = True,
    model_save_path: Optional[Path] = None,
    metadata_save_path: Optional[Path] = None,
    feature_list_save_path: Optional[Path] = None,
    versions_dir: Optional[Path] = None,
    save_model: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run the end-to-end training pipeline:
      1. Clean and join raw datasets (D-008 through D-022)
      2. Engineer features (or use custom feature_cols)
      3. Split data chronologically (70/15/15)
      4. Fit XGBoost Regressor with early stopping
      5. Calculate train, validation, and test performance
      6. Archive prior model & feature_list to version folder
      7. Save new model as active selected_model.pkl and update feature_list.json
    """
    if save_model is not None:
        save_artifacts = save_model

    start_time = time.time()
    logger.info("Starting model training pipeline...")
    np.random.seed(RANDOM_SEED)

    # 1. Pipeline execution
    base_df = run_data_pipeline()
    feature_df = engineer_features(base_df)

    labelled_df = feature_df.dropna(subset=[TARGET_COL]).copy()

    # 2. Features selection (sanitize and filter out dummy/placeholder values such as Swagger's 'string')
    if feature_cols is not None:
        valid_cols = [c.strip() for c in feature_cols if c and c.strip() and c.strip().lower() != "string"]
        features_to_use = valid_cols if valid_cols else FEATURE_NAMES
    else:
        features_to_use = FEATURE_NAMES

    missing_cols = [c for c in features_to_use if c not in labelled_df.columns]
    if missing_cols:
        err_msg = f"Requested features not found in engineered table: {missing_cols}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info(f"Training configured with {len(features_to_use)} features: {features_to_use}")

    # 3. Temporal Split
    train_df, val_df, test_df = split_temporal(labelled_df)

    X_train = train_df[features_to_use]
    y_train = train_df[TARGET_COL]

    X_val = val_df[features_to_use]
    y_val = val_df[TARGET_COL]

    X_test = test_df[features_to_use]
    y_test = test_df[TARGET_COL]

    # 4. Baselines Evaluation
    global_baseline = GlobalMeanBaseline().fit(X_train, y_train)
    user_baseline = UserHistoricalMeanBaseline().fit(X_train, y_train, train_df["user_id"])

    global_test_preds = global_baseline.predict(X_test)
    user_test_preds = user_baseline.predict(test_df["user_id"])

    global_metrics = compute_metrics(y_test, global_test_preds, split_name="Global Mean Floor", n_features=len(features_to_use))
    user_metrics = compute_metrics(y_test, user_test_preds, split_name="User Mean Baseline", n_features=len(features_to_use))

    logger.info(f"Baseline (Global Mean Floor): RMSE={global_metrics['rmse']} | MAE={global_metrics['mae']}")
    logger.info(f"Baseline (User Mean Baseline): RMSE={user_metrics['rmse']} | MAE={user_metrics['mae']}")

    # 5. Train XGBoost
    params = DEFAULT_XGB_PARAMS.copy()
    if custom_params:
        params.update(custom_params)

    logger.info(f"Fitting XGBoost regressor (max_depth={params.get('max_depth')}, lr={params.get('learning_rate'):.4f}, n_est={params.get('n_estimators')})...")
    model = XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    best_iteration = getattr(model, "best_iteration", params.get("n_estimators", 900))

    # 6. Evaluate XGBoost
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    train_metrics = compute_metrics(y_train, train_preds, split_name="Train", n_features=len(features_to_use))
    val_metrics = compute_metrics(y_val, val_preds, split_name="Validation", n_features=len(features_to_use))
    test_metrics = compute_metrics(y_test, test_preds, split_name="Test", n_features=len(features_to_use))

    logger.info(f"XGBoost Test Metrics: RMSE={test_metrics['rmse']} | R²={test_metrics['r2']} | Exact Acc={test_metrics['exact_accuracy']*100:.1f}% | Acc±1={test_metrics['accuracy_pm1']*100:.1f}%")

    elapsed_time = round(time.time() - start_time, 2)

    results = {
        "model_type": "XGBRegressor",
        "n_features": len(features_to_use),
        "features": features_to_use,
        "best_params": params,
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "baselines": {
                "global_mean": global_metrics,
                "user_mean": user_metrics,
            },
        },
        "overfitting_gap": round(train_metrics["r2"] - val_metrics["r2"], 4),
        "training_time_seconds": elapsed_time,
    }

    # 7. Save Artifacts with Version Control in versions folder
    if save_artifacts:
        save_model_file = model_save_path or MODEL_PATH
        save_meta_file = metadata_save_path or MODEL_METADATA_PATH
        save_feat_file = feature_list_save_path or FEATURE_LIST_PATH
        vdir = versions_dir or MODEL_VERSIONS_DIR

        save_model_file.parent.mkdir(parents=True, exist_ok=True)
        vdir.mkdir(parents=True, exist_ok=True)

        # Archive existing selected_model.pkl and feature_list.json into versions/
        archived_ver = archive_existing_model(save_model_file.parent, vdir)
        results["previous_model_archived_as"] = archived_ver
        results["versions_directory"] = str(vdir)

        # Save new active model strictly as selected_model.pkl
        with open(save_model_file, "wb") as f:
            pickle.dump(model, f)

        # Save the specific features used by this model
        with open(save_feat_file, "w") as f:
            json.dump(features_to_use, f, indent=2)

        with open(save_meta_file, "w") as f:
            json.dump(results, f, indent=2)

        results["model_artifact_path"] = str(save_model_file)
        results["feature_list_path"] = str(save_feat_file)
        results["metadata_artifact_path"] = str(save_meta_file)
        logger.info(f"New model saved to {save_model_file} (previous archived as {archived_ver})")

    logger.info(f"Training pipeline finished in {elapsed_time}s")
    return results
