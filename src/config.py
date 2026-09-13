"""
Configuration and settings for AS_UH_Project readiness scoring.
"""
from pathlib import Path
from typing import List, Dict, Any

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "selected_model"
MODEL_VERSIONS_DIR = MODEL_DIR / "versions"
MODEL_SELECTION_DIR = BASE_DIR / "model_selection"
TESTS_DIR = BASE_DIR / "tests"
LOGS_DIR = BASE_DIR / "logs"

# Log file path
LOG_FILE = LOGS_DIR / "app.log"

# File paths
SLEEP_SESSIONS_FILE = DATA_DIR / "sleep_sessions.csv"
NIGHTLY_SIGNALS_FILE = DATA_DIR / "nightly_signals.csv.gz"
USER_PROFILES_FILE = DATA_DIR / "user_profiles.csv"
DAILY_CONTEXT_FILE = DATA_DIR / "daily_context.csv"
MORNING_CHECKINS_FILE = DATA_DIR / "morning_checkins.csv"

# Saved Model Artifacts
MODEL_PATH = MODEL_DIR / "selected_model.pkl"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.json"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Target column
TARGET_COL = "subjective_feeling"

# Selected 21 Features (Enriched Tier 1+2 Model)
FEATURE_NAMES: List[str] = [
    "had_alcohol",
    "alcohol_level",
    "deep_rem_total",
    "total_sleep_minutes_zscore",
    "stress_index_z",
    "alcohol_units",
    "alcohol_x_hrv_z",
    "sleep_debt",
    "rem_minutes_zscore",
    "sleep_user_ratio",
    "recovery_score",
    "avg_hr_bpm_zscore",
    "deep_minutes_zscore",
    "restorative_pct",
    "avg_hrv_rmssd_ms_zscore",
    "feeling_roll5_mean",
    "feeling_ewm_7",
    "hrv_user_ratio",
    "deep_user_ratio",
    "user_expanding_mean",
    "hr_user_ratio",
]

# Random seed for reproducibility
RANDOM_SEED = 42

# Train / Val / Test Split Ratios (Temporal split by checkin date)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Default Tuned XGBoost Parameters from Optuna optimization (21-feature model)
DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "max_depth": 3,
    "learning_rate": 0.02676795328269727,
    "n_estimators": 1699,
    "subsample": 0.6688454150272058,
    "colsample_bytree": 0.4615855771876036,
    "reg_alpha": 0.001241859408864291,
    "reg_lambda": 0.0698072326507175,
    "min_child_weight": 6,
    "gamma": 0.8706521942241263,
    "random_state": RANDOM_SEED,
    "n_jobs": -1
}

# Physiological Plausibility Limits
PHYSIOLOGICAL_BOUNDS = {
    "avg_hr_bpm": (30.0, 120.0),
    "avg_hrv_rmssd_ms": (1.0, 300.0),
    "time_in_bed_minutes": (60.0, 900.0),
    "sleep_efficiency": (0.0, 1.0),
    "subjective_feeling": (1, 5)
}
