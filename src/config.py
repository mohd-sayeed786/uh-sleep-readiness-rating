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

# Selected 13 Features
FEATURE_NAMES: List[str] = [
    "alcohol_units",
    "had_alcohol",
    "alcohol_level",
    "week_of_year",
    "total_sleep_minutes_zscore",
    "avg_hr_bpm_zscore",
    "avg_hrv_rmssd_ms_zscore",
    "subjective_feeling_lag1",
    "days_since_bad_sleep",
    "days_since_great_sleep",
    "checkin_seq_num",
    "deep_rem_total",
    "sleep_debt",
]

# Random seed for reproducibility
RANDOM_SEED = 42

# Train / Val / Test Split Ratios (Temporal split by checkin date)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Default Tuned XGBoost Parameters from Optuna optimization
DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "max_depth": 8,
    "learning_rate": 0.013895075698868577,
    "n_estimators": 900,
    "subsample": 0.7559678262921059,
    "colsample_bytree": 0.818399922679552,
    "reg_alpha": 0.003804459928933268,
    "reg_lambda": 0.5719545331877559,
    "min_child_weight": 9,
    "gamma": 0.42693535444133257,
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
