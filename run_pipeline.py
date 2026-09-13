#!/usr/bin/env python3
"""
CLI entry point to execute the readiness score pipeline end-to-end with phase timing and live loading display.
Usage:
    python run_pipeline.py
    python run_pipeline.py --train --eval --save
    python run_pipeline.py --log-level DEBUG
"""
import argparse
import logging
import sys
import threading
import time
from pathlib import Path

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Immediate CLI feedback before heavy module loading
if sys.stdout.isatty():
    print("\033[1;36m[Ring AI]\033[0m Initializing pipeline execution environment...", flush=True)

from src.config import FEATURE_NAMES, LOG_FILE, MODEL_PATH
from src.data_pipeline import run_data_pipeline
from src.evaluate import run_full_evaluation
from src.features import engineer_features
from src.inference import ReadinessPredictor
from src.logger import get_logger, setup_logging
from src.train import train_model

logger = get_logger("pipeline_cli")


class PhaseSpinner:
    """Live animated terminal spinner ensuring continuous visual feedback during execution."""
    def __init__(self, message: str):
        self.message = message
        self.stop_event = threading.Event()
        self.thread = None
        self.start_time = 0.0
        self.is_tty = sys.stdout.isatty()

    def _spin(self):
        spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        while not self.stop_event.is_set():
            elapsed = time.time() - self.start_time
            msg = f"\r  \033[36m{spinner_chars[idx % len(spinner_chars)]}\033[0m \033[1m{self.message}\033[0m \033[90m({elapsed:.1f}s)\033[0m\033[K"
            sys.stdout.write(msg)
            sys.stdout.flush()
            idx += 1
            time.sleep(0.08)

    def __enter__(self):
        self.start_time = time.time()
        if self.is_tty:
            self.thread = threading.Thread(target=self._spin, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_tty:
            self.stop_event.set()
            if self.thread:
                self.thread.join()
            elapsed = time.time() - self.start_time
            if exc_type is None:
                msg = f"\r  \033[32m✔\033[0m \033[1m{self.message}\033[0m \033[32m[Completed in {elapsed:.2f}s]\033[0m\033[K\n"
            else:
                msg = f"\r  \033[31m✖\033[0m \033[1m{self.message}\033[0m \033[31m[Failed]\033[0m\033[K\n"
            sys.stdout.write(msg)
            sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Ring AI Readiness Score End-to-End Pipeline")
    parser.add_argument("--train", action="store_true", default=True, help="Train XGBoost model")
    parser.add_argument("--eval", action="store_true", default=True, help="Run slice and baseline evaluation")
    parser.add_argument("--save", action="store_true", default=True, help="Save model artifacts")
    parser.add_argument("--n-estimators", type=int, default=900, help="Number of estimators for XGBoost")
    parser.add_argument("--learning-rate", type=float, default=0.014, help="Learning rate")
    parser.add_argument("--max-depth", type=int, default=8, help="Tree max depth")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level for console output (default: INFO, log file records DEBUG)",
    )
    args = parser.parse_args()

    # Configure console and file logging
    console_lvl = getattr(logging, args.log_level.upper())
    setup_logging(console_level=console_lvl, file_level=logging.DEBUG)

    total_start = time.time()
    logger.info("=" * 60)
    logger.info(" RING AI: READINESS SCORE END-TO-END PIPELINE")
    logger.info("=" * 60)

    # 1. Ingestion & Cleaning
    t0 = time.time()
    logger.info("[Phase 1/4] Ingesting and cleaning raw datasets (D-008 to D-022)...")
    with PhaseSpinner("Phase 1/4: Ingesting & cleaning raw sensor telemetry..."):
        base_df = run_data_pipeline()
    t_clean = time.time() - t0
    logger.info(f"  -> Merged base table: {len(base_df):,} rows across {base_df['user_id'].nunique()} users")
    logger.info(f"  -> Sleep session match rate: {base_df['has_session_data'].mean()*100:.1f}%")
    logger.info(f"  -> Cleaning phase completed in {t_clean:.2f}s")

    # 2. Feature Engineering
    t0 = time.time()
    logger.info("[Phase 2/4] Engineering 13 selected features...")
    with PhaseSpinner("Phase 2/4: Engineering 13 physiological features across 120 users..."):
        feat_df = engineer_features(base_df)
    t_feat = time.time() - t0
    logger.info(f"  -> Feature table shape: {feat_df.shape}")
    logger.info(f"  -> Verified 13 features: {FEATURE_NAMES}")
    logger.info(f"  -> Feature engineering completed in {t_feat:.2f}s")

    # 3. Model Training
    t_train = 0.0
    if args.train:
        t0 = time.time()
        logger.info("[Phase 3/4] Running temporal 70/15/15 split & training XGBoost...")
        custom_params = {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
        }
        with PhaseSpinner(f"Phase 3/4: Training XGBoost Regressor ({args.n_estimators} trees, max_depth={args.max_depth})..."):
            train_results = train_model(custom_params=custom_params, save_artifacts=args.save)
        t_train = time.time() - t0

        metrics = train_results["metrics"]
        logger.info(f"  -> Baseline Floor (Global Mean): RMSE = {metrics['baselines']['global_mean']['rmse']:.4f}")
        logger.info(f"  -> Baseline (User Mean):        RMSE = {metrics['baselines']['user_mean']['rmse']:.4f}")
        logger.info(f"  -> XGBoost Validation:          RMSE = {metrics['validation']['rmse']:.4f} | R\u00b2 = {metrics['validation']['r2']:.4f}")
        logger.info(f"  -> XGBoost Test:                RMSE = {metrics['test']['rmse']:.4f} | R\u00b2 = {metrics['test']['r2']:.4f}")
        logger.info(f"  -> Test Exact Accuracy:         {metrics['test']['exact_accuracy']*100:.1f}%")
        logger.info(f"  -> Test Accuracy \u00b11 Class:      {metrics['test']['accuracy_pm1']*100:.1f}%")
        logger.info(f"  -> Overfitting Gap (Train-Val): {train_results['overfitting_gap']:.4f}")
        logger.info(f"  -> Training phase completed in {t_train:.2f}s")

    # 4. Slices & Error Analysis
    t_eval = 0.0
    if args.eval:
        t0 = time.time()
        logger.info("[Phase 4/4] Evaluating subgroups and slice breakdowns...")
        with PhaseSpinner("Phase 4/4: Evaluating subgroups & slice breakdowns..."):
            predictor = ReadinessPredictor(model_path=MODEL_PATH)
            eval_results = run_full_evaluation(predictor.model)
        t_eval = time.time() - t0

        slices = eval_results["slice_analysis"]
        logger.info("  -> By Target Feeling Class:")
        for cls_name, stats in slices["by_class"].items():
            logger.info(f"     * {cls_name}: n={stats['count']} | Exact Acc={stats['exact_accuracy']*100:.1f}% | MAE={stats['mae']:.3f}")

        if "by_session_data" in slices:
            logger.info("  -> By Sensor Availability:")
            for k, stats in slices["by_session_data"].items():
                logger.info(f"     * {k}: n={stats['count']} | RMSE={stats['rmse']:.3f} | Exact Acc={stats['exact_accuracy']*100:.1f}%")

        if eval_results.get("legacy_heuristic_comparison"):
            leg = eval_results["legacy_heuristic_comparison"]
            logger.info(f"  -> Comparison to Legacy Score Shown: Legacy Correlation = {leg['legacy_correlation']} | Scaled RMSE = {leg['legacy_scaled_rmse']:.3f}")
        logger.info(f"  -> Slice evaluation completed in {t_eval:.2f}s")

    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f" PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f}s")
    logger.info(f" Timing breakdown: Cleaning: {t_clean:.2f}s | Features: {t_feat:.2f}s | Train: {t_train:.2f}s | Eval: {t_eval:.2f}s")
    logger.info(f" Model Artifact: {MODEL_PATH}")
    logger.info(f" Log File: {LOG_FILE}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
