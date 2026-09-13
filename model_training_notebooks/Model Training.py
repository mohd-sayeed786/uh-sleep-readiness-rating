# Databricks notebook source
# DBTITLE 1,Model Training — XGBoost with Temporal Validation
# MAGIC %md
# MAGIC # Model Training — XGBoost with Temporal Validation
# MAGIC
# MAGIC Anomaly detection → temporal split → baseline XGBoost → Optuna tuning → overfitting check → feature reduction → final evaluation → save model.

# COMMAND ----------

# DBTITLE 1,Install packages and load data
import subprocess, sys, warnings
warnings.filterwarnings('ignore')

# Install packages if needed
for pkg, install_name in [('xgboost', 'xgboost-cpu'), ('optuna', 'optuna'), ('shap', 'shap')]:
    try:
        __import__(pkg)
        print(f"  {pkg} already installed")
    except ImportError:
        print(f"  Installing {install_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_name, "-q"])
        print(f"  {pkg} installed")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

DATA_DIR = "/Workspace/Users/mohammad.sayeed@tatadigital.com/AS_UH_Project/data"

# Load data
df = pd.read_parquet(f"{DATA_DIR}/feature_engineered_table.parquet")
print(f"\nLoaded: {df.shape[0]:,} rows \u00d7 {df.shape[1]} columns")

# Target
TARGET = "subjective_feeling"
y = df[TARGET]

# Exclude columns (leakage/meta)
exclude_cols = [
    TARGET, "checkin_date", "user_id", "session_id", "date",
    "legacy_readiness_shown", "mood_category", "feeling_delta",
    "good_sleep_streak", "bad_sleep_streak", "holiday_name",
    "day_name", "firmware_version", "sleep_onset_period", "wake_period"
]
exclude_cols = [c for c in exclude_cols if c in df.columns]
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Encode categorical columns
cat_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    print(f"\nEncoding {len(cat_cols)} categorical columns:")
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  {col}: {len(le.classes_)} classes")

print(f"\nExcluded: {len(exclude_cols)} columns")
print(f"Feature candidates: {len(feature_cols)}")
print(f"\nTarget distribution:")
display(y.value_counts().sort_index().to_frame("count"))

# COMMAND ----------

# DBTITLE 1,Anomaly detection on feature set
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Anomaly Detection on Feature Set
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
X_all = df[feature_cols]
anomaly_report = []

# 1. Missing values
null_pct = X_all.isnull().mean() * 100
high_null = null_pct[null_pct > 50]
for feat in high_null.index:
    anomaly_report.append({"Feature": feat, "Issue": f"High missing ({null_pct[feat]:.1f}%)",
                           "Action": "Flag (XGBoost handles NaN)"})
print(f"1. Missing values: {(null_pct > 0).sum()} features have NaNs, "
      f"{len(high_null)} have >50% missing")

# 2. Constant / near-constant
nunique = X_all.nunique()
constant_feats = nunique[nunique < 2].index.tolist()
near_constant = []
for col in X_all.select_dtypes(include=[np.number]).columns:
    if col in constant_feats:
        continue
    mode_pct = X_all[col].value_counts(normalize=True).iloc[0] * 100
    if mode_pct > 99:
        near_constant.append(col)
        anomaly_report.append({"Feature": col, "Issue": f"Near-constant ({mode_pct:.1f}% same value)",
                               "Action": "Drop"})
for col in constant_feats:
    anomaly_report.append({"Feature": col, "Issue": "Constant (0 variance)", "Action": "Drop"})
print(f"2. Constant: {len(constant_feats)} | Near-constant (>99%): {len(near_constant)}")

# 3. Infinite values
inf_counts = np.isinf(X_all.select_dtypes(include=[np.number])).sum()
inf_feats = inf_counts[inf_counts > 0]
for feat in inf_feats.index:
    anomaly_report.append({"Feature": feat, "Issue": f"Infinite values ({inf_feats[feat]})",
                           "Action": "Replace with NaN"})
print(f"3. Infinite values: {len(inf_feats)} features")

# 4. Outlier detection (z-score)
outlier_feats = []
for col in X_all.select_dtypes(include=[np.number]).columns:
    valid = X_all[col].dropna()
    if len(valid) < 10 or valid.std() == 0:
        continue
    z = np.abs((valid - valid.mean()) / valid.std())
    outlier_pct = (z > 3).mean() * 100
    if outlier_pct > 10:
        outlier_feats.append(col)
        anomaly_report.append({"Feature": col, "Issue": f"High outliers ({outlier_pct:.1f}% |z|>3)",
                               "Action": "Flag (tree models robust)"})
print(f"4. Outlier features (>10% |z|>3): {len(outlier_feats)}")

# 5. High cardinality categoricals
cat_remaining = X_all.select_dtypes(include=['object', 'category']).columns
high_card = [c for c in cat_remaining if X_all[c].nunique() > 50]
print(f"5. High cardinality categoricals: {len(high_card)}")

# Summary
anomaly_df = pd.DataFrame(anomaly_report)
if len(anomaly_df) > 0:
    print(f"\n{'='*70}")
    print(f"ANOMALY SUMMARY: {len(anomaly_df)} issues across {anomaly_df['Feature'].nunique()} features")
    print(f"{'='*70}")
    display(anomaly_df)
else:
    print("\n\u2705 No critical anomalies detected.")

# Cleanup: drop constant + near-constant, replace inf with NaN
drop_feats = constant_feats + near_constant
if drop_feats:
    feature_cols = [c for c in feature_cols if c not in drop_feats]
    print(f"\nDropped {len(drop_feats)} constant/near-constant features \u2192 {len(feature_cols)} remaining")

X_clean = df[feature_cols].replace([np.inf, -np.inf], np.nan)
print(f"\nFinal feature set: {len(feature_cols)} features, {X_clean.isnull().sum().sum():,} total NaN values")
print(f"  (XGBoost handles NaN natively \u2014 no imputation needed)")

# COMMAND ----------

# DBTITLE 1,Temporal train / validation / test split
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Temporal Train / Validation / Test Split
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
dates = pd.to_datetime(df["checkin_date"])
unique_dates = sorted(dates.dropna().unique())
n_dates = len(unique_dates)

# 70% train, 15% val, 15% test by DATE
train_cutoff = unique_dates[int(n_dates * 0.70) - 1]
val_cutoff = unique_dates[int(n_dates * 0.85) - 1]

train_mask = dates <= train_cutoff
val_mask = (dates > train_cutoff) & (dates <= val_cutoff)
test_mask = dates > val_cutoff

X_train = X_clean[train_mask].copy()
X_val = X_clean[val_mask].copy()
X_test = X_clean[test_mask].copy()
y_train = y[train_mask].copy()
y_val = y[val_mask].copy()
y_test = y[test_mask].copy()

print(f"{'='*70}")
print(f"TEMPORAL SPLIT (by checkin_date)")
print(f"{'='*70}")
print(f"\n  Train: {len(X_train):,} rows | {dates[train_mask].min().date()} \u2192 {dates[train_mask].max().date()}")
print(f"  Val:   {len(X_val):,} rows  | {dates[val_mask].min().date()} \u2192 {dates[val_mask].max().date()}")
print(f"  Test:  {len(X_test):,} rows  | {dates[test_mask].min().date()} \u2192 {dates[test_mask].max().date()}")
print(f"\n  Features: {X_train.shape[1]}")
print(f"\n  Target distribution:")
for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    dist = y_split.value_counts().sort_index()
    pcts = (dist / len(y_split) * 100).round(1)
    print(f"    {split_name}: " + " | ".join([f"{k}:{v}({pcts[k]}%)" for k, v in dist.items()]))

# COMMAND ----------

# DBTITLE 1,Optuna-tuned XGBoost (from Model Improvement Experiments)
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Optuna-Tuned XGBoost (from Model Improvement Experiments v2) \u2014 Default Hyperparameters
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
def compute_metrics(y_true, y_pred, n_features, label=""):
    """Compute regression + classification metrics."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if n > n_features + 1 else r2
    y_cls = np.clip(np.round(y_pred), 1, 5).astype(int)
    exact_acc = accuracy_score(y_true.astype(int), y_cls)
    acc_pm1 = np.mean(np.abs(y_true - y_cls) <= 1)
    return {"Split": label, "RMSE": rmse, "MAE": mae, "R\u00b2": r2, "Adj R\u00b2": adj_r2,
            "Exact Acc": exact_acc, "Acc \u00b11": acc_pm1, "N": n}

n_feats = X_train.shape[1]

# Best params from Optuna (80 trials in Model Improvement Experiments v2)
best_params = {
    "max_depth": 3,
    "learning_rate": 0.02676795328269727,
    "n_estimators": 1699,
    "subsample": 0.6688454150272058,
    "colsample_bytree": 0.4615855771876036,
    "reg_alpha": 0.001241859408864291,
    "reg_lambda": 0.0698072326507175,
    "min_child_weight": 6,
    "gamma": 0.8706521942241263,
}

tuned_model = xgb.XGBRegressor(
    **best_params, random_state=42, n_jobs=-1, verbosity=0,
    early_stopping_rounds=50
)
tuned_model.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False)

# Metrics
metrics = []
for X_s, y_s, nm in [(X_train, y_train, "Train"), (X_val, y_val, "Val"), (X_test, y_test, "Test")]:
    metrics.append(compute_metrics(y_s, tuned_model.predict(X_s), n_feats, nm))
metrics_df = pd.DataFrame(metrics)

print(f"{'='*70}")
print(f"OPTUNA-TUNED XGBoost (from Model Improvement Experiments v2)")
print(f"  Best iteration: {tuned_model.best_iteration}")
print(f"  Params: max_depth={best_params['max_depth']}, lr={best_params['learning_rate']:.5f}, "
      f"n_est={best_params['n_estimators']}, subsample={best_params['subsample']:.3f}")
print(f"{'='*70}")
display(metrics_df.round(4))

# Overfitting check
train_r2 = metrics_df.loc[metrics_df['Split']=='Train', 'R\u00b2'].values[0]
val_r2 = metrics_df.loc[metrics_df['Split']=='Val', 'R\u00b2'].values[0]
test_r2 = metrics_df.loc[metrics_df['Split']=='Test', 'R\u00b2'].values[0]
gap_tv = train_r2 - val_r2
gap_tt = train_r2 - test_r2
print(f"\nOVERFITTING ANALYSIS:")
print(f"  Train R\u00b2: {train_r2:.4f}")
print(f"  Val R\u00b2:   {val_r2:.4f}  (gap: {gap_tv:.4f})")
print(f"  Test R\u00b2:  {test_r2:.4f}  (gap: {gap_tt:.4f})")
if gap_tv > 0.10:
    print(f"  \u26a0\ufe0f OVERFIT (train-val gap > 0.10)")
else:
    print(f"  \u2705 Not overfit (train-val gap \u2264 0.10)")

# Learning curves
results = tuned_model.evals_result()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(results['validation_0']['rmse'], label='Train RMSE', alpha=0.7)
ax.plot(results['validation_1']['rmse'], label='Val RMSE', alpha=0.7)
ax.axvline(x=tuned_model.best_iteration, color='r', linestyle='--', alpha=0.5,
           label=f'Best iter: {tuned_model.best_iteration}')
ax.set_xlabel('Boosting Round'); ax.set_ylabel('RMSE')
ax.set_title('Optuna-Tuned XGBoost \u2014 Learning Curves')
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Optuna tuning — SKIPPED (pre-tuned)
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Optuna Hyperparameter Tuning (50 trials)
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
print("Optuna tuning skipped — using pre-tuned params from Model Improvement Experiments v2 (80 trials). See cell 5 for best_params.")

# COMMAND ----------

# DBTITLE 1,Tuned model — detailed test set analysis
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Train Tuned Model + Overfitting Analysis
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Tuned model is already trained in cell 5 — this cell does detailed test analysis
print(f"{'='*70}")
print(f"DETAILED TEST SET ANALYSIS")
print(f"{'='*70}")

# Recap metrics from cell 5
print(f"\nMetrics recap (from cell 5):")
display(metrics_df.round(4))

# Confusion matrix on test set
y_test_pred = tuned_model.predict(X_test)
y_test_cls = np.clip(np.round(y_test_pred), 1, 5).astype(int)
y_test_int = y_test.astype(int)

cm = confusion_matrix(y_test_int, y_test_cls, labels=[1, 2, 3, 4, 5])
cm_df = pd.DataFrame(cm, index=[f"True={i}" for i in range(1, 6)],
                       columns=[f"Pred={i}" for i in range(1, 6)])
print(f"\nConfusion Matrix (Test Set):")
display(cm_df)

# Per-class accuracy
print(f"\nPer-class accuracy (Test Set):")
for cls in range(1, 6):
    mask = y_test_int == cls
    if mask.sum() > 0:
        cls_acc = accuracy_score(y_test_int[mask], y_test_cls[mask])
        cls_pm1 = np.mean(np.abs(y_test_int[mask] - y_test_cls[mask]) <= 1)
        print(f"  Class {cls}: n={mask.sum():>4d}  Exact={cls_acc:.1%}  \u00b11={cls_pm1:.1%}")

# Residual analysis
residuals = y_test.values - y_test_pred
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix')

# Residuals distribution
axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Residual (actual - predicted)'); axes[1].set_ylabel('Count')
axes[1].set_title(f'Residuals (mean={residuals.mean():.3f}, std={residuals.std():.3f})')

# Actual vs Predicted
axes[2].scatter(y_test.values, y_test_pred, alpha=0.3, s=10)
axes[2].plot([1, 5], [1, 5], 'r--', label='Perfect')
axes[2].set_xlabel('Actual'); axes[2].set_ylabel('Predicted')
axes[2].set_title('Actual vs Predicted'); axes[2].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Compare: exact top-30 baseline vs new top-30 with Tier 1+2
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Feature Reduction \u2014 Iterative Backward Elimination
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# ======================================================================
# A vs B Comparison: Exact Top-30 Baseline vs New Top-30 (with Tier 1+2)
# ======================================================================

# Tier 1+2 feature names
tier12_features = [
    'hr_z_7d', 'hrv_z_7d', 'sleep_z_7d', 'efficiency_z_7d', 'deep_z_7d', 'rem_z_7d',
    'hr_recovery_gap', 'hrv_recovery_gap', 'deep_recovery_gap', 'eff_recovery_gap',
    'stress_index_log', 'stress_index_z', 'recovery_score', 'good_recovery_flag',
    'hr_volatility', 'hrv_volatility', 'sleep_volatility', 'bedtime_volatility',
    'efficiency_volatility', 'deep_volatility',
    'cumul_sleep_debt_7d', 'deep_rem_ratio', 'wake_consistency', 'social_jetlag',
    'hr_user_ratio', 'hrv_user_ratio', 'sleep_user_ratio', 'eff_user_ratio',
    'deep_user_ratio', 'consistency_composite'
]

# EXACT 30 features from experiment notebook (baseline)
exact_top30 = [
    'alcohol_level', 'alcohol_units', 'had_alcohol', 'deep_rem_total',
    'sleep_debt', 'total_sleep_minutes_zscore', 'rem_minutes_zscore',
    'avg_hr_bpm_zscore', 'avg_hrv_rmssd_ms_zscore', 'restorative_pct',
    'alcohol_x_hrv_z', 'deep_minutes_zscore', 'user_expanding_mean',
    'feeling_ewm_7', 'days_into_study', 'submit_hour', 'feeling_roll7_mean',
    'subjective_feeling_lag3', 'light_pct', 'feeling_roll14_mean',
    'feeling_ewm_3', 'feeling_roll3_mean', 'days_since_great_sleep',
    'feeling_lag1', 'temp_instability', 'feeling_lag3', 'week_of_year',
    'feeling_roll5_mean', 'efficiency_duration', 'fragmented_night'
]

def train_and_eval(feats, label):
    """Train XGB on given features, return metrics dict."""
    m = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0,
                          early_stopping_rounds=50)
    m.fit(X_train[feats], y_train,
          eval_set=[(X_train[feats], y_train), (X_val[feats], y_val)],
          verbose=False)
    row = {"Model": label, "N_Feats": len(feats), "Best_Iter": m.best_iteration}
    for nm, X_s, y_s in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        preds = m.predict(X_s[feats])
        r2 = r2_score(y_s, preds)
        rmse = np.sqrt(mean_squared_error(y_s, preds))
        exact = np.mean(np.clip(np.round(preds), 1, 5).astype(int) == y_s.astype(int).values)
        pm1 = np.mean(np.abs(y_s.values - np.clip(np.round(preds), 1, 5).astype(int)) <= 1)
        row[f"{nm}_R2"] = r2; row[f"{nm}_RMSE"] = rmse
        row[f"{nm}_Exact"] = exact; row[f"{nm}_PM1"] = pm1
    row["Overfit"] = row["Train_R2"] - row["Val_R2"]
    return m, row

# ── MODEL A: Exact top-30 baseline ──
print("Training Model A (exact top-30 baseline)...")
model_A, row_A = train_and_eval(exact_top30, "A: Exact Top-30 (baseline)")

# ── Select NEW top-30 from ALL 260 features ──
print("Training full-feature model to get new importance ranking...")
full_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0,
                               early_stopping_rounds=50)
full_model.fit(X_train, y_train,
               eval_set=[(X_val, y_val)], verbose=False)
imp_new = pd.Series(full_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
new_top30 = imp_new.head(30).index.tolist()

# ── MODEL B: New top-30 (includes Tier 1+2 candidates) ──
print("Training Model B (new top-30 with Tier 1+2)...")
model_B, row_B = train_and_eval(new_top30, "B: New Top-30 (Tier 1+2)")

# ── Comparison table ──
comp_df = pd.DataFrame([row_A, row_B])
print(f"\n{'='*90}")
print(f"MODEL COMPARISON: Exact Top-30 Baseline vs New Top-30 (with Tier 1+2 features)")
print(f"{'='*90}")
display(comp_df.round(4))

# ── Deltas ──
print(f"\n{'-'*70}")
print(f"IMPROVEMENT (B vs A):")
print(f"{'-'*70}")
for metric in ['Train_R2', 'Val_R2', 'Test_R2', 'Test_RMSE', 'Test_Exact', 'Test_PM1', 'Overfit']:
    a, b = row_A[metric], row_B[metric]
    delta = b - a
    better = (delta > 0) if metric not in ['Test_RMSE', 'Overfit'] else (delta < 0)
    arrow = '\u2191' if better else '\u2193' if delta != 0 else '\u2192'
    if 'Exact' in metric or 'PM1' in metric:
        print(f"  {metric:15s}: {a:.1%} \u2192 {b:.1%}  ({delta:+.1%}) {arrow}")
    else:
        print(f"  {metric:15s}: {a:.4f} \u2192 {b:.4f}  ({delta:+.4f}) {arrow}")

# ── Which Tier 1+2 features made it into new top-30? ──
tier12_in_top30 = [f for f in new_top30 if f in tier12_features]
tier12_not_in = [f for f in tier12_features if f not in new_top30]
print(f"\n{'='*70}")
print(f"TIER 1+2 FEATURES IN NEW TOP-30: {len(tier12_in_top30)} of {len(tier12_features)}")
print(f"{'='*70}")
for f in tier12_in_top30:
    rank = new_top30.index(f) + 1
    print(f"  #{rank:2d}  {f}  (importance={imp_new[f]:.6f})")

print(f"\nTier 1+2 features NOT in top-30: {len(tier12_not_in)}")
for f in sorted(tier12_not_in, key=lambda x: imp_new.get(x, 0), reverse=True)[:10]:
    rank = list(imp_new.index).index(f) + 1 if f in imp_new.index else -1
    print(f"  #{rank:3d}  {f}  (importance={imp_new.get(f, 0):.6f})")

# ── New top-30 full list ──
print(f"\n{'='*70}")
print(f"NEW TOP-30 FEATURES (by importance from 260-feature model):")
print(f"{'='*70}")
for i, f in enumerate(new_top30, 1):
    tag = " [NEW Tier1+2]" if f in tier12_features else ""
    in_old = " [was in baseline]" if f in exact_top30 else ""
    print(f"  {i:2d}. {f:<45s} imp={imp_new[f]:.6f}{tag}{in_old}")

print(f"\n\u26d4 STOP: Review comparison above. Tell me how to proceed.")

# === OLD REDUCTION CODE REMOVED ===
DO_NOT_RUN = '''
# Feature reduction starting from TOP-30 enriched model (R²≈0.695)
# Same top-30 features as Model Improvement Experiments v2
import time

print(f"{'='*70}")
print(f"FEATURE REDUCTION — Starting from Top-30 Enriched Model")
print(f"{'='*70}")

# Step 1: Select top-30 features by importance from the 230-feature model
imp_all = pd.Series(tuned_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top30 = imp_all.head(30).index.tolist()
print(f"\nTop 30 features selected from 230-feature importance ranking:")
for i, f in enumerate(top30, 1):
    print(f"  {i:2d}. {f}")

# Step 2: Train the Optuna-tuned model on top-30 features (baseline)
model_30 = xgb.XGBRegressor(
    **best_params, random_state=42, n_jobs=-1, verbosity=0,
    early_stopping_rounds=50
)
model_30.fit(X_train[top30], y_train,
             eval_set=[(X_train[top30], y_train), (X_val[top30], y_val)],
             verbose=False)

# Baseline metrics for 30-feature model
def get_all_metrics(model, feats, X_tr, y_tr, X_v, y_v, X_te, y_te):
    results = {}
    for nm, X_s, y_s in [("Train", X_tr, y_tr), ("Val", X_v, y_v), ("Test", X_te, y_te)]:
        preds = model.predict(X_s[feats])
        rmse = np.sqrt(mean_squared_error(y_s, preds))
        r2 = r2_score(y_s, preds)
        exact = np.mean(np.clip(np.round(preds), 1, 5).astype(int) == y_s.astype(int).values)
        pm1 = np.mean(np.abs(y_s.values - np.clip(np.round(preds), 1, 5).astype(int)) <= 1)
        results[nm] = {"RMSE": rmse, "R2": r2, "Exact": exact, "PM1": pm1}
    return results

base_m = get_all_metrics(model_30, top30, X_train, y_train, X_val, y_val, X_test, y_test)
print(f"\n{'='*70}")
print(f"TOP-30 BASELINE MODEL")
print(f"  Best iteration: {model_30.best_iteration}")
print(f"  Train: R\u00b2={base_m['Train']['R2']:.4f}  RMSE={base_m['Train']['RMSE']:.4f}  Exact={base_m['Train']['Exact']:.1%}")
print(f"  Val:   R\u00b2={base_m['Val']['R2']:.4f}  RMSE={base_m['Val']['RMSE']:.4f}  Exact={base_m['Val']['Exact']:.1%}")
print(f"  Test:  R\u00b2={base_m['Test']['R2']:.4f}  RMSE={base_m['Test']['RMSE']:.4f}  Exact={base_m['Test']['Exact']:.1%}  \u00b11={base_m['Test']['PM1']:.1%}")
print(f"  Overfit gap: {base_m['Train']['R2'] - base_m['Val']['R2']:.4f}")
print(f"{'='*70}")

# Step 3: Iterative backward elimination from 30 down to 5
print(f"\nReducing features one-at-a-time from 30...\n")

current_feats = top30.copy()
reduction_log = []
feature_sets = {}

# Log baseline
reduction_log.append({
    "N": 30, "Val_RMSE": base_m['Val']['RMSE'], "Val_R2": base_m['Val']['R2'],
    "Test_RMSE": base_m['Test']['RMSE'], "Test_R2": base_m['Test']['R2'],
    "Test_Exact": base_m['Test']['Exact'], "Test_PM1": base_m['Test']['PM1'],
    "Train_R2": base_m['Train']['R2'], "Overfit": base_m['Train']['R2'] - base_m['Val']['R2']
})
feature_sets[30] = top30.copy()

t0 = time.time()
for target_n in range(29, 4, -1):  # 29, 28, 27, ..., 5
    # Train on current features to get importances
    m_iter = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0,
                               early_stopping_rounds=30)
    m_iter.fit(X_train[current_feats], y_train,
               eval_set=[(X_val[current_feats], y_val)], verbose=False)
    imp = pd.Series(m_iter.feature_importances_, index=current_feats)
    drop_feat = imp.idxmin()  # remove least important
    remaining = [f for f in current_feats if f != drop_feat]

    # Retrain and evaluate with remaining features
    m_eval = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0,
                               early_stopping_rounds=30)
    m_eval.fit(X_train[remaining], y_train,
               eval_set=[(X_train[remaining], y_train), (X_val[remaining], y_val)],
               verbose=False)
    ms = get_all_metrics(m_eval, remaining, X_train, y_train, X_val, y_val, X_test, y_test)

    reduction_log.append({
        "N": len(remaining), "Val_RMSE": ms['Val']['RMSE'], "Val_R2": ms['Val']['R2'],
        "Test_RMSE": ms['Test']['RMSE'], "Test_R2": ms['Test']['R2'],
        "Test_Exact": ms['Test']['Exact'], "Test_PM1": ms['Test']['PM1'],
        "Train_R2": ms['Train']['R2'], "Overfit": ms['Train']['R2'] - ms['Val']['R2']
    })
    feature_sets[len(remaining)] = remaining.copy()

    marker = " \u2b50" if ms['Test']['R2'] >= base_m['Test']['R2'] else ""
    print(f"  {len(remaining):2d} feats  Val R\u00b2={ms['Val']['R2']:.4f}  Test R\u00b2={ms['Test']['R2']:.4f}  "
          f"Exact={ms['Test']['Exact']:.1%}  \u00b11={ms['Test']['PM1']:.1%}  "
          f"(dropped: {drop_feat}){marker}")

    current_feats = remaining

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s")

# Results table
red_df = pd.DataFrame(reduction_log)
print(f"\n{'='*70}")
print(f"REDUCTION SUMMARY")
print(f"{'='*70}")
display(red_df.round(4))

# Elbow chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(red_df['N'], red_df['Test_R2'], 'o-', markersize=6, color='#e74c3c', label='Test R\u00b2')
axes[0].plot(red_df['N'], red_df['Val_R2'], 's--', markersize=5, color='#3498db', alpha=0.7, label='Val R\u00b2')
best_idx = red_df['Test_R2'].idxmax()
axes[0].axvline(x=red_df.loc[best_idx, 'N'], color='green', linestyle=':', alpha=0.7,
                label=f"Best: {int(red_df.loc[best_idx, 'N'])} feats")
axes[0].set_xlabel('Number of Features'); axes[0].set_ylabel('R\u00b2')
axes[0].set_title('R\u00b2 vs Feature Count'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].invert_xaxis()

axes[1].plot(red_df['N'], red_df['Test_Exact']*100, 'o-', markersize=6, color='#2ecc71')
axes[1].set_xlabel('Features'); axes[1].set_ylabel('Exact Accuracy (%)')
axes[1].set_title('Exact Accuracy vs Feature Count'); axes[1].invert_xaxis()
axes[1].grid(True, alpha=0.3)

axes[2].plot(red_df['N'], red_df['Overfit'], 'o-', markersize=6, color='#9b59b6')
axes[2].axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Overfit threshold')
axes[2].set_xlabel('Features'); axes[2].set_ylabel('Train-Val R\u00b2 Gap')
axes[2].set_title('Overfitting vs Feature Count'); axes[2].invert_xaxis()
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Best point
best_row = red_df.loc[best_idx]
print(f"\n{'='*70}")
print(f"\u2b50 BEST TEST R\u00b2 = {best_row['Test_R2']:.4f} at {int(best_row['N'])} features")
print(f"   Test RMSE={best_row['Test_RMSE']:.4f}, Exact={best_row['Test_Exact']:.1%}, \u00b11={best_row['Test_PM1']:.1%}")
print(f"   Overfit gap: {best_row['Overfit']:.4f}")
print(f"\n\u26d4 STOP: Review the reduction curve above.")
print(f"   Tell me which feature count to use for the final model.")
print(f"{'='*70}")
'''  # end DO_NOT_RUN

# COMMAND ----------

# DBTITLE 1,Backward elimination: new top-30 down to 5 features
# ══════════════════════════════════════════════════════════════
# Backward Elimination — New Top-30 (with Tier 1+2) → 5 features
# ══════════════════════════════════════════════════════════════
import time

print(f"{'='*70}")
print(f"BACKWARD ELIMINATION — Starting from New Top-30 (Model B)")
print(f"  Baseline Test R²=0.6975, Exact=66.3%")
print(f"{'='*70}")

# Use the new_top30 from cell 8
print(f"\nStarting features ({len(new_top30)}):")
for i, f in enumerate(new_top30, 1):
    tag = " [Tier1+2]" if f in tier12_features else ""
    print(f"  {i:2d}. {f}{tag}")

# Baseline: train on all 30
model_base, row_base = train_and_eval(new_top30, "30 feats (baseline)")
reduction_log = [{
    "N": 30, "Dropped": "-",
    "Train_R2": row_base["Train_R2"], "Val_R2": row_base["Val_R2"],
    "Test_R2": row_base["Test_R2"], "Test_RMSE": row_base["Test_RMSE"],
    "Test_Exact": row_base["Test_Exact"], "Test_PM1": row_base["Test_PM1"],
    "Overfit": row_base["Overfit"]
}]
feature_sets = {30: new_top30.copy()}

current_feats = new_top30.copy()
t0 = time.time()

for target_n in range(29, 4, -1):  # 29, 28, ..., 5
    # Train on current features to get importances
    m_iter = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0,
                               early_stopping_rounds=30)
    m_iter.fit(X_train[current_feats], y_train,
               eval_set=[(X_val[current_feats], y_val)], verbose=False)
    imp = pd.Series(m_iter.feature_importances_, index=current_feats)
    drop_feat = imp.idxmin()  # remove least important
    remaining = [f for f in current_feats if f != drop_feat]

    # Retrain and evaluate with remaining features
    m_eval, row_eval = train_and_eval(remaining, f"{len(remaining)} feats")

    reduction_log.append({
        "N": len(remaining), "Dropped": drop_feat,
        "Train_R2": row_eval["Train_R2"], "Val_R2": row_eval["Val_R2"],
        "Test_R2": row_eval["Test_R2"], "Test_RMSE": row_eval["Test_RMSE"],
        "Test_Exact": row_eval["Test_Exact"], "Test_PM1": row_eval["Test_PM1"],
        "Overfit": row_eval["Overfit"]
    })
    feature_sets[len(remaining)] = remaining.copy()

    marker = " ⭐" if row_eval["Test_R2"] >= row_base["Test_R2"] else ""
    tier_tag = " [T1+2]" if drop_feat in tier12_features else ""
    print(f"  {len(remaining):2d} feats  Train R²={row_eval['Train_R2']:.4f}  Val R²={row_eval['Val_R2']:.4f}  "
          f"Test R²={row_eval['Test_R2']:.4f}  Exact={row_eval['Test_Exact']:.1%}  "
          f"±1={row_eval['Test_PM1']:.1%}  Overfit={row_eval['Overfit']:.4f}  "
          f"(dropped: {drop_feat}{tier_tag}){marker}")

    current_feats = remaining

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s")

# ── Results Table ──
red_df = pd.DataFrame(reduction_log)
print(f"\n{'='*70}")
print(f"REDUCTION SUMMARY")
print(f"{'='*70}")
display(red_df.round(4))

# ── Elbow Charts ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# R² curve
axes[0].plot(red_df['N'], red_df['Test_R2'], 'o-', markersize=6, color='#e74c3c', label='Test R²')
axes[0].plot(red_df['N'], red_df['Val_R2'], 's--', markersize=5, color='#3498db', alpha=0.7, label='Val R²')
axes[0].plot(red_df['N'], red_df['Train_R2'], '^:', markersize=4, color='#2ecc71', alpha=0.5, label='Train R²')
best_idx = red_df['Test_R2'].idxmax()
axes[0].axvline(x=red_df.loc[best_idx, 'N'], color='green', linestyle=':', alpha=0.7,
                label=f"Best: {int(red_df.loc[best_idx, 'N'])} feats (R²={red_df.loc[best_idx, 'Test_R2']:.4f})")
axes[0].set_xlabel('Number of Features'); axes[0].set_ylabel('R²')
axes[0].set_title('R² vs Feature Count'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
axes[0].invert_xaxis()

# Exact accuracy
axes[1].plot(red_df['N'], red_df['Test_Exact']*100, 'o-', markersize=6, color='#2ecc71', label='Exact %')
axes[1].plot(red_df['N'], red_df['Test_PM1']*100, 's--', markersize=5, color='#9b59b6', alpha=0.7, label='±1 %')
axes[1].set_xlabel('Features'); axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy vs Feature Count'); axes[1].invert_xaxis()
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Overfit gap
axes[2].plot(red_df['N'], red_df['Overfit'], 'o-', markersize=6, color='#e67e22')
axes[2].axhline(y=0.10, color='red', linestyle='--', alpha=0.5, label='Overfit threshold (0.10)')
axes[2].set_xlabel('Features'); axes[2].set_ylabel('Train-Val R² Gap')
axes[2].set_title('Overfitting vs Feature Count'); axes[2].invert_xaxis()
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ── Highlight best and key points ──
best_row = red_df.loc[best_idx]
print(f"\n{'='*70}")
print(f"⭐ BEST TEST R² = {best_row['Test_R2']:.4f} at {int(best_row['N'])} features")
print(f"   Test RMSE={best_row['Test_RMSE']:.4f}, Exact={best_row['Test_Exact']:.1%}, ±1={best_row['Test_PM1']:.1%}")
print(f"   Overfit gap: {best_row['Overfit']:.4f}")

# Show key checkpoints
print(f"\n{'='*70}")
print(f"KEY CHECKPOINTS:")
print(f"{'='*70}")
for n_feat in [30, 25, 20, 15, 13, 10, 8, 5]:
    row_check = red_df[red_df['N'] == n_feat]
    if len(row_check) > 0:
        r = row_check.iloc[0]
        feats_at_n = feature_sets.get(n_feat, [])
        tier12_count = sum(1 for f in feats_at_n if f in tier12_features)
        print(f"  {int(r['N']):2d} feats: Test R²={r['Test_R2']:.4f}  "
              f"Exact={r['Test_Exact']:.1%}  ±1={r['Test_PM1']:.1%}  "
              f"Overfit={r['Overfit']:.4f}  ({tier12_count} Tier1+2)")

# Show feature lists at key points
for n_feat in [20, 15, 13, 10]:
    if n_feat in feature_sets:
        feats_at_n = feature_sets[n_feat]
        print(f"\n  Features at {n_feat}:")
        for i, f in enumerate(feats_at_n, 1):
            tag = " [T1+2]" if f in tier12_features else ""
            print(f"    {i:2d}. {f}{tag}")

print(f"\n{'='*70}")
print(f"⛔ STOP: Review the reduction curve and checkpoints above.")
print(f"   Tell me which feature count to use for the final model.")
print(f"{'='*70}")

# COMMAND ----------

# DBTITLE 1,Focused Optuna tuning on 21 features + final evaluation
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Final Evaluation on Test Set
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
import optuna, time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 21-feature set from backward elimination
final_feats = feature_sets[21]
print(f"Final 21 features:")
for i, f in enumerate(final_feats, 1):
    tag = " [T1+2]" if f in tier12_features else ""
    print(f"  {i:2d}. {f}{tag}")

# -- Focused Optuna Search (narrowed around current best params) --
print(f"\n{'='*70}")
print(f"FOCUSED OPTUNA TUNING (50 trials on 21 features)")
print(f"{'='*70}")

def objective_21(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 800, 2500),
        'subsample': trial.suggest_float('subsample', 0.55, 0.85),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.35, 0.7),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
        'gamma': trial.suggest_float('gamma', 0.3, 1.5),
    }
    m = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0,
                          early_stopping_rounds=50)
    m.fit(X_train[final_feats], y_train,
          eval_set=[(X_val[final_feats], y_val)], verbose=False)
    preds = m.predict(X_val[final_feats])
    return np.sqrt(mean_squared_error(y_val, preds))

t0 = time.time()
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective_21, n_trials=50)
elapsed = time.time() - t0

new_params = study.best_params
print(f"\nOptuna done in {elapsed:.0f}s")
print(f"Best Val RMSE: {study.best_value:.4f}")
print(f"New params: {new_params}")

# -- Compare: original params vs tuned params on 21 features --
print(f"\n{'='*70}")
print(f"COMPARISON: Original vs Tuned Params (21 features)")
print(f"{'='*70}")

_, row_orig = train_and_eval(final_feats, "21-feat (original params)")

def train_and_eval_params(feats, params, label):
    m = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0,
                          early_stopping_rounds=50)
    m.fit(X_train[feats], y_train,
          eval_set=[(X_train[feats], y_train), (X_val[feats], y_val)],
          verbose=False)
    row = {"Model": label, "N_Feats": len(feats), "Best_Iter": m.best_iteration}
    for nm, X_s, y_s in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        preds = m.predict(X_s[feats])
        r2 = r2_score(y_s, preds)
        rmse = np.sqrt(mean_squared_error(y_s, preds))
        exact = np.mean(np.clip(np.round(preds), 1, 5).astype(int) == y_s.astype(int).values)
        pm1 = np.mean(np.abs(y_s.values - np.clip(np.round(preds), 1, 5).astype(int)) <= 1)
        row[f"{nm}_R2"] = r2; row[f"{nm}_RMSE"] = rmse
        row[f"{nm}_Exact"] = exact; row[f"{nm}_PM1"] = pm1
    row["Overfit"] = row["Train_R2"] - row["Val_R2"]
    return m, row

model_tuned21, row_tuned = train_and_eval_params(final_feats, new_params, "21-feat (Optuna-tuned)")

comp = pd.DataFrame([row_orig, row_tuned])
display(comp.round(4))

# Pick the better model
if row_tuned['Test_R2'] >= row_orig['Test_R2']:
    print(f"\nTuned params IMPROVED Test R2: {row_orig['Test_R2']:.4f} -> {row_tuned['Test_R2']:.4f}")
    final_model = model_tuned21
    final_params = new_params
    final_row = row_tuned
else:
    print(f"\nTuned params did NOT improve. Keeping original params.")
    print(f"  Original Test R2={row_orig['Test_R2']:.4f} vs Tuned={row_tuned['Test_R2']:.4f}")
    final_model_obj, _ = train_and_eval(final_feats, "final")
    final_model = final_model_obj
    final_params = best_params
    final_row = row_orig

# -- FINAL TEST SET EVALUATION --
print(f"\n{'='*70}")
print(f"FINAL MODEL EVALUATION (21 features)")
print(f"{'='*70}")

y_test_pred = final_model.predict(X_test[final_feats])
y_test_cls = np.clip(np.round(y_test_pred), 1, 5).astype(int)
y_test_int = y_test.astype(int)

print(f"  Train R2: {final_row['Train_R2']:.4f}")
print(f"  Val R2:   {final_row['Val_R2']:.4f}")
print(f"  Test R2:  {final_row['Test_R2']:.4f}")
print(f"  Test RMSE: {final_row['Test_RMSE']:.4f}")
print(f"  Exact Acc: {final_row['Test_Exact']:.1%}")
print(f"  +/-1 Acc:  {final_row['Test_PM1']:.1%}")
print(f"  Overfit:   {final_row['Overfit']:.4f}")

# Classification report
labels = sorted(y_test_int.unique())
print(f"\nClassification Report (Reduced Model \u2014 rounded to 1-5):")
print(classification_report(y_test_int, y_test_cls, labels=labels,
                            target_names=[f"Feeling {l}" for l in labels]))
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion matrix + plots
cm = confusion_matrix(y_test_int, y_test_cls, labels=labels)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
axes[0].set_title('Confusion Matrix (21-Feature Model)')

axes[1].scatter(y_test, y_test_pred, alpha=0.3, s=10)
axes[1].plot([1, 5], [1, 5], 'r--', label='Perfect')
axes[1].set_xlabel('Actual Feeling'); axes[1].set_ylabel('Predicted Feeling')
axes[1].set_title('Actual vs Predicted'); axes[1].legend()

residuals = y_test.values - y_test_pred
axes[2].hist(residuals, bins=50, edgecolor='white', alpha=0.7)
axes[2].axvline(x=0, color='r', linestyle='--')
axes[2].set_xlabel('Residual'); axes[2].set_ylabel('Count')
axes[2].set_title(f'Residuals (mean={residuals.mean():.4f}, std={residuals.std():.4f})')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Save final 21-feature model + summary
import pickle, os, json

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Save Final Model + Summary
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
save_dir = f"{DATA_DIR}/../selected_model"
os.makedirs(save_dir, exist_ok=True)

# 1. Save model
pkl_path = f"{save_dir}/xgboost_tuned_reduced.pkl"
with open(pkl_path, 'wb') as f:
    pickle.dump(final_model, f)
size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
print(f"Model saved: {pkl_path} ({size_mb:.2f} MB)")

# 2. Save feature list
feat_path = f"{save_dir}/feature_list.json"
with open(feat_path, 'w') as f:
    json.dump(final_feats, f, indent=2)
print(f"Feature list saved: {feat_path} ({len(final_feats)} features)")

# 3. Save metadata
metadata = {
    "model_type": "XGBRegressor",
    "n_features": len(final_feats),
    "n_features_original": len(feature_cols),
    "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else
                        float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in final_params.items()},
    "best_iteration": int(final_model.best_iteration),
    "train_rows": int(len(X_train)),
    "val_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "test_metrics": {
        "R2": round(final_row['Test_R2'], 4),
        "RMSE": round(final_row['Test_RMSE'], 4),
        "Exact_Acc": round(final_row['Test_Exact'], 4),
        "PM1_Acc": round(final_row['Test_PM1'], 4),
        "Train_R2": round(final_row['Train_R2'], 4),
        "Val_R2": round(final_row['Val_R2'], 4),
        "Overfit": round(final_row['Overfit'], 4),
    },
    "split_type": "temporal (70/15/15 by date)",
    "target": TARGET,
}
meta_path = f"{save_dir}/model_metadata.json"
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"Metadata saved: {meta_path}")

# Summary
print(f"\n{'='*70}")
print(f"MODEL TRAINING PIPELINE - COMPLETE SUMMARY")
print(f"{'='*70}")
print(f"\n  Data: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Target: {TARGET} (1-5 scale)")
print(f"  Split: Temporal (70/15/15)")
print(f"  Features: {len(feature_cols)} -> {len(final_feats)} (after reduction)")
print(f"\n  Optuna: {len(study.trials)} trials, best Val RMSE={study.best_value:.4f}")
print(f"  Best iteration: {final_model.best_iteration}")
print(f"\n  Test Set Performance:")
print(f"    R2:        {final_row['Test_R2']:.4f}")
print(f"    RMSE:      {final_row['Test_RMSE']:.4f}")
print(f"    Exact Acc: {final_row['Test_Exact']:.1%}")
print(f"    Acc +/-1:  {final_row['Test_PM1']:.1%}")
print(f"    Train R2:  {final_row['Train_R2']:.4f}")
print(f"    Overfit:   {final_row['Overfit']:.4f}")
print(f"\n  Model saved to: {save_dir}/")
print(f"    - xgboost_tuned_reduced.pkl ({size_mb:.2f} MB)")
print(f"    - feature_list.json ({len(final_feats)} features)")
print(f"    - model_metadata.json")

print(f"\n  Final 21 features:")
for i, f in enumerate(final_feats, 1):
    tag = " [T1+2]" if f in tier12_features else ""
    print(f"    {i:2d}. {f}{tag}")