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
for pkg in ['xgboost', 'optuna', 'shap']:
    try:
        __import__(pkg)
        print(f"  {pkg} already installed")
    except ImportError:
        print(f"  Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "numpy<2", "-q"])
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

# DBTITLE 1,Baseline XGBoost training + overfitting check
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Baseline XGBoost \u2014 Default Hyperparameters
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

baseline_model = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=7,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    random_state=42, n_jobs=-1, verbosity=0,
    early_stopping_rounds=50
)
baseline_model.fit(X_train, y_train,
                   eval_set=[(X_train, y_train), (X_val, y_val)],
                   verbose=False)

# Metrics
metrics = []
for X_s, y_s, nm in [(X_train, y_train, "Train"), (X_val, y_val, "Val"), (X_test, y_test, "Test")]:
    metrics.append(compute_metrics(y_s, baseline_model.predict(X_s), n_feats, nm))
metrics_df = pd.DataFrame(metrics)

print(f"{'='*70}")
print(f"BASELINE XGBoost (n_estimators=500, max_depth=7, lr=0.05)")
print(f"  Best iteration: {baseline_model.best_iteration}")
print(f"{'='*70}")
display(metrics_df.round(4))

# Overfitting check
train_r2 = metrics_df.loc[metrics_df['Split']=='Train', 'R\u00b2'].values[0]
val_r2 = metrics_df.loc[metrics_df['Split']=='Val', 'R\u00b2'].values[0]
gap = train_r2 - val_r2
print(f"\n\u26a0\ufe0f Overfitting check: Train R\u00b2={train_r2:.4f}, Val R\u00b2={val_r2:.4f}, Gap={gap:.4f}")
if gap > 0.05:
    print(f"  \u2192 Model IS overfit (gap > 0.05). Optuna tuning should add regularization.")
else:
    print(f"  \u2192 Model is NOT overfit (gap \u2264 0.05).")

# Learning curves
results = baseline_model.evals_result()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(results['validation_0']['rmse'], label='Train RMSE', alpha=0.7)
ax.plot(results['validation_1']['rmse'], label='Val RMSE', alpha=0.7)
ax.axvline(x=baseline_model.best_iteration, color='r', linestyle='--', alpha=0.5,
           label=f'Best iter: {baseline_model.best_iteration}')
ax.set_xlabel('Boosting Round'); ax.set_ylabel('RMSE')
ax.set_title('Baseline XGBoost \u2014 Learning Curves')
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Optuna hyperparameter tuning (50 trials)
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Optuna Hyperparameter Tuning (50 trials)
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }
    model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0,
                             early_stopping_rounds=30)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

study = optuna.create_study(direction="minimize", study_name="xgboost_tuning")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"{'='*70}")
print(f"OPTUNA TUNING RESULTS (50 trials)")
print(f"{'='*70}")
print(f"  Best Val RMSE: {study.best_value:.4f}")
print(f"\n  Best Parameters:")
for k, v in study.best_params.items():
    print(f"    {k}: {v}")

# Optimization history
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
vals = [t.value for t in study.trials]
axes[0].plot(vals, 'o-', alpha=0.6, markersize=3)
axes[0].axhline(y=study.best_value, color='r', linestyle='--', label=f"Best: {study.best_value:.4f}")
axes[0].set_xlabel("Trial"); axes[0].set_ylabel("Val RMSE")
axes[0].set_title("Optimization History"); axes[0].legend()

importances = optuna.importance.get_param_importances(study)
top_p = dict(list(importances.items())[:8])
axes[1].barh(list(top_p.keys())[::-1], list(top_p.values())[::-1])
axes[1].set_xlabel("Importance"); axes[1].set_title("Hyperparameter Importance")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Train tuned model + overfitting analysis
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Train Tuned Model + Overfitting Analysis
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
best_params = study.best_params.copy()

tuned_model = xgb.XGBRegressor(
    **best_params, random_state=42, n_jobs=-1, verbosity=0,
    early_stopping_rounds=50
)
tuned_model.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False)

tuned_metrics = []
for X_s, y_s, nm in [(X_train, y_train, "Train"), (X_val, y_val, "Val"), (X_test, y_test, "Test")]:
    tuned_metrics.append(compute_metrics(y_s, tuned_model.predict(X_s), n_feats, nm))
tuned_df = pd.DataFrame(tuned_metrics)

print(f"{'='*70}")
print(f"TUNED XGBoost (Optuna best params)")
print(f"  Best iteration: {tuned_model.best_iteration}")
print(f"{'='*70}")
display(tuned_df.round(4))

# Overfitting analysis
t_r2 = tuned_df.loc[tuned_df['Split']=='Train', 'R\u00b2'].values[0]
v_r2 = tuned_df.loc[tuned_df['Split']=='Val', 'R\u00b2'].values[0]
te_r2 = tuned_df.loc[tuned_df['Split']=='Test', 'R\u00b2'].values[0]
gap_tv = t_r2 - v_r2
gap_tt = t_r2 - te_r2

print(f"\n{'\u2500'*60}")
print(f"OVERFITTING ANALYSIS")
print(f"{'\u2500'*60}")
print(f"  Train R\u00b2:  {t_r2:.4f}")
print(f"  Val R\u00b2:    {v_r2:.4f}   (train-val gap: {gap_tv:.4f})")
print(f"  Test R\u00b2:   {te_r2:.4f}   (train-test gap: {gap_tt:.4f})")
if gap_tv > 0.05:
    print(f"\n  \u26a0\ufe0f Model IS OVERFIT (train-val R\u00b2 gap = {gap_tv:.4f} > 0.05)")
else:
    print(f"\n  \u2705 Model is NOT OVERFIT (train-val R\u00b2 gap = {gap_tv:.4f} \u2264 0.05)")

# Learning curves
results = tuned_model.evals_result()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(results['validation_0']['rmse'], label='Train RMSE', alpha=0.7)
ax.plot(results['validation_1']['rmse'], label='Val RMSE', alpha=0.7)
ax.axvline(x=tuned_model.best_iteration, color='r', linestyle='--', alpha=0.5,
           label=f'Best iter: {tuned_model.best_iteration}')
ax.set_xlabel('Boosting Round'); ax.set_ylabel('RMSE')
ax.set_title('Tuned XGBoost \u2014 Learning Curves')
ax.legend()
plt.tight_layout()
plt.show()

# Baseline vs Tuned comparison
print(f"\n{'\u2500'*60}")
print(f"BASELINE vs TUNED (Val Set)")
print(f"{'\u2500'*60}")
baseline_val = metrics_df[metrics_df['Split']=='Val'].iloc[0]
tuned_val = tuned_df[tuned_df['Split']=='Val'].iloc[0]
for col in ['RMSE', 'MAE', 'R\u00b2', 'Adj R\u00b2', 'Exact Acc', 'Acc \u00b11']:
    b, t = baseline_val[col], tuned_val[col]
    better = '\u2191' if (t > b if col not in ['RMSE','MAE'] else t < b) else '\u2193'
    print(f"  {col:12s}: Baseline={b:.4f} \u2192 Tuned={t:.4f}  {better}")

# COMMAND ----------

# DBTITLE 1,Feature reduction — iterative backward elimination
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Feature Reduction \u2014 Iterative Backward Elimination
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
print(f"Starting feature reduction from {len(feature_cols)} features...")
print(f"Using Optuna-tuned parameters\n")

current_feats = feature_cols.copy()
reduction_log = []

# Baseline val RMSE with all features
preds_all = tuned_model.predict(X_val)
best_rmse = np.sqrt(mean_squared_error(y_val, preds_all))
best_r2 = r2_score(y_val, preds_all)
reduction_log.append({"N_Features": len(current_feats), "Val_RMSE": best_rmse, "Val_R2": best_r2})
print(f"  {len(current_feats)} features \u2192 Val RMSE={best_rmse:.4f}, R\u00b2={best_r2:.4f}")

threshold_rmse = best_rmse * 1.05  # Stop if RMSE increases >5%

while len(current_feats) > 10:
    model_iter = xgb.XGBRegressor(
        **best_params, random_state=42, n_jobs=-1, verbosity=0,
        early_stopping_rounds=30
    )
    model_iter.fit(X_train[current_feats], y_train,
                   eval_set=[(X_val[current_feats], y_val)], verbose=False)

    imp = pd.Series(model_iter.feature_importances_, index=current_feats)
    n_remove = max(1, int(len(current_feats) * 0.10))
    remove_feats = imp.nsmallest(n_remove).index.tolist()
    remaining = [f for f in current_feats if f not in remove_feats]

    # Evaluate with remaining features
    model_eval = xgb.XGBRegressor(
        **best_params, random_state=42, n_jobs=-1, verbosity=0,
        early_stopping_rounds=30
    )
    model_eval.fit(X_train[remaining], y_train,
                   eval_set=[(X_val[remaining], y_val)], verbose=False)
    preds = model_eval.predict(X_val[remaining])
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    reduction_log.append({"N_Features": len(remaining), "Val_RMSE": rmse, "Val_R2": r2})

    print(f"  {len(remaining):3d} features \u2192 Val RMSE={rmse:.4f}, R\u00b2={r2:.4f} (removed {n_remove})")

    if rmse > threshold_rmse:
        print(f"\n  \u26a0\ufe0f RMSE exceeded 5% threshold ({rmse:.4f} > {threshold_rmse:.4f}) \u2014 reverting")
        break
    current_feats = remaining

# Elbow chart
red_df = pd.DataFrame(reduction_log)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(red_df['N_Features'], red_df['Val_RMSE'], 'o-', markersize=5)
axes[0].axhline(y=threshold_rmse, color='r', linestyle='--', alpha=0.5,
                label=f"5% threshold: {threshold_rmse:.4f}")
axes[0].set_xlabel('Number of Features'); axes[0].set_ylabel('Val RMSE')
axes[0].set_title('Feature Reduction \u2014 RMSE vs Feature Count')
axes[0].legend(); axes[0].invert_xaxis()

axes[1].plot(red_df['N_Features'], red_df['Val_R2'], 'o-', markersize=5, color='green')
axes[1].set_xlabel('Number of Features'); axes[1].set_ylabel('Val R\u00b2')
axes[1].set_title('Feature Reduction \u2014 R\u00b2 vs Feature Count')
axes[1].invert_xaxis()
plt.tight_layout()
plt.show()

# Train final reduced model
reduced_feats = current_feats
print(f"\n{'='*70}")
print(f"REDUCED MODEL: {len(reduced_feats)} features (from {len(feature_cols)})")
print(f"{'='*70}")

reduced_model = xgb.XGBRegressor(
    **best_params, random_state=42, n_jobs=-1, verbosity=0,
    early_stopping_rounds=50
)
reduced_model.fit(X_train[reduced_feats], y_train,
                  eval_set=[(X_train[reduced_feats], y_train),
                            (X_val[reduced_feats], y_val)],
                  verbose=False)

red_metrics = []
for X_s, y_s, nm in [(X_train, y_train, "Train"), (X_val, y_val, "Val"), (X_test, y_test, "Test")]:
    red_metrics.append(compute_metrics(y_s, reduced_model.predict(X_s[reduced_feats]), len(reduced_feats), nm))
red_metrics_df = pd.DataFrame(red_metrics)
display(red_metrics_df.round(4))

print(f"\nTop 20 features in reduced model:")
imp_final = pd.Series(reduced_model.feature_importances_, index=reduced_feats)
for i, (feat, score) in enumerate(imp_final.nlargest(20).items(), 1):
    print(f"  {i:2d}. {feat:45s} importance={score:.4f}")

# COMMAND ----------

# DBTITLE 1,Final evaluation on test set
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Final Evaluation on Test Set
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
y_test_pred = reduced_model.predict(X_test[reduced_feats])
y_test_cls = np.clip(np.round(y_test_pred), 1, 5).astype(int)
y_test_int = y_test.astype(int)

# Full model predictions for comparison
y_test_full = tuned_model.predict(X_test)
y_test_full_cls = np.clip(np.round(y_test_full), 1, 5).astype(int)

print(f"{'='*70}")
print(f"FINAL TEST SET EVALUATION")
print(f"{'='*70}")
comp_data = []
for nm, preds, nf in [("Full Model", y_test_full, len(feature_cols)),
                       ("Reduced Model", y_test_pred, len(reduced_feats))]:
    comp_data.append(compute_metrics(y_test, preds, nf, nm))
comp_df = pd.DataFrame(comp_data)
display(comp_df.round(4))

# Classification report
labels = sorted(y_test_int.unique())
print(f"\nClassification Report (Reduced Model \u2014 rounded to 1-5):")
print(classification_report(y_test_int, y_test_cls, labels=labels,
                            target_names=[f"Feeling {l}" for l in labels]))

# Confusion matrix + plots
cm = confusion_matrix(y_test_int, y_test_cls, labels=labels)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')
axes[0].set_title('Confusion Matrix (Reduced Model)')

axes[1].scatter(y_test, y_test_pred, alpha=0.3, s=10)
axes[1].plot([1, 5], [1, 5], 'r--', label='Perfect')
axes[1].set_xlabel('Actual Feeling'); axes[1].set_ylabel('Predicted Feeling')
axes[1].set_title('Actual vs Predicted'); axes[1].legend()

residuals = y_test.values - y_test_pred
axes[2].hist(residuals, bins=50, edgecolor='white', alpha=0.7)
axes[2].axvline(x=0, color='r', linestyle='--')
axes[2].set_xlabel('Residual (Actual - Predicted)'); axes[2].set_ylabel('Count')
axes[2].set_title(f'Residuals (mean={residuals.mean():.4f}, std={residuals.std():.4f})')
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Save model and pipeline summary
import pickle, os, json

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Save Final Model + Summary
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
save_dir = f"{DATA_DIR}/../selected_model"
os.makedirs(save_dir, exist_ok=True)

# 1. Save model
pkl_path = f"{save_dir}/xgboost_tuned_reduced.pkl"
with open(pkl_path, 'wb') as f:
    pickle.dump(reduced_model, f)
size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
print(f"\u2713 Model saved: {pkl_path} ({size_mb:.2f} MB)")

# 2. Save feature list
feat_path = f"{save_dir}/feature_list.json"
with open(feat_path, 'w') as f:
    json.dump(reduced_feats, f, indent=2)
print(f"\u2713 Feature list saved: {feat_path} ({len(reduced_feats)} features)")

# 3. Save metadata
test_m = comp_df[comp_df['Split']=='Reduced Model'].iloc[0].to_dict()
metadata = {
    "model_type": "XGBRegressor",
    "n_features": len(reduced_feats),
    "n_features_original": len(feature_cols),
    "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else
                        float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in best_params.items()},
    "best_iteration": int(reduced_model.best_iteration),
    "train_rows": int(len(X_train)),
    "val_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "test_metrics": {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                     for k, v in test_m.items()},
    "split_type": "temporal (70/15/15 by date)",
    "target": TARGET,
}
meta_path = f"{save_dir}/model_metadata.json"
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2, default=str)
print(f"\u2713 Metadata saved: {meta_path}")

# Summary
print(f"\n{'='*70}")
print(f"MODEL TRAINING PIPELINE \u2014 COMPLETE SUMMARY")
print(f"{'='*70}")
print(f"\n  Data: {df.shape[0]:,} rows \u00d7 {df.shape[1]} columns")
print(f"  Target: {TARGET} (1-5 scale)")
print(f"  Split: Temporal (70/15/15)")
print(f"  Features: {len(feature_cols)} \u2192 {len(reduced_feats)} (after reduction)")
print(f"\n  Optuna: {len(study.trials)} trials, best Val RMSE={study.best_value:.4f}")
print(f"  Best iteration: {reduced_model.best_iteration}")
print(f"\n  Test Set Performance:")
print(f"    RMSE:      {test_m['RMSE']:.4f}")
print(f"    MAE:       {test_m['MAE']:.4f}")
print(f"    R\u00b2:        {test_m['R\u00b2']:.4f}")
print(f"    Adj R\u00b2:    {test_m['Adj R\u00b2']:.4f}")
print(f"    Exact Acc: {test_m['Exact Acc']:.1%}")
print(f"    Acc \u00b11:    {test_m['Acc \u00b11']:.1%}")
print(f"\n  Model saved to: {save_dir}/")
print(f"    \u251c\u2500\u2500 xgboost_tuned_reduced.pkl ({size_mb:.2f} MB)")
print(f"    \u251c\u2500\u2500 feature_list.json ({len(reduced_feats)} features)")
print(f"    \u2514\u2500\u2500 model_metadata.json")