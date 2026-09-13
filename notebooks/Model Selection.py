# Databricks notebook source
# DBTITLE 1,Model Selection — Readiness Score
# MAGIC %md
# MAGIC # Model Selection — Readiness Score
# MAGIC
# MAGIC Compares LightGBM, XGBoost, Random Forest, Ridge Regression, SVR, and CatBoost using stratified K-fold cross-validation. Includes feature selection, SHAP analysis, and final model recommendation.

# COMMAND ----------

# DBTITLE 1,Install packages and setup
# Check what's already installed, only install what's missing
import subprocess, sys

needed = []
for pkg in ["lightgbm", "xgboost", "shap"]:
    try:
        __import__(pkg)
        print(f"  \u2713 {pkg} already installed")
    except ImportError:
        needed.append(pkg)
        print(f"  \u2717 {pkg} needs install")

if needed:
    print(f"\nInstalling: {needed}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + needed + ["numpy<2", "-q"])
    dbutils.library.restartPython()
else:
    print("\nAll packages ready - no install needed!")

# COMMAND ----------

# DBTITLE 1,Load data and prepare feature matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings("ignore")

DATA_DIR = "/Workspace/Users/mohammad.sayeed@tatadigital.com/AS_UH_Project/data"
df = pd.read_parquet(f"{DATA_DIR}/feature_engineered_table.parquet")
df["checkin_date"] = pd.to_datetime(df["checkin_date"])

print(f"Loaded: {df.shape[0]:,} rows \u00d7 {df.shape[1]} cols")
print(f"Target: subjective_feeling (1-5), mean={df['subjective_feeling'].mean():.3f}")

# ═══════════════════════════════════════════════════════
# Define columns to EXCLUDE from features
# ═══════════════════════════════════════════════════════
id_cols = ["user_id", "checkin_date", "session_id", "night_date", "context_date"]
target_col = "subjective_feeling"

# Leakage/validation columns — NOT model inputs
exclude_cols = [
    "legacy_readiness_shown",  # leakage (shown before answering)
    "mood_category",           # recorded at prediction time
    "feeling_delta",           # contains current target
    "good_sleep_streak",       # REMOVED: was leaky in V1 (included current target)
    "bad_sleep_streak",        # REMOVED: was leaky in V1 (included current target)
    "holiday_name",            # free text, not predictive
    "day_name",                # redundant with day_of_week
    "firmware_version",        # string, low signal
    "sleep_onset_period",      # string category
    "wake_period",             # string category
]

# String/category columns that need encoding
cat_encode_cols = ["sex", "plan_tier", "timezone", "country_code",
                   "workout_category", "age_group", "bmi_category",
                   "activity_level", "alcohol_level", "stress_level",
                   "stress_category"]

# All numeric feature candidates (exclude ids, target, leakage, strings)
non_feature_cols = set(id_cols + [target_col] + exclude_cols)
all_feature_cols = [c for c in df.columns if c not in non_feature_cols]

print(f"\nFeature candidates: {len(all_feature_cols)}")
print(f"Categorical to encode: {len(cat_encode_cols)}")
print(f"Excluded (leakage/meta): {len(exclude_cols)}")

# COMMAND ----------

# DBTITLE 1,Data preparation: encoding, null handling, splits
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ═══════════════════════════════════════════════════════
# A. Encode categoricals
# ═══════════════════════════════════════════════════════
le_dict = {}
for col in cat_encode_cols:
    if col in df.columns:
        le = LabelEncoder()
        mask = df[col].notna()
        df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
        df[col] = pd.to_numeric(df[col], errors="coerce")
        le_dict[col] = le
        print(f"  Encoded {col}: {len(le.classes_)} classes")

# ═══════════════════════════════════════════════════════
# B. Build X and y
# ═══════════════════════════════════════════════════════
feature_cols = [c for c in all_feature_cols if c in df.columns and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"\nFeature matrix: {X.shape}")
print(f"Target: {y.shape}")

# ═══════════════════════════════════════════════════════
# C. Null analysis
# ═══════════════════════════════════════════════════════
null_pct = X.isnull().mean().sort_values(ascending=False)
high_null = null_pct[null_pct > 0.80]  # >80% null (stress_score kept via stress_category)
if len(high_null):
    print(f"\nDropping {len(high_null)} features with >80% nulls:")
    for col, pct in high_null.items():
        print(f"  {col}: {pct:.1%}")
    X = X.drop(columns=high_null.index)
    feature_cols = [c for c in feature_cols if c in X.columns]

print(f"\nAfter null filter: {X.shape[1]} features")
print(f"Remaining nulls per row: mean={X.isnull().sum(axis=1).mean():.1f}, max={X.isnull().sum(axis=1).max()}")

# ═══════════════════════════════════════════════════════
# D. Prepare two versions:
#    - X_raw: for tree models (handle NaN natively)
#    - X_imputed + X_scaled: for linear/SVM models
# ═══════════════════════════════════════════════════════
from sklearn.impute import SimpleImputer

X_raw = X.copy()  # LightGBM, XGBoost, CatBoost handle NaN

# Imputed + scaled for RF, Ridge, SVR
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns, index=X.index)

print(f"\nPrepared:")
print(f"  X_raw (for tree models):     {X_raw.shape}, NaN={X_raw.isnull().sum().sum():,}")
print(f"  X_imputed (for RF):           {X_imputed.shape}, NaN={X_imputed.isnull().sum().sum()}")
print(f"  X_scaled (for Ridge/SVR):     {X_scaled.shape}, NaN={X_scaled.isnull().sum().sum()}")

# COMMAND ----------

# DBTITLE 1,Feature selection: correlation + importance pre-filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

# ═══════════════════════════════════════════════════════
# A. Remove near-zero variance features
# ═══════════════════════════════════════════════════════
variances = X_imputed.var()
zero_var = variances[variances < 1e-6]
if len(zero_var):
    print(f"Dropping {len(zero_var)} near-zero variance features: {list(zero_var.index)}")
    drop_zero = list(zero_var.index)
    X_raw = X_raw.drop(columns=drop_zero, errors="ignore")
    X_imputed = X_imputed.drop(columns=drop_zero, errors="ignore")
    X_scaled = X_scaled.drop(columns=drop_zero, errors="ignore")
    feature_cols = [c for c in feature_cols if c in X_raw.columns]

# ═══════════════════════════════════════════════════════
# B. Remove highly correlated feature pairs (|r| > 0.95)
# ═══════════════════════════════════════════════════════
corr_matrix = X_imputed.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# For each pair, drop the one with lower correlation to target
target_corr = X_imputed.corrwith(y).abs()
to_drop = set()
for col in upper.columns:
    correlated = upper.index[upper[col] > 0.95].tolist()
    for other in correlated:
        # Drop the one less correlated with target
        if target_corr.get(col, 0) >= target_corr.get(other, 0):
            to_drop.add(other)
        else:
            to_drop.add(col)

if to_drop:
    print(f"\nDropping {len(to_drop)} features with |r|>0.95 inter-correlation:")
    for c in sorted(to_drop):
        partner = upper.index[upper[c] > 0.95].tolist() if c in upper.columns else []
        partner = [p for p in partner if p not in to_drop or p == c]
        print(f"  {c}")
    X_raw = X_raw.drop(columns=to_drop, errors="ignore")
    X_imputed = X_imputed.drop(columns=to_drop, errors="ignore")
    X_scaled = X_scaled.drop(columns=to_drop, errors="ignore")
    feature_cols = [c for c in feature_cols if c in X_raw.columns]

# ═══════════════════════════════════════════════════════
# C. Quick RF importance for feature ranking
# ═══════════════════════════════════════════════════════
print(f"\nRunning quick RF importance on {X_imputed.shape[1]} features...")
rf_quick = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_quick.fit(X_imputed, y)

importance_df = pd.DataFrame({
    "feature": X_imputed.columns,
    "rf_importance": rf_quick.feature_importances_
}).sort_values("rf_importance", ascending=False)

# Add correlation with target
importance_df["corr_with_target"] = importance_df["feature"].map(
    X_imputed.corrwith(y).abs().to_dict()
)

print(f"\nTop 30 features by RF importance:")
display(importance_df.head(30))

# ═══════════════════════════════════════════════════════
# D. Select top 50 features by importance
# ═══════════════════════════════════════════════════════
N_TOP = 50
selected = importance_df.head(N_TOP)["feature"].tolist()

print(f"\n{'='*70}")
print(f"SELECTED FEATURES: {len(selected)} (top {N_TOP} from {X_imputed.shape[1]})")
print(f"Importance range: {importance_df.iloc[0]['rf_importance']:.5f} "
      f"to {importance_df.iloc[N_TOP-1]['rf_importance']:.5f}")
print(f"{'='*70}")
for i, feat in enumerate(selected):
    imp = importance_df[importance_df['feature']==feat]['rf_importance'].values[0]
    corr = importance_df[importance_df['feature']==feat]['corr_with_target'].values[0]
    print(f"  {i+1:2d}. {feat:45s} imp={imp:.5f}  |r|={corr:.3f}")

# Update feature matrices
X_raw_sel = X_raw[selected]
X_imputed_sel = X_imputed[selected]
X_scaled_sel = X_scaled[selected]

# COMMAND ----------

# DBTITLE 1,K-Fold cross-validation: 5 algorithms (with progress logs)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import time

# ═══════════════════════════════════════════════════════
# Stratified K-Fold on binned target (keeps class balance)
# ═══════════════════════════════════════════════════════
N_FOLDS = 5
y_binned = pd.cut(y, bins=[0, 2, 3, 4, 5], labels=[0, 1, 2, 3])  # for stratification
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Custom scorers
def rmse_scorer(y_true, y_pred):
    return -np.sqrt(mean_squared_error(y_true, y_pred))  # negative for sklearn

def accuracy_1_scorer(y_true, y_pred):
    """Accuracy within \u00b11 of true value (practical accuracy for 1-5 scale)."""
    return np.mean(np.abs(y_true - y_pred) <= 1)

def exact_accuracy_scorer(y_true, y_pred):
    """Exact class accuracy (round prediction to nearest integer)."""
    y_pred_round = np.clip(np.round(y_pred), 1, 5)
    return np.mean(y_pred_round == y_true)

scoring = {
    "neg_rmse": make_scorer(rmse_scorer),
    "neg_mae": "neg_mean_absolute_error",
    "r2": "r2",
    "accuracy_pm1": make_scorer(accuracy_1_scorer),
    "exact_accuracy": make_scorer(exact_accuracy_scorer),
}

# ═══════════════════════════════════════════════════════
# Define models
# ═══════════════════════════════════════════════════════
models = {
    "LightGBM": {
        "model": lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
            verbosity=-1, n_jobs=-1
        ),
        "X": X_raw_sel,  # handles NaN
    },
    "XGBoost": {
        "model": xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
            tree_method="hist", verbosity=0, n_jobs=-1
        ),
        "X": X_raw_sel,  # handles NaN
    },
    "Random Forest": {
        "model": RandomForestRegressor(
            n_estimators=500, max_depth=15, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        "X": X_imputed_sel,  # RF doesn't handle NaN
    },
    "Ridge Regression": {
        "model": Ridge(alpha=1.0),
        "X": X_scaled_sel,  # needs scaling
    },
    "SVR (RBF)": {
        "model": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "X": X_scaled_sel,  # needs scaling
    },
}

# ═══════════════════════════════════════════════════════
# Run K-Fold CV with FOLD-BY-FOLD progress logging
# ═══════════════════════════════════════════════════════
from sklearn.base import clone

results = []
all_fold_logs = []  # detailed per-fold records

print(f"{N_FOLDS}-Fold Stratified Cross-Validation")
print(f"Features: {len(selected)}, Rows: {len(y):,}")
print(f"Models: {', '.join(models.keys())}")
print(f"{'='*90}")

total_models = len(models)
for m_idx, (name, cfg) in enumerate(models.items(), 1):
    t0_model = time.time()
    print(f"\n[{m_idx}/{total_models}] \u25b6 {name}")
    print(f"    {'Fold':>4s}  {'RMSE':>7s}  {'MAE':>7s}  {'R\u00b2':>7s}  {'Acc\u00b11':>7s}  {'Exact':>7s}  {'Time':>6s}")
    print(f"    {'\u2500'*55}")
    
    fold_rmse, fold_mae, fold_r2, fold_acc1, fold_exact = [], [], [], [], []
    X_model = cfg["X"]
    model_template = cfg["model"]
    
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_model, y_binned), 1):
        t0_fold = time.time()
        X_train, X_test = X_model.iloc[train_idx], X_model.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = clone(model_template)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        acc1 = np.mean(np.abs(y_test - y_pred) <= 1)
        exact = np.mean(np.clip(np.round(y_pred), 1, 5) == y_test)
        fold_time = time.time() - t0_fold
        
        fold_rmse.append(rmse); fold_mae.append(mae); fold_r2.append(r2)
        fold_acc1.append(acc1); fold_exact.append(exact)
        
        print(f"    {fold_i:4d}  {rmse:7.4f}  {mae:7.4f}  {r2:7.4f}  {acc1:7.3f}  {exact:7.3f}  {fold_time:5.1f}s")
        
        all_fold_logs.append({
            "Model": name, "Fold": fold_i,
            "RMSE": rmse, "MAE": mae, "R2": r2,
            "Accuracy_pm1": acc1, "Exact_Accuracy": exact,
            "Train_size": len(train_idx), "Test_size": len(test_idx),
            "Time_sec": fold_time
        })
    
    elapsed = time.time() - t0_model
    row = {
        "Model": name,
        "RMSE": np.mean(fold_rmse), "RMSE_std": np.std(fold_rmse),
        "MAE": np.mean(fold_mae), "MAE_std": np.std(fold_mae),
        "R2": np.mean(fold_r2), "R2_std": np.std(fold_r2),
        "Accuracy_pm1": np.mean(fold_acc1), "Acc_pm1_std": np.std(fold_acc1),
        "Exact_Accuracy": np.mean(fold_exact), "Exact_Acc_std": np.std(fold_exact),
        "Time_sec": elapsed,
    }
    results.append(row)
    print(f"    {'\u2500'*55}")
    print(f"    MEAN  {row['RMSE']:7.4f}  {row['MAE']:7.4f}  {row['R2']:7.4f}  "
          f"{row['Accuracy_pm1']:7.3f}  {row['Exact_Accuracy']:7.3f}  {elapsed:5.1f}s")
    print(f"    \u00b1STD  {row['RMSE_std']:7.4f}  {row['MAE_std']:7.4f}  {row['R2_std']:7.4f}  "
          f"{row['Acc_pm1_std']:7.3f}  {row['Exact_Acc_std']:7.3f}")

results_df = pd.DataFrame(results).sort_values("RMSE")
fold_log_df = pd.DataFrame(all_fold_logs)

print(f"\n{'='*90}")
print(f"\u2705 CV complete! All {total_models} models \u00d7 {N_FOLDS} folds = {total_models * N_FOLDS} runs")
print(f"\nFold-level log ({len(fold_log_df)} rows):")
display(fold_log_df)

# COMMAND ----------

# DBTITLE 1,Results comparison table and chart
# ═══════════════════════════════════════════════════════
# A. Results table
# ═══════════════════════════════════════════════════════
print(f"{'='*80}")
print(f"MODEL COMPARISON (5-Fold Stratified CV, {len(selected)} features, {len(y):,} rows)")
print(f"{'='*80}")

# Compute Adjusted R\u00b2: 1 - (1-R\u00b2)(n-1)/(n-p-1)
n_test_avg = len(y) / N_FOLDS
n_feat = len(selected)
results_df["Adj_R2"] = 1 - (1 - results_df["R2"]) * (n_test_avg - 1) / (n_test_avg - n_feat - 1)

display_df = results_df[[
    "Model", "RMSE", "MAE", "R2", "Adj_R2", "Accuracy_pm1", "Exact_Accuracy", "Time_sec"
]].copy()
display_df.columns = ["Model", "RMSE \u2193", "MAE \u2193", "R\u00b2 \u2191", "Adj R\u00b2 \u2191", "Acc \u00b11 \u2191", "Exact Acc \u2191", "Time (s)"]
for c in display_df.columns[1:]:
    display_df[c] = display_df[c].round(4)
display(display_df.reset_index(drop=True))

# Best model
best = results_df.iloc[0]
print(f"\n\u2705 BEST MODEL: {best['Model']}")
print(f"   RMSE = {best['RMSE']:.4f} \u00b1 {best['RMSE_std']:.4f}")
print(f"   MAE  = {best['MAE']:.4f} \u00b1 {best['MAE_std']:.4f}")
print(f"   R\u00b2   = {best['R2']:.4f} \u00b1 {best['R2_std']:.4f}")
print(f"   Adj R\u00b2= {best['Adj_R2']:.4f}")
print(f"   Accuracy \u00b11 = {best['Accuracy_pm1']:.3f} \u00b1 {best['Acc_pm1_std']:.3f}")
print(f"   Exact Accuracy = {best['Exact_Accuracy']:.3f} \u00b1 {best['Exact_Acc_std']:.3f}")

# ═══════════════════════════════════════════════════════
# B. Visual comparison
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors = ["#2ecc71" if m == best["Model"] else "#3498db" for m in results_df["Model"]]

# RMSE (lower is better)
axes[0].barh(results_df["Model"], results_df["RMSE"], color=colors,
             xerr=results_df["RMSE_std"], capsize=3)
axes[0].set_xlabel("RMSE (lower is better)")
axes[0].set_title("RMSE")
axes[0].invert_yaxis()

# R\u00b2 (higher is better)
axes[1].barh(results_df["Model"], results_df["R2"], color=colors,
             xerr=results_df["R2_std"], capsize=3)
axes[1].set_xlabel("R\u00b2 (higher is better)")
axes[1].set_title("R\u00b2 Score")
axes[1].invert_yaxis()

# Accuracy \u00b11 (higher is better)
axes[2].barh(results_df["Model"], results_df["Accuracy_pm1"], color=colors,
             xerr=results_df["Acc_pm1_std"], capsize=3)
axes[2].set_xlabel("Accuracy \u00b11 (higher is better)")
axes[2].set_title("Practical Accuracy (\u00b11 rating)")
axes[2].invert_yaxis()

plt.suptitle(f"Model Comparison \u2014 {N_FOLDS}-Fold CV", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,SHAP analysis for best model (XGBoost)
import shap

# ═══════════════════════════════════════════════════════
# Train best model on full data for SHAP
# ═══════════════════════════════════════════════════════
best_name = results_df.iloc[0]["Model"]
best_cfg = models[best_name]

print(f"Training {best_name} on full dataset for SHAP analysis...")
best_model = best_cfg["model"]
X_shap = best_cfg["X"].copy()
best_model.fit(X_shap, y)

# ═══════════════════════════════════════════════════════
# Compute SHAP values
# ═══════════════════════════════════════════════════════
print("Computing SHAP values (this may take 1-2 minutes)...")

# ── SHAP analysis ──
# Use XGBoost's native SHAP contributions to avoid the SHAP/XGBoost 2.x
# TreeExplainer incompatibility around bracket-formatted base_score.
if len(X_shap) > 2000:
    X_sample = X_shap.sample(2000, random_state=42)
else:
    X_sample = X_shap

booster = best_model.get_booster()
dmatrix_sample = xgb.DMatrix(X_sample, feature_names=X_sample.columns.tolist())
shap_contrib = booster.predict(dmatrix_sample, pred_contribs=True)
shap_values_array = shap_contrib[:, :-1]  # last column is bias term
base_value = shap_contrib[:, -1].mean()
print(f"✅ Native XGBoost SHAP succeeded for {best_name}")
print(f"  mean base value: {base_value:.4f}")

print(f"SHAP values computed for {len(X_sample)} samples, {X_sample.shape[1]} features")

# ═══════════════════════════════════════════════════════
# A. SHAP Summary (bee swarm) — Top 25 features
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 10))
shap.summary_plot(shap_values_array, X_sample, max_display=25, show=False)
plt.title(f"SHAP Feature Importance \u2014 {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════
# B. SHAP bar plot — Top 25 mean |SHAP|
# ═══════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values_array, X_sample, plot_type="bar", max_display=25, show=False)
plt.title(f"Mean |SHAP| \u2014 {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════
# C. SHAP importance table
# ═══════════════════════════════════════════════════════
shap_importance = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": np.abs(shap_values_array).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print(f"\nTop 30 features by mean |SHAP|:")
display(shap_importance.head(30))

# COMMAND ----------

# DBTITLE 1,SHAP interpretation guide
# ═══════════════════════════════════════════════════════
# SHAP Interpretation Guide
# ═══════════════════════════════════════════════════════
total_shap = shap_importance['mean_abs_shap'].sum()
top5_shap = shap_importance.head(5)['mean_abs_shap'].sum()
top10_shap = shap_importance.head(10)['mean_abs_shap'].sum()

print(f"{'='*70}")
print(f"SHAP INTERPRETATION GUIDE \u2014 {best_name}")
print(f"{'='*70}")

print(f"\n\U0001f4ca Bee Swarm Plot (Chart A):")
print(f"  \u2022 Each dot = one sample's SHAP value for that feature.")
print(f"  \u2022 X-axis = SHAP value: how much that feature pushed the prediction")
print(f"    above (+) or below (\u2212) the base value ({base_value:.2f}).")
print(f"  \u2022 Color: RED = high feature value, BLUE = low feature value.")
print(f"  \u2022 Features sorted top-to-bottom by importance.")
print(f"  \u2022 Wide spread = feature has large, variable effects across samples.")

print(f"\n\U0001f4ca Bar Plot (Chart B):")
print(f"  \u2022 Bars = mean |SHAP| \u2014 average absolute contribution per feature.")
print(f"  \u2022 Longer bar = feature influences predictions more on average.")

print(f"\n\U0001f4ca SHAP Concentration:")
print(f"  \u2022 Top 5 features explain {top5_shap/total_shap*100:.1f}% of total model impact")
print(f"  \u2022 Top 10 features explain {top10_shap/total_shap*100:.1f}% of total model impact")
print(f"  \u2022 Remaining {len(selected)-10} features explain {(1-top10_shap/total_shap)*100:.1f}%")

print(f"\n\U0001f4ca Feature-Level Effects (direction of influence):")
for i, (_, row) in enumerate(shap_importance.head(8).iterrows()):
    feat = row['feature']
    col_idx = list(X_sample.columns).index(feat)
    sv = shap_values_array[:, col_idx]
    fv = X_sample[feat].values
    valid = ~np.isnan(fv)
    if valid.sum() < 10:
        continue
    corr = np.corrcoef(fv[valid], sv[valid])[0, 1]
    if corr > 0.05:
        direction = "INCREASES"
    elif corr < -0.05:
        direction = "DECREASES"
    else:
        direction = "has MIXED effect on"
    print(f"  {i+1}. {feat}")
    print(f"     mean |SHAP| = {row['mean_abs_shap']:.4f} | "
          f"Higher values {direction} predicted feeling (r={corr:.3f})")

# COMMAND ----------

# DBTITLE 1,Dependence plot interpretation
# ═══════════════════════════════════════════════════════
# Dependence Plot Interpretation
# ═══════════════════════════════════════════════════════
print(f"{'='*70}")
print(f"DEPENDENCE PLOT INTERPRETATION \u2014 {best_name}")
print(f"{'='*70}")
print(f"\n\U0001f4ca How to Read Dependence Plots:")
print(f"  \u2022 X-axis = actual feature value; Y-axis = SHAP value (marginal effect).")
print(f"  \u2022 Upward trend = higher feature value increases predicted feeling.")
print(f"  \u2022 Downward trend = higher feature value decreases predicted feeling.")
print(f"  \u2022 Color = auto-selected interaction feature (strongest interaction).")
print(f"  \u2022 Vertical spread at a given X = the effect depends on the")
print(f"    interaction variable's value (feature interaction).")
print(f"  \u2022 Flat regions = feature has minimal effect in that value range.")

for feat in top6:
    col_idx = list(X_sample.columns).index(feat)
    sv = shap_values_array[:, col_idx]
    fv = X_sample[feat].values
    valid = ~np.isnan(fv)
    if valid.sum() < 10:
        continue
    corr = np.corrcoef(fv[valid], sv[valid])[0, 1]
    if corr > 0.05:
        direction = "positive (higher value \u2192 higher feeling)"
    elif corr < -0.05:
        direction = "negative (higher value \u2192 lower feeling)"
    else:
        direction = "non-linear / context-dependent"
    print(f"\n  {feat}:")
    print(f"    Relationship: {direction} (r = {corr:.3f})")
    print(f"    SHAP range: [{sv.min():.3f}, {sv.max():.3f}]")
    print(f"    Feature range: [{np.nanmin(fv):.2f}, {np.nanmax(fv):.2f}]")

# COMMAND ----------

# DBTITLE 1,Save models and measure inference time
import pickle, os, time
from sklearn.base import clone

# ═══════════════════════════════════════════════════════
# Save best model per algorithm + measure size & inference time
# ═══════════════════════════════════════════════════════
save_dir = f"{DATA_DIR}/../model_selection"
os.makedirs(save_dir, exist_ok=True)

print(f"Saving trained models to: {save_dir}")
print(f"{'='*80}")

model_meta = []
for name, cfg in models.items():
    model = clone(cfg["model"])
    X_full = cfg["X"]
    model.fit(X_full, y)

    # Save as pickle
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    pkl_path = f"{save_dir}/{safe_name}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)

    size_bytes = os.path.getsize(pkl_path)
    size_mb = size_bytes / (1024 * 1024)

    # Measure inference time (average over 20 runs on 1000 samples)
    X_test = X_full.sample(min(1000, len(X_full)), random_state=42)
    _ = model.predict(X_test)  # warm-up

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model.predict(X_test)
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    per_sample_us = avg_ms / len(X_test) * 1000

    model_meta.append({
        "Model": name,
        "File": os.path.basename(pkl_path),
        "Size_KB": round(size_bytes / 1024, 1),
        "Size_MB": round(size_mb, 3),
        "Inference_ms_1000": round(avg_ms, 2),
        "Per_Sample_\u00b5s": round(per_sample_us, 2),
    })

    print(f"\n  \u2713 {name}")
    print(f"    File: {os.path.basename(pkl_path)}")
    print(f"    Size: {size_mb:.3f} MB ({size_bytes/1024:.1f} KB)")
    print(f"    Inference: {avg_ms:.2f} ms / {len(X_test)} samples "
          f"({per_sample_us:.2f} \u00b5s/sample)")

model_meta_df = pd.DataFrame(model_meta)
print(f"\n{'='*80}")
print(f"Model Size & Inference Time Comparison:")
display(model_meta_df)
print(f"\n\u2713 All models saved to {save_dir}/")

# COMMAND ----------

# DBTITLE 1,Classification accuracy with rounded predictions
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import clone

# ═══════════════════════════════════════════════════════
# Classification evaluation: round best model predictions to 1-5
# ═══════════════════════════════════════════════════════
best_name = results_df.iloc[0]["Model"]
best_cfg = models[best_name]
X_model = best_cfg["X"]
model_template = best_cfg["model"]

print(f"Classification Evaluation \u2014 {best_name} (predictions rounded to 1-5)")
print(f"{'='*70}")

# Collect out-of-fold CV predictions (fair evaluation)
all_y_true = []
all_y_pred_raw = []
all_y_pred_class = []

for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_model, y_binned), 1):
    model = clone(model_template)
    model.fit(X_model.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X_model.iloc[test_idx])
    preds_rounded = np.clip(np.round(preds), 1, 5).astype(int)
    all_y_true.extend(y.iloc[test_idx].astype(int).tolist())
    all_y_pred_raw.extend(preds.tolist())
    all_y_pred_class.extend(preds_rounded.tolist())

all_y_true = np.array(all_y_true)
all_y_pred_class = np.array(all_y_pred_class)
all_y_pred_raw = np.array(all_y_pred_raw)

# \u2500\u2500 Overall accuracy \u2500\u2500
overall_acc = accuracy_score(all_y_true, all_y_pred_class)
print(f"\nOverall Classification Accuracy: {overall_acc:.4f} ({overall_acc:.1%})")
print(f"  ({len(all_y_true):,} samples, {N_FOLDS}-fold CV out-of-fold predictions)")

# \u2500\u2500 Per-class report \u2500\u2500
labels = sorted(np.unique(all_y_true))
print(f"\nClassification Report:")
print(classification_report(all_y_true, all_y_pred_class, labels=labels,
                            target_names=[f"Feeling {l}" for l in labels]))

# \u2500\u2500 Confusion matrix \u2500\u2500
cm = confusion_matrix(all_y_true, all_y_pred_class, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels],
                      columns=[f"Pred {l}" for l in labels])
print(f"Confusion Matrix:")
display(cm_df)

# \u2500\u2500 Confusion matrix heatmap \u2500\u2500
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f"{l}" for l in labels],
            yticklabels=[f"{l}" for l in labels], ax=ax)
ax.set_xlabel("Predicted Feeling")
ax.set_ylabel("True Feeling")
ax.set_title(f"Confusion Matrix \u2014 {best_name} (Rounded to 1-5)")
plt.tight_layout()
plt.show()

# \u2500\u2500 Error distribution \u2500\u2500
errors = all_y_pred_class - all_y_true
print(f"\nPrediction Error Distribution:")
for err_val in sorted(np.unique(errors)):
    count = (errors == err_val).sum()
    pct = count / len(errors) * 100
    print(f"  Error {err_val:+d}: {count:5d} ({pct:5.1f}%)")

# \u2500\u2500 Summary \u2500\u2500
print(f"\n{'='*70}")
print(f"CLASSIFICATION SUMMARY \u2014 {best_name}")
print(f"{'='*70}")
print(f"  \u2022 Exact match accuracy: {overall_acc:.1%}")
print(f"  \u2022 Within \u00b11 accuracy: {np.mean(np.abs(errors) <= 1):.1%}")
print(f"  \u2022 Mean absolute error: {np.mean(np.abs(all_y_pred_raw - all_y_true)):.4f}")
print(f"  \u2022 Class distribution: {dict(zip(*np.unique(all_y_true, return_counts=True)))}")
print(f"  \u2022 Prediction dist:     {dict(zip(*np.unique(all_y_pred_class, return_counts=True)))}")

# COMMAND ----------

# DBTITLE 1,SHAP dependence plots for top features
# ═══════════════════════════════════════════════════════
# SHAP dependence plots for top 6 features
# Shows how each feature value affects the prediction
# ═══════════════════════════════════════════════════════
top6 = shap_importance.head(6)["feature"].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, feat in enumerate(top6):
    ax = axes[i // 3, i % 3]
    shap.dependence_plot(
        feat, shap_values_array, X_sample,
        interaction_index="auto",
        ax=ax, show=False
    )
    ax.set_title(feat, fontsize=11)

plt.suptitle(f"SHAP Dependence Plots \u2014 {best_name} (Top 6 Features)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Conclusion and model recommendation
# ═══════════════════════════════════════════════════════
# FINAL CONCLUSION
# ═══════════════════════════════════════════════════════
print(f"{'='*80}")
print(f"MODEL SELECTION CONCLUSION")
print(f"{'='*80}")

# Rank models
for i, (_, row) in enumerate(results_df.iterrows()):
    marker = "\u2b50" if i == 0 else f" {i+1}."
    print(f"  {marker} {row['Model']:20s} | RMSE={row['RMSE']:.4f}, "
          f"R\u00b2={row['R2']:.4f}, Acc\u00b11={row['Accuracy_pm1']:.3f}, "
          f"Exact={row['Exact_Accuracy']:.3f}")

best = results_df.iloc[0]
print(f"\n{'\u2500'*60}")
print(f"RECOMMENDED MODEL: {best['Model']}")
print(f"{'\u2500'*60}")
print(f"  \u2022 Best RMSE: {best['RMSE']:.4f} (predicting 1-5 scale)")
print(f"  \u2022 R\u00b2: {best['R2']:.4f} | Adj R\u00b2: {best['Adj_R2']:.4f} (variance explained)")
print(f"  \u2022 Practical accuracy (\u00b11): {best['Accuracy_pm1']:.1%}")
print(f"  \u2022 Exact accuracy: {best['Exact_Accuracy']:.1%}")

print(f"\nKey SHAP insights (top 5 drivers):")
for i, (_, row) in enumerate(shap_importance.head(5).iterrows()):
    print(f"  {i+1}. {row['feature']:40s} mean|SHAP|={row['mean_abs_shap']:.4f}")

print(f"\nFeatures used: {len(selected)}")
print(f"Training samples: {len(y):,}")
print(f"\nNext steps:")
print(f"  1. Hyperparameter tuning (Optuna/Bayesian) on the selected model")
print(f"  2. Time-based validation (train on first 60 days, test on last 30)")
print(f"  3. Per-user evaluation (are some users harder to predict?)")
print(f"  4. Ensemble: stack top 2-3 models for marginal improvement")

# Save results
results_df.to_csv(f"{DATA_DIR}/model_comparison_results.csv", index=False)
shap_importance.to_csv(f"{DATA_DIR}/shap_feature_importance.csv", index=False)
print(f"\n\u2713 Results saved to {DATA_DIR}/model_comparison_results.csv")
print(f"\u2713 SHAP importance saved to {DATA_DIR}/shap_feature_importance.csv")