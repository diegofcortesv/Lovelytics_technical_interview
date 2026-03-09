# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Purchase Prediction Model Training
# MAGIC **Fraud Copilot | Regression (purchase_amount)**
# MAGIC
# MAGIC Pipeline approach: ColumnTransformer(OrdinalEncoder) → LightGBM.
# MAGIC Model accepts raw data directly. No external encoding.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

CATALOG = "fraud_agent"
SCHEMA = "default"
TABLE = f"{CATALOG}.{SCHEMA}.product_purchase_dataset"
EXPERIMENT_NAME = f"/fraud-copilot/purchase-model-training"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.purchase_prediction_model"

mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Load & Explore Data

# COMMAND ----------

df = spark.table(TABLE).toPandas()
print(f"Shape: {df.shape}")
print(f"\nTarget (purchase_amount):")
print(df["purchase_amount"].describe())
print(f"\nMean purchase by tier:")
print(df.groupby("membership_tier")["purchase_amount"].mean().sort_values(ascending=False))

# COMMAND ----------

from scipy import stats
skewness = stats.skew(df["purchase_amount"])
print(f"Target skewness: {skewness:.3f}")
USE_LOG_TRANSFORM = abs(skewness) > 2
print(f"Apply log transform: {'Yes' if USE_LOG_TRANSFORM else 'No (skew < 2)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Feature Engineering with Pipeline

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor

NUMERIC_FEATURES = [
    "age", "customer_tenure_days", "num_transactions_last_month",
    "num_transactions_last_year", "avg_transaction_value", "total_spent_last_year",
    "loyalty_points", "customer_satisfaction_score", "email_engagement_rate",
    "cart_abandonment_rate", "distance_to_nearest_store_km",
    "num_customer_service_contacts", "product_return_rate", "last_purchase_days_ago"
]

BINARY_FEATURES = [
    "mobile_app_user", "social_media_follower", "has_credit_card",
    "has_children", "owns_home"
]

CATEGORICAL_FEATURES = [
    "gender", "income_bracket", "membership_tier",
    "preferred_category", "preferred_payment_method",
    "location_type", "occupation_category", "education_level", "marital_status"
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET = "purchase_amount"
GROUP_COL = "membership_tier"

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
    ],
    remainder="drop"
)

lgbm_params = {
    "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
    "num_leaves": 15, "min_child_samples": 10,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "random_state": 42, "verbose": -1,
}

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor(**lgbm_params))
])

X = df[ALL_FEATURES]
y = df[TARGET].values
groups = df[GROUP_COL].values

if USE_LOG_TRANSFORM:
    y_train_target = np.log1p(y)
else:
    y_train_target = y.copy()

print(f"Features: {len(ALL_FEATURES)} ({len(NUMERIC_FEATURES)} num + {len(BINARY_FEATURES)} bin + {len(CATEGORICAL_FEATURES)} cat)")
print(f"Categoricals as RAW strings: {CATEGORICAL_FEATURES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Train with GroupKFold

# COMMAND ----------

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
import json

n_tiers = len(df[GROUP_COL].unique())
gkf = GroupKFold(n_splits=min(n_tiers, 4))
metrics_list = {"rmse": [], "mae": [], "r2": [], "mape": []}
tier_metrics = {}

with mlflow.start_run(run_name="lightgbm-pipeline-purchase-v2") as run:
    mlflow.log_params(lgbm_params)
    mlflow.log_param("n_folds", min(n_tiers, 4))
    mlflow.log_param("n_features", len(ALL_FEATURES))
    mlflow.log_param("n_samples", len(X))
    mlflow.log_param("log_transform", USE_LOG_TRANSFORM)
    mlflow.log_param("pipeline", "ColumnTransformer(OrdinalEncoder) + LGBMRegressor")
    mlflow.log_param("features", json.dumps(ALL_FEATURES))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_train_target, groups)):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y_train_target[train_idx], y_train_target[val_idx]
        y_vl_original = y[val_idx]
        groups_val = groups[val_idx]
        
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_tr, y_tr)
        
        y_pred = fold_pipe.predict(X_vl)
        if USE_LOG_TRANSFORM:
            y_pred_orig = np.expm1(y_pred)
        else:
            y_pred_orig = y_pred
        
        rmse = np.sqrt(mean_squared_error(y_vl_original, y_pred_orig))
        mae = mean_absolute_error(y_vl_original, y_pred_orig)
        r2 = r2_score(y_vl_original, y_pred_orig)
        mape = np.mean(np.abs((y_vl_original - y_pred_orig) / y_vl_original)) * 100
        
        metrics_list["rmse"].append(rmse)
        metrics_list["mae"].append(mae)
        metrics_list["r2"].append(r2)
        metrics_list["mape"].append(mape)
        
        # Per-tier
        vdf = pd.DataFrame({"actual": y_vl_original, "predicted": y_pred_orig, "tier": groups_val})
        for tier in vdf["tier"].unique():
            t = vdf[vdf["tier"]==tier]
            if tier not in tier_metrics: tier_metrics[tier] = {"rmse":[], "mae":[], "count":[]}
            tier_metrics[tier]["rmse"].append(np.sqrt(mean_squared_error(t["actual"], t["predicted"])))
            tier_metrics[tier]["mae"].append(mean_absolute_error(t["actual"], t["predicted"]))
            tier_metrics[tier]["count"].append(len(t))
        
        print(f"Fold {fold+1}: RMSE=${rmse:.2f}  MAE=${mae:.2f}  R2={r2:.4f}  MAPE={mape:.1f}%")
    
    for name, values in metrics_list.items():
        mlflow.log_metric(f"{name}_mean", np.mean(values))
        mlflow.log_metric(f"{name}_std", np.std(values))
    
    print(f"\n{'='*60}")
    print("SEGMENTED BY TIER:")
    for tier, tm in sorted(tier_metrics.items()):
        print(f"  {tier:10s}: RMSE=${np.mean(tm['rmse']):.2f}  MAE=${np.mean(tm['mae']):.2f}  n={sum(tm['count'])}")
    
    print(f"\n{'='*60}")
    print(f"OVERALL ({min(n_tiers,4)}-Fold GroupKFold):")
    print(f"  RMSE:  ${np.mean(metrics_list['rmse']):.2f} +/- ${np.std(metrics_list['rmse']):.2f}")
    print(f"  MAE:   ${np.mean(metrics_list['mae']):.2f} +/- ${np.std(metrics_list['mae']):.2f}")
    print(f"  R2:    {np.mean(metrics_list['r2']):.4f} +/- {np.std(metrics_list['r2']):.4f}")
    print(f"  MAPE:  {np.mean(metrics_list['mape']):.1f}% +/- {np.std(metrics_list['mape']):.1f}%")
    print(f"{'='*60}")
    
    RUN_ID = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 Final Pipeline + SHAP

# COMMAND ----------

import shap
import matplotlib.pyplot as plt

final_pipeline = clone(pipeline)
final_pipeline.fit(X, y_train_target)

X_transformed = final_pipeline.named_steps["preprocessor"].transform(X)
lgbm_model = final_pipeline.named_steps["regressor"]
transformed_names = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_transformed)

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_transformed, feature_names=transformed_names, show=False)
plt.title("SHAP Feature Importance - Purchase Prediction Model")
plt.tight_layout()
with mlflow.start_run(run_id=RUN_ID):
    fig.savefig("/tmp/shap_purchase_summary.png", dpi=150, bbox_inches="tight")
    mlflow.log_artifact("/tmp/shap_purchase_summary.png", "plots")
display(fig)

# COMMAND ----------

feature_importance = pd.DataFrame({
    "feature": transformed_names,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("Top 10 Features:")
for _, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:35s} {row['mean_abs_shap']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 Register Pipeline in Unity Catalog

# COMMAND ----------

from mlflow.models.signature import infer_signature

X_sample = X.iloc[:5]
y_sample = final_pipeline.predict(X_sample)
signature = infer_signature(X_sample, y_sample)

with mlflow.start_run(run_id=RUN_ID):
    model_info = mlflow.sklearn.log_model(
        sk_model=final_pipeline,
        artifact_path="purchase_model",
        signature=signature,
        input_example=X_sample.iloc[:2],
        registered_model_name=MODEL_NAME,
    )
    mlflow.set_tag("model_type", "purchase_prediction")
    mlflow.set_tag("algorithm", "lightgbm_pipeline")
    mlflow.set_tag("preprocessing", "ColumnTransformer(OrdinalEncoder)")

print(f"Model registered: {MODEL_NAME}")

from mlflow import MlflowClient
client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = max(versions, key=lambda v: int(v.version))
client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)
print(f"Alias 'champion' set on version {latest.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.6 Validation with RAW Data

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")

test = X.iloc[:5]
preds = loaded.predict(test)
if USE_LOG_TRANSFORM:
    preds = np.expm1(preds)

print("Predictions with RAW string data:")
for i, (pred, actual) in enumerate(zip(preds, df["purchase_amount"].iloc[:5])):
    print(f"  Customer {i+1}: predicted=${pred:.2f}, actual=${actual:.2f}")
print(f"\nPipeline handled encoding internally.")

# COMMAND ----------

PURCHASE_MODEL_FEATURES = ALL_FEATURES
print(f"Features for agent: {len(PURCHASE_MODEL_FEATURES)}")