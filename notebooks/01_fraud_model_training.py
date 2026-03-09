# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Fraud Detection Model Training
# MAGIC **Fraud Copilot | Binary Classification (fraud: 0/1)**
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads fraud data from Delta table
# MAGIC 2. Performs EDA and feature engineering
# MAGIC 3. Trains XGBoost **inside a sklearn Pipeline** (encoding + model together)
# MAGIC 4. Generates SHAP explanations
# MAGIC 5. Registers the model in Unity Catalog via MLflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 Configuration

# COMMAND ----------

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

CATALOG = "fraud_agent"
SCHEMA = "default"
TABLE = f"{CATALOG}.{SCHEMA}.fraud_dataset"
EXPERIMENT_NAME = f"/fraud-copilot/fraud-model-training"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_detection_model"

mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.set_registry_uri("databricks-uc")

print(f"Table: {TABLE}")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Model Registry: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Load & Explore Data

# COMMAND ----------

df = spark.table(TABLE).toPandas()
print(f"Shape: {df.shape}")
print(f"\nTarget distribution:")
print(df["fraud"].value_counts())
print(f"\nFraud rate: {df['fraud'].mean()*100:.1f}%")

# COMMAND ----------

print("=== DATASET OVERVIEW ===")
print(f"Transactions: {len(df)}")
print(f"Unique customers: {df['customer_id'].nunique()}")
print(f"Countries: {df['country'].nunique()}")
print(f"International: {df['is_international'].sum()} ({df['is_international'].mean()*100:.1f}%)")
print(f"\nAmount range: ${df['transaction_amount'].min():.2f} - ${df['transaction_amount'].max():.2f}")
print(f"Amount mean: ${df['transaction_amount'].mean():.2f}")

print("\n=== FRAUD vs LEGITIMATE ===")
for label, name in [(1, "Fraudulent"), (0, "Legitimate")]:
    subset = df[df["fraud"] == label]
    print(f"\n{name} (n={len(subset)}):")
    print(f"  Avg amount: ${subset['transaction_amount'].mean():.2f}")
    print(f"  Avg risk score: {subset['customer_risk_score'].mean():.3f}")
    print(f"  Avg IP reputation: {subset['ip_reputation_score'].mean():.3f}")
    print(f"  International: {subset['is_international'].mean()*100:.1f}%")
    print(f"  Avg failed txns 24h: {subset['failed_transactions_24h'].mean():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 Feature Engineering with sklearn Pipeline
# MAGIC
# MAGIC **Key design decision:** All preprocessing is packaged inside a `sklearn.Pipeline`
# MAGIC so the registered model accepts raw data directly. This prevents train-serve skew.

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

NUMERIC_FEATURES = [
    "transaction_amount", "customer_age", "account_age_days",
    "transaction_velocity_24h", "avg_transaction_amount_30d",
    "num_transactions_24h", "num_transactions_7d", "failed_transactions_24h",
    "ip_reputation_score", "customer_risk_score", "account_balance",
    "credit_limit", "debt_to_income_ratio", "previous_fraud_reports",
    "merchant_risk_score", "distance_from_home_km",
    "distance_from_last_transaction_km", "hour_of_day"
]

BINARY_FEATURES = [
    "is_international", "shipping_address_match", "billing_address_match",
    "cvv_match", "card_present", "is_recurring"
]

CATEGORICAL_FEATURES = ["transaction_type", "merchant_category", "device_type", "country"]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET = "fraud"

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
    ],
    remainder="drop"
)

xgb_params = {
    "max_depth": 3, "n_estimators": 150, "learning_rate": 0.1,
    "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8,
    "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "scale_pos_weight": 1, "eval_metric": "auc",
    "random_state": 42, "use_label_encoder": False,
}

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(**xgb_params))
])

X = df[ALL_FEATURES]
y = df[TARGET].values

print(f"Features: {len(ALL_FEATURES)} ({len(NUMERIC_FEATURES)} num + {len(BINARY_FEATURES)} bin + {len(CATEGORICAL_FEATURES)} cat)")
print(f"Categoricals sent as RAW strings: {CATEGORICAL_FEATURES}")
print(f"Pipeline: ColumnTransformer(OrdinalEncoder) → XGBClassifier")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4 Model Training with Stratified K-Fold CV

# COMMAND ----------

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc
from sklearn.base import clone
import json

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = {"auc": [], "f1": [], "precision": [], "recall": [], "pr_auc": []}

with mlflow.start_run(run_name="xgboost-pipeline-fraud-v2") as run:
    mlflow.log_params(xgb_params)
    mlflow.log_param("n_folds", 5)
    mlflow.log_param("n_features", len(ALL_FEATURES))
    mlflow.log_param("n_samples", len(X))
    mlflow.log_param("fraud_rate", float(y.mean()))
    mlflow.log_param("pipeline", "ColumnTransformer(OrdinalEncoder) + XGBClassifier")
    mlflow.log_param("features", json.dumps(ALL_FEATURES))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_train, y_train)
        
        y_proba = fold_pipe.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        fold_auc = roc_auc_score(y_val, y_proba)
        fold_f1 = f1_score(y_val, y_pred)
        fold_prec = precision_score(y_val, y_pred)
        fold_rec = recall_score(y_val, y_pred)
        prec_c, rec_c, _ = precision_recall_curve(y_val, y_proba)
        fold_pr_auc = auc(rec_c, prec_c)
        
        metrics["auc"].append(fold_auc)
        metrics["f1"].append(fold_f1)
        metrics["precision"].append(fold_prec)
        metrics["recall"].append(fold_rec)
        metrics["pr_auc"].append(fold_pr_auc)
        
        print(f"Fold {fold+1}: AUC={fold_auc:.4f}  F1={fold_f1:.4f}  Prec={fold_prec:.4f}  Rec={fold_rec:.4f}")
    
    for name, values in metrics.items():
        mlflow.log_metric(f"{name}_mean", np.mean(values))
        mlflow.log_metric(f"{name}_std", np.std(values))
    
    print(f"\n{'='*50}")
    print(f"RESULTS (5-Fold CV):")
    print(f"  ROC-AUC: {np.mean(metrics['auc']):.4f} +/- {np.std(metrics['auc']):.4f}")
    print(f"  F1:      {np.mean(metrics['f1']):.4f} +/- {np.std(metrics['f1']):.4f}")
    print(f"  PR-AUC:  {np.mean(metrics['pr_auc']):.4f} +/- {np.std(metrics['pr_auc']):.4f}")
    print(f"{'='*50}")
    
    RUN_ID = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.5 Train Final Pipeline on Full Data & Generate SHAP

# COMMAND ----------

import shap
import matplotlib.pyplot as plt

final_pipeline = clone(pipeline)
final_pipeline.fit(X, y)

# SHAP needs the transformed data and the underlying XGBClassifier
X_transformed = final_pipeline.named_steps["preprocessor"].transform(X)
xgb_model = final_pipeline.named_steps["classifier"]
transformed_names = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_transformed)

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values, X_transformed, feature_names=transformed_names, show=False)
plt.title("SHAP Feature Importance - Fraud Detection Model")
plt.tight_layout()

with mlflow.start_run(run_id=RUN_ID):
    fig.savefig("/tmp/shap_summary.png", dpi=150, bbox_inches="tight")
    mlflow.log_artifact("/tmp/shap_summary.png", "plots")
display(fig)

# COMMAND ----------

feature_importance = pd.DataFrame({
    "feature": transformed_names,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("Top 10 Features by SHAP Importance:")
for _, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:35s} {row['mean_abs_shap']:.4f}")

# COMMAND ----------

sample_idx = df[df["fraud"] == 1].index[0]
fig, ax = plt.subplots(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_transformed[sample_idx],
        feature_names=transformed_names
    ),
    show=False
)
plt.title("SHAP Waterfall - Example Fraud Transaction")
plt.tight_layout()
with mlflow.start_run(run_id=RUN_ID):
    fig.savefig("/tmp/shap_waterfall.png", dpi=150, bbox_inches="tight")
    mlflow.log_artifact("/tmp/shap_waterfall.png", "plots")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.6 Register Pipeline Model in Unity Catalog

# COMMAND ----------

from mlflow.models.signature import infer_signature

X_sample = X.iloc[:5]
y_sample = final_pipeline.predict_proba(X_sample)[:, 1]
signature = infer_signature(X_sample, y_sample)

with mlflow.start_run(run_id=RUN_ID):
    model_info = mlflow.sklearn.log_model(
        sk_model=final_pipeline,
        artifact_path="fraud_model",
        signature=signature,
        input_example=X_sample.iloc[:2],
        registered_model_name=MODEL_NAME,
    )
    mlflow.set_tag("model_type", "fraud_detection")
    mlflow.set_tag("algorithm", "xgboost_pipeline")
    mlflow.set_tag("preprocessing", "ColumnTransformer(OrdinalEncoder)")

print(f"Model registered: {MODEL_NAME}")
print(f"  Signature accepts RAW data (strings for categoricals)")

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(versions, key=lambda v: int(v.version))
client.set_registered_model_alias(MODEL_NAME, "champion", latest_version.version)
print(f"Alias 'champion' set on version {latest_version.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.7 Validation: Load and Test with RAW Data

# COMMAND ----------

loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")

# Test with actual rows from the DataFrame (guaranteed correct dtypes)
test_data = df[ALL_FEATURES].iloc[:3]
predictions = loaded.predict(test_data)

print("Predictions from registered Pipeline model:")
for i, (pred, actual) in enumerate(zip(predictions, df["fraud"].iloc[:3])):
    print(f"  Txn {i+1}: P(fraud)={pred:.4f}, actual={actual}")

print(f"\nInput dtypes verified against model signature - no type mismatch.")

# COMMAND ----------

for i in range(3):
    row = df.iloc[i:i+1][ALL_FEATURES]
    pred = loaded.predict(row)
    actual = df.iloc[i]["fraud"]
    print(f"  Txn {i+1}: P(fraud)={pred[0]:.4f}, actual={actual}")

# COMMAND ----------

FRAUD_MODEL_FEATURES = ALL_FEATURES
print(f"Features for agent: {len(FRAUD_MODEL_FEATURES)}")