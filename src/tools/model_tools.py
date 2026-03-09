"""
Model inference tools for the Fraud Copilot agent.

Each tool wraps an MLflow-registered sklearn Pipeline model and provides:
- Inference with raw data (Pipeline handles encoding internally)
- SHAP explanations via TreeExplainer on the underlying model
- Risk classification and actionable recommendations

Design note: Models are loaded twice — as pyfunc (for inference with raw data)
and as sklearn (for SHAP access to pipeline internals). This is required because
shap.TreeExplainer cannot operate on a full Pipeline.
"""

import re
import mlflow
import numpy as np
import pandas as pd
import shap
from typing import Optional


class FraudModelTool:
    """Fraud detection model with SHAP explanations."""

    def __init__(
        self,
        model_name: str,
        feature_names: list[str],
        risk_thresholds: dict | None = None,
    ):
        self.model_name = model_name
        self.feature_names = feature_names

        # Risk tier thresholds
        self.thresholds = risk_thresholds or {
            "high": 0.7,
            "medium": 0.4,
        }

        # Load as pyfunc (accepts raw data via Pipeline)
        self.pyfunc_model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")

        # Load as sklearn (for SHAP internals)
        pipeline = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.classifier = pipeline.named_steps["classifier"]
        self.explainer = shap.TreeExplainer(self.classifier)

    @mlflow.trace(span_type="TOOL", name="run_fraud_model")
    def predict(self, features: dict) -> dict:
        """Predict fraud probability with SHAP explanations."""
        input_df = pd.DataFrame([features])[self.feature_names]

        # Cast int64 to int32 to match model signature
        for col in input_df.select_dtypes(include=["int64"]).columns:
            input_df[col] = input_df[col].astype("int32")

        # Inference via Pipeline (handles encoding)
        prob = float(self.pyfunc_model.predict(input_df)[0])
        risk_score = int(prob * 100)

        # Risk tier classification
        if prob >= self.thresholds["high"]:
            risk_tier = "HIGH"
            action = "Block transaction and verify identity immediately"
        elif prob >= self.thresholds["medium"]:
            risk_tier = "MEDIUM"
            action = "Flag for manual review within 24 hours"
        else:
            risk_tier = "LOW"
            action = "No immediate action, monitor patterns"

        # SHAP explanation
        impacts = self._compute_shap(input_df)

        return {
            "probability": round(prob, 4),
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "suggested_action": action,
            "top_features": impacts,
        }

    def _compute_shap(self, input_df: pd.DataFrame, top_k: int = 5) -> list[dict]:
        """Compute SHAP values for the given input."""
        try:
            X_enc = self.preprocessor.transform(input_df)
            sv = self.explainer.shap_values(X_enc)
            if isinstance(sv, list):
                sv = sv[1]  # Binary classification: take positive class
            vals = sv[0] if len(sv.shape) > 1 else sv

            impacts = sorted(
                [
                    {
                        "feature": self.feature_names[i],
                        "shap_value": round(float(vals[i]), 4),
                    }
                    for i in range(min(len(vals), len(self.feature_names)))
                ],
                key=lambda x: abs(x["shap_value"]),
                reverse=True,
            )[:top_k]

            for item in impacts:
                direction = "increases" if item["shap_value"] > 0 else "decreases"
                item["explanation"] = (
                    f"{item['feature']} {direction} fraud risk "
                    f"(SHAP: {item['shap_value']:.4f})"
                )
            return impacts

        except Exception as e:
            return [{"feature": "SHAP error", "explanation": str(e)}]


class PurchaseModelTool:
    """Purchase prediction model with SHAP explanations."""

    def __init__(self, model_name: str, feature_names: list[str]):
        self.model_name = model_name
        self.feature_names = feature_names

        self.pyfunc_model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")

        pipeline = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.regressor = pipeline.named_steps["regressor"]
        self.explainer = shap.TreeExplainer(self.regressor)

    @mlflow.trace(span_type="TOOL", name="run_purchase_model")
    def predict(self, features: dict) -> dict:
        """Predict expected purchase amount."""
        input_df = pd.DataFrame([features])[self.feature_names]

        for col in input_df.select_dtypes(include=["int64"]).columns:
            input_df[col] = input_df[col].astype("int32")
        for col in input_df.select_dtypes(include=["float64"]).columns:
            input_df[col] = input_df[col].astype("float64")

        pred_raw = float(self.pyfunc_model.predict(input_df)[0])

        # Inverse log transform if needed (raw prediction < 10 suggests log space)
        pred = float(np.expm1(pred_raw)) if pred_raw < 10 else pred_raw

        # Confidence interval (±20% margin)
        margin = pred * 0.20

        # SHAP
        impacts = self._compute_shap(input_df)

        return {
            "predicted_amount": round(pred, 2),
            "confidence_interval": {
                "low": round(max(0, pred - margin), 2),
                "high": round(pred + margin, 2),
            },
            "top_features": impacts,
        }

    def _compute_shap(self, input_df: pd.DataFrame, top_k: int = 5) -> list[dict]:
        """Compute SHAP values for the given input."""
        try:
            X_enc = self.preprocessor.transform(input_df)
            sv = self.explainer.shap_values(X_enc)
            vals = sv[0] if len(sv.shape) > 1 else sv

            impacts = sorted(
                [
                    {
                        "feature": self.feature_names[i],
                        "shap_value": round(float(vals[i]), 4),
                    }
                    for i in range(min(len(vals), len(self.feature_names)))
                ],
                key=lambda x: abs(x["shap_value"]),
                reverse=True,
            )[:top_k]
            return impacts

        except Exception:
            return []


# --- Feature extraction helpers ---

def extract_fraud_features(
    query: str,
    feature_names: list[str],
    fraud_df: pd.DataFrame,
) -> dict:
    """
    Extract transaction features from natural language for fraud prediction.

    Attempts regex extraction for common fields, falls back to defaults
    for any missing features. If a customer ID is found, looks up their
    actual data from the dataset.
    """
    q = query.lower()
    features = {}

    # Amount extraction
    m = re.search(r"\$?([\d,]+(?:\.\d{2})?)", query)
    if m:
        features["transaction_amount"] = float(m.group(1).replace(",", ""))

    # International flag
    if "international" in q:
        features["is_international"] = 1
    if "domestic" in q:
        features["is_international"] = 0

    # Merchant category
    category_map = {
        "electronics": "electronics", "jewelry": "jewelry",
        "grocery": "grocery", "restaurant": "restaurant",
        "gaming": "gaming", "luxury": "luxury_goods",
        "travel": "travel", "atm": "ATM",
    }
    for keyword, category in category_map.items():
        if keyword in q:
            features["merchant_category"] = category
            break

    # Time of day
    tm = re.search(r"(\d{1,2})\s*(am|pm)", q, re.IGNORECASE)
    if tm:
        h = int(tm.group(1))
        if tm.group(2).lower() == "pm" and h != 12:
            h += 12
        elif tm.group(2).lower() == "am" and h == 12:
            h = 0
        features["hour_of_day"] = h

    # Account age
    am = re.search(r"(\d+)[- ]month[- ]old", q)
    if am:
        features["account_age_days"] = int(am.group(1)) * 30

    # Customer ID lookup (overrides all extracted features)
    cm = re.search(r"CUST\d+", query, re.IGNORECASE)
    if cm:
        cid = cm.group().upper()
        row = fraud_df[fraud_df["customer_id"] == cid]
        if len(row) > 0:
            row = row.iloc[0]
            return {f: row[f] for f in feature_names if f in row.index}

    # Default values for missing features
    defaults = dict(zip(feature_names, [
        500.0, 35, 365, 2, 300.0, 2, 10, 0, 0.7, 0.3,
        5000.0, 10000, 0.3, 0, 0.3, 10.0, 5.0, 14,
        0, 1, 1, 1, 1, 0,
        "online", "retail", "mobile", "USA",
    ]))
    for k, v in defaults.items():
        if k not in features:
            features[k] = v

    return features


def extract_purchase_features(
    query: str,
    feature_names: list[str],
    purchase_df: pd.DataFrame,
) -> dict:
    """
    Extract customer features from natural language for purchase prediction.

    Uses regex for common fields, fills remaining from a representative row.
    """
    q = query.lower()
    features = {}

    # Age
    am = re.search(r"(\d{2})[- ]year[- ]old", q)
    if am:
        features["age"] = int(am.group(1))

    # Membership tier
    for tier in ["platinum", "gold", "silver", "bronze"]:
        if tier in q:
            features["membership_tier"] = tier
            break

    # Transactions last month
    tm = re.search(r"(\d+)\s*transactions?\s*(?:last|per)\s*month", q)
    if tm:
        features["num_transactions_last_month"] = int(tm.group(1))

    # Fill missing from a representative row
    row = purchase_df.iloc[0]
    defaults = {f: row[f] for f in feature_names if f in row.index}
    for f in feature_names:
        if f not in defaults:
            defaults[f] = 0

    defaults.update(features)
    return defaults


# Module-level convenience functions

_fraud_model: Optional[FraudModelTool] = None
_purchase_model: Optional[PurchaseModelTool] = None


def init_model_tools(
    fraud_model_name: str,
    fraud_features: list[str],
    purchase_model_name: str,
    purchase_features: list[str],
) -> None:
    """Initialize model tools with registered model names and feature lists."""
    global _fraud_model, _purchase_model
    _fraud_model = FraudModelTool(fraud_model_name, fraud_features)
    _purchase_model = PurchaseModelTool(purchase_model_name, purchase_features)


def run_fraud_model(features: dict) -> dict:
    if _fraud_model is None:
        raise RuntimeError("Call init_model_tools() first")
    return _fraud_model.predict(features)


def run_purchase_model(features: dict) -> dict:
    if _purchase_model is None:
        raise RuntimeError("Call init_model_tools() first")
    return _purchase_model.predict(features)
