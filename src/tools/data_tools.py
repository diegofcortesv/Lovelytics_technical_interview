"""
Data query tools for the Fraud Copilot agent.

These tools provide pattern-matched data retrieval against in-memory
pandas DataFrames loaded from Delta tables. In production, these would
generate SQL queries against a Databricks SQL Warehouse.

Design note: Pattern matching was chosen over text-to-SQL for the prototype
because it provides deterministic, testable behavior for the demo queries.
A production version should use LLM-generated SQL with validation guardrails.
"""

import re
import mlflow
import pandas as pd
import numpy as np
from typing import Optional


class FraudDataTool:
    """Query engine for the fraud transaction dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._validate(df)

    def _validate(self, df: pd.DataFrame) -> None:
        required = {"customer_id", "transaction_id", "transaction_amount",
                     "fraud", "is_international", "customer_risk_score",
                     "merchant_risk_score", "ip_reputation_score",
                     "failed_transactions_24h"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Fraud dataset missing columns: {missing}")

    @mlflow.trace(span_type="TOOL", name="query_fraud_data")
    def query(self, query: str) -> dict:
        """Route a natural language query to the appropriate handler."""
        q = query.lower()
        df = self.df

        if "international" in q and ("how many" in q or "count" in q):
            return self._count_international(df)

        if "average" in q and "amount" in q and ("fraud" in q or "legitimate" in q):
            return self._avg_amount_by_fraud(df)

        if "highest" in q and "risk" in q:
            return self._top_risk_scores(df)

        if "failed" in q and ("3" in q or "three" in q):
            return self._failed_transactions(df)

        if "suspicious" in q or ("top" in q and "5" in q):
            return self._suspicious_transactions(df)

        if "total" in q and ("number" in q or "count" in q or "transactions" in q):
            return self._total_count(df)

        if "unique" in q and "customer" in q:
            return self._unique_customers(df)

        if "device" in q and ("distribution" in q or "type" in q):
            return self._device_distribution(df)

        if "percentage" in q and "fraud" in q:
            return self._fraud_percentage(df)

        # Customer lookup
        cust = re.search(r"CUST\d+", query, re.IGNORECASE)
        if cust:
            return self._customer_lookup(df, cust.group().upper())

        return self._dataset_summary(df)

    def _count_international(self, df: pd.DataFrame) -> dict:
        count = int(df["is_international"].sum())
        total = len(df)
        return {
            "result": f"There are {count} international transactions out of {total} total ({count/total*100:.1f}%).",
            "row_count": count,
        }

    def _avg_amount_by_fraud(self, df: pd.DataFrame) -> dict:
        stats = df.groupby("fraud")["transaction_amount"].agg(["mean", "median", "count"]).reset_index()
        stats["label"] = stats["fraud"].map({0: "Legitimate", 1: "Fraudulent"})
        result = "Average transaction amounts:\n"
        for _, r in stats.iterrows():
            result += f"  - {r['label']}: ${r['mean']:.2f} (median: ${r['median']:.2f}, n={int(r['count'])})\n"
        return {"result": result, "row_count": len(stats), "data": stats.to_dict("records")}

    def _top_risk_scores(self, df: pd.DataFrame, n: int = 10) -> dict:
        top = df.nlargest(n, "customer_risk_score")[
            ["customer_id", "transaction_id", "customer_risk_score", "transaction_amount", "fraud"]
        ]
        return {
            "result": f"Top {n} by risk score:\n{top.to_string(index=False)}",
            "row_count": n,
            "data": top.to_dict("records"),
        }

    def _failed_transactions(self, df: pd.DataFrame, threshold: int = 3) -> dict:
        failed = df[df["failed_transactions_24h"] > threshold]
        return {
            "result": f"{len(failed)} transactions had more than {threshold} failed attempts in 24h.",
            "row_count": len(failed),
        }

    def _suspicious_transactions(self, df: pd.DataFrame, n: int = 5) -> dict:
        df_s = df.copy()
        max_failed = max(df_s["failed_transactions_24h"].max(), 1)
        df_s["suspicion_score"] = (
            df_s["customer_risk_score"] * 0.3
            + df_s["merchant_risk_score"] * 0.2
            + (1 - df_s["ip_reputation_score"]) * 0.2
            + df_s["is_international"] * 0.15
            + df_s["failed_transactions_24h"] / max_failed * 0.15
        )
        top = df_s.nlargest(n, "suspicion_score")[
            ["transaction_id", "customer_id", "transaction_amount",
             "merchant_category", "suspicion_score", "customer_risk_score",
             "is_international", "fraud"]
        ]
        return {
            "result": f"Top {n} suspicious:\n{top.to_string(index=False)}",
            "row_count": n,
            "data": top.to_dict("records"),
        }

    def _total_count(self, df: pd.DataFrame) -> dict:
        return {
            "result": f"Total transactions in dataset: {len(df)}.",
            "row_count": len(df),
        }

    def _unique_customers(self, df: pd.DataFrame) -> dict:
        n = df["customer_id"].nunique()
        return {
            "result": f"There are {n} unique customers in the fraud dataset.",
            "row_count": n,
        }

    def _device_distribution(self, df: pd.DataFrame) -> dict:
        dist = df["device_type"].value_counts()
        result = "Device type distribution:\n"
        for device, count in dist.items():
            result += f"  - {device}: {count} ({count/len(df)*100:.1f}%)\n"
        return {"result": result, "row_count": len(dist), "data": dist.to_dict()}

    def _fraud_percentage(self, df: pd.DataFrame) -> dict:
        pct = df["fraud"].mean() * 100
        fraud_n = int(df["fraud"].sum())
        return {
            "result": f"{pct:.1f}% of transactions ({fraud_n} out of {len(df)}) are flagged as fraud.",
            "row_count": fraud_n,
        }

    def _customer_lookup(self, df: pd.DataFrame, customer_id: str) -> dict:
        rows = df[df["customer_id"] == customer_id]
        if len(rows) == 0:
            return {"result": f"No transactions found for {customer_id}.", "row_count": 0}
        return {
            "result": f"{len(rows)} transaction(s) for {customer_id}:\n{rows.to_string(index=False)}",
            "row_count": len(rows),
            "data": rows.to_dict("records"),
        }

    def _dataset_summary(self, df: pd.DataFrame) -> dict:
        return {
            "result": (
                f"Dataset: {len(df)} transactions, {int(df['fraud'].sum())} fraud, "
                f"{df['customer_id'].nunique()} customers, avg ${df['transaction_amount'].mean():.2f}"
            ),
            "row_count": len(df),
        }


class PurchaseDataTool:
    """Query engine for the customer purchase dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @mlflow.trace(span_type="TOOL", name="query_purchase_data")
    def query(self, query: str) -> dict:
        """Route a natural language query to the appropriate handler."""
        q = query.lower()
        df = self.df

        if any(w in q for w in ["tier", "membership", "compare"]):
            return self._tier_comparison(df)

        if "loyalty" in q:
            return self._loyalty_by_tier(df)

        return self._dataset_summary(df)

    def _tier_comparison(self, df: pd.DataFrame) -> dict:
        stats = (
            df.groupby("membership_tier")["purchase_amount"]
            .agg(["mean", "median", "std", "count"])
            .round(2)
            .reset_index()
        )
        result = "Purchase patterns by membership tier:\n"
        for _, r in stats.iterrows():
            result += f"  - {r['membership_tier'].title()}: avg=${r['mean']:.2f}, median=${r['median']:.2f}, n={int(r['count'])}\n"
        return {"result": result, "row_count": len(stats), "data": stats.to_dict("records")}

    def _loyalty_by_tier(self, df: pd.DataFrame) -> dict:
        stats = (
            df.groupby("membership_tier")["loyalty_points"]
            .agg(["mean", "median", "count"])
            .round(0)
            .reset_index()
        )
        result = "Average loyalty points by membership tier:\n"
        for _, r in stats.iterrows():
            result += f"  - {r['membership_tier'].title()}: avg={r['mean']:.0f}, median={r['median']:.0f}\n"
        return {"result": result, "row_count": len(stats), "data": stats.to_dict("records")}

    def _dataset_summary(self, df: pd.DataFrame) -> dict:
        return {
            "result": f"Purchase dataset: {len(df)} customers, avg ${df['purchase_amount'].mean():.2f}",
            "row_count": len(df),
        }


# Module-level convenience functions (for backwards compatibility with notebooks)
_fraud_tool: Optional[FraudDataTool] = None
_purchase_tool: Optional[PurchaseDataTool] = None


def init_data_tools(fraud_df: pd.DataFrame, purchase_df: pd.DataFrame) -> None:
    """Initialize the module-level tool instances."""
    global _fraud_tool, _purchase_tool
    _fraud_tool = FraudDataTool(fraud_df)
    _purchase_tool = PurchaseDataTool(purchase_df)


def query_fraud_data(query: str) -> dict:
    """Query the fraud dataset. Call init_data_tools() first."""
    if _fraud_tool is None:
        raise RuntimeError("Call init_data_tools() before using query_fraud_data()")
    return _fraud_tool.query(query)


def query_purchase_data(query: str) -> dict:
    """Query the purchase dataset. Call init_data_tools() first."""
    if _purchase_tool is None:
        raise RuntimeError("Call init_data_tools() before using query_purchase_data()")
    return _purchase_tool.query(query)
