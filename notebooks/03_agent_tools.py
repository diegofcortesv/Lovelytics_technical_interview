# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Agent Tools
# MAGIC **Fraud Copilot | Initialize & Test Each Tool**
# MAGIC
# MAGIC Tools accept raw data (strings for categoricals). The Pipeline models handle encoding.

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np
import shap

CATALOG = "fraud_agent"
SCHEMA = "default"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 Tool: Query Fraud Data

# COMMAND ----------

fraud_df = spark.table(f"{CATALOG}.{SCHEMA}.fraud_dataset").toPandas()
purchase_df = spark.table(f"{CATALOG}.{SCHEMA}.product_purchase_dataset").toPandas()
print(f"Fraud data: {fraud_df.shape}, Purchase data: {purchase_df.shape}")

# COMMAND ----------

@mlflow.trace(span_type="TOOL", name="query_fraud_data")
def query_fraud_data(query: str) -> dict:
    """Query the fraud dataset based on natural language."""
    q = query.lower()
    df = fraud_df
    
    if "international" in q and ("how many" in q or "count" in q):
        count = int(df["is_international"].sum())
        return {"result": f"There are {count} international transactions out of {len(df)} total ({count/len(df)*100:.1f}%).", "row_count": count}
    
    if "average" in q and "amount" in q and ("fraud" in q or "legitimate" in q):
        stats = df.groupby("fraud")["transaction_amount"].agg(["mean","median","count"]).reset_index()
        stats["label"] = stats["fraud"].map({0:"Legitimate", 1:"Fraudulent"})
        result = "Average transaction amounts:\n"
        for _, r in stats.iterrows():
            result += f"  - {r['label']}: ${r['mean']:.2f} (median: ${r['median']:.2f}, n={int(r['count'])})\n"
        return {"result": result, "row_count": len(stats), "data": stats.to_dict("records")}
    
    if "highest" in q and "risk" in q:
        top = df.nlargest(10, "customer_risk_score")[["customer_id","transaction_id","customer_risk_score","transaction_amount","fraud"]]
        return {"result": f"Top 10 by risk score:\n{top.to_string(index=False)}", "row_count": 10, "data": top.to_dict("records")}
    
    if "failed" in q:
        failed = df[df["failed_transactions_24h"] > 3]
        return {"result": f"{len(failed)} transactions had more than 3 failed attempts in 24h.", "row_count": len(failed)}
    
    if "suspicious" in q or ("top" in q and "5" in q):
        df_s = df.copy()
        df_s["suspicion_score"] = (
            df_s["customer_risk_score"]*0.3 + df_s["merchant_risk_score"]*0.2 +
            (1-df_s["ip_reputation_score"])*0.2 + df_s["is_international"]*0.15 +
            df_s["failed_transactions_24h"]/max(df_s["failed_transactions_24h"].max(),1)*0.15
        )
        top = df_s.nlargest(5, "suspicion_score")[["transaction_id","customer_id","transaction_amount","merchant_category","suspicion_score","customer_risk_score","is_international","fraud"]]
        return {"result": f"Top 5 suspicious:\n{top.to_string(index=False)}", "row_count": 5, "data": top.to_dict("records")}
    
    import re
    cust = re.search(r'CUST\d+', query, re.IGNORECASE)
    if cust:
        cid = cust.group().upper()
        rows = df[df["customer_id"]==cid]
        if len(rows)==0:
            return {"result": f"No transactions found for {cid}.", "row_count": 0}
        return {"result": f"{len(rows)} transaction(s) for {cid}:\n{rows.to_string(index=False)}", "row_count": len(rows), "data": rows.to_dict("records")}
    
    return {"result": f"Dataset: {len(df)} transactions, {int(df['fraud'].sum())} fraud, {df['customer_id'].nunique()} customers, avg ${df['transaction_amount'].mean():.2f}", "row_count": len(df)}


@mlflow.trace(span_type="TOOL", name="query_purchase_data")
def query_purchase_data(query: str) -> dict:
    """Query the purchase dataset."""
    q = query.lower()
    df = purchase_df
    
    if "tier" in q or "membership" in q or "compare" in q:
        stats = df.groupby("membership_tier")["purchase_amount"].agg(["mean","median","std","count"]).round(2).reset_index()
        result = "Purchase patterns by membership tier:\n"
        for _, r in stats.iterrows():
            result += f"  - {r['membership_tier'].title()}: avg=${r['mean']:.2f}, median=${r['median']:.2f}, n={int(r['count'])}\n"
        return {"result": result, "row_count": len(stats), "data": stats.to_dict("records")}
    
    return {"result": f"Purchase dataset: {len(df)} customers, avg ${df['purchase_amount'].mean():.2f}", "row_count": len(df)}


# Tests
print("=== query_fraud_data ===")
print(query_fraud_data("How many international transactions?")["result"])
print("\n=== query_purchase_data ===")
print(query_purchase_data("Compare Gold and Silver tiers")["result"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 Tool: Fraud Model Prediction + SHAP
# MAGIC
# MAGIC The model is a Pipeline. We send raw data (strings). For SHAP, we access the
# MAGIC pipeline internals to get the transformed features and the underlying XGBClassifier.

# COMMAND ----------

FRAUD_MODEL_NAME = f"{CATALOG}.{SCHEMA}.fraud_detection_model"

# Load as pyfunc (for prediction with raw data)
fraud_model = mlflow.pyfunc.load_model(f"models:/{FRAUD_MODEL_NAME}@champion")

# Also load the native sklearn Pipeline (for SHAP access)
fraud_pipeline = mlflow.sklearn.load_model(f"models:/{FRAUD_MODEL_NAME}@champion")
fraud_preprocessor = fraud_pipeline.named_steps["preprocessor"]
fraud_xgb = fraud_pipeline.named_steps["classifier"]
fraud_explainer = shap.TreeExplainer(fraud_xgb)

fraud_features = [inp.name for inp in fraud_model.metadata.signature.inputs.inputs]
print(f"Fraud model loaded: {len(fraud_features)} features, accepts raw strings")

# COMMAND ----------

@mlflow.trace(span_type="TOOL", name="run_fraud_model")
def run_fraud_model(features: dict) -> dict:
    """Predict fraud probability with SHAP explanations. Accepts raw data."""
    input_df = pd.DataFrame([features])[fraud_features]
    
    for col in input_df.select_dtypes(include=['int64']).columns:
        input_df[col] = input_df[col].astype('int32')

    # Predict via pyfunc (Pipeline handles encoding)
    prob = float(fraud_model.predict(input_df)[0])
    risk_score = int(prob * 100)
    
    if prob >= 0.7:
        risk_tier, action = "HIGH", "Block transaction and verify identity immediately"
    elif prob >= 0.4:
        risk_tier, action = "MEDIUM", "Flag for manual review within 24 hours"
    else:
        risk_tier, action = "LOW", "No immediate action, monitor patterns"
    
    # SHAP: transform the raw data, then explain on the underlying XGB model
    try:
        X_enc = fraud_preprocessor.transform(input_df)
        sv = fraud_explainer.shap_values(X_enc)
        if isinstance(sv, list): sv = sv[1]
        vals = sv[0] if len(sv.shape) > 1 else sv
        
        impacts = sorted(
            [{"feature": fraud_features[i], "shap_value": round(float(vals[i]),4)} for i in range(min(len(vals), len(fraud_features)))],
            key=lambda x: abs(x["shap_value"]), reverse=True
        )[:5]
        for item in impacts:
            d = "increases" if item["shap_value"] > 0 else "decreases"
            item["explanation"] = f"{item['feature']} {d} fraud risk (SHAP: {item['shap_value']:.4f})"
    except Exception as e:
        impacts = [{"feature": "SHAP error", "explanation": str(e)}]
    
    return {
        "probability": round(prob, 4),
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "suggested_action": action,
        "top_features": impacts
    }


# Test with a known fraud transaction - using RAW data (strings!)
fraud_row = fraud_df[fraud_df["fraud"]==1].iloc[0]
test_features = {f: fraud_row[f] for f in fraud_features}
result = run_fraud_model(test_features)

print("=== TEST: run_fraud_model (known fraud, RAW strings) ===")
print(f"P(fraud): {result['probability']:.4f}")
print(f"Risk tier: {result['risk_tier']}")
print(f"Action: {result['suggested_action']}")
print("Top features:")
for f in result["top_features"]:
    print(f"  {f.get('explanation','')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 Tool: Purchase Model Prediction + SHAP

# COMMAND ----------

PURCHASE_MODEL_NAME = f"{CATALOG}.{SCHEMA}.purchase_prediction_model"

purchase_model = mlflow.pyfunc.load_model(f"models:/{PURCHASE_MODEL_NAME}@champion")
purchase_pipeline = mlflow.sklearn.load_model(f"models:/{PURCHASE_MODEL_NAME}@champion")
purchase_preprocessor = purchase_pipeline.named_steps["preprocessor"]
purchase_lgbm = purchase_pipeline.named_steps["regressor"]
purchase_explainer = shap.TreeExplainer(purchase_lgbm)

purchase_features = [inp.name for inp in purchase_model.metadata.signature.inputs.inputs]
print(f"Purchase model loaded: {len(purchase_features)} features, accepts raw strings")

# COMMAND ----------

@mlflow.trace(span_type="TOOL", name="run_purchase_model")
def run_purchase_model(features: dict) -> dict:
    """Predict expected purchase amount. Accepts raw data."""
    input_df = pd.DataFrame([features])[purchase_features]
    
    for col in input_df.select_dtypes(include=["int64"]).columns:
        input_df[col] = input_df[col].astype("int32")
    for col in input_df.select_dtypes(include=["float64"]).columns:
        input_df[col] = input_df[col].astype("float64")  

    pred_raw = float(purchase_model.predict(input_df)[0])
    
    if pred_raw < 10:
        pred = float(np.expm1(pred_raw))
    else:
        pred = pred_raw
    
    margin = pred * 0.20
    
    try:
        X_enc = purchase_preprocessor.transform(input_df)
        sv = purchase_explainer.shap_values(X_enc)
        vals = sv[0] if len(sv.shape) > 1 else sv
        impacts = sorted(
            [{"feature": purchase_features[i], "shap_value": round(float(vals[i]),4)} for i in range(min(len(vals), len(purchase_features)))],
            key=lambda x: abs(x["shap_value"]), reverse=True
        )[:5]
    except:
        impacts = []
    
    return {
        "predicted_amount": round(pred, 2),
        "confidence_interval": {"low": round(max(0, pred-margin),2), "high": round(pred+margin,2)},
        "top_features": impacts
    }


# Test with raw data
test_row = purchase_df.iloc[0]
test_pf = {f: test_row[f] for f in purchase_features if f in test_row.index}
for f in purchase_features:
    if f not in test_pf: test_pf[f] = 0
result = run_purchase_model(test_pf)
print(f"=== TEST: run_purchase_model (RAW strings) ===")
print(f"Predicted: ${result['predicted_amount']:.2f}, actual: ${test_row['purchase_amount']:.2f}")
print(f"Interval: ${result['confidence_interval']['low']:.2f} - ${result['confidence_interval']['high']:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 Tool: RAG Search over Financial Documents

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load chunks for TF-IDF fallback
chunks_df = spark.table(f"{CATALOG}.{SCHEMA}.financial_docs_chunks").toPandas()
print(f"Loaded {len(chunks_df)} chunks from {chunks_df['doc_name'].nunique()} documents")

# Build TF-IDF fallback index
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
chunk_vectors = vectorizer.fit_transform(chunks_df["content"].tolist())

# Try to connect to Vector Search
VS_ENDPOINT_NAME = "fraud-copilot-vs-endpoint"
VS_INDEX_NAME = f"{CATALOG}.{SCHEMA}.financial_docs_index"
USE_VECTOR_SEARCH = False

try:
    from databricks.vector_search.client import VectorSearchClient
    vs_client = VectorSearchClient()
    vs_index = vs_client.get_index(
        endpoint_name=VS_ENDPOINT_NAME,
        index_name=VS_INDEX_NAME
    )
    # Verify index is online
    status = vs_index.describe().get("status", {}).get("ready", False)
    if status:
        USE_VECTOR_SEARCH = True
        print(f"Vector Search index ready: {VS_INDEX_NAME}")
    else:
        print("Vector Search index not ready, using TF-IDF fallback")
except Exception as e:
    print(f"Vector Search unavailable ({e}), using TF-IDF fallback")

print(f"RAG backend: {'Vector Search' if USE_VECTOR_SEARCH else 'TF-IDF'}")

# COMMAND ----------

@mlflow.trace(span_type="RETRIEVER", name="search_financial_docs")
def search_financial_docs(query: str, top_k: int = 5) -> dict:
    """Search financial knowledge base with citations."""
    
    if USE_VECTOR_SEARCH:
        # --- Vector Search path ---
        try:
            results_raw = vs_index.similarity_search(
                query_text=query,
                columns=["chunk_id", "content", "source", "doc_name"],
                num_results=top_k,
            )
            
            results = []
            sources = set()
            for row in results_raw.get("result", {}).get("data_array", []):
                # data_array returns lists: [chunk_id, content, source, doc_name, score]
                content = str(row[1])[:1500]
                source = str(row[2])
                score = float(row[-1]) if len(row) > 4 else 0.0
                results.append({
                    "text": content,
                    "source": source,
                    "score": score,
                })
                sources.add(source)
            
            context = "\n\n---\n\n".join(
                [f"[Source: {r['source']}]\n{r['text']}" for r in results]
            )
            return {
                "answer_context": context,
                "sources": list(sources),
                "num_results": len(results),
                "backend": "vector_search",
            }
        except Exception as e:
            print(f"Vector Search query failed ({e}), falling back to TF-IDF")
    
    # --- TF-IDF fallback path ---
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, chunk_vectors).flatten()
    top_idx = np.argsort(sims)[-top_k:][::-1]
    
    results = []
    sources = set()
    for idx in top_idx:
        if sims[idx] > 0.05:
            row = chunks_df.iloc[idx]
            results.append({
                "text": row["content"][:1500],
                "source": row["source"],
                "score": float(sims[idx]),
            })
            sources.add(row["source"])
    
    context = "\n\n---\n\n".join(
        [f"[Source: {r['source']}]\n{r['text']}" for r in results]
    )
    return {
        "answer_context": context,
        "sources": list(sources),
        "num_results": len(results),
        "backend": "tfidf",
    }

# COMMAND ----------

index = vs_client.get_index(
    endpoint_name=VS_ENDPOINT_NAME,
    index_name=VS_INDEX_NAME
)

status = index.describe()
print(f"Status: {status.get('status', {})}")

# Test query
test_results = index.similarity_search(
    query_text="What are the key components of PCI DSS compliance?",
    columns=["content", "source"],
    num_results=3,
)
print(f"Test results: {len(test_results.get('result', {}).get('data_array', []))} hits")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 Tool: Cognitive Load Assessment

# COMMAND ----------

@mlflow.trace(span_type="TOOL", name="assess_analyst_load")
def assess_analyst_load(session_metrics: dict) -> dict:
    """Calculate cognitive load score (0-100)."""
    score = 0.0
    score += min(session_metrics.get("queries_last_hour",0)/15.0, 1.0) * 20
    score += (session_metrics.get("avg_routing_tier",1.0)/4.0) * 20
    score += min(session_metrics.get("session_duration_hours",0)/6.0, 1.0) * 15
    interval = session_metrics.get("avg_query_interval_sec", 300)
    score += max(0, 1.0-interval/300.0) * 15
    score += min(session_metrics.get("followup_rate",0)/0.4, 1.0) * 15
    hour = session_metrics.get("hour_of_day", 12)
    circadian = 0.5 if 10<=hour<=14 else (0.8 if hour>18 or hour<6 else 0.3)
    score += circadian * 15
    
    load_score = min(int(score), 100)
    if load_score <= 30: level = "normal"
    elif load_score <= 60: level = "elevated"
    elif load_score <= 80: level = "high"
    else: level = "critical"
    
    return {"load_score": load_score, "level": level}


print("=== TEST: assess_analyst_load ===")
print("Normal:", assess_analyst_load({"queries_last_hour":3,"avg_routing_tier":1.5,"session_duration_hours":1,"avg_query_interval_sec":200,"followup_rate":0.05,"hour_of_day":10}))
print("High:", assess_analyst_load({"queries_last_hour":14,"avg_routing_tier":3.5,"session_duration_hours":5,"avg_query_interval_sec":30,"followup_rate":0.35,"hour_of_day":19}))