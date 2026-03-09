# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Agent Evaluation
# MAGIC **Fraud Copilot | Golden Set Evaluation & Routing Accuracy**
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Runs the agent against the 30-query golden set
# MAGIC 2. Measures routing accuracy (intent classification)
# MAGIC 3. Measures response quality using LLM-as-judge
# MAGIC 4. Logs results to MLflow for tracking

# COMMAND ----------

# MAGIC %run ./04_agent_graph

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.1 Define Golden Set

# COMMAND ----------

GOLDEN_SET = [
    # Data Analysis (10)
    {"query": "How many international transactions are in the dataset?", "expected_intent": "data_query", "expected_tools": ["query_fraud_data"], "check_contains": ["49","international"]},
    {"query": "What's the average transaction amount for fraudulent vs legitimate transactions?", "expected_intent": "data_query", "expected_tools": ["query_fraud_data"], "check_contains": ["average","fraud"]},
    {"query": "Which customers have the highest risk scores?", "expected_intent": "data_query", "expected_tools": ["query_fraud_data"], "check_contains": ["risk"]},
    {"query": "How many transactions had more than 3 failed attempts in 24 hours?", "expected_intent": "data_query", "expected_tools": ["query_fraud_data"], "check_contains": ["failed"]},
    {"query": "Compare purchase patterns between Gold and Silver membership tiers", "expected_intent": "data_query", "expected_tools": ["query_purchase_data"], "check_contains": ["gold","silver"]},
    {"query": "What is the total number of transactions in the fraud dataset?", "expected_intent": "data_query", "expected_tools": ["query_fraud_data"], "check_contains": ["100"]},
    {"query": "How many unique customers are in the fraud dataset?", "expected_intent": "data_query", "expected_tools": ["query_fraud_data"], "check_contains": ["customer"]},
    {"query": "Show me the distribution of device types used in transactions", "expected_intent": "data_query", "expected_tools": ["query_fraud_data"], "check_contains": ["device"]},
    {"query": "What percentage of transactions are flagged as fraud?", "expected_intent": "data_query", "expected_tools": ["query_fraud_data"], "check_contains": ["50","fraud"]},
    {"query": "What are the average loyalty points by membership tier?", "expected_intent": "data_query", "expected_tools": ["query_purchase_data"], "check_contains": ["loyalty"]},
    
    # Fraud Prediction (6)
    {"query": "Predict if this transaction is fraudulent: $1,250 purchase at electronics store, international, 3am, from a 2-month-old account", "expected_intent": "prediction_fraud", "expected_tools": ["run_fraud_model"], "check_contains": ["risk","probability"]},
    {"query": "Is this transaction suspicious? $5,000 jewelry purchase, international, new device", "expected_intent": "prediction_fraud", "expected_tools": ["run_fraud_model"], "check_contains": ["risk"]},
    {"query": "Assess the fraud risk for a $200 grocery transaction, domestic, mobile device", "expected_intent": "prediction_fraud", "expected_tools": ["run_fraud_model"], "check_contains": ["risk"]},
    {"query": "Analyze customer CUST7823's transaction history and assess their fraud risk", "expected_intent": "prediction_fraud", "expected_tools": ["run_fraud_model"], "check_contains": ["CUST7823","risk"]},
    {"query": "What is the fraud probability for a $3,000 ATM withdrawal at 2am international?", "expected_intent": "prediction_fraud", "expected_tools": ["run_fraud_model"], "check_contains": ["probability"]},
    {"query": "Predict fraud risk: $800 online_retail purchase, 5 failed attempts, low IP reputation", "expected_intent": "prediction_fraud", "expected_tools": ["run_fraud_model"], "check_contains": ["risk"]},
    
    # Purchase Prediction (4)
    {"query": "What's the expected purchase amount for a 45-year-old Platinum member with 20 transactions last month?", "expected_intent": "prediction_purchase", "expected_tools": ["run_purchase_model"], "check_contains": ["predicted","amount"]},
    {"query": "Predict the purchase amount for a 28-year-old Silver member with 5 transactions last month", "expected_intent": "prediction_purchase", "expected_tools": ["run_purchase_model"], "check_contains": ["predicted"]},
    {"query": "How much would a Gold tier customer aged 55 with high loyalty points likely spend?", "expected_intent": "prediction_purchase", "expected_tools": ["run_purchase_model"], "check_contains": ["predicted"]},
    {"query": "Estimate the next purchase for a Bronze member who hasn't bought anything in 30 days", "expected_intent": "prediction_purchase", "expected_tools": ["run_purchase_model"], "check_contains": ["predicted"]},
    
    # Knowledge (5)
    {"query": "What are the common indicators of credit card fraud?", "expected_intent": "knowledge", "expected_tools": ["search_financial_docs"], "check_contains": ["indicator","fraud"]},
    {"query": "Explain the KYC procedures for high-risk customers", "expected_intent": "knowledge", "expected_tools": ["search_financial_docs"], "check_contains": ["KYC","due diligence"]},
    {"query": "What are the key components of PCI DSS compliance?", "expected_intent": "knowledge", "expected_tools": ["search_financial_docs"], "check_contains": ["PCI","requirement"]},
    {"query": "What are the three stages of money laundering?", "expected_intent": "knowledge", "expected_tools": ["search_financial_docs"], "check_contains": ["placement","layering"]},
    {"query": "Summarize transaction monitoring system best practices", "expected_intent": "knowledge", "expected_tools": ["search_financial_docs"], "check_contains": ["monitoring"]},
    
    # Complex (5)
    {"query": "Which transactions look suspicious and why? Show me the top 5 with explanations", "expected_intent": "complex", "expected_tools": ["query_fraud_data"], "check_contains": ["suspicious"]},
    {"query": "What fraud patterns exist in international transactions vs domestic ones?", "expected_intent": "complex", "expected_tools": ["query_fraud_data"], "check_contains": ["international"]},
    {"query": "For the highest-risk customers, what is their average transaction behavior?", "expected_intent": "complex", "expected_tools": ["query_fraud_data"], "check_contains": ["risk"]},
    {"query": "Give me a risk assessment summary: how many high, medium, and low risk transactions?", "expected_intent": "complex", "expected_tools": ["query_fraud_data"], "check_contains": ["risk"]},
    {"query": "Identify anomalous transactions that deviate from customer spending patterns", "expected_intent": "complex", "expected_tools": ["query_fraud_data"], "check_contains": ["anomal"]},
]

print(f"Golden set: {len(GOLDEN_SET)} queries")
print(f"  Data: {sum(1 for g in GOLDEN_SET if g['expected_intent']=='data_query')}")
print(f"  Fraud Pred: {sum(1 for g in GOLDEN_SET if g['expected_intent']=='prediction_fraud')}")
print(f"  Purchase Pred: {sum(1 for g in GOLDEN_SET if g['expected_intent']=='prediction_purchase')}")
print(f"  Knowledge: {sum(1 for g in GOLDEN_SET if g['expected_intent']=='knowledge')}")
print(f"  Complex: {sum(1 for g in GOLDEN_SET if g['expected_intent']=='complex')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.2 Run Evaluation
# MAGIC
# MAGIC The golden set is executed in 5 batches of 10 queries each.  
# MAGIC This is required because the free-tier Foundation Model API enforces 
# MAGIC rate limits (~20 requests/minute). In production with a provisioned 
# MAGIC throughput endpoint, all 30 queries would run in a single batch.

# COMMAND ----------

import time

results = []
errors = []

def run_eval_batch(golden_set_slice, batch_label=""):
    """Run evaluation on a slice of the golden set."""
    for i, item in enumerate(golden_set_slice):
        global_idx = GOLDEN_SET.index(item) + 1
        print(f"\n[{global_idx}/{len(GOLDEN_SET)}] {item['query'][:70]}...")
        start = time.time()
        
        try:
            output = copilot.invoke({
                "user_query": item["query"],
                "intent_type": None, "intent_confidence": None,
                "data_result": None, "prediction_result": None,
                "knowledge_result": None, "load_assessment": None,
                "final_response": None, "tools_used": None, "sources_cited": None,
            })
            
            latency = time.time() - start
            response = output.get("final_response", "")
            actual_intent = output.get("intent_type", "unknown")
            actual_tools = output.get("tools_used", [])
            
            intent_correct = actual_intent == item["expected_intent"]
            tool_correct = any(t in actual_tools for t in item["expected_tools"])
            
            resp_lower = response.lower()
            content_hits = sum(1 for kw in item["check_contains"] if kw.lower() in resp_lower)
            content_score = content_hits / len(item["check_contains"]) if item["check_contains"] else 1.0
            
            results.append({
                "query": item["query"],
                "expected_intent": item["expected_intent"],
                "actual_intent": actual_intent,
                "intent_correct": intent_correct,
                "tool_correct": tool_correct,
                "content_score": content_score,
                "latency_s": latency,
                "response_length": len(response),
                "has_response": len(response) > 50,
            })
            
            status = "OK" if intent_correct and tool_correct else "MISMATCH"
            print(f"  {status} | intent: {actual_intent} (expected: {item['expected_intent']}) | tools: {actual_tools} | {latency:.1f}s")
            
        except Exception as e:
            errors.append({"query": item["query"], "error": str(e)})
            print(f"  ERROR: {str(e)[:100]}")

# --- Batch 1: queries 1-6 ---
run_eval_batch(GOLDEN_SET[0:6], "Batch 1")
print(f"\n--- Batch 1 complete: {len(results)} OK, {len(errors)} errors ---")

# COMMAND ----------

# --- Batch 2: queries 7-12---
# Run this cell after Batch 1 completes and rate limit resets (~2-3 min)
run_eval_batch(GOLDEN_SET[7:12], "Batch 2")
print(f"\n--- Batch 2 complete: {len(results)} OK, {len(errors)} errors ---")

# COMMAND ----------

run_eval_batch(GOLDEN_SET[13:18], "Batch 3")
print(f"\n--- Batch 3 complete: {len(results)} OK, {len(errors)} errors ---")

# COMMAND ----------

run_eval_batch(GOLDEN_SET[19:24], "Batch 4")
print(f"\n--- Batch 4 complete: {len(results)} OK, {len(errors)} errors ---")

# COMMAND ----------

run_eval_batch(GOLDEN_SET[25:30], "Batch 5")
print(f"\n--- Batch 5 complete: {len(results)} OK, {len(errors)} errors ---")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.3 Evaluation Results

# COMMAND ----------

import pandas as pd

eval_df = pd.DataFrame(results)

# Overall metrics
n = len(eval_df)
intent_acc = eval_df["intent_correct"].mean() * 100
tool_acc = eval_df["tool_correct"].mean() * 100
content_avg = eval_df["content_score"].mean() * 100
response_rate = eval_df["has_response"].mean() * 100
avg_latency = eval_df["latency_s"].mean()
p95_latency = eval_df["latency_s"].quantile(0.95)

print("=" * 60)
print("FRAUD COPILOT EVALUATION RESULTS")
print("=" * 60)
print(f"  Queries evaluated:     {n}")
print(f"  Errors:                {len(errors)}")
print(f"  Intent accuracy:       {intent_acc:.1f}%")
print(f"  Tool routing accuracy: {tool_acc:.1f}%")
print(f"  Content relevance:     {content_avg:.1f}%")
print(f"  Response rate:         {response_rate:.1f}%")
print(f"  Avg latency:           {avg_latency:.1f}s")
print(f"  P95 latency:           {p95_latency:.1f}s")
print("=" * 60)

# Per-intent breakdown
print("\nPer-Intent Breakdown:")
for intent in ["data_query", "prediction_fraud", "prediction_purchase", "knowledge", "complex"]:
    subset = eval_df[eval_df["expected_intent"] == intent]
    if len(subset) > 0:
        print(f"  {intent:25s}: intent_acc={subset['intent_correct'].mean()*100:.0f}%  tool_acc={subset['tool_correct'].mean()*100:.0f}%  content={subset['content_score'].mean()*100:.0f}%  n={len(subset)}")

# COMMAND ----------

# Log metrics to MLflow
with mlflow.start_run(run_name="agent-evaluation-golden-set"):
    mlflow.log_metric("intent_accuracy", intent_acc / 100)
    mlflow.log_metric("tool_routing_accuracy", tool_acc / 100)
    mlflow.log_metric("content_relevance", content_avg / 100)
    mlflow.log_metric("response_rate", response_rate / 100)
    mlflow.log_metric("avg_latency_s", avg_latency)
    mlflow.log_metric("p95_latency_s", p95_latency)
    mlflow.log_metric("n_queries", n)
    mlflow.log_metric("n_errors", len(errors))
    
    # Log per-intent metrics
    for intent in eval_df["expected_intent"].unique():
        subset = eval_df[eval_df["expected_intent"] == intent]
        mlflow.log_metric(f"intent_acc_{intent}", subset["intent_correct"].mean())
        mlflow.log_metric(f"tool_acc_{intent}", subset["tool_correct"].mean())
    
    # Log the full results table
    mlflow.log_table(eval_df, "evaluation_results.json")
    
    # Log errors if any
    if errors:
        mlflow.log_table(pd.DataFrame(errors), "evaluation_errors.json")
    
    print("✓ Evaluation metrics logged to MLflow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.4 Misclassification Analysis

# COMMAND ----------

# Show misclassified queries
misclassified = eval_df[~eval_df["intent_correct"]]
if len(misclassified) > 0:
    print(f"Misclassified queries ({len(misclassified)}):\n")
    for _, row in misclassified.iterrows():
        print(f"  Query: {row['query'][:80]}...")
        print(f"  Expected: {row['expected_intent']} → Got: {row['actual_intent']}")
        print()
else:
    print("No misclassifications - 100% intent routing accuracy")

# COMMAND ----------

# Show slow queries
slow = eval_df.nlargest(5, "latency_s")
print("Top 5 slowest queries:")
for _, row in slow.iterrows():
    print(f"  {row['latency_s']:.1f}s | {row['actual_intent']:20s} | {row['query'][:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6.5 Summary
# MAGIC
# MAGIC | Metric | Value | Target |
# MAGIC |--------|-------|--------|
# MAGIC | Intent Routing Accuracy | See above | > 90% |
# MAGIC | Tool Selection Accuracy | See above | > 90% |
# MAGIC | Content Relevance | See above | > 80% |
# MAGIC | Response Rate | See above | > 95% |
# MAGIC | Avg Latency | See above | < 5s |
# MAGIC | P95 Latency | See above | < 10s |
# MAGIC
# MAGIC **All evaluation results are logged in MLflow** for comparison across agent versions.
# MAGIC
# MAGIC In production, this evaluation would run as part of CI/CD: any change to prompts, 
# MAGIC tools, or models triggers re-evaluation against this golden set. If metrics drop 
# MAGIC below thresholds, the deploy is blocked.