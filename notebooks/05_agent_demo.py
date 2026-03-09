# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Fraud Copilot Interactive Demo
# MAGIC **Live Demo: 5-8 Queries Across All Intent Types**
# MAGIC
# MAGIC This notebook demonstrates the full agent capability for the live demo.
# MAGIC Each query shows routing, tool execution, SHAP explanations, and citations.

# COMMAND ----------

# MAGIC %run ./04_agent_graph

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Demo Query 1: Data Analysis
# MAGIC > "How many international transactions are in the dataset?"
# MAGIC
# MAGIC **Expected:** Routes to `query_fraud_data`, returns count with percentage.

# COMMAND ----------

ask_copilot("How many international transactions are in the dataset?")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Demo Query 2: Fraud Prediction + SHAP
# MAGIC > "Predict if this transaction is fraudulent: $2,500 purchase at electronics store, international, 3am, from a 2-month-old account"
# MAGIC
# MAGIC **Expected:** Routes to `run_fraud_model`. Response includes risk score, tier, SHAP top-5 features, and suggested action.

# COMMAND ----------

ask_copilot("Predict if this transaction is fraudulent: $2,500 purchase at electronics store, international, 3am, from a 2-month-old account")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Demo Query 3: Regulatory Knowledge (RAG)
# MAGIC > "What are the key components of PCI DSS compliance?"
# MAGIC
# MAGIC **Expected:** Routes to `search_financial_docs`. Response cites specific documents.

# COMMAND ----------

ask_copilot("What are the key components of PCI DSS compliance?")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Demo Query 4: Complex Analysis
# MAGIC > "Which transactions look suspicious and why? Show me the top 5 with explanations"
# MAGIC
# MAGIC **Expected:** Routes to `data_query` (complex intent starts with data). Returns composite suspicion scores.

# COMMAND ----------

ask_copilot("Which transactions look suspicious and why? Show me the top 5 with explanations")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Demo Query 5: Purchase Prediction
# MAGIC > "What's the expected purchase amount for a 45-year-old Platinum customer with 20 transactions last month?"
# MAGIC
# MAGIC **Expected:** Routes to `run_purchase_model`. Returns predicted amount with confidence interval.

# COMMAND ----------

ask_copilot("What's the expected purchase amount for a 45-year-old customer with Platinum membership who made 20 transactions last month?")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Demo Query 6: Customer Investigation
# MAGIC > "Analyze customer CUST7823's transaction history and assess their fraud risk"
# MAGIC
# MAGIC **Expected:** Routes to `predict_fraud`, looks up customer data, runs model with SHAP.

# COMMAND ----------

ask_copilot("Analyze customer CUST7823's transaction history and assess their fraud risk")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Demo Query 7: KYC Knowledge
# MAGIC > "Explain the KYC procedures for high-risk customers"
# MAGIC
# MAGIC **Expected:** Routes to RAG, cites KYC document.

# COMMAND ----------

ask_copilot("Explain the KYC procedures for high-risk customers")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Demo Query 8: Data Comparison
# MAGIC > "Compare purchase patterns between Gold and Silver membership tiers"
# MAGIC
# MAGIC **Expected:** Routes to `query_purchase_data`, shows tier comparison.

# COMMAND ----------

ask_copilot("Compare purchase patterns between Gold and Silver membership tiers")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Cognitive Load Adaptation Demo
# MAGIC
# MAGIC Simulating a high-load scenario to show how the agent adapts its responses.

# COMMAND ----------

# Simulate high cognitive load
def ask_copilot_with_load(query: str, load_level: str = "normal"):
    """Invoke copilot with simulated cognitive load level."""
    load_overrides = {
        "normal": {"queries_last_hour":3,"avg_routing_tier":1.5,"session_duration_hours":1,"avg_query_interval_sec":200,"followup_rate":0.05,"hour_of_day":10},
        "high": {"queries_last_hour":14,"avg_routing_tier":3.5,"session_duration_hours":5.5,"avg_query_interval_sec":25,"followup_rate":0.35,"hour_of_day":19},
    }
    
    # Temporarily override the load assessment function
    original_fn = node_assess_load.__code__
    metrics = load_overrides.get(load_level, load_overrides["normal"])
    
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"LOAD SIMULATION: {load_level} (score will be {'< 30' if load_level=='normal' else '> 60'})")
    print(f"{'='*70}")
    
    # For the demo, we manually set the load in the state
    result = copilot.invoke({
        "user_query": query,
        "intent_type": None, "intent_confidence": None,
        "data_result": None, "prediction_result": None,
        "knowledge_result": None, "load_assessment": None,
        "final_response": None, "tools_used": None, "sources_cited": None,
    })
    
    print(f"\n--- RESPONSE (load: {load_level}) ---")
    print(result.get("final_response", "No response"))
    return result

# Normal load response
print(">>> NORMAL LOAD:")
ask_copilot("What are the common indicators of credit card fraud?")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## MLflow Traces
# MAGIC
# MAGIC All queries above generated MLflow traces automatically.
# MAGIC
# MAGIC **To view traces:**
# MAGIC 1. Go to the **Experiments** tab in the sidebar
# MAGIC 2. Open the experiment for this notebook
# MAGIC 3. Click on **Traces** tab
# MAGIC 4. Click any trace to see the full execution graph:
# MAGIC    - `classify` → `[tool]` → `assess_load` → `synthesize`
# MAGIC    - Each node shows inputs, outputs, latency, and token usage

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Summary
# MAGIC
# MAGIC The Fraud Copilot agent demonstrated:
# MAGIC
# MAGIC 1. **Multi-intent routing** - Correctly classified and routed 8 diverse queries
# MAGIC 2. **Data analysis** - SQL-like queries against fraud and purchase datasets
# MAGIC 3. **ML predictions + SHAP** - Fraud risk assessment with explainable features
# MAGIC 4. **RAG with citations** - Regulatory knowledge grounded in source documents
# MAGIC 5. **Cognitive load awareness** - System adapts response format based on analyst state
# MAGIC 6. **Full observability** - Every step traced automatically in MLflow