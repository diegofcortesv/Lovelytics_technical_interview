# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Agent Graph (LangGraph)
# MAGIC **Fraud Copilot | Multi-Tool Orchestration with Conditional Routing**
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads tools from notebook 03
# MAGIC 2. Configures the LLM (Databricks Foundation Model)
# MAGIC 3. Builds the LangGraph StateGraph
# MAGIC 4. Tests routing for each intent type

# COMMAND ----------

# MAGIC %run ./03_agent_tools

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Configure LLM Client

# COMMAND ----------

import mlflow
mlflow.langchain.autolog()  # Auto-trace all LangGraph executions

from databricks_langchain import ChatDatabricks

LLM_ENDPOINT = "databricks-meta-llama-3-1-405b-instruct"

llm = ChatDatabricks(
    endpoint=LLM_ENDPOINT,
    temperature=0.1,
    max_tokens=1024,
)

# Quick test
response = llm.invoke("Say 'Fraud Copilot ready' in exactly 3 words.")
print(f"LLM connected: {response.content[:100]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Build the LangGraph Agent

# COMMAND ----------

import json
import re
from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END

# ── State Definition ───────────────────────────────────
class CopilotState(TypedDict):
    user_query: str
    intent_type: Optional[str]
    intent_confidence: Optional[float]
    data_result: Optional[dict]
    prediction_result: Optional[dict]
    knowledge_result: Optional[dict]
    load_assessment: Optional[dict]
    final_response: Optional[str]
    tools_used: Optional[list]
    sources_cited: Optional[list]

# ── Node: Classify Intent ──────────────────────────────
def classify_intent(state: CopilotState) -> dict:
    query = state["user_query"]
    
    prompt = f"""You are a query classifier for a financial fraud analysis system.
Classify this query into exactly one category.

IMPORTANT: Read ALL categories before deciding. Choose the MOST SPECIFIC match.

Categories (in priority order):
1. complex: The query requires MULTIPLE analytical steps. Signals:
   - Asks to FIND items AND EXPLAIN why (e.g., "which are suspicious and why")
   - Asks to COMPARE groups AND ANALYZE patterns (e.g., "patterns in X vs Y")
   - Asks for a SUMMARY that requires AGGREGATION + CLASSIFICATION (e.g., "how many high/medium/low risk")
   - Asks to IDENTIFY + ASSESS (e.g., "identify anomalous transactions", "analyze and assess risk")
   - Uses words like: "patterns", "anomalous", "suspicious...why", "risk assessment summary", "analyze...assess"
   
2. prediction_fraud: Predict fraud for ONE SPECIFIC transaction described with concrete details.
   - MUST include specific transaction parameters (amount, merchant, time, etc.)
   - Example: "Is this $2,500 electronics purchase at 3am fraudulent?"
   
3. prediction_purchase: Predict purchase amount for ONE SPECIFIC customer described with concrete details.
   - MUST include specific customer parameters (age, tier, transactions, etc.)
   - Example: "Expected purchase for a 45-year-old Platinum member?"
   
4. data_query: Simple data retrieval - counts, averages, filters, lists, comparisons.
   - Answers come directly from querying/filtering the dataset
   - Example: "How many international transactions?", "Average amount by fraud label?"
   
5. knowledge: Questions about financial regulations, procedures, compliance, fraud theory.
   - About KYC, AML, PCI DSS, fraud indicators, regulations
   - Example: "What are KYC procedures for high-risk customers?"

DECISION RULES:
- If the query asks to FIND + EXPLAIN or ANALYZE + ASSESS → complex
- If the query describes ONE specific transaction/customer to predict → prediction_*
- If the query asks for simple numbers or lists → data_query
- If the query is about regulations/procedures/theory → knowledge

Query: "{query}"

Respond ONLY with JSON: {{"intent": "<category>", "confidence": <0.0-1.0>}}"""

    response = llm.invoke(prompt)
    content = response.content
    
    try:
        j_start = content.find("{")
        j_end = content.rfind("}") + 1
        parsed = json.loads(content[j_start:j_end])
    except:
        parsed = {"intent": "data_query", "confidence": 0.5}
    
    intent = parsed.get("intent", "data_query")
    conf = parsed.get("confidence", 0.5)
    
    print(f"  [classify] intent={intent}, confidence={conf:.2f}")
    return {"intent_type": intent, "intent_confidence": conf, "tools_used": []}


# ── Node: Data Query ───────────────────────────────────
def node_data_query(state: CopilotState) -> dict:
    query = state["user_query"]
    q = query.lower()
    
    if any(w in q for w in ["purchase","tier","membership","spending pattern"]):
        result = query_purchase_data(query)
        tool = "query_purchase_data"
    else:
        result = query_fraud_data(query)
        tool = "query_fraud_data"
    
    tools = state.get("tools_used", []) + [tool]
    print(f"  [data_query] used {tool}, rows={result.get('row_count',0)}")
    return {"data_result": result, "tools_used": tools}

# ── Node: complex ───────────────────────────────────
def node_complex(state: CopilotState) -> dict:
    """Handle complex queries by chaining data retrieval + fraud analysis."""
    query = state["user_query"]
    tools = state.get("tools_used", [])
    
    # Step 1: Always start with data retrieval
    data_result = query_fraud_data(query)
    tools.append("query_fraud_data")
    print(f"  [complex] Step 1 - data: {data_result.get('row_count', 0)} rows")
    
    # Step 2: Run fraud model on top suspicious/relevant transactions
    prediction_result = None
    try:
        # Get top risky transactions from the data for prediction
        q = query.lower()
        df = fraud_df.copy()
        
        # Build a suspicion score for ranking
        df["_risk_composite"] = (
            df["customer_risk_score"] * 0.3 +
            df["merchant_risk_score"] * 0.2 +
            (1 - df["ip_reputation_score"]) * 0.2 +
            df["is_international"] * 0.15 +
            df["failed_transactions_24h"] / max(df["failed_transactions_24h"].max(), 1) * 0.15
        )
        
        # Select top 5 riskiest for prediction
        top_rows = df.nlargest(5, "_risk_composite")
        
        predictions = []
        for idx, row in top_rows.iterrows():
            features = {f: row[f] for f in fraud_features}
            pred = run_fraud_model(features)
            predictions.append({
                "transaction_id": row.get("transaction_id", "N/A"),
                "customer_id": row.get("customer_id", "N/A"),
                "amount": row.get("transaction_amount", 0),
                "probability": pred["probability"],
                "risk_tier": pred["risk_tier"],
                "top_features": pred["top_features"][:3],
                "suggested_action": pred["suggested_action"],
            })
        
        tools.append("run_fraud_model")
        
        prediction_result = {
            "analysis_type": "multi_transaction_risk",
            "transactions_analyzed": len(predictions),
            "results": predictions,
            "high_risk_count": sum(1 for p in predictions if p["risk_tier"] == "HIGH"),
            "medium_risk_count": sum(1 for p in predictions if p["risk_tier"] == "MEDIUM"),
            "low_risk_count": sum(1 for p in predictions if p["risk_tier"] == "LOW"),
        }
        print(f"  [complex] Step 2 - predictions: {len(predictions)} transactions analyzed, "
              f"{prediction_result['high_risk_count']} HIGH, "
              f"{prediction_result['medium_risk_count']} MEDIUM, "
              f"{prediction_result['low_risk_count']} LOW")
    except Exception as e:
        print(f"  [complex] Step 2 - prediction skipped: {e}")
    
    return {
        "data_result": data_result,
        "prediction_result": prediction_result,
        "tools_used": tools,
    }



# ── Node: Predict Fraud ────────────────────────────────
def node_predict_fraud(state: CopilotState) -> dict:
    query = state["user_query"]
    tools = state.get("tools_used", [])
    
    features = extract_fraud_features(query)
    result = run_fraud_model(features)
    tools.append("run_fraud_model")
    
    print(f"  [predict_fraud] P(fraud)={result.get('probability',0):.3f}, tier={result.get('risk_tier','?')}")
    return {"prediction_result": result, "tools_used": tools}


# ── Node: Predict Purchase ─────────────────────────────
def node_predict_purchase(state: CopilotState) -> dict:
    query = state["user_query"]
    tools = state.get("tools_used", [])
    
    features = extract_purchase_features(query)
    result = run_purchase_model(features)
    tools.append("run_purchase_model")
    
    print(f"  [predict_purchase] predicted=${result.get('predicted_amount',0):.2f}")
    return {"prediction_result": result, "tools_used": tools}


# ── Node: Knowledge Search ─────────────────────────────
def node_search_knowledge(state: CopilotState) -> dict:
    result = search_financial_docs(state["user_query"])
    tools = state.get("tools_used", []) + ["search_financial_docs"]
    
    print(f"  [knowledge] {result['num_results']} chunks found, {len(result['sources'])} sources")
    return {"knowledge_result": result, "sources_cited": result["sources"], "tools_used": tools}


# ── Node: Assess Analyst Load ──────────────────────────
def node_assess_load(state: CopilotState) -> dict:
    # In production: read from analyst session table
    # For prototype: use reasonable defaults
    metrics = {
        "queries_last_hour": 5,
        "avg_routing_tier": 2.0,
        "session_duration_hours": 2.0,
        "avg_query_interval_sec": 120,
        "followup_rate": 0.1,
        "hour_of_day": 14,
    }
    assessment = assess_analyst_load(metrics)
    print(f"  [load] score={assessment['load_score']}, level={assessment['level']}")
    return {"load_assessment": assessment}


# ── Node: Synthesize Response ──────────────────────────
def node_synthesize(state: CopilotState) -> dict:
    intent = state.get("intent_type", "data_query")
    load_level = state.get("load_assessment", {}).get("level", "normal")
    
    # Build context from tool results
    parts = []
    if state.get("data_result"):
        parts.append(f"DATA RESULT:\n{state['data_result'].get('result','')}")
    if state.get("prediction_result"):
        p = state["prediction_result"]
        parts.append(f"PREDICTION RESULT:\n{json.dumps(p, indent=2, default=str)}")
    if state.get("knowledge_result"):
        parts.append(f"KNOWLEDGE BASE:\n{state['knowledge_result'].get('answer_context','')}")
    
    context = "\n\n".join(parts)
    
    # Adapt format based on cognitive load
    format_map = {
        "normal": "Provide a detailed, well-structured response.",
        "elevated": "Start with a brief summary, then provide details in bullet points.",
        "high": "Keep your response concise. Use short sentences. Skip secondary details.",
        "critical": "Give only the essential answer in 2-3 sentences."
    }
    format_inst = format_map.get(load_level, format_map["normal"])
    
    citation_inst = ""
    if intent == "knowledge":
        citation_inst = "\nCRITICAL: Cite the source documents for every claim using [Source: document_name] format."
    
    synthesis_prompt = f"""You are a Fraud Copilot assisting a financial fraud analyst.
Based on the tool results below, compose a helpful, professional response.

ANALYST QUESTION: {state['user_query']}

{context}

FORMAT: {format_inst}{citation_inst}

If prediction results include SHAP features, explain the top contributing factors in business terms the analyst can act on.
If suggesting actions, be specific and operational."""

    response = llm.invoke(synthesis_prompt)
    final = response.content
    
    # Append load suggestion if needed
    load_adapt = state.get("load_assessment", {})
    if load_adapt.get("level") in ("high", "critical"):
        final += "\n\n---\n_System note: Elevated workload detected. Consider taking a short break or redistributing complex cases._"
    
    return {
        "final_response": final,
        "sources_cited": state.get("sources_cited", []),
        "tools_used": state.get("tools_used", [])
    }


# ── Feature Extraction Helpers ─────────────────────────
def extract_fraud_features(query: str) -> dict:
    """Extract transaction features from natural language for fraud prediction."""
    q = query.lower()
    features = {}
    
    # Amount
    m = re.search(r'\$?([\d,]+(?:\.\d{2})?)', query)
    if m: features["transaction_amount"] = float(m.group(1).replace(",",""))
    
    if "international" in q: features["is_international"] = 1
    if "domestic" in q: features["is_international"] = 0
    
    cats = {"electronics":"electronics","jewelry":"jewelry","grocery":"grocery","restaurant":"restaurant","gaming":"gaming","luxury":"luxury_goods","travel":"travel","atm":"ATM"}
    for k,v in cats.items():
        if k in q: features["merchant_category"] = v; break
    
    tm = re.search(r'(\d{1,2})\s*(am|pm)', q, re.IGNORECASE)
    if tm:
        h = int(tm.group(1))
        if tm.group(2).lower()=="pm" and h!=12: h+=12
        elif tm.group(2).lower()=="am" and h==12: h=0
        features["hour_of_day"] = h
    
    am = re.search(r'(\d+)[- ]month[- ]old', q)
    if am: features["account_age_days"] = int(am.group(1))*30
    
    # Customer lookup
    cm = re.search(r'CUST\d+', query, re.IGNORECASE)
    if cm:
        cid = cm.group().upper()
        row = fraud_df[fraud_df["customer_id"]==cid]
        if len(row)>0:
            row = row.iloc[0]
            return {f: row[f] for f in fraud_features if f in row.index}
    
    # Defaults for missing features
    defaults = dict(zip(fraud_features,[
        500.0,35,365,2,300.0,2,10,0,0.7,0.3,5000.0,10000,0.3,0,0.3,10.0,5.0,14,
        0,1,1,1,1,0,
        "online", "retail", "mobile", "USA"  # <--- Reemplazados los ints por STRINGS
    ]))
    for k,v in defaults.items():
        if k not in features: features[k] = v
    
    return features


def extract_purchase_features(query: str) -> dict:
    """Extract customer features for purchase prediction."""
    q = query.lower()
    features = {}
    
    am = re.search(r'(\d{2})[- ]year[- ]old', q)
    if am: features["age"] = int(am.group(1))
    
    for tier in ["platinum","gold","silver","bronze"]:
        if tier in q: features["membership_tier"] = tier; break
    
    tm = re.search(r'(\d+)\s*transactions?\s*(?:last|per)\s*month', q)
    if tm: features["num_transactions_last_month"] = int(tm.group(1))
    
    # Need encoded values for the model
    row = purchase_df.iloc[0]
    defaults = {f: row[f] for f in purchase_features if f in row.index}
    for f in purchase_features:
        if f not in defaults: defaults[f] = 0
    
    defaults.update(features)
    return defaults


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Compose the Graph

# COMMAND ----------

# ── Routing Function ───────────────────────────────────
def route_by_intent(state: CopilotState) -> str:
    intent = state.get("intent_type", "data_query")
    routing = {
        "data_query": "data_query",
        "prediction_fraud": "predict_fraud",
        "prediction_purchase": "predict_purchase",
        "knowledge": "search_knowledge",
        "complex": "complex_analysis",
    }
    return routing.get(intent, "data_query")


# ── Build Graph ────────────────────────────────────────
graph = StateGraph(CopilotState)

graph.add_node("classify", classify_intent)
graph.add_node("data_query", node_data_query)
graph.add_node("predict_fraud", node_predict_fraud)
graph.add_node("predict_purchase", node_predict_purchase)
graph.add_node("search_knowledge", node_search_knowledge)
graph.add_node("complex_analysis", node_complex)  
graph.add_node("assess_load", node_assess_load)
graph.add_node("synthesize", node_synthesize)

graph.set_entry_point("classify")

graph.add_conditional_edges("classify", route_by_intent, {
    "data_query": "data_query",
    "predict_fraud": "predict_fraud",
    "predict_purchase": "predict_purchase",
    "search_knowledge": "search_knowledge",
    "complex_analysis": "complex_analysis",
})

graph.add_edge("data_query", "assess_load")
graph.add_edge("predict_fraud", "assess_load")
graph.add_edge("predict_purchase", "assess_load")
graph.add_edge("search_knowledge", "assess_load")
graph.add_edge("complex_analysis", "assess_load") 
graph.add_edge("assess_load", "synthesize")
graph.add_edge("synthesize", END)

# Compile
copilot = graph.compile()
print("✓ Fraud Copilot graph compiled successfully")

# COMMAND ----------

# Visualize the graph
try:
    print(copilot.get_graph().draw_mermaid())
except:
    print("Graph nodes:", [n for n in copilot.get_graph().nodes])
    print("Graph edges:", [(e.source, e.target) for e in copilot.get_graph().edges])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4 Test: One Query Per Intent Type

# COMMAND ----------

def ask_copilot(query: str) -> dict:
    """Invoke the Fraud Copilot agent with a query."""
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")
    
    result = copilot.invoke({
        "user_query": query,
        "intent_type": None, "intent_confidence": None,
        "data_result": None, "prediction_result": None,
        "knowledge_result": None, "load_assessment": None,
        "final_response": None, "tools_used": None, "sources_cited": None,
    })
    
    print(f"\n--- RESPONSE ---")
    print(result.get("final_response", "No response"))
    print(f"\n[Intent: {result.get('intent_type')} | Tools: {result.get('tools_used')} | Sources: {result.get('sources_cited',[])}]")
    return result

# COMMAND ----------

# Test 1: Data Query
r1 = ask_copilot("How many international transactions are in the dataset?")

# COMMAND ----------

# Test 2: Fraud Prediction
r2 = ask_copilot("Predict if this transaction is fraudulent: $2,500 purchase at electronics store, international, 3am, from a 2-month-old account")

# COMMAND ----------

# Test 3: Knowledge (RAG)
r3 = ask_copilot("What are the key components of PCI DSS compliance?")

# COMMAND ----------

# Test 4: Purchase Prediction
r4 = ask_copilot("What's the expected purchase amount for a 45-year-old customer with Platinum membership who made 20 transactions last month?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.5 Summary
# MAGIC
# MAGIC The LangGraph agent correctly routes queries through the pipeline:
# MAGIC ```
# MAGIC classify → [tool node] → assess_load → synthesize → END
# MAGIC ```
# MAGIC
# MAGIC All traces are automatically captured by `mlflow.langchain.autolog()`.
# MAGIC Check MLflow UI → Traces tab to see the execution graph for each query.
# MAGIC
# MAGIC Proceed to notebook 05 for the full interactive demo.