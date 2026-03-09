"""
LangGraph node implementations for the Fraud Copilot agent.

Each function is a graph node that receives CopilotState and returns
a partial state update. Nodes are composed into a StateGraph in graph.py.
"""

import json
import re
from typing import TypedDict, Optional

from ..tools.data_tools import query_fraud_data, query_purchase_data
from ..tools.model_tools import (
    run_fraud_model,
    run_purchase_model,
    extract_fraud_features,
    extract_purchase_features,
)
from ..tools.rag_tools import search_financial_docs
from .policies import assess_analyst_load, get_format_instruction


# --- State Definition ---

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


# --- Node: Intent Classification ---

CLASSIFY_PROMPT = """You are a query classifier for a financial fraud analysis system.
Classify this query into exactly one category.

IMPORTANT: Read ALL categories before deciding. Choose the MOST SPECIFIC match.

Categories (in priority order):
1. complex: The query requires MULTIPLE analytical steps. Signals:
   - Asks to FIND items AND EXPLAIN why (e.g., "which are suspicious and why")
   - Asks to COMPARE groups AND ANALYZE patterns (e.g., "patterns in X vs Y")
   - Asks for a SUMMARY that requires AGGREGATION + CLASSIFICATION (e.g., "how many high/medium/low risk")
   - Asks to IDENTIFY + ASSESS (e.g., "identify anomalous transactions", "analyze and assess risk")

2. prediction_fraud: Predict fraud for ONE SPECIFIC transaction described with concrete details.
   - MUST include specific transaction parameters (amount, merchant, time, etc.)

3. prediction_purchase: Predict purchase amount for ONE SPECIFIC customer described with concrete details.
   - MUST include specific customer parameters (age, tier, transactions, etc.)

4. data_query: Simple data retrieval - counts, averages, filters, lists, comparisons.
   - Answers come directly from querying/filtering the dataset

5. knowledge: Questions about financial regulations, procedures, compliance, fraud theory.
   - About KYC, AML, PCI DSS, fraud indicators, regulations

DECISION RULES:
- If the query asks to FIND + EXPLAIN or ANALYZE + ASSESS -> complex
- If the query describes ONE specific transaction/customer to predict -> prediction_*
- If the query asks for simple numbers or lists -> data_query
- If the query is about regulations/procedures/theory -> knowledge

Query: "{query}"

Respond ONLY with JSON: {{"intent": "<category>", "confidence": <0.0-1.0>}}"""


def classify_intent(state: CopilotState, llm) -> dict:
    """Classify the analyst's query intent using the LLM."""
    query = state["user_query"]
    prompt = CLASSIFY_PROMPT.format(query=query)

    response = llm.invoke(prompt)
    content = response.content

    try:
        j_start = content.find("{")
        j_end = content.rfind("}") + 1
        parsed = json.loads(content[j_start:j_end])
    except (json.JSONDecodeError, ValueError):
        parsed = {"intent": "data_query", "confidence": 0.5}

    intent = parsed.get("intent", "data_query")
    confidence = parsed.get("confidence", 0.5)

    return {
        "intent_type": intent,
        "intent_confidence": confidence,
        "tools_used": [],
    }


# --- Node: Data Query ---

def node_data_query(state: CopilotState) -> dict:
    """Route data queries to the appropriate dataset tool."""
    query = state["user_query"]
    q = query.lower()

    if any(w in q for w in ["purchase", "tier", "membership", "spending pattern", "loyalty"]):
        result = query_purchase_data(query)
        tool = "query_purchase_data"
    else:
        result = query_fraud_data(query)
        tool = "query_fraud_data"

    tools = (state.get("tools_used") or []) + [tool]
    return {"data_result": result, "tools_used": tools}


# --- Node: Complex Analysis ---

def node_complex(
    state: CopilotState,
    fraud_df,
    fraud_features: list[str],
) -> dict:
    """Handle complex queries by chaining data retrieval + model predictions."""
    query = state["user_query"]
    tools = list(state.get("tools_used") or [])

    # Step 1: Data retrieval
    data_result = query_fraud_data(query)
    tools.append("query_fraud_data")

    # Step 2: Model predictions on top-risk transactions
    prediction_result = None
    try:
        df = fraud_df.copy()
        max_failed = max(df["failed_transactions_24h"].max(), 1)
        df["_risk_composite"] = (
            df["customer_risk_score"] * 0.3
            + df["merchant_risk_score"] * 0.2
            + (1 - df["ip_reputation_score"]) * 0.2
            + df["is_international"] * 0.15
            + df["failed_transactions_24h"] / max_failed * 0.15
        )

        top_rows = df.nlargest(5, "_risk_composite")
        predictions = []
        for _, row in top_rows.iterrows():
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
    except Exception:
        pass

    return {
        "data_result": data_result,
        "prediction_result": prediction_result,
        "tools_used": tools,
    }


# --- Node: Predict Fraud ---

def node_predict_fraud(
    state: CopilotState,
    fraud_df,
    fraud_features: list[str],
) -> dict:
    """Run fraud model on extracted transaction features."""
    query = state["user_query"]
    tools = list(state.get("tools_used") or [])

    features = extract_fraud_features(query, fraud_features, fraud_df)
    result = run_fraud_model(features)
    tools.append("run_fraud_model")

    return {"prediction_result": result, "tools_used": tools}


# --- Node: Predict Purchase ---

def node_predict_purchase(
    state: CopilotState,
    purchase_df,
    purchase_features: list[str],
) -> dict:
    """Run purchase model on extracted customer features."""
    query = state["user_query"]
    tools = list(state.get("tools_used") or [])

    features = extract_purchase_features(query, purchase_features, purchase_df)
    result = run_purchase_model(features)
    tools.append("run_purchase_model")

    return {"prediction_result": result, "tools_used": tools}


# --- Node: Knowledge Search ---

def node_search_knowledge(state: CopilotState) -> dict:
    """Retrieve relevant financial documents via RAG."""
    result = search_financial_docs(state["user_query"])
    tools = (state.get("tools_used") or []) + ["search_financial_docs"]

    return {
        "knowledge_result": result,
        "sources_cited": result["sources"],
        "tools_used": tools,
    }


# --- Node: Assess Analyst Load ---

def node_assess_load(state: CopilotState) -> dict:
    """
    Calculate cognitive load from session metrics.

    In production: read from analyst session Delta table.
    For prototype: use reasonable defaults.
    """
    metrics = {
        "queries_last_hour": 5,
        "avg_routing_tier": 2.0,
        "session_duration_hours": 2.0,
        "avg_query_interval_sec": 120,
        "followup_rate": 0.1,
        "hour_of_day": 14,
    }
    assessment = assess_analyst_load(metrics)
    return {"load_assessment": assessment}


# --- Node: Synthesize Response ---

SYNTHESIS_PROMPT = """You are a Fraud Copilot assisting a financial fraud analyst.
Based on the tool results below, compose a helpful, professional response.

ANALYST QUESTION: {query}

{context}

FORMAT: {format_instruction}{citation_instruction}

If prediction results include SHAP features, explain the top contributing factors in business terms the analyst can act on.
If suggesting actions, be specific and operational."""


def node_synthesize(state: CopilotState, llm) -> dict:
    """Compose the final response using LLM synthesis."""
    intent = state.get("intent_type", "data_query")
    load_level = (state.get("load_assessment") or {}).get("level", "normal")

    # Build context from tool results
    parts = []
    if state.get("data_result"):
        parts.append(f"DATA RESULT:\n{state['data_result'].get('result', '')}")
    if state.get("prediction_result"):
        p = state["prediction_result"]
        parts.append(f"PREDICTION RESULT:\n{json.dumps(p, indent=2, default=str)}")
    if state.get("knowledge_result"):
        parts.append(f"KNOWLEDGE BASE:\n{state['knowledge_result'].get('answer_context', '')}")

    context = "\n\n".join(parts)
    format_instruction = get_format_instruction(load_level)

    citation_instruction = ""
    if intent == "knowledge":
        citation_instruction = (
            "\nCRITICAL: Cite the source documents for every claim "
            "using [Source: document_name] format."
        )

    prompt = SYNTHESIS_PROMPT.format(
        query=state["user_query"],
        context=context,
        format_instruction=format_instruction,
        citation_instruction=citation_instruction,
    )

    response = llm.invoke(prompt)
    final = response.content

    # Append load advisory if needed
    if load_level in ("high", "critical"):
        final += (
            "\n\n---\n_System note: Elevated workload detected. "
            "Consider taking a short break or redistributing complex cases._"
        )

    return {
        "final_response": final,
        "sources_cited": state.get("sources_cited", []),
        "tools_used": state.get("tools_used", []),
    }
