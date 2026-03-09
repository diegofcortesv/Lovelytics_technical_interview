"""
Golden set: 30 evaluation queries with expected intents, tools, and keywords.

Distribution:
    - Data analysis:       10 queries (33%)
    - Fraud prediction:     6 queries (20%)
    - Purchase prediction:  4 queries (13%)
    - Knowledge (RAG):      5 queries (17%)
    - Complex:              5 queries (17%)

Usage:
    from src.evaluation.golden_set import GOLDEN_SET, run_evaluation
    results = run_evaluation(copilot, GOLDEN_SET)
"""

import time
from typing import Any


GOLDEN_SET = [
    # --- Data Analysis (10) ---
    {
        "query": "How many international transactions are in the dataset?",
        "expected_intent": "data_query",
        "expected_tools": ["query_fraud_data"],
        "check_contains": ["49", "international"],
    },
    {
        "query": "What's the average transaction amount for fraudulent vs legitimate transactions?",
        "expected_intent": "data_query",
        "expected_tools": ["query_fraud_data"],
        "check_contains": ["average", "fraud"],
    },
    {
        "query": "Which customers have the highest risk scores?",
        "expected_intent": "data_query",
        "expected_tools": ["query_fraud_data"],
        "check_contains": ["risk"],
    },
    {
        "query": "How many transactions had more than 3 failed attempts in 24 hours?",
        "expected_intent": "data_query",
        "expected_tools": ["query_fraud_data"],
        "check_contains": ["failed"],
    },
    {
        "query": "Compare purchase patterns between Gold and Silver membership tiers",
        "expected_intent": "data_query",
        "expected_tools": ["query_purchase_data"],
        "check_contains": ["gold", "silver"],
    },
    {
        "query": "What is the total number of transactions in the fraud dataset?",
        "expected_intent": "data_query",
        "expected_tools": ["query_fraud_data"],
        "check_contains": ["100"],
    },
    {
        "query": "How many unique customers are in the fraud dataset?",
        "expected_intent": "data_query",
        "expected_tools": ["query_fraud_data"],
        "check_contains": ["customer"],
    },
    {
        "query": "Show me the distribution of device types used in transactions",
        "expected_intent": "data_query",
        "expected_tools": ["query_fraud_data"],
        "check_contains": ["device"],
    },
    {
        "query": "What percentage of transactions are flagged as fraud?",
        "expected_intent": "data_query",
        "expected_tools": ["query_fraud_data"],
        "check_contains": ["50", "fraud"],
    },
    {
        "query": "What are the average loyalty points by membership tier?",
        "expected_intent": "data_query",
        "expected_tools": ["query_purchase_data"],
        "check_contains": ["loyalty"],
    },

    # --- Fraud Prediction (6) ---
    {
        "query": "Predict if this transaction is fraudulent: $1,250 purchase at electronics store, international, 3am, from a 2-month-old account",
        "expected_intent": "prediction_fraud",
        "expected_tools": ["run_fraud_model"],
        "check_contains": ["risk", "probability"],
    },
    {
        "query": "Is this transaction suspicious? $5,000 jewelry purchase, international, new device",
        "expected_intent": "prediction_fraud",
        "expected_tools": ["run_fraud_model"],
        "check_contains": ["risk"],
    },
    {
        "query": "Assess the fraud risk for a $200 grocery transaction, domestic, mobile device",
        "expected_intent": "prediction_fraud",
        "expected_tools": ["run_fraud_model"],
        "check_contains": ["risk"],
    },
    {
        "query": "Analyze customer CUST7823's transaction history and assess their fraud risk",
        "expected_intent": "prediction_fraud",
        "expected_tools": ["run_fraud_model"],
        "check_contains": ["CUST7823", "risk"],
    },
    {
        "query": "What is the fraud probability for a $3,000 ATM withdrawal at 2am international?",
        "expected_intent": "prediction_fraud",
        "expected_tools": ["run_fraud_model"],
        "check_contains": ["probability"],
    },
    {
        "query": "Predict fraud risk: $800 online_retail purchase, 5 failed attempts, low IP reputation",
        "expected_intent": "prediction_fraud",
        "expected_tools": ["run_fraud_model"],
        "check_contains": ["risk"],
    },

    # --- Purchase Prediction (4) ---
    {
        "query": "What's the expected purchase amount for a 45-year-old Platinum member with 20 transactions last month?",
        "expected_intent": "prediction_purchase",
        "expected_tools": ["run_purchase_model"],
        "check_contains": ["predicted", "amount"],
    },
    {
        "query": "Predict the purchase amount for a 28-year-old Silver member with 5 transactions last month",
        "expected_intent": "prediction_purchase",
        "expected_tools": ["run_purchase_model"],
        "check_contains": ["predicted"],
    },
    {
        "query": "How much would a Gold tier customer aged 55 with high loyalty points likely spend?",
        "expected_intent": "prediction_purchase",
        "expected_tools": ["run_purchase_model"],
        "check_contains": ["predicted"],
    },
    {
        "query": "Estimate the next purchase for a Bronze member who hasn't bought anything in 30 days",
        "expected_intent": "prediction_purchase",
        "expected_tools": ["run_purchase_model"],
        "check_contains": ["predicted"],
    },

    # --- Knowledge / RAG (5) ---
    {
        "query": "What are the common indicators of credit card fraud?",
        "expected_intent": "knowledge",
        "expected_tools": ["search_financial_docs"],
        "check_contains": ["indicator", "fraud"],
    },
    {
        "query": "Explain the KYC procedures for high-risk customers",
        "expected_intent": "knowledge",
        "expected_tools": ["search_financial_docs"],
        "check_contains": ["KYC", "due diligence"],
    },
    {
        "query": "What are the key components of PCI DSS compliance?",
        "expected_intent": "knowledge",
        "expected_tools": ["search_financial_docs"],
        "check_contains": ["PCI", "requirement"],
    },
    {
        "query": "What are the three stages of money laundering?",
        "expected_intent": "knowledge",
        "expected_tools": ["search_financial_docs"],
        "check_contains": ["placement", "layering"],
    },
    {
        "query": "Summarize transaction monitoring system best practices",
        "expected_intent": "knowledge",
        "expected_tools": ["search_financial_docs"],
        "check_contains": ["monitoring"],
    },

    # --- Complex (5) ---
    {
        "query": "Which transactions look suspicious and why? Show me the top 5 with explanations",
        "expected_intent": "complex",
        "expected_tools": ["query_fraud_data"],
        "check_contains": ["suspicious"],
    },
    {
        "query": "What fraud patterns exist in international transactions vs domestic ones?",
        "expected_intent": "complex",
        "expected_tools": ["query_fraud_data"],
        "check_contains": ["international"],
    },
    {
        "query": "For the highest-risk customers, what is their average transaction behavior?",
        "expected_intent": "complex",
        "expected_tools": ["query_fraud_data"],
        "check_contains": ["risk"],
    },
    {
        "query": "Give me a risk assessment summary: how many high, medium, and low risk transactions?",
        "expected_intent": "complex",
        "expected_tools": ["query_fraud_data"],
        "check_contains": ["risk"],
    },
    {
        "query": "Identify anomalous transactions that deviate from customer spending patterns",
        "expected_intent": "complex",
        "expected_tools": ["query_fraud_data"],
        "check_contains": ["anomal"],
    },
]


def evaluate_single(
    copilot: Any,
    item: dict,
    invoke_fn=None,
) -> dict:
    """
    Evaluate a single golden set item against the copilot.

    Returns a dict with correctness metrics.
    """
    start = time.time()

    if invoke_fn:
        output = invoke_fn(copilot, item["query"])
    else:
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

    # Scoring
    intent_correct = actual_intent == item["expected_intent"]
    tool_correct = any(t in actual_tools for t in item["expected_tools"])

    resp_lower = response.lower()
    keywords = item.get("check_contains", [])
    content_hits = sum(1 for kw in keywords if kw.lower() in resp_lower)
    content_score = content_hits / len(keywords) if keywords else 1.0

    return {
        "query": item["query"],
        "expected_intent": item["expected_intent"],
        "actual_intent": actual_intent,
        "intent_correct": intent_correct,
        "tool_correct": tool_correct,
        "content_score": content_score,
        "latency_s": latency,
        "response_length": len(response),
        "has_response": len(response) > 50,
    }


def run_evaluation(
    copilot: Any,
    golden_set: list[dict] | None = None,
    invoke_fn=None,
) -> list[dict]:
    """
    Run the full golden set evaluation.

    Returns a list of result dicts, one per query.
    """
    if golden_set is None:
        golden_set = GOLDEN_SET

    results = []
    for i, item in enumerate(golden_set):
        try:
            result = evaluate_single(copilot, item, invoke_fn)
            results.append(result)
            status = "OK" if result["intent_correct"] and result["tool_correct"] else "MISMATCH"
            print(
                f"[{i+1}/{len(golden_set)}] {status} | "
                f"intent: {result['actual_intent']} "
                f"(expected: {item['expected_intent']}) | "
                f"{result['latency_s']:.1f}s"
            )
        except Exception as e:
            results.append({
                "query": item["query"],
                "expected_intent": item["expected_intent"],
                "actual_intent": "ERROR",
                "intent_correct": False,
                "tool_correct": False,
                "content_score": 0.0,
                "latency_s": 0.0,
                "response_length": 0,
                "has_response": False,
                "error": str(e),
            })
            print(f"[{i+1}/{len(golden_set)}] ERROR | {str(e)[:80]}")

    return results


def summarize_results(results: list[dict]) -> dict:
    """Compute aggregate metrics from evaluation results."""
    n = len(results)
    if n == 0:
        return {}

    intent_acc = sum(r["intent_correct"] for r in results) / n
    tool_acc = sum(r["tool_correct"] for r in results) / n
    content_avg = sum(r["content_score"] for r in results) / n
    response_rate = sum(r["has_response"] for r in results) / n
    latencies = [r["latency_s"] for r in results if r["latency_s"] > 0]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    return {
        "n_queries": n,
        "n_errors": sum(1 for r in results if "error" in r),
        "intent_accuracy": round(intent_acc, 4),
        "tool_routing_accuracy": round(tool_acc, 4),
        "content_relevance": round(content_avg, 4),
        "response_rate": round(response_rate, 4),
        "avg_latency_s": round(avg_latency, 2),
        "p95_latency_s": round(p95_latency, 2),
    }
