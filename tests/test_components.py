"""
Unit tests for Fraud Copilot components.

These tests validate tool logic, policy functions, feature extraction,
and evaluation utilities WITHOUT requiring Databricks infrastructure.
All tests use synthetic data and mock objects.

Run: pytest tests/test_components.py -v
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


# ============================================================
# Fixtures: Synthetic data for testing
# ============================================================

@pytest.fixture
def fraud_df():
    """Minimal fraud dataset for testing data tools."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "customer_id": [f"CUST{1000+i}" for i in range(n)],
        "transaction_id": [f"TXN{2000+i}" for i in range(n)],
        "transaction_amount": np.random.uniform(50, 5000, n).round(2),
        "fraud": [0]*10 + [1]*10,
        "is_international": [0]*12 + [1]*8,
        "customer_risk_score": np.random.uniform(0, 1, n).round(3),
        "merchant_risk_score": np.random.uniform(0, 1, n).round(3),
        "ip_reputation_score": np.random.uniform(0, 1, n).round(3),
        "failed_transactions_24h": np.random.randint(0, 6, n),
        "merchant_category": np.random.choice(["electronics", "grocery", "travel"], n),
        "device_type": np.random.choice(["mobile", "desktop", "tablet"], n),
        "country": np.random.choice(["USA", "UK", "BR"], n),
        "transaction_type": np.random.choice(["online", "in_store"], n),
        "customer_age": np.random.randint(20, 70, n),
        "account_age_days": np.random.randint(30, 3000, n),
        "transaction_velocity_24h": np.random.randint(1, 10, n),
        "avg_transaction_amount_30d": np.random.uniform(100, 2000, n).round(2),
        "num_transactions_24h": np.random.randint(1, 8, n),
        "num_transactions_7d": np.random.randint(1, 30, n),
        "account_balance": np.random.uniform(1000, 50000, n).round(2),
        "credit_limit": np.random.randint(5000, 50000, n),
        "debt_to_income_ratio": np.random.uniform(0.1, 0.8, n).round(2),
        "previous_fraud_reports": np.random.randint(0, 3, n),
        "distance_from_home_km": np.random.uniform(0, 500, n).round(1),
        "distance_from_last_transaction_km": np.random.uniform(0, 200, n).round(1),
        "hour_of_day": np.random.randint(0, 24, n),
        "shipping_address_match": np.random.randint(0, 2, n),
        "billing_address_match": np.random.randint(0, 2, n),
        "cvv_match": np.random.randint(0, 2, n),
        "card_present": np.random.randint(0, 2, n),
        "is_recurring": np.random.randint(0, 2, n),
    })


@pytest.fixture
def purchase_df():
    """Minimal purchase dataset for testing data tools."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "customer_id": [f"PCUST{3000+i}" for i in range(n)],
        "purchase_amount": np.random.uniform(50, 600, n).round(2),
        "membership_tier": np.random.choice(["gold", "silver", "platinum", "bronze"], n),
        "loyalty_points": np.random.randint(100, 5000, n),
        "age": np.random.randint(20, 65, n),
        "gender": np.random.choice(["male", "female"], n),
        "income_bracket": np.random.choice(["low", "medium", "high"], n),
        "preferred_category": np.random.choice(["electronics", "food", "clothing"], n),
        "preferred_payment_method": np.random.choice(["credit", "debit", "cash"], n),
        "location_type": np.random.choice(["urban", "suburban", "rural"], n),
        "occupation_category": np.random.choice(["tech", "finance", "health"], n),
        "education_level": np.random.choice(["bachelor", "master", "phd"], n),
        "marital_status": np.random.choice(["single", "married"], n),
        "customer_tenure_days": np.random.randint(30, 3000, n),
        "num_transactions_last_month": np.random.randint(1, 30, n),
        "num_transactions_last_year": np.random.randint(10, 200, n),
        "avg_transaction_value": np.random.uniform(50, 500, n).round(2),
        "total_spent_last_year": np.random.uniform(1000, 20000, n).round(2),
        "customer_satisfaction_score": np.random.uniform(1, 5, n).round(1),
        "email_engagement_rate": np.random.uniform(0, 1, n).round(2),
        "cart_abandonment_rate": np.random.uniform(0, 1, n).round(2),
        "distance_to_nearest_store_km": np.random.uniform(0.5, 50, n).round(1),
        "num_customer_service_contacts": np.random.randint(0, 10, n),
        "product_return_rate": np.random.uniform(0, 0.3, n).round(2),
        "last_purchase_days_ago": np.random.randint(1, 90, n),
        "mobile_app_user": np.random.randint(0, 2, n),
        "social_media_follower": np.random.randint(0, 2, n),
        "has_credit_card": np.random.randint(0, 2, n),
        "has_children": np.random.randint(0, 2, n),
        "owns_home": np.random.randint(0, 2, n),
    })


# ============================================================
# Tests: Data Tools
# ============================================================

class TestFraudDataTool:
    """Tests for the FraudDataTool query engine."""

    def test_count_international(self, fraud_df):
        from src.tools.data_tools import FraudDataTool
        tool = FraudDataTool(fraud_df)
        result = tool.query("How many international transactions are in the dataset?")
        assert "international" in result["result"].lower()
        assert result["row_count"] == int(fraud_df["is_international"].sum())

    def test_avg_amount_by_fraud(self, fraud_df):
        from src.tools.data_tools import FraudDataTool
        tool = FraudDataTool(fraud_df)
        result = tool.query("What's the average transaction amount for fraudulent vs legitimate?")
        assert "Legitimate" in result["result"]
        assert "Fraudulent" in result["result"]

    def test_top_risk_scores(self, fraud_df):
        from src.tools.data_tools import FraudDataTool
        tool = FraudDataTool(fraud_df)
        result = tool.query("Which customers have the highest risk scores?")
        assert result["row_count"] == 10

    def test_failed_transactions(self, fraud_df):
        from src.tools.data_tools import FraudDataTool
        tool = FraudDataTool(fraud_df)
        result = tool.query("How many transactions had more than 3 failed attempts?")
        expected = len(fraud_df[fraud_df["failed_transactions_24h"] > 3])
        assert result["row_count"] == expected

    def test_suspicious_transactions(self, fraud_df):
        from src.tools.data_tools import FraudDataTool
        tool = FraudDataTool(fraud_df)
        result = tool.query("Show top 5 suspicious transactions")
        assert result["row_count"] == 5

    def test_customer_lookup_found(self, fraud_df):
        from src.tools.data_tools import FraudDataTool
        tool = FraudDataTool(fraud_df)
        cid = fraud_df.iloc[0]["customer_id"]
        result = tool.query(f"Show transactions for {cid}")
        assert result["row_count"] > 0

    def test_customer_lookup_not_found(self, fraud_df):
        from src.tools.data_tools import FraudDataTool
        tool = FraudDataTool(fraud_df)
        result = tool.query("Show transactions for CUST9999")
        assert result["row_count"] == 0

    def test_dataset_summary_fallback(self, fraud_df):
        from src.tools.data_tools import FraudDataTool
        tool = FraudDataTool(fraud_df)
        result = tool.query("Tell me something random")
        assert "transactions" in result["result"].lower()

    def test_total_count(self, fraud_df):
        from src.tools.data_tools import FraudDataTool
        tool = FraudDataTool(fraud_df)
        result = tool.query("What is the total number of transactions?")
        assert result["row_count"] == len(fraud_df)

    def test_unique_customers(self, fraud_df):
        from src.tools.data_tools import FraudDataTool
        tool = FraudDataTool(fraud_df)
        result = tool.query("How many unique customers?")
        assert result["row_count"] == fraud_df["customer_id"].nunique()

    def test_device_distribution(self, fraud_df):
        from src.tools.data_tools import FraudDataTool
        tool = FraudDataTool(fraud_df)
        result = tool.query("Show device type distribution")
        assert "device" in result["result"].lower() or "mobile" in result["result"].lower()

    def test_fraud_percentage(self, fraud_df):
        from src.tools.data_tools import FraudDataTool
        tool = FraudDataTool(fraud_df)
        result = tool.query("What percentage of transactions are fraud?")
        assert "fraud" in result["result"].lower()

    def test_missing_columns_raises(self):
        from src.tools.data_tools import FraudDataTool
        with pytest.raises(ValueError, match="missing columns"):
            FraudDataTool(pd.DataFrame({"a": [1]}))


class TestPurchaseDataTool:
    """Tests for the PurchaseDataTool query engine."""

    def test_tier_comparison(self, purchase_df):
        from src.tools.data_tools import PurchaseDataTool
        tool = PurchaseDataTool(purchase_df)
        result = tool.query("Compare purchase patterns between Gold and Silver tiers")
        assert "tier" in result["result"].lower() or "gold" in result["result"].lower()

    def test_loyalty_by_tier(self, purchase_df):
        from src.tools.data_tools import PurchaseDataTool
        tool = PurchaseDataTool(purchase_df)
        result = tool.query("What are the average loyalty points by tier?")
        assert "loyalty" in result["result"].lower()

    def test_summary_fallback(self, purchase_df):
        from src.tools.data_tools import PurchaseDataTool
        tool = PurchaseDataTool(purchase_df)
        result = tool.query("random query")
        assert "customers" in result["result"].lower()


# ============================================================
# Tests: Policies (Cognitive Load)
# ============================================================

class TestCognitiveLoad:
    """Tests for the cognitive load assessment heuristic."""

    def test_normal_load(self):
        from src.agent.policies import assess_analyst_load
        result = assess_analyst_load({
            "queries_last_hour": 3,
            "avg_routing_tier": 1.5,
            "session_duration_hours": 1,
            "avg_query_interval_sec": 200,
            "followup_rate": 0.05,
            "hour_of_day": 10,
        })
        assert result["level"] == "normal"
        assert result["load_score"] <= 30

    def test_high_load(self):
        from src.agent.policies import assess_analyst_load
        result = assess_analyst_load({
            "queries_last_hour": 14,
            "avg_routing_tier": 3.5,
            "session_duration_hours": 5.5,
            "avg_query_interval_sec": 25,
            "followup_rate": 0.35,
            "hour_of_day": 19,
        })
        assert result["level"] in ("high", "critical")
        assert result["load_score"] >= 60

    def test_empty_metrics_defaults(self):
        from src.agent.policies import assess_analyst_load
        result = assess_analyst_load({})
        assert 0 <= result["load_score"] <= 100
        assert result["level"] in ("normal", "elevated", "high", "critical")

    def test_score_bounds(self):
        from src.agent.policies import assess_analyst_load
        # Max everything
        result = assess_analyst_load({
            "queries_last_hour": 100,
            "avg_routing_tier": 10,
            "session_duration_hours": 24,
            "avg_query_interval_sec": 0,
            "followup_rate": 1.0,
            "hour_of_day": 22,
        })
        assert result["load_score"] <= 100

    def test_format_instructions(self):
        from src.agent.policies import get_format_instruction
        assert "detailed" in get_format_instruction("normal").lower()
        assert "concise" in get_format_instruction("high").lower()
        assert "essential" in get_format_instruction("critical").lower()


class TestGuardrails:
    """Tests for policy guardrail checks."""

    def test_confidence_validation(self):
        from src.agent.policies import validate_intent_confidence
        assert validate_intent_confidence(0.9) is True
        assert validate_intent_confidence(0.5) is False
        assert validate_intent_confidence(0.8) is True

    def test_citation_requirement(self):
        from src.agent.policies import requires_citation
        assert requires_citation("knowledge") is True
        assert requires_citation("data_query") is False

    def test_shap_requirement(self):
        from src.agent.policies import requires_shap
        assert requires_shap("prediction_fraud") is True
        assert requires_shap("prediction_purchase") is True
        assert requires_shap("complex") is True
        assert requires_shap("data_query") is False
        assert requires_shap("knowledge") is False


# ============================================================
# Tests: Feature Extraction
# ============================================================

class TestFeatureExtraction:
    """Tests for NL-to-features extraction utilities."""

    def test_extract_amount(self, fraud_df):
        from src.tools.model_tools import extract_fraud_features
        features = extract_fraud_features(
            "$2,500 electronics purchase international 3am",
            list(fraud_df.columns[:28]),
            fraud_df,
        )
        assert features.get("transaction_amount") == 2500.0

    def test_extract_international(self, fraud_df):
        from src.tools.model_tools import extract_fraud_features
        features = extract_fraud_features(
            "international transaction $100",
            list(fraud_df.columns[:28]),
            fraud_df,
        )
        assert features.get("is_international") == 1

    def test_extract_domestic(self, fraud_df):
        from src.tools.model_tools import extract_fraud_features
        features = extract_fraud_features(
            "domestic transaction $100",
            list(fraud_df.columns[:28]),
            fraud_df,
        )
        assert features.get("is_international") == 0

    def test_extract_merchant_category(self, fraud_df):
        from src.tools.model_tools import extract_fraud_features
        features = extract_fraud_features(
            "$500 jewelry purchase",
            list(fraud_df.columns[:28]),
            fraud_df,
        )
        assert features.get("merchant_category") == "jewelry"

    def test_extract_hour(self, fraud_df):
        from src.tools.model_tools import extract_fraud_features
        features = extract_fraud_features(
            "$100 purchase at 3am",
            list(fraud_df.columns[:28]),
            fraud_df,
        )
        assert features.get("hour_of_day") == 3

    def test_extract_account_age(self, fraud_df):
        from src.tools.model_tools import extract_fraud_features
        features = extract_fraud_features(
            "$100 from a 2-month-old account",
            list(fraud_df.columns[:28]),
            fraud_df,
        )
        assert features.get("account_age_days") == 60

    def test_extract_customer_lookup(self, fraud_df):
        from src.tools.model_tools import extract_fraud_features
        cid = fraud_df.iloc[0]["customer_id"]
        features = extract_fraud_features(
            f"Analyze {cid}'s transaction",
            list(fraud_df.columns[:28]),
            fraud_df,
        )
        # Should return actual row data, not defaults
        assert features.get("transaction_amount") == fraud_df.iloc[0]["transaction_amount"]

    def test_extract_purchase_age(self, purchase_df):
        from src.tools.model_tools import extract_purchase_features
        features = extract_purchase_features(
            "45-year-old Platinum member with 20 transactions last month",
            list(purchase_df.columns),
            purchase_df,
        )
        assert features.get("age") == 45
        assert features.get("membership_tier") == "platinum"
        assert features.get("num_transactions_last_month") == 20


# ============================================================
# Tests: RAG Tool (TF-IDF path)
# ============================================================

class TestRAGTool:
    """Tests for the TF-IDF fallback RAG retrieval."""

    @pytest.fixture
    def chunks_df(self):
        return pd.DataFrame({
            "chunk_id": [0, 1, 2, 3],
            "content": [
                "PCI DSS requires encryption of cardholder data and regular security assessments.",
                "KYC procedures include identity verification and enhanced due diligence for high-risk customers.",
                "Anti-money laundering involves three stages: placement, layering, and integration.",
                "Credit card fraud indicators include unusual transaction patterns and velocity spikes.",
            ],
            "source": [
                "pci_dss_compliance > Requirements",
                "kyc_procedures > Due Diligence",
                "aml_guide > Stages",
                "fraud_indicators > Patterns",
            ],
            "doc_name": ["pci_dss_compliance", "kyc_procedures", "aml_guide", "fraud_indicators"],
            "section": ["Requirements", "Due Diligence", "Stages", "Patterns"],
            "char_count": [80, 90, 85, 82],
        })

    def test_tfidf_search_relevance(self, chunks_df):
        from src.tools.rag_tools import RAGTool
        tool = RAGTool(chunks_df)
        result = tool.search("PCI DSS compliance requirements")
        assert result["num_results"] > 0
        assert result["backend"] == "tfidf"
        assert any("pci" in s.lower() for s in result["sources"])

    def test_tfidf_search_kyc(self, chunks_df):
        from src.tools.rag_tools import RAGTool
        tool = RAGTool(chunks_df)
        result = tool.search("KYC due diligence procedures")
        assert result["num_results"] > 0
        assert any("kyc" in s.lower() for s in result["sources"])

    def test_tfidf_returns_context(self, chunks_df):
        from src.tools.rag_tools import RAGTool
        tool = RAGTool(chunks_df)
        result = tool.search("money laundering stages")
        assert "answer_context" in result
        assert len(result["answer_context"]) > 0

    def test_tfidf_sources_are_set(self, chunks_df):
        from src.tools.rag_tools import RAGTool
        tool = RAGTool(chunks_df)
        result = tool.search("fraud indicators")
        # Sources should be unique
        assert len(result["sources"]) == len(set(result["sources"]))


# ============================================================
# Tests: Evaluation Utilities
# ============================================================

class TestEvaluation:
    """Tests for the golden set evaluation framework."""

    def test_golden_set_size(self):
        from src.evaluation.golden_set import GOLDEN_SET
        assert len(GOLDEN_SET) == 30

    def test_golden_set_distribution(self):
        from src.evaluation.golden_set import GOLDEN_SET
        intents = [g["expected_intent"] for g in GOLDEN_SET]
        assert intents.count("data_query") == 10
        assert intents.count("prediction_fraud") == 6
        assert intents.count("prediction_purchase") == 4
        assert intents.count("knowledge") == 5
        assert intents.count("complex") == 5

    def test_golden_set_has_required_fields(self):
        from src.evaluation.golden_set import GOLDEN_SET
        for item in GOLDEN_SET:
            assert "query" in item
            assert "expected_intent" in item
            assert "expected_tools" in item
            assert "check_contains" in item

    def test_summarize_results(self):
        from src.evaluation.golden_set import summarize_results
        results = [
            {
                "intent_correct": True, "tool_correct": True,
                "content_score": 0.8, "has_response": True,
                "latency_s": 2.5,
            },
            {
                "intent_correct": False, "tool_correct": True,
                "content_score": 0.6, "has_response": True,
                "latency_s": 3.0,
            },
        ]
        summary = summarize_results(results)
        assert summary["n_queries"] == 2
        assert summary["intent_accuracy"] == 0.5
        assert summary["tool_routing_accuracy"] == 1.0

    def test_summarize_empty(self):
        from src.evaluation.golden_set import summarize_results
        assert summarize_results([]) == {}


# ============================================================
# Tests: Graph Routing Logic
# ============================================================

class TestRouting:
    """Tests for the graph routing function (no LLM required)."""

    def test_route_data_query(self):
        from src.agent.graph import _route_by_intent
        state = {"intent_type": "data_query"}
        assert _route_by_intent(state) == "data_query"

    def test_route_prediction_fraud(self):
        from src.agent.graph import _route_by_intent
        state = {"intent_type": "prediction_fraud"}
        assert _route_by_intent(state) == "predict_fraud"

    def test_route_prediction_purchase(self):
        from src.agent.graph import _route_by_intent
        state = {"intent_type": "prediction_purchase"}
        assert _route_by_intent(state) == "predict_purchase"

    def test_route_knowledge(self):
        from src.agent.graph import _route_by_intent
        state = {"intent_type": "knowledge"}
        assert _route_by_intent(state) == "search_knowledge"

    def test_route_complex(self):
        from src.agent.graph import _route_by_intent
        state = {"intent_type": "complex"}
        assert _route_by_intent(state) == "complex_analysis"

    def test_route_unknown_defaults(self):
        from src.agent.graph import _route_by_intent
        state = {"intent_type": "unknown_type"}
        assert _route_by_intent(state) == "data_query"

    def test_route_none_defaults(self):
        from src.agent.graph import _route_by_intent
        state = {"intent_type": None}
        assert _route_by_intent(state) == "data_query"
