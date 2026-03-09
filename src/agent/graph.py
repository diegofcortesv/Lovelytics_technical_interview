"""
LangGraph StateGraph composition for the Fraud Copilot agent.

This module builds the compiled graph that orchestrates:
    classify_intent → [tool_node] → assess_load → synthesize → END

The graph is parameterized by the LLM client, DataFrames, and feature lists
to keep the composition decoupled from Databricks-specific initialization.
"""

from functools import partial
from langgraph.graph import StateGraph, END
from .nodes import (
    CopilotState,
    classify_intent,
    node_data_query,
    node_complex,
    node_predict_fraud,
    node_predict_purchase,
    node_search_knowledge,
    node_assess_load,
    node_synthesize,
)


def _route_by_intent(state: CopilotState) -> str:
    """Conditional edge: route from classify to the appropriate tool node."""
    intent = state.get("intent_type", "data_query")
    routing = {
        "data_query": "data_query",
        "prediction_fraud": "predict_fraud",
        "prediction_purchase": "predict_purchase",
        "knowledge": "search_knowledge",
        "complex": "complex_analysis",
    }
    return routing.get(intent, "data_query")


def build_copilot_graph(
    llm,
    fraud_df,
    fraud_features: list[str],
    purchase_df,
    purchase_features: list[str],
):
    """
    Build and compile the Fraud Copilot LangGraph.

    Args:
        llm: LangChain-compatible LLM client (e.g., ChatDatabricks)
        fraud_df: Pandas DataFrame with fraud transaction data
        fraud_features: List of feature column names for the fraud model
        purchase_df: Pandas DataFrame with purchase data
        purchase_features: List of feature column names for the purchase model

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    graph = StateGraph(CopilotState)

    # Bind dependencies to nodes via partial application
    graph.add_node("classify", partial(classify_intent, llm=llm))
    graph.add_node("data_query", node_data_query)
    graph.add_node("predict_fraud", partial(
        node_predict_fraud, fraud_df=fraud_df, fraud_features=fraud_features
    ))
    graph.add_node("predict_purchase", partial(
        node_predict_purchase, purchase_df=purchase_df, purchase_features=purchase_features
    ))
    graph.add_node("search_knowledge", node_search_knowledge)
    graph.add_node("complex_analysis", partial(
        node_complex, fraud_df=fraud_df, fraud_features=fraud_features
    ))
    graph.add_node("assess_load", node_assess_load)
    graph.add_node("synthesize", partial(node_synthesize, llm=llm))

    # Entry point
    graph.set_entry_point("classify")

    # Conditional routing from classify
    graph.add_conditional_edges("classify", _route_by_intent, {
        "data_query": "data_query",
        "predict_fraud": "predict_fraud",
        "predict_purchase": "predict_purchase",
        "search_knowledge": "search_knowledge",
        "complex_analysis": "complex_analysis",
    })

    # All tool nodes → assess_load → synthesize → END
    for node in ["data_query", "predict_fraud", "predict_purchase",
                  "search_knowledge", "complex_analysis"]:
        graph.add_edge(node, "assess_load")

    graph.add_edge("assess_load", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def invoke_copilot(copilot, query: str) -> dict:
    """
    Convenience function to invoke the compiled graph with a query.

    Returns the full state dict including final_response, tools_used, etc.
    """
    return copilot.invoke({
        "user_query": query,
        "intent_type": None,
        "intent_confidence": None,
        "data_result": None,
        "prediction_result": None,
        "knowledge_result": None,
        "load_assessment": None,
        "final_response": None,
        "tools_used": None,
        "sources_cited": None,
    })
