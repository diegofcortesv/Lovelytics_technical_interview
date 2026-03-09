"""
Agent policies: guardrails and cognitive load assessment.

The cognitive load module estimates analyst fatigue based on observable
session signals. It is heuristic-based (not ML) and designed to be
replaced with a trained model once real session data is available.
"""

import mlflow


# --- Cognitive Load Assessment ---

LOAD_LEVELS = {
    (0, 30): "normal",
    (31, 60): "elevated",
    (61, 80): "high",
    (81, 100): "critical",
}

FORMAT_INSTRUCTIONS = {
    "normal": "Provide a detailed, well-structured response.",
    "elevated": "Start with a brief summary, then provide details in bullet points.",
    "high": "Keep your response concise. Use short sentences. Skip secondary details.",
    "critical": "Give only the essential answer in 2-3 sentences.",
}


@mlflow.trace(span_type="TOOL", name="assess_analyst_load")
def assess_analyst_load(session_metrics: dict) -> dict:
    """
    Calculate cognitive load score (0-100) from session signals.

    Signals and weights:
        - queries_last_hour (20%): More queries = more load
        - avg_routing_tier (20%): Higher tier = more complex cases
        - session_duration_hours (15%): Fatigue after 4+ hours
        - avg_query_interval_sec (15%): Decreasing interval = urgency
        - followup_rate (15%): More clarifications = confusion
        - hour_of_day (15%): Circadian fatigue factor
    """
    score = 0.0

    # Query volume (15 queries/hour = max)
    queries_1h = session_metrics.get("queries_last_hour", 0)
    score += min(queries_1h / 15.0, 1.0) * 20

    # Case complexity (tier 4 = max)
    avg_complexity = session_metrics.get("avg_routing_tier", 1.0)
    score += (avg_complexity / 4.0) * 20

    # Session duration (6 hours = max fatigue)
    session_hours = session_metrics.get("session_duration_hours", 0)
    score += min(session_hours / 6.0, 1.0) * 15

    # Query velocity (< 30s interval = max urgency)
    interval = session_metrics.get("avg_query_interval_sec", 300)
    score += max(0, 1.0 - interval / 300.0) * 15

    # Follow-up rate (40% = max confusion)
    followup_rate = session_metrics.get("followup_rate", 0)
    score += min(followup_rate / 0.4, 1.0) * 15

    # Circadian factor
    hour = session_metrics.get("hour_of_day", 12)
    if 10 <= hour <= 14:
        circadian = 0.5
    elif hour > 18 or hour < 6:
        circadian = 0.8
    else:
        circadian = 0.3
    score += circadian * 15

    load_score = min(int(score), 100)

    # Determine level
    level = "normal"
    for (low, high), lvl in LOAD_LEVELS.items():
        if low <= load_score <= high:
            level = lvl
            break

    return {"load_score": load_score, "level": level}


def get_format_instruction(level: str) -> str:
    """Get response format instructions based on cognitive load level."""
    return FORMAT_INSTRUCTIONS.get(level, FORMAT_INSTRUCTIONS["normal"])


# --- Guardrail Checks ---

def validate_intent_confidence(confidence: float, threshold: float = 0.8) -> bool:
    """Check if intent classification confidence meets the threshold."""
    return confidence >= threshold


def requires_citation(intent: str) -> bool:
    """Determine if the intent type requires source document citations."""
    return intent == "knowledge"


def requires_shap(intent: str) -> bool:
    """Determine if the intent type requires SHAP explanations."""
    return intent in ("prediction_fraud", "prediction_purchase", "complex")
