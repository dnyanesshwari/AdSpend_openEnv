# server/tasks/task_hard.py

TASK_CONFIG = {
    "name": "whale_hunter_hard",
    "description": (
        "After a noon budget shock cuts remaining budget in half, continue prioritizing "
        "high-value whale customers while maintaining healthy post-shock efficiency."
    ),
    "daily_budget": 1000.0,
    "seed": 300,
    "max_slots": 48,
    "difficulty": "hard",
    "budget_shock_slot": 24,
    "budget_shock_multiplier": 0.5,
}


def grade(env_state: dict) -> float:
    """Hard task: whale targeting plus efficient recovery after a mid-day shock."""

    whale_conversions = int(env_state.get("whale_conversions", 0))
    conversions = int(env_state.get("conversions", 0))
    revenue = float(env_state.get("total_revenue", 0.0))
    spend = float(env_state.get("spend_so_far", 0.0))
    post_shock_spend = float(env_state.get("post_shock_spend", 0.0))
    shock_budget = float(env_state.get("shock_budget_reference", TASK_CONFIG["daily_budget"] * TASK_CONFIG["budget_shock_multiplier"]))

    whale_score = min(whale_conversions / 3.0, 1.0)
    quality_score = min((whale_conversions / max(conversions, 1)) / 0.5, 1.0) if conversions > 0 else 0.0

    if spend <= 0.0:
        roas_score = 0.0
    else:
        roas_score = min((revenue / spend) / 3.0, 1.0)

    if shock_budget <= 0.0:
        post_shock_pacing = 0.0
    else:
        post_shock_ratio = post_shock_spend / shock_budget
        post_shock_pacing = 1.0 if post_shock_ratio <= 0.85 else max(0.0, 1.0 - min((post_shock_ratio - 0.85) / 0.65, 1.0))

    score = (0.45 * whale_score) + (0.20 * quality_score) + (0.20 * roas_score) + (0.15 * post_shock_pacing)
    return round(min(max(score, 0.0), 1.0), 4)
