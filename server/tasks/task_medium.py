# server/tasks/task_medium.py

TASK_CONFIG = {
    "name": "maximize_roas_medium",
    "description": (
        "Maximize return on ad spend while pacing total spend into the preferred "
        "30%-60% budget band on a $1000 campaign day."
    ),
    "daily_budget": 1000.0,
    "seed": 200,
    "max_slots": 48,
    "difficulty": "medium",
}


def _band_score(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        return max(0.0, value / low)
    # Penalize overspending more steeply than underspending.
    return max(0.0, 1.0 - min((value - high) / high, 1.0))


def grade(env_state: dict) -> float:
    """Medium task: balance efficiency with realistic campaign pacing."""

    revenue = float(env_state.get("total_revenue", 0.0))
    spend = float(env_state.get("spend_so_far", 0.0))
    budget = float(env_state.get("daily_budget", TASK_CONFIG["daily_budget"]))

    spend_ratio = (spend / budget) if budget > 0 else 0.0
    pacing_score = _band_score(spend_ratio, 0.30, 0.60)

    if spend <= 0.0:
        roas_score = 0.0
    else:
        roas = revenue / spend
        roas_score = min(roas / 4.5, 1.0)

    score = (0.65 * roas_score) + (0.35 * pacing_score)
    return round(min(max(score, 0.0), 1.0), 4)
