# server/tasks/task_easy.py

TASK_CONFIG = {
    "name": "budget_pacing_easy",
    "description": (
        "Deliver 5 conversions while keeping spend at or below a $200 daily budget. "
        "The ideal policy reaches the target without materially overspending."
    ),
    "daily_budget": 200.0,
    "seed": 100,
    "max_slots": 48,
    "difficulty": "easy",
}


def grade(env_state: dict) -> float:
    """Easy task: hit a concrete conversion target while respecting budget."""

    conversions = int(env_state.get("conversions", 0))
    spend = float(env_state.get("spend_so_far", 0.0))
    budget = float(env_state.get("daily_budget", TASK_CONFIG["daily_budget"]))

    target_conversions = 5
    conversion_progress = min(conversions / target_conversions, 1.0)
    exact_target_bonus = 1.0 if conversions == target_conversions else max(0.0, 1.0 - 0.18 * abs(conversions - target_conversions))

    budget_ratio = spend / budget if budget > 0 else 1.0
    if budget_ratio <= 0.75:
        budget_efficiency = 0.75 + 0.25 * (budget_ratio / 0.75)
    elif budget_ratio <= 1.0:
        budget_efficiency = 1.0
    else:
        budget_efficiency = max(0.0, 1.0 - min(budget_ratio - 1.0, 1.0))

    score = (0.5 * conversion_progress) + (0.25 * exact_target_bonus) + (0.25 * budget_efficiency)
    return round(min(max(score, 0.0), 1.0), 4)
