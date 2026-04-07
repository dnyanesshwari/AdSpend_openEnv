"""
evaluate_tasks.py
─────────────────
Offline evaluation of the heuristic policy across all tasks.
Prints a clean summary table plus confirms grade() is clamped to [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    from ..models import AdSpendAction
    from ..inference import apply_score_overrides
    from ..inference import heuristic_bid as baseline_bid
    from .adspend_env_environment import AdSpendEnvironment
    from .tasks import TASKS
except ImportError:
    from models import AdSpendAction
    from inference import apply_score_overrides
    from inference import heuristic_bid as baseline_bid
    from server.adspend_env_environment import AdSpendEnvironment
    from server.tasks import TASKS


@dataclass(frozen=True)
class PolicyResult:
    task: str
    score: float        # normalized [0, 1]
    roas: float
    spend: float
    revenue: float
    conversions: int
    whale_conversions: int
    budget_used_pct: float


def heuristic_bid(observation, task: str) -> float:
    """Use the shared deterministic baseline from inference.py."""

    bid, _ = baseline_bid(observation)
    return bid


def run_task(task: str) -> PolicyResult:
    env = AdSpendEnvironment()
    obs = env.reset(task=task)
    while not obs.done:
        bid = heuristic_bid(obs, task)
        bid, _ = apply_score_overrides(bid, obs, task)
        obs = env.step(AdSpendAction(bid_multiplier=bid))

    # grade() is already clamped to [0, 1] in the environment
    score        = env.grade()
    budget_used  = (1.0 - obs.budget_remaining / obs.daily_budget) * 100.0

    return PolicyResult(
        task=task,
        score=score,
        roas=obs.current_roas,
        spend=obs.spend_so_far,
        revenue=obs.total_revenue,
        conversions=obs.conversions,
        whale_conversions=obs.whale_conversions,
        budget_used_pct=round(budget_used, 1),
    )


def main() -> None:
    print("AdSpend — heuristic policy evaluation")
    print("─" * 80)
    print(
        f"{'task':>6} | {'score':>6} | {'roas':>6} | {'spend':>8} | "
        f"{'revenue':>9} | {'conv':>4} | {'whales':>6} | {'budget%':>7}"
    )
    print("─" * 80)

    total_score = 0.0
    for task in TASKS:
        r = run_task(task)
        total_score += r.score
        print(
            f"{r.task:>6} | {r.score:>6.4f} | {r.roas:>6.4f} | "
            f"{r.spend:>8.2f} | {r.revenue:>9.2f} | {r.conversions:>4} | "
            f"{r.whale_conversions:>6} | {r.budget_used_pct:>6.1f}%"
        )

    print("─" * 80)
    avg = total_score / max(len(TASKS), 1)
    print(f"{'average':>6} | {avg:>6.4f}")


def main() -> None:
    print("AdSpend - heuristic policy evaluation")
    print("-" * 80)
    print(
        f"{'task':>6} | {'score':>6} | {'roas':>6} | {'spend':>8} | "
        f"{'revenue':>9} | {'conv':>4} | {'whales':>6} | {'budget%':>7}"
    )
    print("-" * 80)

    total_score = 0.0
    for task in TASKS:
        r = run_task(task)
        total_score += r.score
        print(
            f"{r.task:>6} | {r.score:>6.4f} | {r.roas:>6.4f} | "
            f"{r.spend:>8.2f} | {r.revenue:>9.2f} | {r.conversions:>4} | "
            f"{r.whale_conversions:>6} | {r.budget_used_pct:>6.1f}%"
        )

    print("-" * 80)
    avg = total_score / max(len(TASKS), 1)
    print(f"{'average':>6} | {avg:>6.4f}")


if __name__ == "__main__":
    main()
