"""
adspend_env_environment.py
──────────────────────────
Ad-spend pacing environment with task-aware evaluation.

One episode = one simulated ad-campaign day split into 48 half-hour slots.
The selected task controls the budget, seed, grading rubric, and optional
task-specific dynamics (e.g. hard-mode noon budget shock).
"""
from __future__ import annotations

import math
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import (
        AdSpendAction,
        AdSpendObservation,
        AdSpendReward,
        AdSpendState,
        AdSpendStepInfo,
    )
    from .tasks import get_task_definition, list_task_metadata
except ImportError:
    from models import (
        AdSpendAction,
        AdSpendObservation,
        AdSpendReward,
        AdSpendState,
        AdSpendStepInfo,
    )
    from server.tasks import get_task_definition, list_task_metadata


class AdSpendEnvironment(Environment):
    """Task-aware ad-spend RL environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    concurrency_safe                   = True

    def __init__(self) -> None:
        super().__init__()
        self._rng = random.Random()
        self._task_definition = get_task_definition("medium")
        self._shock_applied = False
        self._last_info = AdSpendStepInfo(task=self._task_definition.key)
        self._state = self._build_state()
        self._reset_internal_for_task(seed=self._task_definition.seed)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task: str = "medium",
        **_: object,
    ) -> AdSpendObservation:
        """Reset the environment for a specific task."""
        self._task_definition = get_task_definition(task)
        self._state = self._build_state(episode_id=episode_id)
        effective_seed = self._task_definition.seed if seed is None else seed
        self._reset_internal_for_task(seed=effective_seed)
        self._last_info = AdSpendStepInfo(task=self._task_definition.key)
        return self._build_obs(
            reward=0.0,
            done=False,
            info=self._last_info.model_copy(update={"score_after_step": round(self.grade(), 4)}),
        )

    def step(
        self,
        action: AdSpendAction,
        timeout_s: float | None = None,
        **_: object,
    ) -> AdSpendObservation:
        """Advance the simulation by one slot."""
        del timeout_s

        if self._slot >= self._task_definition.max_slots or self._budget <= 0:
            terminal_info = self._last_info.model_copy(update={"score_after_step": round(self.grade(), 4)})
            return self._build_obs(reward=0.0, done=True, info=terminal_info)

        shock_applied = self._apply_task_events()

        bid    = max(0.0, min(2.0, action.bid_multiplier))
        market = self._get_market_data(self._slot)
        result = self._run_auction(bid, market)

        # ── Update state ─────────────────────────────────────────────────────
        self._spend  += result["cost"]
        self._budget  = max(0.0, self._budget - result["cost"])
        self._revenue += result["revenue"]
        if self._shock_applied:
            self._post_shock_spend += result["cost"]

        conversion_count = int(result["conversion_count"])
        if conversion_count:
            self._conversions += conversion_count
            if market["is_whale"]:
                self._whale_conversions += conversion_count

        reward_breakdown = self._compute_reward(action=bid, result=result, market=market, shock_applied=shock_applied)
        reward = reward_breakdown.reward
        self._slot += 1
        self._state = self._build_state(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count + 1,
        )

        done = (self._slot >= self._task_definition.max_slots) or (self._budget <= 0)

        # Bonus for exceptional end-of-episode ROAS
        if done and self._spend > 0:
            roas = self.current_roas
            if roas > 4.0:
                bonus = (roas - 4.0) * 5.0
                reward += bonus
                reward_breakdown.reward = reward
                reward_breakdown.outcome_component += bonus

        self._last_info = AdSpendStepInfo(
            task=self._task_definition.key,
            action_applied=bid,
            won_auction=bool(result["won"]),
            clicks=int(result["clicks"]),
            conversions_gained=conversion_count,
            whale_conversions_gained=conversion_count if bool(market["is_whale"]) else 0,
            cost_incurred=round(float(result["cost"]), 4),
            revenue_generated=round(float(result["revenue"]), 2),
            budget_shock_applied=shock_applied,
            score_after_step=round(self.grade(), 4),
            reward_breakdown=reward_breakdown,
        )

        return self._build_obs(reward=round(reward, 4), done=done, info=self._last_info)

    @property
    def state(self) -> AdSpendState:
        return self._build_state(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="adspend-personalizer",
            description=(
                "RL environment for budget pacing across ad slots with task-aware "
                "grading for easy, medium, and hard campaign scenarios."
            ),
            version="0.2.0",
            readme_content=(
                "Tasks are selected via reset(task=...) and evaluated with the "
                "matching task rubric."
            ),
        )

    @property
    def current_roas(self) -> float:
        return self._revenue / self._spend if self._spend > 0 else 0.0

    def grade(self) -> float:
        """Return a normalized score in [0, 1]."""
        raw = self._task_definition.grader(self._snapshot_metrics())
        return min(max(float(raw), 0.0), 1.0)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_state(
        self,
        episode_id: str | None = None,
        step_count: int = 0,
    ) -> AdSpendState:
        return AdSpendState(
            episode_id=episode_id or str(uuid4()),
            step_count=step_count,
            task=self._task_definition.key,
            task_name=self._task_definition.name,
            daily_budget=self._task_definition.daily_budget,
            max_slots=self._task_definition.max_slots,
            current_slot=getattr(self, "_slot", 0),
            budget_remaining=round(getattr(self, "_budget", self._task_definition.daily_budget), 2),
            spend_so_far=round(getattr(self, "_spend", 0.0), 2),
            total_revenue=round(getattr(self, "_revenue", 0.0), 2),
            conversions=getattr(self, "_conversions", 0),
            whale_conversions=getattr(self, "_whale_conversions", 0),
            current_roas=round(self.current_roas if hasattr(self, "_spend") else 0.0, 4),
            task_score=round(self.grade(), 4) if hasattr(self, "_spend") else 0.0,
            budget_shock_active=getattr(self, "_shock_applied", False),
            last_info=getattr(self, "_last_info", AdSpendStepInfo(task=self._task_definition.key)),
        )

    def _reset_internal_for_task(self, seed: int) -> None:
        self._seed           = seed
        self._rng            = random.Random(seed)
        self._budget         = self._task_definition.daily_budget
        self._spend          = 0.0
        self._revenue        = 0.0
        self._conversions    = 0
        self._whale_conversions = 0
        self._slot           = 0
        self._shock_applied  = False
        self._post_shock_spend = 0.0
        self._shock_budget_reference = self._task_definition.daily_budget * self._task_definition.budget_shock_multiplier
        self._market_cache: dict[int, dict[str, float | bool]] = {}
        self._last_info = AdSpendStepInfo(task=self._task_definition.key)

    def _build_obs(self, reward: float, done: bool, info: AdSpendStepInfo | None = None) -> AdSpendObservation:
        market = self._get_market_data(min(self._slot, self._task_definition.max_slots - 1))
        return AdSpendObservation(
            done=done,
            reward=reward,
            time_slot=self._slot,
            budget_remaining=round(self._budget, 2),
            bid_price=market["bid_price"],
            estimated_traffic=int(market["traffic_volume"]),
            estimated_ctr=market["estimated_ctr"],
            conversion_probability=market["conversion_prob"],
            customer_value=market["customer_value"],
            slots_remaining=max(0, self._task_definition.max_slots - self._slot),
            spend_so_far=round(self._spend, 2),
            total_revenue=round(self._revenue, 2),
            conversions=self._conversions,
            whale_conversions=self._whale_conversions,
            task=self._task_definition.key,
            task_name=self._task_definition.name,
            daily_budget=self._task_definition.daily_budget,
            current_roas=round(self.current_roas, 4),
            task_score=round(self.grade(), 4),
            budget_shock_active=self._shock_applied,
            info=info or self._last_info,
            metadata={
                "task": self._task_definition.to_metadata(),
                "metrics": self._snapshot_metrics(),
                "available_tasks": list_task_metadata(),
            },
        )

    def _snapshot_metrics(self) -> dict[str, float | int | bool | str]:
        return {
            "task": self._task_definition.key,
            "task_name": self._task_definition.name,
            "budget_remaining": round(self._budget, 2),
            "spend_so_far": round(self._spend, 2),
            "total_revenue": round(self._revenue, 2),
            "conversions": self._conversions,
            "whale_conversions": self._whale_conversions,
            "current_slot": self._slot,
            "max_slots": self._task_definition.max_slots,
            "daily_budget": self._task_definition.daily_budget,
            "current_roas": round(self.current_roas, 4),
            "budget_shock_active": self._shock_applied,
            "post_shock_spend": round(getattr(self, "_post_shock_spend", 0.0), 2),
            "shock_budget_reference": round(getattr(self, "_shock_budget_reference", 0.0), 2),
        }

    def _apply_task_events(self) -> bool:
        shock_slot = self._task_definition.budget_shock_slot
        if (
            shock_slot is not None
            and not self._shock_applied
            and self._slot >= shock_slot
        ):
            self._shock_budget_reference = self._budget * self._task_definition.budget_shock_multiplier
            self._budget *= self._task_definition.budget_shock_multiplier
            self._shock_applied = True
            return True
        return False

    def _get_market_data(self, slot: int) -> dict[str, float | bool]:
        if slot in self._market_cache:
            return dict(self._market_cache[slot])

        slot_rng = random.Random((self._seed * 1009) + (slot * 7919))
        hour     = (slot * 0.5) % 24

        morning     = 2.5 * math.exp(-0.5 * ((hour - 9.5) / 1.5) ** 2)
        evening_bid = 1.5 * math.exp(-0.5 * ((hour - 20)  / 2.0) ** 2)
        bid_price   = round(0.30 + morning + evening_bid, 3)

        ctr  = round(0.03 + 0.04 * math.exp(-0.5 * ((hour - 13) / 4)   ** 2), 4)
        conv = round(min(0.99, 0.02 + 0.06 * math.exp(-0.5 * ((hour - 20) / 3) ** 2)), 4)
        traffic_volume = max(
            8,
            int(round(
                18
                + 42 * math.exp(-0.5 * ((hour - 13) / 4.5) ** 2)
                + 55 * math.exp(-0.5 * ((hour - 20) / 2.8) ** 2)
            )),
        )

        whale_bias = 0.05 + (0.08 if hour >= 17 else 0.0) + (0.04 if self._task_definition.key == "hard" else 0.0)
        is_whale       = slot_rng.random() < min(0.25, whale_bias)
        customer_value = (
            round(slot_rng.uniform(80, 200), 2)
            if is_whale
            else round(slot_rng.uniform(5, 40), 2)
        )

        def noise(value: float) -> float:
            return max(0.001, value * (1 + slot_rng.gauss(0, 0.2)))

        market: dict[str, float | bool] = {
            "bid_price":      round(noise(bid_price), 3),
            "estimated_ctr":  round(min(0.99, noise(ctr)), 4),
            "conversion_prob": round(min(0.99, noise(conv)), 4),
            "customer_value": customer_value,
            "traffic_volume": max(1, int(round(noise(float(traffic_volume))))),
            "is_whale":       is_whale,
        }
        self._market_cache[slot] = market
        return dict(market)

    def _run_auction(
        self,
        bid_multiplier: float,
        market: dict[str, float | bool],
    ) -> dict[str, float | bool]:
        _zero: dict[str, float | bool] = {
            "won": False, "clicked": False, "converted": False,
            "clicks": 0, "conversion_count": 0, "cost": 0.0, "revenue": 0.0,
        }

        if bid_multiplier <= 0.0:
            return _zero

        win_prob = min(0.95, 0.3 + 0.55 * bid_multiplier)
        if not (self._rng.random() < win_prob):
            return _zero

        available_traffic = int(market["traffic_volume"])
        traffic_captured  = max(
            1,
            int(round(available_traffic * min(1.0, 0.35 + 0.4 * bid_multiplier))),
        )
        clicks = sum(
            1
            for _ in range(traffic_captured)
            if self._rng.random() < float(market["estimated_ctr"])
        )
        if clicks == 0:
            return {**_zero, "won": True}

        raw_cost = round(clicks * float(market["bid_price"]) * bid_multiplier, 4)
        affordable_clicks = clicks
        if raw_cost > self._budget and self._budget > 0.0:
            max_affordable = int(self._budget / max(float(market["bid_price"]) * bid_multiplier, 0.0001))
            affordable_clicks = max(1, min(clicks, max_affordable))
        cost = round(affordable_clicks * float(market["bid_price"]) * bid_multiplier, 4)

        if cost > self._budget:
            cost = round(self._budget, 4)
            affordable_clicks = min(affordable_clicks, clicks)

        conversion_count = sum(
            1
            for _ in range(affordable_clicks)
            if self._rng.random() < float(market["conversion_prob"])
        )
        revenue = float(market["customer_value"]) * conversion_count

        return {
            "won": True,
            "clicked": affordable_clicks > 0,
            "converted": conversion_count > 0,
            "clicks": affordable_clicks,
            "conversion_count": conversion_count,
            "cost": cost,
            "revenue": round(revenue, 2),
        }

    def _compute_reward(
        self,
        action: float,
        result: dict[str, float | bool],
        market: dict[str, float | bool],
        shock_applied: bool,
    ) -> AdSpendReward:
        revenue_component = float(result["revenue"]) * 0.08
        cost_component = -float(result["cost"]) * 0.06
        pacing_component = 0.0
        outcome_component = 0.0
        penalty_component = 0.0

        spend_ratio = self._spend / max(self._task_definition.daily_budget, 1.0)
        expected_progress = (self._slot + 1) / self._task_definition.max_slots

        if self._task_definition.key == "easy":
            target_gap_before = abs(5 - max(self._conversions - int(result["conversion_count"]), 0))
            target_gap_after = abs(5 - self._conversions)
            outcome_component += max(0.0, float(target_gap_before - target_gap_after)) * 1.5
            if self._conversions == 5:
                outcome_component += 2.5
            if self._conversions > 5:
                penalty_component -= min(2.0, 0.6 * (self._conversions - 5))
            if spend_ratio > 1.0:
                penalty_component -= min(3.0, 4.0 * (spend_ratio - 1.0))
            pacing_component += 0.5 * max(0.0, 1.0 - abs(spend_ratio - expected_progress) / 0.55)

        elif self._task_definition.key == "medium":
            if float(result["cost"]) == 0.0 and expected_progress < 0.3 and action == 0.0:
                pacing_component += 0.08
            if 0.30 <= spend_ratio <= 0.60:
                pacing_component += 0.9
            else:
                pacing_component += max(0.0, 0.9 - 1.5 * abs(spend_ratio - 0.45))
            if self.current_roas > 0.0:
                outcome_component += min(self.current_roas / 6.0, 1.0) * 1.2

        else:
            if bool(market["is_whale"]) and action > 0.0:
                outcome_component += 0.7
            if bool(market["is_whale"]) and bool(result["converted"]):
                outcome_component += 4.0 * int(result["conversion_count"])
            if shock_applied:
                pacing_component += 0.6
            if self._shock_applied and self._shock_budget_reference > 0:
                post_shock_ratio = self._post_shock_spend / self._shock_budget_reference
                pacing_component += max(0.0, 0.8 - max(0.0, post_shock_ratio - 0.85))
            if self.current_roas > 0.0:
                outcome_component += min(self.current_roas / 4.0, 1.0) * 0.7

        if self._budget <= 0 and self._slot < 38:
            penalty_component -= 10.0

        if action <= 0.0 and self._slot < 18 and (float(market["conversion_prob"]) * float(market["customer_value"]) / max(float(market["bid_price"]), 0.05)) < 0.8:
            pacing_component += 0.1

        reward = revenue_component + cost_component + pacing_component + outcome_component + penalty_component
        return AdSpendReward(
            reward=round(reward, 4),
            revenue_component=round(revenue_component, 4),
            cost_component=round(cost_component, 4),
            pacing_component=round(pacing_component, 4),
            outcome_component=round(outcome_component, 4),
            penalty_component=round(penalty_component, 4),
        )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    env = AdSpendEnvironment()
    for task_name in ("easy", "medium", "hard"):
        obs = env.reset(task=task_name)
        while not obs.done:
            bid = 0.2 if obs.time_slot < 18 else 1.0
            obs = env.step(AdSpendAction(bid_multiplier=bid))
        grade = env.grade()
        print(
            f"task={task_name} grade={grade:.4f} score={obs.task_score:.4f} "
            f"roas={obs.current_roas:.4f} conversions={obs.conversions} "
            f"whales={obs.whale_conversions}"
        )
