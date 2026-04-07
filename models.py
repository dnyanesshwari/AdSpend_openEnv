# models.py
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server import Action, Observation
from openenv.core.env_server.types import State


# ── What the agent SEES each step ────────────────────────────────────────────

class AdSpendObservation(Observation):
    """
    Inheriting from openenv Observation gives you:
      - done:   bool   — is episode over?
      - reward: float  — per-step reward
    Domain-specific fields are added below.
    """
    time_slot: int                        = 0
    budget_remaining: float               = 1000.0
    bid_price: float                      = 0.5
    estimated_traffic: int                = 0
    estimated_ctr: float                  = 0.03
    conversion_probability: float         = 0.02
    customer_value: float                 = 20.0
    slots_remaining: int                  = 48
    spend_so_far: float                   = 0.0
    total_revenue: float                  = 0.0
    conversions: int                      = 0
    whale_conversions: int                = 0
    task: str                             = "medium"
    task_name: str                        = "maximize_roas_medium"
    daily_budget: float                   = 1000.0
    current_roas: float                   = 0.0
    task_score: float                     = 0.0
    budget_shock_active: bool             = False
    # Optional metadata blob passed by the environment server
    metadata: Optional[Dict[str, Any]]   = None


# ── What the agent DOES each step ────────────────────────────────────────────

class AdSpendAction(Action):
    """
    0.0  = skip this slot entirely (save budget)
    1.0  = bid at market price
    2.0  = bid aggressively (you really want this customer)
    """
    bid_multiplier: float = 0.5     # range: 0.0 → 2.0
    task: str             = "medium"


# ── Environment state (exposed via /state endpoint) ───────────────────────────

class AdSpendState(State):
    task: str                 = "medium"
    task_name: str            = "maximize_roas_medium"
    daily_budget: float       = 1000.0
    max_slots: int            = 48
    current_slot: int         = 0
    budget_remaining: float   = 1000.0
    spend_so_far: float       = 0.0
    total_revenue: float      = 0.0
    conversions: int          = 0
    whale_conversions: int    = 0
    current_roas: float       = 0.0
    task_score: float         = 0.0
    budget_shock_active: bool = False


class AdSpendReward(BaseModel):
    """Typed reward decomposition for interpretability and debugging."""

    reward: float = Field(default=0.0)
    revenue_component: float = Field(default=0.0)
    cost_component: float = Field(default=0.0)
    pacing_component: float = Field(default=0.0)
    outcome_component: float = Field(default=0.0)
    penalty_component: float = Field(default=0.0)


class AdSpendStepInfo(BaseModel):
    """Typed per-step info emitted via observation/state."""

    task: str = Field(default="medium")
    action_applied: float = Field(default=0.0, ge=0.0, le=2.0)
    won_auction: bool = Field(default=False)
    clicks: int = Field(default=0, ge=0)
    conversions_gained: int = Field(default=0, ge=0)
    whale_conversions_gained: int = Field(default=0, ge=0)
    cost_incurred: float = Field(default=0.0, ge=0.0)
    revenue_generated: float = Field(default=0.0, ge=0.0)
    budget_shock_applied: bool = Field(default=False)
    score_after_step: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_breakdown: AdSpendReward = Field(default_factory=AdSpendReward)


class AdSpendObservation(Observation):
    """Typed observation returned by reset() and step()."""

    time_slot: int = Field(default=0, ge=0, le=48)
    budget_remaining: float = Field(default=1000.0, ge=0.0)
    bid_price: float = Field(default=0.5, ge=0.0)
    estimated_traffic: int = Field(default=0, ge=0)
    estimated_ctr: float = Field(default=0.03, ge=0.0, le=1.0)
    conversion_probability: float = Field(default=0.02, ge=0.0, le=1.0)
    customer_value: float = Field(default=20.0, ge=0.0)
    slots_remaining: int = Field(default=48, ge=0, le=48)
    spend_so_far: float = Field(default=0.0, ge=0.0)
    total_revenue: float = Field(default=0.0, ge=0.0)
    conversions: int = Field(default=0, ge=0)
    whale_conversions: int = Field(default=0, ge=0)
    task: str = Field(default="medium")
    task_name: str = Field(default="maximize_roas_medium")
    daily_budget: float = Field(default=1000.0, gt=0.0)
    current_roas: float = Field(default=0.0, ge=0.0)
    task_score: float = Field(default=0.0, ge=0.0, le=1.0)
    budget_shock_active: bool = Field(default=False)
    info: AdSpendStepInfo = Field(default_factory=AdSpendStepInfo)
    metadata: Optional[Dict[str, Any]] = None


class AdSpendAction(Action):
    """Bid decision for one auction slot."""

    bid_multiplier: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="0.0 skips the slot, 1.0 bids at market price, 2.0 bids aggressively.",
    )


class AdSpendState(State):
    """Full environment state exposed by /state."""

    task: str = Field(default="medium")
    task_name: str = Field(default="maximize_roas_medium")
    daily_budget: float = Field(default=1000.0, gt=0.0)
    max_slots: int = Field(default=48, ge=1)
    current_slot: int = Field(default=0, ge=0, le=48)
    budget_remaining: float = Field(default=1000.0, ge=0.0)
    spend_so_far: float = Field(default=0.0, ge=0.0)
    total_revenue: float = Field(default=0.0, ge=0.0)
    conversions: int = Field(default=0, ge=0)
    whale_conversions: int = Field(default=0, ge=0)
    current_roas: float = Field(default=0.0, ge=0.0)
    task_score: float = Field(default=0.0, ge=0.0, le=1.0)
    budget_shock_active: bool = Field(default=False)
    last_info: AdSpendStepInfo = Field(default_factory=AdSpendStepInfo)
