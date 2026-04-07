# client.py
from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import AdSpendAction, AdSpendObservation, AdSpendState
except ImportError:
    from models import AdSpendAction, AdSpendObservation, AdSpendState


class AdSpendEnv(EnvClient[AdSpendAction, AdSpendObservation, AdSpendState]):
    """
    Typed HTTP/WebSocket client for the AdSpend Personalizer environment.

    Usage (against a running server):
        env = AdSpendEnv(base_url="http://localhost:8000")
        obs = env.reset(task="medium")
        while not obs.done:
            obs = env.step(AdSpendAction(bid_multiplier=1.0))
    """

    def _step_payload(self, action: AdSpendAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[AdSpendObservation]:
        observation = AdSpendObservation(**payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> AdSpendState:
        return AdSpendState(**payload)