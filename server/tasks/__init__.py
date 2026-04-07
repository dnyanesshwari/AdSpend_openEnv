from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import task_easy, task_hard, task_medium


GradeFn = Callable[[dict[str, Any]], float]


@dataclass(frozen=True)
class TaskDefinition:
    key: str
    name: str
    difficulty: str
    description: str
    daily_budget: float
    seed: int
    max_slots: int
    grader: GradeFn
    budget_shock_slot: int | None = None
    budget_shock_multiplier: float = 1.0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "difficulty": self.difficulty,
            "description": self.description,
            "daily_budget": self.daily_budget,
            "seed": self.seed,
            "max_slots": self.max_slots,
            "budget_shock_slot": self.budget_shock_slot,
            "budget_shock_multiplier": self.budget_shock_multiplier,
        }


def _build_definition(module: Any) -> TaskDefinition:
    config = module.TASK_CONFIG
    return TaskDefinition(
        key=config["difficulty"],
        name=config["name"],
        difficulty=config["difficulty"],
        description=config["description"],
        daily_budget=float(config["daily_budget"]),
        seed=int(config["seed"]),
        max_slots=int(config["max_slots"]),
        grader=module.grade,
        budget_shock_slot=config.get("budget_shock_slot"),
        budget_shock_multiplier=float(config.get("budget_shock_multiplier", 1.0)),
    )


TASKS: dict[str, TaskDefinition] = {
    definition.key: definition
    for definition in (
        _build_definition(task_easy),
        _build_definition(task_medium),
        _build_definition(task_hard),
    )
}

TASK_NAME_TO_KEY: dict[str, str] = {
    definition.name: definition.key for definition in TASKS.values()
}


def get_task_definition(task: str | None) -> TaskDefinition:
    normalized = (task or "medium").strip().lower()
    normalized = TASK_NAME_TO_KEY.get(normalized, normalized)
    if normalized not in TASKS:
        supported = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task '{task}'. Supported tasks: {supported}")
    return TASKS[normalized]


def list_task_metadata() -> list[dict[str, Any]]:
    return [definition.to_metadata() for definition in TASKS.values()]
