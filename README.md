# AdSpend Personalizer

Real-world OpenEnv environment for ad-campaign budget pacing and customer-value-aware bidding. An agent decides how aggressively to bid across 48 half-hour auction windows in a simulated campaign day while balancing conversions, ROAS, pacing, and whale-customer capture.

## Why this environment

Digital marketing teams really do this work: they pace spend, chase efficient conversion windows, protect ROAS, and adapt to budget changes during a live campaign. This environment turns that workflow into a compact but meaningful RL and agent benchmark rather than a toy game.

What makes it useful:

- real business objective instead of a game loop
- dense trajectory reward rather than purely sparse terminal scoring
- multiple evaluation modes: conversion target, ROAS pacing, and shock adaptation
- typed OpenEnv API suitable for agent benchmarking

## OpenEnv contract

The environment implements the standard OpenEnv interface:

- `reset(task=..., seed=...) -> AdSpendObservation`
- `step(AdSpendAction) -> AdSpendObservation`
- `state -> AdSpendState`

Validation:

```bash
uv run openenv validate .
```

Current result:

```text
[OK] : Ready for multi-mode deployment
```

## Environment design

One episode is one campaign day split into 48 half-hour slots.

At each slot the agent observes:

- slot index and slots remaining
- remaining budget and spend so far
- market bid price
- estimated traffic
- estimated CTR
- estimated conversion probability
- customer value
- current ROAS
- conversion counts
- task score and budget-shock flag
- typed step diagnostics and reward breakdown

The agent acts with:

```python
AdSpendAction(
    bid_multiplier: float  # constrained to [0.0, 2.0]
)
```

Interpretation:

- `0.0` = skip the slot
- `1.0` = bid at market level
- `2.0` = bid aggressively

## Tasks

### Easy: `budget_pacing_easy`

Goal:
Reach 5 conversions while keeping spend at or below a $200 budget.

Grader values:

- conversion progress toward 5
- landing near the exact target
- budget efficiency

### Medium: `maximize_roas_medium`

Goal:
Maximize ROAS while pacing spend toward the preferred 30%-60% band of a $1000 daily budget.

Grader values:

- ROAS
- spend pacing into the preferred budget band

### Hard: `whale_hunter_hard`

Goal:
Survive a noon budget shock, keep bidding selectively, and capture high-value whale customers efficiently.

Grader values:

- whale conversions
- quality of conversions
- post-shock efficiency
- post-shock pacing discipline

## Reward function

The environment uses dense reward with multiple components:

- realized revenue reward
- spend cost penalty
- task-specific pacing reward
- conversion and whale bonuses
- penalties for early budget exhaustion and other undesirable behavior

This gives the agent useful learning signal across the whole episode instead of only at the end.

## Reproducible baseline

Deterministic heuristic baseline:

| Task | Score | ROAS | Spend | Revenue |
|---|---:|---:|---:|---:|
| easy | 0.8118 | 8.0881 | 46.26 | 374.12 |
| medium | 0.4657 | 2.1373 | 134.55 | 287.58 |
| hard | 0.7000 | 4.6015 | 28.33 | 130.36 |
| average | 0.6592 | - | - | - |

Run baseline:

```bash
uv sync
uv run python inference.py --policy heuristic --task all
uv run python evaluate_tasks.py
```

## OpenAI model baseline

The inference script uses the OpenAI Python client.

Environment variables:

- `OPENAI_API_KEY` required for model runs
- `API_BASE_URL` optional, defaults to OpenAI
- `MODEL_NAME` optional, defaults to `gpt-4o-mini`

PowerShell example:

```bash
$env:OPENAI_API_KEY="your_key"
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
uv run python inference.py --task all
```

## Commands to run the complete project

### 1. Install dependencies

```bash
uv sync
```

### 2. Validate OpenEnv spec

```bash
uv run openenv validate .
```

### 3. Run local environment server

```bash
uv run python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 4. Run deterministic heuristic benchmark

```bash
uv run python inference.py --policy heuristic --task all
```

### 5. Run offline evaluation table

```bash
uv run python evaluate_tasks.py
```

### 6. Run a single task

```bash
uv run python inference.py --policy heuristic --task easy
uv run python inference.py --policy heuristic --task medium
uv run python inference.py --policy heuristic --task hard
```

### 7. Run with an OpenAI model

```bash
$env:OPENAI_API_KEY="your_key"
uv run python inference.py --task all
```

### 8. Build and run the container

```bash
docker build -t adspend-env .
docker run -p 7860:7860 -e PORT=7860 adspend-env
```

### 9. Health check the running server

Local uvicorn:

```bash
curl http://localhost:8000/health
```

Docker or HF-style port:

```bash
curl http://localhost:7860/health
```

## Hugging Face Space notes

Recommended configuration:

- SDK: `docker`
- Port: `7860`
- Tag: `openenv`

The project includes a root `Dockerfile` and the FastAPI server reads `PORT`, which makes it suitable for container deployment.

 

## Project structure

```text
adspend_env/
├── inference.py
├── evaluate_tasks.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── models.py
├── client.py
└── server/
    ├── app.py
    ├── adspend_env_environment.py
    ├── evaluate_tasks.py
    └── tasks/
        ├── task_easy.py
        ├── task_medium.py
        └── task_hard.py
```

  
