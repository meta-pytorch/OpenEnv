---
title: Grid World Environment Server
emoji: 🎻
colorFrom: yellow
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Grid World Environment

Grid World is a simple 5x5 navigation task with a fixed goal at (4, 4). The agent moves with cardinal actions and receives a small step penalty until it reaches the goal. Each observation also includes a `suggested_action` that you can pass directly into the next step.

## Architecture

```
┌────────────────────────────────────┐
│ RL Client                           │
│   GridWorldEnv.step(action)         │
└──────────────┬─────────────────────┘
               │ HTTP / WebSocket
┌──────────────▼─────────────────────┐
│ FastAPI Server (Docker)             │
│   GridWorldEnvironment              │
│     ├─ Reset/Step/State endpoints   │
│     ├─ Reward + termination         │
│     └─ Action validation            │
└────────────────────────────────────┘
```

## Installation & Usage

### Option 1: Local Development (without Docker)

**Requirements:**
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

```bash
cd envs/grid_world

# Install the package and dependencies
uv pip install -e .
# or
pip install -e .
```

Run the server locally:

```bash
cd envs/grid_world
uv run --project . server --port 8000
# or
uvicorn server.app:app --reload --port 8000
```

Connect with the client:

```python
from grid_world import GridWorldAction, GridWorldEnv

env = GridWorldEnv(base_url="http://localhost:8000")
result = env.reset()
print(result.observation.message)

action = result.observation.suggested_action
result = env.step(GridWorldAction(action=action))
print(result.observation.suggested_action, result.reward)

env.close()
```

### Option 2: Docker (Recommended)

Build the image from the repo root:

```bash
cd /path/to/OpenEnv
docker build -f envs/grid_world/server/Dockerfile -t grid-world-env:latest .
```

Run the container:

```bash
docker run -p 8000:8000 grid-world-env:latest
```

Use with `from_docker_image()`:

```python
from grid_world import GridWorldAction, GridWorldEnv

env = None
try:
    # Create environment from Docker image
    env = GridWorldEnv.from_docker_image("grid-world-env:latest")

    # Reset to start a new episode
    result = env.reset()
    print(f"Initial suggested action: {result.observation.suggested_action}")
    print(f"Message: {result.observation.message}")

    # Play until done
    while not result.done:
        action = result.observation.suggested_action
        result = env.step(GridWorldAction(action=action))
        print(f"Reward: {result.reward}, Done: {result.done}")

finally:
    if env is not None:
        env.close()
```

## API Endpoints

- `GET /health` - Container health check
- `POST /reset` - Reset the environment
- `POST /step` - Execute an action
- `GET /state` - Fetch current state
- `GET /schema` - Action/observation schema
- `WS /ws` - WebSocket endpoint for low-latency sessions

## Environment Details

### Actions

**GridWorldAction**
- `action` (enum): `UP`, `DOWN`, `LEFT`, `RIGHT`

### Observations

**GridWorldObservation**
- `x` (int): Agent x position
- `y` (int): Agent y position
- `suggested_action` (MoveAction | null): Recommended next move toward the goal
- `message` (str): Status message
- `reward` (float | null): Reward for the transition
- `done` (bool): Episode termination flag

### Rewards & Termination

- Each step gives `-0.1` reward.
- Reaching `(4, 4)` yields `+1.0` and `done = True`.
- Reset returns `reward = null` and `done = False`.

### State

`GET /state` returns:
- `episode_id`, `step_count`
- `agent_x`, `agent_y`
- `goal_x`, `goal_y`
- `grid_size`, `episode_steps`


## Project Structure

```
grid_world/
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # GridWorldEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── grid_world_environment.py  # Core environment logic
    ├── app.py             # FastAPI application
    └── Dockerfile         # Container image definition
```
