---
title: Reasoning Gym Environment
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reasoning
---

# Reasoning Gym Environment

A pure MCP environment that integrates with the [Reasoning Gym](https://github.com/open-thought/reasoning-gym) library, providing 100+ procedurally generated reasoning tasks with algorithmic verification.

## Quick Start

```python
from reasoning_gym_env import ReasoningGymEnv

with ReasoningGymEnv(base_url="http://localhost:8000") as env:
    env.reset()

    # Get the question
    question = env.call_tool("get_question")
    print(question["question"])  # "How many legs do 2 dogs have?"

    # Submit an answer
    result = env.call_tool("submit_answer", answer="8")
    print(f"Score: {result['score']}")  # 1.0
    print(f"Correct answer: {result['correct_answer']}")  # "8"
```

## Available Tools

| Tool | Description |
|------|-------------|
| `get_question()` | Returns current question and task type |
| `submit_answer(answer: str)` | Submits answer, returns score (0.0-1.0) and correct answer |
| `get_task_info()` | Returns available tasks and configuration |

## Configuration

Pass configuration directly to `reset()` to change tasks or settings at runtime:

```python
from reasoning_gym_env import ReasoningGymEnv

with ReasoningGymEnv(base_url="http://localhost:8000") as env:
    # Configure at reset time
    env.reset(task_name="basic_arithmetic", task_config={"max_value": 100})

    question = env.call_tool("get_question")
    result = env.call_tool("submit_answer", answer="42")

    # Switch to different task mid-session
    env.reset(task_name="chain_sum", dataset_size=50, seed=42)

    # Use composite dataset
    env.reset(task_specs=[
        {"name": "leg_counting", "weight": 2},
        {"name": "basic_arithmetic", "weight": 1},
    ])
```

**Available reset() parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `task_name` | `str` | Task type from reasoning-gym |
| `task_config` | `dict` | Task-specific configuration |
| `task_specs` | `list` | Composite dataset specs (mutually exclusive with task_name) |
| `dataset_size` | `int` | Number of questions in the dataset |
| `seed` | `int` | Random seed for reproducibility |

When any configuration parameter changes, the dataset is automatically rebuilt.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         SERVER SIDE                              │
│  ReasoningGymEnvironment ──► FastAPI App ──► HTTP Server :8000  │
│  (holds reasoning-gym        (app.py)                           │
│   dataset & MCP tools)                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↕ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT SIDE                              │
│  ReasoningGymEnv(MCPToolClient) ──► connects via base_url       │
│  (thin HTTP client, no reasoning-gym dependency)                │
└─────────────────────────────────────────────────────────────────┘
```

The **server** holds the reasoning-gym dataset and configuration. The **client** is a thin HTTP wrapper that sends tool calls to the server.

## Running the Server

### With uv (development)

```bash
cd envs/reasoning_gym_env
uv run server
```

### With uvicorn

```bash
cd envs/reasoning_gym_env
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### With Docker

```bash
# Build image (from project root)
docker build -t reasoning-gym-env:latest -f envs/reasoning_gym_env/Dockerfile .

# Run container
docker run -p 8000:8000 reasoning-gym-env:latest
```

## Example Agent Loop

```python
from reasoning_gym_env import ReasoningGymEnv

with ReasoningGymEnv(base_url="http://localhost:8000") as env:
    for episode in range(10):
        env.reset()

        # Get question
        q = env.call_tool("get_question")
        print(f"Question: {q['question']}")
        print(f"Task: {q['task']}")

        # Your agent generates an answer here
        answer = generate_answer(q['question'])

        # Submit and get feedback
        result = env.call_tool("submit_answer", answer=answer)
        print(f"Score: {result['score']}, Correct: {result['correct_answer']}")
```

## Available Tasks

Reasoning Gym provides 100+ task types including:

- **Arithmetic**: basic_arithmetic, chain_sum, fraction_simplification
- **Logic**: boolean_logic, propositional_logic, syllogism
- **Counting**: leg_counting, letter_counting, word_sorting
- **Pattern**: pattern_completion, sequence_completion
- **And many more...**

See the [Reasoning Gym documentation](https://github.com/open-thought/reasoning-gym) for a full list.

## Project Structure

```
reasoning_gym_env/
├── __init__.py              # Module exports
├── README.md                # This file
├── client.py                # ReasoningGymEnv client
├── openenv.yaml             # Environment manifest
├── pyproject.toml           # Dependencies
└── server/
    ├── __init__.py          # Server module exports
    ├── reasoning_gym_environment.py  # Core environment logic
    └── app.py               # FastAPI application
```
