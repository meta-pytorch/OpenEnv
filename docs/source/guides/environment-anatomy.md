# Environment Anatomy

:::{note}
Coming Soon
    This page is under construction.

A deep dive

A deep dive into the structure of OpenEnv environments.

## Components

Every OpenEnv environment consists of:

```
my_env/
├── openenv.yaml          # Manifest file
├── my_env/
│   ├── __init__.py
│   ├── client.py         # Client classes
│   ├── server.py         # Server/Environment
│   └── models.py         # Pydantic models
├── Dockerfile            # Container definition
├── pyproject.toml        # Package metadata
└── README.md             # Documentation
```

## The Manifest (openenv.yaml)

```yaml
name: my_env
version: 0.1.0
description: My custom environment

client:
  class_name: MyEnvClient
  module: my_env.client

action:
  class_name: MyAction
  module: my_env.models

observation:
  class_name: MyObservation
  module: my_env.models

default_image: my-env:latest
spec_version: 1
```

## Models (Pydantic)

```python
from pydantic import BaseModel

class MyAction(BaseModel):
    command: str
    args: list[str] = []

class MyObservation(BaseModel):
    output: str
    success: bool

class MyState(BaseModel):
    history: list[str] = []
```

## Environment Class

```python
from openenv.core import Environment

class MyEnvironment(Environment[MyAction, MyObservation, MyState]):
    def reset(self) -> MyObservation:
        ...

    def step(self, action: MyAction) -> tuple[MyObservation, float, bool]:
        ...

    def get_state(self) -> MyState:
        ...
```

## Server (FastAPI)

```python
from fastapi import FastAPI
from openenv.server import create_server

app = create_server(MyEnvironment)
```

## Rewards via the Rubric

Rewards in OpenEnv are computed **inside the environment** — they never come from outside the `step` call. The base `Environment` accepts an optional `rubric` parameter on `__init__` and stores it as `self.rubric`. The recommended flow is:

1. Build the rubric and pass it to `super().__init__(rubric=...)` so `self.rubric` is set.
2. Call `self._reset_rubric()` at the start of `reset` so trajectory state does not leak between episodes.
3. Call `self._apply_rubric(action, observation)` inside `step` and attach the result to `observation.reward`. Use `self._apply_rubric_async(...)` from `step_async`.

The rubric system (`openenv.core.rubrics`) ships with composable containers (`WeightedSum`, `Gate`, `Sequential`), an `LLMJudge` for subjective criteria, and `TrajectoryRubric` for delayed rewards. The [Rubrics tutorial](../tutorials/rubrics.md) covers the full API.

## Next Steps

- [Deployment](deployment.md) - Deploy your environment
- [Your First Environment](first-environment.md) - Build step by step
:::
