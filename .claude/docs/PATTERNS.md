# Code Patterns & Conventions

This document describes the canonical patterns for OpenEnv code. Follow these patterns for consistency.

## Environment Structure

Every environment follows this structure:
```
my_env/
├── __init__.py          # Export Action, Observation, Client
├── models.py            # Action, Observation, State (Pydantic)
├── client.py            # EnvClient[ActT, ObsT, StateT] subclass
├── openenv.yaml         # Environment manifest
├── pyproject.toml       # Dependencies
└── server/
    ├── my_environment.py  # Environment[ActT, ObsT, StateT] subclass
    ├── app.py             # create_app() with HTTPEnvServer
    ├── requirements.txt   # Docker dependencies
    └── Dockerfile
```

Use `openenv init <name>` to scaffold this structure.

## Type Safety Pattern

Always use generics for type safety across the wire:

```python
# models.py
from pydantic import BaseModel

class MyAction(BaseModel):
    command: str

class MyObservation(BaseModel):
    result: str
    reward: float
    done: bool

class MyState(BaseModel):
    episode_id: str
    step_count: int
```

```python
# client.py
from openenv.core import EnvClient, StepResult

class MyEnv(EnvClient[MyAction, MyObservation, MyState]):
    def _step_payload(self, action: MyAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[MyObservation]:
        obs = MyObservation(**payload["observation"])
        return StepResult(observation=obs, reward=obs.reward, done=obs.done)

    def _parse_state(self, payload: dict) -> MyState:
        return MyState(**payload)
```

```python
# server/my_environment.py
from openenv.core.env_server import Environment

class MyEnvironment(Environment[MyAction, MyObservation, MyState]):
    def reset(self, seed=None, episode_id=None) -> MyObservation:
        ...

    def step(self, action: MyAction) -> MyObservation:
        ...

    @property
    def state(self) -> MyState:
        ...
```

## Pydantic Models

- All wire types must be Pydantic models
- Use `Field()` for validation constraints
- Enable `arbitrary_types_allowed` for numpy/torch types

```python
from pydantic import BaseModel, Field
import numpy as np

class MyObservation(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    grid: np.ndarray
    score: float = Field(ge=0.0)
```

## Error Handling

- Return error info in observations, don't raise exceptions
- Use `done=True` with error observation for fatal errors
- Reserve exceptions for truly exceptional cases (server crashes)

```python
def step(self, action: MyAction) -> MyObservation:
    try:
        result = self._execute(action)
        return MyObservation(result=result, error=None, done=False)
    except InvalidAction as e:
        return MyObservation(result="", error=str(e), done=False)
    except FatalError as e:
        return MyObservation(result="", error=str(e), done=True)
```

## Reward Computation

Rewards are computed inside the environment, not externally:

```python
def step(self, action: MyAction) -> MyObservation:
    # Execute action
    new_state = self._apply_action(action)

    # Compute reward inside environment
    reward = self._compute_reward(new_state)

    return MyObservation(
        state=new_state,
        reward=reward,
        done=self._is_terminal(new_state)
    )
```

## FastAPI App Pattern

```python
# server/app.py
from openenv.core.env_server import create_app
from .my_environment import MyEnvironment
from ..models import MyAction, MyObservation

env = MyEnvironment()
app = create_app(env, MyAction, MyObservation)
```
