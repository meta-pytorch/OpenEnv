# Maze Environment

Integration of Maze game with the OpenEnv framework.

## Architecture

```
┌────────────────────────────────────┐
│ RL Training Code (Client)          │
│   MazeEnv.step(action)             │
└──────────────┬─────────────────────┘
               │ HTTP
┌──────────────▼─────────────────────┐
│ FastAPI Server (Docker)            │
│   MazeEnvironment                  │
│     ├─ Wraps Maze environment      │
│     └─ Agent controls player       │
└────────────────────────────────────┘
```

## Installation & Usage

### Option 1: Local Development (without Docker)

**Requirements:**
- Python 3.11+
- Numpy

```python
from envs.maze_env import MazeEnv, MazeAction

# Start local server manually
# python -m envs.maze_env.server.app

# Connect to local server
env = MazeEnv(base_url="http://localhost:8000")

# Reset environment
result = env.reset()
print(f"Initial state: {result.observation.info_state}")
print(f"Legal actions: {result.observation.legal_actions}")

# Take actions
for _ in range(10):
    action_id = result.observation.legal_actions[0]  # Choose first legal action
    result = env.step(MazeAction(action_id=action_id))
    print(f"Reward: {result.reward}, Done: {result.done}")
    if result.done:
        break

# Cleanup
env.close()
```

### Option 2: Docker (Recommended)

**Build Docker image:**

```bash
cd OpenEnv
docker build -f src/envs/maze_env/server/Dockerfile -t maze-env:latest .
```

**Use with from_docker_image():**

```python
from envs.maze_env import MazeEnv, MazeAction

# Automatically starts container
env = MazeEnv.from_docker_image("maze-env:latest")

result = env.reset()
result = env.step(MazeAction(action_id=0))

env.close()  # Stops container
```

## Configuration

### Variables

- `maze` : Maze as a numpy array saved in mazearray.py

### Example

```bash
docker run -p 8000:8000 maze-env:latest
```

## API Reference

### MazeAction

```python
@dataclass
class MazeAction(Action):
    action: int                        # Action to be taken
```

### MazeObservation

```python
@dataclass
class MazeObservation(Observation):
    position: List[int]  # [row, col]
    total_reward: float  # Total reward
    legal_actions: List[int] = field(default_factory=list)  # Legal action based on the current position
```

### MazeState

```python
@dataclass
class MazeState(State):
    episode_id: str     # Episode
    step_count: int     # Number of steps
    done: bool = False  # Solve status

```

## References

- [Maze Environment](https://github.com/erikdelange/Reinforcement-Learning-Maze)
