---
title: Warehouse Env Environment Server
emoji: 🏭
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /demo
tags:
  - openenv
  - reinforcement-learning
  - logistics
  - warehouse
  - robotics
---

# Warehouse Optimization Environment

A grid-based warehouse logistics optimization environment for reinforcement learning. This environment simulates a warehouse robot that must navigate through obstacles, pick up packages from pickup zones, and deliver them to designated dropoff zones while optimizing for time and efficiency.

## Overview

The Warehouse Environment is designed for training reinforcement learning agents on logistics and pathfinding tasks. It features:

- **Grid-based navigation** with walls and obstacles
- **Package pickup and delivery** mechanics
- **Multi-objective optimization** (speed, deliveries, efficiency)
- **Scalable difficulty** levels (1-5)
- **Dense reward signals** for effective learning
- **ASCII visualization** for debugging

## Quick Start

### Using Docker (Recommended)

```bash
# Build the Docker image (from OpenEnv root)
cd /path/to/OpenEnv
docker build -f src/envs/warehouse_env/server/Dockerfile -t warehouse-env:latest .

# Run with default settings (difficulty level 2)
docker run -p 8000:8000 warehouse-env:latest

# Run with custom difficulty
docker run -p 8000:8000 -e DIFFICULTY_LEVEL=3 warehouse-env:latest
```

### Using Python Client

```python
from envs.warehouse_env import WarehouseEnv, WarehouseAction

# Connect to server (or start from Docker)
env = WarehouseEnv.from_docker_image(
    "warehouse-env:latest",
    environment={"DIFFICULTY_LEVEL": "2"}
)

# Reset environment
result = env.reset()
print(f"Warehouse size: {len(result.observation.grid)}x{len(result.observation.grid[0])}")
print(f"Packages to deliver: {result.observation.total_packages}")

# Run episode
done = False
while not done:
    # Simple policy: move toward pickup if not carrying, else toward dropoff
    if result.observation.robot_carrying is None:
        action = WarehouseAction(action_id=4)  # Try to pick up
    else:
        action = WarehouseAction(action_id=5)  # Try to drop off

    result = env.step(action)
    print(f"Step {result.observation.step_count}: {result.observation.message}")
    print(f"Reward: {result.reward:.2f}")

    done = result.done

print(f"\nEpisode finished!")
print(f"Delivered: {result.observation.packages_delivered}/{result.observation.total_packages}")
print(f"Total reward: {env.state().cum_reward:.2f}")

env.close()
```

## Environment Specification

### State Space

The environment provides rich observations including:

- **Grid layout**: 2D array with cell types (empty, wall, shelf, pickup zone, dropoff zone)
- **Robot state**: Position, carrying status
- **Package information**: Locations, status (waiting/picked/delivered), priorities
- **Episode metrics**: Step count, deliveries, time remaining

### Action Space

6 discrete actions:

| Action ID | Action Name | Description |
|-----------|-------------|-------------|
| 0 | MOVE_UP | Move robot one cell up |
| 1 | MOVE_DOWN | Move robot one cell down |
| 2 | MOVE_LEFT | Move robot one cell left |
| 3 | MOVE_RIGHT | Move robot one cell right |
| 4 | PICK_UP | Pick up package at current location |
| 5 | DROP_OFF | Drop off package at current location |

### Reward Structure

Multi-component reward function:

- **+100**: Successful package delivery
- **+10**: Successful package pickup
- **+0.1 × time_remaining**: Time bonus for fast deliveries
- **+200**: Completion bonus (all packages delivered)
- **-0.1**: Small step penalty (encourages efficiency)
- **-1**: Invalid action penalty

### Episode Termination

Episodes end when:
- All packages are delivered (success!)
- Maximum steps reached (timeout)

## Difficulty Levels

### Level 1: Simple
- Grid: 5×5
- Packages: 1
- Obstacles: 0
- Max steps: 50
- **Best for**: Testing, debugging, quick validation

### Level 2: Easy (Default)
- Grid: 8×8
- Packages: 2
- Obstacles: 3
- Max steps: 100
- **Best for**: Initial training, curriculum learning start

### Level 3: Medium
- Grid: 10×10
- Packages: 3
- Obstacles: 8
- Max steps: 150
- **Best for**: Intermediate training, testing learned policies

### Level 4: Hard
- Grid: 15×15
- Packages: 5
- Obstacles: 20
- Max steps: 250
- **Best for**: Advanced training, evaluation

### Level 5: Expert
- Grid: 20×20
- Packages: 8
- Obstacles: 40
- Max steps: 400
- **Best for**: Final evaluation, research benchmarks

## Configuration

### Environment Variables

Configure the warehouse via environment variables:

```bash
# Difficulty level (1-5)
DIFFICULTY_LEVEL=2

# Custom grid size (overrides difficulty)
GRID_WIDTH=12
GRID_HEIGHT=12

# Custom package count (overrides difficulty)
NUM_PACKAGES=4

# Custom step limit (overrides difficulty)
MAX_STEPS=200

# Random seed for reproducibility
RANDOM_SEED=42
```

### Docker Example

```bash
docker run -p 8000:8000 \
  -e DIFFICULTY_LEVEL=3 \
  -e RANDOM_SEED=42 \
  warehouse-env:latest
```

### Python Client Example

```python
env = WarehouseEnv.from_docker_image(
    "warehouse-env:latest",
    environment={
        "DIFFICULTY_LEVEL": "3",
        "GRID_WIDTH": "12",
        "GRID_HEIGHT": "12",
        "NUM_PACKAGES": "4",
        "MAX_STEPS": "200",
        "RANDOM_SEED": "42"
    }
)
```

## Visualization

### ASCII Rendering

Get a visual representation of the warehouse state:

```python
# Get ASCII visualization
ascii_art = env.render_ascii()
print(ascii_art)
```

Example output:
```
=================================
Step: 15/100 | Delivered: 1/2 | Reward: 109.9
=================================
█ █ █ █ █ █ █ █
█ P . . . # . █
█ . # . . . . █
█ . . R . # . █
█ . # . . . . █
█ . . . . D . █
█ . . . . . . █
█ █ █ █ █ █ █ █
=================================
Robot at (3, 3), carrying: 1
✓ Package #0: delivered (P(1,1)→D(5,5))
↻ Package #1: picked (P(1,1)→D(5,5))
=================================
Legend: r/R=Robot(empty/carrying), P=Pickup, D=Dropoff, #=Shelf, █=Wall
```

## Training Examples

### Random Agent

```python
import random
from envs.warehouse_env import WarehouseEnv, WarehouseAction

env = WarehouseEnv.from_docker_image("warehouse-env:latest")

for episode in range(100):
    result = env.reset()
    done = False

    while not done:
        # Random action
        action = WarehouseAction(action_id=random.randint(0, 5))
        result = env.step(action)
        done = result.done

    print(f"Episode {episode}: Delivered {result.observation.packages_delivered}")

env.close()
```

### Greedy Agent (Move toward target)

```python
from envs.warehouse_env import WarehouseEnv, WarehouseAction

def get_greedy_action(obs):
    """Simple greedy policy: move toward nearest target."""
    robot_x, robot_y = obs.robot_position

    # If not carrying, move toward nearest waiting package
    if obs.robot_carrying is None:
        for pkg in obs.packages:
            if pkg["status"] == "waiting":
                target_x, target_y = pkg["pickup_location"]
                break
        else:
            return 4  # Try to pick up if at location
    else:
        # Move toward dropoff zone
        pkg = next(p for p in obs.packages if p["id"] == obs.robot_carrying)
        target_x, target_y = pkg["dropoff_location"]

    # Simple pathfinding: move closer on one axis
    if robot_x < target_x:
        return 3  # RIGHT
    elif robot_x > target_x:
        return 2  # LEFT
    elif robot_y < target_y:
        return 1  # DOWN
    elif robot_y > target_y:
        return 0  # UP
    else:
        # At target location
        return 4 if obs.robot_carrying is None else 5

env = WarehouseEnv.from_docker_image("warehouse-env:latest")

for episode in range(10):
    result = env.reset()
    done = False

    while not done:
        action_id = get_greedy_action(result.observation)
        action = WarehouseAction(action_id=action_id)
        result = env.step(action)
        done = result.done

    state = env.state()
    print(f"Episode {episode}: {state.packages_delivered}/{state.total_packages} delivered, "
          f"reward: {state.cum_reward:.2f}")

env.close()
```

### Integration with RL Libraries

#### Stable Baselines 3

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from envs.warehouse_env import WarehouseEnv, WarehouseAction

class WarehouseGymWrapper(gym.Env):
    """Gymnasium wrapper for Warehouse environment."""

    def __init__(self, base_url="http://localhost:8000"):
        super().__init__()
        self.env = WarehouseEnv(base_url=base_url)

        # Define spaces (simplified)
        self.action_space = gym.spaces.Discrete(6)

        # Observation: grid + robot state + package info
        # For simplicity, use flattened representation
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(200,),  # Adjust based on grid size
            dtype=np.float32
        )

    def reset(self, **kwargs):
        result = self.env.reset()
        obs = self._process_obs(result.observation)
        return obs, {}

    def step(self, action):
        result = self.env.step(WarehouseAction(action_id=int(action)))
        obs = self._process_obs(result.observation)
        return obs, result.reward, result.done, False, {}

    def _process_obs(self, observation):
        # Flatten grid and add robot/package info
        grid_flat = np.array(observation.grid).flatten()
        robot_pos = np.array(observation.robot_position)
        carrying = np.array([1 if observation.robot_carrying else 0])

        # Pad or truncate to fixed size
        obs = np.concatenate([
            grid_flat[:196],  # Grid (max 14x14)
            robot_pos,        # Robot position (2)
            carrying,         # Carrying status (1)
            [observation.packages_delivered]  # Progress (1)
        ])
        return obs.astype(np.float32)

    def close(self):
        self.env.close()

# Train with PPO
env = WarehouseGymWrapper()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("warehouse_ppo")

env.close()
```

## API Reference

### WarehouseAction

```python
@dataclass
class WarehouseAction(Action):
    action_id: int  # 0-5
```

### WarehouseObservation

```python
@dataclass
class WarehouseObservation(Observation):
    grid: List[List[int]]              # Warehouse layout
    robot_position: tuple[int, int]    # Robot (x, y)
    robot_carrying: Optional[int]      # Package ID or None
    packages: List[Dict[str, Any]]     # Package states
    step_count: int                    # Current step
    packages_delivered: int            # Successful deliveries
    total_packages: int                # Total packages
    time_remaining: int                # Steps left
    action_success: bool               # Last action valid
    message: str                       # Status message
```

### WarehouseState

```python
@dataclass
class WarehouseState(State):
    episode_id: str                    # Unique episode ID
    step_count: int                    # Steps taken
    packages_delivered: int            # Deliveries
    total_packages: int                # Total packages
    difficulty_level: int              # Difficulty (1-5)
    grid_size: tuple[int, int]         # Grid dimensions
    cum_reward: float                  # Cumulative reward
    is_done: bool                      # Episode finished
```

## Development

### Local Setup (without Docker)

```bash
# Install dependencies
cd OpenEnv/src/envs/warehouse_env
pip install -r server/requirements.txt

# Run server
python -m uvicorn envs.warehouse_env.server.app:app --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Run basic test
python examples/warehouse_simple.py
```

## Architecture

```
┌─────────────────────────────────────┐
│  RL Training Framework (Client)     │
│  ┌──────────────────────────────┐   │
│  │ Agent Policy (PPO/DQN/etc)   │   │
│  └──────────┬───────────────────┘   │
│             │                        │
│  ┌──────────▼───────────────────┐   │
│  │ WarehouseEnv (HTTPEnvClient) │   │
│  └──────────┬───────────────────┘   │
└─────────────┼───────────────────────┘
              │ HTTP/JSON
┌─────────────▼───────────────────────┐
│  Docker Container                   │
│  ┌─────────────────────────────┐    │
│  │ FastAPI Server              │    │
│  └──────────┬──────────────────┘    │
│  ┌──────────▼──────────────────┐    │
│  │ WarehouseEnvironment        │    │
│  │ - Grid generation           │    │
│  │ - Collision detection       │    │
│  │ - Reward calculation        │    │
│  │ - Package management        │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

## Real-World Applications

This environment simulates real warehouse optimization problems:

- **Amazon fulfillment centers**: Robot pathfinding and package routing
- **Manufacturing warehouses**: Material handling optimization
- **Distribution centers**: Inventory management and delivery sequencing
- **Automated storage**: Efficient retrieval systems

## Research & Benchmarking

The warehouse environment is suitable for research on:

- **Pathfinding algorithms**: A*, Dijkstra, learned policies
- **Multi-objective RL**: Balancing speed, safety, and coverage
- **Curriculum learning**: Progressive difficulty scaling
- **Transfer learning**: Generalization across warehouse layouts
- **Hierarchical RL**: High-level planning + low-level control

## Contributing

We welcome contributions! Areas for enhancement:

- **Multi-robot coordination**: Multiple robots working together
- **Dynamic obstacles**: Moving shelves or other robots
- **Battery management**: Energy constraints and charging stations
- **Priority queuing**: Handling different package urgencies
- **3D visualization**: Enhanced rendering

## License

BSD 3-Clause License (see LICENSE file)

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{warehouse_env_openenv,
  title = {Warehouse Optimization Environment for OpenEnv},
  author = {OpenEnv Contributors},
  year = {2024},
  url = {https://github.com/meta-pytorch/OpenEnv}
}
```

## References

- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
- [Gymnasium API](https://gymnasium.farama.org/)
- [Warehouse Robotics Research](https://arxiv.org/abs/2006.14876)
