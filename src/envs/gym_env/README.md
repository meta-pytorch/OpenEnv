---
title: Gym Environment Server
emoji: 🎮
colorFrom: '#0E84B5'
colorTo: '#34D399'
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Gym Environment

Integration of OpenAI Gym/Gymnasium environments with the OpenEnv framework. Gymnasium provides a wide variety of environments for reinforcement learning research and development.

## Supported Environments

Gymnasium includes numerous environments across different categories:

### Classic Control
- **CartPole** - Balance a pole on a moving cart
- **Pendulum** - Swing up and balance an inverted pendulum
- **Acrobot** - Swing up a two-link robotic arm
- **MountainCar** - Drive up a mountain with limited power

### Box2D
- **LunarLander** - Land a spacecraft safely
- **BipedalWalker** - Train a 2D biped to walk
- **CarRacing** - Race a car around a track

And many more! For a complete list, see [Gymnasium documentation](https://gymnasium.farama.org/environments/classic_control/).

## Architecture

```
┌────────────────────────────────────┐
│ RL Training Code (Client)          │
│   GymEnv.step(action)              │
└──────────────┬─────────────────────┘
               │ HTTP
┌──────────────▼─────────────────────┐
│ FastAPI Server (Docker)            │
│   GymEnvironment                   │
│     ├─ Wraps Gymnasium Env         │
│     ├─ Handles observations        │
│     └─ Action execution            │
└────────────────────────────────────┘
```

## Installation & Usage

### Option 1: Local Development (without Docker)

**Requirements:**
- Python 3.11+
- gymnasium installed: `pip install gymnasium`

```python
# Connect to local server
from envs.gym_env import GymEnvironment, GymAction

# Start local server manually
# python -m envs.gym_env.server.app

env = GymEnvironment(base_url="http://0.0.0.0:8000")

# Reset environment
result = env.reset()
print(f"Observation : {result.observation.state}")
print(f"Action space: {result.observation.legal_actions}")

# Take actions
for _ in range(100):
    action = 1  # Example action
    result = env.step(GymAction(action=[action]))
    print(f"Reward: {result.reward}, Done: {result.done}")
    if result.done:
        break

# Cleanup
env.close()

```

### Option 2: Docker (Recommended)

**Build Gym image:**

```bash
cd OpenEnv

# Build the image
docker build \
  -f src/envs/gym_env/server/Dockerfile \
  -t gym-env:latest \
  .
```