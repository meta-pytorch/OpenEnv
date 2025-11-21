---
title: Doom Environment Server
emoji: 🎮
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - vizdoom
  - reinforcement-learning
---

# Doom Environment

A ViZDoom-based environment for OpenEnv. ViZDoom is a Doom-based AI research platform for visual reinforcement learning, allowing agents to play Doom using only visual information.

## Overview

This environment wraps ViZDoom scenarios and exposes them through the OpenEnv API. It provides:
- **Visual observations**: RGB or grayscale screen buffers
- **Game variables**: Health, ammo, kills, etc.
- **Flexible action space**: Discrete actions or button combinations
- **Multiple scenarios**: Built-in scenarios like "basic", "deadly_corridor", "defend_the_center", etc.

## Quick Start

The simplest way to use the Doom environment is through the `DoomEnv` class:

```python
from doom_env import DoomAction, DoomEnv

try:
    # Create environment from Docker image
    doom_env = DoomEnv.from_docker_image("doom-env:latest")

    # Reset to start a new episode
    result = doom_env.reset()
    print(f"Screen shape: {result.observation.screen_shape}")
    print(f"Available actions: {result.observation.available_actions}")

    # Take actions
    for i in range(100):
        # Use discrete action (e.g., 0=no-op, 1-7=various actions)
        result = doom_env.step(DoomAction(action_id=1))

        print(f"Step {i}:")
        print(f"  Reward: {result.reward}")
        print(f"  Done: {result.observation.done}")
        print(f"  Game variables: {result.observation.game_variables}")

        if result.observation.done:
            print("Episode finished!")
            break

finally:
    # Always clean up
    doom_env.close()
```

That's it! The `DoomEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From the doom_env directory
cd src/envs/doom_env
docker build -t doom-env:latest -f server/Dockerfile .

# Or from project root
docker build -t doom-env:latest -f src/envs/doom_env/server/Dockerfile src/envs/doom_env
```

## Environment Details

### Action Space

Actions can be specified in two ways:

1. **Discrete Actions** (recommended for most use cases):
   ```python
   DoomAction(action_id=2)  # Single integer action
   ```

   Available discrete actions (depends on scenario):
   - `0`: No-op (do nothing)
   - `1-N`: Various single button presses (move left, right, shoot, etc.)

2. **Button Combinations**:
   ```python
   DoomAction(buttons=[1, 0, 1, 0])  # Press specific buttons
   ```

   Each element is 0 (not pressed) or 1 (pressed).

### Observation Space

**DoomObservation** contains:
- `screen_buffer` (List[int]): Flattened screen pixels
  - RGB: Shape [height, width, 3] before flattening
  - Grayscale: Shape [height, width] before flattening
- `screen_shape` (List[int]): Original shape of the screen
- `game_variables` (List[float]): Health, ammo, kills, etc.
- `available_actions` (List[int]): Valid action IDs
- `episode_finished` (bool): Whether episode has ended
- `reward` (float): Reward from last action
- `done` (bool): Same as episode_finished
- `metadata` (dict): Additional info (scenario name, available buttons)

### Scenarios

ViZDoom comes with several built-in scenarios:

- **basic**: Simple scenario to learn basic movement and shooting
- **deadly_corridor**: Navigate a corridor while avoiding/killing monsters
- **defend_the_center**: Stay alive as long as possible in the center
- **defend_the_line**: Defend a line against incoming monsters
- **health_gathering**: Collect health packs to survive
- **my_way_home**: Navigate to a specific location
- **predict_position**: Predict where an object will be
- **take_cover**: Learn to take cover from enemy fire

## Advanced Usage

### Custom Configuration

You can customize the environment when creating the server:

```python
from doom_env.server.doom_env_environment import DoomEnvironment

# Create with custom settings
env = DoomEnvironment(
    scenario="deadly_corridor",
    screen_resolution="RES_320X240",  # Higher resolution
    screen_format="GRAY8",             # Grayscale instead of RGB
    window_visible=True,               # Show game window
    use_discrete_actions=True          # Use discrete action space
)
```

### Connecting to an Existing Server

If you already have a Doom environment server running:

```python
from doom_env import DoomEnv

# Connect to existing server
doom_env = DoomEnv(base_url="http://localhost:8000")

# Use as normal
result = doom_env.reset()
result = doom_env.step(DoomAction(action_id=1))
```

Note: When connecting to an existing server, `doom_env.close()` will NOT stop the server.

### Processing Visual Observations

The screen buffer is flattened for JSON serialization. To use it:

```python
import numpy as np

result = doom_env.reset()
obs = result.observation

# Reshape to original dimensions
screen = np.array(obs.screen_buffer).reshape(obs.screen_shape)

# screen is now a numpy array with shape [height, width, channels]
# You can visualize it, pass to a neural network, etc.
```

## Development & Testing

### Local Development

Install dependencies and run locally without Docker:

```bash
# Install the environment in development mode
cd src/envs/doom_env
uv pip install -e .

# Or using pip
pip install -e .

# Run the server locally
uv run server --host 0.0.0.0 --port 8000

# Or using uvicorn directly
uvicorn server.app:app --reload
```

### Testing

Test the environment logic directly:

```bash
# From the doom_env directory
python3 -c "
from server.doom_env_environment import DoomEnvironment
from models import DoomAction

env = DoomEnvironment(scenario='basic')
obs = env.reset()
print(f'Initial observation shape: {obs.screen_shape}')

for i in range(10):
    obs = env.step(DoomAction(action_id=1))
    print(f'Step {i}: reward={obs.reward}, done={obs.done}')
"
```

## Deploying to Hugging Face Spaces

Deploy your Doom environment to Hugging Face Spaces:

```bash
# From the doom_env directory
openenv push

# Or specify options
openenv push --repo-id my-org/doom-env --private
```

The `openenv push` command will:
1. Validate the environment setup
2. Prepare for Hugging Face Docker space
3. Upload to Hugging Face

After deployment, your space will include:
- **Web Interface** at `/web` - Interactive UI
- **API Documentation** at `/docs` - OpenAPI/Swagger
- **Health Check** at `/health` - Monitoring

## Project Structure

```
doom_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports (DoomAction, DoomObservation, DoomEnv)
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Dependencies (vizdoom, numpy, etc.)
├── uv.lock                # Locked dependencies
├── client.py              # DoomEnv HTTP client
├── models.py              # DoomAction and DoomObservation dataclasses
└── server/
    ├── __init__.py        # Server module exports
    ├── doom_env_environment.py  # Core ViZDoom wrapper
    ├── app.py             # FastAPI application
    └── Dockerfile         # Container with ViZDoom dependencies
```

## Dependencies

- **ViZDoom**: Doom-based AI research platform
- **NumPy**: Array operations for screen buffers
- **OpenEnv Core**: Base framework
- **FastAPI/Uvicorn**: HTTP server
- **System libraries**: SDL2, Boost, OpenGL, etc. (handled in Dockerfile)

## Troubleshooting

### ViZDoom Installation Issues

If you encounter issues installing ViZDoom:

```bash
# Make sure you have system dependencies (Ubuntu/Debian)
sudo apt-get install cmake libboost-all-dev libsdl2-dev libfreetype6-dev

# Then install ViZDoom
pip install vizdoom
```

### Docker Build Issues

If Docker build fails with ViZDoom dependencies:
- Ensure you have sufficient disk space
- Check that the base image is accessible
- Verify system dependencies in Dockerfile

### Runtime Errors

- **"Could not load scenario"**: Check scenario name or path
- **"Invalid action_id"**: Ensure action_id is within valid range
- **Screen buffer issues**: Verify screen format and resolution settings

## References

- [ViZDoom Documentation](http://vizdoom.cs.put.edu.pl/)
- [ViZDoom GitHub](https://github.com/mwydmuch/ViZDoom)
- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)

## License

BSD 3-Clause License (see LICENSE file in repository root)
