---
title: Atari Environment Server
emoji: 🕹️
colorFrom: '#FF6200'
colorTo: '#D4151B'
sdk: docker
pinned: false
app_port: 8000
base_path: /web
---

# Atari Environment

Integration of Atari 2600 games with the OpenEnv framework via the Arcade Learning Environment (ALE). ALE provides access to 100+ classic Atari games for RL research.

## Supported Games

ALE supports 100+ Atari 2600 games including:

### Popular Games
- **Pong** - Classic two-player tennis
- **Breakout** - Break bricks with a ball
- **Space Invaders** - Shoot descending aliens
- **Pac-Man / Ms. Pac-Man** - Navigate mazes and eat pellets
- **Asteroids** - Destroy asteroids in space
- **Defender** - Side-scrolling space shooter
- **Centipede** - Shoot segmented centipede
- **Donkey Kong** - Jump over barrels to save princess
- **Frogger** - Cross road and river safely
- **Q*bert** - Jump on pyramid cubes

And many more! For a complete list, see [ALE documentation](https://ale.farama.org/environments/complete_list/).

## Architecture

```
┌────────────────────────────────────┐
│ RL Training Code (Client)          │
│   AtariEnv.step(action)            │
└──────────────┬─────────────────────┘
               │ HTTP
┌──────────────▼─────────────────────┐
│ FastAPI Server (Docker)            │
│   AtariEnvironment                 │
│     ├─ Wraps ALEInterface          │
│     ├─ Handles observations        │
│     └─ Action execution            │
└────────────────────────────────────┘
```

## Installation & Usage

### Option 1: Local Development (without Docker)

**Requirements:**
- Python 3.11+
- ale-py installed: `pip install ale-py`

```python
from envs.atari_env import AtariEnv, AtariAction

# Start local server manually
# python -m envs.atari_env.server.app

# Connect to local server
env = AtariEnv(base_url="http://localhost:8000")

# Reset environment
result = env.reset()
print(f"Screen shape: {result.observation.screen_shape}")
print(f"Legal actions: {result.observation.legal_actions}")
print(f"Lives: {result.observation.lives}")

# Take actions
for _ in range(10):
    action_id = 2  # UP action
    result = env.step(AtariAction(action_id=action_id, game_name="pong"))
    print(f"Reward: {result.reward}, Done: {result.done}")
    if result.done:
        break

# Cleanup
env.close()
```

### Option 2: Docker (Recommended)

**Build Atari image:**

```bash
cd OpenEnv

# Build the image
docker build \
  -f src/envs/atari_env/server/Dockerfile \
  -t atari-env:latest \
  .
```

**Run specific games:**

```bash
# Pong (default)
docker run -p 8000:8000 atari-env:latest

# Breakout
docker run -p 8000:8000 -e ATARI_GAME=breakout atari-env:latest

# Space Invaders with grayscale observation
docker run -p 8000:8000 \
  -e ATARI_GAME=space_invaders \
  -e ATARI_OBS_TYPE=grayscale \
  atari-env:latest

# Ms. Pac-Man with full action space
docker run -p 8000:8000 \
  -e ATARI_GAME=ms_pacman \
  -e ATARI_FULL_ACTION_SPACE=true \
  atari-env:latest
```

**Use with from_docker_image():**

```python
from envs.atari_env import AtariEnv, AtariAction
import numpy as np

# Automatically starts container
env = AtariEnv.from_docker_image("atari-env:latest")

result = env.reset()
result = env.step(AtariAction(action_id=2))  # UP

# Reshape screen for visualization
screen = np.array(result.observation.screen).reshape(result.observation.screen_shape)
print(f"Screen shape: {screen.shape}")  # (210, 160, 3) for RGB

env.close()  # Stops container
```

## Observation Types

### 1. RGB (Default)
- **Shape**: [210, 160, 3]
- **Description**: Full-color screen observation
- **Usage**: Most realistic, good for vision-based learning

```python
docker run -p 8000:8000 -e ATARI_OBS_TYPE=rgb atari-env:latest
```

### 2. Grayscale
- **Shape**: [210, 160]
- **Description**: Grayscale screen observation
- **Usage**: Reduced dimensionality, faster processing

```python
docker run -p 8000:8000 -e ATARI_OBS_TYPE=grayscale atari-env:latest
```

### 3. RAM
- **Shape**: [128]
- **Description**: Raw 128-byte Atari 2600 RAM contents
- **Usage**: Compact representation, useful for specific research

```python
docker run -p 8000:8000 -e ATARI_OBS_TYPE=ram atari-env:latest
```

## Action Spaces

### Minimal Action Set (Default)
Game-specific minimal actions (typically 4-9 actions).
- Pong: 6 actions (NOOP, FIRE, UP, DOWN, etc.)
- Breakout: 4 actions (NOOP, FIRE, LEFT, RIGHT)

```python
docker run -p 8000:8000 -e ATARI_FULL_ACTION_SPACE=false atari-env:latest
```

### Full Action Set
All 18 possible Atari 2600 actions:
0. NOOP
1. FIRE
2. UP
3. RIGHT
4. LEFT
5. DOWN
6. UPRIGHT
7. UPLEFT
8. DOWNRIGHT
9. DOWNLEFT
10. UPFIRE
11. RIGHTFIRE
12. LEFTFIRE
13. DOWNFIRE
14. UPRIGHTFIRE
15. UPLEFTFIRE
16. DOWNRIGHTFIRE
17. DOWNLEFTFIRE

```python
docker run -p 8000:8000 -e ATARI_FULL_ACTION_SPACE=true atari-env:latest
```

## Configuration

### Environment Variables

- `ATARI_GAME`: Game name (default: "pong")
- `ATARI_OBS_TYPE`: Observation type - "rgb", "grayscale", "ram" (default: "rgb")
- `ATARI_FULL_ACTION_SPACE`: Use full action space - "true"/"false" (default: "false")
- `ATARI_MODE`: Game mode (optional, game-specific)
- `ATARI_DIFFICULTY`: Game difficulty (optional, game-specific)
- `ATARI_REPEAT_ACTION_PROB`: Sticky action probability 0.0-1.0 (default: "0.0")
- `ATARI_FRAMESKIP`: Frames to skip per action (default: "4")

### Example: Breakout with Custom Settings

```bash
docker run -p 8000:8000 \
  -e ATARI_GAME=breakout \
  -e ATARI_OBS_TYPE=grayscale \
  -e ATARI_FULL_ACTION_SPACE=true \
  -e ATARI_REPEAT_ACTION_PROB=0.25 \
  -e ATARI_FRAMESKIP=4 \
  atari-env:latest
```

## API Reference

### AtariAction

```python
@dataclass
class AtariAction(Action):
    action_id: int                  # Action index to execute
    game_name: str = "pong"         # Game name
    obs_type: str = "rgb"           # Observation type
    full_action_space: bool = False # Full or minimal action space
```

### AtariObservation

```python
@dataclass
class AtariObservation(Observation):
    screen: List[int]               # Flattened screen pixels
    screen_shape: List[int]         # Original screen shape
    legal_actions: List[int]        # Legal action indices
    lives: int                      # Lives remaining
    episode_frame_number: int       # Frame # in episode
    frame_number: int               # Total frame #
    done: bool                      # Episode finished
    reward: Optional[float]         # Reward from last action
```

### AtariState

```python
@dataclass
class AtariState(State):
    episode_id: str                      # Unique episode ID
    step_count: int                      # Number of steps
    game_name: str                       # Game name
    obs_type: str                        # Observation type
    full_action_space: bool              # Action space type
    mode: Optional[int]                  # Game mode
    difficulty: Optional[int]            # Game difficulty
    repeat_action_probability: float     # Sticky action prob
    frameskip: int                       # Frameskip setting
```

## Example Script

```python
#!/usr/bin/env python3
"""Example training loop with Atari environment."""

import numpy as np
from envs.atari_env import AtariEnv, AtariAction

# Start environment
env = AtariEnv.from_docker_image("atari-env:latest")

# Training loop
for episode in range(10):
    result = env.reset()
    episode_reward = 0
    steps = 0

    while not result.done:
        # Random policy (replace with your RL agent)
        action_id = np.random.choice(result.observation.legal_actions)

        # Take action
        result = env.step(AtariAction(action_id=action_id))

        episode_reward += result.reward or 0
        steps += 1

        # Reshape screen for processing
        screen = np.array(result.observation.screen).reshape(
            result.observation.screen_shape
        )

        # Your RL training code here
        # ...

    print(f"Episode {episode}: reward={episode_reward:.2f}, steps={steps}")

env.close()
```

## Testing

### Local Testing

```bash
# Install dependencies
pip install ale-py fastapi uvicorn requests

# Start server
cd /Users/sanyambhutani/OpenEnv/OpenEnv
export PYTHONPATH=/Users/sanyambhutani/OpenEnv/OpenEnv/src
python -m envs.atari_env.server.app

# Test from another terminal
python -c "
from envs.atari_env import AtariEnv, AtariAction
env = AtariEnv(base_url='http://localhost:8000')
result = env.reset()
print(f'Initial obs: {result.observation.screen_shape}')
result = env.step(AtariAction(action_id=2))
print(f'After step: reward={result.reward}, done={result.done}')
env.close()
"
```

### Docker Testing

```bash
# Build and run
docker build -f src/envs/atari_env/server/Dockerfile -t atari-env:latest .
docker run -p 8000:8000 atari-env:latest

# Test in another terminal
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset
```

## Popular Games and Their Characteristics

| Game | Minimal Actions | Lives | Difficulty | Notes |
|------|----------------|-------|-----------|-------|
| Pong | 6 | 1 | Low | Good for learning basics |
| Breakout | 4 | 5 | Medium | Classic RL benchmark |
| Space Invaders | 6 | 3 | Medium | Shooting game |
| Ms. Pac-Man | 9 | 3 | High | Complex navigation |
| Asteroids | 14 | 3 | Medium | Continuous shooting |
| Montezuma's Revenge | 18 | 5 | Very High | Exploration challenge |
| Pitfall | 18 | 1 | High | Platformer |
| Seaquest | 18 | 3 | High | Submarine rescue |

## Limitations & Notes

- **Frame perfect timing**: Some games require precise timing
- **Exploration**: Games like Montezuma's Revenge are notoriously difficult
- **Observation delay**: HTTP adds minimal latency vs local gym
- **Determinism**: Set `ATARI_REPEAT_ACTION_PROB=0.0` for deterministic behavior
- **ROMs**: All ROMs are bundled with ale-py package

## References

- [Arcade Learning Environment Paper (2013)](https://jair.org/index.php/jair/article/view/10819)
- [ALE GitHub](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
- [ALE Documentation](https://ale.farama.org/)
- [Gymnasium Atari Environments](https://gymnasium.farama.org/environments/atari/)

## Citation

If you use ALE in your research, please cite:

```bibtex
@Article{bellemare13arcade,
    author = {{Bellemare}, M.~G. and {Naddaf}, Y. and {Veness}, J. and {Bowling}, M.},
    title = {The Arcade Learning Environment: An Evaluation Platform for General Agents},
    journal = {Journal of Artificial Intelligence Research},
    year = "2013",
    month = "jun",
    volume = "47",
    pages = "253--279",
}
```
