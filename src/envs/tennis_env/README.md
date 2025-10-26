# Tennis Environment

Specialized Atari Tennis environment for strategic two-player gameplay with LLMs. Extends `atari_env` with symbolic state extraction (ball position, opponent position, rally length), dynamic reward shaping, and simulated multi-agent capabilities for self-play and tournament scenarios.

## Why Tennis Models?

OpenEnv's standard `atari_env` provides basic Atari game support but cannot accommodate strategic two-player games that require:

- **Opponent modeling** - Track and reason about opponent position/behavior
- **Rally tracking** - Monitor and reward sustained gameplay (consecutive hits)
- **Ball tracking** - Symbolic ball position for strategic positioning
- **Perspective switching** - Enable one model to play both sides (self-play)
- **Dynamic rewards** - Configurable reward shaping for experimentation

`tennis_env` adds custom models (`TennisAction`, `TennisObservation`, `TennisState`) and environment logic specifically for these capabilities:

| Feature | atari_env | tennis_env |
|---------|-----------|------------|
| **Opponent position** | ❌ Not tracked | ✅ `opponent_position` field |
| **Rally tracking** | ❌ Not available | ✅ `rally_length` field |
| **Ball tracking** | ❌ Not tracked | ✅ `ball_side` field |
| **Score tracking** | ❌ Basic | ✅ `(agent, opp)` tuple |
| **Dynamic rewards** | ❌ Static | ✅ 7 configurable parameters |
| **Simulated multi-agent** | ❌ No | ✅ Via perspective switching |

## Installation

```bash
# Install dependencies
pip install ale-py gymnasium fastapi uvicorn requests

# Or with uv
uv pip install ale-py gymnasium fastapi uvicorn requests
```

## Quick Start

### Local Environment

```python
from envs.tennis_env.server.tennis_environment import TennisEnvironment
from envs.tennis_env.models import TennisAction

# Create environment with dynamic rewards
env = TennisEnvironment(
    score_reward=15.0,        # Reward for scoring
    rally_bonus_scale=0.2,    # Emphasize rallies
)

# Reset and get initial observation
obs = env.reset()
print(f"Score: {obs.score}")
print(f"Ball: {obs.ball_side}")
print(f"My position: {obs.my_position}")
print(f"Opponent position: {obs.opponent_position}")
print(f"Rally length: {obs.rally_length}")

# Take actions
action = TennisAction(action_id=2, action_name="UP")
obs = env.step(action)
print(f"Reward: {obs.reward}")
```

### HTTP Client-Server

**Start server:**

```bash
# From OpenEnv root directory
export TENNIS_SCORE_REWARD=20.0
export TENNIS_RALLY_BONUS_SCALE=0.3

python -m uvicorn envs.tennis_env.server.app:app --host 0.0.0.0 --port 8000
```

**Connect client:**

```python
from envs.tennis_env import TennisEnv, TennisAction

# Connect to server
client = TennisEnv(base_url="http://localhost:8000")

# Use same interface as local environment
result = client.reset()
result = client.step(TennisAction(action_id=2))

print(f"Opponent at: {result.observation.opponent_position}")
print(f"Rally length: {result.observation.rally_length}")
```

## Observation Space

`TennisObservation` extends `atari_env` observations with strategic features:

| Field | Type | Description | New in tennis_env? |
|-------|------|-------------|-------------------|
| `screen_rgb` | `List[int]` | Flattened RGB (210×160×3) | No (standard) |
| `screen_shape` | `List[int]` | Shape `[210, 160, 3]` | No (standard) |
| `score` | `Tuple[int, int]` | **(agent, opponent) scores** | **Yes** |
| `ball_side` | `str` | `"left"/"center"/"right"/"unknown"` | **Yes** |
| `my_position` | `str` | `"top"/"middle"/"bottom"/"unknown"` | **Yes** |
| `opponent_position` | `str` | `"top"/"middle"/"bottom"/"unknown"` | **Yes** |
| `rally_length` | `int` | **Consecutive hits in rally** | **Yes** |
| `legal_actions` | `List[int]` | Available actions | No (standard) |
| `action_meanings` | `List[str]` | Human-readable names | **Yes** |
| `lives` | `int` | Lives remaining | No (standard) |
| `done` | `bool` | Episode ended | No (standard) |
| `reward` | `float` | **Shaped reward** | Enhanced |

## Action Space

18 discrete actions (full Atari action space):

| ID | Action | ID | Action | ID | Action |
|----|--------|----|--------|----|--------|
| 0 | NOOP | 6 | UPRIGHT | 12 | LEFTFIRE |
| 1 | FIRE | 7 | UPLEFT | 13 | DOWNFIRE |
| 2 | UP | 8 | DOWNRIGHT | 14 | UPRIGHTFIRE |
| 3 | RIGHT | 9 | DOWNLEFT | 15 | UPLEFTFIRE |
| 4 | LEFT | 10 | UPFIRE | 16 | DOWNRIGHTFIRE |
| 5 | DOWN | 11 | RIGHTFIRE | 17 | DOWNLEFTFIRE |

## Dynamic Reward Shaping

### Configurable Parameters

Tennis_env supports **7 dynamic reward parameters** for RL experimentation:

```python
env = TennisEnvironment(
    score_reward=10.0,        # Reward for scoring a point
    score_penalty=-5.0,       # Penalty for opponent scoring
    rally_bonus_max=1.0,      # Maximum rally bonus
    rally_bonus_scale=0.1,    # Rally bonus scale factor
    movement_bonus=0.05,      # Reward for active movement
    positioning_bonus=0.1,    # Reward for good positioning
    center_bonus=0.2,         # Reward for center court control
)
```

### Environment Variables

Configure via environment variables before starting server:

| Variable | Default | Description |
|----------|---------|-------------|
| `TENNIS_SCORE_REWARD` | `10.0` | Reward for scoring |
| `TENNIS_SCORE_PENALTY` | `-5.0` | Penalty for opponent scoring |
| `TENNIS_RALLY_BONUS_MAX` | `1.0` | Max rally bonus |
| `TENNIS_RALLY_BONUS_SCALE` | `0.1` | Rally scale factor |
| `TENNIS_MOVEMENT_BONUS` | `0.05` | Movement reward |
| `TENNIS_POSITIONING_BONUS` | `0.1` | Positioning reward |
| `TENNIS_CENTER_BONUS` | `0.2` | Center court reward |

## Simulated Multi-Agent Capabilities

Tennis_env enables **simulated multi-agent** scenarios via symbolic state and perspective switching:

### 1. Self-Play (One Model, Two Sides)

```python
# Orange player prompt
prompt_orange = f"""
You are the ORANGE player on the LEFT.
Ball is on {obs.ball_side}, you're at {obs.my_position}, opponent at {obs.opponent_position}
"""

# Blue player prompt (flipped perspective)
prompt_blue = f"""
You are the BLUE player on the RIGHT.
Ball is on {obs.ball_side}, you're at {obs.opponent_position}, opponent at {obs.my_position}
"""

# One model learns both sides!
```

### 2. Tournament Play (Two Models Compete)

```python
if obs.ball_side == "left":
    action = model_3b.generate(prompt_orange)   # 3B controls orange
else:
    action = model_70b.generate(prompt_blue)    # 70B controls blue
```

### 3. Opponent Modeling

The symbolic state enables strategic reasoning:

```python
def strategy(ball_side, my_pos, opp_pos):
    # Exploit opponent position
    if opp_pos == "top" and ball_side == "right":
        return 5  # DOWN - hit to bottom where opponent isn't
    elif opp_pos == "bottom" and my_pos == "middle":
        return 2  # UP - move to cover court
    return 0      # NOOP
```

**Note**: This is simulated multi-agent (via symbolic state + perspective switching), not true multi-agent like PettingZoo.

## Docker Support

### Build Image

```bash
docker build -f src/envs/tennis_env/server/Dockerfile -t tennis-env:latest .
```

### Run Container with Custom Rewards

```bash
docker run -p 8000:8000 \
  -e TENNIS_SCORE_REWARD=15.0 \
  -e TENNIS_RALLY_BONUS_SCALE=0.2 \
  -e TENNIS_MOVEMENT_BONUS=0.1 \
  tennis-env:latest
```

### Test Docker Setup

```bash
# Comprehensive test with dynamic reward validation
./src/envs/tennis_env/test_tennis_docker.sh
```

**Tests performed:**
- ✅ Image build
- ✅ Container startup
- ✅ Health endpoint
- ✅ Reset with symbolic features
- ✅ Step endpoint
- ✅ Dynamic reward configuration
- ✅ Error checking

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TENNIS_MODE` | `None` | Game mode (0-2) |
| `TENNIS_DIFFICULTY` | `None` | Difficulty (0-3) |
| `TENNIS_REPEAT_ACTION_PROB` | `0.25` | Sticky action probability |
| `TENNIS_FRAMESKIP` | `4` | Frames to skip per action |
| **Dynamic Rewards** | | **See Dynamic Reward Shaping section** |

## Testing

Run the consolidated test suite:

```bash
# All tests (12 tests covering models, environment, integration)
pytest src/envs/tennis_env/tests/test_tennis_env.py -v

# Specific test categories
pytest src/envs/tennis_env/tests/test_tennis_env.py::test_dynamic_reward_configuration -v
```

**Test coverage:**
- ✅ 3 model tests (Action, Observation, State)
- ✅ 6 environment tests (Init, reset, step, symbolic features, dynamic rewards, reward shaping)
- ✅ 3 integration tests (HTTP client-server, full episodes)

## Architecture

```
src/envs/tennis_env/
├── __init__.py                     # Package exports
├── models.py                       # TennisAction, TennisObservation, TennisState
├── client.py                       # TennisEnv HTTP client
├── server/
│   ├── __init__.py
│   ├── app.py                      # FastAPI server with dynamic rewards
│   ├── tennis_environment.py       # Core env with symbolic extraction
│   └── Dockerfile                  # Container build
├── tests/
│   ├── __init__.py
│   └── test_tennis_env.py          # Consolidated test suite (12 tests)
├── test_tennis_docker.sh           # Docker test script
└── README.md                       # This file
```

## API Reference

### TennisAction

```python
@dataclass
class TennisAction(Action):
    action_id: int
    action_name: Optional[str] = None
```

### TennisObservation

```python
@dataclass
class TennisObservation(Observation):
    screen_rgb: List[int]
    screen_shape: List[int]
    score: Tuple[int, int]              # (agent, opponent)
    ball_side: str                      # "left"/"center"/"right"/"unknown"
    my_position: str                    # "top"/"middle"/"bottom"/"unknown"
    opponent_position: str              # "top"/"middle"/"bottom"/"unknown"
    rally_length: int
    legal_actions: List[int]
    action_meanings: List[str]
    lives: int
    done: bool
    reward: Optional[float]
    metadata: Dict[str, Any]
```

### TennisState

```python
@dataclass
class TennisState(State):
    previous_score: Tuple[int, int]
    rally_length: int
    total_points: int
    agent_games_won: int
    opponent_games_won: int
```

### TennisEnvironment

```python
class TennisEnvironment:
    def __init__(
        self,
        mode: Optional[int] = None,
        difficulty: Optional[int] = None,
        repeat_action_probability: float = 0.25,
        frameskip: int = 4,
        render_mode: Optional[str] = None,
        # Dynamic reward shaping parameters
        score_reward: float = 10.0,
        score_penalty: float = -5.0,
        rally_bonus_max: float = 1.0,
        rally_bonus_scale: float = 0.1,
        movement_bonus: float = 0.05,
        positioning_bonus: float = 0.1,
        center_bonus: float = 0.2,
    )

    def reset(self) -> TennisObservation
    def step(self, action: TennisAction) -> TennisObservation
```

## Limitations & Future Work

### Current Limitations

1. **Simulated Multi-Agent**: Not true multi-agent (PettingZoo-style), uses perspective switching
2. **Score Tracking**: Approximate (reward-based, not exact tennis scoring)
3. **Symbolic Extraction**: Best-effort pixel detection (may return "unknown")

### Planned Features

- [ ] True multi-agent with PettingZoo integration
- [ ] Improved score tracking via RAM reading
- [ ] Ball trajectory prediction
- [ ] Pre-trained model checkpoints

## References

- [ALE Tennis Documentation](https://github.com/Farama-Foundation/Arcade-Learning-Environment/tree/master/docs)

## License

BSD-style license (see LICENSE file in repository root)

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{openenv_tennis,
  title = {Tennis Environment for OpenEnv: Simulated Multi-Agent RL with LLMs},
  author = {OpenEnv Contributors},
  year = {2025},
  url = {https://github.com/meta-pytorch/OpenEnv}
}
```
