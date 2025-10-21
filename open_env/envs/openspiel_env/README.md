# OpenSpiel Environment

Integration of OpenSpiel games with the OpenEnv framework. OpenSpiel (https://github.com/google-deepmind/open_spiel) is DeepMind's collection of 70+ game environments for RL research.

## Supported Games

This environment supports 6 games across different categories:

### Single-Player Games (No Opponent)
1. **Catch** - Move horizontally to catch a falling ball
2. **Cliff Walking** - Navigate grid without falling off cliff (Sutton & Barto benchmark)
3. **2048** - Classic tile-merging puzzle game
4. **Blackjack** - Simplified blackjack (HIT/STAND only)

### Multi-Player Games (with Bot Opponent)
5. **Tic-Tac-Toe** - Classic 3x3 game
6. **Kuhn Poker** - 2-player simplified poker (game theory benchmark)

## Architecture

```
┌────────────────────────────────────┐
│ RL Training Code (Client)          │
│   OpenSpielEnv.step(action)        │
└──────────────┬─────────────────────┘
               │ HTTP
┌──────────────▼─────────────────────┐
│ FastAPI Server (Docker)            │
│   OpenSpielEnvironment             │
│     ├─ Wraps rl_environment.Env    │
│     ├─ Agent controls player 0     │
│     └─ Opponent: Random/Fixed      │
└────────────────────────────────────┘
```

## Installation & Usage

### Option 1: Local Development (without Docker)

**Requirements:**
- OpenSpiel must be installed (see https://github.com/google-deepmind/open_spiel)
- Python 3.11+

```python
from envs.openspiel_env import OpenSpielEnv, OpenSpielAction

# Start local server manually
# python -m envs.openspiel_env.server.app

# Connect to local server
env = OpenSpielEnv(base_url="http://localhost:8000")

# Reset environment
result = env.reset()
print(f"Initial state: {result.observation.info_state}")
print(f"Legal actions: {result.observation.legal_actions}")

# Take actions
for _ in range(10):
    action_id = result.observation.legal_actions[0]  # Choose first legal action
    result = env.step(OpenSpielAction(action_id=action_id))
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
docker build -f src/envs/openspiel_env/server/Dockerfile -t openspiel-env:latest .
```

**Run specific games:**

```bash
# Catch (default)
docker run -p 8000:8000 openspiel-env:latest

# Tic-Tac-Toe with random opponent
docker run -p 8000:8000 -e OPENSPIEL_GAME=tic_tac_toe openspiel-env:latest

# Kuhn Poker
docker run -p 8000:8000 -e OPENSPIEL_GAME=kuhn_poker openspiel-env:latest

# 2048
docker run -p 8000:8000 -e OPENSPIEL_GAME=2048 openspiel-env:latest
```

**Use with from_docker_image():**

```python
from envs.openspiel_env import OpenSpielEnv, OpenSpielAction

# Automatically starts container
env = OpenSpielEnv.from_docker_image("openspiel-env:latest")

result = env.reset()
result = env.step(OpenSpielAction(action_id=0))

env.close()  # Stops container
```

## Game-Specific Information

### 1. Catch
- **Type**: Single-player
- **Action Space**: 3 actions (left, stay, right)
- **Observation**: 5x5 grid flattened (25 dimensions)
- **Reward**: +1 for catching ball, 0 otherwise
- **Episode Length**: ~10 steps

```python
env = OpenSpielEnv.from_docker_image("openspiel-env:latest")
# Or set OPENSPIEL_GAME=catch
```

### 2. Tic-Tac-Toe
- **Type**: 2-player turn-based, perfect information
- **Players**: Agent (X) vs Random Bot (O)
- **Action Space**: 9 positions
- **Observation**: 27 dimensions (3x3 board + game state)
- **Reward**: +1 win, -1 loss, 0 draw/mid-game

```python
# Set environment variable or run directly
docker run -p 8000:8000 -e OPENSPIEL_GAME=tic_tac_toe openspiel-env:latest
```

### 3. Kuhn Poker
- **Type**: 2-player turn-based, imperfect information
- **Players**: Agent vs Random Bot
- **Action Space**: 2 actions (pass/fold, bet/call)
- **Observation**: 6 dimensions (card + betting history)
- **Reward**: Pot winnings (typically -1, 0, +1, +2)
- **Notes**: THE benchmark for imperfect-information RL

```python
docker run -p 8000:8000 -e OPENSPIEL_GAME=kuhn_poker openspiel-env:latest
```

### 4. Cliff Walking
- **Type**: Single-player grid world
- **Action Space**: 4 actions (up, down, left, right)
- **Observation**: Position encoding
- **Reward**: -1 per step, -100 for falling off cliff
- **Notes**: Classic RL benchmark from Sutton & Barto

```python
docker run -p 8000:8000 -e OPENSPIEL_GAME=cliff_walking openspiel-env:latest
```

### 5. 2048
- **Type**: Single-player puzzle
- **Action Space**: 4 actions (up, down, left, right)
- **Observation**: 4x4 grid with tile values
- **Reward**: Points from merging tiles
- **Notes**: Stochastic tile spawning

```python
docker run -p 8000:8000 -e OPENSPIEL_GAME=2048 openspiel-env:latest
```

### 6. Blackjack
- **Type**: Single-player vs dealer
- **Action Space**: 2 actions (HIT, STAND)
- **Observation**: Player hand + dealer's visible card
- **Reward**: +1 win, -1 loss, 0 draw
- **Notes**: Simplified version, no double/split

```python
docker run -p 8000:8000 -e OPENSPIEL_GAME=blackjack openspiel-env:latest
```

## Configuration

### Environment Variables

- `OPENSPIEL_GAME`: Game name (default: "catch")
- `OPENSPIEL_AGENT_PLAYER`: Player ID for agent (default: 0)
- `OPENSPIEL_OPPONENT_POLICY`: Opponent policy for multi-player games
  - `random`: Uniform random (default)
  - `first`: Always picks first legal action
  - `last`: Always picks last legal action

### Example: Tic-Tac-Toe with Fixed Opponent

```bash
docker run -p 8000:8000 \
  -e OPENSPIEL_GAME=tic_tac_toe \
  -e OPENSPIEL_OPPONENT_POLICY=first \
  openspiel-env:latest
```

## API Reference

### OpenSpielAction

```python
@dataclass
class OpenSpielAction(Action):
    action_id: int                      # Action to take
    game_name: str = "catch"            # Game name
    game_params: Dict[str, Any] = {}    # Optional game parameters
```

### OpenSpielObservation

```python
@dataclass
class OpenSpielObservation(Observation):
    info_state: List[float]             # Agent's information state
    legal_actions: List[int]            # Legal action IDs
    game_phase: str                     # "initial", "playing", "terminal"
    current_player_id: int              # Current player (-1 for simultaneous)
    opponent_last_action: Optional[int] # Last opponent action (if available)
    done: bool                          # Episode finished
    reward: Optional[float]             # Reward for last action
```

### OpenSpielState

```python
@dataclass
class OpenSpielState(State):
    episode_id: str                     # Unique episode ID
    step_count: int                     # Number of steps
    game_name: str                      # Game name
    agent_player: int                   # Agent's player ID
    opponent_policy: str                # Opponent policy name
    num_players: int                    # Total players
```

## Testing

### Automated Testing (All 6 Games)

**Quick test of all games in Docker:**
```bash
./test_docker_all_games.sh
```

This automated script will:
- Build and run Docker containers for each game
- Test reset, step, and state APIs
- Verify episode completion
- Report pass/fail for all 6 games

**Expected output:**
```
========================================
OpenSpiel Docker Integration Test
========================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Testing: catch
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🐳 Starting Docker container...
  ⏳ Waiting for server to be ready...
  ✓ Server ready (2s)
  🎮 Running Python client test...
  ✓ PASSED - Episode completed successfully

[... tests all 6 games ...]

========================================
Test Summary
========================================

  ✓ catch
  ✓ tic_tac_toe
  ✓ kuhn_poker
  ✓ cliff_walking
  ✓ 2048
  ✓ blackjack

Total: 6 passed, 0 failed out of 6 games

========================================
All tests PASSED! 🎉
========================================
```

### Manual Testing

```bash
# Local (requires OpenSpiel installed)
python -m pytest src/envs/openspiel_env/

# Docker build
docker build -f src/envs/openspiel_env/server/Dockerfile -t openspiel-env:latest .

# Run specific game
docker run -p 8000:8000 openspiel-env:latest

# Test from another terminal
python3 examples/openspiel_simple.py
```

## Development

### Adding New Games

To add support for more OpenSpiel games:

1. Verify the game works with `rl_environment.Environment`
2. Test with different opponent policies if multi-player
3. Document game-specific configuration
4. Add example script

## Limitations

- **Simultaneous-move games**: Only agent_player=0 supported
- **Multi-agent training**: Single agent only (no self-play yet)
- **Opponent policies**: Random and fixed only (no MCTS yet)
- **Build time**: Docker image takes ~5-10 minutes to build (compiles C++)

## Future Work

- MCTS opponent policies
- Self-play support (multiple agents)
- More games (Chess, Go, Poker Hold'em)
- Faster build with pre-built OpenSpiel base image
- Game-specific reward shaping options

## References

- [OpenSpiel Paper (2019)](https://arxiv.org/abs/1908.09453)
- [OpenSpiel GitHub](https://github.com/google-deepmind/open_spiel)
- [OpenSpiel Documentation](https://openspiel.readthedocs.io/)
