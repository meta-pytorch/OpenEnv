# Chess Environment

A chess reinforcement learning environment for OpenEnv, powered by the [moonfish](https://github.com/luccabb/moonfish) chess engine.

## Features

- **Full chess rules** via python-chess library
- **Configurable opponent**: moonfish engine, random moves, or self-play
- **Position evaluation**: Uses moonfish's PSQT-based evaluation
- **Standard OpenEnv interface**: reset(), step(), state

## Quick Start

### Using Docker

```bash
# Build the image
docker build -t chess-env:latest -f envs/chess_env/server/Dockerfile .

# Run the server
docker run -p 8000:8000 chess-env:latest
```

### Using the Client

```python
from envs.chess_env import ChessEnv, ChessAction

# Connect to server
with ChessEnv(base_url="http://localhost:8000") as env:
    # Reset for a new game
    result = env.reset()
    print(f"Starting position: {result.observation.fen}")
    print(f"Legal moves: {result.observation.legal_moves}")

    # Make a move
    result = env.step(ChessAction(move="e2e4"))
    print(f"Reward: {result.reward}, Done: {result.done}")

    # Play until game ends
    while not result.done:
        # Your policy here
        move = result.observation.legal_moves[0]
        result = env.step(ChessAction(move=move))

    print(f"Game result: {result.observation.result}")
```

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `fen` | str | Board position in FEN notation |
| `legal_moves` | List[str] | Legal moves in UCI format |
| `is_check` | bool | Whether current player is in check |
| `done` | bool | Whether game has ended |
| `reward` | float | Reward for last action |
| `result` | str | Game result ("1-0", "0-1", "1/2-1/2") |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `move` | str | UCI format move (e.g., "e2e4", "e7e8q") |

## Rewards

| Outcome | Reward |
|---------|--------|
| Win | +1.0 |
| Loss | -1.0 |
| Draw | 0.0 |
| Illegal move | -0.1 |

## Configuration

The environment supports these configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `opponent` | "moonfish" | Opponent type: "moonfish", "random", or None |
| `opponent_depth` | 2 | Search depth for moonfish opponent |
| `max_moves` | 500 | Maximum half-moves before draw |
| `agent_color` | None | Agent color: "white", "black", or None (alternate each episode) |

## Links

- [moonfish GitHub](https://github.com/luccabb/moonfish)
- [Play online](https://huggingface.co/spaces/luccabb/moonfish_chess)
