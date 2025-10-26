# Connect4 Environment

This environment wraps the [`gym-connect4`](https://github.com/Danielhp95/gym-connect4) implementation inside OpenEnv. It exposes a turn-based 6x7 Connect Four board where the agent plays as player `+1` against the built-in opponent logic supplied by the Gym environment.

## Action, Observation, State

| Type | Fields | Description |
| --- | --- | --- |
| `Connect4Action` | `column: int` | 0-based column where the agent drops a disc. |
| `Connect4Observation` | `board: list[list[int]]`<br>`legal_actions: list[int]`<br>`current_player: int`<br>`last_move: Optional[int]`<br>`info: dict` | Board uses `1` for the agent, `-1` for the opponent, `0` for empty. Legal actions are the playable columns. When `done=True`, `legal_actions` is empty. Any metadata from Gym is forwarded through `info`. |
| `Connect4State` | `episode_id: str`<br>`step_count: int`<br>`rows: int`<br>`cols: int` | Mirrors the generic OpenEnv state and records the board geometry. |

Rewards from Gym can be scalars or a 2-element vector. The server always scalarizes them into an agent-centric `float` (`r_agent - r_opponent` when two values are supplied).

## Running the server

```bash
uvicorn envs.connect4_env.server.app:app --host 0.0.0.0 --port 8000
```

Set `GYM_CONNECT4_ID` if you need a custom Gym registration ID (default `Connect4-v0`).

## Client usage

```python
from envs.connect4_env import Connect4Env, Connect4Action

client = Connect4Env(base_url="http://localhost:8000")

result = client.reset()
print(result.observation.board)

while not result.done:
    action = Connect4Action(column=result.observation.legal_actions[0])
    result = client.step(action)

print("Episode reward:", result.reward)
```
