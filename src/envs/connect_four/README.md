# Connect Four (OpenSpiel) — OpenEnv Wrapper

This environment wraps **OpenSpiel**’s `connect_four` and exposes an OpenEnv-style API.

## Observation
- **Board**: `6 x 7` int grid in the _agent’s_ view
  - `0` empty, `+1` agent discs (player 0), `-1` opponent discs (player 1).
- **Legal actions**: playable columns `[0..6]`.
- **current_player**: `+1` if agent to move, `-1` otherwise.
- **reward**: scalar, agent centric (`+1` win, `-1` loss, `0` otherwise).

## Endpoints
- `POST /reset` → `{ observation, state }`
- `POST /step` w/ `{"column": int}` → `{ observation, state }`
- `GET /state` → current metadata
- `POST /close` → cleanup

## Local run
```bash
pip install "open_spiel>=1.6" fastapi "uvicorn[standard]" numpy
uvicorn src.envs.connect_four.server.app:app --host 0.0.0.0 --port 8020
