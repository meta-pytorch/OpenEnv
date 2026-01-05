# TextArena Environment

Generic wrapper for any [TextArena](https://www.textarena.ai/docs/overview) game inside OpenEnv. This module exposes the TextArena `Env` interface through the standard HTTP server/client APIs used by other OpenEnv environments, enabling quick experimentation with the full suite of word, reasoning, and multi-agent games.

## Features
- Works with any registered TextArena game (e.g. `Wordle-v0`, `GuessTheNumber-v0`, `Chess-v0`, ...).
- Transparent access to TextArena message streams, rewards, and state snapshots.
- Docker image for easy deployment with Python 3.11 and preinstalled dependencies.
- Example client demonstrating end-to-end interaction.

## Docker

Build the container from the project root:

```bash
docker build -f envs/textarena_env/server/Dockerfile -t textarena-env:latest .
```

Run it with your desired game (default is `Wordle-v0`). Environment configuration is handled via env vars:

```bash
docker run -p 8000:8000 \
  -e TEXTARENA_ENV_ID=GuessTheNumber-v0 \
  -e TEXTARENA_NUM_PLAYERS=1 \
  textarena-env:latest
```

Additional environment arguments can be passed using the `TEXTARENA_KW_` prefix. For example, to enable `hardcore=True`:

```bash
docker run -p 8000:8000 \
  -e TEXTARENA_ENV_ID=Wordle-v0 \
  -e TEXTARENA_KW_hardcore=true \
  textarena-env:latest
```

## Python Example

The repository ships with a simple client script that connects to a running server (local or Docker) and plays a few turns. Run it from the repo root:

```bash
python examples/textarena_simple.py
```

The script uses `TextArenaEnv.from_docker_image` to automatically build/run the container if needed. Review the source (`examples/textarena_simple.py`) for more details and to customize the gameplay loop.

