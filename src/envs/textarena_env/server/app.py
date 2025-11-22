# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the TextArena Environment.

This module creates an HTTP server that exposes the TextArenaEnvironment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os

try:
    from openenv_core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv_core is required for the web interface. Install dependencies with '\n"
        "    uv sync\n'"
    ) from e

from server.environment import TextArenaEnvironment
from models import TextArenaAction, TextArenaObservation


def _parse_env_kwargs(prefix: str = "TEXTARENA_KW_") -> dict[str, str]:
    """Collect arbitrary environment kwargs from the process environment."""

    env_kwargs: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            env_key = key[len(prefix) :].lower()
            env_kwargs[env_key] = value
    return env_kwargs


env_id = os.getenv("TEXTARENA_ENV_ID", "Wordle-v0")
num_players = int(os.getenv("TEXTARENA_NUM_PLAYERS", "1"))
max_turns_env = os.getenv("TEXTARENA_MAX_TURNS")
max_turns = int(max_turns_env) if max_turns_env is not None else None
download_nltk = os.getenv("TEXTARENA_DOWNLOAD_NLTK", "1") in {"1", "true", "True"}

extra_kwargs = _parse_env_kwargs()

environment = TextArenaEnvironment(
    env_id=env_id,
    num_players=num_players,
    max_turns=max_turns,
    download_nltk=download_nltk,
    env_kwargs=extra_kwargs,
)

app = create_app(
    environment, TextArenaAction, TextArenaObservation, env_name="textarena_env"
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m textarena.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn textarena.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
