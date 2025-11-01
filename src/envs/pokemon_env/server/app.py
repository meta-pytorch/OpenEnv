
"""
FastAPI application for the Pokemon Battle Environment.

This module creates an HTTP server that exposes Pokemon battles
over HTTP endpoints, making them compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.pokemon_env.server.app:app --reload --host 0.0.0.0 --port 9980

    # Production:
    uvicorn envs.pokemon_env.server.app:app --host 0.0.0.0 --port 9980 --workers 4

    # Or run directly:
    python -m envs.pokemon_env.server.app

Environment variables:
    POKEMON_BATTLE_FORMAT: Battle format (default: "gen8randombattle")
    POKEMON_PLAYER_USERNAME: Player username (default: "player")
    POKEMON_SERVER_URL: Pokemon Showdown server URL (default: "localhost:8000")
"""

import os

from core.env_server import create_app

from ..models import PokemonAction, PokemonObservation
from .pokemon_environment import PokemonEnvironment

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

battle_format = os.getenv("POKEMON_BATTLE_FORMAT", "gen9randombattle")
player_username = os.getenv("POKEMON_PLAYER_USERNAME", "player")
reward_mode = os.getenv("POKEMON_REWARD_MODE", "sparse")
max_turns = int(os.getenv("POKEMON_MAX_TURNS", "1000"))

env = PokemonEnvironment(
    battle_format=battle_format,
    player_username=player_username,
    reward_mode=reward_mode,
    max_turns=max_turns,
)

app = create_app(env, PokemonAction, PokemonObservation, env_name="pokemon_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9980)
