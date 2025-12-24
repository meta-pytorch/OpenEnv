# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the OpenSpiel Environment.

This module creates an HTTP server that exposes OpenSpiel games
over HTTP and WebSocket endpoints, compatible with EnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.openspiel_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.openspiel_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.openspiel_env.server.app

Environment variables:
    OPENSPIEL_GAME: Game name to serve (default: "catch")
    OPENSPIEL_AGENT_PLAYER: Agent player ID (default: 0)
    OPENSPIEL_OPPONENT_POLICY: Opponent policy (default: "random")
"""

import os

from openenv.core.env_server import create_app

from ..models import OpenSpielAction, OpenSpielObservation
from .openspiel_environment import OpenSpielEnvironment

# Get game configuration from environment variables
game_name = os.getenv("OPENSPIEL_GAME", "catch")
agent_player = int(os.getenv("OPENSPIEL_AGENT_PLAYER", "0"))
opponent_policy = os.getenv("OPENSPIEL_OPPONENT_POLICY", "random")


# Factory function to create OpenSpielEnvironment instances
def create_openspiel_environment():
    """Factory function that creates OpenSpielEnvironment with config."""
    return OpenSpielEnvironment(
        game_name=game_name,
        agent_player=agent_player,
        opponent_policy=opponent_policy,
    )


# Create the FastAPI app with web interface and README integration
# Pass the factory function instead of an instance for WebSocket session support
app = create_app(create_openspiel_environment, OpenSpielAction, OpenSpielObservation, env_name="openspiel_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
