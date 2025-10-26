# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Tennis Environment.

This module creates an HTTP server that exposes the Tennis game
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.tennis_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.tennis_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.tennis_env.server.app

Environment variables:
    TENNIS_MODE: Game mode (optional, typically 0)
    TENNIS_DIFFICULTY: Game difficulty (optional, 0-3)
    TENNIS_REPEAT_ACTION_PROB: Sticky action probability (default: "0.25")
    TENNIS_FRAMESKIP: Frameskip (default: "4")

    Dynamic reward shaping (optional):
    TENNIS_SCORE_REWARD: Reward for scoring a point (default: "10.0")
    TENNIS_SCORE_PENALTY: Penalty for opponent scoring (default: "-5.0")
    TENNIS_RALLY_BONUS_MAX: Maximum rally bonus (default: "1.0")
    TENNIS_RALLY_BONUS_SCALE: Rally bonus scale factor (default: "0.1")
    TENNIS_MOVEMENT_BONUS: Reward for movement (default: "0.05")
    TENNIS_POSITIONING_BONUS: Reward for good positioning (default: "0.1")
    TENNIS_CENTER_BONUS: Reward for center positioning (default: "0.2")
"""

import os

from core.env_server import create_app

from ..models import TennisAction, TennisObservation
from .tennis_environment import TennisEnvironment

# Get configuration from environment variables
mode = os.getenv("TENNIS_MODE")
difficulty = os.getenv("TENNIS_DIFFICULTY")
repeat_action_prob = float(os.getenv("TENNIS_REPEAT_ACTION_PROB", "0.25"))
frameskip = int(os.getenv("TENNIS_FRAMESKIP", "4"))

# Reward shaping parameters (dynamic RL)
score_reward = float(os.getenv("TENNIS_SCORE_REWARD", "10.0"))
score_penalty = float(os.getenv("TENNIS_SCORE_PENALTY", "-5.0"))
rally_bonus_max = float(os.getenv("TENNIS_RALLY_BONUS_MAX", "1.0"))
rally_bonus_scale = float(os.getenv("TENNIS_RALLY_BONUS_SCALE", "0.1"))
movement_bonus = float(os.getenv("TENNIS_MOVEMENT_BONUS", "0.05"))
positioning_bonus = float(os.getenv("TENNIS_POSITIONING_BONUS", "0.1"))
center_bonus = float(os.getenv("TENNIS_CENTER_BONUS", "0.2"))

# Convert to int if specified
mode = int(mode) if mode is not None else None
difficulty = int(difficulty) if difficulty is not None else None

# Create the environment instance with dynamic rewards
env = TennisEnvironment(
    mode=mode,
    difficulty=difficulty,
    repeat_action_probability=repeat_action_prob,
    frameskip=frameskip,
    score_reward=score_reward,
    score_penalty=score_penalty,
    rally_bonus_max=rally_bonus_max,
    rally_bonus_scale=rally_bonus_scale,
    movement_bonus=movement_bonus,
    positioning_bonus=positioning_bonus,
    center_bonus=center_bonus,
)

# Create the FastAPI app with web interface and README integration
app = create_app(env, TennisAction, TennisObservation, env_name="tennis_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
