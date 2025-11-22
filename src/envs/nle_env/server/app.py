# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the NetHack Learning Environment.

This module creates an HTTP server that exposes the NLE environment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.nle_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.nle_env.server.app:app --host 0.0.0.0 --port 8000 --workers 1

    # Or run directly:
    python -m envs.nle_env.server.app

Note:
    NLE is single-threaded (uses C extension with global state), so workers=1
"""

import os

from core.env_server.http_server import create_app

from ..models import NLEAction, NLEObservation
from .nle_environment import NLEEnvironment

# Read configuration from environment variables
TASK_NAME = os.getenv("NLE_TASK", "score")
CHARACTER = os.getenv("NLE_CHARACTER", "mon-hum-neu-mal")
MAX_STEPS = int(os.getenv("NLE_MAX_STEPS", "5000"))

# Create the environment instance
env = NLEEnvironment(
    task_name=TASK_NAME,
    character=CHARACTER,
    max_episode_steps=MAX_STEPS,
)

# Create the app with web interface and README integration
app = create_app(env, NLEAction, NLEObservation, env_name="nle_env")


if __name__ == "__main__":
    import uvicorn

    # NLE must run single-threaded (workers=1) due to C extension
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
