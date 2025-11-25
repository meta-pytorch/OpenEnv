# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Doom Env Environment.

This module creates an HTTP server that exposes the DoomEnvironment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv_core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv_core is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import DoomAction, DoomObservation
    from .doom_env_environment import DoomEnvironment
except ImportError:
    from doom_env.server.doom_env_environment import DoomEnvironment
    from doom_env.models import DoomAction, DoomObservation

import os

# Read configuration from environment variables
scenario = os.getenv("DOOM_SCENARIO", "basic")
screen_resolution = os.getenv("DOOM_SCREEN_RESOLUTION", "RES_160X120")
screen_format = os.getenv("DOOM_SCREEN_FORMAT", "RGB24")
window_visible = os.getenv("DOOM_WINDOW_VISIBLE", "false").lower() in ("true", "1", "yes")

# Create the environment instance with configuration
env = DoomEnvironment(
    scenario=scenario,
    screen_resolution=screen_resolution,
    screen_format=screen_format,
    window_visible=window_visible,
    use_discrete_actions=True,
)

# Create the app with web interface and README integration
app = create_app(
    env,
    DoomAction,
    DoomObservation,
    env_name="doom_env",
)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m envs.doom_env.server.app
        openenv serve doom_env

    For production deployments, use uvicorn directly:
        uvicorn envs.doom_env.server.app:app --workers 4
    """
    import uvicorn

    print(f"Doom Environment Configuration:")
    print(f"  Scenario: {scenario}")
    print(f"  Resolution: {screen_resolution}")
    print(f"  Format: {screen_format}")
    print(f"  Window Visible: {window_visible}")
    print(f"\nStarting server on http://0.0.0.0:8000")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
