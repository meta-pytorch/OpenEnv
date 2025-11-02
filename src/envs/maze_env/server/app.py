# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Maze Environment.

This module creates an HTTP server that exposes Maze game
over HTTP endpoints, making them compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.maze_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.maze_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.maze_env.server.app

Variables:
    maze: np.array - Maze as a numpy array
"""

from core.env_server import create_app
from ..models import MazeAction, MazeObservation
from .maze_environment import MazeEnvironment
from .mazearray import maze
# Get game configuration from environment variables

# Create the environment instance
env = MazeEnvironment(maze_array=maze,start_cell=(0,0),exit_cell=(7,7))

# Create the FastAPI app with web interface and README integration
app = create_app(env, MazeAction, MazeObservation, env_name="maze_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
