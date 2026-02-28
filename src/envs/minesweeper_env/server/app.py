# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Minesweeper Environment."""

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server import create_app
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server import create_app

from ..models import MinesweeperAction, MinesweeperObservation
from .minesweeper_environment import MinesweeperEnvironment

# Create the FastAPI app
# Pass the class (factory) instead of an instance for WebSocket session support
app = create_app(
    MinesweeperEnvironment,
    MinesweeperAction,
    MinesweeperObservation,
    env_name="minesweeper_env"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
