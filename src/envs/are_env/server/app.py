# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the ARE Environment.

This module creates an HTTP server that exposes the ARE (Agents Research
Environment) over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.are_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.are_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.are_env.server.app

Environment variables:
    (Phase 1: No environment variables needed for dummy implementation)
    (Phase 2+: Will add ARE-specific configuration)
"""

from core.env_server import create_app

from ..models import AREAction, AREObservation
from .are_environment import AREEnvironment

# Create the environment instance
# Phase 1: Simple instantiation with no configuration
env = AREEnvironment()

# Create the FastAPI app with web interface and README integration
app = create_app(env, AREAction, AREObservation, env_name="are_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
