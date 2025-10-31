# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Echo Environment.

This module creates an HTTP server that exposes the EchoEnvironment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.echo_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.echo_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.echo_env.server.app
"""

from core.env_server.http_server import create_app

from ..models import EchoAction, EchoObservation
from .echo_environment import EchoEnvironment

# Create the environment instance
env = EchoEnvironment()

# Create the app with web interface and README integration
app = create_app(env, EchoAction, EchoObservation, env_name="echo_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)