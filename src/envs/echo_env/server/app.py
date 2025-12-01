# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Echo Environment (MCP-based).

This module creates an HTTP server that exposes the EchoEnvironment
over HTTP endpoints with MCP tool support.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server
"""

try:
    from core.env_server import create_app
    from core.env_server.mcp_types import Action, Observation

    from .echo_environment import EchoEnvironment
except ImportError:
    from openenv_core.env_server import create_app
    from openenv_core.env_server.mcp_types import Action, Observation
    from server.echo_environment import EchoEnvironment


# Create the environment instance (MCP client is configured internally)
env = EchoEnvironment()

# Create the FastAPI app
# Note: We use Action and Observation base classes since echo_env
# uses MCP actions (ListToolsAction, CallToolAction) instead of custom types
app = create_app(env, Action, Observation)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m envs.echo_env.server.app
        openenv serve echo_env
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
