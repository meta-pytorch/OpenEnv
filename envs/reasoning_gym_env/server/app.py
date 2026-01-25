# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Reasoning Gym Environment.

This module creates an HTTP server that exposes the ReasoningGymEnvironment
over HTTP and WebSocket endpoints, compatible with MCPToolClient.

Clients configure the task at runtime via reset():
    env.reset(task_name="basic_arithmetic", task_config={"max_value": 100})

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Or run directly:
    uv run --project . server
"""

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .reasoning_gym_environment import ReasoningGymEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.reasoning_gym_environment import ReasoningGymEnvironment


def create_configured_environment() -> ReasoningGymEnvironment:
    """Factory function that creates a ReasoningGymEnvironment with defaults.

    Clients configure the task via reset():
        env.reset(task_name="basic_arithmetic", task_config={"max_value": 100})
    """
    return ReasoningGymEnvironment()


# Create the app with the configured environment factory
# The factory is called for each new WebSocket session
app = create_app(
    create_configured_environment,
    CallToolAction,
    CallToolObservation,
    env_name="reasoning_gym_env",
)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m envs.reasoning_gym_env.server.app
        openenv serve reasoning_gym_env
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
