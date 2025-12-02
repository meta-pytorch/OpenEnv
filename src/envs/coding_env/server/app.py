# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Coding Environment.

This module creates an HTTP server that exposes the PythonCodeActEnv
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.coding_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.coding_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.coding_env.server.app
"""

# Support both standalone and in-repo imports
try:
    # Standalone imports (when installed from pip)
    from openenv_core.env_server import create_app
    from coding_env.models import CodeAction, CodeObservation
    from coding_env.server.python_codeact_env import PythonCodeActEnv
except ImportError:
    # In-repo imports (when running from OpenEnv repository)
    from core.env_server import create_app
    from envs.coding_env.models import CodeAction, CodeObservation
    from envs.coding_env.server.python_codeact_env import PythonCodeActEnv

# Create the environment instance
env = PythonCodeActEnv()

# Create the app with web interface and README integration
app = create_app(env, CodeAction, CodeObservation, env_name="coding_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


def main():
    """Main entry point for running the server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
