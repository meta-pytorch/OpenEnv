#!/usr/bin/env python3

"""
FastAPI application for Git Environment.

This module creates an HTTP server for the Git environment that connects
to a shared external Gitea service for fast, isolated task resets.

Environment variables:
    GITEA_URL: URL of shared Gitea service (default: http://localhost:3000)
    GITEA_USERNAME: Gitea username (default: openenv)
    GITEA_PASSWORD: Gitea password (default: openenv)
    WORKSPACE_DIR: Workspace directory (optional, default: /workspace)

Usage:
    # Development (with auto-reload):
    uvicorn envs.git_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.git_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # With custom Gitea:
    GITEA_URL=http://my-gitea:3000 uvicorn envs.git_env.server.app:app --host 0.0.0.0 --port 8000
"""

import os

from openenv.core.env_server import create_app

from ..models import GitAction, GitObservation
from .git_task_environment import GitTaskEnvironment

# Read configuration from environment variables
gitea_url = os.getenv("GITEA_URL", "http://localhost:3000")
gitea_username = os.getenv("GITEA_USERNAME", "openenv")
gitea_password = os.getenv("GITEA_PASSWORD", "openenv")
workspace_dir = os.getenv("WORKSPACE_DIR", "/workspace")


# Factory function to create GitTaskEnvironment instances
def create_git_environment():
    """Factory function that creates GitTaskEnvironment with config."""
    return GitTaskEnvironment(
        gitea_url=gitea_url,
        username=gitea_username,
        password=gitea_password,
        workspace_dir=workspace_dir,
    )


# Create the app with web interface and README integration
# Pass the factory function instead of an instance for WebSocket session support
app = create_app(create_git_environment, GitAction, GitObservation, env_name="git_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
