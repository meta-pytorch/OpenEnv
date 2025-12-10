#!/usr/bin/env python3

"""
FastAPI application for Git Environment.

This module creates an HTTP server for the Git environment that connects
to a shared external Gitea service for fast, isolated task resets.

Environment variables (required):
    GITEA_URL: URL of shared Gitea service
    GITEA_USERNAME: Gitea username
    GITEA_PASSWORD: Gitea password
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
gitea_url = os.getenv("GITEA_URL")
gitea_username = os.getenv("GITEA_USERNAME")
gitea_password = os.getenv("GITEA_PASSWORD")
workspace_dir = os.getenv("WORKSPACE_DIR", "/workspace")

# Validate required environment variables
if not gitea_url:
    raise RuntimeError("GITEA_URL environment variable is required")
if not gitea_username:
    raise RuntimeError("GITEA_USERNAME environment variable is required")
if not gitea_password:
    raise RuntimeError("GITEA_PASSWORD environment variable is required")

# Create the environment instance (connects to external Gitea)
env = GitTaskEnvironment(
    gitea_url=gitea_url,
    username=gitea_username,
    password=gitea_password,
    workspace_dir=workspace_dir,
)

# Create the app with web interface and README integration
app = create_app(env, GitAction, GitObservation, env_name="git_env")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
