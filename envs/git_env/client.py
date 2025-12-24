#!/usr/bin/env python3
"""
GitEnv Client
-------------
Client-side wrapper for the Git environment server.

This client maintains a persistent WebSocket connection to the environment
server, enabling efficient multi-step interactions with lower latency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import GitAction, GitObservation, GitState

if TYPE_CHECKING:
    from openenv.core.containers.runtime import ContainerProvider


class GitEnv(EnvClient[GitAction, GitObservation, GitState]):
    """
    Client for Git Environment with Gitea server.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions for Git operations.

    The environment connects to a shared external Gitea service. Repositories
    must be pre-migrated to Gitea before use.

    Example:
        >>> # From Docker image
        >>> client = GitEnv.from_docker_image("git-env:latest")
        >>> try:
        ...     result = client.reset()
        ...
        ...     # List available repositories
        ...     from envs.git_env import GitAction
        ...     result = client.step(GitAction(action_type="list_repos"))
        ...     print(result.observation.repos)
        ...
        ...     # Clone repository to workspace
        ...     result = client.step(GitAction(action_type="clone_repo", repo_name="OpenEnv"))
        ...
        ...     # Execute git commands
        ...     result = client.step(GitAction(
        ...         action_type="execute_git_command",
        ...         command="status",
        ...         working_dir="OpenEnv"
        ...     ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: GitAction) -> dict:
        """
        Convert action to payload for server's /step endpoint.

        Args:
            action: GitAction to send to server

        Returns:
            Dictionary payload for HTTP request
        """
        # Convert action to dictionary
        payload = {
            "action_type": action.action_type,
        }

        # Add type-specific fields for supported actions
        if hasattr(action, "repo_name"):
            payload["repo_name"] = action.repo_name
        if hasattr(action, "target_dir"):
            payload["target_dir"] = action.target_dir
        if hasattr(action, "command"):
            payload["command"] = action.command
        if hasattr(action, "working_dir"):
            payload["working_dir"] = action.working_dir

        return payload

    def _parse_result(self, payload: dict) -> StepResult[GitObservation]:
        """
        Parse server response into StepResult.

        Args:
            payload: JSON response from /step endpoint

        Returns:
            StepResult containing GitObservation
        """
        obs = GitObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> GitState:
        """
        Parse server response into GitState object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            GitState object with environment state
        """
        return GitState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            gitea_ready=payload.get("gitea_ready", False),
            workspace_path=payload.get("workspace_path", "/workspace"),
        )
