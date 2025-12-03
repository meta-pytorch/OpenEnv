"""
Git Environment - Git server with Gitea support.

This environment connects to a shared Gitea service for task-based isolation,
allowing agents to clone repositories, execute git commands, and manage workspaces.

Note: Repository migration is done externally via Gitea API before environment use.
"""

from .client import GitEnv
from .models import GitAction, GitObservation, GitState

__all__ = [
    "GitEnv",
    "GitAction",
    "GitObservation",
    "GitState",
]
