"""
envs/coding_env/models.py
--------------------------------
Action/Observation types for the Coding environment.
"""

from __future__ import annotations

from dataclasses import dataclass

# Support both standalone and in-repo imports
try:
    # Standalone imports (when installed from pip)
    from openenv_core.env_server.types import Action, Observation, State
except ImportError:
    # In-repo imports (when running from OpenEnv repository)
    from core.env_server.types import Action, Observation, State


@dataclass
class CodeAction(Action):
    """
    Represents a single code execution request.
    """

    code: str
    # Optional: future fields like 'lint': bool, 'timeout_s': float, etc.


@dataclass
class CodeObservation(Observation):
    """
    Result of executing code in the environment.
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


@dataclass
class CodeState(State):
    """State for CodeAct environment with persistent execution context."""

    last_exit_code: int = 0
