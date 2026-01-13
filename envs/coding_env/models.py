"""
envs/coding_env/models.py
--------------------------------
Action/Observation types for the Coding environment.
"""

from __future__ import annotations

from openenv.core.env_server.interfaces import Action, Observation, State
class CodeAction(Action):
    """
    Represents a single code execution request.
    """

    code: str
    # Optional: future fields like 'lint': bool, 'timeout_s': float, etc.


class CodeObservation(Observation):
    """
    Result of executing code in the environment.
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

class CodeState(State):
    """State for CodeAct environment with persistent execution context."""

    last_exit_code: int = 0
