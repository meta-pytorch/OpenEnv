"""
Julia Code Action Environment.

This environment mirrors the PythonCodeActEnv but runs Julia code instead.
It executes Julia code from CodeAction using subprocess, captures output,
tracks the last exit code, and returns a CodeObservation.
"""

import uuid
import subprocess
import tempfile
import os

from core.env_server import Action, Environment, Observation
from ..models import CodeAction, CodeObservation, CodeState
from .transforms import create_safe_coding_transform


class JuliaCodeActEnv(Environment):
    """
    Julia Code Action Environment for executing code and tracking state.

    This environment executes Julia code submitted as CodeAction during step,
    maintains the last exit code in its state, and returns results wrapped
    in CodeObservation.

    Example:
        >>> env = JuliaCodeActEnv()
        >>> obs = env.reset()
        >>> action = CodeAction(code='println("Hello, Julia!")')
        >>> obs = env.step(action)
        >>> print(obs.stdout)  # "Hello, Julia!\n"
        >>> print(obs.exit_code)  # 0
        >>> print(env.state.last_exit_code)  # 0
    """

    def __init__(self, additional_imports=None):
        """
        Args:
            additional_imports: optional list of Julia packages to `using` by default
        """
        self.transform = create_safe_coding_transform()
        self.additional_imports = additional_imports or []
        self._state = CodeState()

    # --------------------------------------------------
    def reset(self) -> Observation:
        """
        Reset environment for a fresh Julia execution session.
        Returns an empty CodeObservation with exit_code=0.
        """
        self._state = CodeState(episode_id=str(uuid.uuid4()), step_count=0)
        self._state.last_exit_code = 0

        # Reset transform to clear any state
        self.transform = create_safe_coding_transform()

        obs = CodeObservation(stdout="", stderr="", exit_code=0)
        return self._apply_transform(obs)

    # --------------------------------------------------
    def step(self, action: Action) -> Observation:
        """
        Execute Julia code from a CodeAction and return the result as CodeObservation.
        """
        if not isinstance(action, CodeAction):
            raise ValueError(f"Expected CodeAction, got {type(action)}")

        # Construct a temporary file for the code
        with tempfile.TemporaryDirectory() as td:
            code_path = os.path.join(td, "code.jl")

            # Add imports + user code
            prelude = "\n".join([f"using {pkg}" for pkg in self.additional_imports])
            full_code = f"{prelude}\n{action.code}"

            open(code_path, "w").write(full_code)

            # Run Julia in a sandboxed subprocess
            result = subprocess.run(
                ["julia", "--project", code_path],
                capture_output=True,
                text=True,
                timeout=10,
            )

        # Update environment state
        self._state.step_count += 1
        self._state.last_exit_code = result.returncode

        # Build observation
        obs = CodeObservation(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )

        return self._apply_transform(obs)

    # --------------------------------------------------
    @property
    def state(self) -> CodeState:
        """Return current environment state."""
        return self._state
