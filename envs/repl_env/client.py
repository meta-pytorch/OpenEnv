# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
REPL Environment Client.

This module provides the client for connecting to a REPL Environment server
via WebSocket for persistent sessions.
"""

from typing import Any, Dict, List, Optional

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from .models import REPLAction, REPLObservation, REPLState, CodeBlockResult
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from models import REPLAction, REPLObservation, REPLState, CodeBlockResult


class REPLEnv(EnvClient[REPLAction, REPLObservation, REPLState]):
    """
    Client for the REPL Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with REPLEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(context="Hello World", task_prompt="Count chars")
        ...     print(result.observation.context_preview)
        ...
        ...     result = client.execute("result = len(context)")
        ...     print(result.observation.result.success)
        ...
        ...     result = client.execute("print(f'FINAL({result})')")
        ...     print(result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = REPLEnv.from_docker_image("repl-env:latest")
        >>> try:
        ...     result = client.reset(context="Large document...")
        ...     result = client.execute("chunks = context.split('\\\\n')")
        ...     result = client.submit_final_answer("42")
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: REPLAction) -> Dict:
        """
        Convert REPLAction to JSON payload for step request.

        Args:
            action: REPLAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "code": action.code,
            "is_final": action.is_final,
            "final_answer": action.final_answer,
        }

    def _parse_result(self, payload: Dict) -> StepResult[REPLObservation]:
        """
        Parse server response into StepResult[REPLObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with REPLObservation
        """
        obs_data = payload.get("observation", {})
        result_data = obs_data.get("result", {})

        observation = REPLObservation(
            result=CodeBlockResult(
                stdout=result_data.get("stdout", ""),
                stderr=result_data.get("stderr", ""),
                locals_snapshot=result_data.get("locals_snapshot", {}),
                execution_time=result_data.get("execution_time", 0.0),
                success=result_data.get("success", True),
                exception=result_data.get("exception"),
            ),
            context_preview=obs_data.get("context_preview"),
            context_length=obs_data.get("context_length", 0),
            available_variables=obs_data.get("available_variables", []),
            iteration=obs_data.get("iteration", 0),
            max_iterations=obs_data.get("max_iterations", 30),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> REPLState:
        """
        Parse server response into REPLState object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            REPLState object
        """
        return REPLState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            context=payload.get("context"),
            task_prompt=payload.get("task_prompt"),
            iteration=payload.get("iteration", 0),
            max_iterations=payload.get("max_iterations", 30),
            namespace_keys=payload.get("namespace_keys", []),
            final_answer=payload.get("final_answer"),
            total_execution_time=payload.get("total_execution_time", 0.0),
        )

    # Convenience methods for better developer experience

    def execute(self, code: str) -> StepResult[REPLObservation]:
        """
        Execute Python code in the REPL.

        Convenience method that wraps step() with a code-only action.

        Args:
            code: Python code to execute

        Returns:
            StepResult with execution observation
        """
        return self.step(REPLAction(code=code))

    def submit_final_answer(self, answer: str) -> StepResult[REPLObservation]:
        """
        Submit a final answer and terminate the episode.

        Args:
            answer: The final answer string

        Returns:
            StepResult with done=True
        """
        return self.step(REPLAction(code="", is_final=True, final_answer=answer))

    def get_variable(self, name: str) -> StepResult[REPLObservation]:
        """
        Retrieve and print a variable from the REPL namespace.

        Args:
            name: Variable name to retrieve

        Returns:
            StepResult with variable value in stdout
        """
        return self.execute(f"print(repr({name}))")

    def list_variables(self) -> List[str]:
        """
        Get list of available variables in the current session.

        Note: This requires an active session (after reset).

        Returns:
            List of variable names
        """
        state = self.state()
        return state.namespace_keys if state else []
