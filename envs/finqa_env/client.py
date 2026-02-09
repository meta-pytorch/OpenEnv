# envs/finqa_env/client.py
"""
HTTP client for the FinQA environment.

This client connects to a running FinQA environment server and provides
a Python interface for interacting with it.

Example:
    >>> from envs.finqa_env import FinQAEnv, FinQAAction
    >>>
    >>> # Connect to a running server
    >>> client = FinQAEnv(base_url="http://localhost:8000")
    >>>
    >>> # Or start from a Docker image
    >>> client = FinQAEnv.from_docker_image("finqa-env:latest")
    >>>
    >>> # Reset to start a new episode
    >>> result = client.reset()
    >>> print(f"Question: {result.observation.question}")
    >>>
    >>> # Take actions
    >>> action = FinQAAction(
    ...     tool_name="get_descriptions",
    ...     tool_args={"company_name": result.observation.company}
    ... )
    >>> result = client.step(action)
    >>> print(f"Result: {result.observation.tool_result}")
    >>>
    >>> # Continue until done
    >>> while not result.done:
    ...     # Your agent's logic here
    ...     action = FinQAAction(tool_name="submit_answer", tool_args={"answer": "6.118"})
    ...     result = client.step(action)
    >>>
    >>> print(f"Reward: {result.reward}")
    >>> client.close()
"""

from typing import Dict, Any, List

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from .models import FinQAAction, FinQAObservation, FinQAState, AVAILABLE_TOOLS


class FinQAEnv(EnvClient[FinQAAction, FinQAObservation, FinQAState]):
    """
    Client for the FinQA environment.

    Inherits from EnvClient and implements the serialization/deserialization
    logic for FinQA-specific action and observation types.
    """

    def __init__(self, base_url: str, timeout: float = 120.0, provider=None):
        """
        Initialize the client.

        Args:
            base_url: URL of the running environment server
            timeout: Request timeout in seconds (default: 120s for SQL queries)
            provider: Optional container provider for lifecycle management
        """
        super().__init__(base_url=base_url, request_timeout_s=timeout, provider=provider)

    def _step_payload(self, action: FinQAAction) -> dict:
        """
        Convert a FinQAAction to a JSON-serializable dictionary.

        Args:
            action: The action to serialize

        Returns:
            Dictionary to send as request body
        """
        return {
            "tool_name": action.tool_name,
            "tool_args": action.tool_args,
        }

    def _parse_result(self, payload: dict) -> StepResult[FinQAObservation]:
        """
        Parse server response into a StepResult.

        Handles both /reset and /step responses, which may have different
        nesting structures.

        Args:
            payload: Raw dictionary from server response

        Returns:
            Structured StepResult object
        """
        # Handle potential double-nesting from /step endpoint
        obs_data = payload.get("observation")

        if isinstance(obs_data, dict) and "observation" in obs_data:
            # Double-nested structure from /step
            actual_obs_data = obs_data.get("observation", {})
            reward = obs_data.get("reward", payload.get("reward"))
            done = obs_data.get("done", payload.get("done", False))
        else:
            # Single-nested structure from /reset or direct
            actual_obs_data = obs_data if isinstance(obs_data, dict) else {}
            reward = payload.get("reward")
            done = payload.get("done", False)

        if not isinstance(actual_obs_data, dict):
            actual_obs_data = {}

        obs = FinQAObservation(
            question=actual_obs_data.get("question", ""),
            company=actual_obs_data.get("company", ""),
            tool_result=actual_obs_data.get("tool_result", ""),
            history=actual_obs_data.get("history", []),
            step_count=actual_obs_data.get("step_count", 0),
            available_tools=actual_obs_data.get("available_tools", AVAILABLE_TOOLS.copy()),
            done=actual_obs_data.get("done", done),
            reward=actual_obs_data.get("reward", reward),
            metadata=payload.get("metadata", {}),
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict) -> FinQAState:
        """
        Parse server /state response into a FinQAState.

        Args:
            payload: Raw dictionary from server response

        Returns:
            Structured FinQAState object
        """
        return FinQAState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_question=payload.get("current_question"),
            current_company=payload.get("current_company"),
            ground_truth=payload.get("ground_truth"),
            question_id=payload.get("question_id"),
        )
