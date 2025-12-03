# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FinRL Environment HTTP Client.

This module provides the client for connecting to a FinRL Environment server
over HTTP.
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult

from openenv.core.env_server.types import State
from openenv.core.http_env_client import HTTPEnvClient

from .models import FinRLAction, FinRLObservation


class FinRLEnv(HTTPEnvClient[FinRLAction, FinRLObservation]):
    """
    HTTP client for the FinRL Environment.

    This client connects to a FinRLEnvironment HTTP server and provides
    methods to interact with it for stock trading RL tasks.

    Example:
        >>> # Connect to a running server
        >>> client = FinRLEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.state)
        >>> print(result.observation.portfolio_value)
        >>>
        >>> # Execute a trading action
        >>> action = FinRLAction(actions=[0.5, -0.3])  # Buy stock 0, sell stock 1
        >>> result = client.step(action)
        >>> print(result.reward)
        >>> print(result.observation.portfolio_value)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = FinRLEnv.from_docker_image("finrl-env:latest")
        >>> result = client.reset()
        >>> result = client.step(FinRLAction(actions=[0.1]))
        >>> client.close()

    Example training loop:
        >>> import numpy as np
        >>> from envs.finrl_env import FinRLEnv, FinRLAction
        >>>
        >>> client = FinRLEnv(base_url="http://localhost:8000")
        >>>
        >>> # Training loop
        >>> for episode in range(10):
        >>>     result = client.reset()
        >>>     done = False
        >>>     episode_reward = 0
        >>>
        >>>     while not done:
        >>>         # Get state
        >>>         state = result.observation.state
        >>>
        >>>         # Simple random policy (replace with your RL agent)
        >>>         num_stocks = len(state) // 7  # Simplified calculation
        >>>         actions = np.random.uniform(-1, 1, size=num_stocks).tolist()
        >>>
        >>>         # Execute action
        >>>         result = client.step(FinRLAction(actions=actions))
        >>>
        >>>         episode_reward += result.reward or 0
        >>>         done = result.done
        >>>
        >>>     print(f"Episode {episode}: reward={episode_reward:.2f}, "
        >>>           f"final value={result.observation.portfolio_value:.2f}")
        >>>
        >>> client.close()
    """

    def get_config(self) -> Dict[str, Any]:
        """
        Get the environment configuration from the server.

        Returns:
            Dictionary containing environment configuration
        """
        response = self.session.get(f"{self.base_url}/config")
        response.raise_for_status()
        return response.json()

    def _step_payload(self, action: FinRLAction) -> Dict:
        """
        Convert FinRLAction to JSON payload for step request.

        Args:
            action: FinRLAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "actions": action.actions,
        }

    def _parse_result(self, payload: Dict) -> StepResult[FinRLObservation]:
        """
        Parse server response into StepResult[FinRLObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with FinRLObservation
        """
        obs_data = payload.get("observation", {})
        observation = FinRLObservation(
            state=obs_data.get("state", []),
            portfolio_value=obs_data.get("portfolio_value", 0.0),
            date=obs_data.get("date", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
