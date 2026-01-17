# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grid World Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import GridWorldAction, GridWorldObservation, GridWorldState, MoveAction


class GridWorldEnv(
    EnvClient[GridWorldAction, GridWorldObservation, GridWorldState]
):
    """
    Client for the Grid World Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with GridWorldEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.message)
        ...
        ...     result = client.step(GridWorldAction(action="UP"))
        ...     print(result.observation.message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = GridWorldEnv.from_docker_image("grid_world-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(GridWorldAction(action="UP"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: GridWorldAction) -> Dict:
        """
        Convert GridWorldAction to JSON payload for step message.

        Args:
            action: GridWorldAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {"action": action.action}

    def _parse_result(self, payload: Dict) -> StepResult[GridWorldObservation]:
        """
        Parse server response into StepResult[GridWorldObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with GridWorldObservation
        """
        obs_data = payload.get("observation", {})
        
        suggested_action = obs_data.get("suggested_action")
        if suggested_action is not None:
            suggested_action = MoveAction(suggested_action)

        observation = GridWorldObservation(
            x=obs_data.get("x", ""),
            y=obs_data.get("y", ""),
            suggested_action=suggested_action,
            message=obs_data.get("message", ""),
            reward=payload.get("reward"),
            done=payload.get("done", False),
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
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return GridWorldState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            agent_x=payload.get("agent_x", 0),
            agent_y=payload.get("agent_y", 0),
            goal_x=payload.get("goal_x", 0),
            goal_y=payload.get("goal_y", 0),
            grid_size=payload.get("grid_size", 0),
            episode_steps=payload.get("episode_steps", 0)
        )
