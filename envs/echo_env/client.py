# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Echo Environment Client.

This module provides the client for connecting to an Echo Environment server
via WebSocket for persistent sessions.
"""

from typing import Any, Dict

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    from openenv.core.env_client import EnvClient
    from .models import EchoAction, EchoObservation
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    from openenv.core.env_client import EnvClient
    from models import EchoAction, EchoObservation


class EchoEnv(EnvClient[EchoAction, EchoObservation, State]):
    """
    Client for the Echo Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with EchoEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(EchoAction(message="Hello!"))
        ...     print(result.observation.echoed_message)
        ...     print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = EchoEnv.from_docker_image("echo-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(EchoAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: EchoAction) -> Dict:
        """
        Convert EchoAction to JSON payload for step request.

        Args:
            action: EchoAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[EchoObservation]:
        """
        Parse server response into StepResult[EchoObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with EchoObservation
        """
        obs_data = payload.get("observation", {})
        observation = EchoObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
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
