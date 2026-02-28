# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HTTP client for generic Gymnasium environments served over HTTP."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.client_types import StepResult

from core.http_env_client import HTTPEnvClient

from .models import GymAction, GymObservation, GymState

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class GymEnvironment(HTTPEnvClient[GymAction, GymObservation]):
    """Client for interacting with Gymnasium environments over HTTP.

    Example:
        >>> client = GymEnvironment(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.state)
        >>> result = client.step(GymAction(action=1))
        >>> print(result.reward, result.done)

    Example with Docker:
        >>> client = GymEnvironment.from_docker_image("generic-gym-env:latest")
        >>> _ = client.reset()
        >>> _ = client.step(GymAction(action=0))
    """

    def _step_payload(self, action: GymAction) -> Dict[str, Any]:
        """
        Convert GymAction to JSON payload for step request.

        Args:
            action: GymAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        payload: Dict[str, Any] = {"action": action.action}
        if action.metadata:
            payload["metadata"] = action.metadata
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[GymObservation]:
        """
        Parse server response into StepResult[GymObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with GymObservation.
        """
        obs_data = payload.get("observation", {})

        observation = GymObservation(
            state=obs_data.get("state"),
            legal_actions=obs_data.get("legal_actions"),
            episode_length=obs_data.get("episode_length", 0),
            total_reward=obs_data.get("total_reward", 0.0),
            done=bool(payload.get("done", False)),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> GymState:
        """
        Parse server response into GymState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            GymState object with environment state information.
        """
        return GymState(
            env_id=payload.get("env_id", "Unknown"),
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            render_mode=payload.get("render_mode"),
            max_steps=payload.get("max_steps"),
            seed=payload.get("seed"),
            episode_length=payload.get("episode_length", 0),
            total_reward=payload.get("total_reward", 0.0),
        )
