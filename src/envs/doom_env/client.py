# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Doom Environment HTTP Client.

This module provides the client for connecting to a Doom Environment server
over HTTP.
"""

from typing import Any, Dict

from openenv_core.client_types import StepResult
from openenv_core.env_server.types import State
from openenv_core.http_env_client import HTTPEnvClient

from .models import DoomAction, DoomObservation


class DoomEnv(HTTPEnvClient[DoomAction, DoomObservation]):
    """
    HTTP client for the Doom Environment.

    This client connects to a DoomEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    The Doom environment wraps ViZDoom scenarios for visual RL research.

    Example:
        >>> # Connect to a running server
        >>> client = DoomEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.screen_shape)
        >>>
        >>> # Take an action
        >>> result = client.step(DoomAction(action_id=2))
        >>> print(result.observation.reward, result.observation.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = DoomEnv.from_docker_image("doom-env:latest")
        >>> result = client.reset()
        >>> result = client.step(DoomAction(action_id=0))
        >>> client.close()
    """

    def _step_payload(self, action: DoomAction) -> Dict:
        """
        Convert DoomAction to JSON payload for step request.

        Args:
            action: DoomAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {}
        if action.action_id is not None:
            payload["action_id"] = action.action_id
        if action.buttons is not None:
            payload["buttons"] = action.buttons
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[DoomObservation]:
        """
        Parse server response into StepResult[DoomObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with DoomObservation
        """
        obs_data = payload.get("observation", {})
        observation = DoomObservation(
            screen_buffer=obs_data.get("screen_buffer", []),
            screen_shape=obs_data.get("screen_shape", [120, 160, 3]),
            game_variables=obs_data.get("game_variables"),
            available_actions=obs_data.get("available_actions"),
            episode_finished=obs_data.get("episode_finished", False),
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
