# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tennis Environment HTTP Client.

This module provides the client for connecting to a Tennis Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.client_types import StepResult
from core.http_env_client import HTTPEnvClient

from .models import TennisAction, TennisObservation, TennisState

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class TennisEnv(HTTPEnvClient[TennisAction, TennisObservation]):
    """
    HTTP client for Tennis Environment.

    This client connects to a TennisEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = TennisEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.score)
        >>>
        >>> # Take an action
        >>> result = client.step(TennisAction(action_id=2, action_name="UP"))
        >>> print(result.reward, result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = TennisEnv.from_docker_image("tennis-env:latest")
        >>> result = client.reset()
        >>> result = client.step(TennisAction(action_id=0, action_name="NOOP"))
    """

    def _step_payload(self, action: TennisAction) -> Dict[str, Any]:
        """
        Convert TennisAction to JSON payload for step request.

        Args:
            action: TennisAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "action_id": action.action_id,
            "action_name": action.action_name,
            "player_id": action.player_id,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[TennisObservation]:
        """
        Parse server response into StepResult[TennisObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with TennisObservation.
        """
        obs_data = payload.get("observation", {})

        observation = TennisObservation(
            screen_rgb=obs_data.get("screen_rgb", []),
            screen_shape=obs_data.get("screen_shape", [210, 160, 3]),
            score=tuple(obs_data.get("score", [0, 0])),
            ball_side=obs_data.get("ball_side", "unknown"),
            my_position=obs_data.get("my_position", "unknown"),
            opponent_position=obs_data.get("opponent_position", "unknown"),
            legal_actions=obs_data.get("legal_actions", []),
            action_meanings=obs_data.get("action_meanings", []),
            rally_length=obs_data.get("rally_length", 0),
            lives=obs_data.get("lives", 0),
            episode_frame_number=obs_data.get("episode_frame_number", 0),
            frame_number=obs_data.get("frame_number", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> TennisState:
        """
        Parse server response into TennisState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            TennisState object with environment state information.
        """
        return TennisState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            previous_score=tuple(payload.get("previous_score", [0, 0])),
            rally_length=payload.get("rally_length", 0),
            total_points=payload.get("total_points", 0),
            agent_games_won=payload.get("agent_games_won", 0),
            opponent_games_won=payload.get("opponent_games_won", 0),
        )
