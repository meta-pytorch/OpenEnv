# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ARE Environment HTTP Client.

This module provides the client for connecting to an ARE Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.http_env_client import HTTPEnvClient
from core.client_types import StepResult

from .models import (
    AREAction,
    AREObservation,
    AREState,
    CallToolAction,
    GetStateAction,
    InitializeAction,
    ListAppsAction,
    TickAction,
)

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class AREEnv(HTTPEnvClient[AREAction, AREObservation]):
    """
    HTTP client for ARE Environment.

    This client connects to an AREEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = AREEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.environment_state)
        >>>
        >>> # Initialize a scenario
        >>> action = InitializeAction(scenario_path="/path/to/scenario.json")
        >>> result = client.step(action)
        >>> print(result.observation.action_success)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = AREEnv.from_docker_image("are-env:latest")
        >>> result = client.reset()
        >>> result = client.step(TickAction(num_ticks=5))
    """

    def _step_payload(self, action: AREAction) -> Dict[str, Any]:
        """
        Convert AREAction to JSON payload for step request.

        Args:
            action: One of the AREAction types.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        # Handle different action types
        if isinstance(action, InitializeAction):
            return {
                "action_type": "initialize",
                "scenario_path": action.scenario_path,
                "scenario_config": action.scenario_config,
            }
        elif isinstance(action, TickAction):
            return {
                "action_type": "tick",
                "num_ticks": action.num_ticks,
            }
        elif isinstance(action, ListAppsAction):
            return {
                "action_type": "list_apps",
            }
        elif isinstance(action, CallToolAction):
            return {
                "action_type": "call_tool",
                "app_name": action.app_name,
                "tool_name": action.tool_name,
                "tool_args": action.tool_args,
                "advance_time": action.advance_time,
            }
        elif isinstance(action, GetStateAction):
            return {
                "action_type": "get_state",
                "include_event_log": action.include_event_log,
                "include_event_queue": action.include_event_queue,
                "include_apps_state": action.include_apps_state,
            }
        else:
            raise ValueError(f"Unknown action type: {type(action)}")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AREObservation]:
        """
        Parse server response into StepResult[AREObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with AREObservation.
        """
        obs_data = payload.get("observation", {})

        observation = AREObservation(
            current_time=obs_data.get("current_time", 0.0),
            tick_count=obs_data.get("tick_count", 0),
            action_success=obs_data.get("action_success", True),
            action_result=obs_data.get("action_result"),
            action_error=obs_data.get("action_error"),
            notifications=obs_data.get("notifications", []),
            environment_state=obs_data.get("environment_state", "SETUP"),
            event_queue_length=obs_data.get("event_queue_length", 0),
            event_log_length=obs_data.get("event_log_length", 0),
            available_apps=obs_data.get("available_apps"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AREState:
        """
        Parse server response into AREState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            AREState object with environment state information.
        """
        return AREState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            scenario_loaded=payload.get("scenario_loaded", False),
            scenario_path=payload.get("scenario_path"),
            current_time=payload.get("current_time", 0.0),
            tick_count=payload.get("tick_count", 0),
            environment_state=payload.get("environment_state", "SETUP"),
        )
