# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Android Environment HTTP Client.

This module provides the client for connecting to an Android Environment server
over HTTP.
"""

from typing import Any, Dict

from core.client_types import StepResult
from core.env_server.types import State
from core.http_env_client import HTTPEnvClient

from .models import AndroidAction, AndroidObservation


class AndroidEnv(HTTPEnvClient[AndroidAction, AndroidObservation]):
    """
    HTTP client for the Android Environment.

    This client connects to an AndroidEnvironment HTTP server running in a
    container with an Android emulator. It provides methods to interact with
    Android applications through touchscreen gestures.

    Example:
        >>> # Connect to a running server
        >>> client = AndroidEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.screen_width, result.observation.screen_height)
        >>>
        >>> # Tap on the screen
        >>> result = client.step(
        ...     AndroidAction(tool_name="tap", parameters={"x": 0.5, "y": 0.3})
        ... )
        >>> print(result.reward, result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = AndroidEnv.from_docker_image(
        ...     "android-env:latest",
        ...     environment={
        ...         "ANDROID_AVD_NAME": "Pixel_6_API_33",
        ...         "ANDROID_TASK_PATH": "/workspace/tasks/my_task.textproto"
        ...     }
        ... )
        >>> result = client.reset()
        >>> result = client.step(
        ...     AndroidAction(tool_name="tap", parameters={"x": 0.5, "y": 0.5})
        ... )
        >>> # View screen image (base64)
        >>> print(result.observation.screen_image[:50])  # First 50 chars
        >>> client.close()

    Example with high-level gestures:
        >>> # Swipe gesture
        >>> result = client.step(AndroidAction(
        ...     tool_name="swipe",
        ...     parameters={"x1": 0.5, "y1": 0.8, "x2": 0.5, "y2": 0.2}
        ... ))
        >>>
        >>> # Type text (if supported by task)
        >>> result = client.step(AndroidAction(
        ...     tool_name="type_text",
        ...     parameters={"text": "Hello Android"}
        ... ))
        >>>
        >>> # Press system button
        >>> result = client.step(AndroidAction(
        ...     tool_name="press_button",
        ...     parameters={"button": "HOME"}
        ... ))
    """

    def _step_payload(self, action: AndroidAction) -> Dict:
        """
        Convert AndroidAction to JSON payload for step request.

        Args:
            action: AndroidAction instance with tool_name and parameters.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "tool_name": action.tool_name,
            "parameters": action.parameters,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AndroidObservation]:
        """
        Parse server response into StepResult[AndroidObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with AndroidObservation containing screen state.
        """
        obs_data = payload.get("observation", {})

        observation = AndroidObservation(
            screen_image=obs_data.get("screen_image", ""),
            screen_width=obs_data.get("screen_width", 0),
            screen_height=obs_data.get("screen_height", 0),
            timestamp_ms=obs_data.get("timestamp_ms", 0),
            orientation=obs_data.get("orientation", 0),
            extras=obs_data.get("extras", {}),
            pixels_shape=obs_data.get("pixels_shape"),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=obs_data.get("reward"),
            done=obs_data.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            State object with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
