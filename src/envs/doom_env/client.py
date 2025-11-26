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

from typing import Any, Dict, Optional

import numpy as np

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

    Example with rendering:
        >>> client = DoomEnv.from_docker_image("doom-env:latest")
        >>> result = client.reset()
        >>> for _ in range(100):
        >>>     result = client.step(DoomAction(action_id=1))
        >>>     client.render()  # Display the game
        >>> client.close()
    """

    def __init__(self, *args, **kwargs):
        """Initialize DoomEnv client."""
        super().__init__(*args, **kwargs)
        self._render_window = None
        self._last_observation = None

    def _step_payload(self, action: DoomAction) -> Dict:
        """
        Convert DoomAction to JSON payload for step request.

        Args:
            action: DoomAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        # Use dataclasses.asdict to ensure proper serialization
        from dataclasses import asdict

        # Convert to dict and filter out None values
        action_dict = asdict(action)

        # Convert numpy types to native Python types for JSON serialization
        result = {}
        for k, v in action_dict.items():
            if v is None:
                continue
            # Handle numpy integers and floats
            if hasattr(v, 'item'):  # numpy scalar types
                result[k] = v.item()
            # Handle numpy arrays/lists
            elif isinstance(v, (list, tuple)):
                result[k] = [x.item() if hasattr(x, 'item') else x for x in v]
            else:
                result[k] = v

        return result

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

        # Store for rendering
        self._last_observation = observation

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

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the current observation.

        Args:
            mode: Render mode - "human" for window display, "rgb_array" for array return.

        Returns:
            RGB array if mode is "rgb_array", None otherwise.
        """
        if self._last_observation is None:
            print("Warning: No observation to render. Call reset() or step() first.")
            return None

        # Get screen from observation
        screen_buffer = self._last_observation.screen_buffer
        screen_shape = self._last_observation.screen_shape

        if not screen_buffer or not screen_shape:
            return None

        # Reshape screen buffer to original dimensions
        screen = np.array(screen_buffer, dtype=np.uint8).reshape(screen_shape)

        if mode == "rgb_array":
            return screen
        elif mode == "human":
            # Display using cv2 or matplotlib
            try:
                import cv2

                # Create window if it doesn't exist
                if self._render_window is None:
                    self._render_window = "ViZDoom - Doom Environment"
                    cv2.namedWindow(self._render_window, cv2.WINDOW_NORMAL)

                # Convert to BGR for OpenCV (if RGB)
                if len(screen.shape) == 3 and screen.shape[2] == 3:
                    screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                else:
                    screen_bgr = screen

                # Display
                cv2.imshow(self._render_window, screen_bgr)
                cv2.waitKey(1)

            except ImportError:
                # Fallback to matplotlib
                try:
                    import matplotlib.pyplot as plt

                    if self._render_window is None:
                        plt.ion()
                        self._render_window = plt.figure(figsize=(8, 6))
                        self._render_window.canvas.manager.set_window_title(
                            "ViZDoom - Doom Environment"
                        )

                    plt.clf()
                    if len(screen.shape) == 3:
                        plt.imshow(screen)
                    else:
                        plt.imshow(screen, cmap="gray")
                    plt.axis("off")
                    plt.pause(0.001)

                except ImportError:
                    print(
                        "Warning: Neither cv2 nor matplotlib available for rendering. "
                        "Install with: pip install opencv-python or pip install matplotlib"
                    )
            return None
        else:
            raise ValueError(
                f"Invalid render mode: {mode}. Use 'human' or 'rgb_array'."
            )

    def close(self) -> None:
        """Close the environment and clean up resources."""
        # Close render window if it exists
        if self._render_window is not None:
            try:
                import cv2

                cv2.destroyAllWindows()
            except ImportError:
                try:
                    import matplotlib.pyplot as plt

                    plt.close("all")
                except ImportError:
                    pass
            self._render_window = None

        # Call parent close
        super().close()
