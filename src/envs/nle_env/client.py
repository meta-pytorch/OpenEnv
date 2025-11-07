# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NetHack Learning Environment HTTP Client.

This module provides the client for connecting to an NLE Environment server
over HTTP.
"""

from typing import Dict

from core.client_types import StepResult
from core.http_env_client import HTTPEnvClient

from .models import NLEAction, NLEObservation, NLEState


class NLEEnv(HTTPEnvClient[NLEAction, NLEObservation]):
    """
    HTTP client for the NetHack Learning Environment.

    This client connects to an NLEEnvironment HTTP server and provides
    methods to interact with NetHack: reset(), step(), and state access.

    With beefy compute, we use simple JSON serialization. The server sends
    all observation arrays as nested lists, which we keep as-is or convert
    back to numpy arrays as needed.

    Example:
        >>> # Connect to a running server
        >>> client = NLEEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.blstats)  # [HP, MaxHP, ...]
        >>>
        >>> # Take a step (move north)
        >>> result = client.step(NLEAction(action_id=0))
        >>> print(result.reward)
        >>> print(result.done)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = NLEEnv.from_docker_image("nle-env:latest")
        >>> result = client.reset()
        >>>
        >>> # Play NetHack!
        >>> for _ in range(100):
        ...     action = NLEAction(action_id=random.randint(0, 112))
        ...     result = client.step(action)
        ...     if result.done:
        ...         break
    """

    def _step_payload(self, action: NLEAction) -> Dict:
        """
        Convert NLEAction to JSON payload for step request.

        Args:
            action: NLEAction instance with action_id

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_id": action.action_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[NLEObservation]:
        """
        Parse server response into StepResult[NLEObservation].

        The server sends all arrays as nested lists. With beefy compute,
        we just keep them as lists - no need to convert back to numpy
        unless the user specifically needs it.

        Args:
            payload: JSON response from server

        Returns:
            StepResult with NLEObservation
        """
        obs_data = payload.get("observation", {})

        # Extract standard fields
        done = obs_data.get("done", False)
        reward = obs_data.get("reward")
        metadata = obs_data.get("metadata", {})

        # Build observation with all the array fields
        # Keep them as lists - simple and works great with beefy compute
        observation = NLEObservation(
            # Core observations
            glyphs=obs_data.get("glyphs"),
            blstats=obs_data.get("blstats"),
            message=obs_data.get("message"),
            # Visual observations
            chars=obs_data.get("chars"),
            colors=obs_data.get("colors"),
            specials=obs_data.get("specials"),
            # Inventory observations
            inv_glyphs=obs_data.get("inv_glyphs"),
            inv_strs=obs_data.get("inv_strs"),
            inv_letters=obs_data.get("inv_letters"),
            inv_oclasses=obs_data.get("inv_oclasses"),
            # Terminal observations
            tty_chars=obs_data.get("tty_chars"),
            tty_colors=obs_data.get("tty_colors"),
            tty_cursor=obs_data.get("tty_cursor"),
            # Extended observations
            screen_descriptions=obs_data.get("screen_descriptions"),
            program_state=obs_data.get("program_state"),
            internal=obs_data.get("internal"),
            misc=obs_data.get("misc"),
            # Standard fields
            done=done,
            reward=reward,
            metadata=metadata,
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict) -> NLEState:
        """
        Parse server response into NLEState object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            NLEState object with episode and game information
        """
        return NLEState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            game_over=payload.get("game_over", False),
            end_status=payload.get("end_status", "RUNNING"),
            in_normal_game=payload.get("in_normal_game", False),
            character=payload.get("character", "mon-hum-neu-mal"),
            task_name=payload.get("task_name", "NetHackScore-v0"),
        )
