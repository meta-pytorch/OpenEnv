# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Searchr1 Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv_core.env_server.interfaces import Environment
from openenv_core.env_server.types import State

from searchr1_env.models import Searchr1Action, Searchr1Observation


class Searchr1Environment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = Searchr1Environment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Searchr1 Env environment ready!"
        >>>
        >>> obs = env.step(Searchr1Action(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    def __init__(self):
        """Initialize the searchr1_env environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(self) -> Searchr1Observation:
        """
        Reset the environment.

        Returns:
            Searchr1Observation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return Searchr1Observation(
            echoed_message="Searchr1 Env environment ready!",
            message_length=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: Searchr1Action) -> Searchr1Observation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: Searchr1Action containing the message to echo

        Returns:
            Searchr1Observation with the echoed message and its length
        """
        self._state.step_count += 1

        message = action.message
        length = len(message)

        # Simple reward: longer messages get higher rewards
        reward = length * 0.1

        return Searchr1Observation(
            echoed_message=message,
            message_length=length,
            done=False,
            reward=reward,
            metadata={"original_message": message, "step": self._state.step_count},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

