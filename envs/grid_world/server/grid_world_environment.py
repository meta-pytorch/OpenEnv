# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Grid World Environment Implementation.

Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from grid_world.models import (
    GridWorldAction,
    GridWorldObservation,
    GridWorldState,
    MoveAction,
)


class GridWorldEnvironment(Environment):
    """
    A grid environment to reach the goal pose.
    This environment is designed for testing the HTTP server infrastructure.
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the grid_world environment."""
        super().__init__()
        
        self.grid_size = 5
        self.goal_pos = [4, 4]

        self._state = GridWorldState(
            agent_x=0,
            agent_y=0,
            goal_x=self.goal_pos[0],
            goal_y=self.goal_pos[1],
            grid_size=self.grid_size,
            episode_steps=0
        )

    def reset(self) -> GridWorldObservation:
        """
        Reset the environment.

        Returns:
            GridWorldObservation
        """
        self._state.episode_id = str(uuid4())
        self._state.step_count = 0

        # Update State
        self._state.agent_x = 0
        self._state.agent_y = 0
        self._state.episode_steps = 0

        return GridWorldObservation(
            x=self._state.agent_x,
            y=self._state.agent_y,
            suggested_action=self._next_action(self._state.agent_x, self._state.agent_y),
            message="Welcome to Grid World! Goal is at [4, 4].",
            reward=None,
            done=False
        )

    def step(self, action: GridWorldAction) -> GridWorldObservation:  # type: ignore[override]
        """
        Execute a step in the environment.

        Args:
            action: GridWorldAction 

        Returns:
            GridWorldObservation 
        """
        self._state.step_count += 1
        self._state.episode_steps += 1

        current_x = self._state.agent_x
        current_y = self._state.agent_y

        move = action.action
        if move == MoveAction.UP:
            current_x -= 1
        elif move == MoveAction.DOWN:
            current_x += 1
        elif move == MoveAction.LEFT:
            current_y -= 1
        elif move == MoveAction.RIGHT:
            current_y += 1
        
        # Update state
        self._state.agent_x = current_x
        self._state.agent_y = current_y

        reward, done, message = self._reward(current_x, current_y)

        return GridWorldObservation(
            x=current_x,
            y=current_y,
            suggested_action=self._next_action(current_x, current_y),
            reward=reward,
            done=done,
            message=message
        )

    def _next_action(self, current_x, current_y):
        if [current_x, current_y] == self.goal_pos:
            return None
        if current_x < self.goal_pos[0]:
            return MoveAction.DOWN
        if current_x > self.goal_pos[0]:
            return MoveAction.UP
        if current_y < self.goal_pos[1]:
            return MoveAction.RIGHT
        return MoveAction.LEFT

    def _reward(self, current_x, current_y):
        reward = 0.0
        done = False
        message = "Keep going..."

        if [current_x, current_y] == self.goal_pos:
            reward = 1.0
            done = True
            message = "Found the goal!"
        else:
            reward = -0.1
        
        return reward, done, message
        

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
