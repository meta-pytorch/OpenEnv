# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Grid World Environment.
"""

from typing import Optional
from enum import Enum

from openenv.core.env_server.types import Action, Observation, State

class MoveAction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

class GridWorldAction(Action):
    """Action for the Grid World environment"""
    action: MoveAction

class GridWorldObservation(Observation):
    """Observation from the Grid World environment"""
    # have the value to pass directly to the action
    x: int = 0
    y: int = 0
    suggested_action: Optional[MoveAction] = None
    message: str = ""
    reward: Optional[float] = None
    done: bool = False

class GridWorldState(State):
    agent_x: int = 0
    agent_y: int = 0
    goal_x: int = 0
    goal_y: int = 0
    grid_size: int = 0
    episode_steps: int = 0
