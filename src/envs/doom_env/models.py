# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Doom Environment.

The doom_env environment wraps ViZDoom for reinforcement learning research.
ViZDoom is a Doom-based AI research platform for visual RL.
"""

from dataclasses import dataclass
from typing import List, Optional

from openenv_core.env_server.types import Action, Observation


@dataclass(kw_only=True)
class DoomAction(Action):
    """
    Action for the Doom environment.

    Actions are specified as a list of button states. Each element corresponds to
    a button (e.g., MOVE_LEFT, MOVE_RIGHT, ATTACK, etc.) with value 0 (not pressed)
    or 1 (pressed).

    Attributes:
        buttons: List of button states (0 or 1). The length and meaning depend on
                 the available buttons in the scenario.
        action_id: Optional pre-defined action ID if using discrete action space.
                   If provided, this will be used instead of buttons.

    Example:
        # Use discrete action (move left)
        DoomAction(action_id=0)

        # Use button combination (move forward and shoot)
        DoomAction(buttons=[1, 0, 0, 1])
    """

    buttons: Optional[List[int]] = None
    action_id: Optional[int] = None


@dataclass(kw_only=True)
class DoomObservation(Observation):
    """
    Observation from the Doom environment.

    Contains the screen buffer, game variables, and episode information.

    Attributes:
        screen_buffer: Flattened screen pixels as a list of integers.
                      Shape is [height, width, channels] before flattening.
        screen_shape: Original shape of the screen [height, width, channels].
        game_variables: Current values of game variables (health, ammo, etc.).
        available_actions: List of available action IDs if using discrete actions.
        episode_finished: Whether the episode has ended.
    """

    screen_buffer: List[int]
    screen_shape: List[int]
    game_variables: List[float] = None
    available_actions: List[int] = None
    episode_finished: bool = False

