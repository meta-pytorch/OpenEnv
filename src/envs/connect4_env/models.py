# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Connect4 Environment.

This module defines the Action, Observation, and State types for Connect4 games
via the OpenEnv interface.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import List

from core.env_server import Action, Observation, State


@dataclass
class Connect4Action(Action):
    """
    Action for Connect4 environment.

    Attributes:
        column: The column index (0 to 6) where the piece will be placed.
    """
    column: int


@dataclass(kw_only=True)
class Connect4Observation(Observation):
    """
    Observation for Connect4 environment.

    Attributes:
        board: The current board as a 2D list (6 rows x 7 columns).
               1 = current player, -1 = opponent, 0 = empty.
        legal_actions: List of column indices that are valid moves.
        done: Whether the game is over.
        reward: Reward for the last action.
    """
    
    board: List[List[int]]
    legal_actions: List[int]
    done: bool = False
    reward: float = 0.0
    metadata: dict = field(default_factory=dict)
    


@dataclass(kw_only=True)
class Connect4State(State):
    """
    State for Connect4 environment.

    Attributes:
        episode_id: Unique ID for the current game.
        board: Current board state (rows x columns), 0 = empty, 1 = player, -1 = opponent.
        next_player: Whose turn it is (1 or -1).
        step_count: Number of steps taken in the game.
    """
    episode_id: str
    board: List[List[int]] = field(default_factory=lambda: np.zeros((6,7), dtype=int).tolist())
    next_player: int = 1
    step_count: int = 0
