# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Maze Environment.

This module defines the Action, Observation, and State types for Maze games.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from core.env_server import Action, Observation, State


@dataclass
class MazeAction(Action):
    action: int


@dataclass
class MazeObservation(Observation):
    position: List[int]  # [row, col]
    total_reward: float
    legal_actions: List[int] = field(default_factory=list)

@dataclass
class MazeState(State):
    episode_id: str
    step_count: int
    done: bool = False
