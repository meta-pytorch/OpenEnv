# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Tennis Environment.

This module defines the Action, Observation, and State types for the Tennis game
via the Arcade Learning Environment (ALE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.env_server import Action, Observation, State


@dataclass
class TennisAction(Action):
    """
    Action for Tennis environment.

    Attributes:
        action_id: The integer action ID to take (0-17 for Tennis full action space).
        action_name: Human-readable action name (e.g., "UP", "DOWNLEFTFIRE").
        player_id: Optional player ID for multi-agent mode (None for single-agent).
    """
    action_id: int
    action_name: str = ""
    player_id: Optional[str] = None


@dataclass
class TennisObservation(Observation):
    """
    Observation from Tennis environment.

    This represents what the agent sees after taking an action.

    Attributes:
        screen_rgb: Screen observation as a flattened list of RGB pixels.
                   Shape: [210, 160, 3] flattened to 100,800 values.
        screen_shape: Original shape of the screen [210, 160, 3].
        score: Current game score as (player_score, opponent_score).
        ball_side: Simplified ball position - "left", "center", "right", or "unknown".
        my_position: Agent's position - "top", "middle", "bottom", or "unknown".
        opponent_position: Opponent's position - "top", "middle", "bottom", or "unknown".
        legal_actions: List of legal action IDs the agent can take.
        action_meanings: Human-readable names for each action.
        rally_length: Number of consecutive hits in current rally.
        lives: Number of lives remaining (typically not used in tennis).
        episode_frame_number: Frame number within current episode.
        frame_number: Total frame number since environment creation.
    """
    screen_rgb: List[int]
    screen_shape: List[int]
    score: Tuple[int, int]
    ball_side: str
    my_position: str
    opponent_position: str
    legal_actions: List[int]
    action_meanings: List[str]
    rally_length: int = 0
    lives: int = 0
    episode_frame_number: int = 0
    frame_number: int = 0


@dataclass
class TennisState(State):
    """
    State for Tennis environment.

    Tracks game-specific state information beyond basic episode tracking.

    Attributes:
        episode_id: Unique identifier for current episode.
        step_count: Number of steps taken in current episode.
        previous_score: Score from previous step, for detecting scoring events.
        rally_length: Number of steps since last point scored.
        total_points: Total points scored in current episode (by both players).
        agent_games_won: Number of games won by agent in current set.
        opponent_games_won: Number of games won by opponent in current set.
    """
    previous_score: Tuple[int, int] = (0, 0)
    rally_length: int = 0
    total_points: int = 0
    agent_games_won: int = 0
    opponent_games_won: int = 0
