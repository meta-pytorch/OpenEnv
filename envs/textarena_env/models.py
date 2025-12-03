# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Common data models for the TextArena environment wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State


@dataclass
class TextArenaMessage:
    """Single message observed by a player."""

    sender_id: int
    content: str
    category: str


@dataclass(kw_only=True)
class TextArenaAction(Action):
    """Action issued by the agent for TextArena games."""

    message: str


@dataclass(kw_only=True)
class TextArenaObservation(Observation):
    """Observation returned from any TextArena game."""

    prompt: str
    messages: List[TextArenaMessage] = field(default_factory=list)
    current_player_id: int = 0
    legal_players: List[int] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class TextArenaState(State):
    """Structured state snapshot for the server."""

    env_id: str
    num_players: int
    max_turns: Optional[int] = None
    turn: int = 0
    last_reward: float = 0.0
    last_info: Dict[str, Any] = field(default_factory=dict)
    raw_state: Dict[str, Any] = field(default_factory=dict)

