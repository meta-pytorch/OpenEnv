# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Gymnasium-based environments.

This module defines generic Action, Observation, and State representations
used by the Gym environment integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from core.env_server import Action, Observation, State


@dataclass
class GymAction(Action):
    """Generic action wrapper for Gymnasium environments."""

    action: Any


@dataclass
class GymObservation(Observation):
    """Observation returned by a Gymnasium environment."""

    state: Any
    legal_actions: Optional[Any] = None
    episode_length: int = 0
    total_reward: float = 0.0


@dataclass
class GymState(State):
    """Server-side state snapshot for Gymnasium environments."""

    env_id: str = "Unknown"
    render_mode: Optional[str] = None
    max_steps: Optional[int] = None
    seed: Optional[int] = None
    episode_length: int = 0
    total_reward: float = 0.0
