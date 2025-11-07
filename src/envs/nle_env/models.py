# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the NetHack Learning Environment (NLE).

The NLE environment wraps the NetHack 3.6.6 game as a reinforcement learning
environment, providing rich observations and a complex action space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.env_server import Action, Observation, State


@dataclass
class NLEAction(Action):
    """
    Action for the NetHack Learning Environment.

    Uses discrete action space where action_id maps to NetHack commands
    (movement, interactions, etc.). The action space has ~113 actions.

    Examples:
        - action_id=0: Move North (k)
        - action_id=1: Move East (l)
        - action_id=37: Eat (e)
        - action_id=50: Search (s)
    """

    action_id: int  # Index into nethack.USEFUL_ACTIONS (0-112)


@dataclass
class NLEObservation(Observation):
    """
    Observation from the NetHack Learning Environment.

    Contains a subset of NLE's 14+ observation types. All numpy arrays are
    serialized as nested lists for JSON compatibility.

    Observation types (all optional, configured at env creation):
        - glyphs: (21, 79) - Symbolic dungeon map representation
        - chars: (21, 79) - ASCII character display
        - colors: (21, 79) - Color codes for display
        - specials: (21, 79) - Special attributes
        - blstats: (26,) - Bottom-line stats (HP, XP, gold, etc.)
        - message: (256,) - Game message as byte array
        - inv_glyphs: (55,) - Inventory item glyphs
        - inv_strs: (55, 80) - Inventory item descriptions
        - inv_letters: (55,) - Inventory item letters (a-z, A-Z)
        - inv_oclasses: (55,) - Inventory object classes
        - tty_chars: (24, 80) - Full terminal character display
        - tty_colors: (24, 80) - Full terminal colors
        - tty_cursor: (2,) - Terminal cursor position [row, col]
        - screen_descriptions: (21, 79, 80) - Text descriptions of dungeon

    With beefy compute, we include all observations by default.
    """

    # Core observations (always useful)
    glyphs: Optional[List[List[int]]] = None
    blstats: Optional[List[int]] = None
    message: Optional[List[int]] = None

    # Visual observations
    chars: Optional[List[List[int]]] = None
    colors: Optional[List[List[int]]] = None
    specials: Optional[List[List[int]]] = None

    # Inventory observations
    inv_glyphs: Optional[List[int]] = None
    inv_strs: Optional[List[List[int]]] = None
    inv_letters: Optional[List[int]] = None
    inv_oclasses: Optional[List[int]] = None

    # Terminal observations (for rendering)
    tty_chars: Optional[List[List[int]]] = None
    tty_colors: Optional[List[List[int]]] = None
    tty_cursor: Optional[List[int]] = None

    # Extended observations
    screen_descriptions: Optional[List[List[List[int]]]] = None
    program_state: Optional[List[int]] = None
    internal: Optional[List[int]] = None
    misc: Optional[List[int]] = None


@dataclass
class NLEState(State):
    """
    Extended state for the NLE environment.

    Includes NetHack-specific state information beyond basic episode tracking.
    """

    # NLE-specific state
    game_over: bool = False
    end_status: str = "RUNNING"  # RUNNING, DEATH, TASK_SUCCESSFUL, ABORTED
    in_normal_game: bool = False
    character: str = "mon-hum-neu-mal"  # role-race-gender-alignment

    # Task-specific info
    task_name: str = "NetHackScore-v0"
