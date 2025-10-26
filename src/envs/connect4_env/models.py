"""Data models for the Connect4 OpenEnv environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.env_server.types import Action, Observation, State


@dataclass(kw_only=True)
class Connect4Action(Action):
    """Selects the column (0-indexed) where the agent wants to drop a disc."""

    column: int


@dataclass(kw_only=True)
class Connect4Observation(Observation):
    """Observation returned after every step/reset."""

    board: List[List[int]]  # 6x7 grid with 1 (agent), -1 (opponent), 0 (empty)
    legal_actions: List[int]
    current_player: int
    last_move: Optional[int] = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Connect4State(State):
    """Track episode metadata plus board geometry for convenience."""

    rows: int = 6
    cols: int = 7
