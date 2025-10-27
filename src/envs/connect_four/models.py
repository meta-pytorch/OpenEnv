from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ConnectFourAction(BaseModel):
    column: int = Field(..., ge=0, le=6, description="Playable column 0..6")


class ConnectFourObservation(BaseModel):
    # 6x7 int grid: 0 empty, +1 agent discs, -1 opponent discs
    board: List[List[int]]
    # list of playable columns (0..6), empty when done=True
    legal_actions: List[int]
    # +1 if agent (player 0) to move, -1 otherwise
    current_player: int
    # last column played, or None at the start
    last_move: Optional[int] = None
    # terminal flag
    done: bool
    # scalar reward in agent’s perspective: +1 win, -1 loss, 0 else
    reward: float
    # passthrough metadata
    info: Dict[str, Any] = {}


class ConnectFourState(BaseModel):
    rows: int = 6
    cols: int = 7
    move_count: int = 0
    episode_id: str = ""
