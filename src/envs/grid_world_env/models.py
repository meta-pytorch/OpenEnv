
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.env_server import Action, Observation, State

# --- Action Models ---
class MoveAction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

@dataclass
class GridWorldAction(Action):
    action: MoveAction

# --- Observation Model ---
@dataclass
class GridWorldObservation(Observation):
    x: int = 0
    y: int = 0
    message: str = ""
    reward: Optional[float] = None
    done: bool = False

# --- State Model ---
@dataclass
class GridWorldState(State):
    agent_x: int = 0
    agent_y: int = 0
    goal_x: int = 0
    goal_y: int = 0
    grid_size: int = 0
    episode_steps: int = 0