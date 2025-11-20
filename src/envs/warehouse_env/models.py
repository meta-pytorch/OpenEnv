"""
Data models for Warehouse Optimization Environment.

This module defines the Action, Observation, and State dataclasses
for the warehouse logistics optimization environment.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from core.env_server import Action, Observation, State
except ImportError:
    # Standalone imports (when environment is standalone with openenv-core from pip)
    from openenv_core.env_server import Action, Observation, State


@dataclass
class WarehouseAction(Action):
    """
    Action for the warehouse robot.

    Actions:
        0: MOVE_UP - Move robot one cell up
        1: MOVE_DOWN - Move robot one cell down
        2: MOVE_LEFT - Move robot one cell left
        3: MOVE_RIGHT - Move robot one cell right
        4: PICK_UP - Pick up package at current location
        5: DROP_OFF - Drop off package at current location
    """

    action_id: int  # 0-5

    # Action names for reference
    ACTION_NAMES = {
        0: "MOVE_UP",
        1: "MOVE_DOWN",
        2: "MOVE_LEFT",
        3: "MOVE_RIGHT",
        4: "PICK_UP",
        5: "DROP_OFF",
    }

    def __post_init__(self):
        """Validate action ID."""
        if self.action_id not in range(6):
            raise ValueError(f"action_id must be 0-5, got {self.action_id}")

    @property
    def action_name(self) -> str:
        """Get human-readable action name."""
        return self.ACTION_NAMES.get(self.action_id, "UNKNOWN")


@dataclass
class Package:
    """Represents a package in the warehouse."""

    id: int
    status: str  # "waiting", "picked", "delivered"
    pickup_location: tuple[int, int]
    dropoff_location: tuple[int, int]
    priority: int  # 1 (low), 2 (medium), 3 (high)
    time_waiting: int  # Steps since created


@dataclass(kw_only=True)
class WarehouseObservation(Observation):
    """
    Observation returned after each step in the warehouse environment.

    Attributes:
        grid: 2D list representing the warehouse layout
              0=empty, 1=wall, 2=shelf, 3=pickup_zone, 4=dropoff_zone
        robot_position: Current (x, y) position of the robot
        robot_carrying: Package ID if carrying, None otherwise
        packages: List of all packages and their states
        step_count: Current step number in episode
        packages_delivered: Number of packages successfully delivered
        total_packages: Total number of packages in episode
        time_remaining: Steps remaining before timeout
        action_success: Whether the last action was successful
        message: Human-readable message about last action
    """

    grid: List[List[int]]
    robot_position: tuple[int, int]
    robot_carrying: Optional[int]
    packages: List[Dict[str, Any]]
    step_count: int
    packages_delivered: int
    total_packages: int
    time_remaining: int
    action_success: bool
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class WarehouseState(State):
    """
    Episode state tracking for the warehouse environment.

    Attributes:
        episode_id: Unique identifier for this episode
        step_count: Number of steps taken
        packages_delivered: Packages successfully delivered
        total_packages: Total packages in episode
        difficulty_level: Difficulty setting (1-5)
        grid_size: (width, height) of warehouse
        cum_reward: Cumulative reward for episode
        is_done: Whether episode has ended
    """

    episode_id: str
    step_count: int
    packages_delivered: int
    total_packages: int
    difficulty_level: int
    grid_size: tuple[int, int]
    cum_reward: float
    is_done: bool
