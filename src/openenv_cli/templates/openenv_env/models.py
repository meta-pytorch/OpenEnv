from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """Base Action for __ENV_NAME__."""
    pass


@dataclass
class Observation:
    """Base Observation for __ENV_NAME__."""
    message: str = ""


@dataclass
class State:
    """Episode state."""
    episode_id: str = ""
    step_count: int = 0


