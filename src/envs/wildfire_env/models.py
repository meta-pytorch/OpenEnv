from dataclasses import dataclass, field
from typing import List, Optional

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from core.env_server import Action, Observation, State
except ImportError:
    # Standalone imports (when environment is standalone with openenv-core from pip)
    from openenv_core.env_server import Action, Observation, State

# Grid cell encoding:
# 0 = empty/ash, 1 = fuel (healthy), 2 = burning, 3 = firebreak, 4 = watered (damp)
# (You can tweak encodings, but keep them ints for compact obs.)

@dataclass
class WildfireAction(Action):
    # action: "break" (build firebreak), "water" (drop water), "wait"
    action: str
    x: Optional[int] = None
    y: Optional[int] = None

@dataclass
class WildfireObservation(Observation):
    grid: List[int]                 # flattened grid H*W, ints in {0..4}
    width: int
    height: int
    step: int
    wind_dir: str                   # e.g. "N","NE","E","SE","S","SW","W","NW","CALM"
    humidity: float                 # [0,1]
    burning_count: int
    burned_count: int               # total ash (0) cells (cumulative)
    reward_hint: float = 0.0        # optional shaping info
    remaining_water: int = 0
    remaining_breaks: int = 0

@dataclass
class WildfireState(State):
    episode_id: str = ""
    step_count: int = 0
    total_burned: int = 0
    total_extinguished: int = 0
    last_action: str = "reset"
    # For visibility / debugging (not required by core):
    width: int = 0
    height: int = 0
    wind_dir: str = "CALM"
    humidity: float = 0.25
    remaining_water: int = 20       # simple resource constraint
    remaining_breaks: int = 50
    # internal full grid as flattened ints
    grid: List[int] = field(default_factory=list)
    # burn timers for each cell (track how long cells have been burning/damp)
    burn_timers: List[int] = field(default_factory=list)
