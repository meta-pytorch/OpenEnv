"""Data models for the BrowserGym environment.

BrowserGym is a unified framework for web-based agent tasks, combining multiple
benchmarks including MiniWoB (training), WebArena (evaluation), VisualWebArena,
and more under a single Gymnasium-compatible API.
"""

from dataclasses import dataclass
from typing import List, Optional

from openenv_core.env_server.types import Action, Observation, State


@dataclass(kw_only=True)
class BrowserGymAction(Action):
    """Action to be executed in the BrowserGym environment.

    BrowserGym supports high-level natural language actions that can be parsed
    into browser operations.

    Example actions:
    - "click('Submit button')"
    - "fill('username', 'john@example.com')"
    - "goto('https://example.com')"
    - "scroll(down)"
    - "send_keys('Enter')"
    """

    action_str: str
    """Natural language action string (e.g., "click('Submit')")"""


@dataclass(kw_only=True)
class BrowserGymObservation(Observation):
    """Observation returned from the BrowserGym environment.

    Contains multiple observation modalities including text (accessibility tree
    or DOM), visual (screenshot), and page metadata.
    """

    text: str = ""
    """Text representation of the page (accessibility tree or DOM)"""

    url: str = ""
    """Current URL of the page"""

    screenshot: Optional[List[List[List[int]]]] = None
    """Screenshot as numpy array [height, width, channels] (if visual observation enabled)"""

    goal: str = ""
    """Task goal/instruction for the current episode"""

    axtree_txt: str = ""
    """Full accessibility tree as text"""

    pruned_html: str = ""
    """Pruned HTML content (interactive elements only)"""

    error: str = ""
    """Error message if action execution failed"""

    last_action_error: bool = False
    """Whether the last action resulted in an error"""


@dataclass
class BrowserGymState(State):
    """State of the BrowserGym environment.

    Tracks the current benchmark, task, and progress through an episode.
    """

    benchmark: str = ""
    """Benchmark name (e.g., 'miniwob', 'webarena', 'visualwebarena')"""

    task_name: str = ""
    """Specific task within the benchmark (e.g., 'click-test', 'click-button')"""

    task_id: Optional[str] = None
    """Task ID for evaluation benchmarks (e.g., WebArena task number)"""

    goal: str = ""
    """Task goal/instruction"""

    current_url: str = ""
    """Current URL of the active page"""

    max_steps: Optional[int] = None
    """Maximum steps allowed for this task"""

    cum_reward: float = 0.0
    """Cumulative reward for the current episode"""
