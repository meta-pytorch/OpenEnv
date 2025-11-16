"""Data models for custom BrowserGym tasks.

These models are used specifically for custom tasks that are not part of the
official BrowserGym benchmarks (miniwob, webarena, visualwebarena, workarena).
"""

import sys
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Add src directory to path for core imports
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from core.env_server.types import Action, Observation, State


@dataclass(kw_only=True)
class CustomGymAction(Action):
    """Action to be executed in a custom BrowserGym environment.

    Custom actions support the same BrowserGym action format but may include
    additional custom fields specific to your task.

    Example actions:
    - "click('Submit button')"
    - "fill('username', 'john@example.com')"
    - "goto('https://example.com')"
    - "scroll(down)"
    - "send_keys('Enter')"
    """

    action_str: str
    """Natural language action string (e.g., "click('Submit')")"""

    metadata: Optional[Dict[str, Any]] = None
    """Optional metadata for custom task-specific data"""


@dataclass(kw_only=True)
class CustomGymObservation(Observation):
    """Observation returned from a custom BrowserGym environment.

    Contains multiple observation modalities including text (accessibility tree
    or DOM), visual (screenshot), and page metadata, plus custom fields.
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

    custom_data: Optional[Dict[str, Any]] = None
    """Optional custom data specific to your task"""


@dataclass
class CustomGymState(State):
    """State of a custom BrowserGym environment.

    Tracks the current task, and progress through an episode, plus custom state fields.
    """

    benchmark: str = "custom"
    """Benchmark name (always 'custom' for custom tasks)"""

    task_name: str = ""
    """Specific custom task name (e.g., 'copy-paste', 'data-entry')"""

    task_id: Optional[str] = None
    """Task ID for custom task tracking"""

    goal: str = ""
    """Task goal/instruction"""

    current_url: str = ""
    """Current URL of the active page"""

    max_steps: Optional[int] = None
    """Maximum steps allowed for this task"""

    cum_reward: float = 0.0
    """Cumulative reward for the current episode"""

    custom_state: Optional[Dict[str, Any]] = None
    """Optional custom state data specific to your task"""
