"""Data models for custom BrowserGym tasks.

These models are used specifically for custom tasks that are not part of the
official BrowserGym benchmarks (miniwob, webarena, visualwebarena, workarena).
"""

from typing import List, Optional, Dict, Any
from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


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

    action_str: str = Field(..., description="Natural language action string")


class CustomGymObservation(Observation):
    """Observation returned from a custom BrowserGym environment.

    Contains multiple observation modalities including text (accessibility tree
    or DOM), visual (screenshot), and page metadata, plus custom fields.
    """

    text: str = Field(default="", description="Text representation of the page")
    url: str = Field(default="", description="Current URL of the page")
    screenshot: Optional[List[List[List[int]]]] = Field(
        default=None, description="Screenshot as array [height, width, channels]"
    )
    goal: str = Field(default="", description="Task goal/instruction")
    axtree_txt: str = Field(default="", description="Full accessibility tree as text")
    pruned_html: str = Field(default="", description="Pruned HTML content")
    error: str = Field(default="", description="Error message if action failed")
    last_action_error: bool = Field(
        default=False, description="Whether last action resulted in error"
    )
    custom_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional custom task-specific data"
    )


class CustomGymState(State):
    """State of a custom BrowserGym environment.

    Tracks the current task and progress through an episode, plus custom state fields.
    """

    benchmark: str = Field(
        default="custom", description="Benchmark name (always 'custom')"
    )
    task_name: str = Field(default="", description="Specific custom task name")
    task_id: Optional[str] = Field(
        default=None, description="Task ID for custom task tracking"
    )
    goal: str = Field(default="", description="Task goal/instruction")
    current_url: str = Field(default="", description="Current URL of the active page")
    max_steps: Optional[int] = Field(
        default=None, description="Maximum steps allowed for this task"
    )
    cum_reward: float = Field(
        default=0.0, description="Cumulative reward for the current episode"
    )
    custom_state: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional custom state data"
    )
