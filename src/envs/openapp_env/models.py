# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the OpenApp Environment.

The OpenApp environment provides a simulated web application environment
for training and evaluating UI agents that interact with various apps
(calendar, todo, messenger, maps, etc.) using browser actions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from core.env_server.types import Action, Observation
except ImportError:
    # Standalone imports (when environment is standalone with openenv-core from pip)
    from openenv_core.env_server.types import Action, Observation


@dataclass(kw_only=True)
class OpenAppAction(Action):
    """
    Action for the OpenApp environment.

    Supports BrowserGym-style actions for web interaction:
    - click: Click on an element (requires bid - BrowserGym ID)
    - fill: Fill a text field (requires bid and text)
    - select_option: Select from dropdown (requires bid and value)
    - goto: Navigate to URL (requires url)
    - scroll: Scroll the page (requires direction)
    - send_keys: Send keyboard input (requires text)
    - noop: No operation

    Attributes:
        action_type: Type of action to perform
        bid: BrowserGym element ID (for click, fill, select_option)
        text: Text content (for fill, send_keys)
        value: Value to select (for select_option)
        url: URL to navigate to (for goto)
        direction: Scroll direction - 'up' or 'down' (for scroll)
        metadata: Additional action parameters
    """

    action_type: (
        str  # "click", "fill", "select_option", "goto", "scroll", "send_keys", "noop"
    )
    bid: Optional[str] = None  # BrowserGym element ID
    text: Optional[str] = None  # For fill or send_keys
    value: Optional[str] = None  # For select_option
    url: Optional[str] = None  # For goto
    direction: Optional[str] = None  # For scroll: "up" or "down"
    metadata: Dict[str, Any] = None  # Additional parameters

    def __post_init__(self):
        """Validate action parameters."""
        if self.metadata is None:
            self.metadata = {}

        # Validate required parameters for each action type
        if self.action_type == "click" and not self.bid:
            raise ValueError("click action requires 'bid' parameter")
        elif self.action_type == "fill" and (not self.bid or not self.text):
            raise ValueError("fill action requires 'bid' and 'text' parameters")
        elif self.action_type == "select_option" and (not self.bid or not self.value):
            raise ValueError(
                "select_option action requires 'bid' and 'value' parameters"
            )
        elif self.action_type == "goto" and not self.url:
            raise ValueError("goto action requires 'url' parameter")
        elif self.action_type == "scroll" and not self.direction:
            raise ValueError("scroll action requires 'direction' parameter")
        elif self.action_type == "send_keys" and not self.text:
            raise ValueError("send_keys action requires 'text' parameter")


@dataclass(kw_only=True)
class OpenAppObservation(Observation):
    """
    Observation from the OpenApp environment.

    Provides comprehensive state information about the web apps and browser state.

    Attributes:
        html: Current page HTML content
        url: Current page URL
        open_pages_urls: List of all open page URLs
        active_page_index: Index of currently active page
        screenshot: Base64-encoded screenshot (optional)
        axtree_txt: Accessibility tree as text (for element interaction)
        app_state: Current state of all apps (calendar, todo, messenger, map)
        task_info: Information about the current task (if any)
        last_action_error: Error message from last action (if failed)
    """

    html: str = ""
    url: str = ""
    open_pages_urls: List[str] = None
    active_page_index: int = 0
    screenshot: Optional[str] = None  # Base64-encoded
    axtree_txt: str = ""  # Accessibility tree
    app_state: Dict[str, Any] = None  # State of all apps
    task_info: Optional[Dict[str, Any]] = None  # Current task information
    last_action_error: Optional[str] = None  # Error from last action

    def __post_init__(self):
        """Initialize default values."""
        if self.open_pages_urls is None:
            self.open_pages_urls = []
        if self.app_state is None:
            self.app_state = {}
        if self.metadata is None:
            self.metadata = {}
