# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Android Environment.

The Android environment provides access to Android applications and the
Android OS through a touchscreen interface. Actions represent touch events
and gestures, while observations contain screen pixels and metadata.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.env_server.types import Action, Observation


@dataclass(kw_only=True)
class AndroidAction(Action):
    """Action for the Android environment.

    Supports multiple interaction types following RFC 004's ToolCallAction pattern.

    Examples:
        # Tap at specific coordinates
        AndroidAction(
            tool_name="tap",
            parameters={"x": 0.5, "y": 0.3}
        )

        # Swipe gesture
        AndroidAction(
            tool_name="swipe",
            parameters={"x1": 0.2, "y1": 0.5, "x2": 0.8, "y2": 0.5, "duration_ms": 300}
        )

        # Type text
        AndroidAction(
            tool_name="type_text",
            parameters={"text": "Hello World"}
        )

        # Press system button
        AndroidAction(
            tool_name="press_button",
            parameters={"button": "HOME"}  # HOME, BACK, MENU, etc.
        )

        # Raw touch event (for advanced control)
        AndroidAction(
            tool_name="touch_event",
            parameters={
                "action_type": "TOUCH",  # TOUCH, LIFT, REPEAT
                "touch_position": [0.5, 0.3],  # normalized [0, 1]
                "duration_ms": 100
            }
        )
    """

    tool_name: str  # Action type: "tap", "swipe", "type_text", "press_button", "touch_event"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class AndroidObservation(Observation):
    """Observation from the Android environment.

    Contains the current screen state as an image plus additional metadata
    about the Android system and task state.

    Attributes:
        screen_image: Base64-encoded image (JPEG or PNG) of current screen.
        screen_width: Width of the screen in pixels.
        screen_height: Height of the screen in pixels.
        timestamp_ms: Timestamp of the observation in milliseconds.
        orientation: Screen orientation (0, 90, 180, 270 degrees).
        extras: Additional task-specific information (e.g., accessibility tree,
                current app package, system state).
    """

    screen_image: str  # Base64-encoded image
    screen_width: int
    screen_height: int
    timestamp_ms: int = 0
    orientation: int = 0  # degrees: 0, 90, 180, 270

    # Task extras from android_env (accessibility info, package names, etc.)
    extras: Dict[str, Any] = field(default_factory=dict)

    # Optional: Include raw pixels shape for reference
    pixels_shape: Optional[tuple[int, int, int]] = None  # (height, width, channels)
