# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Gesture and action utilities for Android environment.

This module provides helper classes for composing complex gestures
from primitive touch events.
"""

import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class TouchPoint:
    """A point in a touch gesture with timing."""
    x: float  # Normalized x coordinate [0, 1]
    y: float  # Normalized y coordinate [0, 1]
    duration_ms: int = 100  # How long to hold this position


class GestureBuilder:
    """Helper class for building complex gestures from touch primitives."""

    @staticmethod
    def tap(x: float, y: float, duration_ms: int = 100) -> List[dict]:
        """Create a tap gesture (touch + lift).

        Args:
            x: Normalized x coordinate [0, 1]
            y: Normalized y coordinate [0, 1]
            duration_ms: How long to hold the touch

        Returns:
            List of action dicts representing the tap sequence
        """
        return [
            {"action_type": 0, "x": x, "y": y, "duration_ms": duration_ms},  # TOUCH
            {"action_type": 1, "x": x, "y": y, "duration_ms": 50},  # LIFT
        ]

    @staticmethod
    def swipe(
        x1: float, y1: float, x2: float, y2: float,
        duration_ms: int = 300, steps: int = 10
    ) -> List[dict]:
        """Create a swipe gesture from (x1, y1) to (x2, y2).

        Args:
            x1, y1: Start position (normalized [0, 1])
            x2, y2: End position (normalized [0, 1])
            duration_ms: Total duration of the swipe
            steps: Number of intermediate points

        Returns:
            List of action dicts representing the swipe sequence
        """
        actions = []
        step_duration = duration_ms // steps

        # Touch down at start
        actions.append({"action_type": 0, "x": x1, "y": y1, "duration_ms": step_duration})

        # Move through intermediate points
        for i in range(1, steps):
            t = i / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            actions.append({"action_type": 2, "x": x, "y": y, "duration_ms": step_duration})  # REPEAT

        # Lift at end
        actions.append({"action_type": 1, "x": x2, "y": y2, "duration_ms": 50})

        return actions

    @staticmethod
    def long_press(x: float, y: float, duration_ms: int = 1000) -> List[dict]:
        """Create a long press gesture.

        Args:
            x, y: Position (normalized [0, 1])
            duration_ms: How long to hold

        Returns:
            List of action dicts representing the long press
        """
        return [
            {"action_type": 0, "x": x, "y": y, "duration_ms": duration_ms},  # TOUCH
            {"action_type": 1, "x": x, "y": y, "duration_ms": 50},  # LIFT
        ]

    @staticmethod
    def double_tap(x: float, y: float, gap_ms: int = 100) -> List[dict]:
        """Create a double tap gesture.

        Args:
            x, y: Position (normalized [0, 1])
            gap_ms: Time between taps

        Returns:
            List of action dicts representing the double tap
        """
        actions = []

        # First tap
        actions.extend(GestureBuilder.tap(x, y, duration_ms=100))

        # Gap (represented as a REPEAT at same position)
        actions.append({"action_type": 2, "x": x, "y": y, "duration_ms": gap_ms})

        # Second tap
        actions.extend(GestureBuilder.tap(x, y, duration_ms=100))

        return actions

    @staticmethod
    def scroll_down(x: float = 0.5, distance: float = 0.5, duration_ms: int = 300) -> List[dict]:
        """Scroll down (swipe up).

        Args:
            x: Horizontal position (normalized [0, 1])
            distance: How far to scroll (normalized [0, 1])
            duration_ms: Duration of scroll

        Returns:
            List of action dicts representing the scroll
        """
        y_start = 0.7
        y_end = max(0.2, y_start - distance)
        return GestureBuilder.swipe(x, y_start, x, y_end, duration_ms=duration_ms)

    @staticmethod
    def scroll_up(x: float = 0.5, distance: float = 0.5, duration_ms: int = 300) -> List[dict]:
        """Scroll up (swipe down).

        Args:
            x: Horizontal position (normalized [0, 1])
            distance: How far to scroll (normalized [0, 1])
            duration_ms: Duration of scroll

        Returns:
            List of action dicts representing the scroll
        """
        y_start = 0.3
        y_end = min(0.8, y_start + distance)
        return GestureBuilder.swipe(x, y_start, x, y_end, duration_ms=duration_ms)

    @staticmethod
    def swipe_left(y: float = 0.5, distance: float = 0.5, duration_ms: int = 300) -> List[dict]:
        """Swipe left.

        Args:
            y: Vertical position (normalized [0, 1])
            distance: How far to swipe (normalized [0, 1])
            duration_ms: Duration of swipe

        Returns:
            List of action dicts representing the swipe
        """
        x_start = 0.7
        x_end = max(0.2, x_start - distance)
        return GestureBuilder.swipe(x_start, y, x_end, y, duration_ms=duration_ms)

    @staticmethod
    def swipe_right(y: float = 0.5, distance: float = 0.5, duration_ms: int = 300) -> List[dict]:
        """Swipe right.

        Args:
            y: Vertical position (normalized [0, 1])
            distance: How far to swipe (normalized [0, 1])
            duration_ms: Duration of swipe

        Returns:
            List of action dicts representing the swipe
        """
        x_start = 0.3
        x_end = min(0.8, x_start + distance)
        return GestureBuilder.swipe(x_start, y, x_end, y, duration_ms=duration_ms)


class ADBCommands:
    """Helper class for ADB commands."""

    @staticmethod
    def text_input(text: str) -> str:
        """Generate ADB command for text input.

        Args:
            text: Text to input

        Returns:
            ADB command string
        """
        # Escape special characters for ADB
        # Use double quotes and escape backslashes, double quotes, and spaces
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace(" ", "%s")
        return f'input text "{escaped}"'

    @staticmethod
    def keyevent(keycode: str) -> str:
        """Generate ADB command for key event.

        Args:
            keycode: Android keycode (e.g., "KEYCODE_HOME", "KEYCODE_BACK")

        Returns:
            ADB command string
        """
        return f"input keyevent {keycode}"

    @staticmethod
    def tap_coordinates(x: int, y: int) -> str:
        """Generate ADB command for tap at pixel coordinates.

        Args:
            x, y: Pixel coordinates

        Returns:
            ADB command string
        """
        return f"input tap {x} {y}"

    @staticmethod
    def swipe_coordinates(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> str:
        """Generate ADB command for swipe.

        Args:
            x1, y1: Start pixel coordinates
            x2, y2: End pixel coordinates
            duration_ms: Duration in milliseconds

        Returns:
            ADB command string
        """
        return f"input swipe {x1} {y1} {x2} {y2} {duration_ms}"

    # Common Android keycodes
    KEYCODE_HOME = "KEYCODE_HOME"
    KEYCODE_BACK = "KEYCODE_BACK"
    KEYCODE_MENU = "KEYCODE_MENU"
    KEYCODE_SEARCH = "KEYCODE_SEARCH"
    KEYCODE_ENTER = "KEYCODE_ENTER"
    KEYCODE_DEL = "KEYCODE_DEL"
    KEYCODE_VOLUME_UP = "KEYCODE_VOLUME_UP"
    KEYCODE_VOLUME_DOWN = "KEYCODE_VOLUME_DOWN"
    KEYCODE_POWER = "KEYCODE_POWER"
    KEYCODE_CAMERA = "KEYCODE_CAMERA"
    KEYCODE_TAB = "KEYCODE_TAB"
    KEYCODE_SPACE = "KEYCODE_SPACE"
