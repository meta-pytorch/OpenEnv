# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for ARE Environment.

This module defines the Action, Observation, and State types for the
Agents Research Environment (ARE) integration with OpenEnv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

from core.env_server import Action, Observation, State


# Action Types (Union type for different action kinds)


@dataclass
class InitializeAction(Action):
    """
    Initialize the ARE environment with a scenario.

    Attributes:
        action_type: Literal "initialize"
        scenario_path: Path to scenario JSON/YAML file
        scenario_config: Optional override for scenario configuration
    """

    action_type: Literal["initialize"] = "initialize"
    scenario_path: str = ""
    scenario_config: Optional[Dict[str, Any]] = None


@dataclass
class TickAction(Action):
    """
    Advance simulation time by ticking.

    Attributes:
        action_type: Literal "tick"
        num_ticks: Number of ticks to advance (default 1)
    """

    action_type: Literal["tick"] = "tick"
    num_ticks: int = 1


@dataclass
class ListAppsAction(Action):
    """
    Get available apps and their tools.

    Attributes:
        action_type: Literal "list_apps"
    """

    action_type: Literal["list_apps"] = "list_apps"


@dataclass
class CallToolAction(Action):
    """
    Call a tool on a specific app.

    Attributes:
        action_type: Literal "call_tool"
        app_name: Name of the app
        tool_name: Name of the tool/function
        tool_args: Arguments for the tool
        advance_time: Whether to tick after tool call (default True)
    """

    action_type: Literal["call_tool"] = "call_tool"
    app_name: str = ""
    tool_name: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    advance_time: bool = True


@dataclass
class GetStateAction(Action):
    """
    Get detailed environment state.

    Attributes:
        action_type: Literal "get_state"
        include_event_log: Include event log in response
        include_event_queue: Include event queue in response
        include_apps_state: Include apps state in response
    """

    action_type: Literal["get_state"] = "get_state"
    include_event_log: bool = True
    include_event_queue: bool = False
    include_apps_state: bool = True


# Union type for all actions
AREAction = Union[
    InitializeAction,
    TickAction,
    ListAppsAction,
    CallToolAction,
    GetStateAction,
]


@dataclass
class AREObservation(Observation):
    """
    Observation from ARE environment.

    This represents what the agent sees after taking an action.

    Attributes:
        current_time: Current simulation time
        tick_count: Number of ticks elapsed
        action_success: Whether the action succeeded
        action_result: Result of the action (tool call result, app list, etc.)
        action_error: Error message if action failed
        notifications: List of notification events since last observation
        environment_state: State of the environment (SETUP, RUNNING, PAUSED, STOPPED, FAILED)
        event_queue_length: Number of events in the queue
        event_log_length: Number of events in the log
        available_apps: List of app names
        metadata: Additional metadata
    """

    current_time: float = 0.0
    tick_count: int = 0
    action_success: bool = True
    action_result: Optional[Dict[str, Any]] = None
    action_error: Optional[str] = None
    notifications: List[Dict[str, Any]] = field(default_factory=list)
    environment_state: str = "SETUP"
    event_queue_length: int = 0
    event_log_length: int = 0
    available_apps: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AREState(State):
    """
    State for ARE environment.

    This extends the base State with ARE-specific information,
    including the full internal state from ARE's get_state() method.

    Attributes:
        scenario_loaded: Whether a scenario is currently loaded
        scenario_path: Path to the loaded scenario (if any)
        current_time: Current simulation time
        tick_count: Number of ticks elapsed
        environment_state: State of the environment (SETUP, RUNNING, PAUSED, etc.)
        are_internal_state: Full internal state from ARE environment's get_state()
                           Includes: apps state, event_log, event_queue, start_time,
                           duration, time_increment_in_seconds
    """

    scenario_loaded: bool = False
    scenario_path: Optional[str] = None
    current_time: float = 0.0
    tick_count: int = 0
    environment_state: str = "SETUP"
    are_internal_state: Optional[Dict[str, Any]] = None
