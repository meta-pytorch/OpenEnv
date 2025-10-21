# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ARE Environment Server Implementation.

This module wraps the ARE (Agents Research Environment) and exposes it
via the OpenEnv Environment interface.

Phase 3.1: Real ARE integration - Initialization
"""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from are.simulation.benchmark.scenario_loader import load_scenario
from are.simulation.environment import (
    Environment as ARESimulationEnvironment,
    EnvironmentConfig,
)
from are.simulation.scenarios.scenario import Scenario

from core.env_server import Action, Environment, Observation, State

from ..models import (
    AREAction,
    AREObservation,
    AREState,
    CallToolAction,
    GetStateAction,
    InitializeAction,
    ListAppsAction,
    TickAction,
)


class AREEnvironment(Environment):
    """
    ARE Environment wrapper for OpenEnv.

    This environment wraps the Agents Research Environment (ARE) which is
    event-driven and simulates scenarios involving apps and tool calling.

    Phase 1: This is a dummy implementation for testing the HTTP infrastructure.
    The real ARE integration will be added in Phase 2.

    Args:
        None (Phase 1 dummy implementation)

    Example:
        >>> env = AREEnvironment()
        >>> obs = env.reset()
        >>> print(obs.environment_state)  # "SETUP"
        >>> obs = env.step(InitializeAction(scenario_path="/path/to/scenario.json"))
        >>> print(obs.action_success)  # True
    """

    def __init__(self):
        """Initialize ARE environment."""
        super().__init__()

        # Phase 3.1: Real ARE integration
        self._are_env: Optional[ARESimulationEnvironment] = None
        self._scenario: Optional[Scenario] = None
        self._scenario_loaded = False

        # Track only episode-specific data not in ARE
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._scenario_path: Optional[str] = None

        # Notification buffer
        self._notifications_buffer = []

    def reset(self) -> Observation:
        """
        Reset the environment and return initial observation.

        Clears the ARE environment, scenario, and all state.

        Returns:
            Initial observation for the agent.
        """
        # Stop and clear ARE environment if it exists
        if self._are_env is not None:
            self._are_env.stop()
            self._are_env = None

        self._scenario = None
        self._scenario_loaded = False
        self._scenario_path = None

        # Reset episode tracking
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0

        # Clear buffers
        self._notifications_buffer = []

        return self._make_observation(
            action_success=True,
            action_result={"message": "ARE environment reset successfully"},
            environment_state="SETUP",
        )

    def step(self, action: Action) -> Observation:
        """
        Execute agent's action and return resulting observation.

        Args:
            action: One of the AREAction types (Initialize, Tick, ListApps, CallTool, GetState)

        Returns:
            Observation after action execution.

        Raises:
            ValueError: If action type is not recognized.
        """
        self._step_count += 1

        # Dispatch to appropriate handler based on action type
        if isinstance(action, InitializeAction):
            return self._handle_initialize(action)
        elif isinstance(action, TickAction):
            return self._handle_tick(action)
        elif isinstance(action, ListAppsAction):
            return self._handle_list_apps(action)
        elif isinstance(action, CallToolAction):
            return self._handle_call_tool(action)
        elif isinstance(action, GetStateAction):
            return self._handle_get_state(action)
        else:
            return self._make_observation(
                action_success=False,
                action_error=f"Unknown action type: {type(action)}",
            )

    @property
    def state(self) -> AREState:
        """
        Get current environment state.

        Builds AREState dynamically from self._are_env instead of maintaining
        a duplicate copy. Includes full ARE internal state.
        """
        # Build state dynamically from ARE environment
        if self._are_env:
            are_internal_state = self._are_env.get_state()
            current_time = self._are_env.current_time
            tick_count = self._are_env.tick_count
            env_state = self._are_env.state.value
        else:
            are_internal_state = None
            current_time = 0.0
            tick_count = 0
            env_state = "SETUP"

        return AREState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            scenario_loaded=self._scenario_loaded,
            scenario_path=self._scenario_path,
            current_time=current_time,
            tick_count=tick_count,
            environment_state=env_state,
            are_internal_state=are_internal_state,
        )

    def _handle_initialize(self, action: InitializeAction) -> AREObservation:
        """
        Handle initialize action - Load and initialize a scenario.

        Loads a scenario from either a JSON file path or a JSON string,
        initializes the ARE environment, and prepares it for execution.

        Args:
            action: InitializeAction with scenario_path and optional config

        Returns:
            Observation indicating success/failure of initialization
        """
        try:
            # Load scenario from path or JSON string
            scenario_json = None

            # Check if it's a file path or JSON string
            # First try to determine if it's a JSON string by checking for opening brace
            if action.scenario_path.strip().startswith("{"):
                # Assume it's a JSON string
                scenario_json = action.scenario_path
            else:
                # Try as file path
                scenario_path = Path(action.scenario_path)
                if scenario_path.exists() and scenario_path.suffix == ".json":
                    # Load from file
                    with open(scenario_path, "r") as f:
                        scenario_json = f.read()
                else:
                    return self._make_observation(
                        action_success=False,
                        action_error=f"File not found: {action.scenario_path}",
                        environment_state="FAILED",
                    )

            # Validate it's valid JSON
            try:
                json.loads(scenario_json)
            except json.JSONDecodeError as e:
                return self._make_observation(
                    action_success=False,
                    action_error=f"Invalid JSON: {str(e)}",
                    environment_state="FAILED",
                )

            # Use ARE's scenario loader
            scenario, _ = load_scenario(
                scenario_json,
                scenario_path=f"scenario_{self._episode_id}",
                load_completed_events=False,
            )

            if scenario is None:
                return self._make_observation(
                    action_success=False,
                    action_error="Failed to load scenario - scenario loader returned None",
                    environment_state="FAILED",
                )

            # Apply config overrides if provided
            if action.scenario_config:
                # TODO: Apply scenario config overrides
                pass

            # Initialize scenario
            scenario.initialize()
            self._scenario = scenario

            # Create ARE environment
            config = EnvironmentConfig(
                start_time=0,
                duration=None,  # No duration limit for now
                time_increment_in_seconds=scenario.time_increment_in_seconds,
                oracle_mode=True,  # We control events
                exit_when_no_events=False,
                queue_based_loop=False,  # Time-based
                verbose=False,
            )

            self._are_env = ARESimulationEnvironment(config=config)

            # Run scenario but don't start event loop yet
            self._are_env.run(scenario, wait_for_end=False, schedule_events=True)

            # Pause immediately to control ticking
            self._are_env.pause()

            # Update state
            self._scenario_loaded = True
            self._scenario_path = action.scenario_path

            return self._make_observation(
                action_success=True,
                action_result={
                    "scenario_id": scenario.scenario_id,
                    "duration": scenario.duration,
                    "time_increment": scenario.time_increment_in_seconds,
                },
                environment_state="RUNNING",
            )

        except Exception as e:
            return self._make_observation(
                action_success=False,
                action_error=f"Initialization failed: {str(e)}",
                environment_state="FAILED",
            )

    def _handle_tick(self, action: TickAction) -> AREObservation:
        """
        Handle tick action - Advance simulation time.

        Resumes the ARE environment, executes the specified number of ticks,
        then pauses again. Collects notifications generated during ticking.

        Args:
            action: TickAction with num_ticks parameter

        Returns:
            Observation after ticking with notifications and events processed
        """
        if not self._scenario_loaded or self._are_env is None:
            return self._make_observation(
                action_success=False,
                action_error="No scenario loaded. Call initialize first.",
                environment_state="SETUP",
            )

        try:
            # Capture event log length before ticking to track new events
            event_log_length_before = len(self._are_env.event_log)

            # Resume environment
            self._are_env.resume()

            # Execute ticks
            for _ in range(action.num_ticks):
                # Advance time manager by time_increment BEFORE ticking
                # so that tick() sees the new time when it calls time_manager.time()
                self._are_env.time_manager.add_offset(
                    self._are_env.time_increment_in_seconds
                )

                # Execute one tick of the event loop
                # This will update current_time from time_manager at the start
                self._are_env.tick()

                # Increment tick count (similar to _time_based_loop)
                self._are_env.tick_count += 1

            # Pause again
            self._are_env.pause()

            # Collect notifications generated during ticks
            self._collect_notifications()

            # Get newly executed events during this tick
            event_log_length_after = len(self._are_env.event_log)
            events_executed = event_log_length_after - event_log_length_before

            return self._make_observation(
                action_success=True,
                action_result={
                    "ticks_executed": action.num_ticks,
                    "events_executed": events_executed,
                    "notifications_generated": len(self._notifications_buffer),
                },
            )

        except Exception as e:
            return self._make_observation(
                action_success=False, action_error=f"Tick failed: {str(e)}"
            )

    def _handle_list_apps(self, action: ListAppsAction) -> AREObservation:
        """
        Handle list_apps action - List all available apps and their tools.

        Returns detailed information about each app and its tools, including
        tool names, descriptions, and parameter schemas.

        Args:
            action: ListAppsAction

        Returns:
            Observation with apps info in action_result
        """
        if not self._scenario_loaded or self._are_env is None:
            return self._make_observation(
                action_success=False,
                action_error="No scenario loaded. Call initialize first.",
                environment_state="SETUP",
            )

        try:
            apps_info = {}

            # Get apps - need to use get_app() method if apps is just names
            for app_or_name in self._are_env.apps:
                # Get actual app object
                if isinstance(app_or_name, str):
                    app = self._are_env.get_app(app_or_name)
                    if not app:
                        continue
                else:
                    app = app_or_name

                tools = app.get_tools()
                apps_info[app.name] = [tool.to_metadata_dict() for tool in tools]

            obs = self._make_observation(
                action_success=True,
                action_result={"apps": apps_info},
            )
            # Override available_apps specifically for this observation
            obs.available_apps = list(apps_info.keys())
            return obs
        except Exception as e:
            return self._make_observation(
                action_success=False, action_error=f"Failed to list apps: {str(e)}"
            )

    def _handle_call_tool(self, action: CallToolAction) -> AREObservation:
        """
        Handle call_tool action - Execute a tool on an app.

        Executes the specified tool on the specified app and optionally
        advances time by one tick.

        Args:
            action: CallToolAction with app_name, tool_name, tool_args, advance_time

        Returns:
            Observation with tool execution result
        """
        if not self._scenario_loaded or self._are_env is None:
            return self._make_observation(
                action_success=False,
                action_error="No scenario loaded. Call initialize first.",
                environment_state="SETUP",
            )

        try:
            # Import here to avoid circular dependency
            from are.simulation.types import Action as AREAction, Event, EventType

            # Get the app
            app = self._are_env.get_app(action.app_name)
            if app is None:
                return self._make_observation(
                    action_success=False,
                    action_error=f"App '{action.app_name}' not found",
                )

            # Get the tool
            tools = app.get_tools()
            tool = next((t for t in tools if t.name == action.tool_name), None)
            if tool is None:
                return self._make_observation(
                    action_success=False,
                    action_error=f"Tool '{action.tool_name}' not found on app '{action.app_name}'",
                )

            # Create and execute action
            are_action = AREAction(
                app=app, function=tool.function, args=action.tool_args
            )

            # Create event
            event = Event(
                event_id=f"openenv_{uuid.uuid4()}",
                event_time=self._are_env.current_time,
                event_type=EventType.AGENT,
                action=are_action,
            )

            # Execute event
            completed_event = event.execute()

            # Add to event log
            self._are_env.add_to_log(completed_event)

            # Get result - success is determined by absence of exception
            success = (
                completed_event.metadata.exception is None
                if completed_event.metadata
                else True
            )
            tool_result = {
                "success": success,
                "result": (
                    completed_event.metadata.return_value
                    if completed_event.metadata
                    else None
                ),
                "error": (
                    completed_event.metadata.exception
                    if completed_event.metadata
                    else None
                ),
            }

            # Optionally advance time
            events_executed = 0
            if action.advance_time:
                # Capture event log before tick
                event_log_length_before = len(self._are_env.event_log)

                # Resume, tick once, pause
                self._are_env.resume()
                
                # Advance time and tick
                self._are_env.time_manager.add_offset(
                    self._are_env.time_increment_in_seconds
                )
                self._are_env.tick()
                self._are_env.tick_count += 1
                
                self._are_env.pause()

                # Collect notifications
                self._collect_notifications()

                # Count events executed during tick
                event_log_length_after = len(self._are_env.event_log)
                events_executed = event_log_length_after - event_log_length_before

            # Add metadata to result
            tool_result["time_advanced"] = action.advance_time
            if action.advance_time:
                tool_result["events_executed"] = events_executed
                tool_result["notifications_generated"] = len(self._notifications_buffer)

            return self._make_observation(
                action_success=True, action_result=tool_result
            )

        except Exception as e:
            return self._make_observation(
                action_success=False, action_error=f"Tool execution failed: {str(e)}"
            )

    def _handle_get_state(self, action: GetStateAction) -> AREObservation:
        """
        Handle get_state action - Get detailed environment state.

        Returns detailed information about the environment state including
        event log, event queue, and app states based on requested fields.

        Args:
            action: GetStateAction with optional flags for what to include

        Returns:
            Observation with detailed state information
        """
        if not self._scenario_loaded or self._are_env is None:
            return self._make_observation(
                action_success=False,
                action_error="No scenario loaded. Call initialize first.",
                environment_state="SETUP",
            )

        try:
            state_info = {
                "current_time": self._are_env.current_time,
                "tick_count": self._are_env.tick_count,
                "environment_state": self._are_env.state.value,
                "duration": self._are_env.duration,
                "time_increment": self._are_env.time_increment_in_seconds,
            }

            if action.include_event_log:
                state_info["event_log"] = [
                    {
                        "event_id": e.event_id,
                        "event_time": e.event_time,
                        "event_type": e.event_type.value,
                        "success": (
                            e.metadata.exception is None if e.metadata else True
                        ),
                        "app_name": (
                            e.action.app.name
                            if e.action and hasattr(e.action, "app")
                            else None
                        ),
                        "return_value": (
                            e.metadata.return_value
                            if e.metadata and e.metadata.exception is None
                            else None
                        ),
                        "exception": (
                            str(e.metadata.exception) if e.metadata else None
                        ),
                    }
                    for e in self._are_env.event_log.list_view()
                ]
                state_info["event_log_summary"] = self._get_event_summary()

            if action.include_event_queue:
                state_info["event_queue"] = [
                    {
                        "event_id": e.event_id,
                        "event_time": e.event_time,
                        "event_type": e.event_type.value,
                        "app_name": (
                            e.action.app.name
                            if e.action and hasattr(e.action, "app")
                            else None
                        ),
                    }
                    for e in self._are_env.event_queue.list_view()
                ]
                state_info["queue_summary"] = self._get_queue_summary()

            if action.include_apps_state:
                apps_state = {}
                for app_or_name in self._are_env.apps:
                    # Handle both app objects and strings
                    if isinstance(app_or_name, str):
                        app = self._are_env.get_app(app_or_name)
                        if app:
                            app_state = app.get_state()
                            # Add tools info to app state
                            app_state["tools"] = [
                                tool.to_metadata_dict() for tool in app.get_tools()
                            ]
                            apps_state[app.name] = app_state
                    else:
                        app_state = app_or_name.get_state()
                        app_state["tools"] = [
                            tool.to_metadata_dict() for tool in app_or_name.get_tools()
                        ]
                        apps_state[app_or_name.name] = app_state
                state_info["apps_state"] = apps_state

            return self._make_observation(action_success=True, action_result=state_info)

        except Exception as e:
            return self._make_observation(
                action_success=False, action_error=f"Get state failed: {str(e)}"
            )

    def _collect_notifications(self) -> None:
        """
        Collect notifications from ARE notification system.

        Extracts all messages from the notification system's message queue
        up to the current simulation time and adds them to the notification buffer.
        """
        if self._are_env is None or self._are_env.notification_system is None:
            return

        try:
            from datetime import datetime, timezone

            current_time = self._are_env.current_time
            current_timestamp = datetime.fromtimestamp(current_time, tz=timezone.utc)

            # Extract all messages up to current time
            messages = self._are_env.notification_system.message_queue.get_by_timestamp(
                current_timestamp
            )

            for message in messages:
                notification = {
                    "type": message.message_type.value,
                    "message": message.message,
                    "timestamp": message.timestamp.isoformat(),
                }

                # Add attachments if present
                if message.attachments:
                    notification["attachments"] = message.attachments

                self._notifications_buffer.append(notification)

        except Exception as e:
            # Don't fail the whole action if notification collection fails
            pass

    def _get_event_summary(self, num_recent: int = 5) -> Dict[str, Any]:
        """
        Get a summary of recent events from the event log.

        Args:
            num_recent: Number of most recent events to include

        Returns:
            Dictionary containing event summary information
        """
        if self._are_env is None or len(self._are_env.event_log) == 0:
            return {
                "total_events": 0,
                "recent_events": [],
                "event_types": {},
            }

        events = self._are_env.event_log.list_view()
        recent = list(events[-num_recent:])

        # Count event types
        event_types = {}
        for event in events:
            event_type = event.event_type.value
            event_types[event_type] = event_types.get(event_type, 0) + 1

        return {
            "total_events": len(events),
            "recent_events": [
                {
                    "event_id": e.event_id,
                    "event_time": e.event_time,
                    "event_type": e.event_type.value,
                    "success": e.metadata.exception is None if e.metadata else True,
                    "app_name": (
                        e.action.app.name
                        if e.action and hasattr(e.action, "app")
                        else None
                    ),
                }
                for e in recent
            ],
            "event_types": event_types,
        }

    def _get_queue_summary(self, num_upcoming: int = 5) -> Dict[str, Any]:
        """
        Get a summary of upcoming events in the event queue.

        Args:
            num_upcoming: Number of upcoming events to include

        Returns:
            Dictionary containing event queue summary information
        """
        if self._are_env is None or len(self._are_env.event_queue) == 0:
            return {
                "total_queued": 0,
                "upcoming_events": [],
                "next_event_time": None,
            }

        events = self._are_env.event_queue.list_view()
        upcoming = list(events[:num_upcoming])

        return {
            "total_queued": len(events),
            "upcoming_events": [
                {
                    "event_id": e.event_id,
                    "event_time": e.event_time,
                    "event_type": e.event_type.value,
                }
                for e in upcoming
            ],
            "next_event_time": events[0].event_time if events else None,
        }

    def _make_observation(
        self,
        action_success: bool = True,
        action_result: Optional[Dict[str, Any]] = None,
        action_error: Optional[str] = None,
        environment_state: Optional[str] = None,
    ) -> AREObservation:
        """
        Create an AREObservation from current state.

        Args:
            action_success: Whether the action succeeded
            action_result: Result of the action
            action_error: Error message if action failed
            environment_state: Override environment state (for SETUP, FAILED)

        Returns:
            AREObservation for the agent.
        """
        # Get state from ARE environment if available
        if self._are_env:
            current_time = self._are_env.current_time
            tick_count = self._are_env.tick_count
            event_queue_length = len(self._are_env.event_queue)
            event_log_length = len(self._are_env.event_log)
            # Handle apps - might be strings or App objects
            available_apps = []
            for app in self._are_env.apps:
                if isinstance(app, str):
                    available_apps.append(app)
                else:
                    available_apps.append(
                        app.name if hasattr(app, "name") else str(app)
                    )
            env_state = environment_state or self._are_env.state.value
        else:
            current_time = 0.0
            tick_count = 0
            event_queue_length = 0
            event_log_length = 0
            available_apps = []
            env_state = environment_state or "SETUP"

        # Get notifications (don't clear buffer yet - let actions control that)
        notifications = self._notifications_buffer.copy()

        # Build enriched metadata with event and queue summaries
        metadata = {
            "scenario_loaded": self._scenario_loaded,
            "scenario_path": self._scenario_path,
        }

        # Add event summary if environment is initialized
        if self._are_env:
            metadata["event_summary"] = self._get_event_summary()
            metadata["queue_summary"] = self._get_queue_summary()

        obs = AREObservation(
            current_time=current_time,
            tick_count=tick_count,
            action_success=action_success,
            action_result=action_result,
            action_error=action_error,
            notifications=notifications,
            environment_state=env_state,
            event_queue_length=event_queue_length,
            event_log_length=event_log_length,
            available_apps=available_apps if available_apps else None,
            done=False,  # TODO: Determine done based on scenario completion
            reward=1.0 if action_success else 0.0,  # Simple reward
            metadata=metadata,
        )

        return obs
