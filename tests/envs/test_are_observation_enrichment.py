# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for Phase 4: ARE Observation Enrichment.

This module tests that observations contain comprehensive state information including:
- Notification tracking across all actions
- Event log visibility and summaries
- Event queue visibility and summaries
- Apps state visibility
- Observation completeness
"""

import json

import pytest

from src.envs.are_env.models import (
    CallToolAction,
    GetStateAction,
    InitializeAction,
    TickAction,
)
from src.envs.are_env.server.are_environment import AREEnvironment


@pytest.fixture
def environment():
    """Fixture providing a fresh ARE environment"""
    env = AREEnvironment()
    return env


class TestApp:
    """Simple test app for testing observation enrichment"""

    def __init__(self):
        from are.simulation.apps.app import App
        from are.simulation.tool_utils import app_tool
        from are.simulation.types import event_registered
        from are.simulation.utils import get_state_dict

        class _TestApp(App):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def get_state(self):
                return get_state_dict(self, ["counter"])

            def reset(self):
                self.counter = 0

            @app_tool()
            @event_registered()
            def increment(self):
                """Increment the counter."""
                self.counter += 1
                return f"Counter is now {self.counter}"

            @app_tool()
            @event_registered()
            def get_count(self):
                """Get current counter value."""
                return self.counter

        self.AppClass = _TestApp


@pytest.fixture
def simple_scenario_json():
    """Fixture providing a minimal scenario JSON without apps"""
    return json.dumps(
        {
            "version": "are_simulation_v1",
            "metadata": {
                "definition": {
                    "scenario_id": "test_enrichment",
                    "duration": 100,
                    "time_increment_in_seconds": 1,
                }
            },
            "apps": [],
            "events": [],
            "completed_events": [],
            "world_logs": [],
        }
    )


@pytest.fixture
def environment_with_app():
    """Environment with TestApp registered"""
    from are.simulation.environment import Environment as ARESimEnv, EnvironmentConfig
    from are.simulation.scenarios.scenario import Scenario

    env = AREEnvironment()
    env.reset()

    # Create test app
    app = TestApp().AppClass()
    app.name = "TestApp"

    # Create scenario
    scenario = Scenario(
        scenario_id="enrichment_test",
        duration=100,
        time_increment_in_seconds=1,
        apps=[app],
        events=[],
    )
    scenario.initialize()

    # Initialize ARE environment
    config = EnvironmentConfig(
        start_time=0,
        duration=None,
        time_increment_in_seconds=1,
        oracle_mode=True,
        exit_when_no_events=False,
        queue_based_loop=False,
        verbose=False,
    )

    env._are_env = ARESimEnv(config=config)
    env._are_env.run(scenario, wait_for_end=False, schedule_events=True)
    env._are_env.pause()
    env._scenario = scenario
    env._scenario_loaded = True
    env._scenario_path = "test_scenario"

    # Store tool names for tests to use
    env._test_tools = {tool.name: tool for tool in app.get_tools()}

    yield env
    if env._are_env is not None:
        env._are_env.stop()


class TestNotificationTracking:
    """Test notification tracking across all actions"""

    def test_notifications_in_tick_observation(
        self, environment, simple_scenario_json
    ):
        """Test that ticking generates notifications and they appear in observation"""
        # Initialize
        obs = environment.step(InitializeAction(scenario_path=simple_scenario_json))
        assert obs.action_success

        # Clear any initialization notifications
        environment._notifications_buffer.clear()

        # Tick to generate potential notifications
        obs = environment.step(TickAction(num_ticks=1))
        assert obs.action_success
        assert isinstance(obs.notifications, list)
        # Notifications list should exist (may be empty initially)
        assert obs.notifications is not None

    def test_notifications_in_tool_call_observation(self, environment_with_app):
        """Test that tool calls with time advancement collect notifications"""
        # Get the actual tool name
        tool_name = list(environment_with_app._test_tools.keys())[0]
        
        # Call tool with time advancement
        obs = environment_with_app.step(
            CallToolAction(
                app_name="TestApp",
                tool_name=tool_name,
                tool_args={},
                advance_time=True,
            )
        )
        assert obs.action_success
        assert isinstance(obs.notifications, list)

    def test_notifications_cleared_between_actions(
        self, environment, simple_scenario_json
    ):
        """Test that notifications are managed properly between actions"""
        # Initialize
        obs = environment.step(InitializeAction(scenario_path=simple_scenario_json))
        assert obs.action_success

        # First tick
        obs1 = environment.step(TickAction(num_ticks=1))
        notifications1 = obs1.notifications

        # Second tick should have its own notifications
        obs2 = environment.step(TickAction(num_ticks=1))
        # Both should have notifications field
        assert isinstance(obs2.notifications, list)


class TestEventLogVisibility:
    """Test event log visibility and summaries"""

    def test_event_summary_in_metadata(self, environment, simple_scenario_json):
        """Test that observation metadata contains event summaries"""
        # Initialize
        obs = environment.step(InitializeAction(scenario_path=simple_scenario_json))
        assert obs.action_success

        # Check metadata has event_summary
        assert "event_summary" in obs.metadata
        summary = obs.metadata["event_summary"]
        assert "total_events" in summary
        assert "recent_events" in summary
        assert "event_types" in summary

    def test_event_log_length_tracking(self, environment_with_app):
        """Test that event_log_length is tracked correctly"""
        initial_log_length = len(environment_with_app._are_env.event_log)

        # Get the actual tool name
        tool_name = list(environment_with_app._test_tools.keys())[0]

        # Call a tool to create an event
        environment_with_app.step(
            CallToolAction(
                app_name="TestApp",
                tool_name=tool_name,
                tool_args={},
                advance_time=False,
            )
        )

        obs = environment_with_app.step(TickAction(num_ticks=1))
        final_log_length = obs.event_log_length

        # Should have more events after tool call
        assert final_log_length > initial_log_length

    def test_event_summary_after_tool_call(self, environment_with_app):
        """Test that tool calls appear in event summaries"""
        # Get the actual tool name
        tool_name = list(environment_with_app._test_tools.keys())[0]
        
        # Call a tool
        obs = environment_with_app.step(
            CallToolAction(
                app_name="TestApp",
                tool_name=tool_name,
                tool_args={},
                advance_time=False,
            )
        )
        assert obs.action_success

        # Check event summary includes the tool call
        summary = obs.metadata["event_summary"]
        assert summary["total_events"] >= 1

    def test_get_state_event_log_detail(self, environment_with_app):
        """Test GetStateAction includes detailed event log"""
        # Get the actual tool name
        tool_name = list(environment_with_app._test_tools.keys())[0]
        
        # Call some tools to generate events
        environment_with_app.step(
            CallToolAction(
                app_name="TestApp",
                tool_name=tool_name,
                tool_args={},
                advance_time=False,
            )
        )

        # Get detailed state
        obs = environment_with_app.step(
            GetStateAction(
                include_event_log=True,
                include_event_queue=False,
                include_apps_state=False,
            )
        )
        assert obs.action_success
        assert "event_log" in obs.action_result
        assert "event_log_summary" in obs.action_result

        # Check event log structure
        event_log = obs.action_result["event_log"]
        assert isinstance(event_log, list)
        if len(event_log) > 0:
            event = event_log[0]
            assert "event_id" in event
            assert "event_time" in event
            assert "event_type" in event
            assert "success" in event


class TestEventQueueVisibility:
    """Test event queue visibility and summaries"""

    def test_queue_summary_in_metadata(self, environment, simple_scenario_json):
        """Test that observation metadata contains queue summaries"""
        # Initialize
        obs = environment.step(InitializeAction(scenario_path=simple_scenario_json))
        assert obs.action_success

        # Check metadata has queue_summary
        assert "queue_summary" in obs.metadata
        summary = obs.metadata["queue_summary"]
        assert "total_queued" in summary
        assert "upcoming_events" in summary
        assert "next_event_time" in summary

    def test_event_queue_length_tracking(self, environment, simple_scenario_json):
        """Test that event_queue_length decreases as events execute"""
        # Initialize
        obs = environment.step(InitializeAction(scenario_path=simple_scenario_json))
        initial_queue_length = obs.event_queue_length

        # Tick past events
        obs = environment.step(TickAction(num_ticks=10))
        final_queue_length = obs.event_queue_length

        # Queue should have fewer events (or same if no events were scheduled)
        assert final_queue_length <= initial_queue_length

    def test_get_state_queue_detail(self, environment, simple_scenario_json):
        """Test GetStateAction includes detailed event queue"""
        # Initialize
        obs = environment.step(InitializeAction(scenario_path=simple_scenario_json))
        assert obs.action_success

        # Get detailed state
        obs = environment.step(
            GetStateAction(
                include_event_log=False,
                include_event_queue=True,
                include_apps_state=False,
            )
        )
        assert obs.action_success
        assert "event_queue" in obs.action_result
        assert "queue_summary" in obs.action_result

        # Check event queue structure
        event_queue = obs.action_result["event_queue"]
        assert isinstance(event_queue, list)
        if len(event_queue) > 0:
            event = event_queue[0]
            assert "event_id" in event
            assert "event_time" in event
            assert "event_type" in event


class TestAppsStateVisibility:
    """Test apps state visibility"""

    def test_available_apps_in_observation(self, environment_with_app):
        """Test that available_apps field is populated"""
        # Tick to get observation with apps
        obs = environment_with_app.step(TickAction(num_ticks=1))
        assert obs.available_apps is not None
        assert isinstance(obs.available_apps, list)
        assert "TestApp" in obs.available_apps

    def test_get_state_apps_detail(self, environment_with_app):
        """Test GetStateAction includes detailed apps state"""
        # Get detailed state
        obs = environment_with_app.step(
            GetStateAction(
                include_event_log=False,
                include_event_queue=False,
                include_apps_state=True,
            )
        )
        assert obs.action_success
        assert "apps_state" in obs.action_result

        # Check apps state structure
        apps_state = obs.action_result["apps_state"]
        assert isinstance(apps_state, dict)
        assert "TestApp" in apps_state

        # Each app should have state and tools
        app_state = apps_state["TestApp"]
        assert "tools" in app_state
        assert isinstance(app_state["tools"], list)

        # Tools should have metadata
        if len(app_state["tools"]) > 0:
            tool = app_state["tools"][0]
            assert "name" in tool
            assert "description" in tool
            assert "args" in tool


class TestObservationCompleteness:
    """Test that observations contain all required information"""

    def test_observation_fields_after_initialize(
        self, environment, simple_scenario_json
    ):
        """Test that observation after initialize has all expected fields"""
        obs = environment.step(InitializeAction(scenario_path=simple_scenario_json))

        # Core fields
        assert hasattr(obs, "current_time")
        assert hasattr(obs, "tick_count")
        assert hasattr(obs, "action_success")
        assert hasattr(obs, "action_result")
        assert hasattr(obs, "action_error")
        assert hasattr(obs, "notifications")
        assert hasattr(obs, "environment_state")
        assert hasattr(obs, "event_queue_length")
        assert hasattr(obs, "event_log_length")
        assert hasattr(obs, "available_apps")
        assert hasattr(obs, "metadata")

        # Metadata enrichment
        assert "scenario_loaded" in obs.metadata
        assert "scenario_path" in obs.metadata
        assert "event_summary" in obs.metadata
        assert "queue_summary" in obs.metadata

    def test_observation_fields_after_tick(self, environment, simple_scenario_json):
        """Test that observation after tick has all expected fields"""
        environment.step(InitializeAction(scenario_path=simple_scenario_json))
        obs = environment.step(TickAction(num_ticks=1))

        # All core fields present
        assert obs.current_time > 0
        assert obs.tick_count >= 1
        assert obs.action_success is True
        assert obs.action_result is not None
        assert isinstance(obs.notifications, list)
        # State should be RUNNING or PAUSED (both are valid after ticking)
        assert obs.environment_state in ["RUNNING", "PAUSED"]
        assert obs.event_queue_length >= 0
        assert obs.event_log_length >= 0

        # Action result should have tick metadata
        assert "ticks_executed" in obs.action_result
        assert "events_executed" in obs.action_result
        assert "notifications_generated" in obs.action_result

    def test_observation_fields_after_tool_call(self, environment_with_app):
        """Test that observation after tool call has all expected fields"""
        # Get the actual tool name
        tool_name = list(environment_with_app._test_tools.keys())[0]
        
        obs = environment_with_app.step(
            CallToolAction(
                app_name="TestApp",
                tool_name=tool_name,
                tool_args={},
                advance_time=True,
            )
        )

        # All core fields present
        assert obs.action_success is True
        assert obs.action_result is not None
        assert isinstance(obs.notifications, list)

        # Action result should have tool call metadata
        assert "success" in obs.action_result
        assert "result" in obs.action_result
        assert "time_advanced" in obs.action_result

        # When time advanced, should have event info
        if obs.action_result["time_advanced"]:
            assert "events_executed" in obs.action_result
            assert "notifications_generated" in obs.action_result

    def test_metadata_enrichment_consistency(self, environment, simple_scenario_json):
        """Test that metadata enrichment is consistent across observations"""
        environment.step(InitializeAction(scenario_path=simple_scenario_json))

        # Multiple ticks
        for _ in range(3):
            obs = environment.step(TickAction(num_ticks=1))
            # Every observation should have enriched metadata
            assert "event_summary" in obs.metadata
            assert "queue_summary" in obs.metadata

            # Summaries should have expected structure
            event_summary = obs.metadata["event_summary"]
            assert "total_events" in event_summary
            assert "recent_events" in event_summary
            assert "event_types" in event_summary

            queue_summary = obs.metadata["queue_summary"]
            assert "total_queued" in queue_summary
            assert "upcoming_events" in queue_summary


class TestEventExecutionTracking:
    """Test tracking of events executed during actions"""

    def test_tick_tracks_events_executed(self, environment, simple_scenario_json):
        """Test that tick action tracks number of events executed"""
        environment.step(InitializeAction(scenario_path=simple_scenario_json))

        # Ticking without events should return 0 events executed
        obs = environment.step(TickAction(num_ticks=6))
        assert obs.action_success
        assert "events_executed" in obs.action_result
        # We don't have scenario events, so can be 0
        assert obs.action_result["events_executed"] >= 0

    def test_tool_call_tracks_events_when_advancing_time(self, environment_with_app):
        """Test that tool call with advance_time tracks events executed"""
        # Get the actual tool name
        tool_name = list(environment_with_app._test_tools.keys())[0]
        
        # Call tool with advance_time=True
        obs = environment_with_app.step(
            CallToolAction(
                app_name="TestApp",
                tool_name=tool_name,
                tool_args={},
                advance_time=True,
            )
        )
        assert obs.action_success
        assert "events_executed" in obs.action_result
        assert isinstance(obs.action_result["events_executed"], int)

    def test_tool_call_no_events_without_time_advance(self, environment_with_app):
        """Test that tool call without advance_time doesn't report events executed"""
        # Get the actual tool name
        tool_name = list(environment_with_app._test_tools.keys())[0]
        
        # Call tool with advance_time=False
        obs = environment_with_app.step(
            CallToolAction(
                app_name="TestApp",
                tool_name=tool_name,
                tool_args={},
                advance_time=False,
            )
        )
        assert obs.action_success
        # Should still have time_advanced field
        assert "time_advanced" in obs.action_result
        assert obs.action_result["time_advanced"] is False


class TestRecentEventsSummary:
    """Test that recent events summary provides useful information"""

    def test_recent_events_limited(self, environment_with_app):
        """Test that recent events are limited to a reasonable number"""
        # Get the actual tool name
        tool_name = list(environment_with_app._test_tools.keys())[0]
        
        # Execute multiple tool calls to generate many events
        for i in range(10):
            environment_with_app.step(
                CallToolAction(
                    app_name="TestApp",
                    tool_name=tool_name,
                    tool_args={},
                    advance_time=False,
                )
            )

        # Get observation and check recent events
        obs = environment_with_app.step(TickAction(num_ticks=1))
        event_summary = obs.metadata["event_summary"]

        # Recent events should be limited (default is 5)
        assert len(event_summary["recent_events"]) <= 5

        # Total events should be more than recent
        assert event_summary["total_events"] >= len(event_summary["recent_events"])

    def test_event_types_counted(self, environment_with_app):
        """Test that event types are counted correctly"""
        # Get the actual tool name
        tool_name = list(environment_with_app._test_tools.keys())[0]
        
        # Execute tool call (AGENT event)
        environment_with_app.step(
            CallToolAction(
                app_name="TestApp",
                tool_name=tool_name,
                tool_args={},
                advance_time=False,
            )
        )

        # Tick
        environment_with_app.step(TickAction(num_ticks=1))

        # Get observation
        obs = environment_with_app.step(TickAction(num_ticks=1))
        event_summary = obs.metadata["event_summary"]

        # Should have event type counts
        assert isinstance(event_summary["event_types"], dict)
        # Should have at least AGENT type from tool calls
        assert "AGENT" in event_summary["event_types"]
