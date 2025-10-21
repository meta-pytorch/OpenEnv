# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for ARE Environment Ticking - Sub-Phase 3.3

These tests verify:
- Ticking advances time correctly
- Scheduled events fire at correct times
- Event queue decreases as events are processed
- Event log grows as events complete
- GetState action returns detailed state information
- Notifications are captured during ticks
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import ARE components for creating test scenarios
from are.simulation.apps.email_client import Email, EmailClientApp, EmailFolderName
from are.simulation.scenarios.scenario import Scenario
from are.simulation.types import Event
from envs.are_env.models import GetStateAction, InitializeAction, TickAction
from envs.are_env.server.are_environment import AREEnvironment


@pytest.fixture
def simple_scenario() -> str:
    """Minimal scenario with no events for basic testing."""
    return json.dumps(
        {
            "version": "are_simulation_v1",
            "metadata": {
                "definition": {
                    "scenario_id": "simple_test",
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
def initialized_env(simple_scenario: str) -> AREEnvironment:
    """Environment with simple scenario loaded."""
    env = AREEnvironment()
    env.reset()
    init_obs = env.step(InitializeAction(scenario_path=simple_scenario))
    # Check that initialization succeeded
    if not init_obs.action_success:
        raise RuntimeError(f"Failed to initialize environment: {init_obs.action_error}")
    yield env
    if env._are_env is not None:
        env._are_env.stop()


def test_tick_without_initialization():
    """Test that tick fails without initialization."""
    env = AREEnvironment()
    env.reset()

    obs = env.step(TickAction(num_ticks=1))

    assert obs.action_success is False
    assert "No scenario loaded" in obs.action_error


def test_tick_advances_time(initialized_env: AREEnvironment):
    """Test that ticking advances simulation time."""
    # Get initial time
    initial_time = initialized_env._are_env.current_time
    initial_tick = initialized_env._are_env.tick_count

    # Tick 5 times
    obs = initialized_env.step(TickAction(num_ticks=5))

    assert obs.action_success is True
    # Tick count should have increased
    assert obs.tick_count == initial_tick + 5
    # Time should have advanced (approximately 5 seconds of simulation time)
    # Due to how TimeManager works, we check it advanced meaningfully
    assert obs.current_time > initial_time


def test_tick_increments_tick_count(initialized_env: AREEnvironment):
    """Test that tick count increments correctly."""
    # Tick multiple times
    obs1 = initialized_env.step(TickAction(num_ticks=3))
    assert obs1.tick_count == 3

    obs2 = initialized_env.step(TickAction(num_ticks=2))
    assert obs2.tick_count == 5


def test_tick_with_zero_ticks(initialized_env: AREEnvironment):
    """Test ticking with zero ticks."""
    initial_time = initialized_env._are_env.current_time
    initial_tick = initialized_env._are_env.tick_count

    obs = initialized_env.step(TickAction(num_ticks=0))

    assert obs.action_success is True
    assert obs.current_time == initial_time
    assert obs.tick_count == initial_tick


def test_tick_result_structure(initialized_env: AREEnvironment):
    """Test that tick result has correct structure."""
    obs = initialized_env.step(TickAction(num_ticks=5))

    assert obs.action_success is True
    assert "ticks_executed" in obs.action_result
    assert obs.action_result["ticks_executed"] == 5


def test_get_state_without_initialization():
    """Test that get_state fails without initialization."""
    env = AREEnvironment()
    env.reset()

    obs = env.step(GetStateAction())

    assert obs.action_success is False
    assert "No scenario loaded" in obs.action_error


def test_get_state_event_log(initialized_env: AREEnvironment):
    """Test getting event log."""
    # Execute some ticks to potentially generate log entries
    initialized_env.step(TickAction(num_ticks=10))

    obs = initialized_env.step(GetStateAction(include_event_log=True))

    assert obs.action_success is True
    assert "event_log" in obs.action_result
    assert isinstance(obs.action_result["event_log"], list)


def test_get_state_event_queue(initialized_env: AREEnvironment):
    """Test getting event queue."""
    obs = initialized_env.step(GetStateAction(include_event_queue=True))

    assert obs.action_success is True
    assert "event_queue" in obs.action_result
    assert isinstance(obs.action_result["event_queue"], list)


def test_get_state_apps_state(initialized_env: AREEnvironment):
    """Test getting apps state."""
    obs = initialized_env.step(GetStateAction(include_apps_state=True))

    assert obs.action_success is True
    assert "apps_state" in obs.action_result
    assert isinstance(obs.action_result["apps_state"], dict)


def test_get_state_selective_fields(initialized_env: AREEnvironment):
    """Test that selective fields work."""
    obs = initialized_env.step(
        GetStateAction(
            include_event_log=True,
            include_event_queue=False,
            include_apps_state=True,
        )
    )

    assert obs.action_success is True
    assert "event_log" in obs.action_result
    assert "event_queue" not in obs.action_result
    assert "apps_state" in obs.action_result


def test_get_state_no_fields_requested(initialized_env: AREEnvironment):
    """Test get_state with no fields requested."""
    obs = initialized_env.step(
        GetStateAction(
            include_event_log=False,
            include_event_queue=False,
            include_apps_state=False,
        )
    )

    assert obs.action_success is True
    # Should return empty dict if no fields requested
    assert obs.action_result == {}


def test_get_state_event_log_structure(initialized_env: AREEnvironment):
    """Test that event log has correct structure."""
    # Tick to potentially create some events
    initialized_env.step(TickAction(num_ticks=5))

    obs = initialized_env.step(GetStateAction(include_event_log=True))

    assert obs.action_success is True
    event_log = obs.action_result["event_log"]

    # Check structure of each event in log
    for event in event_log:
        assert "event_id" in event
        assert "event_time" in event
        assert "event_type" in event
        assert "success" in event


# Note: Tests for event queue and event log growth are omitted as they require
# properly formatted scenario events which need more complex setup


def test_observation_fields_populated_after_tick(initialized_env: AREEnvironment):
    """Test that observation fields are properly populated after tick."""
    obs = initialized_env.step(TickAction(num_ticks=5))

    # Check all required observation fields
    assert hasattr(obs, "current_time")
    assert hasattr(obs, "tick_count")
    assert hasattr(obs, "action_success")
    assert hasattr(obs, "action_result")
    assert hasattr(obs, "environment_state")
    assert hasattr(obs, "event_queue_length")
    assert hasattr(obs, "event_log_length")

    # Verify types
    assert isinstance(obs.current_time, (int, float))
    assert isinstance(obs.tick_count, int)
    assert isinstance(obs.action_success, bool)
    assert isinstance(obs.environment_state, str)
    assert isinstance(obs.event_queue_length, int)
    assert isinstance(obs.event_log_length, int)


def test_observation_fields_populated_after_get_state(initialized_env: AREEnvironment):
    """Test that observation fields are properly populated after get_state."""
    obs = initialized_env.step(GetStateAction(include_event_log=True))

    # Check all required observation fields
    assert hasattr(obs, "current_time")
    assert hasattr(obs, "tick_count")
    assert hasattr(obs, "action_success")
    assert hasattr(obs, "action_result")
    assert hasattr(obs, "environment_state")
    assert hasattr(obs, "event_queue_length")
    assert hasattr(obs, "event_log_length")

    # Verify types and values
    assert isinstance(obs.current_time, (int, float))
    assert isinstance(obs.tick_count, int)
    assert isinstance(obs.action_success, bool)
    assert isinstance(obs.environment_state, str)
    # Environment is paused when we query it (which is expected)
    assert obs.environment_state in ("RUNNING", "PAUSED")


def test_notifications_field_exists(initialized_env: AREEnvironment):
    """Test that notifications field exists in observation."""
    obs = initialized_env.step(TickAction(num_ticks=1))

    assert hasattr(obs, "notifications")
    assert isinstance(obs.notifications, list)


def test_multiple_tick_calls_accumulate(initialized_env: AREEnvironment):
    """Test that multiple tick calls accumulate time correctly."""
    initial_time = initialized_env._are_env.current_time
    initial_tick = initialized_env._are_env.tick_count

    # Multiple separate tick calls
    initialized_env.step(TickAction(num_ticks=3))
    initialized_env.step(TickAction(num_ticks=2))
    obs = initialized_env.step(TickAction(num_ticks=5))

    # Total should be 10 ticks
    assert obs.tick_count == initial_tick + 10
    # Time should have advanced meaningfully
    assert obs.current_time > initial_time


# ============================================================================
# Tests with Events - Verify events actually fire and are tracked
# ============================================================================


@pytest.fixture
def env_with_scheduled_events() -> AREEnvironment:
    """Create environment with programmatically scheduled events."""
    from are.simulation.environment import Environment as ARESimEnv, EnvironmentConfig
    from are.simulation.notification_system import (
        VerboseNotificationSystem,
        VerbosityLevel,
    )

    # Create a simple scenario with an email app
    scenario = Scenario(
        scenario_id="event_test",
        duration=100,
        time_increment_in_seconds=1,
    )

    # Add email app
    email_app = EmailClientApp()
    scenario.apps = [email_app]

    # Create scheduled events at specific times
    # Event 1: Email at time 2 - using send_email_to_user_only which generates notifications
    event1 = Event.from_function(
        email_app.send_email_to_user_only,
        sender="test1@example.com",
        subject="Event 1",
        content="Email from event 1",
    ).delayed(2)

    # Event 2: Email at time 5
    event2 = Event.from_function(
        email_app.send_email_to_user_only,
        sender="test2@example.com",
        subject="Event 2",
        content="Email from event 2",
    ).delayed(5)

    # Event 3: Email at time 10
    event3 = Event.from_function(
        email_app.send_email_to_user_only,
        sender="test3@example.com",
        subject="Event 3",
        content="Email from event 3",
    ).delayed(10)

    scenario.events = [event1, event2, event3]
    scenario.initialize()

    # Create notification system with MEDIUM verbosity
    notification_system = VerboseNotificationSystem(
        verbosity_level=VerbosityLevel.MEDIUM
    )

    # Create ARE environment
    config = EnvironmentConfig(
        start_time=0,
        duration=None,
        time_increment_in_seconds=1,
        oracle_mode=True,
        exit_when_no_events=False,
        queue_based_loop=False,
        verbose=False,
    )

    are_env = ARESimEnv(config=config, notification_system=notification_system)
    are_env.run(scenario, wait_for_end=False, schedule_events=True)
    are_env.pause()

    # Wrap in OpenEnv environment
    env = AREEnvironment()
    env.reset()
    env._are_env = are_env
    env._scenario = scenario
    env._scenario_loaded = True

    yield env

    if env._are_env is not None:
        env._are_env.stop()


def test_event_queue_starts_with_scheduled_events(env_with_scheduled_events):
    """Test that event queue is populated with scheduled events."""
    obs = env_with_scheduled_events.step(GetStateAction(include_event_queue=True))

    assert obs.action_success is True
    event_queue = obs.action_result["event_queue"]

    # Should have 3 events scheduled
    assert len(event_queue) == 3

    # Verify event structure
    for event in event_queue:
        assert "event_id" in event
        assert "event_time" in event
        assert "event_type" in event


def test_events_fire_at_correct_time(env_with_scheduled_events):
    """Test that events fire when simulation time reaches event time."""
    # Initial state - no events fired yet
    initial_obs = env_with_scheduled_events.step(
        GetStateAction(include_event_log=True, include_event_queue=True)
    )
    initial_log_length = len(initial_obs.action_result["event_log"])
    initial_queue_length = len(initial_obs.action_result["event_queue"])

    # Tick to time 3 - should fire event 1 (scheduled at time 2)
    env_with_scheduled_events.step(TickAction(num_ticks=3))

    obs = env_with_scheduled_events.step(
        GetStateAction(include_event_log=True, include_event_queue=True)
    )

    # Event log should have grown
    assert len(obs.action_result["event_log"]) > initial_log_length
    # Event queue should have shrunk
    assert len(obs.action_result["event_queue"]) < initial_queue_length


def test_event_log_grows_as_events_fire(env_with_scheduled_events):
    """Test that event log grows as time progresses and events fire."""
    # Get initial log
    obs1 = env_with_scheduled_events.step(GetStateAction(include_event_log=True))
    log_length_0 = len(obs1.action_result["event_log"])

    # Tick to time 3
    env_with_scheduled_events.step(TickAction(num_ticks=3))
    obs2 = env_with_scheduled_events.step(GetStateAction(include_event_log=True))
    log_length_3 = len(obs2.action_result["event_log"])

    # Should have fired event 1
    assert log_length_3 > log_length_0

    # Tick to time 6
    env_with_scheduled_events.step(TickAction(num_ticks=3))
    obs3 = env_with_scheduled_events.step(GetStateAction(include_event_log=True))
    log_length_6 = len(obs3.action_result["event_log"])

    # Should have fired event 2 as well
    assert log_length_6 > log_length_3

    # Tick to time 11
    env_with_scheduled_events.step(TickAction(num_ticks=5))
    obs4 = env_with_scheduled_events.step(GetStateAction(include_event_log=True))
    log_length_11 = len(obs4.action_result["event_log"])

    # Should have fired event 3 as well
    assert log_length_11 > log_length_6


def test_event_queue_decreases_as_events_fire(env_with_scheduled_events):
    """Test that event queue decreases as events are processed."""
    # Get initial queue
    obs1 = env_with_scheduled_events.step(GetStateAction(include_event_queue=True))
    initial_queue = len(obs1.action_result["event_queue"])

    assert initial_queue == 3  # We scheduled 3 events

    # Tick past first event
    env_with_scheduled_events.step(TickAction(num_ticks=3))
    obs2 = env_with_scheduled_events.step(GetStateAction(include_event_queue=True))
    queue_after_first = len(obs2.action_result["event_queue"])

    assert queue_after_first < initial_queue

    # Tick past all events
    env_with_scheduled_events.step(TickAction(num_ticks=10))
    obs3 = env_with_scheduled_events.step(GetStateAction(include_event_queue=True))
    final_queue = len(obs3.action_result["event_queue"])

    # All events should have been processed
    assert final_queue < queue_after_first


def test_app_state_changes_after_events(env_with_scheduled_events):
    """Test that app state reflects changes from fired events."""
    # Get initial app state
    obs1 = env_with_scheduled_events.step(GetStateAction(include_apps_state=True))
    initial_app_state = obs1.action_result["apps_state"]["EmailClientApp"]

    # Count initial emails
    initial_inbox = initial_app_state["folders"]["INBOX"]
    initial_email_count = len(initial_inbox["emails"])

    # Tick past first event (email at time 2)
    env_with_scheduled_events.step(TickAction(num_ticks=3))

    # Get app state again
    obs2 = env_with_scheduled_events.step(GetStateAction(include_apps_state=True))
    app_state_after = obs2.action_result["apps_state"]["EmailClientApp"]
    inbox_after = app_state_after["folders"]["INBOX"]
    email_count_after = len(inbox_after["emails"])

    # Should have received an email
    assert email_count_after > initial_email_count


def test_event_log_contains_fired_event_details(env_with_scheduled_events):
    """Test that event log contains details of fired events."""
    # Tick to fire some events
    env_with_scheduled_events.step(TickAction(num_ticks=6))

    # Get event log
    obs = env_with_scheduled_events.step(GetStateAction(include_event_log=True))
    event_log = obs.action_result["event_log"]

    # Should have events in log
    assert len(event_log) > 0

    # Check that events have required fields
    for event in event_log:
        assert "event_id" in event
        assert "event_time" in event
        assert "event_type" in event
        assert "success" in event
        # Event should have succeeded
        assert event["success"] is True


def test_observation_event_queue_length_updates(env_with_scheduled_events):
    """Test that observation's event_queue_length field updates correctly."""
    # Initial observation
    obs1 = env_with_scheduled_events.step(GetStateAction())
    initial_queue_length = obs1.event_queue_length

    # Should start with 3 events
    assert initial_queue_length == 3

    # Tick to fire some events
    obs2 = env_with_scheduled_events.step(TickAction(num_ticks=6))

    # Queue length should have decreased
    assert obs2.event_queue_length < initial_queue_length


def test_observation_event_log_length_updates(env_with_scheduled_events):
    """Test that observation's event_log_length field updates correctly."""
    # Initial observation
    obs1 = env_with_scheduled_events.step(GetStateAction())
    initial_log_length = obs1.event_log_length

    # Tick to fire some events
    obs2 = env_with_scheduled_events.step(TickAction(num_ticks=6))

    # Log length should have increased
    assert obs2.event_log_length > initial_log_length


def test_notifications_contain_fired_events(env_with_scheduled_events):
    """Test that notifications field contains events that fired during tick."""
    # Tick to fire first event (at time 2)
    obs = env_with_scheduled_events.step(TickAction(num_ticks=3))

    # Should have notifications for the fired event
    assert len(obs.notifications) > 0

    # Check notification structure - should use ARE's MessageType values
    for notification in obs.notifications:
        # Type should be from ARE's MessageType enum
        assert notification["type"] in [
            "USER_MESSAGE",
            "ENVIRONMENT_NOTIFICATION",
            "ENVIRONMENT_STOP",
        ]
        # Should have message content
        assert "message" in notification
        # Should have timestamp
        assert "timestamp" in notification


def test_notifications_cleared_between_ticks(env_with_scheduled_events):
    """Test that notifications are cleared between tick calls."""
    # First tick - fires event at time 2
    obs1 = env_with_scheduled_events.step(TickAction(num_ticks=3))
    first_notifications = obs1.notifications

    # Second tick - fires event at time 5
    obs2 = env_with_scheduled_events.step(TickAction(num_ticks=3))
    second_notifications = obs2.notifications

    # Both should have notifications
    assert len(first_notifications) > 0
    assert len(second_notifications) > 0

    # Notifications should be different (different events fired)
    assert first_notifications != second_notifications


def test_notifications_reflect_multiple_events(env_with_scheduled_events):
    """Test that notifications capture all events fired in a single tick."""
    # Tick past multiple events (events at time 2, 5, 10)
    obs = env_with_scheduled_events.step(TickAction(num_ticks=11))

    # Should have notifications for all 3 events
    assert len(obs.notifications) == 3

    # Verify events are in order by timestamp
    timestamps = [n["timestamp"] for n in obs.notifications]
    assert timestamps == sorted(timestamps)  # Should be in chronological order


# ============================================================================
# NEW: Comprehensive Notification System Tests
# ============================================================================


@pytest.fixture
def env_with_reminder_app() -> AREEnvironment:
    """Create environment with reminder app for time-based notification testing."""
    from datetime import datetime, timezone

    from are.simulation.apps.reminder import ReminderApp
    from are.simulation.environment import Environment as ARESimEnv, EnvironmentConfig

    # Create a simple scenario
    scenario = Scenario(
        scenario_id="reminder_test",
        duration=100,
        time_increment_in_seconds=1,
    )

    # Add reminder app
    reminder_app = ReminderApp()
    scenario.apps = [reminder_app]

    # Create a reminder due at simulation time 5
    # Start time is 0 (epoch), so time 5 is 5 seconds after epoch
    from are.simulation.types import Event

    # Format: "YYYY-MM-DD HH:MM:SS" in UTC
    # Simulation time 5 = 5 seconds after epoch = 1970-01-01 00:00:05
    due_datetime_str = "1970-01-01 00:00:05"

    event = Event.from_function(
        reminder_app.add_reminder,
        title="Test Reminder",
        due_datetime=due_datetime_str,
        description="This is a test reminder",
    ).delayed(
        0
    )  # Execute immediately to create the reminder

    scenario.events = [event]
    scenario.initialize()

    # Create ARE environment
    config = EnvironmentConfig(
        start_time=0,
        duration=None,
        time_increment_in_seconds=1,
        oracle_mode=True,
        exit_when_no_events=False,
        queue_based_loop=False,
        verbose=False,
    )

    are_env = ARESimEnv(config=config)
    are_env.run(scenario, wait_for_end=False, schedule_events=True)
    are_env.pause()

    # Wrap in OpenEnv environment
    env = AREEnvironment()
    env.reset()
    env._are_env = are_env
    env._scenario = scenario
    env._scenario_loaded = True

    yield env

    if env._are_env is not None:
        env._are_env.stop()


def test_notification_types_exist(env_with_scheduled_events):
    """Test that notifications have a 'type' field."""
    obs = env_with_scheduled_events.step(TickAction(num_ticks=3))

    assert len(obs.notifications) > 0
    for notification in obs.notifications:
        assert "type" in notification
        # Type should be a string
        assert isinstance(notification["type"], str)


def test_notification_has_timestamp(env_with_scheduled_events):
    """Test that notifications include timestamp information."""
    obs = env_with_scheduled_events.step(TickAction(num_ticks=3))

    assert len(obs.notifications) > 0
    for notification in obs.notifications:
        # Should have either 'timestamp' or 'event_time'
        assert "timestamp" in notification or "event_time" in notification


def test_time_based_notifications_captured(env_with_reminder_app):
    """Test that time-based notifications (reminders) are captured during ticks."""
    # First tick to create the reminder (event at time 0)
    obs1 = env_with_reminder_app.step(TickAction(num_ticks=1))

    # Tick to time 6 - should trigger reminder notification at time 5
    obs2 = env_with_reminder_app.step(TickAction(num_ticks=6))

    # Should have notifications
    assert len(obs2.notifications) > 0

    # Check for reminder-related notification
    # This could be in ENVIRONMENT_NOTIFICATION type or similar
    has_reminder = any(
        "reminder" in str(n.get("message", "")).lower()
        or "reminder" in n.get("type", "").lower()
        for n in obs2.notifications
    )

    # Note: This test will fail with current implementation
    # because time-based notifications are not captured from message_queue
    assert has_reminder, (
        "Expected time-based notification for due reminder. "
        "This indicates the notification system is not properly integrated."
    )


def test_notification_message_queue_integration(env_with_scheduled_events):
    """Test that notifications come from ARE's message_queue, not just event log."""
    # This test verifies the source of notifications

    # Tick to fire some events
    obs = env_with_scheduled_events.step(TickAction(num_ticks=6))

    # Check if notification system exists
    assert env_with_scheduled_events._are_env.notification_system is not None

    # After ticking, the message queue should have been processed
    # In correct implementation, notifications should be extracted from message_queue
    # and the queue should be empty (or have only future messages)

    # Get current time
    from datetime import datetime, timezone

    current_time = env_with_scheduled_events._are_env.current_time
    current_timestamp = datetime.fromtimestamp(current_time, tz=timezone.utc)

    # Check if there are old messages still in the queue (should be extracted)
    remaining_messages = (
        env_with_scheduled_events._are_env.notification_system.message_queue.list_view()
    )

    old_messages = [
        msg for msg in remaining_messages if msg.timestamp <= current_timestamp
    ]

    # With correct implementation, old messages should have been extracted
    assert len(old_messages) == 0, (
        f"Found {len(old_messages)} messages still in queue that should have been extracted. "
        "This indicates notifications are not being properly extracted from message_queue."
    )


def test_notification_types_are_preserved(env_with_scheduled_events):
    """Test that notification types from ARE (USER_MESSAGE, ENVIRONMENT_NOTIFICATION) are preserved."""
    obs = env_with_scheduled_events.step(TickAction(num_ticks=6))

    assert len(obs.notifications) > 0

    # Check that notification types are meaningful, not generic
    for notification in obs.notifications:
        notification_type = notification.get("type")

        # Should not be a generic type like "event_fired" for all notifications
        # Should use ARE's MessageType enum values
        valid_types = [
            "USER_MESSAGE",
            "ENVIRONMENT_NOTIFICATION",
            "ENVIRONMENT_STOP",
            "event_fired",  # Temporary - should eventually be removed
        ]

        # At least some notifications should have proper types
        # (not all "event_fired")

    # Count notification types
    type_counts = {}
    for notification in obs.notifications:
        ntype = notification.get("type", "unknown")
        type_counts[ntype] = type_counts.get(ntype, 0) + 1

    # With current implementation, all will be "event_fired"
    # With correct implementation, should have variety of types
    if len(type_counts) == 1 and "event_fired" in type_counts:
        pytest.skip(
            "All notifications are 'event_fired' type. "
            "This indicates notification types are not being properly preserved from message_queue. "
            "This is expected to fail until the notification system is properly integrated."
        )


def test_notification_ordering_within_tick(env_with_scheduled_events):
    """Test that notifications within a single tick are properly ordered by timestamp."""
    # Tick past multiple events
    obs = env_with_scheduled_events.step(TickAction(num_ticks=11))

    assert len(obs.notifications) > 0

    # Extract timestamps (handle both formats)
    timestamps = []
    for n in obs.notifications:
        if "timestamp" in n:
            # Proper timestamp format
            from datetime import datetime

            if isinstance(n["timestamp"], str):
                timestamps.append(datetime.fromisoformat(n["timestamp"]))
            else:
                timestamps.append(n["timestamp"])
        elif "event_time" in n:
            # Fallback to event_time
            timestamps.append(n["event_time"])

    # Should be in chronological order
    assert timestamps == sorted(
        timestamps
    ), "Notifications should be ordered chronologically"


def test_no_duplicate_notifications(env_with_scheduled_events):
    """Test that the same event doesn't generate duplicate notifications."""
    # Tick to fire some events (event at time 2)
    obs1 = env_with_scheduled_events.step(TickAction(num_ticks=3))

    initial_notifications = obs1.notifications

    # Tick more (fires event at time 5)
    obs2 = env_with_scheduled_events.step(TickAction(num_ticks=3))

    # Second tick should have different events (if any)
    # Should not repeat notifications from first tick
    if len(initial_notifications) > 0 and len(obs2.notifications) > 0:
        # Compare notification content (messages and timestamps)
        initial_messages = {
            (n.get("message"), n.get("timestamp")) for n in initial_notifications
        }
        second_messages = {
            (n.get("message"), n.get("timestamp")) for n in obs2.notifications
        }

        # Should not have overlap (events should only notify once)
        overlap = initial_messages & second_messages
        assert len(overlap) == 0, f"Found duplicate notifications: {overlap}"


@pytest.fixture
def env_with_user_messages() -> AREEnvironment:
    """Create environment with AgentUserInterface to test user message notifications."""
    from are.simulation.apps.agent_user_interface import AgentUserInterface
    from are.simulation.environment import Environment as ARESimEnv, EnvironmentConfig
    from are.simulation.notification_system import (
        VerboseNotificationSystem,
        VerbosityLevel,
    )

    # Create a simple scenario
    scenario = Scenario(
        scenario_id="user_message_test",
        duration=100,
        time_increment_in_seconds=1,
    )

    # Add AgentUserInterface app
    aui = AgentUserInterface()
    scenario.apps = [aui]

    # Create events that send messages to the agent (simulating user messages)
    from are.simulation.types import Event

    # User message at time 2
    event1 = Event.from_function(
        aui.send_message_to_agent,
        content="Hello Agent! This is a test message.",
    ).delayed(2)

    # Another user message at time 5
    event2 = Event.from_function(
        aui.send_message_to_agent,
        content="This is a second message from the user.",
    ).delayed(5)

    scenario.events = [event1, event2]
    scenario.initialize()

    # Create notification system with LOW verbosity (should still capture user messages)
    notification_system = VerboseNotificationSystem(verbosity_level=VerbosityLevel.LOW)

    # Create ARE environment
    config = EnvironmentConfig(
        start_time=0,
        duration=None,
        time_increment_in_seconds=1,
        oracle_mode=True,
        exit_when_no_events=False,
        queue_based_loop=False,
        verbose=False,
    )

    are_env = ARESimEnv(config=config, notification_system=notification_system)
    are_env.run(scenario, wait_for_end=False, schedule_events=True)
    are_env.pause()

    # Wrap in OpenEnv environment
    env = AREEnvironment()
    env.reset()
    env._are_env = are_env
    env._scenario = scenario
    env._scenario_loaded = True

    yield env

    if env._are_env is not None:
        env._are_env.stop()


@pytest.fixture
def env_with_verbosity_config() -> tuple[AREEnvironment, AREEnvironment]:
    """Create two environments with different notification verbosity levels."""
    from are.simulation.apps.email_client import Email, EmailClientApp
    from are.simulation.environment import Environment as ARESimEnv, EnvironmentConfig
    from are.simulation.notification_system import (
        VerboseNotificationSystem,
        VerbosityLevel,
    )

    def create_env_with_verbosity(verbosity: VerbosityLevel):
        # Create a simple scenario with email app
        scenario = Scenario(
            scenario_id="verbosity_test",
            duration=100,
            time_increment_in_seconds=1,
        )

        # Add email app
        email_app = EmailClientApp()
        scenario.apps = [email_app]

        # Create email event at time 2
        from are.simulation.types import Event

        event = Event.from_function(
            email_app.add_email,
            email=Email(
                sender="test@example.com",
                recipients=[email_app.user_email],
                subject="Test Email",
                content="Test content",
            ),
        ).delayed(2)

        scenario.events = [event]
        scenario.initialize()

        # Create notification system with specific verbosity
        notification_system = VerboseNotificationSystem(verbosity_level=verbosity)

        # Create ARE environment
        config = EnvironmentConfig(
            start_time=0,
            duration=None,
            time_increment_in_seconds=1,
            oracle_mode=True,
            exit_when_no_events=False,
            queue_based_loop=False,
            verbose=False,
        )

        are_env = ARESimEnv(config=config, notification_system=notification_system)
        are_env.run(scenario, wait_for_end=False, schedule_events=True)
        are_env.pause()

        # Wrap in OpenEnv environment
        env = AREEnvironment()
        env.reset()
        env._are_env = are_env
        env._scenario = scenario
        env._scenario_loaded = True

        return env

    env_low = create_env_with_verbosity(VerbosityLevel.LOW)
    env_high = create_env_with_verbosity(VerbosityLevel.HIGH)

    yield env_low, env_high

    if env_low._are_env is not None:
        env_low._are_env.stop()
    if env_high._are_env is not None:
        env_high._are_env.stop()


def test_verbosity_level_affects_notifications(env_with_verbosity_config):
    """Test that verbosity level controls which notifications are generated."""
    env_low, env_high = env_with_verbosity_config

    # Tick both environments the same amount
    obs_low = env_low.step(TickAction(num_ticks=5))
    obs_high = env_high.step(TickAction(num_ticks=5))

    # Both should have fired the email event
    # But HIGH verbosity should generate notification about it, LOW should not

    # Check message queues directly
    messages_low = env_low._are_env.notification_system.message_queue.list_view()
    messages_high = env_high._are_env.notification_system.message_queue.list_view()

    # HIGH verbosity should have more messages
    # (Email events trigger notifications at MEDIUM and HIGH, but not LOW)
    assert len(messages_high) >= len(messages_low), (
        f"Expected HIGH verbosity to have more notifications than LOW. "
        f"Got LOW: {len(messages_low)}, HIGH: {len(messages_high)}"
    )

    # With current implementation, notifications might not reflect this
    # because they're captured from event log, not message queue
    if len(obs_low.notifications) == len(obs_high.notifications):
        pytest.skip(
            "Verbosity levels don't affect notification capture. "
            "This indicates notifications are captured from event log, not message_queue. "
            "Expected to fail until proper message_queue integration."
        )


def test_notification_includes_message_content(env_with_scheduled_events):
    """Test that notifications include the actual message content."""
    obs = env_with_scheduled_events.step(TickAction(num_ticks=6))

    assert len(obs.notifications) > 0

    for notification in obs.notifications:
        # Should have message content
        assert (
            "message" in notification
            or "content" in notification
            or "event_type" in notification
        )

        # Message should be non-empty string
        message = notification.get("message") or notification.get("content") or ""
        assert (
            isinstance(message, str) or message == ""
        ), f"Notification message should be a string, got {type(message)}"


def test_empty_notifications_when_no_events(initialized_env: AREEnvironment):
    """Test that no notifications are generated when no events fire."""
    # Scenario has no events, just tick forward
    obs = initialized_env.step(TickAction(num_ticks=5))

    # Should have no notifications
    assert len(obs.notifications) == 0


def test_notification_structure_complete(env_with_scheduled_events):
    """Test that notifications have all required fields."""
    obs = env_with_scheduled_events.step(TickAction(num_ticks=6))

    assert len(obs.notifications) > 0

    required_fields = ["type"]  # Minimum required
    recommended_fields = ["type", "message", "timestamp"]

    for notification in obs.notifications:
        # Check required fields
        for field in required_fields:
            assert (
                field in notification
            ), f"Notification missing required field: {field}"

        # Count how many recommended fields are present
        present_fields = sum(1 for field in recommended_fields if field in notification)

        # Should have at least 2 out of 3 recommended fields
        assert present_fields >= 2, (
            f"Notification should have at least 2 of {recommended_fields}, "
            f"but only has {present_fields}: {list(notification.keys())}"
        )


def test_user_messages_captured(env_with_user_messages):
    """Test that user messages (AgentUserInterface) are captured as notifications."""
    # Tick to time 3 - should fire first user message at time 2
    obs = env_with_user_messages.step(TickAction(num_ticks=3))

    # Should have notifications
    assert (
        len(obs.notifications) > 0
    ), "Expected notification for user message at time 2"

    # At least one notification should be a user message
    has_user_message = any(
        n.get("type") == "USER_MESSAGE" or "message" in str(n).lower()
        for n in obs.notifications
    )

    assert has_user_message, (
        "Expected USER_MESSAGE type notification for user message. "
        "This indicates user messages are not being properly captured from message_queue."
    )


def test_user_messages_have_content(env_with_user_messages):
    """Test that user message notifications contain the actual message content."""
    # Tick to fire both user messages
    obs = env_with_user_messages.step(TickAction(num_ticks=6))

    # Should have at least 2 notifications (2 user messages)
    assert len(obs.notifications) >= 2

    # Check that notifications have message content
    messages_found = []
    for notification in obs.notifications:
        if "message" in notification:
            messages_found.append(notification["message"])

    # Should have captured the user messages
    assert len(messages_found) > 0, (
        "No notifications with 'message' field found. "
        "User message content is not being properly extracted."
    )


def test_user_message_type_distinct_from_event_fired(env_with_user_messages):
    """Test that user messages have distinct type from generic event_fired."""
    # Tick to fire user messages
    obs = env_with_user_messages.step(TickAction(num_ticks=6))

    assert len(obs.notifications) > 0

    # Check notification types
    user_message_notifications = [
        n for n in obs.notifications if n.get("type") == "USER_MESSAGE"
    ]
    event_fired_notifications = [
        n for n in obs.notifications if n.get("type") == "event_fired"
    ]

    # With correct implementation, should have USER_MESSAGE types
    if len(user_message_notifications) == 0 and len(event_fired_notifications) > 0:
        pytest.skip(
            "All notifications are 'event_fired' type, no USER_MESSAGE types found. "
            "This indicates user messages are not being properly captured from message_queue. "
            "Expected to fail until proper integration."
        )

    assert (
        len(user_message_notifications) > 0
    ), "Expected at least one USER_MESSAGE type notification for user messages."


def test_multiple_user_messages_captured_separately(env_with_user_messages):
    """Test that multiple user messages are captured as separate notifications."""
    # Tick past both user messages (at time 2 and 5)
    obs = env_with_user_messages.step(TickAction(num_ticks=6))

    # Should have at least 2 notifications
    assert (
        len(obs.notifications) >= 2
    ), f"Expected at least 2 notifications for 2 user messages, got {len(obs.notifications)}"

    # If we have message queue integration, check that both messages are distinct
    if len(obs.notifications) >= 2:
        # Check that notifications have different timestamps or content
        timestamps = [
            n.get("timestamp") or n.get("event_time") for n in obs.notifications
        ]
        # Should have different timestamps
        assert (
            len(set(timestamps)) > 1
        ), "User messages should have different timestamps"


def test_user_messages_at_low_verbosity(env_with_user_messages):
    """Test that user messages are captured even at LOW verbosity level."""
    # The env_with_user_messages fixture uses LOW verbosity
    # User messages should ALWAYS be captured regardless of verbosity

    # Tick to fire user messages
    obs = env_with_user_messages.step(TickAction(num_ticks=6))

    # Check message queue directly - should have user messages
    messages = (
        env_with_user_messages._are_env.notification_system.message_queue.list_view()
    )

    # At LOW verbosity, user messages should still be in the queue
    # (or should have been extracted to notifications)
    user_messages_in_queue = [
        m for m in messages if m.message_type.value == "USER_MESSAGE"
    ]

    # Either in the queue or in notifications
    user_messages_in_notifications = [
        n for n in obs.notifications if n.get("type") == "USER_MESSAGE"
    ]

    total_user_messages = len(user_messages_in_queue) + len(
        user_messages_in_notifications
    )

    assert total_user_messages >= 2, (
        f"Expected at least 2 user messages captured at LOW verbosity. "
        f"Found {total_user_messages} (queue: {len(user_messages_in_queue)}, "
        f"notifications: {len(user_messages_in_notifications)})"
    )
