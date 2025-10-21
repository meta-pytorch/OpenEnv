"""
Tests for ARE Environment Initialization (Sub-Phase 3.1).

These tests verify that the ARE environment can be properly initialized
with scenarios from JSON files and JSON strings, and that reset/cleanup
works correctly.
"""

import json

# Add src to path so we can import modules
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from envs.are_env.models import InitializeAction
from envs.are_env.server.are_environment import AREEnvironment


@pytest.fixture
def simple_scenario_json():
    """Minimal valid scenario JSON"""
    return json.dumps(
        {
            "version": "are_simulation_v1",  # Required by ExportedTrace - correct version string
            "metadata": {  # Required by ExportedTrace
                "definition": {
                    "scenario_id": "test_scenario",
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
def are_env():
    """Create fresh environment for each test"""
    env = AREEnvironment()
    yield env
    # Cleanup
    if env._are_env is not None:
        env._are_env.stop()


def test_reset_initializes_clean_state(are_env):
    """Test that reset() creates clean state"""
    obs = are_env.reset()

    assert obs.action_success is True
    assert obs.environment_state == "SETUP"
    assert obs.current_time == 0.0
    assert obs.tick_count == 0
    assert are_env._are_env is None
    assert are_env._scenario is None
    assert are_env._scenario_loaded is False


def test_initialize_with_valid_json_string(are_env, simple_scenario_json):
    """Test initializing with JSON string"""
    are_env.reset()

    action = InitializeAction(scenario_path=simple_scenario_json)
    obs = are_env.step(action)

    assert obs.action_success is True
    assert obs.environment_state == "RUNNING"
    assert are_env._scenario_loaded is True
    assert are_env._are_env is not None
    assert are_env._scenario is not None


def test_initialize_with_file_path(are_env, tmp_path, simple_scenario_json):
    """Test initializing with file path"""
    # Write scenario to temp file
    scenario_file = tmp_path / "test_scenario.json"
    scenario_file.write_text(simple_scenario_json)

    are_env.reset()
    action = InitializeAction(scenario_path=str(scenario_file))
    obs = are_env.step(action)

    assert obs.action_success is True
    assert obs.environment_state == "RUNNING"


def test_initialize_invalid_json(are_env):
    """Test that invalid JSON returns error"""
    are_env.reset()

    action = InitializeAction(scenario_path="invalid json {{{")
    obs = are_env.step(action)

    assert obs.action_success is False
    assert obs.action_error is not None
    assert obs.environment_state == "FAILED"


def test_initialize_missing_required_fields(are_env):
    """Test scenario missing required fields"""
    are_env.reset()

    invalid_scenario = json.dumps({"scenario_id": "test"})  # Missing duration, etc.
    action = InitializeAction(scenario_path=invalid_scenario)
    obs = are_env.step(action)

    assert obs.action_success is False
    assert obs.action_error is not None


def test_reset_after_initialization(are_env, simple_scenario_json):
    """Test that reset cleans up after initialization"""
    are_env.reset()

    # Initialize
    action = InitializeAction(scenario_path=simple_scenario_json)
    are_env.step(action)
    assert are_env._scenario_loaded is True

    # Reset again
    obs = are_env.reset()

    assert obs.action_success is True
    assert obs.environment_state == "SETUP"
    assert are_env._are_env is None
    assert are_env._scenario_loaded is False


def test_state_property_tracking(are_env, simple_scenario_json):
    """Test that state property tracks episode correctly"""
    obs1 = are_env.reset()
    episode_id_1 = are_env.state.episode_id

    # Initialize and step
    action = InitializeAction(scenario_path=simple_scenario_json)
    are_env.step(action)

    assert are_env.state.episode_id == episode_id_1
    assert are_env.state.step_count == 1

    # Reset should change episode ID
    obs2 = are_env.reset()
    episode_id_2 = are_env.state.episode_id

    assert episode_id_2 != episode_id_1
    assert are_env.state.step_count == 0


def test_are_internal_state_populated(are_env, simple_scenario_json):
    """Test that ARE internal state is properly populated after initialization"""
    # Before initialization, internal state should be None
    state_before = are_env.state
    assert state_before.are_internal_state is None
    assert state_before.scenario_loaded is False

    # Initialize
    are_env.reset()
    action = InitializeAction(scenario_path=simple_scenario_json)
    are_env.step(action)

    # After initialization, internal state should be populated
    state_after = are_env.state
    assert state_after.are_internal_state is not None
    assert state_after.scenario_loaded is True

    # Verify internal state contains expected keys from ARE's get_state()
    internal_state = state_after.are_internal_state
    assert "event_log" in internal_state
    assert "event_queue" in internal_state
    assert "apps" in internal_state
    assert "current_time" in internal_state
    assert "start_time" in internal_state
    assert "duration" in internal_state
    assert "time_increment_in_seconds" in internal_state

    # Verify state fields match ARE environment
    assert state_after.current_time == are_env._are_env.current_time
    assert state_after.tick_count == are_env._are_env.tick_count
    assert state_after.environment_state == are_env._are_env.state.value
