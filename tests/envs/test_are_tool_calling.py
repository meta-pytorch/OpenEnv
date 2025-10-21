# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for ARE Environment Tool Calling - Sub-Phase 3.2

These tests verify:
- list_apps action returns real apps from ARE environment
- list_apps includes tool names, descriptions, and parameters
- call_tool can execute tools on apps
- call_tool handles invalid app/tool names gracefully
- call_tool respects advance_time parameter
- Tool execution results are captured and returned
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import DummyApp for testing
from are.simulation.apps.app import App
from are.simulation.tool_utils import app_tool
from are.simulation.types import event_registered
from are.simulation.utils import get_state_dict
from envs.are_env.models import CallToolAction, InitializeAction, ListAppsAction
from envs.are_env.server.are_environment import AREEnvironment


class DummyApp(App):
    """Simple test app with one tool for testing."""

    def __init__(self):
        super().__init__()
        self.logs = []

    def get_state(self):
        return get_state_dict(self, ["logs"])

    def reset(self):
        self.logs = []

    @app_tool()
    @event_registered()
    def log_stuff(self, message: str):
        """Log a message to the app's log list."""
        self.logs.append(message)
        return f"Logged: {message}"


@pytest.fixture
def simple_scenario() -> str:
    """Minimal scenario with no apps for basic testing."""
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
            "apps": [],  # No apps - simplifies testing
            "events": [],
            "completed_events": [],
            "world_logs": [],
        }
    )


@pytest.fixture
def scenario_with_dummy_app() -> str:
    """Scenario with DummyApp for tool calling tests."""
    return json.dumps(
        {
            "version": "are_simulation_v1",
            "metadata": {
                "definition": {
                    "scenario_id": "dummy_app_test",
                    "duration": 100,
                    "time_increment_in_seconds": 1,
                }
            },
            "apps": [{"app_type": "DummyApp", "name": "test_dummy"}],
            "events": [],
            "completed_events": [],
            "world_logs": [],
        }
    )


@pytest.fixture
def initialized_env(simple_scenario: str) -> AREEnvironment:
    """Environment with simple scenario (no apps) loaded."""
    env = AREEnvironment()
    env.reset()
    init_obs = env.step(InitializeAction(scenario_path=simple_scenario))
    # Check that initialization succeeded
    if not init_obs.action_success:
        raise RuntimeError(f"Failed to initialize environment: {init_obs.action_error}")
    yield env
    if env._are_env is not None:
        env._are_env.stop()


def test_list_apps_without_initialization():
    """Test that list_apps fails without initialization."""
    env = AREEnvironment()
    env.reset()

    obs = env.step(ListAppsAction())

    assert obs.action_success is False
    assert "No scenario loaded" in obs.action_error


def test_list_apps_returns_empty_with_no_apps(initialized_env: AREEnvironment):
    """Test listing apps when scenario has no apps."""
    obs = initialized_env.step(ListAppsAction())

    assert obs.action_success is True
    assert "apps" in obs.action_result
    apps = obs.action_result["apps"]

    # Should return empty dict for no apps
    assert isinstance(apps, dict)
    assert len(apps) == 0


def test_list_apps_schema_structure(initialized_env: AREEnvironment):
    """Test that app listing has correct schema structure."""
    obs = initialized_env.step(ListAppsAction())

    assert obs.action_success is True
    assert "apps" in obs.action_result
    assert isinstance(obs.action_result["apps"], dict)


def test_call_tool_without_initialization():
    """Test that tool calling fails without initialization."""
    env = AREEnvironment()
    env.reset()

    obs = env.step(
        CallToolAction(app_name="calendar", tool_name="get_events", tool_args={})
    )

    assert obs.action_success is False
    assert "No scenario loaded" in obs.action_error


def test_call_tool_invalid_app(initialized_env: AREEnvironment):
    """Test calling tool on non-existent app."""
    obs = initialized_env.step(
        CallToolAction(app_name="nonexistent_app", tool_name="some_tool", tool_args={})
    )

    assert obs.action_success is False
    assert (
        "not found" in obs.action_error.lower()
        or "nonexistent" in obs.action_error.lower()
    )


def test_list_apps_available_apps_field(initialized_env: AREEnvironment):
    """Test that available_apps field is set correctly."""
    obs = initialized_env.step(ListAppsAction())

    assert obs.action_success is True
    # With no apps, available_apps should be an empty list
    assert obs.available_apps is not None
    assert isinstance(obs.available_apps, list)
    assert len(obs.available_apps) == 0


def test_call_tool_error_handling(initialized_env: AREEnvironment):
    """Test that tool call errors are handled properly."""
    obs = initialized_env.step(
        CallToolAction(
            app_name="fake_app",
            tool_name="fake_tool",
            tool_args={},
            advance_time=False,
        )
    )

    # Should fail gracefully
    assert obs.action_success is False
    assert obs.action_error is not None
    assert isinstance(obs.action_error, str)


def test_multiple_list_apps_calls(initialized_env: AREEnvironment):
    """Test calling list_apps multiple times."""
    # Should be idempotent
    obs1 = initialized_env.step(ListAppsAction())
    obs2 = initialized_env.step(ListAppsAction())

    assert obs1.action_success is True
    assert obs2.action_success is True
    assert obs1.action_result == obs2.action_result


def test_observation_fields_populated(initialized_env: AREEnvironment):
    """Test that observation fields are properly populated."""
    obs = initialized_env.step(ListAppsAction())

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


# Tests with DummyApp


@pytest.fixture
def env_with_dummy_app() -> AREEnvironment:
    """Environment with DummyApp registered."""
    env = AREEnvironment()
    env.reset()

    # Register DummyApp manually before initialization
    # (since scenario loading from JSON might not find the class)
    from are.simulation.scenarios.scenario import Scenario

    dummy_app = DummyApp()
    dummy_app.name = "test_dummy"

    # Create minimal scenario
    scenario = Scenario(
        scenario_id="dummy_test",
        duration=100,
        time_increment_in_seconds=1,
        apps=[dummy_app],
        events=[],
    )
    scenario.initialize()

    # Initialize ARE environment with the scenario
    from are.simulation.environment import Environment as ARESimEnv, EnvironmentConfig

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

    yield env
    if env._are_env is not None:
        env._are_env.stop()


def test_list_apps_with_dummy_app(env_with_dummy_app: AREEnvironment):
    """Test listing apps returns DummyApp with its tools."""
    obs = env_with_dummy_app.step(ListAppsAction())

    assert obs.action_success is True
    apps = obs.action_result["apps"]

    # Should have the dummy app
    assert "test_dummy" in apps
    tools = apps["test_dummy"]

    # Should have at least the log_stuff tool
    assert len(tools) > 0
    tool_names = [t["name"] for t in tools]
    assert any("log_stuff" in name for name in tool_names)


def test_tool_metadata_structure(env_with_dummy_app: AREEnvironment):
    """Test that tool metadata from to_metadata_dict has correct structure."""
    obs = env_with_dummy_app.step(ListAppsAction())

    assert obs.action_success is True
    tools = obs.action_result["apps"]["test_dummy"]

    for tool in tools:
        # Check metadata_dict structure
        assert "name" in tool
        assert "description" in tool
        assert "args" in tool
        assert isinstance(tool["args"], list)


def test_call_tool_success(env_with_dummy_app: AREEnvironment):
    """Test successfully calling a tool with arguments."""
    # First get the tool name
    list_obs = env_with_dummy_app.step(ListAppsAction())
    tools = list_obs.action_result["apps"]["test_dummy"]
    log_tool = next(t for t in tools if "log_stuff" in t["name"])

    # Call the tool
    obs = env_with_dummy_app.step(
        CallToolAction(
            app_name="test_dummy",
            tool_name=log_tool["name"],
            tool_args={"message": "test message"},
            advance_time=False,
        )
    )

    # Tool call action should succeed (even if tool itself has issues)
    assert obs.action_success is True
    assert obs.action_result is not None
    # Result should have success and result/error fields
    assert "success" in obs.action_result
    assert "result" in obs.action_result or "error" in obs.action_result


def test_call_tool_advances_time(env_with_dummy_app: AREEnvironment):
    """Test that calling a tool with advance_time=True advances time."""
    initial_time = env_with_dummy_app._are_env.current_time

    # Get tool name
    list_obs = env_with_dummy_app.step(ListAppsAction())
    tools = list_obs.action_result["apps"]["test_dummy"]
    log_tool = next(t for t in tools if "log_stuff" in t["name"])

    # Call with advance_time=True
    obs = env_with_dummy_app.step(
        CallToolAction(
            app_name="test_dummy",
            tool_name=log_tool["name"],
            tool_args={"message": "test"},
            advance_time=True,
        )
    )

    assert obs.action_success is True
    assert env_with_dummy_app._are_env.current_time > initial_time


def test_call_tool_no_time_advance(env_with_dummy_app: AREEnvironment):
    """Test that calling a tool with advance_time=False keeps time same."""
    initial_time = env_with_dummy_app._are_env.current_time

    # Get tool name
    list_obs = env_with_dummy_app.step(ListAppsAction())
    tools = list_obs.action_result["apps"]["test_dummy"]
    log_tool = next(t for t in tools if "log_stuff" in t["name"])

    # Call with advance_time=False
    obs = env_with_dummy_app.step(
        CallToolAction(
            app_name="test_dummy",
            tool_name=log_tool["name"],
            tool_args={"message": "test"},
            advance_time=False,
        )
    )

    assert obs.action_success is True
    assert env_with_dummy_app._are_env.current_time == initial_time
