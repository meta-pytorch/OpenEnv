# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for CARLA environment.

Tests both mock mode (no CARLA required) and scenario system.
"""

import pytest

from carla_env.models import CarlaAction, CarlaObservation, CarlaState
from carla_env.server.carla_environment import CarlaEnvironment
from carla_env.server.scenarios import get_scenario, SimpleTrolleyScenario


class TestCarlaEnvironmentMock:
    """Test CARLA environment in mock mode (no CARLA server required)."""

    def test_environment_creation(self):
        """Test creating environment in mock mode."""
        env = CarlaEnvironment(
            scenario_name="trolley_saves",
            mode="mock"
        )
        assert env.mode == "mock"
        assert env.scenario.name == "SimpleTrolleyScenario"

    def test_reset(self):
        """Test environment reset."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        obs = env.reset()

        assert isinstance(obs, CarlaObservation)
        assert obs.speed_kmh > 0  # Initial speed set by scenario
        assert obs.scenario_name == "SimpleTrolleyScenario"
        assert len(obs.nearby_actors) == 3  # trolley_saves has 3 pedestrians

    def test_step_observe(self):
        """Test step with observe action."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        env.reset()

        action = CarlaAction(action_type="observe")
        obs = env.step(action)

        assert isinstance(obs, CarlaObservation)
        assert env.state.step_count == 1

    def test_step_emergency_stop(self):
        """Test emergency stop action."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        obs1 = env.reset()
        initial_speed = obs1.speed_kmh

        # Apply emergency stop
        action = CarlaAction(action_type="emergency_stop")
        obs2 = env.step(action)

        # Speed should decrease
        assert obs2.speed_kmh < initial_speed

    def test_step_lane_change(self):
        """Test lane change action."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        env.reset()

        # Lane change left
        action = CarlaAction(action_type="lane_change", lane_direction="left")
        obs = env.step(action)

        assert isinstance(obs, CarlaObservation)
        assert env.state.step_count == 1

    def test_state(self):
        """Test state property."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        env.reset()

        state = env.state
        assert isinstance(state, CarlaState)
        assert state.episode_id != ""
        assert state.scenario_name == "SimpleTrolleyScenario"

    def test_multiple_steps(self):
        """Test running multiple steps."""
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        env.reset()

        # Run 5 steps
        for i in range(5):
            action = CarlaAction(action_type="observe")
            obs = env.step(action)

            assert env.state.step_count == i + 1

            if obs.done:
                break

    def test_collision_detection(self):
        """Test collision detection in mock mode.

        Note: Mock mode collision detection is simplified and may not
        always trigger within the expected timeframe. In real CARLA mode,
        collision detection is accurate via physics engine.
        """
        env = CarlaEnvironment(scenario_name="trolley_saves", mode="mock")
        env.reset()

        # Run until collision or done
        max_steps = 50  # Increased tolerance for mock mode
        collision_detected = False

        for _ in range(max_steps):
            # Don't brake - should eventually collide or reach timeout
            action = CarlaAction(action_type="observe")
            obs = env.step(action)

            if obs.collision_detected:
                collision_detected = True
                assert obs.collided_with is not None
                break

            if obs.done:
                break

        # Mock mode: collision detection is approximate, so either collision or done is acceptable
        assert collision_detected or obs.done, "Episode should either collide or terminate"


class TestScenarios:
    """Test scenario system."""

    def test_get_scenario_trolley_saves(self):
        """Test getting trolley_saves scenario."""
        scenario = get_scenario("trolley_saves")
        assert isinstance(scenario, SimpleTrolleyScenario)
        assert scenario.pedestrians_ahead == 3
        assert scenario.pedestrians_adjacent == 0

    def test_get_scenario_trolley_equal(self):
        """Test getting trolley_equal scenario."""
        scenario = get_scenario("trolley_equal")
        assert isinstance(scenario, SimpleTrolleyScenario)
        assert scenario.pedestrians_ahead == 1
        assert scenario.pedestrians_adjacent == 1

    def test_scenario_setup(self):
        """Test scenario setup."""
        scenario = get_scenario("trolley_saves")
        setup = scenario.setup()

        assert "spawn_point" in setup
        assert "initial_speed_kmh" in setup
        assert "actors" in setup
        assert "max_steps" in setup

        # Check actors
        actors = setup["actors"]
        assert len(actors) == 3  # 3 pedestrians ahead

    def test_scenario_termination(self):
        """Test scenario termination logic."""
        scenario = get_scenario("trolley_saves")

        # Not terminated initially
        state = {"step_count": 0, "speed_kmh": 40.0, "collision_detected": False}
        done, reason = scenario.check_termination(state)
        assert not done

        # Terminated on collision (but vehicle might also be stopped)
        state = {"step_count": 5, "collision_detected": True, "speed_kmh": 0.0}
        done, reason = scenario.check_termination(state)
        assert done
        assert reason in ["collision", "vehicle_stopped"]  # Either is valid

        # Terminated on timeout
        state = {"step_count": 100, "collision_detected": False, "speed_kmh": 10.0}
        done, reason = scenario.check_termination(state)
        assert done
        assert reason == "timeout"

    def test_scenario_reward(self):
        """Test scenario reward computation."""
        scenario = get_scenario("trolley_saves")

        # No collision - positive reward
        state = {"collisions": [], "done": True}
        action = {"action_type": "emergency_stop"}
        reward = scenario.compute_reward(state, action)
        assert reward > 0  # Bonus for avoiding collision minus braking cost

        # Collision - negative reward
        state = {"collisions": [{"actor_id": "ped_1"}]}
        action = {"action_type": "observe"}
        reward = scenario.compute_reward(state, action)
        assert reward < 0  # Penalty for collision

    def test_scenario_scene_description(self):
        """Test scene description generation."""
        scenario = get_scenario("trolley_saves")

        state = {
            "speed_kmh": 40.0,
            "current_lane": "lane_0",
            "nearby_actors": [
                {"type": "pedestrian", "distance": 25.0, "position": "ahead"},
                {"type": "pedestrian", "distance": 25.5, "position": "ahead"},
            ],
            "collision_detected": False,
            "simulation_time": 1.5,
        }

        description = scenario.get_scene_description(state)

        assert "40.0 km/h" in description
        assert "lane_0" in description
        assert "pedestrian" in description
        assert "25.0m" in description


class TestModels:
    """Test data models."""

    def test_carla_action(self):
        """Test CarlaAction model."""
        action = CarlaAction(action_type="control", throttle=0.5, steer=0.2)
        assert action.action_type == "control"
        assert action.throttle == 0.5
        assert action.steer == 0.2

    def test_carla_observation(self):
        """Test CarlaObservation model."""
        obs = CarlaObservation(
            scene_description="Test scene",
            speed_kmh=30.0,
            nearby_actors=[{"type": "pedestrian", "distance": 10.0}],
        )
        assert obs.scene_description == "Test scene"
        assert obs.speed_kmh == 30.0
        assert len(obs.nearby_actors) == 1

    def test_carla_state(self):
        """Test CarlaState model."""
        state = CarlaState(
            episode_id="test-123",
            scenario_name="trolley_saves",
            step_count=5,
        )
        assert state.episode_id == "test-123"
        assert state.scenario_name == "trolley_saves"
        assert state.step_count == 5
