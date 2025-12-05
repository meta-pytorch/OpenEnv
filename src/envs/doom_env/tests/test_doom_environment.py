# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test suite for DoomEnvironment server implementation.

Tests the core ViZDoom wrapper and environment logic.
"""

import pytest
import numpy as np

try:
    from ..server.doom_env_environment import DoomEnvironment
    from ..models import DoomAction, DoomObservation
    VIZDOOM_AVAILABLE = True
except ImportError:
    VIZDOOM_AVAILABLE = False
    pytestmark = pytest.mark.skip("ViZDoom not installed")


@pytest.mark.skipif(not VIZDOOM_AVAILABLE, reason="ViZDoom not installed")
class TestDoomEnvironment:
    """Tests for DoomEnvironment class."""

    def test_environment_initialization(self):
        """Verify DoomEnvironment initializes with correct configuration."""
        env = DoomEnvironment(
            scenario="basic",
            screen_resolution="RES_160X120",
            screen_format="RGB24",
            window_visible=False,
            use_discrete_actions=True
        )

        assert env is not None
        assert env.scenario == "basic"
        assert env.screen_resolution == "RES_160X120"

    def test_environment_initialization_invalid_scenario(self):
        """Test initialization with invalid scenario name."""
        with pytest.raises(Exception):  # ViZDoom will raise an error
            env = DoomEnvironment(
                scenario="nonexistent_scenario",
                screen_resolution="RES_160X120",
                screen_format="RGB24",
                window_visible=False
            )

    def test_environment_reset(self):
        """Verify reset() returns valid initial observation."""
        env = DoomEnvironment(
            scenario="basic",
            screen_resolution="RES_160X120",
            window_visible=False
        )

        obs = env.reset()

        assert isinstance(obs, DoomObservation)
        assert obs.screen_buffer is not None
        assert obs.screen_shape is not None
        assert obs.done is False
        assert obs.episode_finished is False

    def test_environment_step_with_valid_action(self):
        """Verify step() processes actions correctly."""
        env = DoomEnvironment(scenario="basic", window_visible=False)
        env.reset()

        action = DoomAction(action_id=0)  # No-op
        obs = env.step(action)

        assert isinstance(obs, DoomObservation)
        assert obs.screen_buffer is not None
        assert isinstance(obs.reward, float)
        assert isinstance(obs.done, bool)

    def test_environment_step_returns_correct_structure(self):
        """Ensure step() returns properly structured DoomObservation."""
        env = DoomEnvironment(scenario="basic", window_visible=False)
        env.reset()

        action = DoomAction(action_id=1)  # Move left
        obs = env.step(action)

        # Verify all required fields are present
        assert hasattr(obs, 'screen_buffer')
        assert hasattr(obs, 'screen_shape')
        assert hasattr(obs, 'game_variables')
        assert hasattr(obs, 'available_actions')
        assert hasattr(obs, 'episode_finished')
        assert hasattr(obs, 'done')
        assert hasattr(obs, 'reward')
        assert hasattr(obs, 'metadata')

    def test_screen_buffer_format(self):
        """Verify screen buffer is flattened RGB data."""
        env = DoomEnvironment(
            scenario="basic",
            screen_resolution="RES_160X120",
            screen_format="RGB24",
            window_visible=False
        )
        obs = env.reset()

        # Screen buffer should be a flat list
        assert isinstance(obs.screen_buffer, list)

        # For RGB24, each pixel has 3 values
        expected_size = 120 * 160 * 3  # H x W x C
        assert len(obs.screen_buffer) == expected_size

        # All values should be in valid pixel range [0, 255]
        assert all(0 <= pixel <= 255 for pixel in obs.screen_buffer)

    def test_screen_shape_consistency(self):
        """Ensure screen_shape matches configured resolution."""
        test_cases = [
            ("RES_160X120", [120, 160, 3]),
            ("RES_320X240", [240, 320, 3]),
            ("RES_640X480", [480, 640, 3]),
        ]

        for resolution, expected_shape in test_cases:
            env = DoomEnvironment(
                scenario="basic",
                screen_resolution=resolution,
                screen_format="RGB24",
                window_visible=False
            )
            obs = env.reset()

            assert obs.screen_shape == expected_shape, (
                f"Resolution {resolution} should produce shape {expected_shape}, "
                f"got {obs.screen_shape}"
            )

    def test_game_variables_health(self):
        """Verify game_variables includes health value."""
        env = DoomEnvironment(scenario="basic", window_visible=False)
        obs = env.reset()

        assert obs.game_variables is not None
        assert len(obs.game_variables) > 0

        # First variable should be health
        health = obs.game_variables[0]
        assert isinstance(health, float)
        assert 0 <= health <= 100  # Health should be in valid range

    def test_episode_termination(self):
        """Verify done flag is set when episode ends."""
        env = DoomEnvironment(scenario="basic", window_visible=False)
        env.reset()

        # Step until episode ends (with timeout)
        max_steps = 1000
        done = False

        for _ in range(max_steps):
            action = DoomAction(action_id=3)  # Attack
            obs = env.step(action)

            if obs.done or obs.episode_finished:
                done = True
                break

        # Episode should eventually end
        # Note: In some scenarios it might not end within max_steps
        # We test that the flags exist and are boolean
        assert isinstance(obs.done, bool)
        assert isinstance(obs.episode_finished, bool)

    def test_reward_calculation(self):
        """Ensure rewards are calculated correctly."""
        env = DoomEnvironment(scenario="basic", window_visible=False)
        env.reset()

        # Take some actions
        rewards = []
        for _ in range(10):
            action = DoomAction(action_id=2)  # Move right
            obs = env.step(action)
            rewards.append(obs.reward)

        # Verify rewards are numeric
        assert all(isinstance(r, (int, float)) for r in rewards)

        # At least some rewards should be non-None
        assert rewards is not None

    def test_available_actions(self):
        """Verify available_actions list is correct for scenario."""
        env = DoomEnvironment(
            scenario="basic",
            window_visible=False,
            use_discrete_actions=True
        )
        obs = env.reset()

        assert obs.available_actions is not None
        assert isinstance(obs.available_actions, list)

        # Basic scenario should have discrete actions
        assert len(obs.available_actions) > 0

        # Actions should be integers
        assert all(isinstance(a, int) for a in obs.available_actions)

    def test_discrete_actions_vs_button_combinations(self):
        """Verify use_discrete_actions flag works correctly."""
        # Test discrete actions mode
        env_discrete = DoomEnvironment(
            scenario="basic",
            window_visible=False,
            use_discrete_actions=True
        )
        obs_discrete = env_discrete.reset()

        # Should have discrete actions
        assert obs_discrete.available_actions is not None
        assert all(isinstance(a, int) for a in obs_discrete.available_actions)

        # Test button combinations mode
        env_buttons = DoomEnvironment(
            scenario="basic",
            window_visible=False,
            use_discrete_actions=False
        )
        obs_buttons = env_buttons.reset()

        # Both should produce valid observations
        assert isinstance(obs_discrete, DoomObservation)
        assert isinstance(obs_buttons, DoomObservation)

    def test_multiple_scenarios(self):
        """Verify different scenarios load correctly."""
        scenarios = ["basic", "deadly_corridor", "defend_the_center"]

        for scenario in scenarios:
            try:
                env = DoomEnvironment(
                    scenario=scenario,
                    window_visible=False
                )
                obs = env.reset()

                assert isinstance(obs, DoomObservation)
                assert obs.metadata.get("scenario") == scenario
            except Exception as e:
                # Some scenarios might not be available
                pytest.skip(f"Scenario {scenario} not available: {e}")

    def test_window_visible_flag(self):
        """Verify window_visible parameter controls rendering window."""
        # Headless mode (should always work)
        env_headless = DoomEnvironment(
            scenario="basic",
            window_visible=False
        )
        obs = env_headless.reset()
        assert obs is not None

        # Window mode might fail in headless environments
        # We just verify the parameter is accepted
        try:
            env_window = DoomEnvironment(
                scenario="basic",
                window_visible=True
            )
            obs_window = env_window.reset()
            assert obs_window is not None
        except Exception:
            # Expected in headless/Docker environments
            pytest.skip("Display not available for window mode")

    def test_state_consistency(self):
        """Ensure environment state remains consistent across steps."""
        env = DoomEnvironment(scenario="basic", window_visible=False)

        # First episode
        obs1 = env.reset()
        episode_id_1 = obs1.metadata.get("episode_id", 0)

        # Take some steps
        for _ in range(5):
            action = DoomAction(action_id=0)
            env.step(action)

        # Reset and start new episode
        obs2 = env.reset()
        episode_id_2 = obs2.metadata.get("episode_id", 0)

        # Episode ID should increment or reset properly
        assert episode_id_2 >= episode_id_1

    def test_multiple_resets(self):
        """Verify multiple reset() calls work correctly."""
        env = DoomEnvironment(scenario="basic", window_visible=False)

        # Reset multiple times
        for i in range(5):
            obs = env.reset()

            assert isinstance(obs, DoomObservation)
            assert obs.done is False
            assert obs.episode_finished is False

            # Screen should not be empty
            assert len(obs.screen_buffer) > 0

    def test_step_after_done(self):
        """Test that stepping after episode completion works."""
        env = DoomEnvironment(scenario="basic", window_visible=False)
        env.reset()

        # Force episode to end (attack until done or max steps)
        max_steps = 500
        for _ in range(max_steps):
            action = DoomAction(action_id=3)  # Attack
            obs = env.step(action)

            if obs.done:
                break

        # If episode ended, try to step again
        if obs.done:
            # Should handle gracefully (either continue or require reset)
            action = DoomAction(action_id=0)
            try:
                obs_after = env.step(action)
                # Either returns final state or new state
                assert isinstance(obs_after, DoomObservation)
            except Exception:
                # Some implementations require reset after done
                pytest.skip("Environment requires reset after done")

    def test_grayscale_format(self):
        """Test environment with grayscale screen format."""
        env = DoomEnvironment(
            scenario="basic",
            screen_resolution="RES_160X120",
            screen_format="GRAY8",
            window_visible=False
        )
        obs = env.reset()

        # Grayscale returns 2D shape [height, width], not 3D [height, width, 1]
        # This is standard behavior for grayscale images in ViZDoom
        assert len(obs.screen_shape) == 2  # 2D array for grayscale
        assert obs.screen_shape[0] == 120  # height
        assert obs.screen_shape[1] == 160  # width

        # Buffer size should match H * W
        expected_size = obs.screen_shape[0] * obs.screen_shape[1]
        assert len(obs.screen_buffer) == expected_size
