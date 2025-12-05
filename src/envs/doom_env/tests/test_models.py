# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test suite for Doom models (DoomAction and DoomObservation).

Tests data model validation, serialization, and edge cases.
"""

import pytest
import numpy as np
from dataclasses import asdict

from ..models import DoomAction, DoomObservation


class TestDoomAction:
    """Tests for DoomAction model."""

    def test_doom_action_creation(self):
        """Verify that DoomAction can be created with valid action_id."""
        action = DoomAction(action_id=0)
        assert action.action_id == 0

        action = DoomAction(action_id=3)
        assert action.action_id == 3

    def test_doom_action_with_negative_id(self):
        """Test DoomAction with negative action_id (edge case)."""
        # Negative action_id should be allowed (let ViZDoom validate)
        action = DoomAction(action_id=-1)
        assert action.action_id == -1

    def test_doom_action_with_buttons(self):
        """Test DoomAction with custom button list."""
        buttons = [True, False, True]
        action = DoomAction(action_id=0, buttons=buttons)
        assert action.action_id == 0
        assert action.buttons == buttons

    def test_doom_action_serialization(self):
        """Verify DoomAction can be serialized to dict."""
        action = DoomAction(action_id=2)
        action_dict = asdict(action)

        assert isinstance(action_dict, dict)
        assert "action_id" in action_dict
        assert action_dict["action_id"] == 2

    def test_doom_action_numpy_int64_conversion(self):
        """Test DoomAction with numpy int64 (common edge case)."""
        # This is important for when actions come from np.random.choice
        np_action_id = np.int64(5)
        action = DoomAction(action_id=int(np_action_id))
        assert isinstance(action.action_id, int)
        assert action.action_id == 5

    def test_doom_action_with_none_buttons(self):
        """Test DoomAction with None buttons (default)."""
        action = DoomAction(action_id=1, buttons=None)
        assert action.action_id == 1
        assert action.buttons is None


class TestDoomObservation:
    """Tests for DoomObservation model."""

    def test_doom_observation_structure(self):
        """Verify DoomObservation has all required fields."""
        obs = DoomObservation(
            screen_buffer=[0, 255, 128],
            screen_shape=[120, 160, 3],
            game_variables=[100.0],
            available_actions=[0, 1, 2, 3],
            episode_finished=False,
            done=False,
            reward=0.0,
            metadata={}
        )

        assert obs.screen_buffer == [0, 255, 128]
        assert obs.screen_shape == [120, 160, 3]
        assert obs.game_variables == [100.0]
        assert obs.available_actions == [0, 1, 2, 3]
        assert obs.episode_finished is False
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.metadata == {}

    def test_doom_observation_screen_buffer_validation(self):
        """Ensure screen_buffer is a list of integers."""
        # Valid screen buffer
        obs = DoomObservation(
            screen_buffer=[0, 128, 255] * 100,
            screen_shape=[10, 10, 3],
            game_variables=None,
            available_actions=None,
            episode_finished=False,
            done=False,
            reward=0.0,
            metadata={}
        )
        assert isinstance(obs.screen_buffer, list)
        assert all(isinstance(x, int) for x in obs.screen_buffer)

    def test_doom_observation_screen_shape_validation(self):
        """Ensure screen_shape matches expected format [H, W, C]."""
        # Test different resolutions
        resolutions = [
            [120, 160, 3],  # RES_160X120
            [240, 320, 3],  # RES_320X240
            [480, 640, 3],  # RES_640X480
            [120, 160, 1],  # Grayscale
        ]

        for shape in resolutions:
            buffer_size = shape[0] * shape[1] * shape[2]
            obs = DoomObservation(
                screen_buffer=[0] * buffer_size,
                screen_shape=shape,
                game_variables=None,
                available_actions=None,
                episode_finished=False,
                done=False,
                reward=0.0,
                metadata={}
            )
            assert obs.screen_shape == shape
            assert len(obs.screen_shape) == 3

    def test_doom_observation_game_variables(self):
        """Verify game_variables field holds ViZDoom state info."""
        # With health only
        obs = DoomObservation(
            screen_buffer=[0],
            screen_shape=[1, 1, 1],
            game_variables=[100.0],
            available_actions=None,
            episode_finished=False,
            done=False,
            reward=0.0,
            metadata={}
        )
        assert obs.game_variables == [100.0]

        # With None (no variables available)
        obs2 = DoomObservation(
            screen_buffer=[0],
            screen_shape=[1, 1, 1],
            game_variables=None,
            available_actions=None,
            episode_finished=False,
            done=False,
            reward=0.0,
            metadata={}
        )
        assert obs2.game_variables is None

        # With multiple variables (health, ammo, etc.)
        obs3 = DoomObservation(
            screen_buffer=[0],
            screen_shape=[1, 1, 1],
            game_variables=[100.0, 50.0, 26.0],
            available_actions=None,
            episode_finished=False,
            done=False,
            reward=0.0,
            metadata={}
        )
        assert len(obs3.game_variables) == 3

    def test_doom_observation_serialization(self):
        """Verify DoomObservation can serialize/deserialize correctly."""
        original_obs = DoomObservation(
            screen_buffer=[255, 128, 0] * 10,
            screen_shape=[5, 2, 3],
            game_variables=[100.0, 50.0],
            available_actions=[0, 1, 2, 3],
            episode_finished=False,
            done=False,
            reward=5.0,
            metadata={"episode": 1}
        )

        # Serialize to dict
        obs_dict = asdict(original_obs)

        # Verify structure
        assert isinstance(obs_dict, dict)
        assert obs_dict["screen_buffer"] == original_obs.screen_buffer
        assert obs_dict["screen_shape"] == original_obs.screen_shape
        assert obs_dict["game_variables"] == original_obs.game_variables
        assert obs_dict["reward"] == 5.0

        # Deserialize back
        reconstructed_obs = DoomObservation(**obs_dict)
        assert reconstructed_obs.screen_buffer == original_obs.screen_buffer
        assert reconstructed_obs.screen_shape == original_obs.screen_shape
        assert reconstructed_obs.reward == original_obs.reward

    def test_doom_observation_empty_screen_buffer(self):
        """Test observation with empty screen buffer (edge case)."""
        obs = DoomObservation(
            screen_buffer=[],
            screen_shape=[0, 0, 0],
            game_variables=None,
            available_actions=None,
            episode_finished=True,
            done=True,
            reward=0.0,
            metadata={}
        )
        assert obs.screen_buffer == []
        assert obs.done is True

    def test_doom_observation_large_screen_buffer(self):
        """Test observation with large screen buffer (high resolution)."""
        # 800x600 RGB
        height, width, channels = 600, 800, 3
        buffer_size = height * width * channels

        obs = DoomObservation(
            screen_buffer=[128] * buffer_size,
            screen_shape=[height, width, channels],
            game_variables=[100.0],
            available_actions=[0, 1, 2, 3],
            episode_finished=False,
            done=False,
            reward=0.0,
            metadata={}
        )

        assert len(obs.screen_buffer) == buffer_size
        assert obs.screen_shape == [height, width, channels]

    def test_doom_observation_reward_types(self):
        """Test observation with different reward values."""
        # Positive reward (kill)
        obs1 = DoomObservation(
            screen_buffer=[0],
            screen_shape=[1, 1, 1],
            game_variables=None,
            available_actions=None,
            episode_finished=False,
            done=False,
            reward=100.0,
            metadata={}
        )
        assert obs1.reward == 100.0

        # Negative reward (damage)
        obs2 = DoomObservation(
            screen_buffer=[0],
            screen_shape=[1, 1, 1],
            game_variables=None,
            available_actions=None,
            episode_finished=False,
            done=False,
            reward=-10.0,
            metadata={}
        )
        assert obs2.reward == -10.0

        # Zero reward (no event)
        obs3 = DoomObservation(
            screen_buffer=[0],
            screen_shape=[1, 1, 1],
            game_variables=None,
            available_actions=None,
            episode_finished=False,
            done=False,
            reward=0.0,
            metadata={}
        )
        assert obs3.reward == 0.0

    def test_doom_observation_metadata(self):
        """Test observation metadata field."""
        metadata = {
            "episode": 5,
            "total_steps": 1000,
            "scenario": "basic"
        }

        obs = DoomObservation(
            screen_buffer=[0],
            screen_shape=[1, 1, 1],
            game_variables=None,
            available_actions=None,
            episode_finished=False,
            done=False,
            reward=0.0,
            metadata=metadata
        )

        assert obs.metadata == metadata
        assert obs.metadata["episode"] == 5
        assert obs.metadata["scenario"] == "basic"
