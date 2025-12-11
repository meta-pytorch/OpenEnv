# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test suite for DoomEnv HTTP client.

Tests client-server communication, serialization, and rendering.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from ..client import DoomEnv
from ..models import DoomAction, DoomObservation
from openenv_core.client_types import StepResult
from openenv_core.env_server.types import State


class TestDoomClient:
    """Tests for DoomEnv HTTP client."""

    def test_client_initialization(self):
        """Verify DoomEnv client can be created with base_url."""
        client = DoomEnv(base_url="http://localhost:8000")

        assert client is not None
        # DoomEnv inherits from HTTPEnvClient which stores URL internally
        # We verify initialization succeeded by checking internal state
        assert hasattr(client, '_last_observation')
        assert hasattr(client, '_render_window')

    def test_client_initialization_invalid_url(self):
        """Test client creation with invalid URL format."""
        # Should still create client (connection tested on first request)
        client = DoomEnv(base_url="not-a-valid-url")
        assert client is not None

    def test_step_payload_serialization(self):
        """Ensure _step_payload() converts DoomAction to JSON correctly."""
        client = DoomEnv(base_url="http://localhost:8000")

        # Test with regular Python int
        action1 = DoomAction(action_id=5)
        payload1 = client._step_payload(action1)

        assert isinstance(payload1, dict)
        assert "action_id" in payload1
        assert payload1["action_id"] == 5
        assert isinstance(payload1["action_id"], int)

    def test_step_payload_numpy_int64_conversion(self):
        """Test _step_payload() converts numpy int64 to native Python int."""
        client = DoomEnv(base_url="http://localhost:8000")

        # Test with numpy int64 (common from np.random.choice)
        np_action_id = np.int64(7)
        action = DoomAction(action_id=int(np_action_id))
        payload = client._step_payload(action)

        assert isinstance(payload["action_id"], int)
        assert payload["action_id"] == 7

    def test_step_payload_with_buttons(self):
        """Test _step_payload() handles button list correctly."""
        client = DoomEnv(base_url="http://localhost:8000")

        buttons = [True, False, True, False]
        action = DoomAction(action_id=0, buttons=buttons)
        payload = client._step_payload(action)

        assert "action_id" in payload
        assert "buttons" in payload
        assert payload["buttons"] == buttons

    def test_step_payload_filters_none_values(self):
        """Test _step_payload() filters out None values."""
        client = DoomEnv(base_url="http://localhost:8000")

        action = DoomAction(action_id=3, buttons=None)
        payload = client._step_payload(action)

        # None values should be filtered out
        assert "action_id" in payload
        assert "buttons" not in payload  # Should be filtered since it's None

    def test_parse_result_structure(self):
        """Verify _parse_result() converts JSON to DoomObservation."""
        client = DoomEnv(base_url="http://localhost:8000")

        # Mock server response
        server_response = {
            "observation": {
                "screen_buffer": [255, 128, 0] * 100,
                "screen_shape": [10, 10, 3],
                "game_variables": [100.0, 50.0],
                "available_actions": [0, 1, 2, 3],
                "episode_finished": False,
                "done": False,
                "metadata": {"episode": 1}
            },
            "reward": 5.0,
            "done": False
        }

        result = client._parse_result(server_response)

        assert isinstance(result, StepResult)
        assert isinstance(result.observation, DoomObservation)
        assert result.reward == 5.0
        assert result.done is False

    def test_parse_result_observation_fields(self):
        """Test _parse_result() correctly parses all observation fields."""
        client = DoomEnv(base_url="http://localhost:8000")

        server_response = {
            "observation": {
                "screen_buffer": [128] * 300,
                "screen_shape": [10, 10, 3],
                "game_variables": [75.0],
                "available_actions": [0, 1, 2],
                "episode_finished": False,
                "done": False,
                "metadata": {"scenario": "basic"}
            },
            "reward": 10.0,
            "done": False
        }

        result = client._parse_result(server_response)
        obs = result.observation

        assert obs.screen_buffer == [128] * 300
        assert obs.screen_shape == [10, 10, 3]
        assert obs.game_variables == [75.0]
        assert obs.available_actions == [0, 1, 2]
        assert obs.episode_finished is False
        assert obs.metadata == {"scenario": "basic"}

    def test_parse_result_with_missing_fields(self):
        """Test _parse_result() handles missing optional fields."""
        client = DoomEnv(base_url="http://localhost:8000")

        # Minimal response
        server_response = {
            "observation": {
                "screen_buffer": [],
                "screen_shape": [120, 160, 3],
                "episode_finished": False,
                "done": True,
                "metadata": {}
            },
            "reward": 0.0,
            "done": True
        }

        result = client._parse_result(server_response)

        assert isinstance(result.observation, DoomObservation)
        assert result.observation.game_variables is None  # Should default to None
        assert result.done is True

    def test_parse_state(self):
        """Verify _parse_state() converts JSON to State object."""
        client = DoomEnv(base_url="http://localhost:8000")

        state_response = {
            "episode_id": 5,
            "step_count": 42
        }

        state = client._parse_state(state_response)

        assert isinstance(state, State)
        assert state.episode_id == 5
        assert state.step_count == 42

    def test_parse_state_with_missing_step_count(self):
        """Test _parse_state() handles missing step_count."""
        client = DoomEnv(base_url="http://localhost:8000")

        state_response = {
            "episode_id": 3
        }

        state = client._parse_state(state_response)

        assert state.episode_id == 3
        assert state.step_count == 0  # Should default to 0

    def test_render_rgb_array_mode(self):
        """Verify render(mode='rgb_array') returns numpy array."""
        client = DoomEnv(base_url="http://localhost:8000")

        # Mock last observation with correct buffer size (10 * 10 * 3 = 300)
        client._last_observation = DoomObservation(
            screen_buffer=[255, 128, 0] * 100,  # 300 elements total
            screen_shape=[10, 10, 3],
            game_variables=None,
            available_actions=None,
            episode_finished=False,
            done=False,
            reward=0.0,
            metadata={}
        )

        result = client.render(mode="rgb_array")

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10, 3)
        assert result.dtype == np.uint8

    def test_render_rgb_array_pixel_values(self):
        """Test render() produces correct pixel values."""
        client = DoomEnv(base_url="http://localhost:8000")

        # Create specific pattern
        screen_buffer = [255, 0, 0, 0, 255, 0]  # Red pixel, Green pixel
        client._last_observation = DoomObservation(
            screen_buffer=screen_buffer,
            screen_shape=[1, 2, 3],
            game_variables=None,
            available_actions=None,
            episode_finished=False,
            done=False,
            reward=0.0,
            metadata={}
        )

        result = client.render(mode="rgb_array")

        assert result[0, 0, 0] == 255  # Red channel
        assert result[0, 0, 1] == 0    # Green channel
        assert result[0, 0, 2] == 0    # Blue channel
        assert result[0, 1, 0] == 0    # Second pixel red
        assert result[0, 1, 1] == 255  # Second pixel green

    def test_render_without_observation(self):
        """Test render() when no observation is available."""
        client = DoomEnv(base_url="http://localhost:8000")

        # No observation set
        result = client.render(mode="rgb_array")

        assert result is None

    def test_render_invalid_mode(self):
        """Test render() with invalid mode raises error."""
        client = DoomEnv(base_url="http://localhost:8000")

        client._last_observation = DoomObservation(
            screen_buffer=[0],
            screen_shape=[1, 1, 1],
            game_variables=None,
            available_actions=None,
            episode_finished=False,
            done=False,
            reward=0.0,
            metadata={}
        )

        with pytest.raises(ValueError, match="Invalid render mode"):
            client.render(mode="invalid_mode")

    @patch('cv2.namedWindow')
    @patch('cv2.imshow')
    @patch('cv2.waitKey', return_value=ord('q'))
    @patch('cv2.cvtColor', return_value=np.zeros((10, 10, 3), dtype=np.uint8))
    def test_render_human_mode_with_cv2(self, mock_cvtColor, mock_waitKey, mock_imshow, mock_namedWindow):
        """Test render(mode='human') uses OpenCV when available."""
        client = DoomEnv(base_url="http://localhost:8000")

        # Mock last observation with correct buffer size
        client._last_observation = DoomObservation(
            screen_buffer=[128] * 300,  # 10 * 10 * 3 = 300
            screen_shape=[10, 10, 3],
            game_variables=[100.0],
            available_actions=[0, 1, 2, 3],
            episode_finished=False,
            done=False,
            reward=5.0,
            metadata={}
        )

        result = client.render(mode="human")

        # Should return None in human mode
        assert result is None

        # Should have called cv2 functions
        mock_namedWindow.assert_called_once()
        mock_imshow.assert_called_once()

    def test_close_cleanup(self):
        """Verify client.close() cleans up resources."""
        client = DoomEnv(base_url="http://localhost:8000")

        # Set up some state
        client._last_observation = DoomObservation(
            screen_buffer=[0],
            screen_shape=[1, 1, 1],
            game_variables=None,
            available_actions=None,
            episode_finished=False,
            done=False,
            reward=0.0,
            metadata={}
        )

        # Close should not raise error
        client.close()

        # Window should be cleared
        assert client._render_window is None

    @patch('cv2.destroyAllWindows')
    def test_close_destroys_cv2_windows(self, mock_destroyAllWindows):
        """Test close() destroys OpenCV windows if they exist."""
        client = DoomEnv(base_url="http://localhost:8000")

        # Simulate window creation
        client._render_window = "test_window"

        client.close()

        # Should have called destroyAllWindows
        mock_destroyAllWindows.assert_called_once()
        assert client._render_window is None

    def test_step_payload_with_numpy_array_buttons(self):
        """Test _step_payload() handles numpy arrays in buttons."""
        client = DoomEnv(base_url="http://localhost:8000")

        # Buttons as numpy array
        buttons = np.array([True, False, True])
        action = DoomAction(action_id=0, buttons=buttons.tolist())
        payload = client._step_payload(action)

        assert "buttons" in payload
        assert isinstance(payload["buttons"], list)
        assert payload["buttons"] == [True, False, True]

    def test_observation_stored_for_rendering(self):
        """Test that _parse_result() stores observation for rendering."""
        client = DoomEnv(base_url="http://localhost:8000")

        server_response = {
            "observation": {
                "screen_buffer": [255] * 100,
                "screen_shape": [10, 10, 1],
                "game_variables": None,
                "available_actions": None,
                "episode_finished": False,
                "done": False,
                "metadata": {}
            },
            "reward": 0.0,
            "done": False
        }

        result = client._parse_result(server_response)

        # Observation should be stored in _last_observation
        assert client._last_observation is not None
        assert client._last_observation == result.observation
        assert client._last_observation.screen_buffer == [255] * 100
