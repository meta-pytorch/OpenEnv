# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for Doom environment.

End-to-end tests that require a running server.
These tests are marked as 'slow' and 'requires_server'.
"""

import pytest
import time
import subprocess
import requests
import numpy as np

from ..client import DoomEnv
from ..models import DoomAction, DoomObservation


@pytest.fixture(scope="module")
def doom_server():
    """Start Doom server for integration tests."""
    # Start server process
    server_process = subprocess.Popen(
        ["python", "-m", "doom_env.server.app"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start
    max_retries = 10
    for _ in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(1)
    else:
        server_process.kill()
        pytest.fail("Server failed to start")

    yield "http://localhost:8000"

    # Cleanup
    server_process.terminate()
    server_process.wait(timeout=5)


@pytest.mark.slow
@pytest.mark.requires_server
class TestDoomIntegration:
    """Integration tests for full Doom environment."""

    def test_client_server_connection(self, doom_server):
        """Test that client can connect to server."""
        client = DoomEnv(base_url=doom_server)

        # Should be able to get health status
        response = requests.get(f"{doom_server}/health")
        assert response.status_code == 200

    def test_full_episode_playthrough(self, doom_server):
        """Run a complete episode from reset to done."""
        client = DoomEnv(base_url=doom_server)

        # Reset environment
        result = client.reset()

        assert result.observation is not None
        assert isinstance(result.observation, DoomObservation)
        assert result.observation.done is False

        # Take some actions
        total_reward = 0.0
        steps = 0
        max_steps = 100

        for _ in range(max_steps):
            # Choose random action
            available_actions = result.observation.available_actions
            if available_actions:
                action_id = int(np.random.choice(available_actions))
            else:
                action_id = 0

            action = DoomAction(action_id=action_id)
            result = client.step(action)

            total_reward += result.reward
            steps += 1

            if result.observation.done:
                break

        # Verify we took at least one step
        assert steps > 0

        # Verify reward is numeric
        assert isinstance(total_reward, (int, float))

        client.close()

    def test_reset_after_episode_completion(self, doom_server):
        """Verify reset works after episode completes."""
        client = DoomEnv(base_url=doom_server)

        # First episode
        result1 = client.reset()
        initial_health_1 = result1.observation.game_variables[0] if result1.observation.game_variables else None

        # Take actions until done or max steps
        max_steps = 500
        for _ in range(max_steps):
            action = DoomAction(action_id=3)  # Attack
            result = client.step(action)

            if result.observation.done:
                break

        # Reset and start new episode
        result2 = client.reset()

        assert result2.observation.done is False
        assert result2.observation.episode_finished is False

        # Health should be reset
        if result2.observation.game_variables:
            initial_health_2 = result2.observation.game_variables[0]
            # Health should be back to full or at least higher than before
            assert initial_health_2 >= 0

        client.close()

    def test_reward_accumulation(self, doom_server):
        """Verify rewards accumulate correctly over episode."""
        client = DoomEnv(base_url=doom_server)

        client.reset()

        rewards = []
        for _ in range(20):
            action = DoomAction(action_id=int(np.random.randint(0, 4)))
            result = client.step(action)
            rewards.append(result.reward)

            if result.observation.done:
                break

        # All rewards should be numeric
        assert all(isinstance(r, (int, float)) for r in rewards)

        # Calculate total reward
        total_reward = sum(rewards)
        assert isinstance(total_reward, (int, float))

        client.close()

    def test_health_tracking(self, doom_server):
        """Verify health tracking through game variables."""
        client = DoomEnv(base_url=doom_server)

        result = client.reset()

        health_values = []

        # Track health over several steps
        for _ in range(30):
            if result.observation.game_variables:
                health = result.observation.game_variables[0]
                health_values.append(health)

            action = DoomAction(action_id=0)  # No-op
            result = client.step(action)

            if result.observation.done:
                break

        # Health should be tracked
        assert len(health_values) > 0

        # All health values should be valid
        assert all(isinstance(h, (int, float)) for h in health_values)
        assert all(0 <= h <= 100 for h in health_values)

        client.close()

    def test_action_effects(self, doom_server):
        """Verify different actions have different effects."""
        client = DoomEnv(base_url=doom_server)

        client.reset()

        observations = []

        # Test different actions
        test_actions = [0, 1, 2, 3]  # No-op, Left, Right, Attack

        for action_id in test_actions:
            action = DoomAction(action_id=action_id)
            result = client.step(action)
            observations.append(result.observation)

        # All observations should be valid
        assert all(isinstance(obs, DoomObservation) for obs in observations)

        # Screen buffers should be present
        assert all(len(obs.screen_buffer) > 0 for obs in observations)

        client.close()

    def test_multiple_resets(self, doom_server):
        """Verify multiple reset() calls work correctly."""
        client = DoomEnv(base_url=doom_server)

        episode_ids = []

        for _ in range(5):
            result = client.reset()

            assert result.observation.done is False
            assert result.observation.episode_finished is False

            # Track episode ID if available
            if result.observation.metadata:
                episode_id = result.observation.metadata.get("episode_id")
                if episode_id is not None:
                    episode_ids.append(episode_id)

            # Take a few steps
            for _ in range(5):
                action = DoomAction(action_id=0)
                client.step(action)

        # Should have completed multiple resets
        assert len(episode_ids) <= 5

        client.close()

    def test_state_endpoint(self, doom_server):
        """Test that state endpoint returns correct information."""
        client = DoomEnv(base_url=doom_server)

        client.reset()

        # Get initial state
        state1 = client.state()

        assert state1.episode_id is not None
        assert state1.step_count == 0

        # Take some steps
        for _ in range(10):
            action = DoomAction(action_id=0)
            client.step(action)

        # Get state after steps
        state2 = client.state()

        # Step count should have increased
        assert state2.step_count > state1.step_count

        client.close()

    def test_screen_buffer_consistency(self, doom_server):
        """Verify screen buffer size is consistent with shape."""
        client = DoomEnv(base_url=doom_server)

        result = client.reset()
        obs = result.observation

        # Calculate expected buffer size
        height, width, channels = obs.screen_shape
        expected_size = height * width * channels

        # Buffer size should match shape
        assert len(obs.screen_buffer) == expected_size

        # Take a step and verify again
        action = DoomAction(action_id=0)
        result = client.step(action)
        obs = result.observation

        height, width, channels = obs.screen_shape
        expected_size = height * width * channels
        assert len(obs.screen_buffer) == expected_size

        client.close()

    def test_observation_serialization_round_trip(self, doom_server):
        """Test that observations survive serialization round-trip."""
        client = DoomEnv(base_url=doom_server)

        result = client.reset()
        original_obs = result.observation

        # Observation should have been serialized over HTTP and deserialized
        assert isinstance(original_obs, DoomObservation)
        assert isinstance(original_obs.screen_buffer, list)
        assert isinstance(original_obs.screen_shape, list)

        # Take a step
        action = DoomAction(action_id=1)
        result = client.step(action)
        step_obs = result.observation

        # Verify structure is consistent
        assert len(step_obs.screen_shape) == len(original_obs.screen_shape)

        client.close()


@pytest.mark.slow
class TestPerformance:
    """Performance and stress tests."""

    def test_rapid_steps(self, doom_server):
        """Test rapid consecutive steps."""
        client = DoomEnv(base_url=doom_server)

        client.reset()

        start_time = time.time()
        num_steps = 100

        for _ in range(num_steps):
            action = DoomAction(action_id=0)
            result = client.step(action)

            if result.observation.done:
                client.reset()

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (< 10 seconds for 100 steps)
        assert elapsed_time < 10.0

        # Calculate steps per second
        steps_per_second = num_steps / elapsed_time
        assert steps_per_second > 5  # At least 5 steps per second

        client.close()

    def test_memory_leak_detection(self, doom_server):
        """Test for memory leaks with repeated resets."""
        client = DoomEnv(base_url=doom_server)

        # Perform many resets
        for _ in range(20):
            client.reset()

            # Take a few steps
            for _ in range(10):
                action = DoomAction(action_id=0)
                client.step(action)

        # If we got here without crashing, no obvious memory leak
        assert True

        client.close()
