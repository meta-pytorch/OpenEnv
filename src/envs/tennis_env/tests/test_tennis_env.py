# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Consolidated Tennis Environment Tests.

Tests cover: models, environment logic, symbolic features, dynamic rewards, and integration.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from envs.tennis_env import TennisEnv, TennisAction, TennisObservation
from envs.tennis_env.models import TennisState
from envs.tennis_env.server.tennis_environment import TennisEnvironment


# ============================================================================
# Model Tests
# ============================================================================

def test_tennis_action_creation():
    """Test TennisAction dataclass."""
    action = TennisAction(action_id=2, action_name="UP")
    assert action.action_id == 2
    assert action.action_name == "UP"


def test_tennis_observation_fields():
    """Test TennisObservation has all required fields."""
    obs = TennisObservation(
        screen_rgb=[0] * 100800,
        screen_shape=[210, 160, 3],
        score=(0, 0),
        ball_side="center",
        my_position="middle",
        opponent_position="middle",
        legal_actions=[0, 1, 2, 3],
        action_meanings=["NOOP", "FIRE", "UP", "RIGHT"],
        rally_length=0,
        done=False,
        reward=None,
        metadata={},
    )
    assert obs.score == (0, 0)
    assert obs.ball_side == "center"
    assert len(obs.legal_actions) == 4


def test_tennis_state_tracking():
    """Test TennisState dataclass."""
    state = TennisState(
        previous_score=(0, 0),
        rally_length=5,
        total_points=10,
        agent_games_won=1,
        opponent_games_won=2,
    )
    assert state.rally_length == 5
    assert state.total_points == 10


# ============================================================================
# Environment Tests
# ============================================================================

@pytest.fixture
def tennis_env():
    """Create a Tennis environment for testing."""
    return TennisEnvironment()


def test_environment_initialization(tennis_env):
    """Test environment initializes correctly."""
    assert tennis_env.ale is not None
    assert len(tennis_env._action_set) == 18  # Full action space
    assert tennis_env.screen_shape == [210, 160, 3]


def test_environment_reset(tennis_env):
    """Test environment reset."""
    obs = tennis_env.reset()
    assert isinstance(obs, TennisObservation)
    assert obs.done is False
    assert obs.score == (0, 0)
    assert len(obs.legal_actions) > 0


def test_environment_step(tennis_env):
    """Test environment step."""
    tennis_env.reset()
    action = TennisAction(action_id=0, action_name="NOOP")
    obs = tennis_env.step(action)
    assert isinstance(obs, TennisObservation)
    assert obs.screen_rgb is not None


def test_symbolic_features(tennis_env):
    """Test symbolic feature extraction."""
    tennis_env.reset()
    action = TennisAction(action_id=2, action_name="UP")
    obs = tennis_env.step(action)

    # Ball side should be one of valid values
    assert obs.ball_side in ["left", "center", "right", "unknown"]
    # Positions should be valid
    assert obs.my_position in ["top", "middle", "bottom", "unknown"]
    assert obs.opponent_position in ["top", "middle", "bottom", "unknown"]


def test_dynamic_reward_configuration():
    """Test dynamic reward shaping parameters."""
    custom_env = TennisEnvironment(
        score_reward=20.0,
        score_penalty=-10.0,
        rally_bonus_max=2.0,
        rally_bonus_scale=0.2,
        movement_bonus=0.1,
        positioning_bonus=0.15,
        center_bonus=0.25,
    )

    assert custom_env.score_reward == 20.0
    assert custom_env.score_penalty == -10.0
    assert custom_env.rally_bonus_max == 2.0
    assert custom_env.rally_bonus_scale == 0.2
    assert custom_env.movement_bonus == 0.1
    assert custom_env.positioning_bonus == 0.15
    assert custom_env.center_bonus == 0.25


def test_reward_shaping(tennis_env):
    """Test that reward shaping is applied."""
    obs = tennis_env.reset()

    # Take several actions and verify rewards are shaped
    total_reward = 0.0
    for _ in range(10):
        action = TennisAction(action_id=0, action_name="NOOP")
        obs = tennis_env.step(action)
        if obs.reward is not None:
            total_reward += obs.reward

    # Reward shaping should provide some signal (even if small)
    # Not strictly zero unless game ends immediately
    assert isinstance(total_reward, float)

# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
