# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Connect4 trajectory rubrics."""

import pytest
from dataclasses import dataclass
from typing import Any, List

from connect4_env.rubrics import Connect4WinLossRubric


@dataclass
class MockObservation:
    """Mock observation for testing."""

    done: bool = False
    reward: float = 0.0


@dataclass
class MockAction:
    """Mock action for testing."""

    column: int = 0


class TestConnect4WinLossRubric:
    """Test Connect4 trajectory rubric."""

    def test_returns_intermediate_until_done(self):
        """Returns 0.0 for non-terminal steps."""
        rubric = Connect4WinLossRubric(gamma=0.95)

        action = MockAction(column=3)
        obs = MockObservation(done=False, reward=0.0)

        reward = rubric(action, obs)
        assert reward == 0.0

    def test_win_returns_one(self):
        """Returns 1.0 when player wins."""
        rubric = Connect4WinLossRubric(gamma=0.95)

        # Simulate 3 moves leading to win
        rubric(MockAction(column=0), MockObservation(done=False))
        rubric(MockAction(column=1), MockObservation(done=False))
        reward = rubric(MockAction(column=2), MockObservation(done=True, reward=1.0))

        assert reward == 1.0

    def test_loss_returns_zero(self):
        """Returns 0.0 when player loses (invalid move)."""
        rubric = Connect4WinLossRubric(gamma=0.95)

        reward = rubric(MockAction(column=0), MockObservation(done=True, reward=-1))

        assert reward == 0.0  # default invalid_move_penalty

    def test_draw_returns_half(self):
        """Returns 0.5 for draw (board full, no winner)."""
        rubric = Connect4WinLossRubric(gamma=0.95)

        reward = rubric(MockAction(column=0), MockObservation(done=True, reward=0.0))

        assert reward == 0.5

    def test_discounting_with_gamma_one(self):
        """With gamma=1.0, all steps get equal credit."""
        rubric = Connect4WinLossRubric(gamma=1.0)

        # 5-step episode with win
        for _ in range(4):
            rubric(MockAction(column=0), MockObservation(done=False))
        rubric(MockAction(column=0), MockObservation(done=True, reward=1.0))

        step_rewards = rubric.compute_step_rewards()

        assert len(step_rewards) == 5
        assert all(r == 1.0 for r in step_rewards)

    def test_discounting_with_gamma_095(self):
        """With gamma=0.95, later steps get more credit."""
        rubric = Connect4WinLossRubric(gamma=0.95)

        # 3-step episode with win
        rubric(MockAction(column=0), MockObservation(done=False))
        rubric(MockAction(column=1), MockObservation(done=False))
        rubric(MockAction(column=2), MockObservation(done=True, reward=1.0))

        step_rewards = rubric.compute_step_rewards()

        # r_t = gamma^(T-1-t) * final_score
        # t=0: 0.95^2 = 0.9025
        # t=1: 0.95^1 = 0.95
        # t=2: 0.95^0 = 1.0
        assert step_rewards[0] == pytest.approx(0.9025)
        assert step_rewards[1] == pytest.approx(0.95)
        assert step_rewards[2] == pytest.approx(1.0)

    def test_reset_clears_trajectory(self):
        """Reset clears accumulated trajectory."""
        rubric = Connect4WinLossRubric(gamma=0.95)

        rubric(MockAction(column=0), MockObservation(done=False))
        rubric(MockAction(column=1), MockObservation(done=False))

        assert len(rubric._trajectory) == 2

        rubric.reset()

        assert len(rubric._trajectory) == 0

    def test_custom_invalid_move_penalty(self):
        """Can customize penalty for invalid moves."""
        rubric = Connect4WinLossRubric(gamma=0.95, invalid_move_penalty=-1.0)

        reward = rubric(MockAction(column=0), MockObservation(done=True, reward=-1))

        assert reward == -1.0

    def test_state_dict_serialization(self):
        """State dict includes custom parameters."""
        rubric = Connect4WinLossRubric(
            gamma=0.9,
            invalid_move_penalty=0.1,
            player_id=-1,
        )

        state = rubric.state_dict()

        assert state["gamma"] == 0.9
        assert state["invalid_move_penalty"] == 0.1
        assert state["player_id"] == -1

    def test_load_state_dict(self):
        """Can load configuration from state dict."""
        rubric = Connect4WinLossRubric(gamma=0.95)

        rubric.load_state_dict(
            {
                "gamma": 0.8,
                "invalid_move_penalty": 0.5,
                "player_id": -1,
            }
        )

        assert rubric.gamma == 0.8
        assert rubric.invalid_move_penalty == 0.5
        assert rubric.player_id == -1


class TestConnect4RubricIntegration:
    """Integration tests with Connect4 environment patterns."""

    def test_multiple_episodes(self):
        """Handles multiple episodes correctly."""
        rubric = Connect4WinLossRubric(gamma=0.95)

        # Episode 1: Win
        rubric(MockAction(column=0), MockObservation(done=False))
        rubric(MockAction(column=0), MockObservation(done=True, reward=1.0))
        ep1_rewards = rubric.compute_step_rewards()

        rubric.reset()

        # Episode 2: Loss
        rubric(MockAction(column=0), MockObservation(done=True, reward=-1))
        ep2_rewards = rubric.compute_step_rewards()

        assert ep1_rewards[-1] == 1.0  # Win
        assert ep2_rewards[0] == 0.0  # Loss (invalid move penalty)

    def test_long_game(self):
        """Handles longer games correctly."""
        rubric = Connect4WinLossRubric(gamma=0.99)

        # Simulate a 20-move game ending in win
        for _ in range(19):
            rubric(MockAction(column=0), MockObservation(done=False))
        rubric(MockAction(column=0), MockObservation(done=True, reward=1.0))

        step_rewards = rubric.compute_step_rewards()

        assert len(step_rewards) == 20
        # First move should have gamma^19 reward
        assert step_rewards[0] == pytest.approx(0.99**19)
        # Last move gets full reward
        assert step_rewards[-1] == 1.0
