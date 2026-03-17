# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trajectory-based rubrics for Connect4 environment.

This module demonstrates the TrajectoryRubric pattern from RFC 004
for terminal games where reward signals depend on game outcome.

Connect4 is ideal for demonstrating trajectory rubrics because:
- Win/loss is only known at game end
- Clear semantics: 1.0 for win, 0.0 for loss, 0.5 for draw
- Discounting can assign more credit to decisive late-game moves
"""

from typing import Any, Dict, List, Tuple

from openenv.core.rubrics import ExponentialDiscountingTrajectoryRubric


class Connect4WinLossRubric(ExponentialDiscountingTrajectoryRubric):
    """Trajectory rubric that scores Connect4 games based on outcome.

    Scores:
    - 1.0 for win (player made 4 in a row)
    - 0.0 for loss (opponent made 4 in a row, or player made invalid move)
    - 0.5 for draw (board full, no winner)

    With exponential discounting, later moves (closer to the decisive
    outcome) receive higher rewards. This helps credit assignment:
    the move that completes 4-in-a-row gets the most credit.

    Usage:
        rubric = Connect4WinLossRubric(gamma=0.95)
        env = Connect4Environment(rubric=rubric)

        obs = env.reset()
        while not obs.done:
            action = agent.act(obs)
            obs = env.step(action)

        # Get per-step rewards for training
        step_rewards = rubric.compute_step_rewards()
        # step_rewards[i] = gamma^(T-1-i) * final_score
    """

    def __init__(
        self,
        gamma: float = 0.95,
        invalid_move_penalty: float = 0.0,
        player_id: int = 1,
    ):
        """Initialize Connect4 trajectory rubric.

        Args:
            gamma: Discount factor for credit assignment. 0.95 gives
                more credit to later (decisive) moves.
            invalid_move_penalty: Score when player makes invalid move.
                Default 0.0 (treat as loss).
            player_id: Which player we're scoring for (1 or -1).
        """
        super().__init__(gamma=gamma, intermediate_reward=0.0)
        self.invalid_move_penalty = invalid_move_penalty
        self.player_id = player_id

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """Score based on game outcome.

        Returns:
            1.0 for win, 0.0 for loss, 0.5 for draw.
        """
        if not trajectory:
            return 0.0

        _, final_obs = trajectory[-1]

        # Check for done observation
        if not getattr(final_obs, "done", False):
            return 0.0

        # Get reward from observation
        reward = getattr(final_obs, "reward", 0.0)
        if reward is None:
            reward = 0.0

        # Interpret reward:
        # -1 = invalid move (loss)
        # 1.0 = win
        # 0.0 with done = draw
        if reward == -1:
            return self.invalid_move_penalty
        elif reward == 1.0:
            return 1.0
        elif reward == 0.0:
            # Draw (board full, no winner)
            return 0.5
        else:
            # Unexpected value, treat as loss
            return 0.0

    def state_dict(self) -> Dict[str, Any]:
        """Serialize configuration."""
        state = super().state_dict()
        state["invalid_move_penalty"] = self.invalid_move_penalty
        state["player_id"] = self.player_id
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load configuration from checkpoint."""
        super().load_state_dict(state)
        if "invalid_move_penalty" in state:
            self.invalid_move_penalty = state["invalid_move_penalty"]
        if "player_id" in state:
            self.player_id = state["player_id"]


__all__ = [
    "Connect4WinLossRubric",
]
