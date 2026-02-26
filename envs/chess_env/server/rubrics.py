# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Chess-specific rubrics for reward computation.

Provides ChessWinLossRubric that extends ExponentialDiscountingTrajectoryRubric
to compute temporally-discounted rewards based on chess game outcomes.

See RFC 004 for rubric design: rfcs/004-rubrics.md
"""

from typing import Any, List, Tuple

from openenv.core.rubrics.trajectory import ExponentialDiscountingTrajectoryRubric


class ChessWinLossRubric(ExponentialDiscountingTrajectoryRubric):
    """Score chess game based on win/loss/draw outcome with temporal discounting.

    Per-step reward: r_t = gamma^(T-1-t) * R_final

    Terminal rewards:
    - Win:  +1.0
    - Loss: -1.0
    - Draw:  0.0

    Note: Unlike the base class convention of 0.0-1.0 scores, chess uses
    -1.0 for losses to match standard chess RL conventions.

    Usage:
        rubric = ChessWinLossRubric(gamma=0.99)
        rubric.reset()
        for action, obs in episode:
            reward = rubric(action, obs)  # 0.0 until done
        step_rewards = rubric.compute_step_rewards()
    """

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """Score based on game outcome from final observation.

        Reads the reward from the terminal observation, which encodes:
        +1.0 for agent win, -1.0 for agent loss, 0.0 for draw.

        Args:
            trajectory: List of (action, observation) tuples.

        Returns:
            Terminal reward: +1.0 (win), -1.0 (loss), 0.0 (draw).
        """
        if not trajectory:
            return 0.0
        _, final_obs = trajectory[-1]
        return getattr(final_obs, "reward", 0.0)
