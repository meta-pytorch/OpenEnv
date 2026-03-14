# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Rubric implementation for the QED Math Environment.

Wraps LLMJudge to grade math proofs on a 0-7 scale and normalizes
the score to a [0, 1] reward signal.

Reference:
    eval/src/imobench/evaluation.py - Core evaluation logic
    eval/configs/prompts/proofbench.txt - Grading rubric structure
"""


class MathProofRubric:
    """
    LLM-based rubric for grading mathematical proofs.

    Grades proofs on a 0-7 scale using an LLM judge, then normalizes
    the score to a [0, 1] reward.
    """

    def grade(self, proof: str, problem: str, reference_solution: str) -> tuple:
        """
        Grade a proof submission.

        Args:
            proof: The proof text submitted by the agent.
            problem: The problem statement.
            reference_solution: The ground truth solution.

        Returns:
            Tuple of (score: int, feedback: str) where score is 0-7.
        """
        raise NotImplementedError

    def normalize_reward(self, score: int) -> float:
        """Normalize a 0-7 score to a [0, 1] reward."""
        return score / 7.0
