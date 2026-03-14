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

    def __init__(
        self,
        grader_model: str = "gemini-2.0-flash",
        prompt_template: str = "",
        custom_threshold: bool = False,
    ):
        """Initialize rubric configuration.

        Args:
            grader_model: Model identifier to use for grading.
            prompt_template: Prompt template for the grader.
            custom_threshold: Whether custom thresholding is enabled.
        """
        self.grader_model = grader_model
        self.prompt_template = prompt_template
        self.custom_threshold = custom_threshold

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
