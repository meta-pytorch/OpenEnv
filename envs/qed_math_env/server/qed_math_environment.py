# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
QED Math Environment Implementation.

A math proof environment that presents problems to agents and evaluates
submitted proofs using LLM-based rubric grading (0-7 scale).
"""

from uuid import uuid4
from typing import Any, Optional

from fastmcp import FastMCP

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State


class QEDMathEnvironment(MCPEnvironment):
    """
    Math proof environment with MCP tools and rubric-based grading.

    This environment provides MCP tools for:
    - get_problem(): Return current problem statement, reference solution,
      and grading guidelines.
    - submit_proof(proof): Grade a proof via MathProofRubric and return
      score (0-7) and normalized reward.
    - get_grading_guidelines(): Return the rubric for the current problem.

    The full implementation includes:
    - Dataset loading from QED-Nano format
    - LLM-based proof grading via rubric
    - Reward normalization to [0, 1]

    Reference:
        - envs/echo_env/server/echo_environment.py - MCP environment pattern
        - QED-Nano: training/pipelinerl/domains/math/rollouts.py - Rollout logic
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        grader_model: str = "gemini-2.0-flash",
        prompt_name: str = "v2",
    ):
        """
        Initialize the QED Math environment.

        Args:
            dataset_path: Optional path to problems dataset.
            grader_model: LLM model to use for grading (default: gemini-2.0-flash).
            prompt_name: Grading prompt version (default: v2).
        """
        mcp = FastMCP("qed_math_env")

        @mcp.tool
        def get_problem() -> dict:
            """
            Get the current problem statement.

            Returns:
                dict with keys: problem, reference_solution, grading_guidelines,
                problem_id, dataset_source.
            """
            raise NotImplementedError("get_problem will be implemented in Phase 2")

        @mcp.tool
        def submit_proof(proof: str) -> dict:
            """
            Submit a proof for grading.

            Args:
                proof: The proof text submitted by the agent.

            Returns:
                dict with keys: score (0-7), feedback, reward (0.0-1.0), done.
            """
            raise NotImplementedError("submit_proof will be implemented in Phase 2")

        @mcp.tool
        def get_grading_guidelines() -> dict:
            """
            Get the grading rubric for the current problem.

            Returns:
                dict with key: grading_guidelines (Markdown rubric string).
            """
            raise NotImplementedError(
                "get_grading_guidelines will be implemented in Phase 2"
            )

        super().__init__(mcp)

        self._dataset_path = dataset_path
        self._grader_model = grader_model
        self._prompt_name = prompt_name
        self._problems: list[dict] = []
        self._current_problem: dict | None = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        problem_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment and load a new problem.

        Args:
            seed: Optional random seed for problem selection.
            problem_id: Optional specific problem ID to load.
            **kwargs: Additional reset parameters.

        Returns:
            ProblemObservation with problem details.

        Note:
            Full implementation in Phase 2 will load problems from dataset
            and return ProblemObservation with problem, reference_solution,
            grading_guidelines, problem_id, and dataset_source.
        """
        self._state = State(
            episode_id=str(uuid4()),
            step_count=0,
        )
        self._reset_count += 1

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "message": "QED Math environment ready. Load a problem to begin.",
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions.

        Args:
            action: The action to execute.
            timeout_s: Optional timeout.
            **kwargs: Additional arguments.

        Returns:
            Observation with error message.
        """
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use MCP tools (CallToolAction) for interactions."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step in the environment.

        Args:
            action: The action to execute.
            timeout_s: Optional timeout for the action.
            **kwargs: Additional arguments.

        Returns:
            Observation from the action execution.
        """
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
