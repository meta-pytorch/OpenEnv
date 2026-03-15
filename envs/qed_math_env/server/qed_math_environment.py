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

import json
from dataclasses import dataclass
from pathlib import Path
import random
from uuid import uuid4
from typing import Any, Optional

from fastmcp import FastMCP

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from openenv.core.env_server.types import Action, Observation, State

try:
    from ..models import ProblemObservation, ProofSubmissionObservation
except ImportError:
    from models import ProblemObservation, ProofSubmissionObservation

from .mcp_server import register_mcp_tools
from .rubric import GradingResult, MathProofRubric, parse_schema


DEFAULT_EVALUATOR_PROMPT = (
    "You are a strict math proof grader. Score the submission from 0 to 7 based on "
    "mathematical correctness, completeness, and logical rigor."
)


@dataclass
class QEDMathConfig:
    """Configuration for QEDMathEnvironment rubric and dataset behavior."""

    dataset_path: str | None = None
    grader_model: str = "gemini-3-pro"
    prompt_name: str = "v2"
    custom_reward_threshold: bool = False


def load_evaluator_prompt(prompt_name: str) -> str:
    """Load an evaluator prompt template by name.

    Prompt lookup order:
    1) env local: envs/qed_math_env/prompts/evaluator_prompts/{prompt_name}.md
    2) fallback to default in-code template
    """
    prompt_path = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / "evaluator_prompts"
        / f"{prompt_name}.md"
    )
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return DEFAULT_EVALUATOR_PROMPT


def _normalize_problem(raw_problem: dict, index: int, dataset_source: str) -> dict:
    """Map heterogeneous dataset rows to canonical QED-Math fields."""
    return {
        "problem": raw_problem.get("problem", ""),
        "reference_solution": raw_problem.get("solution", ""),
        "grading_guidelines": raw_problem.get("grading_guidelines")
        or raw_problem.get("rubrics", ""),
        "problem_id": raw_problem.get("problem_id")
        or raw_problem.get("id")
        or f"problem_{index:06d}",
        "dataset_source": raw_problem.get("dataset", dataset_source),
    }


def load_problems(dataset_path: str | None) -> list[dict]:
    """Load problems from local JSON/JSONL or return a bootstrap sample.

    Supported local formats:
    - JSONL: one problem object per line
    - JSON: list[object] or {"problems": list[object]}
    """
    if not dataset_path:
        return [
            {
                "problem": "Prove that the sum of two even integers is even.",
                "reference_solution": "Let a=2m and b=2n. Then a+b=2(m+n), so it is even.",
                "grading_guidelines": "Award full credit for a correct parity argument.",
                "problem_id": "bootstrap_000001",
                "dataset_source": "bootstrap",
            }
        ]

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    rows: list[dict] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
    elif path.suffix.lower() == ".json":
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, list):
            rows = [item for item in parsed if isinstance(item, dict)]
        elif isinstance(parsed, dict) and isinstance(parsed.get("problems"), list):
            rows = [item for item in parsed["problems"] if isinstance(item, dict)]
        else:
            raise ValueError(
                "JSON dataset must be a list of problem objects or contain 'problems'."
            )
    else:
        raise ValueError(
            "Unsupported dataset format. Expected .jsonl or .json."
        )

    dataset_source = path.stem
    return [
        _normalize_problem(problem, idx + 1, dataset_source)
        for idx, problem in enumerate(rows)
    ]


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
        grader_model: str | None = None,
        prompt_name: str | None = None,
        custom_reward_threshold: bool | None = None,
        config: QEDMathConfig | None = None,
    ):
        """
        Initialize the QED Math environment.

        Args:
            dataset_path: Optional path to problems dataset.
            grader_model: LLM model to use for grading.
            prompt_name: Grading prompt version.
            custom_reward_threshold: Apply QED-Nano-style score thresholding.
            config: Optional consolidated configuration object.
        """
        mcp = FastMCP("qed_math_env")

        register_mcp_tools(mcp, self)

        super().__init__(mcp)

        # Prefer explicit constructor args when provided; otherwise use config.
        base_config = config or QEDMathConfig()
        self._config = QEDMathConfig(
            dataset_path=(
                dataset_path if dataset_path is not None else base_config.dataset_path
            ),
            grader_model=(
                grader_model if grader_model is not None else base_config.grader_model
            ),
            prompt_name=(
                prompt_name if prompt_name is not None else base_config.prompt_name
            ),
            custom_reward_threshold=(
                custom_reward_threshold
                if custom_reward_threshold is not None
                else base_config.custom_reward_threshold
            ),
        )

        self._dataset_path = self._config.dataset_path
        self._grader_model = self._config.grader_model
        self._prompt_name = self._config.prompt_name
        self._prompt_template = load_evaluator_prompt(self._config.prompt_name)
        self._problems: list[dict] = load_problems(self._config.dataset_path)
        self._current_problem: dict | None = None
        self._rubric = MathProofRubric(
            grader_model=self._config.grader_model,
            prompt_template=self._prompt_template,
            custom_threshold=self._config.custom_reward_threshold,
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment and load a new problem.

        Args:
            seed: Optional random seed for problem selection.
            episode_id: Optional episode identifier.
            problem_id: Optional specific problem ID to load (via kwargs).
            **kwargs: Additional reset parameters.

        Returns:
            ProblemObservation with problem details.

        Note:
            Full implementation will load problems from dataset
            and return ProblemObservation with problem, reference_solution,
            grading_guidelines, problem_id, and dataset_source.
        """
        selected_problem_id = kwargs.pop("problem_id", None)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._reset_count += 1

        # Reset rubric state if the rubric exposes a reset hook.
        rubric_reset = getattr(self._rubric, "reset", None)
        if callable(rubric_reset):
            rubric_reset()

        if self._problems:
            if selected_problem_id is not None:
                selected = next(
                    (
                        problem
                        for problem in self._problems
                        if problem.get("problem_id") == selected_problem_id
                    ),
                    None,
                )
                self._current_problem = selected or self._problems[0]
            elif seed is not None:
                rng = random.Random(seed)
                self._current_problem = rng.choice(self._problems)
            else:
                self._current_problem = self._problems[0]
        else:
            self._current_problem = None

        if self._current_problem is None:
            return Observation(
                done=False,
                reward=0.0,
                metadata={
                    "error": "No problems loaded. Provide a valid dataset_path.",
                    "status": "empty",
                },
            )

        return ProblemObservation(
            problem=self._current_problem.get("problem", ""),
            reference_solution=self._current_problem.get("reference_solution", ""),
            grading_guidelines=parse_schema(self._current_problem.get("grading_guidelines", "") or ""),
            problem_id=self._current_problem.get("problem_id", ""),
            dataset_source=self._current_problem.get("dataset_source", ""),
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "reset_count": self._reset_count,
                "step_count": self._state.step_count,
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
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # For proof submissions, surface proof-grading semantics
        # on the returned observation while preserving MCP compatibility.
        if (
            isinstance(action, CallToolAction)
            and action.tool_name == "submit_proof"
            and isinstance(obs, CallToolObservation)
            and obs.error is None
        ):
            payload = self._extract_tool_payload(obs.result)
            if isinstance(payload, dict):
                proof_obs = ProofSubmissionObservation(
                    proof=str(payload.get("proof", "")),
                    score=int(payload.get("score", 0)),
                    feedback=str(payload.get("feedback", "")),
                    done=bool(payload.get("done", True)),
                    reward=float(payload.get("reward", 0.0)),
                )
                metadata = dict(obs.metadata)
                metadata["proof_submission"] = proof_obs.model_dump()
                return CallToolObservation(
                    tool_name=obs.tool_name,
                    result=obs.result,
                    error=obs.error,
                    done=proof_obs.done,
                    reward=proof_obs.reward,
                    metadata=metadata,
                )

        return obs

    @staticmethod
    def _extract_tool_payload(tool_result: Any) -> Any:
        """Extract a structured tool payload from FastMCP CallToolResult-like values."""
        if tool_result is None:
            return None

        data = getattr(tool_result, "data", None)
        if data is not None:
            return data

        structured_content = getattr(tool_result, "structured_content", None)
        if structured_content is not None:
            return structured_content

        if isinstance(tool_result, dict):
            return tool_result

        return None

    def get_problem_payload(self) -> dict:
        """Payload for MCP tool get_problem."""
        if self._current_problem is None:
            return {
                "error": "No active problem. Call reset() first.",
                "done": False,
                "reward": 0.0,
            }

        return {
            "problem": self._current_problem.get("problem", ""),
            "reference_solution": self._current_problem.get("reference_solution", ""),
            "grading_guidelines": self._current_problem.get("grading_guidelines", ""),
            "problem_id": self._current_problem.get("problem_id", ""),
            "dataset_source": self._current_problem.get("dataset_source", ""),
            "done": False,
            "reward": 0.0,
        }

    async def submit_proof_payload(self, proof: str) -> dict:
        """Payload for MCP tool submit_proof."""
        if self._current_problem is None:
            return ProofSubmissionObservation(
                proof=proof,
                score=0,
                feedback="Proof not graded because no problem is active.",
                done=True,
                reward=0.0,
                metadata={"error": "No active problem. Call reset() first."},
            ).model_dump()

        if not proof.strip():
            return ProofSubmissionObservation(
                proof=proof,
                score=0,
                feedback="Empty proof submission.",
                done=True,
                reward=0.0,
            ).model_dump()

        problem = self._current_problem.get("problem", "")
        reference_solution = self._current_problem.get("reference_solution", "")
        grading_guidelines = parse_schema(
            self._current_problem.get("grading_guidelines", "") or ""
        )

        result: GradingResult = await self._rubric.grade(
            proof, problem, reference_solution, grading_guidelines
        )

        return ProofSubmissionObservation(
            proof=proof,
            score=result.score,
            feedback=result.feedback,
            done=True,
            reward=result.reward,
        ).model_dump()

    def get_grading_guidelines_payload(self) -> dict:
        """Payload for MCP tool get_grading_guidelines."""
        if self._current_problem is None:
            return {
                "error": "No active problem. Call reset() first.",
                "grading_guidelines": "",
                "done": False,
                "reward": 0.0,
            }

        return {
            "grading_guidelines": self._current_problem.get("grading_guidelines", ""),
            "problem_id": self._current_problem.get("problem_id", ""),
            "done": False,
            "reward": 0.0,
        }

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
