"""
Task Evaluator for generated tasks.

Given a generated (prompt, verifier_code) and environment config, runs k rollouts
across m models via the Fleet harness (POST /v1/jobs) and returns structured
results for reward computation.

This is the inner loop of the task generation RL pipeline:
    1. Task generator outputs (prompt, verifier) for an environment
    2. TaskEvaluator imports the task to Fleet, creates a harness job
    3. Harness runs k × m rollouts (env provisioning, model calls, verification)
    4. Results feed into reward computation (variance + separation)
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default models for evaluation (must match Fleet models table IDs)
DEFAULT_MODELS = ["claude-sonnet-4.5"]


@dataclass
class EvaluationResult:
    """Aggregated results from k × m rollout evaluation."""

    results_per_model: Dict[str, List[float]] = field(default_factory=dict)
    total_duration_s: float = 0.0
    num_sessions: int = 0
    num_errors: int = 0
    job_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results_per_model": self.results_per_model,
            "total_duration_s": self.total_duration_s,
            "num_rollouts": self.num_sessions,
            "num_errors": self.num_errors,
            "job_id": self.job_id,
        }


class TaskEvaluator:
    """Evaluates generated tasks by submitting jobs to the Fleet harness.

    For each generated task:
    1. Imports the task to Fleet via fleet.import_task()
    2. Creates a harness job via fleet.create_job() with specified models and pass_k
    3. Polls for job completion
    4. Extracts per-session verifier scores
    5. Returns results_per_model for reward computation

    Args:
        api_key: Fleet API key
        k_rollouts: Number of rollouts per model (pass_k in Fleet terms)
        models: List of Fleet model IDs (e.g., ["claude-sonnet-4.5", "claude-opus-4.5"])
        max_steps: Maximum agent steps per session
        poll_interval_s: Seconds between job status polls (default: 10)
        max_poll_time_s: Maximum time to wait for job completion (default: 1800 = 30 min)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        k_rollouts: int = 4,
        models: Optional[List[str]] = None,
        max_steps: int = 30,
        poll_interval_s: int = 10,
        max_poll_time_s: int = 1800,
        **kwargs,
    ):
        self.api_key = api_key or os.environ.get("FLEET_API_KEY")
        if not self.api_key:
            raise ValueError("Fleet API key required")

        self.k_rollouts = k_rollouts
        self.models = models or DEFAULT_MODELS
        self.max_steps = max_steps
        self.poll_interval_s = poll_interval_s
        self.max_poll_time_s = max_poll_time_s

        # Initialize Fleet SDK client
        self._fleet_client = None

    def _get_fleet_client(self):
        """Lazy-init Fleet SDK client."""
        if self._fleet_client is None:
            from fleet import Fleet

            self._fleet_client = Fleet(api_key=self.api_key)
        return self._fleet_client

    async def evaluate(
        self,
        prompt: str,
        verifier_code: str,
        env_key: str,
        env_version: str = "",
        env_variables: Optional[Dict[str, Any]] = None,
        data_key: Optional[str] = None,
        data_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run k × m rollouts via Fleet harness and return structured results.

        Flow:
        1. Create a Fleet Task object with the generated prompt + verifier
        2. Import it to Fleet via POST /v1/tasks
        3. Create a harness job via POST /v1/jobs
        4. Poll until job completes
        5. Extract per-model, per-session verifier scores

        Args:
            prompt: The generated task prompt
            verifier_code: The generated verifier code
            env_key: Fleet environment key
            env_version: Fleet environment version
            env_variables: Optional environment variables
            data_key: Optional data key
            data_version: Optional data version

        Returns:
            Dict with 'results_per_model' mapping model_id -> list[float]
        """
        start_time = time.time()
        result = EvaluationResult()
        for model_id in self.models:
            result.results_per_model[model_id] = []

        fleet = self._get_fleet_client()
        task_key = f"taskgen_{uuid.uuid4().hex[:12]}"

        try:
            # 1. Create Fleet Task object
            from fleet.tasks import Task

            task = Task(
                key=task_key,
                prompt=prompt,
                env_id=env_key,
                version=env_version or None,
                verifier_func=verifier_code,
                data_id=data_key,
                data_version=data_version,
                env_variables=env_variables or {},
            )

            # 2. Import task to Fleet
            import_response = fleet.import_single_task(task)
            if import_response is None:
                logger.error(f"[{task_key}] Failed to import task to Fleet")
                result.num_errors = 1
                result.total_duration_s = time.time() - start_time
                return result.to_dict()

            logger.info(f"[{task_key}] Task imported to Fleet")

            # 3. Create harness job
            job_response = fleet.create_job(
                models=self.models,
                task_keys=[task_key],
                pass_k=self.k_rollouts,
                max_steps=self.max_steps,
                mode="tool-use",
                name=f"taskgen-eval-{task_key}",
            )
            job_id = job_response.job_id
            result.job_id = job_id
            logger.info(
                f"[{task_key}] Harness job created: {job_id} "
                f"(models={self.models}, pass_k={self.k_rollouts})"
            )

            # 4. Poll for job completion
            job_status = self._poll_job(fleet, job_id)
            if job_status not in ("completed",):
                logger.warning(
                    f"[{task_key}] Job {job_id} ended with status: {job_status}"
                )
                result.num_errors = 1
                result.total_duration_s = time.time() - start_time
                return result.to_dict()

            # 5. Extract per-session scores
            sessions_response = fleet.list_job_sessions(job_id)
            for task_group in sessions_response.tasks:
                for session in task_group.sessions:
                    model_id = session.model
                    score = 0.0
                    if session.verifier_execution and session.verifier_execution.score is not None:
                        score = float(session.verifier_execution.score)
                    elif session.verifier_execution and session.verifier_execution.success:
                        score = 1.0

                    if model_id in result.results_per_model:
                        result.results_per_model[model_id].append(score)
                    else:
                        result.results_per_model[model_id] = [score]

                    result.num_sessions += 1

            logger.info(
                f"[{task_key}] Evaluation complete: "
                f"{result.num_sessions} sessions across {len(self.models)} models. "
                f"Results: {{{', '.join(f'{m}: {scores}' for m, scores in result.results_per_model.items())}}}"
            )

        except Exception as e:
            logger.error(f"[{task_key}] Evaluation failed: {e}")
            result.num_errors += 1

        result.total_duration_s = time.time() - start_time
        return result.to_dict()

    def _poll_job(self, fleet, job_id: str) -> str:
        """Poll Fleet job until completion or timeout.

        Returns:
            Final job status string.
        """
        start = time.time()
        while time.time() - start < self.max_poll_time_s:
            try:
                job = fleet.get_job(job_id)
                status = job.status
                if status in ("completed", "cancelled", "errored"):
                    return status
            except Exception as e:
                logger.warning(f"Error polling job {job_id}: {e}")

            time.sleep(self.poll_interval_s)

        logger.error(f"Job {job_id} timed out after {self.max_poll_time_s}s")
        return "timeout"


async def evaluate_task(
    prompt: str,
    verifier_code: str,
    env_key: str,
    env_version: str = "",
    api_key: Optional[str] = None,
    k_rollouts: int = 4,
    models: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convenience function for one-off task evaluation.

    Args:
        prompt: Task prompt to evaluate
        verifier_code: Verifier code for the task
        env_key: Fleet environment key
        env_version: Fleet environment version
        api_key: Fleet API key
        k_rollouts: Number of rollouts per model
        models: List of Fleet model IDs

    Returns:
        Evaluation results dict
    """
    evaluator = TaskEvaluator(
        api_key=api_key,
        k_rollouts=k_rollouts,
        models=models,
    )
    return await evaluator.evaluate(
        prompt=prompt,
        verifier_code=verifier_code,
        env_key=env_key,
        env_version=env_version,
    )
