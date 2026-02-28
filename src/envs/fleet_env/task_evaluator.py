"""
Task Evaluator for generated tasks.

Given a generated (prompt, verifier_code) and environment config, runs k rollouts
across m models and returns structured results for reward computation.

This is the inner loop of the task generation RL pipeline:
    1. Task generator outputs (prompt, verifier) for an environment
    2. TaskEvaluator runs k × m rollouts on Fleet infrastructure
    3. Results feed into reward computation (variance + separation)

Uses OpenEnv's FleetTaskEnv for environment management and rollout execution.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .task_env import FleetTaskEnv

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    """Result from a single agent rollout."""

    model_id: str
    reward: float
    steps: int
    done_reason: str  # "agent_done", "max_steps", "error"
    duration_s: float
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """Aggregated results from k × m rollout evaluation."""

    results_per_model: Dict[str, List[float]] = field(default_factory=dict)
    rollouts: List[RolloutResult] = field(default_factory=list)
    total_duration_s: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results_per_model": self.results_per_model,
            "total_duration_s": self.total_duration_s,
            "num_rollouts": len(self.rollouts),
            "num_errors": len(self.errors),
        }


class TaskEvaluator:
    """Evaluates generated tasks by running k × m rollouts on Fleet.

    For each generated task, creates Fleet environment instances and runs
    agent rollouts using specified models. Collects pass/fail per rollout
    for reward computation.

    Args:
        api_key: Fleet API key
        k_rollouts: Number of rollouts per model (default: 4)
        models: List of model IDs to evaluate with (default: ["weak"])
        max_steps: Maximum steps per rollout (default: 30)
        ttl_seconds: TTL for Fleet instances (default: 300)
        max_concurrent: Maximum concurrent rollouts (default: 4)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        k_rollouts: int = 4,
        models: Optional[List[str]] = None,
        max_steps: int = 30,
        ttl_seconds: int = 300,
        max_concurrent: int = 4,
    ):
        self.api_key = api_key or os.environ.get("FLEET_API_KEY")
        if not self.api_key:
            raise ValueError("Fleet API key required")

        self.k_rollouts = k_rollouts
        self.models = models or ["weak"]
        self.max_steps = max_steps
        self.ttl_seconds = ttl_seconds
        self.max_concurrent = max_concurrent

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
        """Run k × m rollouts and return structured results.

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

        # Build task config for FleetTaskEnv
        task_config = {
            "task_key": f"taskgen_{int(time.time())}",
            "prompt": prompt,
            "env_key": env_key,
            "env_version": env_version,
            "verifier_code": verifier_code,
            "task_modality": "tool_use",
        }
        if env_variables:
            task_config["env_variables"] = env_variables
        if data_key:
            task_config["data_key"] = data_key
        if data_version:
            task_config["data_version"] = data_version

        # Run rollouts with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = []

        for model_id in self.models:
            result.results_per_model[model_id] = []
            for rollout_idx in range(self.k_rollouts):
                tasks.append(
                    self._run_rollout_with_semaphore(
                        semaphore, task_config, model_id, rollout_idx
                    )
                )

        rollout_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for r in rollout_results:
            if isinstance(r, Exception):
                result.errors.append(str(r))
                logger.error(f"Rollout failed: {r}")
                continue
            if isinstance(r, RolloutResult):
                result.rollouts.append(r)
                result.results_per_model[r.model_id].append(r.reward)

        result.total_duration_s = time.time() - start_time

        logger.info(
            f"Evaluation complete: {len(result.rollouts)} rollouts, "
            f"{len(result.errors)} errors, {result.total_duration_s:.1f}s"
        )

        return result.to_dict()

    async def _run_rollout_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        task_config: Dict[str, Any],
        model_id: str,
        rollout_idx: int,
    ) -> RolloutResult:
        """Run a single rollout with concurrency limiting."""
        async with semaphore:
            return await self._run_rollout(task_config, model_id, rollout_idx)

    async def _run_rollout(
        self,
        task_config: Dict[str, Any],
        model_id: str,
        rollout_idx: int,
    ) -> RolloutResult:
        """Run a single agent rollout on a Fleet environment.

        Creates a FleetTaskEnv instance, runs a simple agent loop
        (prompt → model → tool_call → env → ... → done), and returns
        the final reward from the verifier.

        Args:
            task_config: Task configuration for FleetTaskEnv
            model_id: Model identifier for inference
            rollout_idx: Rollout index (for logging)

        Returns:
            RolloutResult with reward and metadata
        """
        start_time = time.time()
        steps = 0
        done_reason = "unknown"
        error = None
        reward = 0.0

        env = None
        try:
            # Create Fleet environment
            env = FleetTaskEnv(
                task_config=task_config,
                api_key=self.api_key,
                ttl_seconds=self.ttl_seconds,
                max_steps=self.max_steps,
            )

            # Reset environment
            obs = await env.reset_async()
            tools = obs.get("tools", [])

            if not tools:
                return RolloutResult(
                    model_id=model_id,
                    reward=0.0,
                    steps=0,
                    done_reason="no_tools",
                    duration_s=time.time() - start_time,
                    error="No tools available in environment",
                )

            # Simple agent loop
            done = False
            while not done and steps < self.max_steps:
                # Get model response
                agent_response = await self._get_model_response(
                    prompt=task_config["prompt"],
                    tools=tools,
                    model_id=model_id,
                    obs=obs,
                    step=steps,
                )

                # Build action from model response
                action = self._parse_agent_response(agent_response)

                # Step environment
                obs, reward, done, info = await env.step_async(action)
                steps += 1
                done_reason = info.get("done_reason", "continue")

            if not done:
                done_reason = "max_steps"

        except Exception as e:
            error = str(e)
            done_reason = "error"
            logger.warning(
                f"Rollout {model_id}[{rollout_idx}] failed at step {steps}: {e}"
            )

        finally:
            if env:
                try:
                    env.close()
                except Exception:
                    pass

        duration = time.time() - start_time
        return RolloutResult(
            model_id=model_id,
            reward=reward,
            steps=steps,
            done_reason=done_reason,
            duration_s=duration,
            error=error,
        )

    async def _get_model_response(
        self,
        prompt: str,
        tools: List[Dict],
        model_id: str,
        obs: Dict[str, Any],
        step: int,
    ) -> str:
        """Get a response from the agent model.

        Uses Anthropic's API for inference. The model_id maps to
        Claude model variants.

        Args:
            prompt: The task prompt
            tools: Available tools
            model_id: Model identifier ("weak", "strong", or specific model ID)
            obs: Current observation from the environment
            step: Current step number

        Returns:
            Model's text response
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for model inference. "
                "Install with: pip install anthropic"
            )

        # Map model_id to actual Anthropic model
        model_map = {
            "weak": "claude-haiku-4-5-20251001",
            "strong": "claude-sonnet-4-5-20250929",
        }
        actual_model = model_map.get(model_id, model_id)

        # Build messages for the model
        tools_json = json.dumps(tools, indent=2)
        current_date = datetime.now().strftime("%Y-%m-%d")

        system_content = f"""You are a helpful agent. Complete the task by calling tools.

## Current Date
Today's date is {current_date}.

## Available Tools
{tools_json}

## Tool Call Format
<tool_call>{{"name": "tool_name", "arguments": {{"param": "value"}}}}</tool_call>

## Response Format
EVERY response MUST end with exactly ONE of:
1. A tool call: <tool_call>...</tool_call>
2. Done signal: <done> (ONLY when task is fully complete)"""

        messages = [{"role": "user", "content": prompt}]

        # Add observation context if we have previous results
        observation = obs.get("observation", {})
        if step > 0 and observation:
            obs_str = json.dumps(observation) if isinstance(observation, dict) else str(observation)
            messages.append({"role": "assistant", "content": "(previous action)"})
            messages.append({"role": "user", "content": f"Tool result:\n{obs_str}"})

        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=actual_model,
            max_tokens=2048,
            system=system_content,
            messages=messages,
        )

        return response.content[0].text if response.content else "<done>"

    def _parse_agent_response(self, response: str) -> Dict[str, Any]:
        """Parse agent response into action dict for FleetTaskEnv.

        Extracts tool calls or done signals from the model's response.

        Args:
            response: Model's text response

        Returns:
            Action dict with 'tool', 'params', 'done' keys
        """
        import re

        # Check for done signal
        agent_done = "<done>" in response.lower()

        # Try to extract tool call
        tool_call = None
        for tag in ["tool_call", "function_call"]:
            match = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
            if not match:
                match = re.search(rf"<{tag}>(.*?)(?:<\||\Z)", response, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(1).strip())
                    if isinstance(parsed, dict):
                        name = parsed.get("name") or parsed.get("tool")
                        args = parsed.get("arguments") or parsed.get("params", {})
                        if name:
                            tool_call = {"name": name, "arguments": args}
                            break
                except (json.JSONDecodeError, ValueError):
                    pass

        action = {"done": agent_done}
        if tool_call:
            action["tool"] = tool_call["name"]
            action["params"] = tool_call.get("arguments", {})

        return action


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
        models: List of model IDs

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
