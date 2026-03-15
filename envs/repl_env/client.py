# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
REPL Environment clients.

`REPLEnv` is the standard async OpenEnv client for remote/server-backed usage.
Use `async with` / `await` directly, or call `.sync()` for synchronous code.

`LocalREPLEnv` is an explicit in-process helper for local experiments, tests,
and notebook workflows where starting a server would be unnecessary.

This separation matches current OpenEnv client conventions while preserving the
RLM-style local REPL workflow used by this package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import CodeBlockResult, REPLAction, REPLObservation, REPLState
except ImportError:
    from models import CodeBlockResult, REPLAction, REPLObservation, REPLState
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient


class REPLEnv(EnvClient[REPLAction, REPLObservation, REPLState]):
    """
    Async client for the remote REPL environment.

    Use this client when connecting to a running OpenEnv server over WebSocket.
    For synchronous code, call `.sync()` on an instance.

    Example:
        >>> async with REPLEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(context="Hello World", task_prompt="Count chars")
        ...     result = await env.execute("count = len(context)")
        ...     result = await env.execute("print(f'FINAL({count})')")
        ...     print(result.done)

        >>> with REPLEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(context="Hello World", task_prompt="Count chars")
        ...     result = env.execute("count = len(context)")
        ...     result = env.execute("print(f'FINAL({count})')")
        ...     print(result.done)
    """

    def _step_payload(self, action: REPLAction) -> dict[str, Any]:
        return {
            "code": action.code,
            "is_final": action.is_final,
            "final_answer": action.final_answer,
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[REPLObservation]:
        obs_data = payload.get("observation", {})
        result_data = obs_data.get("result", {})

        observation = REPLObservation(
            result=CodeBlockResult(
                stdout=result_data.get("stdout", ""),
                stderr=result_data.get("stderr", ""),
                locals_snapshot=result_data.get("locals_snapshot", {}),
                execution_time=result_data.get("execution_time", 0.0),
                success=result_data.get("success", True),
                exception=result_data.get("exception"),
            ),
            context_preview=obs_data.get("context_preview"),
            context_length=obs_data.get("context_length", 0),
            available_variables=obs_data.get("available_variables", []),
            iteration=obs_data.get("iteration", 0),
            max_iterations=obs_data.get("max_iterations", 30),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> REPLState:
        return REPLState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            context=payload.get("context"),
            task_prompt=payload.get("task_prompt"),
            iteration=payload.get("iteration", 0),
            max_iterations=payload.get("max_iterations", 30),
            namespace_keys=payload.get("namespace_keys", []),
            final_answer=payload.get("final_answer"),
            total_execution_time=payload.get("total_execution_time", 0.0),
        )

    async def execute(self, code: str) -> StepResult[REPLObservation]:
        """Execute Python code in the REPL."""
        return await self.step(REPLAction(code=code))

    async def submit_final_answer(self, answer: str) -> StepResult[REPLObservation]:
        """Submit a final answer and terminate the episode."""
        return await self.step(REPLAction(code="", is_final=True, final_answer=answer))

    async def get_variable(self, name: str) -> StepResult[REPLObservation]:
        """Retrieve and print a variable from the REPL namespace."""
        return await self.execute(f"print(repr({name}))")

    async def list_variables(self) -> list[str]:
        """Return the current REPL namespace keys."""
        return (await self.state()).namespace_keys


class LocalREPLEnv:
    """
    Explicit in-process REPL helper for local experimentation.

    This helper preserves the prior local `repl_env` workflow but keeps that
    behavior separate from the standard remote `EnvClient` API.
    """

    def __init__(
        self,
        *,
        llm_query_fn: Optional[Callable[[str], str]] = None,
        llm_batch_fn: Optional[Callable[[list[str]], list[str]]] = None,
        max_output_length: int = 8192,
        context_preview_length: int = 500,
        reward_on_success: float = 1.0,
        reward_on_iteration: float = 0.0,
        reward_on_failure: float = -0.1,
        reward_on_error: float = -0.05,
    ):
        from .server.repl_environment import REPLEnvironment

        self._env = REPLEnvironment(
            max_output_length=max_output_length,
            context_preview_length=context_preview_length,
            reward_on_success=reward_on_success,
            reward_on_iteration=reward_on_iteration,
            reward_on_failure=reward_on_failure,
            reward_on_error=reward_on_error,
            llm_query_fn=llm_query_fn,
            llm_batch_fn=llm_batch_fn,
        )

    def reset(
        self,
        *,
        context: str = "",
        task_prompt: str = "",
        max_iterations: int = 30,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> StepResult[REPLObservation]:
        self._env.max_iterations = max_iterations
        obs = self._env.reset(
            seed=seed,
            episode_id=episode_id,
            context=context,
            task_prompt=task_prompt,
            hf_token=hf_token,
            llm_model=llm_model,
        )
        return self._wrap_observation(obs)

    def step(self, action: REPLAction) -> StepResult[REPLObservation]:
        return self._wrap_observation(self._env.step(action))

    def execute(self, code: str) -> StepResult[REPLObservation]:
        return self.step(REPLAction(code=code))

    def submit_final_answer(self, answer: str) -> StepResult[REPLObservation]:
        return self.step(REPLAction(code="", is_final=True, final_answer=answer))

    def get_variable(self, name: str) -> StepResult[REPLObservation]:
        return self.execute(f"print(repr({name}))")

    def state(self) -> REPLState:
        return self._env.state

    def list_variables(self) -> list[str]:
        return self.state().namespace_keys

    def close(self) -> None:
        self._env.close()

    def __enter__(self) -> "LocalREPLEnv":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def _wrap_observation(obs: REPLObservation) -> StepResult[REPLObservation]:
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )
