# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Local recursive RLM runner for repl_env.

This keeps the iterative prompting/orchestration layer outside the environment,
following the same separation used by the official RLM implementation and DSPy:
- `REPLEnvironment` executes code and exposes tools
- `LocalRLMRunner` owns prompting, message history, and recursive child runs
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .client import LocalREPLEnv
from .prompts import (
    QueryMetadata,
    RLM_SYSTEM_PROMPT,
    build_rlm_system_prompt,
    build_user_prompt,
    extract_code_blocks,
    format_observations,
)


ChatFn = Callable[..., str]


@dataclass
class RLMRunResult:
    final_answer: str | None
    messages: list[dict[str, str]]
    iterations: int
    depth: int


class LocalRLMRunner:
    """Local recursive RLM orchestrator built on top of LocalREPLEnv."""

    def __init__(
        self,
        llm_chat_fn: ChatFn,
        *,
        system_prompt: str = RLM_SYSTEM_PROMPT,
        max_iterations: int = 30,
        max_depth: int = 2,
        depth: int = 0,
        env_max_iterations_multiplier: int = 5,
        max_batch_workers: int = 8,
        verbose: bool = False,
    ) -> None:
        self.llm_chat_fn = llm_chat_fn
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.depth = depth
        self.env_max_iterations_multiplier = env_max_iterations_multiplier
        self.max_batch_workers = max_batch_workers
        self.verbose = verbose

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        try:
            return self.llm_chat_fn([{"role": "user", "content": prompt}], model)
        except TypeError:
            return self.llm_chat_fn([{"role": "user", "content": prompt}])

    def _llm_query_batched(
        self, prompts: list[str], model: str | None = None
    ) -> list[str]:
        if not prompts:
            return []

        max_workers = min(len(prompts), self.max_batch_workers)
        results: list[str] = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self._llm_query, prompt, model): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = f"Error: {exc}"
        return results

    def _subcall(self, prompt: str, model: str | None = None) -> str:
        next_depth = self.depth + 1
        if next_depth >= self.max_depth:
            return self._llm_query(prompt, model)

        child = LocalRLMRunner(
            self.llm_chat_fn,
            system_prompt=self.system_prompt,
            max_iterations=self.max_iterations,
            max_depth=self.max_depth,
            depth=next_depth,
            env_max_iterations_multiplier=self.env_max_iterations_multiplier,
            max_batch_workers=self.max_batch_workers,
            verbose=self.verbose,
        )
        result = child.run(prompt, prompt, model=model)
        return result.final_answer or ""

    def run(
        self,
        context: str,
        task_prompt: str,
        *,
        model: str | None = None,
    ) -> RLMRunResult:
        with LocalREPLEnv(
            llm_query_fn=self._llm_query,
            llm_batch_fn=self._llm_query_batched,
            subcall_fn=self._subcall,
        ) as env:
            result = env.reset(
                context=context,
                task_prompt=task_prompt,
                max_iterations=self.max_iterations * self.env_max_iterations_multiplier,
                llm_model=model,
            )
            obs = result.observation

            query_metadata = QueryMetadata(
                context_lengths=[obs.context_length],
                context_total_length=obs.context_length,
                context_type="str",
            )
            messages = build_rlm_system_prompt(self.system_prompt, query_metadata)
            messages.append(build_user_prompt(root_prompt=task_prompt, iteration=0))

            for iteration in range(1, self.max_iterations + 1):
                response = self._chat(messages, model)
                code_blocks = extract_code_blocks(response)
                code_block_observations = []

                if self.verbose:
                    print(
                        f"[depth={self.depth}] iteration={iteration} code_blocks={len(code_blocks)}"
                    )

                if not code_blocks:
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Please continue by writing Python code in ```repl``` blocks, "
                                "or submit the final answer with FINAL(...) / FINAL_VAR(...)."
                            ),
                        }
                    )
                    continue

                for code in code_blocks:
                    result = env.execute(code)
                    code_block_observations.append(result.observation)
                    if result.done:
                        return RLMRunResult(
                            final_answer=env.state().final_answer,
                            messages=messages + [{"role": "assistant", "content": response}],
                            iterations=iteration,
                            depth=self.depth,
                        )

                observation_text = format_observations(code_block_observations)
                next_prompt = build_user_prompt(
                    root_prompt=task_prompt,
                    iteration=iteration,
                )
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": observation_text + "\n\n" + next_prompt["content"],
                    }
                )

            return RLMRunResult(
                final_answer=env.state().final_answer,
                messages=messages,
                iterations=self.max_iterations,
                depth=self.depth,
            )

    def _chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        try:
            return self.llm_chat_fn(messages, model)
        except TypeError:
            return self.llm_chat_fn(messages)
