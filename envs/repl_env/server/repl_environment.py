# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
REPL Environment Implementation.

A Python REPL environment for training language models on code execution tasks,
based on the Recursive Language Models (RLM) paradigm.

References:
- RLM Paper: https://arxiv.org/abs/2512.24601
- Prime Intellect Blog: https://www.primeintellect.ai/blog/rlm
- Alex Zhang Blog: https://alexzhang13.github.io/blog/2025/rlm/
"""

import os
import re
from collections.abc import Callable
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import EnvironmentMetadata
except ImportError:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import REPLAction, REPLObservation, REPLState, CodeBlockResult
except ImportError:
    from models import REPLAction, REPLObservation, REPLState, CodeBlockResult

try:
    from .python_executor import PythonExecutor
except ImportError:
    from python_executor import PythonExecutor


class REPLEnvironment(Environment):
    """
    A REPL environment for training language models to use code execution.

    Based on the Recursive Language Models (RLM) paradigm, this environment allows
    language models to:
    - Execute Python code in a sandboxed REPL
    - Work with large contexts loaded as variables
    - Finalize answers via FINAL(), FINAL_VAR(), or answer dict pattern
    - Optionally make recursive LLM calls via llm_query() / llm_query_batched()

    Supports two finalization patterns:
    1. RLM-style: print('FINAL(answer)') or print('FINAL_VAR(var_name)')
    2. Prime Intellect style: answer = {"content": "...", "ready": True}

    Example:
        >>> env = REPLEnvironment(context="Hello World", task_prompt="Count chars")
        >>> obs = env.reset()
        >>> print(obs.context_preview)  # "Hello World"
        >>>
        >>> obs = env.step(REPLAction(code="result = len(context)"))
        >>> print(obs.result.success)  # True
        >>> print(obs.available_variables)  # ["context", "result", "answer"]
        >>>
        >>> obs = env.step(REPLAction(code="print(f'FINAL({result})')"))
        >>> print(obs.done)  # True
        >>> print(obs.metadata["final_answer"])  # "11"
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        context: Optional[str] = None,
        task_prompt: Optional[str] = None,
        max_iterations: int = 30,
        max_output_length: int = 8192,
        context_preview_length: int = 500,
        reward_on_success: float = 1.0,
        reward_on_iteration: float = 0.0,
        reward_on_failure: float = -0.1,
        reward_on_error: float = -0.05,
        llm_query_fn: Optional[Callable[[str], str]] = None,
        llm_batch_fn: Optional[Callable[[List[str]], List[str]]] = None,
    ):
        """Initialize the REPL environment.

        Args:
            context: Initial context to load (can also be set via REPL_CONTEXT env var)
            task_prompt: Task description (can also be set via REPL_TASK_PROMPT env var)
            max_iterations: Maximum steps per episode (default 30, env var REPL_MAX_ITERATIONS)
            max_output_length: Max chars for stdout/stderr per turn (default 8192)
            context_preview_length: Chars to show in context preview (default 500)
            reward_on_success: Reward when final answer is submitted (default 1.0)
            reward_on_iteration: Reward per iteration step (default 0.0)
            reward_on_failure: Reward when max iterations reached (default -0.1)
            reward_on_error: Reward when code execution fails (default -0.05)
            llm_query_fn: Optional function for llm_query() support
            llm_batch_fn: Optional function for llm_query_batched() support
        """
        self.initial_context = context or os.environ.get("REPL_CONTEXT", "")
        self.initial_task_prompt = task_prompt or os.environ.get(
            "REPL_TASK_PROMPT", ""
        )
        self.max_iterations = int(
            os.environ.get("REPL_MAX_ITERATIONS", max_iterations)
        )
        self.max_output_length = max_output_length
        self.context_preview_length = context_preview_length

        # Reward configuration
        self.reward_on_success = reward_on_success
        self.reward_on_iteration = reward_on_iteration
        self.reward_on_failure = reward_on_failure
        self.reward_on_error = reward_on_error

        # Optional LLM functions for recursive calls
        self.llm_query_fn = llm_query_fn
        self.llm_batch_fn = llm_batch_fn

        # State (initialized on reset)
        self._state: Optional[REPLState] = None
        self._executor: Optional[PythonExecutor] = None

    def _create_llm_functions(
        self,
        hf_token: str,
        llm_model: Optional[str] = None,
    ) -> None:
        """Create LLM functions dynamically using client-provided token.

        This allows clients to use their own HF token instead of the server's.

        Security: The token is used only to initialize the InferenceClient
        and is NOT stored in state, logged, or persisted anywhere.

        Args:
            hf_token: HuggingFace API token (not logged or persisted)
            llm_model: Model to use (default: Qwen/Qwen3-Coder-480B-A35B-Instruct)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            # huggingface_hub not installed, skip LLM functions
            return

        model = llm_model or os.environ.get(
            "LLM_MODEL", "Qwen/Qwen3-Coder-480B-A35B-Instruct"
        )
        client = InferenceClient(model=model, token=hf_token)

        def llm_query(prompt: str) -> str:
            """Query the LLM with a prompt and return the response."""
            try:
                messages = [{"role": "user", "content": prompt}]
                response = client.chat_completion(
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.7,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                return f"Error calling LLM: {e}"

        def llm_query_batched(prompts: List[str]) -> List[str]:
            """Query the LLM with multiple prompts in parallel."""
            if not prompts:
                return []

            max_workers = min(len(prompts), 8)
            results: List[str] = [""] * len(prompts)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(llm_query, prompt): idx
                    for idx, prompt in enumerate(prompts)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = f"Error: {e}"

            return results

        self.llm_query_fn = llm_query
        self.llm_batch_fn = llm_query_batched

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        context: Optional[str] = None,
        task_prompt: Optional[str] = None,
        hf_token: Optional[str] = None,
        llm_model: Optional[str] = None,
        **kwargs: Any,
    ) -> REPLObservation:
        """Reset the environment with optional new context.

        Args:
            seed: Optional random seed (for reproducibility)
            episode_id: Optional episode identifier (if not provided, one is generated)
            context: Context to load (overrides initial_context)
            task_prompt: Task description (overrides initial_task_prompt)
            hf_token: Optional HuggingFace token for llm_query/llm_query_batched.
                      If provided, creates LLM functions using this token.
                      Security: Token is NOT stored in state or logged.
            llm_model: Optional model name for LLM functions (default: from env or Qwen3-Coder)
            **kwargs: Additional reset parameters

        Returns:
            Initial REPLObservation with environment ready message
        """
        effective_context = context or self.initial_context
        effective_task_prompt = task_prompt or self.initial_task_prompt

        # Create LLM functions if not already provided at init
        # Priority: client hf_token > server HF_TOKEN env var
        if not self.llm_query_fn:
            effective_token = hf_token or os.environ.get("HF_TOKEN")
            if effective_token:
                self._create_llm_functions(effective_token, llm_model)

        # Initialize state
        self._state = REPLState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            context=effective_context,
            task_prompt=effective_task_prompt,
            iteration=0,
            max_iterations=self.max_iterations,
            namespace_keys=[],
            final_answer=None,
            total_execution_time=0.0,
        )

        # Initialize executor
        self._executor = PythonExecutor(
            max_output_length=self.max_output_length
        )

        # Initialize answer dict (Prime Intellect style)
        self._executor.set_variable("answer", {"content": "", "ready": False})

        # Load context into namespace if provided
        if effective_context:
            self._executor.set_context(effective_context)

        # Inject LLM functions if provided
        # Names: llm_query (single), llm_query_batched (official RLM), llm_batch (alias)
        if self.llm_query_fn:
            self._executor.inject_function("llm_query", self.llm_query_fn)
        if self.llm_batch_fn:
            self._executor.inject_function(
                "llm_query_batched", self.llm_batch_fn
            )  # Official name
            self._executor.inject_function(
                "llm_batch", self.llm_batch_fn
            )  # Alias

        # Inject FINAL helper function so both FINAL(x) and print(f'FINAL({x})') work
        # Returns the FINAL pattern as a string so it appears in stdout for detection
        def final_helper(value):
            """Helper that returns FINAL(value) string for detection."""
            return f"FINAL({value})"

        self._executor.inject_function("FINAL", final_helper)

        # Inject FINAL_VAR helper that looks up variable and returns FINAL(value)
        # This matches official RLM behavior - strips quotes from var_name and looks up in namespace
        executor = self._executor  # Capture for closure

        def final_var_helper(var_name: str):
            """Look up variable by name and return FINAL(value) for detection."""
            # Strip quotes if present (handles both FINAL_VAR("x") and FINAL_VAR(x))
            var_name_clean = str(var_name).strip().strip("\"'")
            # Look up variable in executor namespace
            value = executor.get_variable(var_name_clean)
            if value is not None:
                return f"FINAL({value})"
            return (
                f"FINAL_VAR({var_name_clean})"  # Fallback for regex detection
            )

        self._executor.inject_function("FINAL_VAR", final_var_helper)

        # Update namespace keys
        self._state.namespace_keys = self._executor.list_variables()

        # Build initial message
        message_parts = ["REPL environment initialized."]
        if effective_context:
            message_parts.append(
                f"Context loaded ({len(effective_context)} chars). Use 'context' variable to access it."
            )
        if effective_task_prompt:
            message_parts.append(f"Task: {effective_task_prompt}")
        message_parts.append(
            "Use answer['content'] to store your answer, and set answer['ready'] = True when done."
        )

        return REPLObservation(
            result=CodeBlockResult(
                stdout="\n".join(message_parts),
                stderr="",
                locals_snapshot={},
                execution_time=0.0,
                success=True,
                exception=None,
            ),
            context_preview=(
                effective_context[: self.context_preview_length]
                if effective_context
                else None
            ),
            context_length=len(effective_context) if effective_context else 0,
            available_variables=self._state.namespace_keys,
            iteration=0,
            max_iterations=self.max_iterations,
            done=False,
            reward=0.0,
            metadata={
                "task_prompt": effective_task_prompt,
                "message": "Environment ready.",
            },
        )

    def step(
        self,
        action: REPLAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> REPLObservation:
        """Execute code and return observation.

        Args:
            action: REPLAction containing code to execute
            timeout_s: Optional timeout in seconds (not currently used)
            **kwargs: Additional step parameters

        Returns:
            REPLObservation with execution results
        """
        if self._state is None or self._executor is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() first."
            )

        self._state.step_count += 1
        self._state.iteration += 1

        # Check if agent explicitly signals final answer
        if action.is_final:
            self._state.final_answer = action.final_answer or ""
            return self._create_final_observation(
                success=True,
                message="Final answer submitted.",
                reward=self.reward_on_success,
            )

        # Check iteration limit
        if self._state.iteration >= self.max_iterations:
            # Check if there's a partial answer in the answer dict
            answer_var = self._executor.get_variable("answer")
            if isinstance(answer_var, dict) and answer_var.get("content"):
                self._state.final_answer = str(answer_var.get("content", ""))
            return self._create_final_observation(
                success=False,
                message=f"Maximum iterations ({self.max_iterations}) reached.",
                reward=self.reward_on_failure,
            )

        # Execute code
        result = self._executor.execute(action.code)
        self._state.total_execution_time += result["execution_time"]
        self._state.namespace_keys = self._executor.list_variables()

        # Calculate reward
        reward = self.reward_on_iteration
        if not result["success"]:
            reward += self.reward_on_error

        # Check for final answer patterns
        final_answer = self._extract_final_answer(result["stdout"])
        done = final_answer is not None

        if done:
            self._state.final_answer = final_answer
            reward = self.reward_on_success

        return REPLObservation(
            result=CodeBlockResult(
                stdout=result["stdout"],
                stderr=result["stderr"],
                locals_snapshot=result["locals_snapshot"],
                execution_time=result["execution_time"],
                success=result["success"],
                exception=result["exception"],
            ),
            context_preview=(
                self._state.context[: self.context_preview_length]
                if self._state.context
                else None
            ),
            context_length=len(self._state.context)
            if self._state.context
            else 0,
            available_variables=self._state.namespace_keys,
            iteration=self._state.iteration,
            max_iterations=self.max_iterations,
            done=done,
            reward=reward,
            metadata={
                "task_prompt": self._state.task_prompt,
                "final_answer": final_answer,
                "execution_time": result["execution_time"],
            },
        )

    def _extract_final_answer(self, stdout: str) -> Optional[str]:
        """Extract final answer from output.

        Supports multiple patterns:
        1. RLM-style: FINAL(answer) in stdout
        2. RLM-style: FINAL_VAR(variable_name) in stdout
        3. Prime Intellect style: answer = {"content": "...", "ready": True} in namespace

        Args:
            stdout: Standard output from code execution

        Returns:
            Final answer string or None if not found
        """
        # Pattern 1: RLM-style FINAL(answer)
        final_match = re.search(r"FINAL\((.*?)\)", stdout, re.DOTALL)
        if final_match:
            return final_match.group(1).strip()

        # Pattern 2: RLM-style FINAL_VAR(variable_name)
        final_var_match = re.search(r"FINAL_VAR\((\w+)\)", stdout)
        if final_var_match and self._executor:
            var_name = final_var_match.group(1)
            value = self._executor.get_variable(var_name)
            if value is not None:
                return str(value)

        # Pattern 3: Prime Intellect style answer dict
        if self._executor:
            answer_var = self._executor.get_variable("answer")
            if isinstance(answer_var, dict):
                if answer_var.get("ready", False):
                    return str(answer_var.get("content", ""))

        return None

    def _create_final_observation(
        self, success: bool, message: str, reward: float
    ) -> REPLObservation:
        """Create observation for episode termination.

        Args:
            success: Whether the episode ended successfully
            message: Termination message
            reward: Final reward value

        Returns:
            Final REPLObservation with done=True
        """
        return REPLObservation(
            result=CodeBlockResult(
                stdout=message,
                stderr="",
                locals_snapshot={},
                execution_time=0.0,
                success=success,
                exception=None,
            ),
            context_preview=None,
            context_length=0,
            available_variables=[],
            iteration=self._state.iteration if self._state else 0,
            max_iterations=self.max_iterations,
            done=True,
            reward=reward,
            metadata={
                "final_answer": self._state.final_answer
                if self._state
                else None,
                "total_execution_time": (
                    self._state.total_execution_time if self._state else 0
                ),
                "total_iterations": self._state.iteration if self._state else 0,
            },
        )

    @property
    def state(self) -> REPLState:
        """Get the current environment state.

        Returns:
            Current REPLState

        Raises:
            RuntimeError: If environment not initialized
        """
        if self._state is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() first."
            )
        return self._state

    def close(self) -> None:
        """Cleanup resources."""
        self._executor = None
        self._state = None

    def get_metadata(self) -> EnvironmentMetadata:
        """Get environment metadata.

        Returns:
            EnvironmentMetadata with environment info
        """
        return EnvironmentMetadata(
            name="repl_env",
            description="Python REPL environment for RLM-style code execution",
            version="0.1.0",
        )
