# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
REPL Environment Client.

This module provides a unified client for the REPL Environment that works
with both remote servers (via WebSocket) and local execution (no server needed).

Examples:
    # Connect to remote server with your HF token for sub-LLM calls
    env = REPLEnv(base_url="https://my-server.hf.space")
    result = env.reset(
        context="...",
        task_prompt="...",
        hf_token=os.environ["HF_TOKEN"],  # Server uses this for llm_query
    )

    # Run locally (no server)
    env = REPLEnv()

    # Local with LLM support
    env = REPLEnv(llm_query_fn=my_llm, llm_batch_fn=my_batch)

    # All use the same interface
    result = env.execute("x = len(context)")
    env.close()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

# Support both in-repo and standalone imports
try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from .models import REPLAction, REPLObservation, REPLState, CodeBlockResult
except ImportError:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from models import REPLAction, REPLObservation, REPLState, CodeBlockResult

if TYPE_CHECKING:
    from .server.repl_environment import REPLEnvironment


class REPLEnv:
    """
    Unified client for the REPL Environment.

    Works with both remote servers and local execution, providing the same
    interface regardless of where the code runs.

    Examples:
        >>> # Connect to a running server
        >>> with REPLEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset(context="Hello World", task_prompt="Count chars")
        ...     result = env.execute("count = len(context)")
        ...     result = env.execute("print(f'FINAL({count})')")
        ...     print(result.done)  # True

        >>> # Run locally without a server
        >>> with REPLEnv() as env:
        ...     result = env.reset(context="Hello World", task_prompt="Count chars")
        ...     result = env.execute("count = len(context)")
        ...     print(result.observation.result.success)  # True

        >>> # Local with LLM support for recursive calls
        >>> def my_llm(prompt: str) -> str:
        ...     return "LLM response"
        >>> with REPLEnv(llm_query_fn=my_llm) as env:
        ...     result = env.reset(context="...")
        ...     result = env.execute("response = llm_query('Summarize: ' + context)")

        >>> # From Docker image
        >>> env = REPLEnv.from_docker_image("repl-env:latest")

        >>> # From HuggingFace Hub
        >>> env = REPLEnv.from_hub("openenv/repl-env")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        # Local-only options (ignored when base_url is set)
        llm_query_fn: Optional[Callable[[str], str]] = None,
        llm_batch_fn: Optional[Callable[[List[str]], List[str]]] = None,
        max_output_length: int = 8192,
        context_preview_length: int = 500,
        reward_on_success: float = 1.0,
        reward_on_iteration: float = 0.0,
        reward_on_failure: float = -0.1,
        reward_on_error: float = -0.05,
        # Connection options (ignored when running locally)
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 60.0,
    ):
        """
        Initialize REPL environment.

        Args:
            base_url: Server URL. If None, runs locally without a server.
            llm_query_fn: Function for llm_query() calls (local mode only).
            llm_batch_fn: Function for llm_query_batched() calls (local mode only).
            max_output_length: Max stdout/stderr chars per execution (local only).
            context_preview_length: Chars to show in context preview (local only).
            reward_on_success: Reward when final answer submitted (local only).
            reward_on_iteration: Reward per iteration step (local only).
            reward_on_failure: Reward when max iterations reached (local only).
            reward_on_error: Reward when code execution fails (local only).
            connect_timeout_s: WebSocket connection timeout (remote only).
            message_timeout_s: Message response timeout (remote only).
        """
        self._base_url = base_url
        self._local_env: Optional[REPLEnvironment] = None
        self._remote_client: Optional[_RemoteREPLClient] = None

        # Store local-mode options
        self._llm_query_fn = llm_query_fn
        self._llm_batch_fn = llm_batch_fn
        self._max_output_length = max_output_length
        self._context_preview_length = context_preview_length
        self._reward_on_success = reward_on_success
        self._reward_on_iteration = reward_on_iteration
        self._reward_on_failure = reward_on_failure
        self._reward_on_error = reward_on_error

        # Store remote-mode options
        self._connect_timeout_s = connect_timeout_s
        self._message_timeout_s = message_timeout_s

        # Provider for container/runtime lifecycle (set by factory methods)
        self._provider = None

    def _ensure_initialized(self) -> None:
        """Initialize the appropriate backend (local or remote)."""
        if self._local_env is not None or self._remote_client is not None:
            return

        if self._base_url is None:
            # Local mode: create REPLEnvironment directly
            from .server.repl_environment import REPLEnvironment

            self._local_env = REPLEnvironment(
                max_output_length=self._max_output_length,
                context_preview_length=self._context_preview_length,
                reward_on_success=self._reward_on_success,
                reward_on_iteration=self._reward_on_iteration,
                reward_on_failure=self._reward_on_failure,
                reward_on_error=self._reward_on_error,
                llm_query_fn=self._llm_query_fn,
                llm_batch_fn=self._llm_batch_fn,
            )
        else:
            # Remote mode: create WebSocket client
            self._remote_client = _RemoteREPLClient(
                base_url=self._base_url,
                connect_timeout_s=self._connect_timeout_s,
                message_timeout_s=self._message_timeout_s,
                provider=self._provider,
            )
            self._remote_client.connect()

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
        """
        Reset the environment for a new episode.

        Args:
            context: Text content to analyze (accessible as `context` variable).
            task_prompt: Description of the task to solve.
            max_iterations: Maximum code execution steps before timeout.
            seed: Optional random seed for reproducibility.
            episode_id: Optional custom episode identifier.
            hf_token: Optional HuggingFace token for llm_query/llm_query_batched.
                      When provided, the server uses this token for sub-LLM calls
                      instead of its own configured token.
                      Security: Token is NOT stored in state or logged.
            llm_model: Optional model name for LLM functions (default: Qwen3-Coder-480B).

        Returns:
            StepResult with initial observation.
        """
        self._ensure_initialized()

        if self._local_env is not None:
            # Local mode
            self._local_env.max_iterations = max_iterations
            obs = self._local_env.reset(
                seed=seed,
                episode_id=episode_id,
                context=context,
                task_prompt=task_prompt,
                hf_token=hf_token,
                llm_model=llm_model,
            )
            return self._wrap_observation(obs)
        else:
            # Remote mode
            assert self._remote_client is not None
            return self._remote_client.reset(
                context=context,
                task_prompt=task_prompt,
                max_iterations=max_iterations,
                seed=seed,
                episode_id=episode_id,
                hf_token=hf_token,
                llm_model=llm_model,
            )

    def step(self, action: REPLAction) -> StepResult[REPLObservation]:
        """
        Execute a REPL action.

        Args:
            action: REPLAction containing code to execute.

        Returns:
            StepResult with execution observation.
        """
        self._ensure_initialized()

        if self._local_env is not None:
            obs = self._local_env.step(action)
            return self._wrap_observation(obs)
        else:
            assert self._remote_client is not None
            return self._remote_client.step(action)

    def execute(self, code: str) -> StepResult[REPLObservation]:
        """
        Execute Python code in the REPL.

        Convenience method that wraps step() with a code-only action.

        Args:
            code: Python code to execute.

        Returns:
            StepResult with execution observation.
        """
        return self.step(REPLAction(code=code))

    def submit_final_answer(self, answer: str) -> StepResult[REPLObservation]:
        """
        Submit a final answer and terminate the episode.

        Args:
            answer: The final answer string.

        Returns:
            StepResult with done=True.
        """
        return self.step(
            REPLAction(code="", is_final=True, final_answer=answer)
        )

    def get_variable(self, name: str) -> StepResult[REPLObservation]:
        """
        Retrieve and print a variable from the REPL namespace.

        Args:
            name: Variable name to retrieve.

        Returns:
            StepResult with variable value in stdout.
        """
        return self.execute(f"print(repr({name}))")

    def state(self) -> REPLState:
        """
        Get current environment state.

        Returns:
            REPLState with current environment information.
        """
        self._ensure_initialized()

        if self._local_env is not None:
            return self._local_env.state
        else:
            assert self._remote_client is not None
            return self._remote_client.state()

    def list_variables(self) -> List[str]:
        """
        Get list of available variables in the current session.

        Returns:
            List of variable names.
        """
        return self.state().namespace_keys

    def close(self) -> None:
        """Clean up resources."""
        if self._local_env is not None:
            self._local_env.close()
            self._local_env = None

        if self._remote_client is not None:
            self._remote_client.close()
            self._remote_client = None

    def _wrap_observation(
        self, obs: REPLObservation
    ) -> StepResult[REPLObservation]:
        """Wrap a local REPLObservation in a StepResult."""
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    # Context manager support

    def __enter__(self) -> "REPLEnv":
        """Enter context manager."""
        self._ensure_initialized()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.close()

    # Factory methods

    @classmethod
    def from_docker_image(
        cls,
        image: str,
        **kwargs: Any,
    ) -> "REPLEnv":
        """
        Create a REPL environment by spinning up a Docker container.

        Args:
            image: Docker image name to run (e.g., "repl-env:latest").
            **kwargs: Additional arguments passed to container start.

        Returns:
            Connected REPLEnv instance.
        """
        from openenv.core.containers.runtime import LocalDockerProvider

        provider = LocalDockerProvider()
        base_url = provider.start_container(image, **kwargs)
        provider.wait_for_ready(base_url)

        env = cls(base_url=base_url)
        env._provider = provider
        env._ensure_initialized()
        return env

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        *,
        use_docker: bool = True,
        **kwargs: Any,
    ) -> "REPLEnv":
        """
        Create a REPL environment from a HuggingFace Space.

        Args:
            repo_id: HuggingFace space identifier (e.g., "openenv/repl-env").
            use_docker: If True, pull from HF registry. If False, run with UV.
            **kwargs: Additional arguments passed to provider.

        Returns:
            Connected REPLEnv instance.
        """
        if use_docker:
            from openenv.core.containers.runtime import LocalDockerProvider

            provider = LocalDockerProvider()
            tag = kwargs.pop("tag", "latest")
            image = f"registry.hf.space/{repo_id.replace('/', '-')}:{tag}"
            base_url = provider.start_container(image, **kwargs)
            provider.wait_for_ready(base_url)
        else:
            from openenv.core.containers.runtime import UVProvider

            project_path = kwargs.pop(
                "project_path", f"git+https://huggingface.co/spaces/{repo_id}"
            )
            provider = UVProvider(project_path=project_path, **kwargs)
            base_url = provider.start()
            provider.wait_for_ready()

        env = cls(base_url=base_url)
        env._provider = provider
        env._ensure_initialized()
        return env


class _RemoteREPLClient(EnvClient[REPLAction, REPLObservation, REPLState]):
    """
    Internal WebSocket client for remote REPL connections.

    This is the original EnvClient-based implementation, now used internally
    by REPLEnv for remote mode.
    """

    def _step_payload(self, action: REPLAction) -> Dict:
        """Convert REPLAction to JSON payload for step request."""
        return {
            "code": action.code,
            "is_final": action.is_final,
            "final_answer": action.final_answer,
        }

    def _parse_result(self, payload: Dict) -> StepResult[REPLObservation]:
        """Parse server response into StepResult[REPLObservation]."""
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

    def _parse_state(self, payload: Dict) -> REPLState:
        """Parse server response into REPLState object."""
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
