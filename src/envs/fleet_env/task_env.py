"""
Fleet Task Environment - Gymnasium-compatible environment for Fleet tasks.

This module provides a task-oriented wrapper around FleetEnvClient that:
1. Accepts task configs (from export_training_tasks.py)
2. Creates versioned environments on reset
3. Injects task prompt into observations
4. Executes verifier for reward on episode completion
"""

import ast
import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from .client import FleetEnvClient
from .mcp_tools import FleetMCPTools
from .telemetry import (
    fleet_exception,
    fleet_warning,
    fleet_info,
    set_task_context,
    clear_task_context,
)


def _is_tool_error(result: Any) -> Tuple[bool, Optional[str]]:
    """Check if a tool result indicates an error.

    MCP server errors come back as:
    - {"error": "..."} from isError=True responses
    - {"status": "failed", ...} from some tools
    - {"isError": true, ...} in some formats

    Returns:
        (is_error, error_message) tuple
    """
    if not isinstance(result, dict):
        return False, None

    # Direct error field (from FleetMCPClient._extract_tool_result)
    # Check for truthy value to avoid false positives on {"error": null}
    if result.get("error"):
        return True, str(result["error"])

    # Status field pattern
    if result.get("status") == "failed":
        return True, result.get("message") or result.get("error") or "status=failed"

    # isError field pattern
    if result.get("isError"):
        return True, result.get("message") or result.get("error") or "isError=true"

    return False, None


class FleetTaskEnv:
    """Gymnasium-compatible environment for Fleet tasks.

    This class wraps FleetEnvClient to provide a task-oriented interface
    suitable for RL training with SkyRL.

    Args:
        task_config: Task configuration dict with keys:
            - task_key: Unique task identifier
            - prompt: Task instruction for the agent
            - env_key: Environment key (e.g., "booking-com")
            - env_version: Environment version (e.g., "v1.2.3")
            - data_key: Optional data key
            - data_version: Optional data version
            - verifier_code: Python code for verification
            - task_modality: "tool_use" or "computer_use"
        api_key: Fleet API key (defaults to FLEET_API_KEY env var)
        ttl_seconds: Instance TTL in seconds. If None, auto-selects based on
            modality: 1800s (30 min) for computer_use, 900s (15 min) for tool_use.
        max_steps: Maximum steps per episode (default: 50)
        request_timeout_s: HTTP request timeout in seconds (default: 60.0)
        partial_reward: If True, compute partial scores from verifier
            error/success accumulators instead of binary 0/1 (default: False)

    Example:
        >>> task_config = {
        ...     "task_key": "search-flights-001",
        ...     "prompt": "Search for flights from NYC to LA",
        ...     "env_key": "booking-com",
        ...     "env_version": "v1.2.3",
        ...     "verifier_code": "async def verify(env): ...",
        ...     "task_modality": "tool_use",
        ... }
        >>> env = FleetTaskEnv(task_config)
        >>> obs = env.reset()
        >>> obs, reward, done, info = env.step({"tool": "search", "params": {...}})
    """

    def __init__(
        self,
        task_config: Dict[str, Any],
        api_key: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        max_steps: int = 50,
        request_timeout_s: float = 60.0,
        reset_timeout_s: float = 10.0,
        partial_reward: bool = False,
    ):
        self.task = task_config
        self.api_key = api_key or os.environ.get("FLEET_API_KEY")
        self.partial_reward = partial_reward
        # Auto-select TTL based on modality if not explicitly provided
        if ttl_seconds is not None:
            self.ttl_seconds = ttl_seconds
        elif self.modality == "computer_use":
            self.ttl_seconds = 1800  # 30 min — CUA rollouts are slow (browser + inference)
        else:
            self.ttl_seconds = 900   # 15 min — tool_use rollouts need headroom for retries
        self.max_steps = max_steps
        self.request_timeout_s = request_timeout_s
        self.reset_timeout_s = reset_timeout_s

        if not self.api_key:
            raise ValueError(
                "Fleet API key required (pass api_key or set FLEET_API_KEY)"
            )

        self._step_count = 0
        self._done = False
        self._rollout_completed_emitted = False
        self._rollout_started = False
        self._tools_cache: Optional[List[Dict]] = None

        # Set telemetry context so init failures are tracked with full context
        set_task_context(
            env_key=self.env_key,
            env_version=self.env_version,
            task_key=self.task_key,
            modality=self.modality,
        )

        # Provisioning is deferred to _ensure_provisioned() (called from reset_async)
        # to avoid blocking the event loop with sync Fleet.make() calls.
        self._orch = None
        self._tools = None

    @property
    def task_key(self) -> str:
        """Get the task key."""
        return self.task.get("task_key", "unknown")

    @property
    def prompt(self) -> str:
        """Get the task prompt."""
        return self.task.get("prompt", "")

    @property
    def modality(self) -> str:
        """Get the task modality."""
        return self.task.get("task_modality", "tool_use")

    @property
    def env_key(self) -> str:
        """Get the environment key (e.g., 'github', 'amazon')."""
        return self.task.get("env_key", "unknown")

    @property
    def env_version(self) -> str:
        """Get the environment version (e.g., 'v0.0.12')."""
        return self.task.get("env_version", "unknown")

    def _build_env_spec(self) -> str:
        """Build env_key:version spec for Fleet.make()."""
        env_key = self.task.get("env_key")
        env_version = self.task.get("env_version")

        if not env_key:
            raise ValueError("Task config missing env_key")

        if env_version:
            return f"{env_key}:{env_version}"
        return env_key

    def _get_data_key(self) -> Optional[str]:
        """Get data_key from task config."""
        return self.task.get("data_key")

    def _get_data_version(self) -> Optional[str]:
        """Get data_version from task config."""
        return self.task.get("data_version")

    def _get_env_variables(self) -> Optional[Dict[str, Any]]:
        """Get env_variables from task config.

        These variables parameterize the environment with task-specific values
        like names, dates, scenario configurations, etc.
        """
        return self.task.get("env_variables")

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment and return initial observation (sync wrapper).

        This is a sync wrapper around reset_async(). For async code, use reset_async() directly.

        Args:
            seed: Optional random seed (passed to env reset)

        Returns:
            Observation dict with keys:
                - prompt: The task instruction
                - observation: Raw observation from env reset
                - tools: List of available tools (if tool_use modality)
                - step: Current step number (0)
        """
        import asyncio

        return asyncio.run(self.reset_async(seed=seed))

    async def _ensure_provisioned(self):
        """Provision the Fleet environment instance if not already done.

        Uses AsyncFleet.make() to avoid blocking the event loop. This allows
        other async trajectories to progress while waiting for provisioning.
        """
        if self._orch is not None:
            return

        env_spec = self._build_env_spec()
        # computer_use: MCP-enabled container with browser infra (port 8081 aggregator)
        # tool_use: standard container with per-env MCP server (port 3003)
        image_type = "mcp" if self.modality == "computer_use" else "standard"
        self._orch, self._tools = await FleetEnvClient.from_fleet_async(
            api_key=self.api_key,
            env_key=env_spec,
            data_key=self._get_data_key(),
            data_version=self._get_data_version(),
            env_variables=self._get_env_variables(),
            image_type=image_type,
            ttl_seconds=self.ttl_seconds,
            request_timeout_s=self.request_timeout_s,
        )

    async def reset_async(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset episode state and return initial observation.

        Provisions the Fleet environment on first call (async, non-blocking),
        then resets episode state and returns the observation with tools.

        Args:
            seed: Optional random seed (currently unused)

        Returns:
            Observation dict with keys:
                - prompt: The task instruction
                - observation: Observation from env reset (or empty if reset fails)
                - tools: List of available tools (if tool_use modality)
                - step: Current step number (0)
        """
        import logging

        logger = logging.getLogger(__name__)

        # Count this rollout attempt immediately — even if provisioning fails,
        # it's still a rollout attempt (e.g., fostgres health check failures).
        fleet_info("fleet_rollout_started")
        self._rollout_started = True
        self._rollout_completed_emitted = False

        # Provision Fleet env (async, non-blocking) on first call
        try:
            await self._ensure_provisioned()
        except Exception:
            # Emit rollout_completed so init failures are tracked in dashboards
            fleet_info(
                "fleet_rollout_completed",
                step_count=0,
                reward=0.0,
                verifier_success=False,
                failure_reason="init_error",
            )
            self._rollout_completed_emitted = True
            raise

        # Reset episode state
        self._step_count = 0
        self._done = False

        # Reset the environment (use short timeout to avoid blocking on broken manager APIs)
        # reset() failure is non-fatal — env is up, just the manager API timed out
        reset_metadata = {}
        if self._orch:
            try:
                saved_timeout = self._orch._timeout
                self._orch._timeout = self.reset_timeout_s
                try:
                    reset_result = await self._orch.reset_async()
                    reset_metadata = (
                        reset_result.observation.metadata if reset_result else {}
                    )
                finally:
                    self._orch._timeout = saved_timeout
            except Exception as e:
                logger.warning(
                    f"[env={self.env_key}] Fleet env reset failed (timeout={self.reset_timeout_s}s), continuing with empty observation: {e}"
                )
                fleet_warning(
                    "fleet_env_reset_failed",
                    step_count=self._step_count,
                    timeout_s=self.reset_timeout_s,
                    error_type=type(e).__name__,
                    error_message=str(e)[:200],
                )

        # Fetch tools — fatal if MCP call fails (no tools = dead rollout)
        try:
            if self._tools:
                tools_result = await self._tools.list_tools()
                self._tools_cache = tools_result.tools
            if not self._tools_cache:
                raise RuntimeError("list_tools returned no tools")
        except Exception as e:
            fleet_info(
                "fleet_rollout_completed",
                step_count=0,
                reward=0.0,
                verifier_success=False,
                failure_reason="tools_error",
                error_message=str(e)[:200],
            )
            self._rollout_completed_emitted = True
            raise

        # Filter tools based on modality:
        # - computer_use: keep ONLY the 'computer' tool
        # - tool_use: EXCLUDE the 'computer' tool (should only use API tools)
        if self.modality == "tool_use":
            self._tools_cache = [
                t
                for t in self._tools_cache
                if t.get("name") != "computer"
                and t.get("function", {}).get("name") != "computer"
            ]

        # For computer_use, filter to only the 'computer' tool
        if self.modality == "computer_use":
            computer_tools = [
                t
                for t in self._tools_cache
                if t.get("name") == "computer"
                or t.get("function", {}).get("name") == "computer"
            ]
            if not computer_tools:
                available = [
                    t.get("name") or t.get("function", {}).get("name")
                    for t in self._tools_cache
                ]
                fleet_info(
                    "fleet_rollout_completed",
                    step_count=0,
                    reward=0.0,
                    verifier_success=False,
                    failure_reason="computer_tool_missing",
                    available_tools=available,
                )
                self._rollout_completed_emitted = True
                raise RuntimeError(
                    f"computer_use modality but no 'computer' tool found. "
                    f"Available tools: {available}. Check MCP image configuration."
                )
            self._tools_cache = computer_tools

        if not self._tools_cache:
            fleet_info(
                "fleet_rollout_completed",
                step_count=0,
                reward=0.0,
                verifier_success=False,
                failure_reason="tools_error",
                error_message="No tools available after modality filtering",
            )
            self._rollout_completed_emitted = True
            raise RuntimeError("No tools available after filtering")

        # Build observation with cached tools
        obs = {
            "prompt": self.prompt,
            "observation": reset_metadata,
            "step": 0,
            "task_key": self.task_key,
            "modality": self.modality,
            "tools": self._tools_cache,
        }

        # For computer_use, take initial screenshot so VL model can see the screen
        # This is critical for VL models - without visual input they're blind
        if self.modality == "computer_use" and self._tools:
            try:
                screenshot_result = await self._tools.call_tool(
                    "computer", {"action": "screenshot"}
                )
                obs["initial_screenshot"] = screenshot_result
                logger.info(f"Task {self.task_key}: captured initial screenshot")
            except Exception as e:
                logger.warning(
                    f"Task {self.task_key}: failed to capture initial screenshot: {e}"
                )
                fleet_exception(
                    "fleet_screenshot_failed",
                    step_count=self._step_count,
                )

        return obs

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Execute a step in the environment (sync wrapper).

        For async tool calls, use step_async() instead.

        Args:
            action: Action dict. For tool_use modality:
                - tool: Tool name to call
                - params: Tool parameters
                - done: Optional flag to signal episode completion

        Returns:
            Tuple of (observation, reward, done, info)
        """
        import asyncio

        return asyncio.run(self.step_async(action))

    async def step_async(
        self, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """Execute a step in the environment.

        Args:
            action: Action dict. For tool_use modality:
                - tool: Tool name to call
                - params: Tool parameters
                - done: Optional flag to signal episode completion

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        if not self._tools:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self._step_count += 1
        info = {"step": self._step_count}

        # Check if agent signals completion
        agent_done = action.get("done", False)

        # Check max steps
        max_steps_reached = self._step_count >= self.max_steps

        # Execute tool call
        tool_name = action.get("tool")
        tool_params = action.get("params", {})
        tool_result = None

        if tool_name:
            try:
                tool_result = await self._tools.call_tool(tool_name, tool_params)
                info["tool_result"] = tool_result

                # Check for MCP server errors (not Python exceptions)
                is_error, error_msg = _is_tool_error(tool_result)
                if is_error:
                    info["tool_error"] = error_msg
                    logger.warning(
                        f"[env={self.env_key}:{self.env_version}] step {self._step_count}/{self.max_steps} "
                        f"tool_error: {tool_name}() -> {error_msg[:200] if error_msg else 'unknown'}"
                    )
                    fleet_warning(
                        "fleet_mcp_tool_error",
                        step_count=self._step_count,
                        max_steps=self.max_steps,
                        tool_name=tool_name,
                        error_message=error_msg[:500] if error_msg else None,
                    )
            except Exception as e:
                info["tool_error"] = str(e)
                tool_result = {"error": str(e)}
                logger.warning(
                    f"[env={self.env_key}:{self.env_version}] step {self._step_count}/{self.max_steps} "
                    f"tool_call_failed: {tool_name}() -> {type(e).__name__}: {str(e)[:200]}"
                )
                fleet_exception(
                    "fleet_tool_call_failed",
                    step_count=self._step_count,
                    max_steps=self.max_steps,
                    tool_name=tool_name,
                )

        # Determine if done
        self._done = agent_done or max_steps_reached
        info["done_reason"] = (
            "agent_done" if agent_done else "max_steps" if max_steps_reached else None
        )

        # Calculate reward (only on episode completion)
        reward = 0.0
        if self._done:
            reward = await self._compute_reward()
            info["reward_computed"] = True

        # Build observation
        obs = {
            "prompt": self.prompt,
            "observation": tool_result or {},
            "step": self._step_count,
            "task_key": self.task_key,
            "modality": self.modality,
        }

        if self._tools_cache:
            obs["tools"] = self._tools_cache

        return obs, reward, self._done, info

    @staticmethod
    def _parse_partial_reward(stdout: str) -> Optional[float]:
        """Parse partial reward from verifier accumulator output.

        Verifiers print error/success accumulators to stdout. This parses
        them to compute a fractional score (n_success / total_checks).

        Returns:
            Partial score in [0, 1], or None if accumulators not found.
        """
        err_match = re.search(
            r">>> ERROR_ACCUMULATOR >>>\n(.+?)\n<<< ERROR_ACCUMULATOR <<<",
            stdout,
            re.DOTALL,
        )
        suc_match = re.search(
            r">>> SUCCESS_ACCUMULATOR >>>\n(.+?)\n<<< SUCCESS_ACCUMULATOR <<<",
            stdout,
            re.DOTALL,
        )
        if not err_match and not suc_match:
            return None
        try:
            n_errors = len(ast.literal_eval(err_match.group(1))) if err_match else 0
            n_success = len(ast.literal_eval(suc_match.group(1))) if suc_match else 0
            total = n_errors + n_success
            return n_success / total if total > 0 else None
        except Exception:
            return None

    async def _compute_reward(self) -> float:
        """Compute reward by executing the verifier using Fleet SDK.

        Uses Fleet SDK's Task.verify_detailed() which properly sets up the
        verifier namespace with Environment type, helper functions, etc.

        Returns:
            1.0 if verifier passes, 0.0 otherwise (or partial if enabled)
        """
        # Support both field names: verifier_code (OpenEnv) and verifier_func (Fleet SDK)
        verifier_code = self.task.get("verifier_code") or self.task.get("verifier_func")
        score = 0.0
        verifier_success = False
        failure_reason = None

        if not verifier_code:
            # No verifier - return neutral reward
            logger.debug(f"Task {self.task_key}: no verifier_code, returning 0.0")
            failure_reason = "no_verifier"
        elif not self._orch:
            logger.warning(f"Task {self.task_key}: no orchestrator, returning 0.0")
            failure_reason = "no_orchestrator"
        else:
            # Get the Fleet env handle from the orchestrator
            fleet_env = getattr(self._orch, "_fleet_env", None)
            if not fleet_env:
                logger.warning(
                    f"Task {self.task_key}: no Fleet env handle, returning 0.0"
                )
                failure_reason = "no_fleet_env"
            else:
                try:
                    # Use Fleet SDK's Task.verify_detailed() for proper verifier execution
                    from fleet.tasks import Task as FleetTask

                    # Create a Fleet SDK Task object with the verifier
                    fleet_task = FleetTask(
                        key=self.task_key,
                        prompt=self.prompt,
                        env_id=self.task.get("env_key", "unknown"),
                        verifier_func=verifier_code,
                    )

                    # Execute verifier in a thread to avoid blocking the event loop.
                    # verify_detailed() does sync HTTP calls internally.
                    response = await asyncio.to_thread(fleet_task.verify_detailed, fleet_env)

                    # Extract result from response
                    # response.success is bool, response.result is the verifier's return value (0.0 or 1.0)
                    if response.success and response.result is not None:
                        score = float(response.result)
                    elif response.success:
                        # Verifier succeeded but returned None - treat as success
                        score = 1.0
                    else:
                        # Verifier failed (exception or explicit failure)
                        score = 0.0

                    verifier_success = response.success

                    # Partial reward: use accumulator counts instead of binary 0/1
                    partial_score = None
                    if (
                        self.partial_reward
                        and score == 0.0
                        and hasattr(response, "stdout")
                        and response.stdout
                    ):
                        partial_score = self._parse_partial_reward(response.stdout)
                        if partial_score is not None:
                            score = partial_score

                    logger.info(
                        f"Task {self.task_key}: verifier returned success={response.success}, "
                        f"result={response.result}, score={score}"
                        + (f", partial={partial_score:.3f}" if partial_score is not None else "")
                    )

                except ImportError as e:
                    logger.error(f"Fleet SDK not available for verifier execution: {e}")
                    failure_reason = "import_error"
                except Exception as e:
                    logger.error(
                        f"Verifier execution failed for task {self.task_key}: {e}\n"
                        f"Verifier code:\n{verifier_code}"
                    )
                    fleet_exception(
                        "fleet_verifier_failed",
                        step_count=self._step_count,
                        verifier_code_snippet=(
                            verifier_code[:200] if verifier_code else ""
                        ),
                    )
                    failure_reason = "verifier_exception"

        # Always emit rollout completed event
        fleet_info(
            "fleet_rollout_completed",
            step_count=self._step_count,
            max_steps=self.max_steps,
            reward=score,
            verifier_success=verifier_success,
            failure_reason=failure_reason,
        )
        self._rollout_completed_emitted = True
        return score

    def close(self):
        """Close the environment and cleanup resources.

        Emits fleet_rollout_completed if a rollout was started but never
        completed (e.g., caller hit max_turns and stopped without telling us,
        context overflow, job cancellation, TTL expiry).
        """
        try:
            # Emit rollout_completed for orphaned rollouts (started but never completed).
            # This happens when the caller (SkyRL) stops without telling us why:
            # max_turns hit, context overflow, job cancellation, etc.
            if self._rollout_started and not self._rollout_completed_emitted:
                stop_reason = "max_steps" if self._step_count >= self.max_steps else "abandoned"
                fleet_info(
                    "fleet_rollout_completed",
                    step_count=self._step_count,
                    max_steps=self.max_steps,
                    reward=0.0,
                    verifier_success=False,
                    failure_reason=stop_reason,
                )
                self._rollout_completed_emitted = True

            if self._orch:
                try:
                    self._orch.close()
                except Exception:
                    pass  # Expected when instance TTL expired
        finally:
            # Always cleanup state, even if telemetry fails
            self._orch = None
            self._tools = None
            self._tools_cache = None
            self._done = True
            self._rollout_started = False
            clear_task_context()

    async def close_async(self):
        """Async close — avoids blocking the event loop on Fleet instance termination."""
        try:
            if self._rollout_started and not self._rollout_completed_emitted:
                stop_reason = "max_steps" if self._step_count >= self.max_steps else "abandoned"
                fleet_info(
                    "fleet_rollout_completed",
                    step_count=self._step_count,
                    max_steps=self.max_steps,
                    reward=0.0,
                    verifier_success=False,
                    failure_reason=stop_reason,
                )
                self._rollout_completed_emitted = True

            if self._orch:
                try:
                    await self._orch.close_async()
                except Exception:
                    pass  # Expected when instance TTL expired
        finally:
            self._orch = None
            self._tools = None
            self._tools_cache = None
            self._done = True
            self._rollout_started = False
            clear_task_context()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @classmethod
    def from_json_file(cls, json_path: str, task_key: str, **kwargs) -> "FleetTaskEnv":
        """Create FleetTaskEnv from exported JSON file.

        Args:
            json_path: Path to JSON file from export_training_tasks.py
            task_key: Task key to load
            **kwargs: Additional arguments passed to FleetTaskEnv

        Returns:
            FleetTaskEnv instance for the specified task
        """
        import json

        with open(json_path) as f:
            data = json.load(f)

        tasks = data.get("tasks", [])
        task_config = next((t for t in tasks if t["task_key"] == task_key), None)

        if not task_config:
            raise ValueError(f"Task '{task_key}' not found in {json_path}")

        return cls(task_config, **kwargs)

    @classmethod
    def from_json_file_all(cls, json_path: str, **kwargs) -> List["FleetTaskEnv"]:
        """Create FleetTaskEnv instances for all tasks in JSON file.

        Args:
            json_path: Path to JSON file from export_training_tasks.py
            **kwargs: Additional arguments passed to FleetTaskEnv

        Returns:
            List of FleetTaskEnv instances
        """
        import json

        with open(json_path) as f:
            data = json.load(f)

        tasks = data.get("tasks", [])
        return [cls(task, **kwargs) for task in tasks]


def make_fleet_task_env(task_config: Dict[str, Any], **kwargs) -> FleetTaskEnv:
    """Factory function for creating FleetTaskEnv.

    This is the recommended entry point for SkyRL integration.

    Args:
        task_config: Task configuration dict
        **kwargs: Additional arguments passed to FleetTaskEnv

    Returns:
        FleetTaskEnv instance
    """
    return FleetTaskEnv(task_config, **kwargs)
