"""
Fleet Task Environment - Gymnasium-compatible environment for Fleet tasks.

This module provides a task-oriented wrapper around FleetEnvClient that:
1. Accepts task configs (from export_training_tasks.py)
2. Creates versioned environments on reset
3. Injects task prompt into observations
4. Executes verifier for reward on episode completion
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from .client import FleetEnvClient
from .mcp_tools import FleetMCPTools


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
        ttl_seconds: Instance TTL in seconds (default: 600)
        max_steps: Maximum steps per episode (default: 50)
        request_timeout_s: HTTP request timeout in seconds (default: 60.0)

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
        ttl_seconds: int = 600,
        max_steps: int = 50,
        request_timeout_s: float = 60.0,
    ):
        import asyncio

        self.task = task_config
        self.api_key = api_key or os.environ.get("FLEET_API_KEY")
        self.ttl_seconds = ttl_seconds
        self.max_steps = max_steps
        self.request_timeout_s = request_timeout_s

        if not self.api_key:
            raise ValueError("Fleet API key required (pass api_key or set FLEET_API_KEY)")

        self._step_count = 0
        self._done = False
        self._tools_cache: Optional[List[Dict]] = None

        # Create Fleet environment instance (provisions cloud resources)
        env_spec = self._build_env_spec()
        self._orch, self._tools = FleetEnvClient.from_fleet(
            api_key=self.api_key,
            env_key=env_spec,
            data_key=self._get_data_key(),
            data_version=self._get_data_version(),
            ttl_seconds=self.ttl_seconds,
            request_timeout_s=self.request_timeout_s,
        )

        # Fetch tools for tool_use tasks
        # Note: tools are fetched lazily on first reset_async() to avoid
        # asyncio.run() issues when __init__ is called from async context
        self._tools_fetched = False

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

    async def reset_async(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset episode state and return initial observation.

        Environment is already initialized in __init__(). This method resets
        the episode state and returns the observation with cached tools.

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

        # Reset episode state
        self._step_count = 0
        self._done = False

        # Reset the environment
        reset_metadata = {}
        if self._orch:
            try:
                reset_result = self._orch.reset()
                reset_metadata = reset_result.observation.metadata if reset_result else {}
            except Exception as e:
                logger.warning(f"Fleet env reset failed, continuing with empty observation: {e}")

        # Fetch tools lazily on first reset (avoids asyncio.run in __init__)
        if self.modality == "tool_use" and self._tools and not self._tools_fetched:
            try:
                tools_result = await self._tools.list_tools()
                self._tools_cache = tools_result.tools
                self._tools_fetched = True
            except Exception as e:
                logger.warning(f"Failed to fetch tools: {e}")
                self._tools_cache = []
                self._tools_fetched = True

        # Build observation with cached tools
        obs = {
            "prompt": self.prompt,
            "observation": reset_metadata,
            "step": 0,
            "task_key": self.task_key,
            "modality": self.modality,
        }

        if self._tools_cache:
            obs["tools"] = self._tools_cache

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

    async def step_async(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict]:
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
            except Exception as e:
                info["tool_error"] = str(e)
                tool_result = {"error": str(e)}

        # Determine if done
        self._done = agent_done or max_steps_reached
        info["done_reason"] = (
            "agent_done" if agent_done else
            "max_steps" if max_steps_reached else
            None
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

    async def _compute_reward(self) -> float:
        """Compute reward by executing the verifier using Fleet SDK.

        Uses Fleet SDK's Task.verify_detailed() which properly sets up the
        verifier namespace with Environment type, helper functions, etc.

        Returns:
            1.0 if verifier passes, 0.0 otherwise
        """
        # Support both field names: verifier_code (OpenEnv) and verifier_func (Fleet SDK)
        verifier_code = self.task.get("verifier_code") or self.task.get("verifier_func")

        if not verifier_code:
            # No verifier - return neutral reward
            logger.debug(f"Task {self.task_key}: no verifier_code, returning 0.0")
            return 0.0

        if not self._orch:
            logger.warning(f"Task {self.task_key}: no orchestrator, returning 0.0")
            return 0.0

        # Get the Fleet env handle from the orchestrator
        fleet_env = getattr(self._orch, "_fleet_env", None)
        if not fleet_env:
            logger.warning(f"Task {self.task_key}: no Fleet env handle, returning 0.0")
            return 0.0

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

            # Execute verifier via Fleet SDK (handles namespace setup, Environment type, etc.)
            response = fleet_task.verify_detailed(fleet_env)

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

            logger.info(f"Task {self.task_key}: verifier returned success={response.success}, result={response.result}, score={score}")
            return score

        except ImportError as e:
            logger.error(f"Fleet SDK not available for verifier execution: {e}")
            return 0.0
        except Exception as e:
            logger.error(
                f"Verifier execution failed for task {self.task_key}: {e}\n"
                f"Verifier code:\n{verifier_code}"
            )
            return 0.0

    def close(self):
        """Close the environment and cleanup resources."""
        if self._orch:
            try:
                self._orch.close()
            except Exception:
                pass
            self._orch = None
            self._tools = None
            self._tools_cache = None
            self._done = True

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
