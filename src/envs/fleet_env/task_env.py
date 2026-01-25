"""
Fleet Task Environment - Gymnasium-compatible environment for Fleet tasks.

This module provides a task-oriented wrapper around FleetEnvClient that:
1. Accepts task configs (from export_training_tasks.py)
2. Creates versioned environments on reset
3. Injects task prompt into observations
4. Executes verifier for reward on episode completion
"""

import os
from typing import Any, Dict, List, Optional, Tuple

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

        # Fetch tools for tool_use tasks (sync wrapper for async call)
        if self.modality == "tool_use" and self._tools:
            tools_result = asyncio.run(self._tools.list_tools())
            self._tools_cache = tools_result.tools

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
        """Compute reward by executing the verifier.

        Returns:
            1.0 if verifier passes, 0.0 otherwise
        """
        verifier_code = self.task.get("verifier_code")

        if not verifier_code:
            # No verifier - return neutral reward
            return 0.0

        if not self._orch:
            return 0.0

        try:
            # Execute verifier
            # For now, use local execution
            # TODO: Add remote verifier execution support
            result = await self._execute_verifier_local(verifier_code)
            return 1.0 if result else 0.0
        except Exception as e:
            # Verifier failed - treat as unsuccessful
            print(f"Verifier execution failed: {e}")
            return 0.0

    async def _execute_verifier_local(self, verifier_code: str) -> bool:
        """Execute verifier code locally.

        Args:
            verifier_code: Python code string containing verify() function

        Returns:
            True if verification passes, False otherwise
        """
        # Create namespace for verifier execution
        namespace = {}

        # Execute the verifier code to define the function
        exec(verifier_code, namespace)

        # Get the verify function
        verify_func = namespace.get("verify")
        if not verify_func:
            raise ValueError("Verifier code must define a 'verify' function")

        # Call verifier with the orchestrator (env handle)
        result = await verify_func(self._orch)

        # Handle different result formats
        if isinstance(result, bool):
            return result
        if isinstance(result, (int, float)):
            return result > 0
        if isinstance(result, dict):
            return result.get("success", False) or result.get("score", 0) > 0

        return bool(result)

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
