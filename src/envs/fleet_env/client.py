# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fleet Environment client (HTTP orchestration only)."""

import asyncio
import dataclasses
from contextlib import suppress
from typing import Any, Dict, Optional, Tuple, Type

try:
    # In-repo imports
    from core.env_server.types import Action, Observation, State
    from core.http_env_client import HTTPEnvClient
    from core.client_types import StepResult
except ImportError:
    # Standalone imports
    from openenv_core.env_server.types import Action, Observation, State
    from openenv_core.http_env_client import HTTPEnvClient
    from openenv_core.client_types import StepResult

from .mcp_tools import FleetMCPTools
from .models import CallToolAction, ListToolsAction
from .telemetry import fleet_error, fleet_warning, fleet_info


class FleetEnvClient(HTTPEnvClient[Action, Observation]):
    """Orchestrator-facing client for Fleet-hosted environments (HTTP only)."""

    def __init__(
        self,
        base_url: str,
        fleet_env_handle: Any,
        api_key: str,
        mcp_urls: Tuple[str, ...],
        **kwargs: Any,
    ):
        super().__init__(
            base_url=base_url,
            default_headers={"Authorization": f"Bearer {api_key}"},
            **kwargs,
        )
        self._fleet_env = fleet_env_handle
        self._api_key = api_key
        self._mcp_urls = mcp_urls

    @classmethod
    def from_fleet(
        cls: Type["FleetEnvClient"],
        api_key: str,
        env_key: str,
        data_key: str,
        data_version: str,
        image_type: str,
        region: Optional[str] = None,
        ttl_seconds: Optional[int] = 3600,
        env_variables: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple["FleetEnvClient", FleetMCPTools]:
        try:
            from fleet import Fleet
        except ImportError as e:
            raise ImportError(
                "Fleet support requires the optional dependency set. "
                "Install with `pip install openenv[fleet]`."
            ) from e

        # Use synchronous Fleet client for the orchestrator handle.
        # This ensures .close() and other lifecycle methods are synchronous.
        fleet = Fleet(api_key=api_key)

        # Fleet SDK expects data_key in "key:version" format
        data_key_spec = None
        if data_key:
            if data_version:
                data_key_spec = f"{data_key}:{data_version}"
            else:
                data_key_spec = data_key

        import time
        import logging

        _logger = logging.getLogger(__name__)

        _logger.info(f"Creating Fleet instance: env_key={env_key}, ttl={ttl_seconds}s")
        start = time.time()

        # Retry logic for transient Fleet API failures (e.g., health check failures)
        max_retries = 3
        retry_base_delay = 2.0  # seconds
        env = None

        for attempt in range(max_retries):
            try:
                # Fleet SDK expects image_type=None for standard images
                sdk_image_type = image_type if image_type == "mcp" else None
                env = fleet.make(
                    env_key=env_key,
                    region=region,
                    ttl_seconds=ttl_seconds,
                    env_variables=env_variables,
                    image_type=sdk_image_type,
                    data_key=data_key_spec,
                )
                break  # Success
            except Exception as e:
                error_msg = str(e)
                # Retry on transient errors (health check failures, timeouts, etc.)
                is_transient = any(
                    x in error_msg.lower()
                    for x in ["health check", "timeout", "connection", "temporarily"]
                )
                if attempt < max_retries - 1 and is_transient:
                    delay = retry_base_delay * (2**attempt)
                    _logger.warning(
                        f"[env={env_key}] Fleet.make() failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    fleet_warning(
                        "fleet_make_retry",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        retry_delay_s=delay,
                    )
                    time.sleep(delay)
                else:
                    _logger.error(
                        f"[env={env_key}] Fleet.make() failed after {attempt + 1} attempt(s): {e}"
                    )
                    fleet_error(
                        "fleet_make_failed",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    raise

        elapsed = time.time() - start
        instance_id = getattr(env, "instance_id", "unknown")
        _logger.info(f"Fleet instance ready in {elapsed:.1f}s: {instance_id}")

        root = env.urls.root
        # Pick MCP endpoint based on modality:
        # - computer_use: aggregator on port 8081 (has computer tool + API tools)
        # - tool_use: per-env MCP server on port 3003 (API tools only)
        if image_type == "mcp":
            mcp_urls = (f"{root}api/v1/mcp",)
        else:
            mcp_urls = (f"{root}mcp",)

        orch = cls(
            base_url=env.urls.manager.api,
            fleet_env_handle=env,
            api_key=api_key,
            mcp_urls=mcp_urls,
            **kwargs,
        )
        tools = FleetMCPTools(api_key=api_key, mcp_urls=mcp_urls)
        return orch, tools

    @classmethod
    async def from_fleet_async(
        cls: Type["FleetEnvClient"],
        api_key: str,
        env_key: str,
        data_key: str,
        data_version: str,
        image_type: str,
        region: Optional[str] = None,
        ttl_seconds: Optional[int] = 3600,
        env_variables: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple["FleetEnvClient", FleetMCPTools]:
        """Async version of from_fleet() — does not block the event loop.

        Uses AsyncFleet.make() for provisioning and asyncio.sleep() for retries,
        allowing other async trajectories to progress while waiting.
        """
        try:
            from fleet._async import AsyncFleet
            from fleet import Fleet
        except ImportError as e:
            raise ImportError(
                "Fleet support requires the optional dependency set. "
                "Install with `pip install openenv[fleet]`."
            ) from e

        async_fleet = AsyncFleet(api_key=api_key)

        # Fleet SDK expects data_key in "key:version" format
        data_key_spec = None
        if data_key:
            if data_version:
                data_key_spec = f"{data_key}:{data_version}"
            else:
                data_key_spec = data_key

        import time
        import logging

        _logger = logging.getLogger(__name__)

        _logger.info(f"Creating Fleet instance (async): env_key={env_key}, ttl={ttl_seconds}s")
        start = time.time()

        # Retry logic with async sleep (non-blocking)
        max_retries = 3
        retry_base_delay = 2.0  # seconds
        async_env = None

        for attempt in range(max_retries):
            try:
                # Fleet SDK expects image_type=None for standard images
                sdk_image_type = image_type if image_type == "mcp" else None

                # THROWAWAY DEBUG: do NOT call AsyncFleet.make(). Just print every 10s.
                # If these logs stop, your event loop / process is blocked.
                async def _tick_forever() -> None:
                    tick = 0
                    while True:
                        tick += 1
                        _logger.info(
                            f"debug_make_disabled tick={tick} "
                            f"env_key={env_key} region={region} ttl_seconds={ttl_seconds} image_type={image_type} "
                            f"data_key={data_key_spec} attempt={attempt + 1}/{max_retries}"
                        )
                        await asyncio.sleep(10)

                await _tick_forever()
                break  # Success
            except Exception as e:
                error_msg = str(e)
                # Retry on transient errors (health check failures, timeouts, etc.)
                is_transient = any(
                    x in error_msg.lower()
                    for x in ["health check", "timeout", "connection", "temporarily"]
                )
                if attempt < max_retries - 1 and is_transient:
                    delay = retry_base_delay * (2**attempt)
                    _logger.warning(
                        f"[env={env_key}] AsyncFleet.make() failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    fleet_warning(
                        "fleet_make_retry",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        retry_delay_s=delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    _logger.error(
                        f"[env={env_key}] AsyncFleet.make() failed after {attempt + 1} attempt(s): {e}"
                    )
                    fleet_error(
                        "fleet_make_failed",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    raise

        elapsed = time.time() - start
        instance_id = getattr(async_env, "instance_id", "unknown")
        _logger.info(f"Fleet instance ready (async) in {elapsed:.1f}s: {instance_id}")
        fleet_info(
            "fleet_provisioning_completed",
            provisioning_time_s=round(elapsed, 1),
            instance_id=instance_id,
        )

        # Get a sync env handle for close() and verify_detailed() compatibility.
        # This is a fast GET request (~100ms), not a provisioning call.
        try:
            sync_fleet = Fleet(api_key=api_key)
            sync_env = sync_fleet.instance(instance_id)
        except Exception as e:
            # Clean up the async instance we just created
            _logger.error(f"[env={env_key}] Failed to get sync handle for {instance_id}: {e}")
            try:
                await async_env.close()
            except Exception:
                pass
            raise

        root = async_env.urls.root
        # Pick MCP endpoint based on modality:
        # - computer_use (image_type="mcp"): aggregator on port 8081 (has computer tool + API tools)
        # - tool_use: per-env MCP server on port 3003 (API tools only)
        if image_type == "mcp":
            mcp_urls = (f"{root}api/v1/mcp",)
        else:
            mcp_urls = (f"{root}mcp",)

        orch = cls(
            base_url=async_env.urls.manager.api,
            fleet_env_handle=sync_env,
            api_key=api_key,
            mcp_urls=mcp_urls,
            **kwargs,
        )
        tools = FleetMCPTools(api_key=api_key, mcp_urls=mcp_urls)
        return orch, tools

    def _step_payload(self, action: Action) -> dict:
        """Serialize action for HTTP /step."""
        if dataclasses.is_dataclass(action):
            return dataclasses.asdict(action)
        if isinstance(action, dict):
            return action
        raise TypeError(f"Action must be a dataclass or dict, got {type(action)}")

    def _parse_result(self, payload: dict) -> StepResult[Observation]:
        """Parse standard OpenEnv step response."""
        obs_payload = payload.get("observation", {})
        # Ensure obs_payload is a dict before accessing .get()
        if not isinstance(obs_payload, dict):
            # If observation is a primitive (e.g. string), wrap it
            obs_payload = {"content": obs_payload}

        return StepResult(
            observation=Observation(
                metadata=obs_payload,
                reward=payload.get("reward"),
                done=payload.get("done", False),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            try:
                return State(**payload)
            except TypeError:
                pass
        return payload

    def step(self, action: Action) -> StepResult[Observation]:
        # Enforce separation: agent actions are MCP-only (use FleetMCPTools).
        if isinstance(action, (ListToolsAction, CallToolAction)):
            raise TypeError(
                "Agent tool actions are MCP-only. Use FleetMCPTools.list_tools()/call_tool()."
            )
        return super().step(action)

    def close(self) -> None:
        """Terminate the remote Fleet instance (resource cleanup), not an episode reset."""
        if self._fleet_env:
            self._fleet_env.close()
        super().close()
