# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fleet Environment client (HTTP orchestration only)."""

import asyncio
import dataclasses
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
        region: Optional[str] = None,
        ttl_seconds: Optional[int] = 3600,
        env_variables: Optional[Dict[str, Any]] = None,
        image_type: Optional[str] = None,
        data_key: Optional[str] = None,
        data_version: Optional[str] = None,
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
                env = fleet.make(
                    env_key=env_key,
                    region=region,
                    ttl_seconds=ttl_seconds,
                    env_variables=env_variables,
                    image_type=image_type,
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
                        f"Fleet.make() failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    _logger.error(
                        f"Fleet.make() failed after {attempt + 1} attempt(s): {e}"
                    )
                    raise

        _logger.info(f"Fleet instance ready in {time.time() - start:.1f}s: {env.instance_id}")

        root = env.urls.root
        # Fleet currently exposes multiple MCP endpoints. Prefer /api/v1/mcp first.
        mcp_urls = (f"{root}api/v1/mcp", f"{root}mcp")

        orch = cls(
            base_url=env.urls.manager.api,
            fleet_env_handle=env,
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


