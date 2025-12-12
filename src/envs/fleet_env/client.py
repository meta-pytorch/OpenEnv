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
        image_type: str = "mcp",
        **kwargs: Any,
    ) -> Tuple["FleetEnvClient", FleetMCPTools]:
        """
            Instantiate a FleetEnvClient and FleetMCPTools from a Fleet environment.

            Args:
                api_key: The API key for the Fleet environment.
                env_key: The environment key for the Fleet environment.
                region: The region for the Fleet environment.
                ttl_seconds: The TTL for the Fleet environment.
                env_variables: The environment variables for the Fleet environment.
                image_type: The image type for the Fleet environment.
        """
        try:
            from fleet import AsyncFleet
        except ImportError as e:
            raise ImportError(
                "Fleet support requires the optional dependency set. "
                "Install with `pip install openenv-core[fleet]`."
            ) from e

        async def _make_env():
            fleet = AsyncFleet(api_key=api_key)
            return await fleet.make(
                env_key=env_key,
                region=region,
                ttl_seconds=ttl_seconds,
                env_variables=env_variables,
                image_type=image_type,
            )

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            env = loop.run_until_complete(_make_env())
        finally:
            asyncio.set_event_loop(None)
            loop.close()

        root = env.urls.root
        mcp_urls = tuple(sorted({f"{root}api/v1/mcp", f"{root}mcp"}))

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


